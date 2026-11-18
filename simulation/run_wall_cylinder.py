"""Execute Wall-predicted arm/gripper/base actions on the fixed official fingers.

The model observes RGB and robot state. The wrist-cue layout also supplies
initial RGB localization. Object truth and the volume reader score the trial.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from audit_gripper import audit
from carry_env import CAMERAS,CONTROL_DT
from cylinder_actions import relative_to_absolute,cartesian_to_joint_target
from cylinder_joint_actions import (JOINT_ACTION_LAYOUT,JOINT_SERVO_STATE_WIDTH,JOINT_WRIST_CUE_STATE_LAYOUT,
    absolute_joint_chunk,joint_action_to_targets,joint_state_width,robot_joint_observation,bounded_joint_command,
    with_joint_wrist_cue_state)
from cylinder_progress_prompts import CylinderProgressPrompts,PROGRESS_PROTOCOL
from cylinder_tray_vision import TRAY_CUE_PROTOCOLS, capture_initial_tray_cue, with_tray_cue_state
from cylinder_reader import read_volume
from cylinder_wrist_cue import capture_initial_wrist_cue
from inspect_cylinder import CylinderEnv,CYLINDER_INSTRUCTION
from run_wall_carry import CarryWallPolicy,_normalize_real_prefix
from robot_state_normalization import enable_unclipped_robot_state
from robot_action_filter import (RobotActionFilter,limit_pose_step,TemporalActionEnsemble,
                                 average_model_action_samples,GripperOpeningConfirmation)

SOURCE_SHA256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
               for name in ("run_wall_cylinder.py","run_wall_carry.py","cylinder_actions.py",
                            "inspect_cylinder.py","carry_env.py","audit_gripper.py","cylinder_reader.py",
                            "robot_action_filter.py","robot_state_normalization.py","cylinder_joint_actions.py",
                            "cylinder_wrist_cue.py","cylinder_progress_prompts.py","cylinder_tray_vision.py",
                            "scene_config.json")}


class CylinderWallPolicy(CarryWallPolicy):
    @staticmethod
    def _validate_robot_layouts(config):
        if 'follow_right_arm_joint_position_relative' in config['dof_config']:
            if list(config['dof_config'].items())!=list(JOINT_ACTION_LAYOUT.items()):
                raise ValueError('Unsupported cylinder joint action layout')
            joint_state_width(config['agent_pos_config'])
        else:CarryWallPolicy._validate_robot_layouts(config)

    def __init__(self,checkpoint,train_config,norm_key,*,flow_steps=10,flow_noise="random",
                 state_embedding_precision="training",unclipped_robot_state=False):
        if not 1<=flow_steps<=100:raise ValueError("Flow steps must be in [1,100]")
        super().__init__(checkpoint,train_config,norm_key)
        self.flow_steps=flow_steps
        self.action_space=('joint' if 'follow_right_arm_joint_position_relative' in
                           self._train_config['dof_config'] else 'cartesian')
        self.action_width=9 if self.action_space=='joint' else 23
        self.with_odometry="base_odometry" in self._train_config["agent_pos_config"]
        self.state_width=(joint_state_width(self._train_config['agent_pos_config']) if self.action_space=='joint'
                          else 26 if self.with_odometry else 23)
        self.joint_servo_state=self.action_space=='joint' and self.state_width==JOINT_SERVO_STATE_WIDTH
        self.joint_wrist_cue=(self.joint_servo_state and
                             self._train_config['agent_pos_config']==JOINT_WRIST_CUE_STATE_LAYOUT)
        self.progress_protocol=getattr(self._model.config,'joint_progress_protocol',None)
        if self.progress_protocol is not None and (self.progress_protocol!=PROGRESS_PROTOCOL or not self.joint_servo_state):
            raise ValueError('Unsupported progress protocol or robot state layout')
        self.tray_cue_protocol=getattr(self._model.config,'joint_initial_tray_cue',None)
        if self.tray_cue_protocol is not None and (
                self.tray_cue_protocol not in TRAY_CUE_PROTOCOLS or not self.joint_wrist_cue or not self.progress_protocol):
            raise ValueError('Tray cue requires the cue state layout and progress prompts')
        if unclipped_robot_state:
            if self.state_width!=26 or self._model.config.use_state_string_representation:
                raise ValueError("State unclipping experiment requires 26D continuous robot state")
            enable_unclipped_robot_state(self._normalizer_propri)
        if state_embedding_precision not in ("training","native"):
            raise ValueError("Invalid state embedding precision contract")
        self.state_embedding_precision=state_embedding_precision
        if flow_noise not in ("random","zero"):raise ValueError("Invalid flow noise mode")
        if flow_noise=="zero":
            # Pinned Wall-X inference hook. Rebuild the FIRST flow token too;
            # replacing just the integration tensor would leave random noise
            # in the prefill embedding and would not be deterministic.
            prepare=self._model._prepare_flow_action_inputs
            def prepare_zero_noise(*args,**kwargs):
                ctx=prepare(*args,**kwargs)
                ctx["noise"]=torch.zeros_like(ctx["noise"])
                ctx["noisy_action"]=ctx["noise"].clone()
                time=ctx["times"][0].expand(ctx["batch_size"])
                embedding,condition=self._model.action_preprocessor.step(
                    timestep=time,noisy_action=ctx["noisy_action"],dof_mask=ctx["dof_mask"])
                ctx["inputs_embeds"][ctx["flow_action_mask"]]=embedding.reshape(-1,ctx["inputs_embeds"].shape[-1]).to(ctx["inputs_embeds"].dtype)
                ctx["adarms_cond"]=condition
                return ctx
            self._model._prepare_flow_action_inputs=prepare_zero_noise
        self.load_report.update(flow_steps=flow_steps,virtual_tail_after_normalization=0,
                                action_space=self.action_space,active_action_dimensions=self.action_width,
                                robot_state_dimensions=self.state_width,base_odometry=self.with_odometry,
                                joint_servo_state=self.joint_servo_state,joint_wrist_cue=self.joint_wrist_cue,
                                progress_protocol=self.progress_protocol,tray_cue_protocol=self.tray_cue_protocol,
                                flow_noise=flow_noise,state_embedding_precision=state_embedding_precision,
                                unclipped_robot_state=unclipped_robot_state,
                                state_precision_alignment_applies=not self._model.config.use_state_string_representation,
                                token_embedding_dtype=str(self._model.model.embed_tokens.weight.dtype))

    @torch.inference_mode()
    def predict_many(self,observations,instruction=CYLINDER_INSTRUCTION):
        encoded=[]
        for obs in observations:
            assert obs["state"].shape==(self.state_width,)
            state=np.zeros((1,1,26),np.float32);state[...,:self.state_width]=obs["state"]
            mask=np.zeros_like(state);mask[...,:self.state_width]=1
            dof=np.zeros((1,self._action_horizon,26),np.float32);dof[...,:self.action_width]=1
            encoded.append(dict(proprioception=state,agent_pos_mask=mask,dof_mask=dof,**obs["images"]))
        if self.progress_protocol:
            prompts=[self._get_flow_prompt(obs.get('instruction',instruction)) for obs in observations]
            prefixes,postfixes=map(list,zip(*prompts,strict=True))
        else:
            prefix,postfix=self._get_flow_prompt(instruction)
            prefixes,postfixes=[prefix]*len(encoded),[postfix]*len(encoded)
        inputs=self._construct_model_input(encoded,prefixes,postfixes)
        padding=_normalize_real_prefix(self._normalizer_action,torch.zeros(len(encoded),1,26),
                                        inputs["dataset_names"]).to("cuda")
        # Match the training collator's zero padding after normalization.
        # Zero every virtual action/state slot after normalization.
        padding[...,self.action_width:]=0
        inputs["proprioception"][...,self.state_width:]=0
        if self.state_embedding_precision=="training" and not self._model.config.use_state_string_representation:
            # Training's scatter_proprioception_embeddings first rounds state
            # and mask to the token-embedding dtype, then projects in FP32.
            # Native generate_flow_action omits that first cast. Align the
            # actual model boundary, beyond the common preprocessing tensors.
            dtype=self._model.model.embed_tokens.weight.dtype
            inputs["proprioception"]=inputs["proprioception"].to(dtype)
            inputs["agent_pos_mask"]=inputs["agent_pos_mask"].to(dtype)
        output=self._model.generate_flow_action(action_horizon=self._action_horizon,action_dim=26,
            num_inference_timesteps=self.flow_steps,padding_action=padding,**inputs)
        actions=output["predict_action"].float().cpu().numpy()
        if actions.shape!=(len(encoded),self._action_horizon,26) or not np.isfinite(actions).all():
            raise ValueError("Invalid batched Wall actions")
        return actions


class CylinderTrial:
    def __init__(self,case,controller,output,*,record=False,max_seconds=90,image_size=448,action_filter_seconds=0.,with_odometry=False,
                 cartesian_speed_limit=0.,angular_speed_limit=0.,ik_max_residual=.002,temporal_ensemble=0,prediction_samples=1,
                 state_embedding_precision="training",gripper_open_confirmations=0,unclipped_robot_state=False,
                 gripper_open_confirmation_mode="prefix-median",action_space="cartesian",joint_servo_state=False,
                 joint_wrist_cue=False,progress_protocol=None,tray_cue_protocol=None):
        if action_space not in ('cartesian','joint'):raise ValueError('Invalid action space')
        if joint_servo_state and action_space!='joint':raise ValueError('Servo state requires joint actions')
        if joint_wrist_cue and not joint_servo_state:raise ValueError('Wrist cue requires joint servo observations')
        if progress_protocol is not None and (progress_protocol!=PROGRESS_PROTOCOL or not joint_servo_state):
            raise ValueError('Unsupported progress protocol or robot state layout')
        if tray_cue_protocol is not None and (
                tray_cue_protocol not in TRAY_CUE_PROTOCOLS or not joint_wrist_cue or not progress_protocol):
            raise ValueError('Tray cue requires the cue state layout and progress prompts')
        if action_space=='joint' and any((action_filter_seconds,cartesian_speed_limit,angular_speed_limit,
                                        temporal_ensemble,prediction_samples!=1,gripper_open_confirmations)):
            raise ValueError('Joint execution requires single model draws without Cartesian or gripper filters')
        self.action_space=action_space
        self.joint_servo_state=joint_servo_state
        self.max_joint_limit_projection=0.;self.joint_limit_projected_frames=0
        self.case=case;self.output=output;output.mkdir(parents=True,exist_ok=False)
        options=controller.get("environment",{})
        if set(options)-{"grasp_x","grasp_height","solver_impratio","physics_dt","grip_force_target"}:
            raise ValueError("Execution config cannot alter finger geometry/material")
        self.env=CylinderEnv(case["volume"],image_size=768,**options,
            friction_scale=case.get("friction_scale",1.),mass_scale=case.get("mass_scale",1.))
        self.geometry_report=audit(self.env)
        self.env.reset_inspection(seed=case["seed"],jitter=case.get("jitter",.004),yaw_jitter=case.get("yaw_jitter",0.))
        self.initial_wrist_cue=None
        self.tray_cue_protocol=tray_cue_protocol
        self.initial_tray_cue=None
        self.initial_tray_images=[]
        if tray_cue_protocol:
            images,self.initial_tray_cue=capture_initial_tray_cue(self.env)
            if record:
                self.initial_tray_images=list(images)
                for name,rgb in images.items():imageio.imwrite(output/f"initial_tray_{name}.png",rgb)
        elif joint_wrist_cue:
            rgb,self.initial_wrist_cue=capture_initial_wrist_cue(self.env)
            if record:imageio.imwrite(output/"initial_right_wrist_1024.png",rgb)
        self.initial_cylinder=self.env.data.xpos[self.env.cylinder_body].copy()
        self.initial_base=self.env.base_pose().copy()
        self.progress_router=CylinderProgressPrompts(self.initial_base) if progress_protocol else None
        self.progress_changes=[]
        if self.progress_router is not None:
            self.progress_initial_joint_command=bounded_joint_command(self.env).tolist()
            self.update_progress()
        self.action_filter=RobotActionFilter(self.env.state_vector(),action_filter_seconds,CONTROL_DT)
        self.with_odometry=with_odometry or action_space=='joint'
        self.cartesian_speed_limit=cartesian_speed_limit;self.angular_speed_limit=angular_speed_limit
        self.ik_max_residual=ik_max_residual;self.max_attempted_ik_residual=0.
        self.temporal_ensemble=TemporalActionEnsemble(temporal_ensemble)
        self.prediction_samples=prediction_samples;self.sampled_chunks=[]
        self.state_embedding_precision=state_embedding_precision
        self.unclipped_robot_state=unclipped_robot_state
        self.opening_guard=GripperOpeningConfirmation(self.env.state_vector()[19],gripper_open_confirmations,
                                                     gripper_open_confirmation_mode)
        self.image_size=image_size;self.max_seconds=max_seconds
        self.held=False;self.contact_seconds=0.;self.lost_seconds=0.;self.max_slip=0.;self.max_tilt=0.;self.max_lift=0.
        self.done=False;self.success=False;self.reason=None;self.rows=[];self.chunks=[];self.readings=[]
        self.observations=[]
        self.recovery_states=[]
        self.recovery_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.hold_actuator_ids=sorted(self.env.hold_targets)
        self.next_read=0.;self.good_readings=[];self.latencies=[];self.writers={}
        self.task_gates={}
        self.evaluation_interrupted=False
        if record:
            for cam in ("cylinder_overview","face_view"):
                self.writers[cam]=imageio.get_writer(output/f"{cam}.mp4",fps=20,codec="libx264",quality=8,macro_block_size=1)

    def update_progress(self):
        current=self.progress_router.observe(robot_joint_observation(self.env,servo_state=True))
        if not self.progress_changes or self.progress_changes[-1]['task_index']!=current:
            self.progress_changes.append(dict(time_s=float(self.env.data.time),task_index=current))

    def observation(self):
        observation=dict(state=self.env.state_vector().copy(),
                         images={name:self.env.render(name,width=self.image_size) for name in CAMERAS})
        if self.action_space=='joint':
            observation['state']=robot_joint_observation(self.env,servo_state=self.joint_servo_state)
            if self.tray_cue_protocol:
                estimate=self.initial_tray_cue if self.progress_router.index==0 else None
                observation['state']=with_tray_cue_state(observation['state'],estimate,
                    self.env.config['target_tray']['center_xy'],self.tray_cue_protocol)
            elif self.initial_wrist_cue is not None:
                observation['state']=with_joint_wrist_cue_state(observation['state'],self.initial_wrist_cue['world_xy_m'])
        elif self.with_odometry:
            observation["state"]=np.r_[observation["state"],self.env.base_pose()].astype(np.float32)
        record=dict(time=float(self.env.data.time),state=observation["state"].copy())
        if self.progress_router is not None:
            observation['instruction']=record['instruction']=self.progress_router.instruction
        self.observations.append(record)
        physics=np.empty(mujoco.mj_stateSize(self.env.model,self.recovery_spec))
        mujoco.mj_getState(self.env.model,self.env.data,physics,self.recovery_spec)
        self.recovery_states.append(dict(physics=physics,
            hold_targets=np.array([self.env.hold_targets[i] for i in self.hold_actuator_ids]),
            applied_base_action=self.env.applied_action.copy()))
        if self.writers:
            folder=self.output/"model_observations"/f"chunk_{len(self.chunks):04d}"
            folder.mkdir(parents=True,exist_ok=False)
            for name,rgb in observation["images"].items():imageio.imwrite(folder/f"{name}.png",rgb)
        return observation

    def fail(self,reason):
        self.done=True;self.reason=reason

    def step(self,action,*,raw_model_action=None):
        env=self.env;state=env.state_vector();t=float(env.data.time)
        before_metrics=env.metrics()
        raw_action=action.copy() if raw_model_action is None else raw_model_action.copy()
        if self.action_space=='cartesian':
            action=self.action_filter(action)
            action=limit_pose_step(action,state,self.cartesian_speed_limit*CONTROL_DT,self.angular_speed_limit*CONTROL_DT)
        joint_projection=0.;ik_residual=None
        try:
            if self.action_space=='joint':
                joints,opening,forward,yaw,joint_projection=joint_action_to_targets(env,action)
                self.max_joint_limit_projection=max(self.max_joint_limit_projection,joint_projection)
                self.joint_limit_projected_frames+=int(joint_projection>0)
            else:
                joints,opening,forward,yaw=cartesian_to_joint_target(env,action,max_ik_residual=self.ik_max_residual)
                ik_residual=env.last_cartesian_ik_residual
        except ValueError as error:
            if self.action_space=='cartesian':
                self.max_attempted_ik_residual=max(self.max_attempted_ik_residual,getattr(env,"last_cartesian_ik_residual",0.))
            self.fail(str(error));return
        if ik_residual is not None:self.max_attempted_ik_residual=max(self.max_attempted_ik_residual,ik_residual)
        for cam,writer in self.writers.items():writer.append_data(env.render(cam))
        self.rows.append(dict(time=t,state=state,action=action.copy(),raw_action=raw_action,ik_residual=ik_residual,joint_projection=joint_projection,slip=before_metrics["bottle_slip_m"],
                              tilt=before_metrics["bottle_tilt_rad"],contacts=before_metrics["grip_contacts"]))
        if self.action_space=='joint':self.rows[-1]['joint_state']=robot_joint_observation(env)
        _,metrics=env.step_joints(joints,opening=opening,forward=forward,yaw=yaw)
        if self.action_space=='joint':self.rows[-1]['joint_command']=bounded_joint_command(env)
        lifted=float(env.data.xpos[env.cylinder_body,2]-self.initial_cylinder[2]);self.max_lift=max(self.max_lift,lifted)
        if not np.isfinite(env.data.qpos).all():self.fail("nonfinite_state");return
        if metrics["forbidden_contacts"]:self.fail(f"collision:{metrics['forbidden_contacts']}");return
        if self.progress_router is not None:self.update_progress()
        if not self.held:
            self.contact_seconds=self.contact_seconds+CONTROL_DT if metrics["grip_contacts"]==2 and opening<.02 else 0.
            if self.contact_seconds>=.25:
                self.held=True;env.grasp_point_reference=env.grip_point_relative()
        else:
            self.max_slip=max(self.max_slip,metrics["bottle_slip_m"])
            self.max_tilt=max(self.max_tilt,metrics["bottle_tilt_rad"])
            self.lost_seconds=self.lost_seconds+CONTROL_DT if metrics["grip_contacts"]<2 else 0.
            if self.lost_seconds>.2 or self.max_slip>.025:self.fail("lost_grasp");return
            if metrics["bottle_tilt_rad"]>.12:self.fail("cylinder_tilt");return
        # Evaluation-only gates: they never generate or correct an action.
        center=env.data.xpos[env.cylinder_body]+env.data.xmat[env.cylinder_body].reshape(3,3)@np.array([0,0,.125])
        eye=env.data.cam_xpos[env.model.camera("face_view").id]
        retreat=float(self.initial_base[1]-env.base_pose()[1])
        dof=env.model.joint("cylinder_free").dofadr[0]
        linear_speed=float(np.linalg.norm(env.data.qvel[dof:dof+3]))
        angular_speed=float(np.linalg.norm(env.data.qvel[dof+3:dof+6]))
        still=(linear_speed<=.005 and angular_speed<=.03)
        self.task_gates=dict(held=self.held,lifted=self.max_lift>=.04,retreated=retreat>=.20,
            close_to_head=bool(np.linalg.norm(center-eye)<.45),upright=bool(metrics["bottle_tilt_rad"]<=np.deg2rad(3)),
            still=still,base_retreat_m=retreat,camera_distance_m=float(np.linalg.norm(center-eye)),
            linear_speed_m_s=linear_speed,angular_speed_rad_s=angular_speed)
        eligible=(self.held and self.max_lift>=.04 and retreat>=.20 and np.linalg.norm(center-eye)<.45
                  and metrics["bottle_tilt_rad"]<=np.deg2rad(3) and still)
        if env.data.time>=self.next_read:
            self.next_read=float(env.data.time)+.25
            reading=read_volume(env.render("face_view")) if eligible else dict(valid=False,reason="physical_task_gates_not_met")
            self.readings.append(dict(time=float(env.data.time),**reading))
            good=(eligible and reading["valid"] and abs(reading["eye_alignment_error_px"])<=2
                  and reading.get("boundary_support_fraction",0)>=.5)
            if good:self.good_readings.append(reading["volume_ml"])
            else:self.good_readings=[]
            if len(self.good_readings)>=3 and np.ptp(self.good_readings[-3:])<=.4:
                if abs(self.good_readings[-1]-env.volume_ml)<=.5:self.done=True;self.success=True
                else:self.fail("reading_inaccurate")
        if env.data.time>=self.max_seconds and not self.done:self.fail("timeout")

    def finish(self):
        env=self.env
        for writer in self.writers.values():writer.close()
        rgb=env.render("face_view");reading=read_volume(rgb)
        imageio.imwrite(self.output/"final_face.png",rgb)
        summary=dict(policy="wall_oss_full_cylinder",success=self.success,failure_reason=self.reason,
            action_space=self.action_space,
            evaluation_interrupted=self.evaluation_interrupted,source_sha256=SOURCE_SHA256,
            final_task_gates=self.task_gates,
            action_filter_seconds=self.action_filter.seconds,
            temporal_ensemble_predictions=self.temporal_ensemble.max_predictions,
            prediction_samples_per_observation=self.prediction_samples,
            state_embedding_precision=self.state_embedding_precision,
            unclipped_robot_state=self.unclipped_robot_state,
            gripper_opening_confirmation=dict(forecasts=self.opening_guard.confirmations,
                mode=self.opening_guard.mode,
                abrupt_increase_threshold_m_per_finger=.005,decisions=self.opening_guard.decisions),
            sampled_action_decoding=("single model draw" if self.prediction_samples==1 else
                                     "same-observation independent flow draws averaged; rotations projected to SO(3)"),
            robot_servo_envelope=dict(cartesian_speed_limit_m_s=self.cartesian_speed_limit,
                                      angular_speed_limit_rad_s=self.angular_speed_limit),
            ik_projection=(dict(applicable=False,reason="Direct joint position targets; no Cartesian IK is called")
                           if self.action_space=='joint' else dict(applicable=True,max_allowed_weighted_residual=self.ik_max_residual,
                               max_attempted_residual=self.max_attempted_ik_residual,
                               metric="norm([tool_position_error_m,0.25*rotation_error_rad]); official joint limits preserved")),
            joint_limit_projection=dict(maximum_rad=self.max_joint_limit_projection,
                frames=self.joint_limit_projected_frames,margin_rad=.005),
            grip_profile="official",case=self.case,simulated_seconds=float(env.data.time),
            max_slip_m=self.max_slip,max_held_tilt_rad=self.max_tilt,max_lifted_m=self.max_lift,
            expected_volume_ml=env.volume_ml,reading=reading,reading_history=self.readings,
            final_tilt_rad=env.metrics()["bottle_tilt_rad"],geometry_audit=self.geometry_report,
            inference_seconds=self.latencies,
            executed_dimensions=(list(range(9)) if self.action_space=='joint' else list(range(10,21))+[22]),
            ignored_dimensions=(list(range(9,26)) if self.action_space=='joint' else list(range(10))+[21]),
            observation_inputs=("Three RGB cameras, 26D joint/servo state and a prompt selected from measured robot progress; no simulator object state, teacher phase or fill value"
                                if self.progress_router is not None else
                                ("Three RGB cameras, joint/servo state and initial RGB wrist localization" if self.initial_wrist_cue is not None else
                                "Three RGB cameras and 26D joint/servo state" if self.joint_servo_state else
                                "Three RGB cameras and 18D robot joint positions/velocities, aperture, base velocity/odometry" if self.action_space=='joint' else
                                "Three RGB cameras and 26D robot proprioception including base x/y/yaw" if self.with_odometry else
                                "Three RGB cameras and 23D robot proprioception")+"; no simulator object state, phase or fill value"),
            initial_wrist_cue=self.initial_wrist_cue,
            base_odometry=self.with_odometry,
            exact_input_images_recorded=bool(self.writers),
            low_level_controller=("Official joint limits with original5mrad margin and rate-limited joint position servos; no object-pose correction"
                if self.action_space=='joint' else "Generic robot IK, joint-rate limits and optional robot-feedback grip-force closure; no object-pose correction"))
        if self.progress_router is not None:
            summary.update(progress_protocol=PROGRESS_PROTOCOL,prompt_changes=self.progress_changes,
                           progress_initial_joint_command=self.progress_initial_joint_command)
        if self.tray_cue_protocol:
            summary.update(tray_cue_protocol=self.tray_cue_protocol,initial_tray_cue=self.initial_tray_cue,
                           initial_tray_images=self.initial_tray_images,
                           observation_inputs="Three RGB cameras, measured robot progress and joint/servo state "
                           "with the reset RGB tray cue projected along the current base axes during approach")
        if reading["valid"]:summary["absolute_reading_error_ml"]=abs(reading["volume_ml"]-env.volume_ml)
        (self.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        np.save(self.output/"predicted_chunks.npy",np.asarray(self.chunks))
        if self.prediction_samples>1:
            np.save(self.output/"sampled_chunks.npy",np.asarray(self.sampled_chunks))
        archive=dict(timestamp=np.array([r["time"] for r in self.observations]),
                     state=np.array([r["state"] for r in self.observations]))
        if self.progress_router is not None:
            archive['instruction']=np.asarray([r['instruction'] for r in self.observations],dtype=str)
        np.savez_compressed(self.output/"model_observations.npz",**archive)
        # Privileged simulator snapshots are separate from policy inputs. They
        # allow later expert-recovery demonstrations from policy-visited states.
        np.savez_compressed(self.output/"recovery_physics.npz",state_spec=self.recovery_spec,
            full_state=np.array([r["physics"] for r in self.recovery_states]),
            hold_actuator_ids=self.hold_actuator_ids,
            hold_targets=np.array([r["hold_targets"] for r in self.recovery_states]),
            applied_base_action=np.array([r["applied_base_action"] for r in self.recovery_states]))
        extra=(dict(ik_residual=np.array([r['ik_residual'] for r in self.rows])) if self.action_space=='cartesian' else
               dict(joint_limit_projection_rad=np.array([r['joint_projection'] for r in self.rows]),
                    robot_joint_state=np.array([r['joint_state'] for r in self.rows]),
                    joint_command=np.array([r['joint_command'] for r in self.rows])))
        np.savez_compressed(self.output/"trajectory.npz",timestamp=np.array([r["time"] for r in self.rows]),
            state=np.array([r["state"] for r in self.rows]),action=np.array([r["action"] for r in self.rows]),
            unfiltered_model_action=np.array([r["raw_action"] for r in self.rows]),
            slip=np.array([r["slip"] for r in self.rows]),tilt=np.array([r["tilt"] for r in self.rows]),
            contacts=np.array([r["contacts"] for r in self.rows]),phase=np.array(["wall_policy"]*len(self.rows)),**extra)
        env.close();return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,required=True);p.add_argument("--config",type=Path)
    p.add_argument("--norm-key",default="ex_normal");p.add_argument("--controller-config",type=Path,required=True)
    p.add_argument("--cases",type=Path);p.add_argument("--volume",type=float,default=25);p.add_argument("--seed",type=int,default=100)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--execute-steps",type=int,default=4)
    p.add_argument("--max-seconds",type=float,default=90);p.add_argument("--probe-only",action="store_true")
    p.add_argument("--record",action="store_true");p.add_argument("--image-size",type=int,default=448)
    p.add_argument("--policy-batch-size",type=int,default=1,
                   help="Limit inference microbatch size for Thor memory and latency.")
    p.add_argument("--flow-steps",type=int,default=10,choices=range(1,101))
    p.add_argument("--flow-noise",choices=("random","zero"),default="random",
                   help="Diagnostic deterministic zero-latent decoding versus standard Gaussian sampling.")
    p.add_argument("--action-filter-seconds",type=float,default=0.,
                   help="Optional robot-command low-pass time constant; no object/task feedback.")
    p.add_argument("--cartesian-speed-limit",type=float,default=0.,help="Optional pre-IK robot-feedback translation limit, m/s; 0 disables.")
    p.add_argument("--angular-speed-limit",type=float,default=0.,help="Optional pre-IK robot-feedback rotation limit, rad/s; 0 disables.")
    p.add_argument("--ik-max-residual",type=float,default=.002,
                   help="Allowed weighted nearest-IK residual, m/rad*0.25; physical task gates stay fixed.")
    p.add_argument("--temporal-ensemble",type=int,default=0,choices=range(9),
                   help="Average overlapping forecasts for the same tick; newest gripper retained; 0 disables.")
    p.add_argument("--prediction-samples",type=int,default=1,choices=range(1,9),
                   help="Average independent flow draws at the SAME current observation; 1 preserves baseline decoding.")
    p.add_argument("--state-embedding-precision",choices=("training","native"),default="training",
                   help="Match training's state rounding before projection; native preserves the earlier Wall inference path.")
    p.add_argument("--gripper-open-confirmations",type=int,choices=(0,2,3,4),default=0,
                   help="Optional command ablation: confirm abrupt >5mm finger opening across policy forecasts; 0 disables.")
    p.add_argument("--gripper-open-confirmation-mode",choices=("prefix-median","aligned-forecasts"),default="prefix-median",
                   help="Optional gripper ablation: confirm each future tick using overlapping forecasts instead of prefix medians.")
    p.add_argument("--unclipped-robot-state",action="store_true",
                   help="Optional state-input ablation: preserve the existing affine scale without clipping robot states.")
    args=p.parse_args()
    if not np.isfinite(args.action_filter_seconds) or args.action_filter_seconds<0:
        p.error("Action filter time constant must be finite and nonnegative")
    if any(not np.isfinite(x) or x<0 for x in [args.cartesian_speed_limit,args.angular_speed_limit]):
        p.error("Cartesian servo limits must be finite and nonnegative")
    if not np.isfinite(args.ik_max_residual) or not 0<args.ik_max_residual<=.02:
        p.error("IK residual limit must be in (0,0.02]")
    args.output.mkdir(parents=True,exist_ok=False)
    cases=json.loads(args.cases.read_text()) if args.cases else [dict(volume=args.volume,seed=args.seed,jitter=.004)]
    controller=json.loads(args.controller_config.read_text())
    torch.manual_seed(47)
    policy=CylinderWallPolicy(args.model,args.config or args.model/"config.yml",args.norm_key,
                              flow_steps=args.flow_steps,flow_noise=args.flow_noise,
                              state_embedding_precision=args.state_embedding_precision,
                              unclipped_robot_state=args.unclipped_robot_state)
    if not 1<=args.execute_steps<=policy._action_horizon:p.error("Execution prefix exceeds model horizon")
    if args.policy_batch_size<1:p.error("Policy batch size must be positive")
    if not cases or not np.isfinite(args.max_seconds) or args.max_seconds<=0:p.error("Cases and positive duration required")
    (args.output/"model_load.json").write_text(json.dumps(policy.load_report,indent=2)+"\n")
    trials=[CylinderTrial(case,controller,args.output/f"episode_{index:03d}",record=args.record,
                         max_seconds=args.max_seconds,image_size=args.image_size,
                         action_filter_seconds=args.action_filter_seconds,with_odometry=policy.with_odometry,
                         cartesian_speed_limit=args.cartesian_speed_limit,angular_speed_limit=args.angular_speed_limit,
                         ik_max_residual=args.ik_max_residual,temporal_ensemble=args.temporal_ensemble,
                         prediction_samples=args.prediction_samples,
                         state_embedding_precision=args.state_embedding_precision,
                         gripper_open_confirmations=args.gripper_open_confirmations,
                         gripper_open_confirmation_mode=args.gripper_open_confirmation_mode,
                         action_space=policy.action_space,joint_servo_state=policy.joint_servo_state,
                         joint_wrist_cue=policy.joint_wrist_cue,
                         progress_protocol=policy.progress_protocol,tray_cue_protocol=policy.tray_cue_protocol,
                         unclipped_robot_state=args.unclipped_robot_state) for index,case in enumerate(cases)]
    try:
        while True:
            active=[trial for trial in trials if not trial.done]
            if not active:break
            observations=[trial.observation() for trial in active]
            expanded=[obs for obs in observations for _ in range(args.prediction_samples)]
            draws=[];latencies=[];inference_start=time.monotonic()
            for first in range(0,len(expanded),args.policy_batch_size):
                start=time.monotonic()
                predicted=policy.predict_many(expanded[first:first+args.policy_batch_size]);torch.cuda.synchronize()
                latency=time.monotonic()-start
                draws.extend(predicted)
                latencies.extend([latency if args.prediction_samples==1 else latency/len(predicted)]*len(predicted))
            sample_groups=np.asarray(draws).reshape(len(active),args.prediction_samples,policy._action_horizon,26)
            chunks=[average_model_action_samples(group) for group in sample_groups]
            observation_latencies=np.asarray(latencies).reshape(len(active),args.prediction_samples).sum(1)
            for trial,chunk,samples,latency in zip(active,chunks,sample_groups,observation_latencies):
                trial.chunks.append(chunk);trial.latencies.append(float(latency))
                if args.prediction_samples>1:trial.sampled_chunks.append(samples)
            print(json.dumps(dict(active=len(active),chunk=len(active[0].chunks),inference_seconds=time.monotonic()-inference_start,
                                  action_space=policy.action_space,
                                  first_right_action=(chunks[0][0,:9] if policy.action_space=='joint' else chunks[0][0,10:20]).tolist())),flush=True)
            if args.probe_only:
                for trial in active:trial.fail("probe_only_no_execution")
                break
            for trial,chunk,observation in zip(active,chunks,observations):
                try:absolute=(absolute_joint_chunk(chunk,observation['state']) if policy.action_space=='joint'
                              else relative_to_absolute(chunk,observation["state"][:23]))
                except ValueError as error:trial.fail(str(error));continue
                if policy.action_space=='joint':
                    for action in absolute[:args.execute_steps]:
                        trial.step(action)
                        if trial.done:break
                    continue
                guarded=trial.opening_guard(absolute,args.execute_steps)
                trial.temporal_ensemble.add(int(round(trial.env.data.time/CONTROL_DT)),guarded)
                for raw_action,action in zip(absolute[:args.execute_steps],guarded[:args.execute_steps]):
                    command=trial.temporal_ensemble.command(int(round(trial.env.data.time/CONTROL_DT)),action)
                    trial.step(command,raw_model_action=raw_action)
                    if trial.done:break
            print(json.dumps(dict(case_status=[dict(seed=trial.case["seed"],seconds=float(trial.env.data.time),
                held=trial.held,contacts=trial.env.metrics()["grip_contacts"],max_lifted_m=trial.max_lift,
                max_slip_m=trial.max_slip,task_gates=trial.task_gates,
                done=trial.done,reason=trial.reason) for trial in active])),flush=True)
    except BaseException as error:
        for trial in trials:
            if not trial.done:
                trial.evaluation_interrupted=True;trial.fail(f"evaluation_interrupted:{type(error).__name__}")
        raise
    finally:
        results=[trial.finish() for trial in trials]
        report=dict(model=str(args.model.resolve()),config=str((args.config or args.model/"config.yml").resolve()),
                    action_space=policy.action_space,progress_protocol=policy.progress_protocol,
                    tray_cue_protocol=policy.tray_cue_protocol,
                    controller=controller,probe_only=args.probe_only,episodes=results,
                    execution_prefix=args.execute_steps,policy_batch_size=args.policy_batch_size,
                    flow_steps=args.flow_steps,
                    flow_noise=args.flow_noise,
                    action_filter_seconds=args.action_filter_seconds,
                    robot_servo_envelope=dict(cartesian_speed_limit_m_s=args.cartesian_speed_limit,
                                             angular_speed_limit_rad_s=args.angular_speed_limit),
                    ik_max_residual=(None if policy.action_space=='joint' else args.ik_max_residual),
                    temporal_ensemble_predictions=args.temporal_ensemble,
                    prediction_samples_per_observation=args.prediction_samples,
                    state_embedding_precision=args.state_embedding_precision,
                    gripper_open_confirmations=args.gripper_open_confirmations,
                    gripper_open_confirmation_mode=args.gripper_open_confirmation_mode,
                    unclipped_robot_state=args.unclipped_robot_state,
                    evaluation_complete=all(not r["evaluation_interrupted"] for r in results),
                    successes=sum(r["success"] for r in results),total=len(results))
        (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(dict(successes=report["successes"],total=report["total"])),flush=True)


if __name__=="__main__":main()
