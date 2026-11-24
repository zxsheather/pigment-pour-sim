from pathlib import Path
import os,signal,subprocess,json,time
os.chdir('/data/hlou/pigment-pour-study')
os.environ.update(PYTHONPATH='/data/hlou/pigment-pour-study/wall-x',MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
name='cylinder-hold-correction384-aligned-grip-v1';queue=4037304;child=None;paused=False
status=Path('results')/(name+'-status.json')
row=dict(diagnostic_only=True,extra_controller='2 overlapping forecasts agreeing to open at the same future tick',formal_default_training_queue_unchanged=True,physical_task_gates_unchanged=True,ik_limit=.002,not_an_unfiltered_policy_result=True,started=time.time())
def save():status.write_text(json.dumps(row,indent=2)+'\n')
def stop(signum,frame):raise SystemExit(128+signum)
for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
try:
    cmd=Path(f'/proc/{queue}/cmdline').read_bytes()
    assert b'cylinder-hold-correction-bc-v1-evaluations.py' in cmd,'Queue identity mismatch'
    validation=json.loads(Path('runs/cylinder-hold-correction-step384-single-v1/aligned-gripper-validation.json').read_text());assert validation['passed']
    os.kill(queue,signal.SIGSTOP);paused=True;row['queue_temporarily_stopped']=queue;save()
    children=subprocess.run(['ps','--ppid',str(queue),'-o','pid='],capture_output=True,text=True).stdout.strip();assert not children,f'Formal evaluator already active: {children}'
    argv=['.venv-wall/bin/python','pigment-pour-sim/simulation/run_wall_cylinder.py','--model','runs/cylinder-hold-correction-bc-v1/checkpoints/0_384','--config','runs/cylinder-hold-correction-bc-v1/train.yml','--norm-key','cylinder_official_v2','--controller-config','pigment-pour-sim/simulation/configs/cylinder_official_teacher_dev_v1.json','--volume','25','--seed','100','--output','runs/'+name,'--execute-steps','8','--max-seconds','90','--ik-max-residual','.002','--gripper-open-confirmations','2','--gripper-open-confirmation-mode','aligned-forecasts','--record']
    child=subprocess.Popen(argv);row.update(stage='actual_neural_closed_loop',argv=argv,pid=child.pid);save();print('START_ACTUAL_NEURAL_WITH_ALIGNED_GRIP',child.pid,flush=True)
    row.update(exit_code=child.wait(),finished=time.time());save()
finally:
    try:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=10)
    finally:
        if paused:
            cmd=Path(f'/proc/{queue}/cmdline').read_bytes()
            assert b'cylinder-hold-correction-bc-v1-evaluations.py' in cmd
            os.kill(queue,signal.SIGCONT);row.update(queue_resumed=True);save();print('RESUMED_FORMAL_QUEUE',queue,flush=True)
