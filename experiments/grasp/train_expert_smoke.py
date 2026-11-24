"""Launch the official single-GPU trainer with explicit compatibility shims.

Run with torchrun --standalone --nproc_per_node=1. Optional intermediate
snapshots support learning-curve evaluation without restarting the optimizer.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

os.environ.setdefault("WALLX_VISION_ATTN_IMPLEMENTATION", "sdpa")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--snapshot-steps", type=int, nargs="*", default=[])
    p.add_argument("--repeat-frame", type=int, help="Diagnostic only: repeatedly train one raw dataset frame")
    p.add_argument("--fp32-master-weights", action="store_true", help="Accumulate AdamW updates in FP32 while preserving BF16 model compute")
    p.add_argument("--initial-action-expert", type=Path,
                   help="Initialize trainable action tensors from an evaluation overlay; optimizer starts fresh")
    p.add_argument("--dataset-plan", type=Path,
                   help="Verify actual episode/frame membership against a prepared source-group split")
    args = p.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from runtime_compat import configure_runtime
    configure_runtime()
    import torch
    from wall_x.trainer.fsdp_trainer.distribution_strategy import DistributionStrategy
    from wall_x.trainer.fsdp_trainer.fsdp_trainer import FSDPTrainer
    from wall_x.trainer.adapters import resolve_adapter
    from action_checkpoint import export_action_checkpoint

    # Retain the checkpoint's padded vocabulary; tokenizer IDs remain identical.
    cls = resolve_adapter("qwen2_5").model_class()
    resize = cls.resize_token_embeddings
    def preserve_vocabulary(self, new_num_tokens=None, *a, **kw):
        if new_num_tokens is not None and new_num_tokens < self.get_input_embeddings().num_embeddings:
            return self.get_input_embeddings()
        kw.setdefault("mean_resizing", False)
        return resize(self, new_num_tokens, *a, **kw)
    cls.resize_token_embeddings = preserve_vocabulary

    initialization = {}
    if args.initial_action_expert is not None:
        from safetensors.torch import load_file
        import hashlib
        original_load_state = FSDPTrainer.load_state_dict
        def load_base_and_action_expert(self, model, resume_config):
            result = original_load_state(self, model, resume_config)
            if initialization:
                raise RuntimeError("Action initialization must run exactly once, before the optimizer")
            overlay = load_file(str(args.initial_action_expert), device="cpu")
            trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
            if set(overlay) != set(trainable):
                raise RuntimeError("Initial overlay must match the complete trainable action-expert parameter set")
            assert all(overlay[name].shape == parameter.shape for name, parameter in trainable.items())
            with torch.no_grad():
                for name, parameter in trainable.items():
                    value = overlay[name].to(device=parameter.device, dtype=parameter.dtype)
                    parameter.copy_(value)
                    assert torch.equal(parameter, value)
            digest = hashlib.sha256()
            with args.initial_action_expert.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            description_path = args.initial_action_expert.with_name("adaptation.json")
            description = json.loads(description_path.read_text()) if description_path.exists() else {}
            initialization.update(overlay_sha256=digest.hexdigest(), parameter_tensors=len(trainable),
                trainable_parameters=sum(p.numel() for p in trainable.values()),
                all_parameters_match_before_optimizer=True, optimizer_state_restored=False,
                source_completed_updates=description.get("logical_total_weight_updates", description.get("global_step")))
            self.action_expert_initialization = dict(initialization)
            print("ACTION_EXPERT_INITIALIZATION " + json.dumps(initialization), flush=True)
            del overlay
            return result
        FSDPTrainer.load_state_dict = load_base_and_action_expert

    if args.fp32_master_weights:
        from master_optimizer import FP32MasterAdamW
        original_optimizer = FSDPTrainer._create_optimizer
        def optimizer_with_masters(self):
            if self.cfg.distributed.use_fsdp or not self.cfg.distributed.bf16:
                raise ValueError("FP32 master wrapper is restricted to this single-GPU BF16 DDP experiment")
            if Path(self.cfg.checkpoint.resume_from).is_dir():
                raise ValueError("FP32-master experiments start from a base weight file; optimizer resume is unsupported")
            original_optimizer(self)
            self.optimizer = FP32MasterAdamW(self.optimizer)
            print("FP32_MASTER_WEIGHTS enabled with original model compute dtypes", flush=True)
        FSDPTrainer._create_optimizer = optimizer_with_masters

    if args.repeat_frame is not None:
        original_load_dataset = FSDPTrainer.load_dataset
        def load_one_frame(self):
            original_load_dataset(self)
            inner = self.dataset._inner
            previous = inner._dataset
            try:
                inner._dataset = inner.hf_dataset
                sample = inner[args.repeat_frame]
            finally:
                inner._dataset = previous
            collate = self.train_dataloader.collate_fn
            count = self.num_training_steps
            class RepeatedFrameLoader:
                def __len__(self):
                    return count

                def __iter__(self):
                    for _ in range(count):
                        # Resample flow time every iteration, as in normal training.
                        yield collate([sample])
            loader = RepeatedFrameLoader()
            self.train_dataloader = loader
            self.train_num = count
            self.dataset.set_epoch = lambda epoch: None
            self.dataset.get_train_dataloader = lambda: loader
            self.dataset.global_train_iters.value = count
            print(f"FIXED_FRAME_DIAGNOSTIC frame={args.repeat_frame} updates={count}", flush=True)
        FSDPTrainer.load_dataset = load_one_frame

    if args.dataset_plan is not None:
        if args.repeat_frame is not None:
            raise ValueError("Dataset membership auditing is incompatible with repeated-frame diagnostics")
        from collections import Counter
        original_dataset_load = FSDPTrainer.load_dataset
        def load_and_audit_membership(self):
            original_dataset_load(self)
            plan = json.loads(args.dataset_plan.read_text())
            inner = self.dataset._inner
            loaded = inner.hf_dataset
            actual = list(loaded.episodes)
            assert actual == plan["training_episode_ids"], "Episode selection differs from the source-group plan"
            episode_ids = list(loaded.hf_dataset.select_columns(["episode_index"]).with_format(None)["episode_index"])
            counts = Counter(int(episode_ids[int(i)]) for i in inner.train_dataset.indices)
            assert all(counts[e] > 0 for e in plan.get("corrective_episode_ids", [])), (
                "Every corrective demonstration must contribute gradient-training frames")
            audit = {"passed": True, "selected_training_episodes": actual,
                "training_frames": len(inner.train_dataset), "internal_validation_frames": len(inner.val_dataset),
                "gradient_frames_by_episode": dict(sorted(counts.items())),
                "corrective_episode_ids": plan.get("corrective_episode_ids", []),
                "all_corrective_episodes_in_gradient_training": True,
                "source_scene_families_disjoint": plan["no_source_family_overlap"]}
            args.report.parent.mkdir(parents=True, exist_ok=True)
            (args.report.parent / "dataset_membership.json").write_text(json.dumps(audit, indent=2))
            print("DATASET_MEMBERSHIP " + json.dumps(audit), flush=True)
        FSDPTrainer.load_dataset = load_and_audit_membership

    snapshot = {}
    intermediate = []
    requested_steps = set(args.snapshot_steps)
    original_metrics = FSDPTrainer._log_training_metrics
    def metrics_and_snapshot(self, epoch, step, total, loss, total_norm, outputs):
        original_metrics(self, epoch, step, total, loss, total_norm, outputs)
        completed = self.global_step + 1  # Official callback runs after optimizer.step.
        row = {"completed_steps": completed, "epoch": epoch,
               "loss": float(loss.detach()), "grad_norm": float(total_norm),
               "wall_time": time.time()}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with (args.report.parent/"learning_curve.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if completed in requested_steps and completed < self.num_training_steps:
            base = Path(self.cfg.checkpoint.resume_from).parent
            folder = args.report.parent/f"step_{completed:06d}"/"evaluation"
            # Export records the trainer counter, so expose completed updates
            # temporarily and restore it before the official loop increments.
            original_step = self.global_step
            try:
                self.global_step = completed
                description = export_action_checkpoint(self, base, folder)
            finally:
                self.global_step = original_step
            intermediate.append(description)
            print("EVALUATION_SNAPSHOT " + json.dumps(description), flush=True)
    FSDPTrainer._log_training_metrics = metrics_and_snapshot
    original_fit = FSDPTrainer.fit
    def fit_and_export(self):
        original_fit(self)
        base = Path(self.cfg.checkpoint.resume_from).parent
        snapshot.update(export_action_checkpoint(self, base, args.report.parent/"evaluation"))
    FSDPTrainer.fit = fit_and_export

    # Explicit non-reentrant checkpointing supports DDP with frozen parameters.
    def checkpointing(self, model):
        if self._cfg.get("use_gradient_checkpointing", False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    DistributionStrategy._maybe_enable_grad_ckpt = checkpointing
    from wall_x.trainer.fsdp_trainer.train_fsdp import main as official_main
    sys.argv = [sys.argv[0], "--config", str(args.config.resolve())]
    report = {"completed": False, "trainer": "official Wall-X", "method": "action expert only, no LoRA",
              "torch_version": torch.__version__, "native_sdpa": True,
              "repeated_frame_diagnostic": args.repeat_frame,
              "fp32_master_weights": args.fp32_master_weights}
    report["action_expert_initialization"] = initialization
    try:
        official_main()
        report["completed"] = True
        report["snapshot"] = snapshot
        report["intermediate_snapshots"] = intermediate
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if torch.cuda.is_initialized():
            report["peak_allocated_gb"] = torch.cuda.max_memory_allocated()/1e9
            report["peak_reserved_gb"] = torch.cuda.max_memory_reserved()/1e9
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
