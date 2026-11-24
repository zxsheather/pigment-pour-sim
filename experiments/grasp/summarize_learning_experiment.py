"""Render measured learning curves and a portable summary of the experiment."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--additional-experiment", type=Path)
    args = p.parse_args()
    trace = [json.loads(line) for line in (args.experiment/"learning_curve.jsonl").read_text().splitlines()]
    compared = read(args.evaluation/"comparison.json")
    for item in compared:
        prefix=item.get("tag",f"{item['steps']:06d}")
        opened=read(args.evaluation/f"openloop_{prefix}/report.json")
        closed=read(args.evaluation/f"closedloop_{prefix}/report.json")
        orientation={}
        for split in ["train_episode","unused_internal_episode"]:
            errors=[]
            maxima=[]
            for r in opened["records"]:
                if r["split"]!=split:
                    continue
                pred=np.asarray(r["prediction"])
                target=np.asarray(r["target"])
                count=min(8,len(target))
                delta=(Rotation.from_euler("xyz",pred[:count,3:6])
                       *Rotation.from_euler("xyz",target[:count,3:6]).inv())
                errors.extend(np.degrees(delta.magnitude()).tolist())
            for c in closed["cases"]:
                if c["split"]!=split:
                    continue
                rows=read(args.evaluation/f"closedloop_{prefix}"/c["directory"]/"telemetry.json")
                pose=np.asarray([r["right_ee"][3:6] for r in rows])
                maxima.append(float(np.degrees(Rotation.from_euler("xyz",pose).magnitude()).max()))
            orientation[split]={"execute8_angle_error_deg":float(np.mean(errors)),
                                "mean_max_executed_angle_from_expert_deg":float(np.mean(maxima))}
        item["orientation_diagnostics"]=orientation
    steps = np.array([row["completed_steps"] for row in trace])
    losses = np.array([row["loss"] for row in trace])
    points = [row["steps"] for row in compared]
    colors = {"train_episode":"#226d5a", "unused_internal_episode":"#c27b28"}
    labels = {"train_episode":"2 training-pool positions", "unused_internal_episode":"3 unused internal positions"}
    def plot_metric(ax, metric, legend=False):
        for split,color in colors.items():
            continuous=[r for r in compared if r["steps"]!=100 and r.get("variant","original")=="original"]
            ax.plot([r["steps"] for r in continuous],[metric(r,split) for r in continuous],"o-",color=color,label=labels[split])
            for r in compared:
                if r.get("variant","original")!="original":
                    ax.scatter(r["steps"],metric(r,split),marker="D",s=65,color=color,edgecolors="#172b25")
                elif r["steps"]==100:
                    ax.scatter(100,metric(r,split),marker="x",s=60,color=color)
        if legend:
            ax.legend(fontsize=8)
    with plt.rc_context({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "figure.facecolor":"#f7f8f1","axes.facecolor":"#fffef8"}):
        fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
        fig.suptitle("Wall-OSS-0.5 | Fixed-data action-expert training", fontsize=17, fontweight="bold")
        ax = axes[0,0]
        window = min(50, len(losses))
        ax.plot(steps, losses, color="#9bbdb0", alpha=.35, linewidth=.6)
        ax.plot(steps[window-1:], np.convolve(losses,np.ones(window)/window,mode="valid"), color="#226d5a", linewidth=1.5,label="Original BF16 updates")
        if args.additional_experiment:
            extra=[json.loads(line) for line in (args.additional_experiment/"learning_curve.jsonl").read_text().splitlines()]
            extra_steps=np.array([r["completed_steps"] for r in extra])
            extra_losses=np.array([r["loss"] for r in extra])
            extra_window=min(50,len(extra))
            ax.plot(extra_steps[extra_window-1:],np.convolve(extra_losses,np.ones(extra_window)/extra_window,mode="valid"),color="#3f72a8",linewidth=1.5,label="FP32 master updates")
            ax.legend(fontsize=8)
        ax.set(title=f"Training loss ({window}-step moving average)", xlabel="Completed optimizer updates", ylabel="Flow-matching loss", yscale="log")
        ax = axes[0,1]
        plot_metric(ax,lambda r,s:r["openloop"][s]["execute8_xyz_cm"],legend=True)
        ax.set(title="Action position error on expert observations",xlabel="Optimizer updates",ylabel="Mean XYZ distance over first 8 actions (cm)",ylim=(0,None))
        ax = axes[1,0]
        plot_metric(ax,lambda r,s:r["openloop"][s]["execute8_gripper_units"])
        ax.set(title="Gripper error on expert observations",xlabel="Updates (x: earlier reference; diamond: FP32 masters)",ylabel="Mean absolute error (motor units)",ylim=(0,None))
        ax = axes[1,1]
        plot_metric(ax,lambda r,s:r["orientation_diagnostics"][s]["execute8_angle_error_deg"])
        ax.set(title="Orientation error on expert observations",xlabel="Optimizer updates",ylabel="Mean angle error over first 8 actions (degrees)",ylim=(0,None))
        ax = axes[2,1]
        plot_metric(ax,lambda r,s:r["orientation_diagnostics"][s]["mean_max_executed_angle_from_expert_deg"])
        ax.set(title="Executed tool orientation during model rollouts",xlabel="Optimizer updates",ylabel="Maximum rotation from expert pose (degrees)\n(mean across positions)",ylim=(0,180))
        ax = axes[2,0]
        indices=np.arange(len(compared))
        for offset,(split,color) in zip([-.18,.18],colors.items()):
            metrics=[r["closedloop"][split] for r in compared]
            values=[m["successes"]/m["trials"] for m in metrics]
            bars=ax.bar(indices+offset,values,width=.34,color=color,label=labels[split])
            for bar,m in zip(bars,metrics):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,
                        f"{m['successes']}/{m['trials']}",ha="center",fontsize=9)
        ax.set(title="Model-only closed-loop grasping",ylabel="Stable grasps / position trials",ylim=(0,1.15))
        ax.set_xticks(indices,[str(r["steps"])+( "\nFP32" if r.get("variant","original")!="original" else "") for r in compared])
        ax.set_xlabel("Optimizer updates (100-step reference is a separate earlier run)")
        for ax in axes.flat:
            ax.grid(axis="y",alpha=.15)
        fig.savefig(args.evaluation/"learning_curve.png",dpi=180)
        plt.close(fig)
    summary = {"training_steps":int(steps[-1]),"training_parameters":399754240,
               "loss_first50_mean":float(losses[:50].mean()),"loss_last50_mean":float(losses[-50:].mean()),
               "wall_training_seconds_excluding_initialization":float(trace[-1]["wall_time"]-trace[0]["wall_time"]),
               "checkpoints":compared}
    if (args.evaluation/"final_summary.json").exists():
        summary["final_evaluation"]=read(args.evaluation/"final_summary.json")
    if args.additional_experiment:
        summary["additional_training"]={"steps":len(extra),"loss_first50_mean":float(extra_losses[:50].mean()),"loss_last50_mean":float(extra_losses[-50:].mean())}
    (args.evaluation/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ["checkpoints","final_evaluation"]}))


if __name__ == "__main__":
    main()
