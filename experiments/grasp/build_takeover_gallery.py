"""Build paired model-prefix / teacher-correction previews from accepted data."""
from pathlib import Path
import argparse
import concurrent.futures
import json
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/grasp/runs/takeover_precontact_v5"))
    parser.add_argument("--output", type=Path, default=Path("experiments/grasp/runs/demo_gallery_v1/takeover-v5"))
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    assert manifest["source_split"] == "train_episode only"
    assert not manifest["used_for_xyz_normalization_training"]
    accepted = [e for e in manifest["episodes"] if e["accepted"]]
    (args.output / "media").mkdir(parents=True, exist_ok=True)

    def render(entry):
        folder = args.input / f"episode_{entry['episode']:04d}"
        stem = f"takeover-{entry['episode']:04d}"
        for source, suffix in [("student_prefix.mp4", "model"), ("rollout.mp4", "teacher")]:
            target = args.output / "media" / f"{stem}-{suffix}.mp4"
            if not target.exists():
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1",
                    "-i", str(folder / source), "-vf", "setpts=0.8*PTS", "-r", "25", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-threads", "2",
                    "-movflags", "+faststart", str(target)], check=True)
        shutil.copy2(folder / "initial.png", args.output / "media" / (stem + ".png"))
        return {"id": entry["episode"], "scene": entry["source_episode"], "prefix": entry["prefix_steps"],
                "initialTilt": round(entry["initial_bottle_tilt_deg"], 2),
                "frames": entry["frames"], "seconds": round(entry["simulated_seconds"], 2),
                "held": round(entry["max_held_seconds"], 3),
                "model": f"media/{stem}-model.mp4", "teacher": f"media/{stem}-teacher.mp4",
                "poster": f"media/{stem}.png"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        episodes = list(pool.map(render, accepted))
    data = {"attempts": len(manifest["episodes"]), "accepted": len(accepted),
            "frames": sum(e["frames"] for e in accepted), "episodes": episodes}
    template = Path(__file__).with_name("takeover_gallery.html").read_text()
    (args.output / "index.html").write_text(template.replace("__TAKEOVER_DATA__", json.dumps(data).replace("<", "\\u003c")))
    (args.output / "metadata.json").write_text(json.dumps(data, indent=2))
    print(json.dumps({"gallery": str(args.output / "index.html"), "accepted": len(accepted)}))


if __name__ == "__main__":
    main()
