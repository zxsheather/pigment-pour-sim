"""Build a local gallery of accepted reactive-teacher recordings.

Needs only Python's standard library and ffmpeg. No source recording is altered.
The default output is served beneath the existing demonstration gallery.
"""
from pathlib import Path
import argparse
import concurrent.futures
import json
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("experiments/grasp/runs/reactive_train_v4"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/grasp/runs/demo_gallery_v1/reactive-v4"))
    args = parser.parse_args()
    manifest = json.loads((args.input / "manifest.json").read_text())
    assert manifest["controller"] == "privileged reactive expert"
    assert not manifest["model_executed"] and not manifest["grasp_weld"]
    assert not manifest["setup_commands_in_supervision"]
    accepted = [e for e in manifest["episodes"] if e["accepted"]]
    assert len(accepted) == 32 and all(e["numerically_valid"] for e in accepted)
    (args.output / "media").mkdir(parents=True, exist_ok=True)

    def render(entry):
        source = args.input / f"episode_{entry['episode']:04d}"
        stem = f"episode-{entry['episode']:04d}"
        video = args.output / "media" / (stem + ".mp4")
        if not video.exists():
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-threads", "1", "-i", str(source / "rollout.mp4"),
                "-vf", "setpts=0.8*PTS", "-r", "25", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "21", "-threads", "2",
                "-movflags", "+faststart", str(video)], check=True)
        shutil.copy2(source / "initial.png", args.output / "media" / (stem + ".png"))
        return {"episode": entry["episode"], "kind": entry["kind"],
                "video": "media/" + video.name, "poster": "media/" + stem + ".png",
                "frames": entry["frames"], "seconds": round(entry["simulated_seconds"], 2),
                "holdSeconds": round(entry["max_held_seconds"], 3),
                "liftCm": round(entry["max_bottle_bottom_lift_m"] * 100, 2),
                "offsetMm": [round(x * 1000, 1) for x in entry["offset_xy_m"]]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        episodes = list(pool.map(render, accepted))
    data = {"episodes": episodes, "attempted": len(manifest["episodes"]),
            "accepted": len(accepted), "frames": sum(e["frames"] for e in accepted),
            "teacherSourceSha256": manifest["teacher_source_sha256"]}
    template = Path(__file__).with_name("reactive_gallery.html").read_text()
    (args.output / "index.html").write_text(template.replace(
        "__REACTIVE_DATA__", json.dumps(data).replace("<", "\\u003c")))
    (args.output / "metadata.json").write_text(json.dumps(data, indent=2))
    print(json.dumps({"gallery": str(args.output / "index.html"),
                      "episodes": len(episodes), "frames": data["frames"]}))


if __name__ == "__main__":
    main()
