#!/usr/bin/env python3
"""Build an offline gallery from the archived scripted-expert recordings.

Requires Python's standard library, ffmpeg and ffprobe. Generated media stays
under runs/. Original recordings and dataset archives are never modified.
"""

import argparse
import concurrent.futures
import functools
import http.server
import json
import re
import shutil
import struct
import subprocess
import tarfile
import zlib
from pathlib import Path


BATCHES = ("expert_pilot_v1", "expert_train_extra_v1", "expert_heldout_v1")


def run(command):
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def read_metadata(archive):
    episodes = []
    with tarfile.open(archive) as dataset:
        for batch in BATCHES:
            manifest = json.load(dataset.extractfile(f"{batch}/manifest.json"))
            assert manifest["controller"] == "privileged scripted expert"
            assert manifest["model_executed"] is False
            assert manifest["grasp_weld"] is False
            for item in manifest["episodes"]:
                name = f"{batch}/episode_{item['episode']:04d}"
                telemetry = json.load(dataset.extractfile(f"{name}/telemetry.json"))
                index = len(episodes) + 1
                episodes.append({
                    "id": index,
                    "source": name,
                    "split": "heldout" if batch == BATCHES[2] else "train",
                    "offset": [round(x * 1000, 3) for x in item["offset_xy_m"]],
                    "liftCm": round(item["max_bottle_bottom_lift_m"] * 100, 3),
                    "holdSeconds": round(max(x["held_seconds"] for x in telemetry), 3),
                    "accepted": item["accepted"],
                    "valid": item["numerically_valid"],
                    "frames": item["frames"],
                    "simSeconds": round(item["simulated_seconds"], 3),
                    "video": f"media/demo-{index:02d}.mp4",
                    "poster": f"posters/demo-{index:02d}.jpg",
                })
    assert len(episodes) == 32
    assert sum(e["split"] == "heldout" for e in episodes) == 8
    assert all(e["accepted"] and e["valid"] for e in episodes)
    return episodes


def extract_recordings(archive, episodes, output):
    with tarfile.open(archive) as videos:
        for episode in episodes:
            member = videos.getmember(f"{episode['source']}/rollout.mp4")
            target = output / "source" / f"demo-{episode['id']:02d}.mp4"
            if target.exists() and target.stat().st_size == member.size:
                continue
            with videos.extractfile(member) as src, target.open("wb") as dest:
                shutil.copyfileobj(src, dest)


def encode_episode(episode, output):
    video = output / episode["video"]
    poster = output / episode["poster"]
    if not video.exists():
        # Source playback was 20 fps for 25 Hz simulation. Restore simulation
        # speed; initial/final endpoint images add two frames (0.08 seconds).
        run(["ffmpeg", "-v", "error", "-y", "-threads", "1", "-i",
             str(output / "source" / video.name), "-vf", "setpts=0.8*PTS",
             "-r", "25", "-an", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "21", "-threads", "2", "-movflags", "+faststart", str(video)])
    if not poster.exists():
        run(["ffmpeg", "-v", "error", "-y", "-ss", "5.3", "-i", str(video),
             "-frames:v", "1", "-q:v", "3", str(poster)])
    return episode["id"]


# A tiny bitmap font keeps montage numbering independent of platform fonts and
# optional ffmpeg text libraries. The transparent overlay is drawn from scratch.
GLYPHS = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "|": ["00100"] * 7,
}


def montage_labels(path, episodes):
    width, height = 1792, 960
    pixels = bytearray(width * height * 4)

    def rect(x, y, w, h, color):
        for row in range(y, y + h):
            start = (row * width + x) * 4
            pixels[start:start + w * 4] = bytes(color) * w

    def text(value, x, y, color=(240, 245, 244, 255), scale=2):
        for char in value:
            for row, line in enumerate(GLYPHS.get(char, ["00000"] * 7)):
                for col, bit in enumerate(line):
                    if bit == "1":
                        rect(x + col * scale, y + row * scale, scale, scale, color)
            x += 6 * scale

    rect(0, 0, width, 64, (15, 29, 26, 255))
    text("32 SCRIPTED EXPERT DEMOS | NOT WALL-OSS MODEL OUTPUT", 20, 10, scale=3)
    text("01-24 TRAIN CANDIDATES | 25-32 HELDOUT | HEAD CAMERA | SIMULATION ONLY", 20, 42)
    for e in episodes:
        col, row = (e["id"] - 1) % 8, (e["id"] - 1) // 8
        x, y = col * 224, row * 224 + 64
        rect(x, y, 224, 24, (15, 29, 26, 255))
        heldout = e["split"] == "heldout"
        text(f"{e['id']:02d} {'HELDOUT' if heldout else 'TRAIN'}", x + 8, y + 5,
             (255, 189, 102, 255) if heldout else (140, 221, 205, 255))

    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload)))

    raw = b"".join(b"\0" + pixels[y * width * 4:(y + 1) * width * 4]
                   for y in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def encode_montage(episodes, output):
    labels = output / "montage-labels.png"
    montage_labels(labels, episodes)
    overview = output / "overview.mp4"
    if not overview.exists():
        command = ["ffmpeg", "-v", "error", "-y", "-filter_complex_threads", "1"]
        filters = []
        for i, e in enumerate(episodes):
            command += ["-threads", "1", "-i", str(output / e["video"])]
            filters.append(f"[{i}:v]crop=448:448:448:0,scale=224:224,setsar=1[v{i}]")
        command += ["-loop", "1", "-i", str(labels)]
        layout = "|".join(f"{i % 8 * 224}_{i // 8 * 224}" for i in range(32))
        filters.append("".join(f"[v{i}]" for i in range(32))
                       + f"xstack=inputs=32:layout={layout}:shortest=1,"
                         "pad=1792:960:0:64:color=0x0f1d1a[stack]")
        filters.append("[stack][32:v]overlay=shortest=1:format=auto,format=yuv420p[out]")
        command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an",
                    "-t", "6.08", "-r", "25", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "21", "-threads", "4", "-movflags", "+faststart", str(overview)]
        run(command)
    run(["ffmpeg", "-v", "error", "-y", "-ss", "5.3", "-i", str(overview),
         "-frames:v", "1", "-q:v", "2", str(output / "overview.jpg")])


class VideoRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Single-byte-range responses allow native video seeking in browsers."""

    def send_head(self):
        self.range_end = None
        requested_range = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not requested_range or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested_range)
        size = path.stat().st_size
        if not match or not any(match.groups()) or size == 0:
            self.send_error(416, "Unsupported byte range")
            return None
        first, last = match.groups()
        start = int(first) if first else max(0, size - int(last))
        end = min(int(last), size - 1) if first and last else size - 1
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        try:
            source = path.open("rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        source.seek(start)
        self.range_end = end + 1
        return source

    def copyfile(self, source, output):
        if self.range_end is None:
            return super().copyfile(source, output)
        remaining = self.range_end - source.tell()
        while remaining:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            output.write(chunk)
            remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path("experiments/grasp/runs/training_round_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/grasp/runs/demo_gallery_v1"))
    parser.add_argument("--serve", action="store_true", help="Serve an already built gallery")
    parser.add_argument("--port", type=int, default=8843)
    args = parser.parse_args()
    if args.serve:
        if not (args.output / "index.html").is_file():
            parser.error("Build the gallery before starting its preview server")
        handler = functools.partial(VideoRequestHandler, directory=str(args.output))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
        print(f"Preview: http://127.0.0.1:{args.port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return
    for directory in ("source", "media", "posters"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    episodes = read_metadata(args.input_dir / "expert_dataset_v1.tar.gz")
    extract_recordings(args.input_dir / "expert_videos_v1.tar", episodes, args.output)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index in pool.map(lambda e: encode_episode(e, args.output), episodes):
            print(f"Rendered {index:02d}/32", flush=True)
    encode_montage(episodes, args.output)
    data = {"episodes": episodes, "controller": "privileged scripted expert",
            "modelExecuted": False, "simulationFps": 25,
            "sourcePlaybackFps": 20, "displayPlaybackFps": 25,
            "endpointFrames": 2}
    template = Path(__file__).with_name("demo_gallery.html").read_text()
    (args.output / "index.html").write_text(template.replace("__GALLERY_DATA__", json.dumps(data)))
    (args.output / "metadata.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"Gallery: {args.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
