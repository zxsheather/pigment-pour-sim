"""Make an annotated, shareable video without modifying training images."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=2.0, help="Playback multiplier (1..4).")
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    if not 1 <= args.speed <= 4:
        parser.error("speed must be in [1,4]")
    if args.output.exists():
        parser.error("output already exists")
    candidates = [args.font, Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                  Path("/System/Library/Fonts/PingFang.ttc")]
    font = next((path for path in candidates if path and path.exists()), None)
    if font is None:
        parser.error("A Chinese font is required; provide --font")
    title_font, label_font, text_font = (ImageFont.truetype(str(font), size) for size in (28, 22, 19))
    metadata = json.loads((args.episode / "metadata.json").read_text())
    with np.load(args.episode / "trajectory.npz") as data:
        state = data["state"].copy()
        times = data["timestamp"].copy()
        metrics = data["metrics"].copy() if "metrics" in data else None
    fps = metadata["fps"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    background, muted, white, green = "#111827", "#a8b3c7", "#f1f5f9", "#6ee7b7"
    with ExitStack() as stack:
        overview = stack.enter_context(imageio.get_reader(args.episode / "overview.mp4"))
        wrist = stack.enter_context(imageio.get_reader(args.episode / "right_wrist_view.mp4"))
        if len(times) != metadata["frames"] or state.shape != (len(times), 23):
            raise ValueError("Metadata and trajectory lengths differ")
        if metrics is not None and metrics.shape != (len(times), 4):
            raise ValueError("Unexpected metrics shape")
        for reader in (overview, wrist):
            if reader.count_frames() != len(times) or reader.get_meta_data()["fps"] != fps:
                raise ValueError("Video frames or sampling frequency differ from the trajectory")
        writer = stack.enter_context(imageio.get_writer(args.output, fps=fps, codec="libx264",
                                                       quality=8, macro_block_size=1,
                                                       ffmpeg_params=["-movflags", "+faststart"]))
        next_output_time = 0.0
        canvas = None
        for index, (full, grip) in enumerate(zip(overview, wrist)):
            if index >= len(times):
                raise ValueError("More video frames than state samples")
            # Drop frames deterministically to change playback speed while
            # keeping the encoded frame rate compatible with normal players.
            if times[index] + 1e-8 < next_output_time:
                continue
            next_output_time += args.speed / fps
            canvas = Image.new("RGB", (1088, 736), background)
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 16), "持瓶搬运仿真 · 程序控制器", font=title_font, fill=white)
            color_name = {"blue": "蓝色", "red": "红色", "green": "绿色", "yellow": "黄色"}[metadata["color"]]
            draw.text((24, 58), f"{color_name}瓶 → 停靠位 {metadata['goal']}    |    {args.speed:g} 倍速", font=text_font, fill=muted)
            canvas.paste(Image.fromarray(full).resize((640, 640)), (16, 96))
            canvas.paste(Image.fromarray(grip).resize((384, 384)), (688, 96))
            draw.text((32, 108), "全景：绕桌移动", font=label_font, fill="#172033",
                      stroke_width=2, stroke_fill="#ffffff")
            draw.text((704, 108), "腕部：实际接触夹持", font=label_font, fill="#172033",
                      stroke_width=2, stroke_fill="#ffffff")
            draw.text((688, 500), f"仿真时间  {times[index]:5.1f} 秒", font=label_font, fill=white)
            draw.text((688, 538), f"前进速度   {state[index,20]:+.3f} m/s", font=text_font, fill=muted)
            draw.text((688, 570), f"转向速度   {state[index,22]:+.3f} rad/s", font=text_font, fill=muted)
            if metrics is not None:
                error, slip, tilt, contacts = metrics[index]
                draw.text((688, 608), f"到位误差   {error*100:.1f} cm", font=text_font, fill=white)
                draw.text((688, 640), f"夹持滑移   {slip*1000:.1f} mm", font=text_font, fill=white)
                draw.text((688, 672), f"双侧接触   {int(contacts)}/2", font=text_font,
                          fill=green if contacts == 2 else "#fbbf24")
            draw.text((24, 702), "程序控制器 · 起点已夹稳 · 非 Wall 模型成绩", font=text_font,
                      fill="#334155", stroke_width=1, stroke_fill="#ffffff")
            writer.append_data(np.asarray(canvas))
        if canvas is None:
            raise ValueError("Episode contains no video frames")
        result = metadata["result"]
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((176, 286, 912, 470), radius=20, fill="#102b29" if result["success"] else "#3b2025")
        draw.text((208, 308), "搬运成功并停稳" if result["success"] else f"未完成：{result['failure_reason']}",
                  font=title_font, fill=white)
        draw.text((208, 362), f"用时 {result['simulated_seconds']:.2f} 秒  ·  到位误差 {result['position_error_m']*100:.2f} cm",
                  font=label_font, fill=white)
        draw.text((208, 408), f"全程最大滑移 {metadata['worst_slip_m']*1000:.2f} mm  ·  接触夹持，无绑定",
                  font=label_font, fill=muted)
        for _ in range(2 * fps):
            writer.append_data(np.asarray(canvas))
        canvas.save(args.output.with_suffix(".png"))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
