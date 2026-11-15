"""Compose cylinder manipulation and image-based reading into a review video."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

from cylinder_reader import read_volume,annotate

PHASES={"approach":"伸手靠近","advance":"张开夹爪并对准","close":"合拢夹持",
        "lift":"从托盘提起","retract":"回收手臂","back_away":"退离桌边",
        "present":"移到相机前","straighten":"调整量筒竖直","hold_for_reading":"等待稳定",
        "align_eye_level":"液面对齐视线","find_scale":"转动手腕寻找刻度","read_stable":"稳定读取刻度"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--speed",type=float,default=2);p.add_argument("--font",type=Path)
    args=p.parse_args()
    if args.output.exists() or not 1<=args.speed<=4:p.error("output must be new; speed must be in [1,4]")
    font=args.font
    if font is None:
        font=next((path for path in (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                                    Path("/System/Library/Fonts/PingFang.ttc")) if path.exists()),None)
    if font is None:p.error("Chinese font not found; provide --font")
    fonts=[ImageFont.truetype(str(font),size) for size in (30,26,22,19)]
    summary=json.loads((args.episode/"summary.json").read_text())
    with np.load(args.episode/"trajectory.npz") as data:
        times=data["timestamp"].copy();phases=data["phase"].copy();tilt=data["tilt"].copy()
        slip=data["slip"].copy();contacts=data["contacts"].copy()
    history=[row for row in summary["reading_history"] if row["valid"]]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    next_time=0.;last_roi=None;last_reading=None;frame=None
    with imageio.get_reader(args.episode/"cylinder_overview.mp4") as overview, \
            imageio.get_reader(args.episode/"face_view.mp4") as face, \
            imageio.get_writer(args.output,fps=20,codec="libx264",quality=8,
                               macro_block_size=1,ffmpeg_params=["-movflags","+faststart"]) as writer:
        if overview.count_frames()!=len(times) or face.count_frames()!=len(times):
            raise ValueError("Video/trajectory frame counts differ")
        for index,(full,rgb) in enumerate(zip(overview,face)):
            if times[index]+1e-8<next_time:continue
            next_time+=args.speed/20
            if index%10==0 or last_roi is None:
                located=read_volume(rgb)
                if located["valid"]:
                    ax=located["anchor_x"];lx=located["liquid_x"]
                    top=min(located["y35"],located["liquid_y"])-40
                    bottom=max(located["y20"],located["liquid_y"])+50
                    last_roi=(max(0,int(ax-45)),max(0,int(top)),
                              min(rgb.shape[1],int(lx+65)),min(rgb.shape[0],int(bottom)))
            available=[row for row in history if row["time"]<=times[index]+1e-8]
            if available:
                last_reading=available[-1]
                # Use the saved raw-RGB observation, since H.264 may erase a
                # tiny color anchor. This affects the display crop only.
                ax=last_reading["anchor_x"];lx=last_reading["liquid_x"]
                top=min(last_reading["y35"],last_reading["liquid_y"])-40
                bottom=max(last_reading["y20"],last_reading["liquid_y"])+50
                last_roi=(max(0,int(ax-45)),max(0,int(top)),
                          min(rgb.shape[1],int(lx+65)),min(rgb.shape[0],int(bottom)))
            frame=Image.new("RGB",(1280,864),"#111827");draw=ImageDraw.Draw(frame)
            title="量筒稳定优化 · V 形指垫 + 图像读数" if summary.get("grip_profile")=="v-pad" else "量筒拿近与读数 · 接触抓取 + 图像读数"
            draw.text((24,14),title,font=fonts[0],fill="#f1f5f9")
            draw.text((24,56),f"{PHASES.get(str(phases[index]),str(phases[index]))}    |    仿真 {times[index]:.1f} 秒    |    {args.speed:g} 倍速",
                      font=fonts[2],fill="#a8b3c7")
            frame.paste(Image.fromarray(full).resize((720,720)),(16,96))
            if last_reading and phases[index]=="read_stable":
                shown=annotate(rgb,last_reading)
            else:shown=rgb
            image=Image.fromarray(shown)
            if last_roi is not None:image=image.crop(last_roi)
            image=ImageOps.contain(image,(496,512),Image.Resampling.LANCZOS)
            frame.paste(image,(760+(496-image.width)//2,96+(512-image.height)//2))
            draw.text((776,108),"头部相机 · 图像局部放大",font=fonts[2],fill="#f1f5f9",
                      stroke_width=2,stroke_fill="#172033")
            if last_reading:
                prefix="稳定读数" if phases[index]=="read_stable" else "对齐中估读"
                label=f"{prefix}：{last_reading['volume_ml']:.1f} mL";color="#6ee7b7"
            else:label="图像读数：等待稳定观察";color="#a8b3c7"
            draw.text((760,630),label,font=fonts[1],fill=color)
            draw.text((760,680),f"量筒倾斜  {np.degrees(tilt[index]):.1f}°",font=fonts[2],fill="#e2e8f0")
            if phases[index] not in ("approach","advance","close"):
                draw.text((760,722),f"夹持滑移  {slip[index]*1000:.1f} mm   接触 {int(contacts[index])}/2",font=fonts[2],fill="#e2e8f0")
            else:draw.text((760,722),"夹持尚未建立",font=fonts[2],fill="#a8b3c7")
            draw.text((24,829),"程序教师使用仿真位姿；读数只用图像与彩色校准刻度；液面为简化模型",font=fonts[3],fill="#a8b3c7")
            writer.append_data(np.asarray(frame))
        if frame is None:raise ValueError("No recorded video frames")
        draw=ImageDraw.Draw(frame)
        draw.rounded_rectangle((180,290,1100,530),radius=20,fill="#10332e" if summary["success"] else "#45232c")
        draw.text((216,310),"夹起、移近与读数完成" if summary["success"] else "本次动作未完成",font=fonts[0],fill="#f1f5f9")
        if summary["reading"]["valid"]:
            text=f"图像读数 {summary['reading']['volume_ml']:.1f} mL   ·   校验设定 {summary['expected_volume_ml']:.1f} mL"
            draw.text((216,370),text,font=fonts[1],fill="#f1f5f9")
            draw.text((216,420),f"校验误差 {summary['absolute_reading_error_ml']:.2f} mL   ·   最终倾斜 {np.degrees(summary['final_tilt_rad']):.1f}°",
                      font=fonts[2],fill="#b2d8d0")
        draw.text((216,470),"校验设定值只用于评分，没有传入图像读数程序",font=fonts[2],fill="#a8b3c7")
        for _ in range(40):writer.append_data(np.asarray(frame))
        frame.save(args.output.with_suffix(".png"))
    print(args.output.resolve())


if __name__=="__main__":main()
