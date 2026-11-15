"""Image-only liquid reading for the explicitly color-calibrated cylinder.

The reader accepts only RGB pixels, not simulator state or expected volume.
Magenta and orange graduation anchors encode 20 and 35 mL respectively.
This is an instrument-specific CV baseline, not general OCR or Wall inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def components(mask):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return [(index,stats[index],centers[index]) for index in range(1,count) if stats[index,4]>=3],labels


def read_volume(rgb: np.ndarray) -> dict:
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:
        raise ValueError("Reader requires an RGB uint8 image")
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    magenta=(hsv[:,:,0]>=140)&(hsv[:,:,0]<=175)&(hsv[:,:,1]>90)&(hsv[:,:,2]>90)
    orange=(hsv[:,:,0]>=5)&(hsv[:,:,0]<=22)&(hsv[:,:,1]>120)&(hsv[:,:,2]>100)
    pink,_=components(magenta);amber,_=components(orange)
    pairs=[]
    for _,a,p in pink:
        for _,b,q in amber:
            separation=p[1]-q[1]
            if separation<30 or abs(p[0]-q[0])>max(8,separation*.10):continue
            if a[2]<3 or b[2]<3:continue
            score=a[4]+b[4]-2*abs(p[0]-q[0])
            pairs.append((score,p,q))
    if not pairs:return dict(valid=False,reason="calibration_anchors_not_visible")
    _,p,q=max(pairs,key=lambda item:item[0])
    y20,y35=float(p[1]),float(q[1]);anchor_x=float((p[0]+q[0])/2)
    if abs(p[0]-q[0])/(y20-y35)>.0875:
        return dict(valid=False,reason="scale_not_vertical")
    blue=(hsv[:,:,0]>=95)&(hsv[:,:,0]<=135)&(hsv[:,:,1]>90)&(hsv[:,:,2]>60)
    blobs,labels=components(blue)
    relevant=[]
    for index,stats,center in blobs:
        x,y,w,h,area=stats
        if area<40 or w<4 or h<4:continue
        if abs(center[0]-anchor_x)>max(35,(y20-y35)*.25):continue
        if y>y20+2*(y20-y35) or y+h<y35-30:continue
        relevant.append((area,index,stats))
    if not relevant:return dict(valid=False,reason="liquid_not_visible")
    _,index,stats=max(relevant,key=lambda item:item[0])
    x,y,w,h,_=stats
    center_x=int(round(x+w*.5))
    tops=[];columns=[]
    margin=max(1,int(w*.1))
    for column in range(x+margin,x+w-margin):
        pixels=np.flatnonzero(labels[:,column]==index)
        if len(pixels)>3:
            tops.append(float(pixels[0]));columns.append(column)
    if len(tops)<3 or np.ptp(tops)>8:
        return dict(valid=False,reason="ambiguous_liquid_boundary")
    # Black graduation lines or digits can hide the boundary in central
    # columns. Use the highest edge supported by three adjacent columns,
    # with a median rejecting isolated one-pixel noise. No volume prior.
    edges=[float(np.median(tops[i:i+3])) for i in range(len(tops)-2)
           if columns[i+2]-columns[i]==2 and np.ptp(tops[i:i+3])<=1]
    if not edges:return dict(valid=False,reason="ambiguous_liquid_boundary")
    liquid_y=min(edges)
    support=int(np.count_nonzero(np.abs(np.array(tops)-liquid_y)<=1))
    volume=20+(y20-liquid_y)/(y20-y35)*15
    if not 0<=volume<=50:return dict(valid=False,reason="reading_out_of_scale")
    return dict(valid=True,volume_ml=float(volume),liquid_y=liquid_y,liquid_x=center_x,
                y20=y20,y35=y35,anchor_x=anchor_x,liquid_bbox=stats[:4].astype(int).tolist(),
                boundary_support_columns=support,boundary_support_fraction=support/len(tops),
                eye_alignment_error_px=liquid_y-(rgb.shape[0]-1)/2,
                method="two_color_graduation_anchors_and_liquid_boundary")


def annotate(rgb,result):
    out=rgb.copy()
    if result["valid"]:
        x,y,w,h=result["liquid_bbox"]
        cv2.rectangle(out,(x,y),(x+w,y+h),(70,220,120),1)
        ly=int(round(result["liquid_y"]))
        cv2.line(out,(x-25,ly),(x+w+25,ly),(40,230,80),2)
        for level,key in ((20,"y20"),(35,"y35")):
            point=(int(round(result["anchor_x"])),int(round(result[key])))
            cv2.circle(out,point,5,(255,180,40),1)
        label=f"Image reading: {result['volume_ml']:.2f} mL"
    else:label=f"NO READING: {result['reason']}"
    cv2.putText(out,label,(18,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,20,20),3,cv2.LINE_AA)
    cv2.putText(out,label,(18,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,245,250),1,cv2.LINE_AA)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("image",type=Path);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();rgb=cv2.cvtColor(cv2.imread(str(args.image)),cv2.COLOR_BGR2RGB)
    result=read_volume(rgb);args.output.write_text(json.dumps(result,indent=2)+"\n")
    cv2.imwrite(str(args.output.with_suffix(".png")),cv2.cvtColor(annotate(rgb,result),cv2.COLOR_RGB2BGR))
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
