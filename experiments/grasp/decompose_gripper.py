"""Split the compound finger STL into its original convex connected parts."""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

import trimesh
import numpy as np


def decompose(scene, assets, output):
    tree = ET.parse(scene)
    root = tree.getroot()
    mesh_assets = root.find("asset")
    folder = output.parent/"convex_fingers/exact_split"
    folder.mkdir(parents=True, exist_ok=True)
    counts = {}
    for finger in ["left", "right"]:
        source = assets/"urdf/quanta_x1/mesh/collision"/f"gripper_{finger}_link.STL"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        cache = folder/f"{finger}_{digest}"
        manifest = cache/"parts.json"
        if not manifest.exists():
            cache.mkdir(exist_ok=True)
            mesh = trimesh.load_mesh(source)
            parts = mesh.split(only_watertight=False)
            if not parts:
                raise RuntimeError(f"No connected mesh parts: {source}")
            names=[]
            for i, part in enumerate(parts):
                if not part.is_watertight or not np.isclose(abs(part.volume), part.convex_hull.volume, rtol=1e-4):
                    raise RuntimeError(f"Finger component is not convex: {source}, part {i}")
                name=f"part_{i:02d}.stl"
                part.export(cache/name)
                names.append(name)
            manifest.write_text(json.dumps(names))
        parts=[cache/name for name in json.loads(manifest.read_text())]
        counts[finger]=len(parts)
        for i, part in enumerate(parts):
            ET.SubElement(mesh_assets,"mesh",name=f"contact_{finger}_{i}",file=str(part.resolve()))
        for arm in ["left", "right"]:
            body=root.find(f".//body[@name='{arm}_arm_gripper_{finger}_link']")
            colliders=[g for g in body.findall("geom") if g.get("group")=="3"]
            for old in colliders:
                body.remove(old)
                for i in range(len(parts)):
                    attrs=dict(old.attrib)
                    attrs.update(name=f"{arm}_{finger}_contact_{i}",mesh=f"contact_{finger}_{i}",mass="0")
                    attrs.pop("density",None)
                    ET.SubElement(body,"geom",**attrs)
    tree.write(output,encoding="unicode")
    print(json.dumps({"finger_convex_parts":counts,"method":"original convex connected components; no geometry approximation"}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--scene",type=Path,required=True)
    p.add_argument("--assets",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    decompose(args.scene,args.assets,args.output)
