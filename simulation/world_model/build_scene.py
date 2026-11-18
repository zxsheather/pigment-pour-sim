"""Build a photo-guided static scene around the official Quanta X1 URDF.

The upstream URDF and STL assets remain unmodified. The generated MJCF adds
pose joints, cameras, lighting, tables, trays, bottles and a measuring cylinder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from simulation.world_model.paths import ROOT, SCENE_CONFIG


def fmt(values):
    return " ".join(f"{float(v):.8g}" for v in values)


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def camera(parent, name, pos, target, fovy=45):
    pos, target = np.asarray(pos), np.asarray(target)
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return element(parent, "camera", name=name, pos=fmt(pos), xyaxes=fmt([*x, *y]), fovy=fovy)


def tray(world, name, center, size, table_z):
    b = element(world, "body", name=name, pos=fmt([*center, table_z]))
    sx, sy = np.asarray(size) / 2
    element(b, "geom", name=name+"_floor", type="box", pos="0 0 .008", size=fmt([sx, sy, .008]), rgba=".93 .94 .92 1", contype=2, conaffinity=3)
    for axis, half, other in [(0, sx, sy), (1, sy, sx)]:
        for sign in [-1, 1]:
            pos = [0, 0, .028]; pos[axis] = sign*half
            dims = [other, other, .02]; dims[axis] = .006; dims[1-axis] = other+.006
            element(b, "geom", type="box", pos=fmt(pos), size=fmt(dims), rgba=".97 .97 .94 1", contype=2, conaffinity=3)


def build(asset_dir: Path, output_dir: Path, config_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(config_path.read_text())
    urdf = asset_dir / "urdf/quanta_x1/ex001_robot.urdf"
    source = ET.parse(urdf).getroot()
    # Some fixed sensor frames in the CAD export contain singular/negative
    # inertias. Regularize only the generated import; preserve upstream files.
    for inertial in source.findall("link/inertial"):
        mass = inertial.find("mass")
        mass.set("value", str(max(float(mass.get("value")), 1e-5)))
        inertia = inertial.find("inertia")
        a = {k:float(inertia.get(k,"0")) for k in ["ixx","iyy","izz","ixy","ixz","iyz"]}
        matrix=np.array([[a['ixx'],a['ixy'],a['ixz']],[a['ixy'],a['iyy'],a['iyz']],[a['ixz'],a['iyz'],a['izz']]])
        eig,vec=np.linalg.eigh(matrix)
        if eig.min() < 1e-9:
            matrix=(vec*np.maximum(eig,1e-9))@vec.T
            for key,value in zip(["ixx","iyy","izz","ixy","ixz","iyz"],[matrix[0,0],matrix[1,1],matrix[2,2],matrix[0,1],matrix[0,2],matrix[1,2]]): inertia.set(key,str(value))
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    import trimesh
    for mesh in source.iter("mesh"):
        original = (urdf.parent / mesh.get("filename")).resolve()
        name = hashlib.sha1(str(original.relative_to(urdf.parent)).encode()).hexdigest()[:10]+"_"+original.stem+".stl"
        cached = mesh_dir / name
        if not cached.exists():
            geometry = trimesh.load_mesh(original)
            if len(geometry.faces) > 18000:
                geometry = geometry.simplify_quadric_decimation(face_count=18000)
            geometry.export(cached)
        mesh.set("filename", str(cached.resolve()))
    extension = element(source, "mujoco")
    element(extension, "compiler", discardvisual="false", fusestatic="false", strippath="false", balanceinertia="true")
    cleaned = output_dir / "import_robot.urdf"
    ET.ElementTree(source).write(cleaned, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(cleaned))
    robot_xml = output_dir / "robot_import.xml"
    mujoco.mj_saveLastXML(str(robot_xml), model)
    scene = ET.parse(robot_xml).getroot()
    scene.set("model", "CS2971 photo-guided pick-and-pour")
    compiler = scene.find("compiler")
    compiler.set("angle", "radian")
    for mesh in scene.findall("asset/mesh"):
        path = Path(mesh.get("file"))
        if not path.is_absolute():
            mesh.set("file", str((urdf.parent / path).resolve()))
    option = scene.find("option")
    if option is None: option = element(scene, "option")
    option.attrib.update(timestep="0.004", integrator="implicitfast", cone="elliptic", iterations="60", gravity="0 0 -9.81")
    element(scene, "size", njmax="6000", nconmax="2000")
    visual = element(scene, "visual")
    element(visual, "global", offwidth="1280", offheight="800")
    element(visual, "quality", shadowsize="2048", offsamples="4")
    element(visual, "headlight", ambient=".4 .4 .4", diffuse=".65 .65 .65", specular=".15 .15 .15")
    element(visual, "map", znear=".01", zfar="20")
    assets = scene.find("asset")
    element(assets, "texture", name="sky", type="skybox", builtin="gradient", width="512", height="512", rgb1=".76 .80 .79", rgb2=".95 .96 .94")
    element(assets, "texture", name="floor_tex", type="2d", builtin="checker", width="512", height="512", rgb1=".79 .80 .77", rgb2=".87 .88 .84")
    element(assets, "material", name="floor_mat", texture="floor_tex", texrepeat="8 8", reflectance=".08")
    world = scene.find("worldbody")
    imported = list(world)
    robot = element(world, "body", name="robot_planar_root", pos="0 0 0")
    element(robot, "inertial", pos="0 0 0", mass="1", diaginertia=".1 .1 .1")
    element(robot, "joint", name="base_x", type="slide", axis="1 0 0", range="-2 2", limited="true", damping="80")
    element(robot, "joint", name="base_y", type="slide", axis="0 1 0", range="-2 2", limited="true", damping="80")
    element(robot, "joint", name="base_yaw", type="hinge", axis="0 0 1", range="-3.14 3.14", limited="true", damping="30")
    for child in imported:
        world.remove(child); robot.append(child)
    for body in robot.iter("body"):
        body.set("gravcomp", "1")
    for geom in robot.iter("geom"):
        # Official visual geoms have zero contact masks; collision meshes stay separate.
        if geom.get("contype") == "0":
            geom.set("group", "1")
        else:
            geom.set("contype", "1"); geom.set("conaffinity", "2")
            geom.set("group", "3"); geom.set("rgba", ".3 .3 .3 0")
    for side in ["right", "left"]:
        parent = robot.find(f".//body[@name='{side}_arm_gripper_base_link']")
        if parent is None: raise ValueError("Official gripper link missing")
        element(parent, "camera", name=f"{side}_wrist_view", pos=".07 0 .11", xyaxes="0 -1 0 .6 0 .8", fovy="86")
    optical = robot.find(".//body[@name='camera_head_front_color_optical_frame']")
    if optical is not None:
        element(optical, "camera", name="face_view", pos="0 0 0", quat="0 1 0 0", fovy="86")
    element(world, "light", pos="1 -1 4", dir="-.2 .2 -1", diffuse=".85 .85 .8", castshadow="true")
    element(world, "light", pos="-2 2 3", dir="1 -1 -1", diffuse=".5 .55 .6", castshadow="false")
    element(world, "geom", name="ground", type="plane", size="5 5 .1", material="floor_mat", contype="2", conaffinity="3")
    element(world, "geom", name="wall", type="box", pos="-1.15 .1 1.3", size=".04 3 1.3", rgba=".88 .87 .82 1", contype="0", conaffinity="0")
    tab = cfg["table"]; z = tab["height"]
    join = tab.get("join", {"count": 1, "split_axis": "x", "gap_m": 0})
    axis = 0 if join["split_axis"] == "x" else 1
    span = tab["size_xy"][axis]
    panel_size = list(tab["size_xy"])
    panel_size[axis] = (span - join["gap_m"] * (join["count"] - 1)) / join["count"]
    for i in range(join["count"]):
        center = list(tab["center_xy"])
        center[axis] += -span/2 + panel_size[axis]/2 + i * (panel_size[axis] + join["gap_m"])
        element(world, "geom", name=f"tabletop_{i}", type="box", pos=fmt([*center, z-.025]), size=fmt([panel_size[0]/2, panel_size[1]/2, .025]), rgba=".86 .87 .83 1", contype="2", conaffinity="3")
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                x = center[0] + sx * (panel_size[0]/2-.07)
                y = center[1] + sy * (panel_size[1]/2-.10)
                element(world, "geom", name=f"table_{i}_leg_{sx}_{sy}", type="box", pos=fmt([x,y,(z-.05)/2]), size=fmt([.017,.017,(z-.05)/2]), rgba=".65 .67 .63 1", contype="2", conaffinity="3")
    for name in ["source", "target"]:
        p = cfg[name+"_tray"]
        tray(world,name+"_tray",p["center_xy"],p["size_xy"],z)
    for i,(a,b) in enumerate(zip(cfg.get("navigation",{}).get("route_xy",[]),cfg.get("navigation",{}).get("route_xy",[])[1:])):
        element(world,"geom",name=f"route_guide_{i}",type="capsule",fromto=fmt([*a,.018,*b,.018]),size=".006",rgba=".25 .52 .37 .6",contype="0",conaffinity="0")
    element(world,"site",name="target_dock",pos=fmt([cfg["robot"].get("target_base_x",cfg["robot"]["base_x"]),cfg["robot"]["target_base_y"],.025]),size=".04",rgba=".28 .58 .39 .55")
    r, h = cfg["bottle"]["radius"], cfg["bottle"]["height"]
    for spec in cfg["bottles"]:
        color=spec["color"]
        xy=np.asarray(cfg["source_tray"]["center_xy"])+spec["offset_xy"]
        b=element(world,"body",name="bottle_"+color,pos=fmt([*xy,z+.016+h*.5]))
        element(b,"freejoint",name="bottle_"+color+"_free")
        element(b,"geom",name="bottle_"+color+"_shell",type="cylinder",size=fmt([r,h*.5]),rgba=".8 .89 .9 .20",mass=".11",friction="1 .01 .001",contype="2",conaffinity="3")
        element(b,"geom",name="bottle_"+color+"_liquid",type="cylinder",pos=fmt([0,0,-.014]),size=fmt([r*.94,h*.35]),rgba=fmt(spec["rgba"][:3]+[.9]),mass="0",contype="0",conaffinity="0")
        element(b,"geom",type="cylinder",pos=fmt([0,0,h*.5+.007]),size=fmt([r*.7,.009]),rgba=".88 .92 .9 .65",mass=".005",contype="2",conaffinity="3")
        element(b,"site",name="mouth_"+color,pos=fmt([0,0,h*.5+.017]),size=".004",rgba="1 1 1 0")
    c=cfg["cylinder"]; cr,ch=c["radius"],c["height"]
    b=element(world,"body",name="measuring_cylinder",pos=fmt([*c["center_xy"],z+.016]))
    element(b,"geom",name="cylinder_base",type="box",size=".043 .043 .003",pos="0 0 .003",rgba=".83 .89 .89 .7",contype="2",conaffinity="3")
    # An open vessel, represented with separate thin wall segments (no solid cylinder).
    for i in range(24):
        a=2*np.pi*i/24
        element(b,"geom",type="box",pos=fmt([cr*np.cos(a),cr*np.sin(a),ch/2+.006]),size=fmt([.001,cr*np.pi/24,ch/2]),euler=fmt([0,0,a]),rgba=".75 .87 .9 .18",contype="2",conaffinity="3")
    for i in range(1,11):
        zz=.012+i*(ch-.024)/10
        element(b,"geom",type="box",pos=fmt([cr+.002,0,zz]),size=fmt([.001,.011 if i%2==0 else .006,.0008]),rgba=".2 .3 .32 1",contype="0",conaffinity="0")
    element(b,"geom",name="cylinder_liquid",type="cylinder",size=fmt([cr-.002,.0002]),pos="0 0 .008",rgba=".1 .3 .8 .8",contype="0",conaffinity="0")
    element(b,"site",name="cylinder_mouth",pos=fmt([0,0,ch+.006]),size=".005",rgba="1 .5 0 0")
    camera(world,"overview",[2.3,-1.7,2.0],[.4,.4,.77],43)
    cx,cy=cfg["cylinder"]["center_xy"]
    camera(world,"pour_detail",[cx+.9,cy-.85,1.6],[cx,cy,z+.12],43)
    camera(world,"source_detail",[1.0,-.15,1.5],[.22,.58,.88],50)
    dest=output_dir/"scene.xml"
    ET.ElementTree(scene).write(dest,encoding="unicode")
    model=mujoco.MjModel.from_xml_path(str(dest))
    print(f"Built {dest}: {model.nq} qpos, {model.nv} dof, {model.ngeom} geoms",flush=True)
    return dest


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--assets",type=Path,default=ROOT/"third_party/sdk_robot")
    p.add_argument("--output",type=Path,default=ROOT/"generated")
    p.add_argument("--config",type=Path,default=SCENE_CONFIG)
    a=p.parse_args(); build(a.assets.resolve(),a.output.resolve(),a.config.resolve())
