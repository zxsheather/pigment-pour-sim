"""Check the default finger geometry against the pinned official collision STLs."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from inspect_cylinder import CylinderEnv
from carry_env import ROOT


def audit(env=None):
    spec=json.loads((ROOT.parent/"sources.json").read_text())["sdk_robot"]
    asset_root=ROOT.parent/spec["destination"]/"urdf/quanta_x1"
    urdf=asset_root/"ex001_robot.urdf"
    source=ET.parse(urdf).getroot()
    owns_environment=env is None
    if env is None:env=CylinderEnv()
    baseline=json.loads((ROOT.parent/"docs/experiments/2026-11-17-official-parallel-baseline.json").read_text())["audit"]
    expected_hashes={item["source_file"]:item["sha256"] for item in baseline["fingers"]}
    assert hashlib.sha256(urdf.read_bytes()).hexdigest()==baseline["urdf_sha256"]
    nominal=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".01"/></worldbody></mujoco>')
    try:
        assert env.grip_profile=="official"
        assert env.model.neq==0
        assert env.model.opt.noslip_iterations==nominal.opt.noslip_iterations==0
        assert not any("contact_pad" in (env.model.geom(g).name or "") or "vpad" in (env.model.geom(g).name or "")
                       for g in range(env.model.ngeom))
        fingers=[]
        for side in ("left","right"):
            for finger in ("left","right"):
                name=f"{side}_arm_gripper_{finger}_link"
                relative=source.find(f"link[@name='{name}']/collision/geometry/mesh").get("filename")
                path=asset_root/relative
                assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_hashes[relative]
                mesh=trimesh.load_mesh(path)
                parts=mesh.split(only_watertight=False)
                assert all(part.is_convex for part in parts)
                body=env.model.body(name).id
                geoms=[g for g in range(env.model.ngeom)
                       if env.model.geom_bodyid[g]==body and env.model.geom_contype[g]]
                assert len(geoms)==len(parts)
                vertices=[]
                for g in geoms:
                    assert env.model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
                    assert env.model.geom_condim[g]==nominal.geom_condim[0]==3
                    scale=env.friction_scale if side=="right" else 1.
                    np.testing.assert_allclose(env.model.geom_friction[g],nominal.geom_friction[0]*scale)
                    np.testing.assert_allclose(env.model.geom_solref[g],nominal.geom_solref[0])
                    mesh_id=env.model.geom_dataid[g]
                    adr=env.model.mesh_vertadr[mesh_id];count=env.model.mesh_vertnum[mesh_id]
                    rotation=np.empty(9)
                    mujoco.mju_quat2Mat(rotation,env.model.geom_quat[g])
                    vertices.append(env.model.mesh_vert[adr:adr+count]@rotation.reshape(3,3).T+env.model.geom_pos[g])
                compiled=np.concatenate(vertices)
                # Both directions detect inserted, missing or shifted contact
                # vertices; check in the original finger-link coordinate frame.
                error=max(float(cKDTree(compiled).query(mesh.vertices)[0].max()),
                          float(cKDTree(mesh.vertices).query(compiled)[0].max()))
                assert error<1e-6,(name,error)
                joint=f"{side}_arm_gripper_{finger}_joint"
                official_joint=source.find(f"joint[@name='{joint}']")
                limits=official_joint.find("limit")
                np.testing.assert_allclose(env.model.joint(joint).range,
                                           [float(limits.get("lower")),float(limits.get("upper"))])
                np.testing.assert_allclose(env.model.joint(joint).axis,
                                           list(map(float,official_joint.find("axis").get("xyz").split())))
                if side=="right":
                    np.testing.assert_allclose(env.model.actuator(joint+"_servo").forcerange,
                                               [-float(limits.get("effort")),float(limits.get("effort"))])
                fingers.append(dict(link=name,source_file=relative,
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                    convex_solids=len(parts),compiled_contact_geoms=len(geoms),
                                    urdf_effort_limit_n=float(limits.get("effort")),
                                    maximum_vertex_error_m=error))
        return dict(passed=True,default_profile=env.grip_profile,source_commit=spec["commit"],
                    source_url=spec["url"],urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
                    custom_pads=0,geometry_import="Unchanged union of the source convex solids",
                    fingers=fingers,contact_model=dict(friction=nominal.geom_friction[0].tolist(),
                    condim=3,noslip_iterations=0,calibration="Nominal simulator defaults; not measured hardware properties"))
    finally:
        if owns_environment:env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("Output already exists")
    result=audit()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
