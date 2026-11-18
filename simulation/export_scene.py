"""Export the static reconstruction as GLB and rendered views."""
from pathlib import Path
import json
import os
os.environ.setdefault("MUJOCO_GL", "egl" if Path("/dev/nvidia0").exists() else "glfw")
import mujoco
import numpy as np
import trimesh
import imageio.v2 as imageio

from simulation.world_model.paths import ROOT, SCENE_CONFIG
from simulation.benchmarks.pose import set_display_pose


def main():
    cfg=json.loads(SCENE_CONFIG.read_text())
    m=mujoco.MjModel.from_xml_path(str(ROOT/'generated/scene.xml'))
    d=mujoco.MjData(m)
    set_display_pose(m, d, cfg)
    out=ROOT/'reconstruction';out.mkdir(exist_ok=True)
    scene=trimesh.Scene()
    root=m.body('robot_planar_root').id
    count=0
    for gi in range(m.ngeom):
        bi=int(m.geom_bodyid[gi]);cur=bi
        while cur>0 and cur!=root:cur=int(m.body_parentid[cur])
        if cur!=root or m.geom_rgba[gi,3]<.01 or m.geom_group[gi]==3:continue
        typ=int(m.geom_type[gi])
        if typ==int(mujoco.mjtGeom.mjGEOM_MESH):
            mi=int(m.geom_dataid[gi]);va=m.mesh_vertadr[mi];vn=m.mesh_vertnum[mi];fa=m.mesh_faceadr[mi];fn=m.mesh_facenum[mi]
            geom=trimesh.Trimesh(vertices=m.mesh_vert[va:va+vn].copy(),faces=m.mesh_face[fa:fa+fn].copy(),process=False)
        elif typ==int(mujoco.mjtGeom.mjGEOM_BOX):geom=trimesh.creation.box(extents=m.geom_size[gi]*2)
        elif typ==int(mujoco.mjtGeom.mjGEOM_CYLINDER):geom=trimesh.creation.cylinder(radius=m.geom_size[gi,0],height=m.geom_size[gi,1]*2)
        else:continue
        rgba=np.clip(m.geom_rgba[gi]*255,0,255).astype(np.uint8)
        geom.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=rgba,metallicFactor=.08,roughnessFactor=.6))
        transform=np.eye(4);transform[:3,:3]=d.geom_xmat[gi].reshape(3,3);transform[:3,3]=d.geom_xpos[gi]
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,bi) or f'robot_{gi}'
        scene.add_geometry(geom,node_name=f'{name}_{gi}',geom_name=f'{name}_{gi}',transform=transform)
        count+=1
    scene.export(out/'quanta_x1.glb')
    renderer=mujoco.Renderer(m,height=800,width=1120)
    for camera in ['overview','source_detail','pour_detail']:
        renderer.update_scene(d,camera=camera)
        imageio.imwrite(out/(camera+'.png'),renderer.render())
    renderer.close()
    manifest={'status':'static reconstruction only','reference':'场地照片/任务场地.jpg','dimensions':'estimates from a single photo; adjust after measurement','robot_source':'https://github.com/X-Square-Robot/sdk_robot','robot_commit':'63610199bbd70eafd1fc087a015ec5c74a0c906c','robot_visual_geometries':count,'robot_pose':'illustrative parked pose','config':cfg}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps({'output':str(out),'robot_meshes':count,'robot_glb_mb':(out/'quanta_x1.glb').stat().st_size/1e6}),flush=True)


if __name__=='__main__':main()
