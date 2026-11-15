import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
import {RoundedBoxGeometry} from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';

const cfg=structuredClone(window.SCENE_CONFIG);
const initial=structuredClone(cfg);
const colors={green:{label:'绿色',hex:0x20956d},blue:{label:'蓝色',hex:0x356acc},yellow:{label:'黄色',hex:0xe2c726},red:{label:'红色',hex:0xde625a}};
const $=id=>document.getElementById(id);
const host=$('viewport');
const scene=new THREE.Scene();scene.background=new THREE.Color('#e5e9e5');
const camera=new THREE.PerspectiveCamera(40,1,.01,50);camera.up.set(0,0,1);
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.0;
renderer.domElement.setAttribute('aria-label','场景三维视图：拖动旋转，滚轮缩放，右键平移');renderer.domElement.setAttribute('tabindex','0');host.prepend(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.dampingFactor=.08;controls.minDistance=.35;controls.maxDistance=8;controls.maxPolarAngle=Math.PI*.495;controls.target.set(.03,.04,.84);
scene.add(new THREE.HemisphereLight(0xf3f8ff,0x8d9583,1.6));
const sun=new THREE.DirectionalLight(0xfff7e9,2.3);sun.position.set(1.8,-1.8,4.5);sun.castShadow=true;sun.shadow.mapSize.set(2048,2048);sun.shadow.camera.left=-2;sun.shadow.camera.right=2;sun.shadow.camera.top=2;sun.shadow.camera.bottom=-2;sun.shadow.normalBias=.015;sun.shadow.bias=-.00005;scene.add(sun);
const fill=new THREE.DirectionalLight(0xdcecff,1.3);fill.position.set(-2,2,2.5);scene.add(fill);
const floor=new THREE.Mesh(new THREE.PlaneGeometry(12,12),new THREE.MeshStandardMaterial({color:0xcdd1c9,roughness:.95}));floor.receiveShadow=true;scene.add(floor);
const grid=new THREE.GridHelper(8,32,0xa7b0a7,0xc1c8bf);grid.rotation.x=Math.PI/2;grid.position.z=.001;grid.material.transparent=true;grid.material.opacity=.35;scene.add(grid);
const wall=new THREE.Mesh(new THREE.BoxGeometry(.05,6,2.8),new THREE.MeshStandardMaterial({color:0xd8d5cc,roughness:1}));wall.position.set(-1.1,0,1.4);wall.receiveShadow=true;scene.add(wall);
const desk=new THREE.Group();scene.add(desk);
const task=new THREE.Group();scene.add(task);
const robot=new THREE.Group();scene.add(robot);
const route=new THREE.Group();scene.add(route);
const white=new THREE.MeshStandardMaterial({color:0xf5f5ed,roughness:.45});
const objects=[];const labels=[];let selected='blue',sourceCenter,measureCenter,selectedRing;
const axis=new THREE.AxesHelper(.25);axis.position.set(.0,-1.15,.754);axis.visible=false;scene.add(axis);

function mesh(geometry,material,parent,pos=[0,0,0]){const item=new THREE.Mesh(geometry,material);item.position.set(...pos);item.castShadow=true;item.receiveShadow=true;parent.add(item);return item;}
function box(parent,size,pos,material=white,radius=.008){return mesh(new RoundedBoxGeometry(...size,3,radius),material,parent,pos);}
function cylinder(parent,radius,height,pos,material,segments=48){const g=new THREE.CylinderGeometry(radius,radius,height,segments);g.rotateX(Math.PI/2);return mesh(g,material,parent,pos);}
function ring(parent,radius,tube,pos,material){return mesh(new THREE.TorusGeometry(radius,tube,8,64),material,parent,pos);}
function roundedPath(path,x,y,w,h,r){path.moveTo(x+r,y);path.lineTo(x+w-r,y);path.quadraticCurveTo(x+w,y,x+w,y+r);path.lineTo(x+w,y+h-r);path.quadraticCurveTo(x+w,y+h,x+w-r,y+h);path.lineTo(x+r,y+h);path.quadraticCurveTo(x,y+h,x,y+h-r);path.lineTo(x,y+r);path.quadraticCurveTo(x,y,x+r,y);return path;}
function tray(parent,size,center){const [w,h]=size;const g=new THREE.Group();g.position.set(...center,cfg.table.height);parent.add(g);box(g,[w,h,.012],[0,0,.006],white,.025);const s=roundedPath(new THREE.Shape(),-w/2,-h/2,w,h,.035);s.holes.push(roundedPath(new THREE.Path(),-w/2+.014,-h/2+.014,w-.028,h-.028,.025));const rim=new THREE.ExtrudeGeometry(s,{depth:.034,bevelEnabled:true,bevelSegments:3,bevelSize:.003,bevelThickness:.003,curveSegments:10,steps:1});mesh(rim,white,g,[0,0,.01]);return g;}
function label(text,pos,kind=''){const e=document.createElement('div');e.className='scene-label '+kind;e.textContent=text;$('labels').append(e);labels.push({element:e,pos:new THREE.Vector3(...pos),kind});}
function disposeGroup(group){group.traverse(o=>{if(o.geometry)o.geometry.dispose();});group.clear();}
function buildDesk(){
  disposeGroup(desk);
  const t=cfg.table,join=t.join??{count:1,split_axis:'x',gap_m:0};
  const axis=join.split_axis==='x'?0:1,span=t.size_xy[axis],panelSize=[...t.size_xy];
  panelSize[axis]=(span-join.gap_m*(join.count-1))/join.count;
  const topMaterial=new THREE.MeshStandardMaterial({color:0xe3e3da,roughness:.55});
  const legMaterial=new THREE.MeshStandardMaterial({color:0xadafa7,metalness:.25,roughness:.6});
  for(let i=0;i<join.count;i++){
    const center=[...t.center_xy];center[axis]+= -span/2+panelSize[axis]/2+i*(panelSize[axis]+join.gap_m);
    box(desk,[...panelSize,.05],[...center,t.height-.025],topMaterial,.004);
    for(const sx of [-1,1])for(const sy of [-1,1]){
      const x=center[0]+sx*(panelSize[0]/2-.07),y=center[1]+sy*(panelSize[1]/2-.10);
      box(desk,[.034,.034,t.height-.05],[x,y,(t.height-.05)/2],legMaterial,.003);
    }
  }
}
function makeBottle(spec,xy){const g=new THREE.Group();g.position.set(...xy,cfg.table.height+.014);g.userData.color=spec.color;task.add(g);objects.push(g);const r=cfg.bottle.radius,h=cfg.bottle.height;const glass=new THREE.MeshPhysicalMaterial({color:0xd8e7e2,roughness:.15,metalness:0,transparent:true,opacity:.24,depthWrite:false,side:THREE.DoubleSide});const profile=[[0,0],[r*.85,0],[r,.006],[r,h*.72],[r*.85,h*.80],[r*.63,h*.86],[r*.63,h]].map(p=>new THREE.Vector2(...p));const shape=new THREE.LatheGeometry(profile,64);shape.rotateX(Math.PI/2);mesh(shape,glass,g);const liquid=new THREE.MeshPhysicalMaterial({color:colors[spec.color].hex,roughness:.2,metalness:0,transparent:true,opacity:.83});cylinder(g,r*.92,h*.62,[0,0,h*.31+.004],liquid);cylinder(g,r*.92,.001,[0,0,h*.62+.004],new THREE.MeshStandardMaterial({color:colors[spec.color].hex,roughness:.18}));ring(g,r*.63,.0014,[0,0,h],glass);for(let k=0;k<4;k++)ring(g,r*.65,.0007,[0,0,h-.003-k*.003],glass);const tick=new THREE.MeshBasicMaterial({color:0x2d4e47,transparent:true,opacity:.5});for(let k=1;k<5;k++)box(g,[.011,.00045,.0006],[0,-r-.0003,.015+k*.021],tick,.0001);return g;}
function makeCylinder(xy){const g=new THREE.Group();g.position.set(...xy,cfg.table.height+.014);task.add(g);const r=cfg.cylinder.radius,h=cfg.cylinder.height;const glass=new THREE.MeshPhysicalMaterial({color:0xc6dfe0,roughness:.12,transparent:true,opacity:.28,depthWrite:false,side:THREE.DoubleSide});const tube=new THREE.CylinderGeometry(r,r,h,48,1,true);tube.rotateX(Math.PI/2);mesh(tube,glass,g,[0,0,h/2+.006]);box(g,[.044,.044,.005],[0,0,.0025],glass,.004);ring(g,r,.0012,[0,0,h+.006],glass);const tick=new THREE.MeshBasicMaterial({color:0x45666d});const fontCanvas=document.createElement('canvas');fontCanvas.width=192;fontCanvas.height=1024;const ctx=fontCanvas.getContext('2d');ctx.clearRect(0,0,192,1024);ctx.fillStyle='#365f67';ctx.font='32px sans-serif';for(let k=1;k<=10;k++){const z=.018+k*.017;box(g,[.0006,k%2===0?.009:.005,.00055],[r+.0004,0,z],tick,.0001);if(k%2===0)ctx.fillText(String(k*5),25,970-k*83);}ctx.font='bold 33px sans-serif';ctx.fillText('50 mL',20,55);const texture=new THREE.CanvasTexture(fontCanvas);texture.colorSpace=THREE.SRGBColorSpace;const marks=new THREE.Mesh(new THREE.PlaneGeometry(.022,.205),new THREE.MeshBasicMaterial({map:texture,transparent:true,side:THREE.DoubleSide,depthWrite:false}));marks.rotation.x=Math.PI/2;marks.position.set(0,-r-.0005,.12);g.add(marks);return g;}
function buildRoute(){
  disposeGroup(route);
  const points=cfg.navigation.route_xy.map(p=>new THREE.Vector3(...p,.018));
  const mat=new THREE.MeshBasicMaterial({color:0x44765a,transparent:true,opacity:.8});
  for(let i=1;i<points.length;i++){
    const start=points[i-1],end=points[i],direction=end.clone().sub(start),length=direction.length();direction.normalize();
    for(let d=0;d<length;d+=.16){const a=start.clone().addScaledVector(direction,d),b=start.clone().addScaledVector(direction,Math.min(d+.085,length));const piece=mesh(new THREE.CylinderGeometry(.005,.005,a.distanceTo(b),8),mat,route);piece.position.copy(a).add(b).multiplyScalar(.5);piece.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),direction);}
    if(length>.35){const center=start.clone().lerp(end,.68);route.add(new THREE.ArrowHelper(direction,center,.17,0x44765a,.07,.055));}
  }
  const center=new THREE.Vector3(cfg.robot.target_base_x,cfg.robot.target_base_y,.022),yaw=cfg.robot.target_base_yaw;
  const [length,width]=cfg.navigation.base_footprint_xy;
  const corners=[[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2],[-length/2,-width/2]].map(([x,y])=>new THREE.Vector3(x*Math.cos(yaw)-y*Math.sin(yaw)+center.x,x*Math.sin(yaw)+y*Math.cos(yaw)+center.y,center.z));
  route.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(corners),new THREE.LineBasicMaterial({color:0x3e6e51,transparent:true,opacity:.85})));
  route.add(new THREE.ArrowHelper(new THREE.Vector3(Math.cos(yaw),Math.sin(yaw),0),center,.27,0x315d42,.075,.07));
  label('绕桌角',[points[2].x-.03,points[2].y+.23,.035],'route-label');
  label('B 侧停靠',[center.x,center.y-.36,.035],'route-label');
  route.visible=$('show-route').checked;
}
function buildTask(){disposeGroup(task);objects.length=0;labels.splice(0).forEach(l=>l.element.remove());sourceCenter=cfg.source_tray.center_xy;measureCenter=cfg.cylinder.center_xy;tray(task,cfg.source_tray.size_xy,sourceCenter);tray(task,cfg.target_tray.size_xy,cfg.target_tray.center_xy);for(const spec of cfg.bottles){const xy=[sourceCenter[0]+spec.offset_xy[0],sourceCenter[1]+spec.offset_xy[1]];makeBottle(spec,xy);}makeCylinder(measureCenter);selectedRing=ring(task,cfg.bottle.radius+.009,.0016,[0,0,cfg.table.height+.017],new THREE.MeshBasicMaterial({color:colors[selected].hex,transparent:true,opacity:.9}));label('A · 四色颜料',[sourceCenter[0]+.04,sourceCenter[1]+.27,cfg.table.height+.01]);label('B · 量筒（远端桌侧）',[measureCenter[0]+.06,measureCenter[1]-.31,cfg.table.height+.02]);buildRoute();selectBottle(selected);}
function selectBottle(key){selected=key;const spec=cfg.bottles.find(b=>b.color===key);const pos=[sourceCenter[0]+spec.offset_xy[0],sourceCenter[1]+spec.offset_xy[1]];if(selectedRing){selectedRing.position.set(...pos,cfg.table.height+.017);selectedRing.material.color.setHex(colors[key].hex);}$('selected-name').textContent=colors[key].label+'瓶子';$('selected-dot').style.background='#'+colors[key].hex.toString(16).padStart(6,'0');$('selected-location').textContent=`X ${pos[0].toFixed(2)} / Y ${pos[1].toFixed(2)} m`;document.querySelectorAll('[data-color]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.color===key)));}

function setView(name){const [cx,cy]=cfg.cylinder.center_xy;const z=cfg.table.height;const views={photo:{pos:[2.3,-1.7,2.0],target:[.4,.4,.82]},overview:{pos:[2.0,-2.2,2.5],target:[.35,.25,.74]},top:{pos:[.30,.15,4.6],target:[.30,.15,.74]},bottles:{pos:[-.15,-.12,1.25],target:[.23,.58,.84]},cylinder:{pos:[cx+.6,cy-.8,z+.5],target:[cx,cy,z+.11]}};const v=views[name];camera.position.set(...v.pos);controls.target.set(...v.target);controls.update();document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===name)));}
buildDesk();buildTask();setView('photo');
const bytes=Uint8Array.from(atob(window.ROBOT_GLB),c=>c.charCodeAt(0));
new GLTFLoader().parse(bytes.buffer,'',gltf=>{robot.add(gltf.scene);robot.traverse(o=>{if(o.isMesh){o.castShadow=true;o.receiveShadow=true;}});$('load-state').textContent='场景已载入';$('load-state').classList.add('loaded');},error=>{$('load-state').textContent='机器人模型载入失败';console.error(error);});
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
document.querySelectorAll('[data-color]').forEach(b=>b.addEventListener('click',()=>selectBottle(b.dataset.color)));
$('show-robot').addEventListener('change',e=>{robot.visible=e.target.checked;});
$('show-labels').addEventListener('change',e=>{$('labels').hidden=!e.target.checked;});
$('show-grid').addEventListener('change',e=>{grid.visible=e.target.checked;});
$('show-axes').addEventListener('change',e=>{axis.visible=e.target.checked;});
$('show-route').addEventListener('change',e=>{route.visible=e.target.checked;});
$('target-x').addEventListener('input',e=>{const x=Number(e.target.value);cfg.target_tray.center_xy[0]=x;cfg.cylinder.center_xy[0]=x;cfg.robot.target_base_x=x-(initial.cylinder.center_xy[0]-initial.robot.target_base_x);cfg.navigation.route_xy.at(-1)[0]=cfg.robot.target_base_x;$('target-x-value').textContent=x.toFixed(2)+' m';buildTask();});
$('table-height').addEventListener('input',e=>{cfg.table.height=Number(e.target.value);$('height-value').textContent=cfg.table.height.toFixed(2)+' m';buildDesk();buildTask();});
$('reset-layout').addEventListener('click',()=>{Object.assign(cfg,structuredClone(initial));$('target-x').value=cfg.cylinder.center_xy[0];$('target-x-value').textContent=cfg.cylinder.center_xy[0].toFixed(2)+' m';$('table-height').value=cfg.table.height;$('height-value').textContent=cfg.table.height.toFixed(2)+' m';buildDesk();buildTask();setView('photo');});
$('export-config').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(cfg,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='scene_config.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
$('reference-image').src=window.REFERENCE_PHOTO;$('full-reference').src=window.REFERENCE_PHOTO;
$('photo-open').addEventListener('click',()=>$('photo-dialog').showModal());$('photo-close').addEventListener('click',()=>$('photo-dialog').close());$('photo-dialog').addEventListener('click',e=>{if(e.target===$('photo-dialog'))$('photo-dialog').close();});
const pointer=new THREE.Vector2(),raycaster=new THREE.Raycaster();let down=null;
renderer.domElement.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];});
renderer.domElement.addEventListener('pointerup',e=>{if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>5)return;const rect=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera);const hit=raycaster.intersectObjects(objects,true)[0];if(hit){let o=hit.object;while(o&&!o.userData.color)o=o.parent;if(o)selectBottle(o.userData.color);}});
function resize(){const w=host.clientWidth,h=host.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}
new ResizeObserver(resize).observe(host);resize();
const projection=new THREE.Vector3();function draw(){requestAnimationFrame(draw);controls.update();renderer.render(scene,camera);for(const item of labels){projection.copy(item.pos).project(camera);item.element.style.left=((projection.x+1)*.5*host.clientWidth)+'px';item.element.style.top=((-projection.y+1)*.5*host.clientHeight)+'px';item.element.style.display=projection.z>1||(!route.visible&&item.kind==='route-label')?'none':'';}}draw();
window.sceneReconstruction={getConfig:()=>structuredClone(cfg),getObjectCount:()=>objects.length,getView:()=>({camera:camera.position.toArray(),target:controls.target.toArray()})};
