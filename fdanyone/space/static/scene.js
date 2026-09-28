import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {BodyMesh} from '/static/body.js';
import {GaussianScene} from '/static/gaussians.js';

const Y = new THREE.Vector3(0,1,0);
const color = rgb => new THREE.Color().setRGB(...rgb.map(v=>v/255),THREE.SRGBColorSpace);

function ready(video) {
  return new Promise((resolve,reject) => {
    const cleanup = () => { clearTimeout(timer); video.removeEventListener('loadeddata',loaded); video.removeEventListener('error',failed); };
    const loaded = () => { cleanup(); resolve(); };
    const failed = () => { cleanup(); reject(Error('A camera video could not be decoded. Try opening the result again.')); };
    const timer = setTimeout(failed,45000);
    video.addEventListener('loadeddata',loaded,{once:true}); video.addEventListener('error',failed,{once:true});
    video.load();
  });
}

export class SceneViewer {
  constructor(host,{onTime,onSelect,onError,onTraining=()=>{}}) {
    this.host=host;this.onTime=onTime;this.onSelect=onSelect;this.onError=onError;this.dirty=true;
    this.renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.renderer.outputColorSpace=THREE.SRGBColorSpace;
    host.replaceChildren(this.renderer.domElement);this.renderer.domElement.tabIndex=0;
    this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#161616');
    const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(3,5,4);
    this.scene.add(new THREE.HemisphereLight(0xffffff,0x39445b,2),light);
    this.camera=new THREE.PerspectiveCamera(45,1,.01,1000);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;
    this.controls.addEventListener('change',()=>{this.dirty=true;});
    this.controls.minDistance=.1;this.controls.maxDistance=100;this.controls.zoomToCursor=true;
    this.grid=new THREE.GridHelper(30,60,0x505050,0x333333);this.grid.position.y=-.025;this.scene.add(this.grid);
    this.rig=new THREE.Group();this.body=new THREE.Group();this.planes=new THREE.Group();this.scene.add(this.rig,this.body,this.planes);
    this.onTraining=onTraining;
    this.gaussians=new GaussianScene(this.renderer,state=>{
      this.dirty=true;
      if(state.loaded&&!state.error&&!this.has4d){
        this.has4d=true;if(!this.bodyVisibilityChanged)this.body.visible=false;
      }
      host.dataset.model=state.loaded?'ready':this.has4d?'ready':state.loading?'loading':'error';
      if(state.loaded&&state.info)host.dataset.modelVersion=state.info.version;
      onTraining(state);
    });
    this.scene.add(this.gaussians);
    this.entries=[];this.pickables=[];this.playing=false;this.time=0;this.fps=25;this.frames=1;this.radius=3;this.selected=0;this.lastFrame=-1;
    this.home();
    this.resize=new ResizeObserver(()=>{const {width,height}=host.getBoundingClientRect();if(!width||!height)return;this.renderer.setSize(width,height,false);this.camera.aspect=width/height;this.camera.updateProjectionMatrix();this.dirty=true;});this.resize.observe(host);
    this.raycaster=new THREE.Raycaster();this.raycaster.params.Line.threshold=.04;
    this.renderer.domElement.addEventListener('pointerdown',e=>this.down=[e.clientX,e.clientY]);
    this.renderer.domElement.addEventListener('pointerup',e=>{
      if(!this.down||Math.hypot(e.clientX-this.down[0],e.clientY-this.down[1])>5)return;
      const r=host.getBoundingClientRect();this.raycaster.setFromCamera(new THREE.Vector2((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1),this.camera);
      const hit=this.raycaster.intersectObjects(this.pickables).find(h=>h.object.visible && h.object.parent.visible);
      if(hit)this.selectCamera(hit.object.userData.camera);
    });
    this.renderer.domElement.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();this.setPlaying(!this.playing);}if(e.key.toLowerCase()==='r')this.home();});
    this.renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();this.setPlaying(false);onError('The graphics context was lost. Refresh the page to reopen the scene.');});
    this.lastTick=performance.now();this.renderer.setAnimationLoop(now=>this.tick(now));
  }
  clearGroup(group) {
    group.traverse(o=>{o.geometry?.dispose();for(const m of o.material ? (Array.isArray(o.material)?o.material:[o.material]):[]){m.map?.dispose();m.dispose();}});group.clear();
  }
  clear() {
    this.meshAbort?.abort();this.meshAbort=null;this.meshUrl=null;
    this.setPlaying(false);
    for(const e of this.entries){if(e.video){e.video.pause();e.video.removeAttribute('src');e.video.load();e.video.remove();}}
    this.entries=[];this.pickables=[];this.clearGroup(this.rig);this.clearGroup(this.body);this.clearGroup(this.planes);this.lastFrame=-1;this.time=0;
    this.bodyMesh=null;this.bones=null;this.joints=null;
    this.gaussians.clear();this.has4d=false;this.bodyVisibilityChanged=false;this.body.visible=true;this.gaussians.visible=true;
    delete this.host.dataset.model;delete this.host.dataset.modelVersion;this.onTraining({cleared:true});
  }
  setTraining(info) {this.gaussians.setSource(info);}
  requestRender() {this.dirty=true;}
  async setScene(data,urls=[],preserve=false) {
    const oldTime=this.time,oldPlaying=this.playing;
    if(!preserve){this.clear();this.data=data;this.frames=data.frames;this.fps=data.fps;this.buildRig(data.cameras);this.buildBody();this.home();}
    const loads=[];
    if(data.mesh?.url && data.mesh.url!==this.meshUrl)loads.push(this.loadBodyMesh(data.mesh.url));
    for(let i=0;i<urls.length;i++) {
      const e=this.entries[i],url=urls[i];if(!url||!e||e.url===url)continue;
      const video=document.createElement('video');video.muted=true;video.playsInline=true;video.preload='auto';video.src=url;
      video.addEventListener('seeked',()=>{this.dirty=true;});
      e.video=video;e.url=url;
      loads.push(ready(video).then(()=>{
        const texture=new THREE.VideoTexture(video);texture.colorSpace=THREE.SRGBColorSpace;
        e.plane.material.map?.dispose();e.plane.material.map=texture;e.plane.material.color.set(0xffffff);e.plane.material.opacity=1;e.plane.material.needsUpdate=true;
        this.dirty=true;
        video.addEventListener('error',()=>{this.setPlaying(false);this.onError('Camera playback failed. Reopen this result.');});
      }));
    }
    this.setPlaying(false);
    try {await Promise.all(loads);} catch(e) {this.setPlaying(false);throw e;}
    this.seek(preserve?Math.floor(oldTime*this.fps):0);
    this.selectCamera(preserve?this.selected:Math.max(0,urls.findIndex(Boolean)));
    await this.setPlaying(preserve?oldPlaying:this.frames>1);
  }
  buildRig(cameras) {
    const radii=[];
    cameras.forEach((c,i)=>{
      const m=new THREE.Matrix4().set(...c.camera_to_world.flat());const origin=new THREE.Vector3().setFromMatrixPosition(m);radii.push(Math.hypot(origin.x,origin.z));
      const invK=new THREE.Matrix3().set(...c.K.flat()).invert();const depth=.48;
      const corners=[[0,0],[c.image_width,0],[c.image_width,c.image_height],[0,c.image_height]].map(([u,v])=>new THREE.Vector3(u,v,1).applyMatrix3(invK).multiplyScalar(depth).applyMatrix4(m));
      const segments=[];for(let n=0;n<4;n++)segments.push(origin,corners[n],corners[n],corners[(n+1)%4]);
      const lines=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segments),new THREE.LineBasicMaterial({color:0xa0a0a0}));lines.userData.camera=i;this.rig.add(lines);this.pickables.push(lines);
      const geom=new THREE.BufferGeometry().setFromPoints(corners);geom.setIndex([0,1,2,0,2,3]);geom.setAttribute('uv',new THREE.Float32BufferAttribute([0,1,1,1,1,0,0,0],2));
      const plane=new THREE.Mesh(geom,new THREE.MeshBasicMaterial({color:0x626262,side:THREE.DoubleSide,transparent:true,opacity:.18,depthWrite:false}));plane.userData.camera=i;this.planes.add(plane);this.pickables.push(plane);
      const label=document.createElement('canvas');label.width=96;label.height=48;const ctx=label.getContext('2d');ctx.fillStyle='#242424';ctx.fillRect(0,0,96,48);ctx.fillStyle='#e8e8e8';ctx.font='bold 28px sans-serif';ctx.textAlign='center';ctx.fillText(String(c.camera_id).padStart(2,'0'),48,34);
      const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(label),depthTest:false}));sprite.position.copy(origin).add(new THREE.Vector3(0,.12,0));sprite.scale.set(.22,.11,1);this.rig.add(sprite);
      this.entries.push({plane,lines,origin,camera:c});
    });
    this.radius=Math.max(1,...radii);
  }
  buildBody() {
    if(!this.data.keypoints)return;
    const links=this.data.links,ids=this.data.joint_ids;
    this.bones=new THREE.InstancedMesh(new THREE.CylinderGeometry(.012,.012,1,6),new THREE.MeshBasicMaterial(),links.length);
    this.joints=new THREE.InstancedMesh(new THREE.SphereGeometry(.024,8,6),new THREE.MeshBasicMaterial(),ids.length);
    this.bones.frustumCulled=false;this.joints.frustumCulled=false;
    links.forEach((link,i)=>this.bones.setColorAt(i,color(link.color)));ids.forEach((_,i)=>this.joints.setColorAt(i,color(this.data.joint_colors[i])));
    this.body.add(this.bones,this.joints);this.updateBody(0);
  }
  async loadBodyMesh(url) {
    this.meshAbort?.abort();this.meshAbort=new AbortController();
    const {signal}=this.meshAbort;this.meshUrl=url;
    try {
      const mesh=await BodyMesh.load(url,this.frames,signal);
      if(signal.aborted){mesh.geometry.dispose();mesh.material.dispose();return;}
      if(this.bodyMesh){this.body.remove(this.bodyMesh);this.bodyMesh.geometry.dispose();this.bodyMesh.material.dispose();}
      this.bodyMesh=mesh;this.body.add(mesh);
      if(this.bones)this.bones.visible=false;
      if(this.joints)this.joints.visible=false;
      this.lastFrame=-1;this.updateBody(Math.floor(this.time*this.fps));
      this.dirty=true;
    } catch(e) {
      if(!signal.aborted)this.onError(`${e.message} Showing the skeleton instead.`);
    }
  }
  updateBody(frame) {
    frame=Math.max(0,Math.min(this.frames-1,Math.floor(frame)));
    if(this.lastFrame===frame)return;
    if(this.bodyMesh){this.bodyMesh.setFrame(frame);this.lastFrame=frame;return;}
    if(!this.data?.keypoints)return;
    const points=this.data.keypoints[frame];const dummy=new THREE.Object3D();
    this.data.links.forEach((link,i)=>{
      const a=new THREE.Vector3(...points[link.a]),b=new THREE.Vector3(...points[link.b]),delta=b.clone().sub(a);
      dummy.position.copy(a).add(b).multiplyScalar(.5);dummy.scale.set(1,delta.length(),1);dummy.quaternion.setFromUnitVectors(Y,delta.normalize());dummy.updateMatrix();this.bones.setMatrixAt(i,dummy.matrix);
    });
    dummy.quaternion.identity();dummy.scale.set(1,1,1);
    this.data.joint_ids.forEach((id,i)=>{dummy.position.fromArray(points[id]);dummy.updateMatrix();this.joints.setMatrixAt(i,dummy.matrix);});
    this.bones.instanceMatrix.needsUpdate=true;this.joints.instanceMatrix.needsUpdate=true;this.lastFrame=frame;
  }
  home() {
    const center=new THREE.Vector3(0,.9,0);this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(1.15,.85,1.5).multiplyScalar(this.radius*1.5));this.controls.update();
  }
  selectCamera(index) {
    this.dirty=true;
    this.selected=index;this.entries.forEach((e,i)=>e.lines.material.color.set(i===index?0xf09ad6:0xa0a0a0));this.onSelect(index,this.entries[index]?.video,this.entries[index]?.url);
  }
  async setPlaying(value) {
    this.playing=Boolean(value)&&this.frames>1;
    const clips=this.entries.filter(e=>e.video).map(e=>e.video);
    if(!this.playing){
      clips.forEach(v=>v.pause());
      this.seek(Math.floor((clips[0]?.currentTime ?? this.time)*this.fps+1e-5));
      return;
    }
    this.lastTick=performance.now();
    if(this.time>=(this.frames-1)/this.fps)this.seek(0);
    try {await Promise.all(clips.map(v=>v.play()));}catch(e){this.playing=false;clips.forEach(v=>v.pause());if(e.name!=='AbortError')this.onError('Playback was blocked by the browser. Click Play to start.');}
    this.notify();
  }
  seek(frame) {
    this.dirty=true;
    this.time=Math.max(0,Math.min(this.frames-1,frame))/this.fps;
    for(const e of this.entries)if(e.video)e.video.currentTime=this.time;
    this.lastTick=performance.now();this.updateBody(Math.round(this.time*this.fps));this.notify();
  }
  notify() {this.onTime({frame:Math.min(this.frames-1,Math.floor(this.time*this.fps+1e-5)),frames:this.frames,time:this.time,fps:this.fps,playing:this.playing});}
  tick(now) {
    const dt=Math.max(0,Math.min(.1,(now-this.lastTick)/1000));this.lastTick=now;
    if(this.playing) {
      const clips=this.entries.filter(e=>e.video).map(e=>e.video),master=clips[0];
      if(master?.ended || (!master && this.time+dt>=this.frames/this.fps)){this.seek(0);if(master)this.setPlaying(true);}
      else this.time=master?master.currentTime:this.time+dt;
      // All planes and the enlarged camera pane share the same video elements.
      // Correct clock drift; scrubbing explicitly seeks every camera together.
      for(const v of clips.slice(1))if(!v.seeking&&Math.abs(v.currentTime-this.time)>1.5/this.fps)v.currentTime=this.time;
      this.updateBody(Math.min(this.frames-1,Math.floor(this.time*this.fps+1e-5)));this.notify();
    }
    const moved=this.controls.update();this.camera.updateMatrixWorld();
    this.gaussians.update(this.camera,Math.min(1,Math.max(0,this.time*this.fps/Math.max(1,this.frames-1))));
    if(this.playing||moved||this.dirty||this.gaussians.dirty){
      this.renderer.render(this.scene,this.camera);this.dirty=false;this.gaussians.dirty=false;
    }
  }
  dispose() {this.renderer.setAnimationLoop(null);this.clear();this.resize.disconnect();this.controls.dispose();this.grid.geometry.dispose();this.grid.material.dispose();this.renderer.dispose();}
}
