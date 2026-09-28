// Three.js integration of the pinned FreeTimeGsVanilla player (AGPL-3.0).
import * as THREE from 'three';
import {VERTEX,FRAGMENT} from './gaussian-shaders.js';

function matrix(rows) {
  if(!Array.isArray(rows)||rows.length!==4||rows.some(row=>!Array.isArray(row)||row.length!==4||row.some(v=>!Number.isFinite(v))))
    throw Error('The 4D scene is missing a valid coordinate transform.');
  return new THREE.Matrix4().set(...rows.flat());
}

function texture(values,maxSize) {
  const texels=values.length/4;
  const width=Math.min(maxSize,Math.max(1,2**Math.ceil(Math.log2(Math.sqrt(texels)))));
  const height=Math.ceil(texels/width);
  if(height>maxSize)throw Error('This 4D model exceeds the graphics device texture limit.');
  let padded=values;
  if(values.length!==width*height*4){padded=new Float32Array(width*height*4);padded.set(values);}
  const result=new THREE.DataTexture(padded,width,height,THREE.RGBAFormat,THREE.FloatType);
  result.minFilter=result.magFilter=THREE.NearestFilter;result.generateMipmaps=false;result.needsUpdate=true;
  return result;
}

function createMesh(data,info,maxSize) {
  const uniforms={};
  const arrays={positions:data.positionTime,velocities:data.velocityDuration,
    covarianceA:data.covarianceA,covarianceB:data.covarianceB,harmonics:data.sh};
  try {
    for(const [key,value] of Object.entries(arrays))uniforms[key]={value:texture(value,maxSize)};
    const toWorld=matrix(info.training_to_world),toTraining=matrix(info.world_to_training);
    Object.assign(uniforms,{
      view:{value:new THREE.Matrix4()},eye:{value:new THREE.Vector3()},viewport:{value:new THREE.Vector2()},
      time:{value:0},focal:{value:1},nearPlane:{value:.01},farPlane:{value:1000},
      opacityFloor:{value:data.opacityFloor},coefficients:{value:data.coefficients},
      shDegree:{value:data.degree},useVelocity:{value:data.useVelocity},
    });
    const geometry=new THREE.InstancedBufferGeometry();
    geometry.setAttribute('corner',new THREE.Float32BufferAttribute([-3,-3,3,-3,-3,3,3,3],2));
    const ids=new THREE.InstancedBufferAttribute(new Uint32Array(data.count),1);
    ids.setUsage(THREE.DynamicDrawUsage);geometry.setAttribute('splatId',ids);
    geometry.setIndex([0,1,2,2,1,3]);geometry.instanceCount=0;
    const material=new THREE.RawShaderMaterial({vertexShader:VERTEX,fragmentShader:FRAGMENT,uniforms,
      glslVersion:THREE.GLSL3,transparent:true,depthTest:true,depthWrite:false,
      blending:THREE.CustomBlending,blendSrc:THREE.OneFactor,blendDst:THREE.OneMinusSrcAlphaFactor,
      blendEquation:THREE.AddEquation,premultipliedAlpha:true});
    const mesh=new THREE.Mesh(geometry,material);mesh.frustumCulled=false;mesh.renderOrder=1;
    mesh.userData={toWorld,toTraining};
    return mesh;
  } catch(error) {
    for(const uniform of Object.values(uniforms))if(uniform.value?.isTexture)uniform.value.dispose();
    throw error;
  }
}

function disposeMesh(mesh) {
  if(!mesh)return;
  mesh.geometry.dispose();
  for(const uniform of Object.values(mesh.material.uniforms))if(uniform.value?.isTexture)uniform.value.dispose();
  mesh.material.dispose();
}

export class GaussianScene extends THREE.Group {
  constructor(renderer,onState) {
    super();this.renderer=renderer;this.onState=onState;this.sequence=0;this.retryAt=0;
    this.size=new THREE.Vector2();this.lastSort='';
  }
  setSource(info) {
    this.desired=info;
    this.startLoad();
  }
  startLoad() {
    const info=this.desired;
    if(!info?.model||this.loading||this.staged||(this.current?.info.model===info.model&&this.worker)||performance.now()<this.retryAt)return;
    if(!this.worker) {
      const worker=this.worker=new Worker('/static/gaussian-worker.js',{type:'module'});
      worker.onmessage=event=>{if(this.worker===worker)this.receive(event.data);};
      worker.onerror=event=>{
        if(this.worker===worker)this.fail(`4D preview worker failed: ${event.message}`);
      };
    }
    this.loading={id:++this.sequence,info};this.onState({loading:true,info});
    this.worker.postMessage({type:'load',id:this.loading.id,url:info.model});
  }
  receive(data) {
    if(data.type==='loaded'&&data.id===this.loading?.id) {
      try {
        const limit=this.renderer.capabilities.maxTextureSize;
        const mesh=createMesh(data.model,this.loading.info,limit);
        this.staged={...this.loading,mesh};this.loading=null;this.lastSort='';
      } catch(error){this.fail(error.message);}
    } else if(data.type==='sorted') {
      this.sorting=false;
      const target=this.staged?.id===data.id?this.staged:this.current?.id===data.id?this.current:null;
      if(!target)return;
      const ids=target.mesh.geometry.getAttribute('splatId');
      ids.array.set(data.order);ids.clearUpdateRanges();ids.addUpdateRange(0,data.order.length);ids.needsUpdate=true;
      target.mesh.geometry.instanceCount=data.order.length;
      this.dirty=true;
      if(target===this.staged) {
        if(this.current){this.remove(this.current.mesh);disposeMesh(this.current.mesh);}
        this.current=target;this.staged=null;this.add(target.mesh);
        this.worker.postMessage({type:'retain',id:target.id});
        this.onState({loaded:true,info:target.info});
        this.startLoad();
      }
    } else if(data.type==='error') {
      if(data.operation==='sort')this.sorting=false;
      this.fail(data.message);
    }
  }
  fail(message) {
    this.worker?.terminate();this.worker=null;this.sorting=false;this.lastSort='';
    this.loading=null;
    if(this.staged){disposeMesh(this.staged.mesh);this.staged=null;}
    this.retryAt=performance.now()+5000;
    this.onState({error:message,loaded:Boolean(this.current)});
  }
  update(camera,time) {
    this.startLoad();
    this.renderer.getDrawingBufferSize(this.size);
    for(const target of [this.current,this.staged]) {
      if(!target)continue;
      const {mesh}=target,u=mesh.material.uniforms;
      u.view.value.multiplyMatrices(camera.matrixWorldInverse,mesh.userData.toWorld);
      u.eye.value.setFromMatrixPosition(camera.matrixWorld).applyMatrix4(mesh.userData.toTraining);
      u.viewport.value.copy(this.size);u.focal.value=this.size.y/(2*Math.tan(THREE.MathUtils.degToRad(camera.fov)/2));
      u.time.value=time;u.nearPlane.value=camera.near;u.farPlane.value=camera.far;
    }
    const target=this.staged||this.current;
    if(!target||this.sorting||!this.worker)return;
    const view=target.mesh.material.uniforms.view.value.elements;
    const key=`${target.id}:${time}:${view.join(',')}`;
    if(key===this.lastSort)return;
    this.lastSort=key;this.sorting=true;
    this.worker.postMessage({type:'sort',id:target.id,time,view,near:camera.near});
  }
  clear() {
    this.worker?.terminate();this.worker=null;this.sorting=false;
    if(this.current){this.remove(this.current.mesh);disposeMesh(this.current.mesh);}
    if(this.staged)disposeMesh(this.staged.mesh);
    this.current=this.staged=this.loading=this.desired=null;this.lastSort='';this.retryAt=0;
    super.clear();
  }
}
