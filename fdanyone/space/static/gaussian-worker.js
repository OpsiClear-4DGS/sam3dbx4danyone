// FTGS parsing and animated depth sorting come from the pinned AGPL submodule.
import {readModel} from '/ftgs/model.js';
import {sortVisible} from '/ftgs/sort.js';

const models=new Map();
self.onmessage=async({data})=>{
  try {
    if(data.type==='load') {
      const response=await fetch(data.url);
      if(!response.ok)throw Error(`4D preview request failed (${response.status}).`);
      const model=await readModel(await response.blob(),{maxPoints:Infinity});
      if(model.timelineMode!==0)throw Error('This scene requires a continuous 4DGS timeline.');
      models.set(data.id,{positionTime:model.positionTime,velocityDuration:model.velocityDuration,
        alpha:model.alpha,useVelocity:model.useVelocity,opacityFloor:model.opacityFloor});
      // Keep only the compact motion/sort arrays here; transfer GPU staging data.
      const result={count:model.count,degree:model.degree,coefficients:model.coefficients,
        audio:model.audio??null,
        useVelocity:model.useVelocity,opacityFloor:model.opacityFloor,
        positionTime:model.positionTime.slice(),velocityDuration:model.velocityDuration.slice(),
        covarianceA:model.covarianceA,covarianceB:model.covarianceB,sh:model.sh};
      self.postMessage({type:'loaded',id:data.id,model:result},
        [result.positionTime,result.velocityDuration,result.covarianceA,result.covarianceB,result.sh].map(v=>v.buffer));
    } else if(data.type==='sort') {
      const model=models.get(data.id);
      if(!model)return;
      const order=sortVisible(model,data.time,data.view,data.near);
      self.postMessage({type:'sorted',id:data.id,order},[order.buffer]);
    } else if(data.type==='retain') {
      for(const id of models.keys())if(id!==data.id)models.delete(id);
    }
  } catch(error) {
    self.postMessage({type:'error',id:data.id,operation:data.type,message:error.message});
  }
};
