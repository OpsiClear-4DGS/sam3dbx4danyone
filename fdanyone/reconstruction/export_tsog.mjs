// Adapt Vanilla's explicit temporal fields to the pinned original TSOG encoder.
// Encoder and format provenance: docs/THIRD_PARTY_NOTICES.md.
import {open, readFile, writeFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {parseHeader} from '../../third_party/FreeTimeGsVanilla/player/ftgs.js';
import {packageTSOG} from '../../third_party/FreeTimeGsVanilla/player/tsog-package.js';

export function columnsFromPLY(bytes) {
  const header=parseHeader(new TextDecoder('latin1').decode(bytes.subarray(0,1024*1024)));
  if(bytes.length!==header.dataOffset+header.sourceCount*header.stride)
    throw Error('FTGS file size does not match its header.');
  if(!header.nFrames||header.nFrames<2)throw Error('Animated TSOG export requires a recorded frame count of at least two.');
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  const get=(i,key)=>view.getFloat32(header.dataOffset+i*header.stride+header.offsets[key],true);
  const names=['x','y','z',...Array.from({length:3},(_,i)=>`f_dc_${i}`),
    ...Array.from({length:45},(_,i)=>`f_rest_${i}`),...Array.from({length:3},(_,i)=>`scale_${i}`),
    ...Array.from({length:4},(_,i)=>`rot_${i}`),'opacity','t','t_scale','motion_0','motion_1','motion_2'];
  const columns=names.map(name=>{
    let source=name;
    if(name==='t')source='time';
    if(name==='t_scale')source='log_duration';
    if(name.startsWith('motion_'))source=header.useVelocity?name.replace('motion_','velocity_'):null;
    if(name.startsWith('f_rest_')) {
      const index=Number(name.slice(7)),rest=header.coefficients-1;
      source=index%15<rest?`f_rest_${Math.floor(index/15)*rest+index%15}`:null;
    }
    const values=new Float32Array(header.sourceCount);
    for(let i=0;i<values.length;i++) {
      let value=source===null?0:get(i,source);
      if(name==='t_scale')value=Math.max(header.minDuration,Math.exp(value));
      values[i]=value;
      if(!Number.isFinite(values[i]))throw Error(`Non-finite ${name} in Gaussian ${i}.`);
    }
    return {name,values};
  });
  const rotations=columns.filter(column=>column.name.startsWith('rot_'));
  for(let i=0;i<header.sourceCount;i++)
    if(!rotations.some(column=>column.values[i]!==0))throw Error(`Zero quaternion in Gaussian ${i}.`);
  return {header,columns};
}

async function main(args) {
  const {writeTsog,loadSplatTransform}=await import('../../third_party/tsog/dist/index.mjs');
  const peer=await loadSplatTransform();
  if(args.length===1&&args[0]==='--check') {
    await peer.createDevice();
    return;
  }
  const [input,output,fpsText,audioPath]=args,fps=Number(fpsText);
  if(![3,4].includes(args.length)||!output.endsWith('.tsog')||!(fps>0&&Number.isFinite(fps)))
    throw Error('Usage: node export_tsog.mjs input.ftgs.ply output.tsog fps [audio.m4a]');
  const {header,columns}=columnsFromPLY(await readFile(input));
  const table=new peer.DataTable(columns.map(({name,values})=>new peer.Column(name,values)));
  table.tsogSourceMeta={timelineType:1};
  // Deterministic initialization; retain all Gaussians and all SH3 coefficients.
  let seed=42;
  Math.random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
  const handle=await open(output,'wx');
  try {
    await writeTsog(handle,table,output,{cpu:false,iterations:10,bits:{motion:16}});
  } finally {await handle.close();}
  // The original continuous exporter omits clip timing. Repackage metadata
  // without recompressing the attribute images, using the player's extension.
  const packaged=await packageTSOG(new Blob([await readFile(output)]),{
    playback:{fps,duration:(header.nFrames-1)/fps,rate:1,loop:true},
    ...(audioPath?{audio:new File([await readFile(audioPath)],'track.m4a',{type:'audio/mp4'})}:{}),
  });
  await writeFile(output,new Uint8Array(await packaged.arrayBuffer()));
}

if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href) {
  // Dawn retains its device thread. Exit only after all output writes finish.
  try {await main(process.argv.slice(2));process.exit(0);}
  catch(error){console.error(error.stack??error.message);process.exit(1);}
}
