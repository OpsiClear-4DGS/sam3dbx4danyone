// Match the server's end-exclusive, frame-indexed selection for responsive editing.
function lowerBound(times, time) {
  let left=0, right=times.length;
  while(left<right) {
    const middle=Math.floor((left+right)/2);
    if(times[middle]<time)left=middle+1;else right=middle;
  }
  return left;
}

export function previewClip(info, edit) {
  if(!info?.timestamps || !info.chunk_frames)return null;
  const times=info.timestamps, length=info.chunk_frames, stride=edit.skip+1;
  const first=lowerBound(times,edit.start), stop=lowerBound(times,edit.end);
  const kept=Math.ceil((stop-first)/stride);
  if(kept<length)return null;
  const starts=[];
  for(let offset=0;offset+length<=kept;offset+=length)starts.push(offset);
  if(starts.at(-1)+length<kept)starts.push(kept-length);
  const frameDuration=info.fps_den/info.fps_num;
  const chunks=starts.map((offset,index)=>{
    const firstFrame=first+offset*stride, lastFrame=first+(offset+length-1)*stride;
    return {index,label:`Chunk ${index+1}`,first:firstFrame,last:lastFrame,
      start:times[firstFrame],end:Math.min(edit.end,times[lastFrame]+frameDuration),
      overlap_frames:index?Math.max(0,starts[index-1]+length-offset):0};
  });
  return {...edit,stride,kept_frames:kept,chunks,chunk_duration:length*info.fps_den/info.fps_num};
}

export function snapClipEnd(info, edit, width) {
  if(!info?.timestamps || !info.chunk_frames || width<=12)return null;
  const times=info.timestamps, length=info.chunk_frames, stride=edit.skip+1;
  const first=lowerBound(times,edit.start);
  const kept=Math.ceil((lowerBound(times,edit.end)-first)/stride);
  // A small screen-space magnet, capped at eight kept frames even on long clips.
  const radius=Math.min(8*info.duration/(width-12),8*stride*info.fps_den/info.fps_num);
  let target=null, distance=Infinity;
  for(const count of new Set([Math.floor(kept/length),Math.ceil(kept/length)])) {
    if(count<1 || first+(count*length-1)*stride>=times.length)continue;
    const end=times[first+count*length*stride] ?? info.duration;
    // Duplicate presentation times can make a particular boundary unreachable.
    if(Math.ceil((lowerBound(times,end)-first)/stride)!==count*length)continue;
    const delta=Math.abs(end-edit.end);
    if(delta<=radius && delta<distance){target=end;distance=delta;}
  }
  return target;
}
