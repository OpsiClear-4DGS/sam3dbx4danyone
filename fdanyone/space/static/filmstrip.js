// A small, cancellable thumbnail strip decoded in the browser; no generated media is altered.
function waitFor(video,event,ready,signal,action) {
  return new Promise((resolve,reject)=>{
    const cleanup=()=>{clearTimeout(timer);video.removeEventListener(event,done);video.removeEventListener('error',failed);signal.removeEventListener('abort',aborted);};
    const done=()=>{if(ready()){cleanup();resolve();}};
    const failed=()=>{cleanup();reject(Error('Preview thumbnails unavailable'));};
    const aborted=()=>{cleanup();reject(new DOMException('Cancelled','AbortError'));};
    const timer=setTimeout(failed,12000);
    video.addEventListener(event,done);video.addEventListener('error',failed);signal.addEventListener('abort',aborted,{once:true});
    if(signal.aborted){aborted();return;}
    try{action?.();done();}catch(e){cleanup();reject(e);}
  });
}
export async function buildFilmstrip(url,duration,host,signal,onFirstFrame) {
  const video=document.createElement('video');video.muted=true;video.playsInline=true;video.preload='auto';
  try {
    await waitFor(video,'loadeddata',()=>video.readyState>=2,signal,()=>{video.src=url;video.load();});
    const count=12;
    for(let i=0;i<count;i++) {
      if(signal.aborted)break;
      const time=Math.min(duration-.05,duration*(i+.5)/count);
      await waitFor(video,'seeked',()=>!video.seeking && video.readyState>=2,signal,()=>{video.currentTime=Math.max(0,time);});
      if(signal.aborted)break;
      const canvas=document.createElement('canvas');canvas.width=128;canvas.height=72;
      const context=canvas.getContext('2d'), scale=Math.max(128/video.videoWidth,72/video.videoHeight);
      context.drawImage(video,(128-video.videoWidth*scale)/2,(72-video.videoHeight*scale)/2,video.videoWidth*scale,video.videoHeight*scale);
      canvas.style.left=`${i/count*100}%`;canvas.style.width=`${100/count}%`;
      host.append(canvas);
      if(i===0)onFirstFrame?.(canvas.toDataURL('image/jpeg',.75));
    }
  } catch(e) {
    // Trimming still works when the browser cannot decode thumbnails.
    if(e.name!=='AbortError')host.classList.add('unavailable');
  } finally {video.removeAttribute('src');video.load();}
}
