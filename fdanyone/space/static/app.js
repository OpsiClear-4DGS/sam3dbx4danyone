import {buildFilmstrip} from '/static/filmstrip.js';
import {previewClip, snapClipEnd} from '/static/clip-timeline.js';
const $ = id => document.getElementById(id);
let viewer, viewerReady, videos = [], uploading = false, exporting = false, running = false, liveBusy = false, liveVersion = '', pendingLayout = null;
let editing=false, committedClip=null, filmstripAbort=null;
let submitting=false, clipInfo=null, clipPlan=null, clipReady=false, planRevision=0, planTimer, previewBounds=null;
let completedJobs=[], resultPaths=[], refreshPromise=null, deletingJob=false, deleteTarget=null, currentResult='';
const galleryChunks=new Map();
let followJob=true, stopping=false, lastGalleryRefresh=0, gallerySignature='';
let workflow='results', editorOrigin='results', workflowJob='', lastResult='';
let trimPointer=null, trimAlt=false, trimSnapped=false;
const message = (text, error = false) => { $('message').textContent = text; $('message').classList.toggle('error', error); $('message').hidden=!text; };
async function api(path, body, method) {
  const response = await fetch(path, body === undefined ? {method:method||'GET'} : {
    method: method||'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data));
  return data;
}
function buttons() {
  $('generate').disabled = submitting || uploading || running || !videos.length || !clipReady;
  $('settings').hidden=!clipReady || editing;
  $('clip-summary').hidden=!clipReady || editing;
  $('cancel').disabled=!running || stopping; $('cancel').hidden=!running;
  $('resume-live').hidden=!running || followJob;$('resume-live').disabled=exporting || liveBusy || deletingJob;
  $('generate').hidden=running;
  $('views').disabled=submitting;$('mode').disabled=submitting;
  $('output-kind').disabled=submitting || running;
  $('uploads').disabled=submitting || uploading;
  for(const button of document.querySelectorAll('#files button'))button.disabled=submitting || uploading;
  $('clip-editor').hidden=!editing;$('scene-card').hidden=editing;
  $('saved-results').hidden=workflow!=='results' || editing || uploading;
  $('back-to-results').hidden=workflow==='results' || editing;
  $('back-to-results').disabled=exporting || liveBusy || deletingJob || submitting || uploading;
  document.body.classList.toggle('editing',editing);
  $('drop-zone').hidden=Boolean(videos.length);$('uploads').tabIndex=videos.length?-1:0;
  const total=clipReady?clipPlan.chunks.length*Number($('views').value):0;
  $('generate').textContent=submitting?'Starting…':running?'Processing…':$('output-kind').value==='4dgs'?'Create 4D scene':`Generate${total?` ${total} videos`:''}`;
  $('delete-job').disabled=exporting || liveBusy || deletingJob;
  for(const control of document.querySelectorAll('.result-card button,.result-card select'))control.disabled=exporting || liveBusy || deletingJob || control.dataset.unavailable==='true';
}
function renderBatch(report) {
  const done=report.jobs.filter(j=>j.status==='completed').length;
  const failed=report.jobs.filter(j=>j.status==='failed').length;
  const total=report.jobs.length || report.planned_chunks || 0;
  $('batch-progress').hidden=report.status==='completed' || report.status==='idle';
  const training=report.jobs.find(j=>j.status==='running' && j.training?.stage==='training')?.training;
  $('batch-status').textContent=running?(report.stage==='training'?'Training 4D scene':report.stage==='preparing training views'?'Preparing training views':report.jobs.length?'Generating views':total>1?'Preparing clips':'Preparing video'):'Processing stopped';
  $('batch-meter').max=Math.max(1,total);
  if(running && (!report.jobs.length || total===1))$('batch-meter').removeAttribute('value');
  else $('batch-meter').value=done;
  if(running && total===1 && training){$('batch-meter').max=training.steps;$('batch-meter').value=training.step;}
  $('batch-detail').hidden=total<=1;
  $('batch-detail').textContent=total>1?(report.jobs.length?`${done} of ${total} chunks complete${failed?` · ${failed} failed`:''}`
    :`${total} chunks queued`):'';
}
function settings() {
  const invalid=Array.from($('settings').elements).find(e=>e.willValidate && !e.validity.valid);
  if(invalid){invalid.reportValidity();throw Error('Check the highlighted setting.');}
  const pitches=cameraPresets[Number($('views').value)];
  return {views:Number($('views').value)/pitches.length, pitches, yaw:0,
    span:360, turbo:$('mode').value === 'turbo', start_time:0,
    fps:'auto', seed:42, train_4dgs:$('output-kind').value==='4dgs'};
}
$('output-kind').onchange=buttons;
function showUploads() {
  $('files').replaceChildren();
  for (const video of videos) {
    const li=document.createElement('li'), button=document.createElement('button'), thumb=document.createElement('img'), name=document.createElement('span');
    thumb.className='source-thumb';thumb.alt='';thumb.hidden=true;
    button.type='button';button.className='source-select';button.setAttribute('aria-label',`Edit ${video.name}`);
    name.className='source-info';
    const filename=document.createElement('span');filename.className='source-name';filename.textContent=video.name;filename.title=video.name;
    const edit=document.createElement('small');edit.textContent='Edit clip';name.append(filename,edit);button.append(thumb,name);
    button.onclick=()=>{if(clipInfo)openClipEditor();else loadClip().catch(e=>message(e.message,true));};
    const replace=document.createElement('button');replace.type='button';replace.className='source-replace quiet';replace.title='Replace video';replace.setAttribute('aria-label','Replace video');
    replace.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h15m-4-4 4 4-4 4M20 17H5m4-4-4 4 4 4"/></svg>';
    replace.onclick=()=>$('uploads').click();
    li.append(button,replace);$('files').append(li);
  }
  if (videos.length) $('source').src = videos[0].url; else $('source').removeAttribute('src');
  buttons();
}
function upload(file) {
  return new Promise((resolve,reject) => {
    const xhr = new XMLHttpRequest(), form = new FormData(); form.append('file',file);
    xhr.open('POST','/api/uploads');
    xhr.upload.onprogress = e => { if (e.lengthComputable) message(`Uploading ${file.name} · ${Math.round(e.loaded/e.total*100)}%`); };
    xhr.onerror = () => reject(Error('Upload failed. Check the server connection.'));
    xhr.onload = () => { try { const result = JSON.parse(xhr.responseText); xhr.status < 300 ? resolve(result) : reject(Error(result.detail)); } catch(e) { reject(e); } };
    xhr.send(form);
  });
}
async function addFiles(files) {
  if(uploading || submitting)return;
  uploading=true;buttons();
  try {
    if(files.length!==1)throw Error('Choose one video; its chunks will run as a batch.');
    const file=files[0];
    if(file.size>2*1024**3)throw Error(`${file.name} exceeds the 2 GiB upload limit.`);
    const uploaded=await upload(file);
    resetClip(); videos=[uploaded]; showUploads();
    await loadClip();
    message('');
  }catch(e){message(e.message,true);}
  finally{uploading=false;$('uploads').value='';buttons();}
}
function resetClip() {
  clipInfo=null; clipPlan=null; clipReady=false; previewBounds=null;committedClip=null;editing=false;
  filmstripAbort?.abort();$('filmstrip').replaceChildren();$('filmstrip').classList.remove('unavailable');
  ++planRevision;clearTimeout(planTimer);$('source').pause();
  $('clip-summary').textContent='';$('chunks').replaceChildren();$('chunk-segments').replaceChildren();$('source-error').hidden=true;
}
function clipEdit() {
  return {start:Number($('trim-start').value),end:Number($('trim-end').value),skip:Number($('skip-frames').value)};
}
function setClipFields(edit) {
  trimSnapped=false;
  $('trim-start').value=String(edit.start);$('trim-end').value=String(edit.end);$('skip-frames').value=String(edit.skip);
  for(const key of ['start','end']) {
    $(`trim-${key}`).max=String(clipInfo.duration);
    $(`trim-${key}-range`).max=String(clipInfo.duration);
    $(`trim-${key}-range`).value=$(`trim-${key}`).value;
  }
  updateTrimTrack();
}
function updateTrimTrack() {
  const edit=clipEdit(), duration=clipInfo?.duration||1;
  $('trim-track').style.setProperty('--trim-start',`${Math.max(0,Math.min(100,edit.start/duration*100))}%`);
  $('trim-track').style.setProperty('--trim-end',`${Math.max(0,Math.min(100,edit.end/duration*100))}%`);
  $('trim-duration').textContent=edit.end>edit.start?`${clipTime(edit.start)}–${clipTime(edit.end)} · ${(edit.end-edit.start).toFixed(2)}s selected`:'';
  $('trim-start-range').setAttribute('aria-valuetext',clipTime(edit.start));$('trim-end-range').setAttribute('aria-valuetext',clipTime(edit.end));
  $('trim-end-range').classList.toggle('snapped',trimSnapped);
  if(trimSnapped)$('trim-end-range').setAttribute('aria-valuetext',`${clipTime(edit.end)}, snapped to a full chunk boundary`);
  $('speed-label').textContent=edit.skip?`Keep 1 of every ${edit.skip+1} frames`:'Every frame';
}
function openClipEditor() {
  editorOrigin=workflow;workflow='prepare';
  editing=true;followJob=false;
  viewer?.setPlaying(false);previewBounds=clipEdit();buttons();
  $('source').pause();$('source').currentTime=Math.max(0,Math.min(clipInfo.duration,previewBounds.start));
  $('clip-title').focus({preventScroll:true});
  if(!clipPlan)schedulePlan();
}
function leaveClipEditor(restore=true) {
  ++planRevision;clearTimeout(planTimer);
  if(restore && committedClip) {
    clipPlan=committedClip.plan;clipReady=true;setClipFields(committedClip.edit);
    $('source').playbackRate=committedClip.edit.skip+1;
    renderClipPlan(clipPlan);
  }
  if(restore)workflow=editorOrigin;
  editing=false;$('source').pause();$('editor-menu').open=false;buttons();
}
async function loadClip() {
  const id=videos[0].id;
  message('Analyzing video…');
  const info=await api(`/api/uploads/${encodeURIComponent(id)}/info?timeline=true`);
  if(videos[0]?.id!==id)return;
  clipInfo=info;
  setClipFields({start:0,end:info.duration,skip:0});
  openClipEditor();message('');
  $('source-playhead').setAttribute('aria-valuemax',String(info.duration));
  $('source-ruler').replaceChildren(...Array.from({length:5},(_,i)=>{const tick=document.createElement('span');tick.textContent=clipTime(i*info.duration/4);return tick;}));
  filmstripAbort?.abort();filmstripAbort=new AbortController();
  void buildFilmstrip(videos[0].url,info.duration,$('filmstrip'),filmstripAbort.signal,url=>{
    const thumb=document.querySelector('.source-thumb');if(thumb){thumb.src=url;thumb.hidden=false;}
  });
}
const clipTime=s=>{const ticks=Math.round(s*100);return `${Math.floor(ticks/6000)}:${((ticks%6000)/100).toFixed(2).padStart(5,'0')}`;};
function renderClipPlan(plan, validated=true) {
  $('clip-plan-note').classList.remove('error');
  $('clip-plan-note').textContent=`${plan.chunks.length} ${plan.chunks.length===1?'chunk':'chunks'} · ${plan.chunk_duration.toFixed(2)}s each`;
  $('overlap-note').hidden=!trimSnapped && !plan.chunks.at(-1).overlap_frames;
  $('overlap-note').textContent=trimSnapped?'Snapped to chunk boundary':'Final chunk overlaps to include the end.';
  $('chunks').replaceChildren();$('chunk-segments').replaceChildren();
  // Keep the drag scale stable while the selection changes.
  $('source-timeline').style.minWidth=`${Math.max(0,Math.ceil(clipInfo.frames/(clipInfo.chunk_frames||121)/plan.stride)*64)}px`;
  for(const chunk of plan.chunks) {
    const button=document.createElement('button');button.type='button';button.setAttribute('aria-pressed','false');
    button.textContent=String(chunk.index+1).padStart(2,'0');
    button.title=`${chunk.label} · ${clipTime(chunk.start)}–${clipTime(chunk.end)}`;
    button.setAttribute('aria-label',`Preview ${chunk.label}: ${chunk.start.toFixed(2)} to ${chunk.end.toFixed(2)} seconds`);
    button.style.left=`${chunk.start/clipInfo.duration*100}%`;
    if(chunk.overlap_frames)button.classList.add('overlapping');
    button.onclick=()=>{
      for(const other of $('chunks').children)other.setAttribute('aria-pressed',String(other===button));
      previewBounds=chunk;$('source').currentTime=chunk.start;playSource();
    };
    $('chunks').append(button);
    const marker=document.createElement('span');
    marker.style.left=`${chunk.start/clipInfo.duration*100}%`;marker.style.width=`${(chunk.end-chunk.start)/clipInfo.duration*100}%`;
    $('chunk-segments').append(marker);
  }
  $('use-chunks').disabled=!validated;$('source-play').disabled=false;
}
function schedulePlan() {
  $('source').pause();
  clipReady=false;clipPlan=null;buttons();
  const revision=++planRevision;clearTimeout(planTimer);
  $('use-chunks').disabled=true;$('chunks').replaceChildren();$('chunk-segments').replaceChildren();$('overlap-note').hidden=true;
  updateTrimTrack();
  const edit=clipEdit();
  previewBounds={start:edit.start,end:edit.end};
  const valid=clipInfo && ['trim-start','trim-end','skip-frames'].every(id=>$(id).value!=='' && $(id).validity.valid)
    && edit.start<edit.end;
  $('source-play').disabled=!valid;
  if(valid)$('source').playbackRate=edit.skip+1;
  $('clip-plan-note').classList.toggle('error',!valid);
  $('clip-plan-note').textContent=valid?'Planning chunks…':'Choose a valid start, end and skip count.';
  if(!valid)return;
  const preview=previewClip(clipInfo,edit);
  if(preview)renderClipPlan(preview,false);
  const video=videos[0].id;
  planTimer=setTimeout(async()=>{
    try {
      const plan=await api('/api/clips/plan',{video,...edit});
      if(revision!==planRevision || videos[0]?.id!==video)return;
      clipPlan=plan;
      renderClipPlan(plan);
    }catch(e){if(revision===planRevision){$('clip-plan-note').textContent=e.message;$('clip-plan-note').classList.add('error');}}
  },250);
}
for(const key of ['start','end']) {
  const field=$(`trim-${key}`), range=$(`trim-${key}-range`);
  field.oninput=()=>{trimSnapped=false;range.value=field.value;schedulePlan();};
  range.onpointerdown=event=>{trimPointer=event.pointerId;trimAlt=event.altKey;};
  range.oninput=()=>{
    const other=Number($(`trim-${key==='start'?'end':'start'}`).value), value=Number(range.value);
    let end=key==='start'?Math.max(0,Math.min(value,other-.001)):Math.min(clipInfo.duration,Math.max(value,other+.001));
    const snapped=key==='end' && trimPointer!==null && !trimAlt
      ?snapClipEnd(clipInfo,{...clipEdit(),end},range.getBoundingClientRect().width):null;
    trimSnapped=snapped!==null;
    if(trimSnapped)end=snapped;
    field.value=String(end);range.value=field.value;
    $('source').currentTime=Number($(`trim-${key}`).value);schedulePlan();
  };
  range.onkeydown=event=>{
    if(!clipInfo || !['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;
    event.preventDefault();trimPointer=null;
    const direction=['ArrowRight','ArrowUp'].includes(event.key)?1:-1;
    range.value=String(Number(field.value)+direction*(event.shiftKey?10:1)*clipInfo.fps_den/clipInfo.fps_num);
    range.dispatchEvent(new Event('input'));
  };
}
document.addEventListener('pointermove',event=>{if(event.pointerId===trimPointer)trimAlt=event.altKey;},true);
for(const event of ['pointerup','pointercancel'])document.addEventListener(event,()=>{trimPointer=null;},true);
window.addEventListener('blur',()=>{trimPointer=null;});
document.addEventListener('keydown',event=>{if(event.key==='Alt')trimAlt=true;});
document.addEventListener('keyup',event=>{if(event.key==='Alt')trimAlt=false;});
$('skip-frames').oninput=()=>{trimSnapped=false;schedulePlan();};
function syncSourceTime() {
  const video=$('source'), time=Number.isFinite(video.currentTime)?video.currentTime:0, duration=clipInfo?.duration||0;
  $('source-time').textContent=`${clipTime(time)} / ${clipTime(duration)}`;
  $('source-playhead').style.left=`${duration?time/duration*100:0}%`;
  $('source-playhead').setAttribute('aria-valuenow',String(time));$('source-playhead').setAttribute('aria-valuetext',clipTime(time));
  $('source-play').classList.toggle('playing',!video.paused);
  $('source-play').setAttribute('aria-label',video.paused?'Play preview':'Pause preview');
}
function playSource() {$('source').play().catch(()=>{});}
function toggleSource() {
  if($('source-play').disabled)return;
  if($('source').paused)playSource();else $('source').pause();
}
function seekSource(time) {
  $('source').pause();previewBounds=clipEdit();
  for(const button of $('chunks').children)button.setAttribute('aria-pressed','false');
  $('source').currentTime=Math.max(0,Math.min(clipInfo.duration,time));syncSourceTime();
}
$('source-play').onclick=toggleSource;
$('source').onclick=toggleSource;
let scrubPointer=null;
function scrub(event) {
  const box=$('trim-track').getBoundingClientRect();seekSource((event.clientX-box.left)/box.width*clipInfo.duration);
}
$('trim-track').onpointerdown=event=>{
  if(!clipInfo || event.target.closest('input,button') || event.button!==0)return;
  event.preventDefault();scrubPointer=event.pointerId;$('trim-track').setPointerCapture(event.pointerId);scrub(event);$('source-playhead').focus({preventScroll:true});
};
$('trim-track').onpointermove=event=>{if(scrubPointer===event.pointerId)scrub(event);};
for(const event of ['pointerup','pointercancel','lostpointercapture'])$('trim-track').addEventListener(event,()=>{scrubPointer=null;});
$('source-playhead').onkeydown=event=>{
  if(!clipInfo)return;
  const step=clipInfo.fps_den/clipInfo.fps_num;
  if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) {
    event.preventDefault();event.stopPropagation();
    seekSource(event.key==='Home'?0:event.key==='End'?clipInfo.duration:$('source').currentTime+(event.key==='ArrowRight'?1:-1)*step*(event.shiftKey?10:1));
  }
};
$('clip-editor').onkeydown=event=>{
  if(event.target.closest('input,select,button,summary') || event.ctrlKey || event.metaKey || event.altKey)return;
  if(event.repeat){if(event.code==='Space')event.preventDefault();return;}
  if(event.code==='Space'){event.preventDefault();toggleSource();}
  const boundary={i:'start',o:'end'}[event.key.toLowerCase()];
  if(boundary){const field=$(`trim-${boundary}`);field.value=String(Number($('source').currentTime.toFixed(3)));field.dispatchEvent(new Event('input'));event.preventDefault();}
};
for(const event of ['timeupdate','seeked','loadedmetadata','play','pause','ended'])$('source').addEventListener(event,syncSourceTime);
$('reset-trim').onclick=()=>{setClipFields({start:0,end:clipInfo.duration,skip:0});$('source').currentTime=0;schedulePlan();};
let sourceAnimation=0;
function sourceTick() {
  if(previewBounds && !$('source').paused && $('source').currentTime>=previewBounds.end){$('source').pause();$('source').currentTime=previewBounds.start;}
  syncSourceTime();sourceAnimation=$('source').paused?0:requestAnimationFrame(sourceTick);
}
$('source').addEventListener('play',()=>{cancelAnimationFrame(sourceAnimation);sourceTick();});
$('source').addEventListener('pause',()=>{cancelAnimationFrame(sourceAnimation);sourceAnimation=0;});
$('source').ontimeupdate=()=>{if(previewBounds && !$('source').paused && $('source').currentTime>=previewBounds.end){$('source').pause();$('source').currentTime=previewBounds.start;}};
$('source').onplay=()=>{if(previewBounds && ($('source').currentTime<previewBounds.start || $('source').currentTime>=previewBounds.end))$('source').currentTime=previewBounds.start;};
$('source').onerror=()=>{$('source-error').hidden=false;$('source-error').textContent='This browser cannot preview the upload. You can still trim using the time fields.';};
$('close-clip').onclick=()=>{leaveClipEditor();document.querySelector('.source-select')?.focus();};
$('use-chunks').onclick=()=>{
  $('editor-menu').open=false;
  if(!clipPlan)return;
  committedClip={plan:clipPlan,edit:clipEdit()};clipReady=true;leaveClipEditor(false);
  $('clip-summary').textContent=`${clipTime(clipPlan.start)}–${clipTime(clipPlan.end)} · ${clipPlan.stride}× · ${clipPlan.chunks.length} ${clipPlan.chunks.length===1?'chunk':'chunks'}`;
  requestLayoutPreview();$('views').focus({preventScroll:true});
};
$('uploads').onchange=event=>addFiles(Array.from(event.target.files));
$('drop-zone').ondragover=event=>{event.preventDefault();$('drop-zone').classList.add('dragging');};
$('drop-zone').ondragleave=()=>$('drop-zone').classList.remove('dragging');
$('drop-zone').ondrop=event=>{event.preventDefault();$('drop-zone').classList.remove('dragging');addFiles(Array.from(event.dataTransfer.files));};
function fitMenu(menu) {
  const panel=menu.querySelector('.editor-menu-content');
  if(!menu.open || !panel)return;
  panel.style.transform='';
  const margin=12, container=menu.closest('.scene-card,.clip-editor,aside')?.getBoundingClientRect();
  const top=Math.max(margin,(container?.top||0)+4), bottom=Math.min(innerHeight-margin,container?.bottom||innerHeight);
  panel.style.maxHeight=`${Math.max(64,bottom-top)}px`;
  const box=panel.getBoundingClientRect();
  const x=Math.max(margin-box.left,Math.min(0,innerWidth-margin-box.right));
  const y=Math.max(top-box.top,Math.min(0,bottom-box.bottom));
  panel.style.transform=`translate(${x}px,${y}px)`;
}
for(const menu of document.querySelectorAll('.editor-menu')) {
  menu.addEventListener('toggle',()=>{if(menu.open)requestAnimationFrame(()=>fitMenu(menu));});
  menu.addEventListener('keydown',e=>{if(e.key==='Escape'){e.stopPropagation();menu.open=false;menu.querySelector('summary').focus();}});
  menu.addEventListener('click',e=>{if(e.target.closest('button'))menu.open=false;});
}
window.addEventListener('resize',()=>{for(const menu of document.querySelectorAll('.editor-menu[open]'))fitMenu(menu);});
document.addEventListener('pointerdown',e=>{for(const menu of document.querySelectorAll('.editor-menu[open]'))if(!menu.contains(e.target))menu.open=false;});
const cameraPresets = {6:[15], 8:[15], 12:[15], 16:[0,30], 18:[0,30], 24:[0,30], 36:[-15,15,45]};
function updateCameraHint() {
  const count=Number($('views').value), pitches=cameraPresets[count];
  $('camera-summary').textContent=`${pitches.length} ${pitches.length===1?'ring':'rings'} · ${pitches.map(p=>`${p}°`).join(' / ')}`;
  $('camera-summary').title=`${count/pitches.length} cameras per ring · full orbit`;
}
$('mode').onchange=()=>{$('model-menu').open=false;};
$('views').onchange=()=>{updateCameraHint();buttons();requestLayoutPreview();};
updateCameraHint();
async function showScene(scene, urls = [], preserve = false) {
  if(!preserve){trainingWatch=null;liveDirectory='';}
  await viewerReady;
  if(!preserve) $('camera').replaceChildren(...scene.cameras.map((c,i)=>new Option(String(c.camera_id).padStart(2,'0'),String(i))));
  $('scene-card').classList.toggle('layout-preview',scene.frames===1 && !urls.length);
  await viewer.setScene(scene,urls,preserve);
  const bodyLabel=viewer.bodyMesh?'Body mesh':'Body skeleton';
  $('show-body').setAttribute('aria-label',bodyLabel);$('show-body').parentElement.title=bodyLabel;
  if(editing)viewer.setPlaying(false);
  $('scene-count').textContent=scene.cameras.length?`${scene.cameras.length} views${scene.frames>1?` · ${(scene.frames/scene.fps).toFixed(2)}s`:''}`:'';
  $('show-body').disabled=!scene.mesh && !scene.keypoints;
  $('show-body').checked=viewer.body.visible;
  $('show-videos').disabled=!urls.some(Boolean);
}
$('scene-play').onclick = () => viewer?.setPlaying(!viewer.playing);
$('timeline').oninput = event => { const frame=Number(event.target.value); viewer?.setPlaying(false); viewer?.seek(frame); };
$('scene-reset').onclick = () => viewer?.home();
$('scene-fullscreen').onclick = () => {if(document.fullscreenElement)document.exitFullscreen();else $('scene-card').requestFullscreen().catch(e=>message(e.message,true));};
for(const [id,group] of [['show-grid','grid'],['show-cameras','rig'],['show-body','body'],['show-videos','planes'],['show-4d','gaussians']]) {
  $(id).onchange = event => {if(viewer){viewer[group].visible=event.target.checked;viewer.requestRender();if(group==='body')viewer.bodyVisibilityChanged=true;}};
}
let trainingWatch=null,liveDirectory='',trainingPollBusy=false,loadedTraining=null;
function updateTraining(info) {
  trainingWatch=info?.status==='running'?info.watch:null;
  viewer?.setTraining(info);
  if(loadedTraining?.live)trainingNote(info);
}
function trainingNote(info) {
  if(!loadedTraining)return;
  $('scene-note').hidden=!loadedTraining.live;
  if(loadedTraining.live){
    const state=info?.status==='failed'||info?.status==='cancelled'?'Training stopped':'Training preview';
    $('scene-note').textContent=`${state} · step ${loadedTraining.preview_step.toLocaleString()}${state==='Training preview'?' · updates automatically':''}`;
  }
}
$('active-job').onchange=()=>{liveVersion='';autoOpenedBatch=undefined;followJob=true;explicitInitialResult=false;buttons();};
$('resume-live').onclick=()=>{followJob=true;explicitInitialResult=false;liveVersion='';autoOpenedBatch=undefined;pendingLayout=null;buttons();void poll();};
async function updateLive(report) {
  if(liveBusy || exporting || deletingJob || !followJob || report.status!=='running')return;
  liveBusy=true; buttons();
  const index=$('active-job').value||'0';
  try {
    const result=await api(`/api/jobs/preview?index=${index}&version=${encodeURIComponent(liveVersion)}`);
    if(result.scene && !exporting && followJob && index===($('active-job').value||'0')) {
      await showScene(result.scene,result.videos,Boolean(liveVersion));
      currentResult='';$('scene-title').textContent='Live preview';
      markGallerySelection();
      liveVersion=result.version; $('scene-note').hidden=false; $('scene-note').textContent=result.note.replace('SAM body ready','Body tracking ready').replace('generated cameras','views');
      message('');
    }
    if(followJob&&index===($('active-job').value||'0')) {
      if(result.directory)liveDirectory=result.directory;
      if(result.training)updateTraining(result.training);
    }
  } catch(e) { message(`Live preview: ${e.message}`,true); }
  finally {liveBusy=false; buttons(); flushLayoutPreview();}
}
async function exportingAction(action) {
  if (exporting || liveBusy || deletingJob) return;
  exporting = true; buttons();
  try { await action(); } catch(e) { message(e.message,true); }
  finally { exporting = false; buttons(); flushLayoutPreview(); }
}
function requestLayoutPreview() {
  workflow='prepare';
  const count=Number($('views').value), pitches=cameraPresets[count];
  // Preview depends only on camera geometry, not clip timing or upload state.
  pendingLayout={views:count/pitches.length,pitches,yaw:0,span:360,turbo:$('mode').value==='turbo'};
  followJob=false; liveVersion='';
  flushLayoutPreview();
}
function flushLayoutPreview() {
  if(!pendingLayout || exporting || liveBusy || deletingJob)return;
  const options=pendingLayout; pendingLayout=null;
  void exportingAction(async () => {
    message('Preparing camera layout…');
    const result=await api('/api/layout',options);
    if(pendingLayout)return; // A newer selection is queued; render only the latest.
    await showScene(result.scene);
    currentResult='';
    markGallerySelection();
    $('scene-title').textContent='Camera layout';$('scene-note').hidden=true;
    message('');
  });
}
$('settings').onsubmit = async event => {
  event.preventDefault(); if (submitting || running || uploading || !clipReady) return;
  submitting=true;buttons();
  try {
    const result=await api('/api/jobs',{videos:videos.map(v=>v.id),settings:settings(),clip:committedClip.edit});
    workflow='processing';workflowJob=result.directory;
    running = true; explicitInitialResult=false; liveVersion=''; followJob=true; buttons(); message(''); await poll();
  } catch(e) { message(e.message,true); }
  finally {submitting=false;buttons();}
};
$('cancel').onclick = async () => {
  if(stopping)return;stopping=true;buttons(); message('Stopping the batch…');
  try { const result = await api('/api/cancel',{}); message(result.message); await poll(); }
  catch(e) { message(e.message,true); }
  finally{stopping=false;buttons();}
};
const jobDate=timestamp=>new Intl.DateTimeFormat(undefined,{year:'numeric',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}).format(new Date(timestamp*1000));
const jobSize=bytes=>bytes>=1e9?`${(bytes/1e9).toFixed(1)} GB`:bytes>=1e6?`${(bytes/1e6).toFixed(1)} MB`:`${Math.ceil(bytes/1000)} KB`;
function markGallerySelection() {
  for(const card of $('result-gallery').querySelectorAll('.result-card')) {
    const job=completedJobs.find(j=>j.id===card.dataset.job);
    const active=job?.results.find(r=>r.directory===currentResult);
    if(active && card.dataset.directory!==currentResult) {
      galleryChunks.set(job.id,currentResult);
      card.setResult(active);
    }
    card.classList.toggle('selected',Boolean(active));
    const button=card.querySelector('.result-open');
    if(active)button.setAttribute('aria-current','true');else button.removeAttribute('aria-current');
  }
}
function renderGallery() {
  const query=$('result-search').value.trim().toLowerCase();
  $('result-search').hidden=completedJobs.length<6 && !query;
  $('result-count').textContent=completedJobs.length||'';
  $('result-gallery').replaceChildren();
  for(const job of completedJobs.filter(j=>`${j.name} ${jobDate(j.completed_at)}`.toLowerCase().includes(query))) {
    const card=document.createElement('article');card.className='result-card';card.dataset.job=job.id;
    const open=document.createElement('button');open.type='button';open.className='result-open';open.dataset.unavailable=String(!job.results.length);
    const preview=document.createElement('span');preview.className='result-preview';
    const placeholder=document.createElement('span');placeholder.className='result-placeholder';
    const image=document.createElement('img');image.alt='';image.loading='lazy';image.decoding='async';image.width=480;image.height=288;
    image.onload=()=>{placeholder.hidden=true;image.style.opacity='1';};
    image.onerror=()=>{image.style.opacity='0';placeholder.hidden=false;placeholder.textContent='Preview unavailable';};
    const badge=document.createElement('span');badge.className='result-badge';badge.textContent=`${job.cameras} views`;
    preview.append(placeholder,image,badge);
    const caption=document.createElement('span');caption.className='result-caption';
    const name=document.createElement('span');name.className='result-name';name.textContent=job.name;name.title=job.name;
    const date=document.createElement('time');date.className='result-date';date.dateTime=new Date(job.completed_at*1000).toISOString();date.title=jobDate(job.completed_at);
    date.textContent=new Intl.DateTimeFormat(undefined,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}).format(new Date(job.completed_at*1000));
    caption.append(name,date);open.append(preview,caption);card.append(open);
    const select=document.createElement('select');select.setAttribute('aria-label',`Chunk for ${job.name}`);
    job.results.forEach(result=>select.add(new Option(result.label,result.directory)));
    if(job.results.length>1){select.className='result-chunk';card.append(select);}
    card.setResult=result=>{
      badge.textContent=result?.training==='completed'?'4D scene':result?.training==='failed'||result?.training==='cancelled'?`${job.cameras} views · training stopped`:`${job.cameras} views`;
      card.dataset.directory=result?.directory||'';select.value=result?.directory||'';
      open.setAttribute('aria-label',`Open ${job.name}${job.results.length>1?`, ${result?.label}`:''}, ${jobDate(job.completed_at)}`);
      placeholder.hidden=false;placeholder.textContent=result?.thumbnail?'Loading preview…':'Preview unavailable';image.hidden=!result?.thumbnail;image.style.opacity='0';
      if(result?.thumbnail)image.src=result.thumbnail;else image.removeAttribute('src');
    };
    card.setResult(job.results.find(r=>r.directory===currentResult)||job.results.find(r=>r.directory===galleryChunks.get(job.id))||job.results[0]);
    open.onclick=()=>{
      if(exporting || liveBusy || deletingJob)return;
      followJob=false;pendingLayout=null;
      void exportingAction(async()=>{
        await openResult(card.dataset.directory);
        if(matchMedia('(max-width:760px)').matches)$('scene-card').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'auto':'smooth',block:'start'});
      });
    };
    select.onchange=()=>{galleryChunks.set(job.id,select.value);card.setResult(job.results.find(r=>r.directory===select.value));open.click();};
    const remove=document.createElement('button');remove.type='button';remove.className='result-delete';remove.title='Delete result';
    remove.setAttribute('aria-label',`Delete ${job.name}, ${jobDate(job.completed_at)}`);
    remove.innerHTML='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/></svg>';
    remove.onclick=()=>{
      deleteTarget=job;
      $('delete-job-name').textContent=`${job.name} · ${job.chunks} ${job.chunks===1?'chunk':'chunks'} · ${jobSize(job.bytes)}`;
      $('delete-job-error').hidden=true;$('delete-job-dialog').showModal();
    };
    card.append(remove);$('result-gallery').append(card);
  }
  if(!$('result-gallery').children.length){const empty=document.createElement('p');empty.className='gallery-empty';empty.textContent=query?'No matching results.':'Your generated scenes will appear here.';$('result-gallery').append(empty);}
  markGallerySelection();buttons();
}
$('result-search').oninput=renderGallery;
async function refreshResults() {
  if(refreshPromise)return refreshPromise;
  refreshPromise=(async()=>{
    const [paths,jobs]=await Promise.all([api('/api/results'),api('/api/jobs')]);
    resultPaths=paths;lastGalleryRefresh=Date.now();
    const signature=JSON.stringify(jobs);if(signature===gallerySignature)return;
    gallerySignature=signature;completedJobs=jobs;
    for(const id of galleryChunks.keys())if(!jobs.some(j=>j.id===id))galleryChunks.delete(id);
    renderGallery();
  })();
  try{await refreshPromise;}finally{refreshPromise=null;}
}
$('keep-job').onclick=()=>{if(!deletingJob)$('delete-job-dialog').close();};
$('delete-job-dialog').addEventListener('cancel',e=>{if(deletingJob)e.preventDefault();});
$('delete-job').onclick=async()=>{
  if(!deleteTarget || deletingJob || exporting || liveBusy)return;
  const job=deleteTarget, viewed=currentResult.startsWith(job.directory+'/'), keepLive=running && followJob;
  deletingJob=true;buttons();$('delete-job').disabled=true;$('keep-job').disabled=true;$('delete-job').textContent='Deleting…';
  let deleted=false;
  try {
    await api(`/api/jobs/${encodeURIComponent(job.id)}`,undefined,'DELETE');deleted=true;
    $('delete-job-dialog').close();
    if(viewed){
      if(!keepLive)followJob=false;
      currentResult='';liveVersion='';markGallerySelection();
      await showScene({cameras:[],frames:1,fps:25,keypoints:null,links:[],joint_ids:[],joint_colors:[]});
      $('scene-title').textContent='Your scene';
      $('scene-note').hidden=false;$('scene-note').textContent='Choose a result or add a video.';
    }
    await refreshPromise?.catch(()=>{});await refreshResults();await poll();
    message('Result deleted.');
  } catch(e){
    if(deleted)message(e.message,true);
    else{$('delete-job-error').textContent=e.message;$('delete-job-error').hidden=false;}
  } finally {
    deletingJob=false;buttons();$('keep-job').disabled=false;$('delete-job').textContent='Delete';flushLayoutPreview();
    if(deleted)$('result-gallery').querySelector('.result-open:not(:disabled)')?.focus({preventScroll:true});
  }
  if(deleted && viewed && !editing && !keepLive && resultPaths.length)await exportingAction(()=>openResult(resultPaths[0]));
};
function chooseCamera() { viewer?.selectCamera(Number($('camera').value)); }
$('back-to-results').onclick=async()=>{
  followJob=false;pendingLayout=null;workflow='results';buttons();
  await exportingAction(async()=>{
    const directory=resultPaths.includes(lastResult)?lastResult:resultPaths[0];
    if(directory)await openResult(directory);
    else {
      currentResult='';markGallerySelection();
      await showScene({cameras:[],frames:1,fps:25,keypoints:null,links:[],joint_ids:[],joint_colors:[]});
      $('scene-title').textContent='Your scene';$('scene-note').hidden=false;
      $('scene-note').textContent='Add a video to create your first scene.';
    }
  });
  ($('result-gallery').querySelector('.selected .result-open') || $('results-title')).focus();
};
async function openResult(directory) {
  if(editing)leaveClipEditor();
  if(!directory.trim())throw Error('Choose a saved result.');
  message('Loading result…');
  const result = await api('/api/results/open',{directory:directory.trim()});
  await showScene(result.scene,result.videos,directory===liveDirectory||directory===currentResult);
  updateTraining(result.training);
  currentResult=result.directory||directory.trim();
  lastResult=currentResult;
  markGallerySelection();
  const owner=completedJobs.find(j=>j.results.some(r=>r.directory===currentResult));
  const chunk=owner?.results.find(r=>r.directory===currentResult);
  $('scene-title').textContent=owner?`${owner.name}${owner.chunks>1?` · ${chunk.label}`:''}`:'Scene';
  $('scene-title').title=$('scene-title').textContent;
  $('scene-note').hidden=true; message('');
}
$('camera').onchange = chooseCamera;
let previousStatus, autoOpenedBatch, explicitInitialResult=false;
async function poll() {
  try {
    const {report,logs} = await api('/api/status');
    $('connection').hidden=true;running=report.status==='running';
    if(workflow==='processing' && report.directory===workflowJob && !running)workflow='results';
    renderBatch(report);
    $('logs').textContent=logs||'';$('logs').hidden=!logs;
    const jobIndex=$('active-job').value;
    const options=report.jobs.map((j,i)=>new Option(`${j.label||`Chunk ${i+1}`} · ${j.status}`,String(i)));
    // Avoid replacing an open native select on every poll.
    const signature=options.map(o=>o.textContent).join('|');
    if($('active-job').dataset.options!==signature){
      $('active-job').replaceChildren(...options);$('active-job').dataset.options=signature;
      if(Number(jobIndex)<report.jobs.length)$('active-job').value=jobIndex;
    }
    $('active-job').hidden=report.jobs.length<2;
    const statusKey=`${report.directory}:${report.status}:${report.jobs.filter(j=>j.status==='completed').length}`;
    if(previousStatus!==statusKey || Date.now()-lastGalleryRefresh>15000)await refreshResults();
    previousStatus = statusKey; buttons();
    const selectedJob=Number($('active-job').value)||0;
    const resultKey=`${report.directory}:${selectedJob}`;
    if (autoOpenedBatch !== resultKey && !exporting && !liveBusy && !deletingJob && followJob && !explicitInitialResult && !editing) {
      const job = report.jobs[selectedJob];
      if (job?.status==='completed' && job.result_dir) {
        autoOpenedBatch = resultKey;
        await exportingAction(()=>openResult(job.result_dir));
      }
    }
    void updateLive(report);
    // Also follow training started from the CLI on an existing gallery result.
    if(trainingWatch&&!trainingPollBusy&&(report.status!=='running'||!followJob)) {
      const watch=trainingWatch;
      trainingPollBusy=true;
      void api(watch).then(info=>{if(trainingWatch===watch)updateTraining(info);})
        .catch(()=>{}).finally(()=>{trainingPollBusy=false;});
    }
  } catch(e) { $('connection').textContent='Reconnecting…';$('connection').hidden=false; }
}
async function initialize() {
  buttons();
  viewerReady = (async () => {
    const {SceneViewer}=await import('/static/scene.js');
    viewer=new SceneViewer($('viewer'),{
      onTime: state => {
        $('scene-play').disabled=state.frames<2; $('scene-play').textContent=state.playing?'Pause':'Play';
        $('timeline').disabled=state.frames<2; $('timeline').max=state.frames-1; $('timeline').value=state.frame;
        const stamp=t=>`${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`;
        $('scene-time').textContent=`${stamp(state.time)} / ${stamp(state.frames/state.fps)}`;
        $('timeline').title=`Frame ${state.frame+1} of ${state.frames}`;
      },
      onSelect: (index,video,url) => {
        $('camera').value=String(index); $('playback').replaceChildren();
        if(video)$('playback').append(video);else $('playback').textContent=viewer?.frames===1?'Camera layout preview':'This camera video is not ready yet.';
        $('download').hidden=!url;if(url)$('download').href=url;
      },
      onError: text => message(text,true),
      onTraining: state => {
        if(state.cleared){
          loadedTraining=null;$('show-4d-label').hidden=true;$('show-4d').disabled=true;$('show-4d').checked=true;$('download-model').hidden=true;
        } else if(state.loaded&&!state.error){
          loadedTraining=state.info;
          $('show-4d-label').hidden=false;$('show-4d').disabled=false;
          $('show-body').checked=viewer.body.visible;
          $('download-model').hidden=false;$('download-model').href=state.info.model;
          trainingNote(state.info);message('');
        } else if(state.loading&&!loadedTraining){
          $('scene-note').hidden=false;$('scene-note').textContent='Loading 4D scene…';
        } else if(state.error){
          $('scene-note').hidden=false;$('scene-note').textContent=loadedTraining?'Preview update unavailable; keeping the last model.':state.error;
        }
      }
    });
  })();
  viewerReady.catch(e=>{ message(`3D viewer could not start: ${e.message}`,true); });
  try {
    const config = await api('/api/config'); videos=config.videos.slice(0,1); showUploads();
    if(!config.training_available){$('output-kind').value='videos';$('output-kind').options[0].disabled=true;$('output-kind').options[0].textContent='4D scene (setup required)';}
    explicitInitialResult=Boolean(config.output_dir);
    await poll();
    if(config.output_dir) await exportingAction(()=>openResult(config.output_dir));
    else if(!running && !viewer?.data && resultPaths.length) await exportingAction(()=>openResult(resultPaths[0]));
    if(videos.length)await loadClip();
  } catch(e) { message(e.message,true); }
  const tick = async () => { await poll(); setTimeout(tick,2000); }; setTimeout(tick,2000);
}
window.addEventListener('beforeunload',()=>{filmstripAbort?.abort();cancelAnimationFrame(sourceAnimation);viewer?.dispose();});
initialize();
