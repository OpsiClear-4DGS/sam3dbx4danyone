import {test} from 'node:test';
import assert from 'node:assert/strict';
import {columnsFromPLY} from '../fdanyone/reconstruction/export_tsog.mjs';

function fixture({degree=1,moving=true,frames=121,value=1}={}) {
  const rest=(degree+1)**2-1;
  const values={x:1,y:2,z:3,opacity:-1,time:.75,log_duration:Math.log(.2),
    rot_0:1,rot_1:0,rot_2:0,rot_3:0};
  for(let i=0;i<3;i++){
    values[`f_dc_${i}`]=value;values[`scale_${i}`]=-2;values[`velocity_${i}`]=i+1;
  }
  for(let i=0;i<rest*3;i++)values[`f_rest_${i}`]=i+1;
  const fields=Object.keys(values);
  const header=['ply','format binary_little_endian 1.0','comment ftgs_version 1',
    'comment time_units normalized',`comment sh_degree ${degree}`,`comment use_velocity ${+moving}`,
    'comment min_duration 0.001','comment opacity_floor 0.0001',`comment n_frames ${frames}`,
    'element vertex 1',...fields.map(name=>'property float '+name),'end_header',''].join('\n');
  const row=Buffer.alloc(fields.length*4);
  fields.forEach((name,i)=>row.writeFloatLE(values[name],i*4));
  return Buffer.concat([Buffer.from(header),row]);
}

test('Vanilla temporal units and channel-major lower SH bands map to TSOG',()=>{
  const {header,columns}=columnsFromPLY(fixture());
  const values=Object.fromEntries(columns.map(({name,values})=>[name,values[0]]));
  assert.equal(header.nFrames,121);assert.equal(values.t,.75);
  assert.ok(Math.abs(values.t_scale-.2)<1e-7);
  assert.equal(values.motion_0,1);assert.equal(values.motion_2,3);
  for(let channel=0;channel<3;channel++)for(let k=0;k<15;k++)
    assert.equal(values[`f_rest_${channel*15+k}`],k<3?channel*3+k+1:0);
});

test('disabled motion stays disabled and truncated/nonfinite models fail',()=>{
  const {columns}=columnsFromPLY(fixture({degree:0,moving:false}));
  for(const {name,values} of columns)
    if(name.startsWith('motion_')||name.startsWith('f_rest_'))assert.equal(values[0],0);
  assert.throws(()=>columnsFromPLY(fixture().subarray(0,-4)),/file size/);
  assert.throws(()=>columnsFromPLY(fixture({value:NaN})),/Non-finite/);
  assert.throws(()=>columnsFromPLY(fixture({frames:1})),/frame count/);
});
