// Adapted from FreeTimeGsVanilla/player/renderer.js at a3b2292 (AGPL-3.0).
// Three.js supplies attributes and GLSL version; projected depth lets the body,
// grid and camera rig share the scene depth buffer. Retains the upstream FTGS math.
export const VERTEX = `
precision highp float;
precision highp int;
precision highp sampler2D;
in vec2 corner;
in uint splatId;
uniform sampler2D positions, velocities, covarianceA, covarianceB, harmonics;
uniform mat4 view;
uniform vec3 eye;
uniform vec2 viewport;
uniform float time, focal, nearPlane, farPlane, opacityFloor;
uniform int coefficients, shDegree;
uniform bool useVelocity;
out vec2 gaussian;
flat out vec4 color;
vec4 load(sampler2D tex, int i) {
  int width = textureSize(tex,0).x;
  return texelFetch(tex,ivec2(i%width,i/width),0);
}
vec3 shColor(int index, vec3 d) {
  float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z;
  float b[16];
  b[0]=0.28209479177387814;
  b[1]=-0.4886025119029199*y; b[2]=0.4886025119029199*z; b[3]=-0.4886025119029199*x;
  b[4]=1.0925484305920792*x*y; b[5]=-1.0925484305920792*y*z;
  b[6]=0.31539156525252005*(2.0*zz-xx-yy); b[7]=-1.0925484305920792*x*z;
  b[8]=0.5462742152960396*(xx-yy);
  b[9]=-0.5900435899266435*y*(3.0*xx-yy); b[10]=2.890611442640554*x*y*z;
  b[11]=-0.4570457994644658*y*(4.0*zz-xx-yy);
  b[12]=0.3731763325901154*z*(2.0*zz-3.0*xx-3.0*yy);
  b[13]=-0.4570457994644658*x*(4.0*zz-xx-yy);
  b[14]=1.445305721320277*z*(xx-yy); b[15]=-0.5900435899266435*x*(xx-3.0*yy);
  vec3 rgb=vec3(0.5);
  int bandCount=(shDegree+1)*(shDegree+1);
  for(int i=0;i<16;i++) {
    if(i>=bandCount) break;
    rgb += b[i]*load(harmonics,index*coefficients+i).xyz;
  }
  return max(rgb,vec3(0.0));
}
void main() {
  int id=int(splatId);
  vec4 p=load(positions,id), v=load(velocities,id), a=load(covarianceA,id), b=load(covarianceB,id);
  float dt=time-p.w;
  vec3 world=p.xyz+(useVelocity ? v.xyz*dt : vec3(0.0));
  vec3 center=(view*vec4(world,1.0)).xyz;
  float depth=-center.z;
  float opacity=max(opacityFloor,b.z*exp(-0.5*(dt/v.w)*(dt/v.w)));
  gaussian=corner; color=vec4(0.0);
  if(depth<=nearPlane || opacity<1.0/255.0) { gl_Position=vec4(2.0,2.0,0.0,1.0); return; }
  mat3 C=mat3(a.x,a.y,a.z, a.y,a.w,b.x, a.z,b.x,b.y);
  vec3 right=vec3(view[0][0],view[1][0],view[2][0]);
  vec3 up=vec3(view[0][1],view[1][1],view[2][1]);
  vec3 back=vec3(view[0][2],view[1][2],view[2][2]);
  // Bound the covariance Jacobian outside the frustum, as in the CUDA rasterizer.
  // Without this, near-camera offscreen centers produce screen-filling ellipses.
  vec2 limit=1.3*viewport/(2.0*focal);
  vec2 slope=clamp(center.xy/depth,-limit,limit);
  vec3 jx=(focal/depth)*(right+slope.x*back);
  vec3 jy=(focal/depth)*(up+slope.y*back);
  float aa=dot(jx,C*jx)+0.3, ab=dot(jx,C*jy), bb=dot(jy,C*jy)+0.3;
  float mid=0.5*(aa+bb), delta=length(vec2(0.5*(aa-bb),ab));
  float l1=max(mid+delta,0.1), l2=max(mid-delta,0.1);
  vec2 axis=abs(ab)>0.000001 ? normalize(vec2(ab,l1-aa)) : (aa>=bb ? vec2(1,0) : vec2(0,1));
  vec2 offset=corner.x*sqrt(l1)*axis+corner.y*sqrt(l2)*vec2(-axis.y,axis.x);
  vec2 ndc=(focal*center.xy/depth+offset)*2.0/viewport;
  float clipDepth=(farPlane+nearPlane)/(farPlane-nearPlane)-2.0*farPlane*nearPlane/((farPlane-nearPlane)*depth);
  gl_Position=vec4(ndc,clipDepth,1.0);
  vec3 direction=world-eye;
  direction /= max(length(direction),0.000001);
  color=vec4(shColor(id,direction),opacity);
}`;

export const FRAGMENT = `
precision highp float;
in vec2 gaussian;
flat in vec4 color;
out vec4 fragColor;
void main() {
  float radius=dot(gaussian,gaussian);
  if(radius>9.0) discard;
  float alpha=min(0.99,color.a*exp(-0.5*radius));
  if(alpha<1.0/255.0) discard;
  fragColor=vec4(color.rgb*alpha,alpha);
}`;
