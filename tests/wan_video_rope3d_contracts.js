'use strict';
// Independent Wan2.2 video RoPE oracle; original source revision:
// Wan-Video/Wan2.2@1ea34ff48f87168174e12956e200b1d908b1c5ff
// wan/modules/model.py lines 27-66, 126-154.
// This oracle does not call the implementation's metadata helper.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const ND=require('../nd_tensor.js'),L=require('../learning_lab.js'),
 E=require('../element_lineage.js'),A=require('../nd_autodiff.js'),
 M=require('../compute_microscope.js'),Bridge=require('../coordinate_vjp_bridge.js');
const eps=1e-6,near=(a,b,e=3e-6,where='')=>assert.ok(
 Math.abs(a-b)<=e,where+': '+a+' != '+b);
const independent=(x,F,H,W,base)=>{
 const shape=ND.shape(x),complex=shape[3]/2,part=Math.floor(complex/3),
  split=[complex-2*part,part,part];
 return ND.build(shape,idx=>{
  const [batch,head,t,d]=idx,pair=Math.floor(d/2);
  const f=Math.floor(t/(H*W)),h=Math.floor(t/W)%H,w=t%W;
  const coord=pair<split[0]?f:pair<split[0]+split[1]?h:w;
  // Equivalent to complex multiplication in rope_apply: global-frequency
  // channel offset is NOT reset at the F/H/W boundary.
  const angle=coord*Math.pow(base,-2*pair/shape[3]);
  const a=ND.at(x,[batch,head,t,2*pair]),b=ND.at(x,[batch,head,t,2*pair+1]);
  return d%2===0?a*Math.cos(angle)-b*Math.sin(angle):
   a*Math.sin(angle)+b*Math.cos(angle);
 });
};
const srcTensor=ND.seedTensor([1,2,8,6],7);
for(const cfg of [[2,2,2,10000],[1,2,4,5000],[2,1,4,1000]]){
 const [F,H,W,base]=cfg,actual=ND.wanVideoRoPE3D4(srcTensor,F,H,W,base),
  ref=independent(srcTensor,F,H,W,base);
 ND.indices(ND.shape(actual),idx=>near(ND.at(actual,idx),ND.at(ref,idx),1e-12,'Wan source '+cfg+'/'+idx));
 // Preserve orthonormality for each complex pair at all grid positions.
 for(let t=0;t<8;t++)for(let h=0;h<2;h++)for(let p=0;p<3;p++){
  const x=srcTensor[0][h][t],out=actual[0][h][t];
  near(x[2*p]**2+x[2*p+1]**2,out[2*p]**2+out[2*p+1]**2,1e-12,'rotation norm');
 }
 const upstream=ND.seedTensor([1,2,8,6],11),g=A.vjp('nd_wan_rope3d',
  {x:srcTensor},{frames:F,height:H,width:W,base},upstream,actual).x;
 ND.indices(ND.shape(srcTensor),idx=>{
  const j=idx[3],meta=ND.wanVideoRoPE3DCoordinate(ND.shape(srcTensor),idx,F,H,W,base),
   partner=[...idx.slice(0,3),meta.partner],d=idx[3]%2===0?
   meta.cos*ND.at(upstream,idx)+meta.sin*ND.at(upstream,partner):
   meta.cos*ND.at(upstream,idx)-meta.sin*ND.at(upstream,partner);
  near(ND.at(g,idx),d,1e-12,'local VJP');
  const bump=sign=>ND.build(ND.shape(srcTensor),coords=>ND.at(srcTensor,coords)+
   (coords.every((v,i)=>v===idx[i])?sign*eps:0));
  const weighted=trial=>ND.flatten(independent(trial,F,H,W,base))
   .reduce((sum,v,i)=>sum+v*ND.flatten(upstream)[i],0);
  near(ND.at(g,idx),(weighted(bump(1))-weighted(bump(-1)))/(2*eps),
   2e-9,'finite difference '+cfg+'/'+idx);
 });
}
// F, H and W are not interchangeable positions. With D=6 the global
// complex pairs 0,1,2 encode temporal, vertical and horizontal separately.
const origin=ND.wanVideoRoPE3DCoordinate([1,2,8,6],[0,0,0,0],2,2,2,10000);
assert.deepEqual([origin.temporalPairs,origin.heightPairs,origin.widthPairs],[1,1,1]);
for(const [d,axis,pos] of [[0,'F',1],[2,'H',1],[4,'W',1]]){
 const t=d===0?4:d===2?2:1,m=ND.wanVideoRoPE3DCoordinate(
  [1,2,8,6],[0,0,t,d],2,2,2,10000);
 assert.equal(m.axis,axis);assert.equal(m.position,pos);
 near(m.angle,Math.pow(10000,-d/6),1e-12,'global complex-pair exponent');
}
// A wrong reset-per-axis frequency and a one-dimensional token index give
// different answers at a frame/height/width decision boundary.
const wrong=ND.rope4(srcTensor,0,6,10000);
assert.notDeepEqual(wrong,ND.wanVideoRoPE3D4(srcTensor,2,2,2,10000));
const graph=L.preset('nd_wan_video_rope_attention');
const run=L.evaluate(graph),ref=run.steps.find(s=>s.id==='context').values;
assert.deepEqual(ND.shape(ref),[1,2,8,6]);
const byId=id=>run.steps.find(s=>s.id===id).values;
for(const [id,field] of [['q_video_rope','query'],['k_video_rope','key']])
 assert.deepEqual(byId(id),independent(byId(field),2,2,2,10000));
// The FULL attention output is independently recomputed without using the
// native Wan rotary implementation: QK^T / √D -> row Softmax -> PV.
const kTranspose=ND.permute(independent(byId('key'),2,2,2,10000),[0,1,3,2]),
 scores=ND.scale(ND.matmul4(independent(byId('query'),2,2,2,10000),kTranspose),1/Math.sqrt(6)),
 recomputed=ND.matmul4(ND.softmaxLast(scores),byId('value'));
 ND.indices([1,2,8,6],idx=>near(ND.at(ref,idx),ND.at(recomputed,idx),1e-11,'whole graph forward'));
const numerical=M.learning(graph,run,L);
for(const [target,index,axis] of [
 ['q_video_rope',[0,0,4,0],'F'],['q_video_rope',[0,0,2,2],'H'],
 ['q_video_rope',[0,0,1,4],'W']]){
 const proof=E.inspect(graph,{targetId:target,outputIndex:index,trace:run});
 assert.equal(proof.complete,true);
 const v=proof.vertices.find(v=>v.id===target);
 assert.equal(v.stages[0].stage,'wan_video_3d_rope');
 assert.equal(v.stages[0].axis,axis);
 const report=Bridge.inspect(graph,run,numerical,{stepId:target,coordinate:index});
 assert.equal(report.backward.complete,true);
 assert.equal(report.backward.internalStages[0].axis,axis);
 assert.equal(report.loss.status,'NO_LOSS_OBJECTIVE');
}
const scalar=E.inspect(graph,{targetId:'context',outputIndex:[0,0,5,0],
 trace:run,maxVertices:4096,maxEdges:16384});
assert.equal(scalar.complete,true);
assert.ok(scalar.vertices.some(v=>v.id==='query'&&v.status==='source'));
assert.ok(scalar.vertices.some(v=>v.id==='key'&&v.status==='source'));
assert.ok(scalar.vertices.some(v=>v.id==='value'&&v.status==='source'));
const h=1e-6;
for(const v of scalar.vertices.filter(v=>v.status==='source').slice(0,14)){
 // Move each source input to a trainable parameter and perturb the actual
 // graph re-evaluation, independent of the scalar VJP implementation.
 const raw=structuredClone(graph),n=raw.nodes.find(n=>n.id===v.id);
 const original=L.evaluate(graph).steps.find(s=>s.id===v.id).values;
 n.kind='nd_parameter';n.params={values:original,
  optimizer:{t:0,m:ND.build(ND.shape(original),()=>0),v:ND.build(ND.shape(original),()=>0)}};
 const sample=sign=>{
  const g=structuredClone(raw),p=g.nodes.find(x=>x.id===v.id).params.values;
  let cur=p;for(const i of v.index.slice(0,-1))cur=cur[i];
  cur[v.index.at(-1)]+=sign*h;
  const trace=L.evaluate(g);
  return ND.at(trace.steps.find(s=>s.id==='context').values,[0,0,5,0]);
 };
 near(v.sensitivity,(sample(1)-sample(-1))/(2*h),1e-5,'full attention FD '+v.key);
}
for(const p of [
 {frames:2,height:2,width:1,base:10000},{frames:2,height:2,width:3,base:10000},
 {frames:2,height:2,width:2,base:1.2}]){
 assert.throws(()=>ND.wanVideoRoPE3D4(srcTensor,p.frames,p.height,p.width,p.base),
 /Wan video RoPE/);
}
assert.throws(()=>ND.wanVideoRoPE3D4(ND.seedTensor([1,2,8,4],3),2,2,2),
 /Wan video RoPE/);
const stale=structuredClone(run);stale.steps.find(s=>s.id==='q_video_rope').values[0][0][0][0]+=.4;
assert.throws(()=>E.inspect(graph,{targetId:'context',outputIndex:[0,0,0,0],trace:stale}),
 /stale or altered/);
const html=fs.readFileSync(path.join(__dirname,'../learn.html'),'utf8');
assert.ok(html.includes('data-preset="nd_wan_video_rope_attention"'));
assert.ok(html.includes('"nd_wan_video_rope_attention"'));
assert.ok(html.includes('"nd_wan_qkv_fullwidth_attention"'));
console.log('WAN_VIDEO_ROPE3D_PASS: source-pinned F/H/W complex-phase, 3 head sizes, independent finite differences, full video attention scalar Jacobians and negative tests');
