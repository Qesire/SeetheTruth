'use strict';
const assert=require('node:assert/strict');
const V=require('../video_causal_conv3d.js');
const L=require('../learning_lab.js'),C=require('../concept_flow.js'),R=require('../graph_readiness.js'),K=require('../block_knowledge.js');
const p={frames:4,height:2,width:2,channels:1,kernelFrames:2,kernelHeight:2,kernelWidth:1,stride:1,outChannels:1};
const x=Array.from({length:8},(_,i)=>[2*i+1,2*i+2]);
const w=[[1,0,0,0]];
assert.deepEqual(V.shape({x:[8,2],kernel:[1,4]},p),[4,2]);
assert.deepEqual(V.execute(x,w,p),[[0,0],[1,2],[5,6],[9,10]]);
const a=V.makeNativeVideo(x,p),b=V.makeNativeKernel(w,p),n=V.nativeConv3D(a,b,p);
assert.deepEqual(a.shape,[1,4,2,2,1]);assert.deepEqual(a.axes,["B","T","H","W","C"]);
assert.deepEqual(b.shape,[1,2,2,1,1]);assert.deepEqual(b.axes,["O","Kt","Kh","Kw","I"]);
assert.deepEqual(n.shape,[1,1,4,1,2]);assert.deepEqual(n.axes,["B","O","T","H","W"]);
assert.deepEqual(V.toCarrier(n,p),V.execute(x,w,p));
function chunks(sizes){
 let state=V.initialize(p,w),output=[],at=0;
 for(const size of sizes){
  const part=x.slice(at*p.height,(at+size)*p.height);
  const before=JSON.stringify(state);
  const r=V.pushChunk(state,part);
  assert.equal(JSON.stringify(state),before,'stream state must remain immutable');
  output.push(...r.output);state=r.state;at+=size;
 }
 assert.equal(state.processed,p.frames);
 assert.equal(state.history.length,(p.kernelFrames-1)*p.height);
 assert.deepEqual(output,V.execute(x,w,p));
}
chunks([1,1,1,1]);chunks([1,2,1]);chunks([2,2]);chunks([4]);
const changed=x.map(row=>row.slice());changed[6][0]=100000;changed[7][1]=-100000;
assert.deepEqual(V.execute(changed,w,p).slice(0,3),V.execute(x,w,p).slice(0,3),'future cannot change earlier outputs');
assert.notDeepEqual(V.execute(changed,[[0,0,1,0]],p).at(-1),V.execute(x,[[0,0,1,0]],p).at(-1));
const one={...p,kernelFrames:1};
let st=V.initialize(one,[[1,0]]);st=V.pushChunk(st,x.slice(0,2)).state;
assert.equal(st.history.length,0,'1-frame temporal conv has no cache');
assert.throws(()=>V.shape({x:[8,2],kernel:[1,3]},p),/kernel/);
assert.throws(()=>V.pushChunk(V.initialize(p,w),x.slice(0,1)),/complete THWC frames/);
assert.throws(()=>V.pushChunk(V.initialize(p,w),x.concat(x)),/exceeds declared total frames/);
assert.ok(K.get("video_causal_conv3d")?.refs.length,'source-grounded Wan VAE knowledge card');
const graph=L.preset("video_causal_conv3d"),e=L.evaluate(graph),c=C.evaluate(graph),r=R.check(graph);
assert.equal(r.executable,true);assert.equal(e.steps.length,graph.nodes.length);
for(const step of e.steps)assert.deepEqual(step.shape,c.steps.find(z=>z.id===step.id).shape);
assert.ok(e.steps.find(z=>z.id==="causal_conv").values.flat().every(Number.isFinite));
console.log('VIDEO_CAUSAL_CONV3D_CONTRACTS_PASS rank5 axes, strict causality, immutable streaming chunks, graph shapes');
