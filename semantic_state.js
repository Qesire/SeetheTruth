/* SeetheTruth explicit axis and immutable streaming-state contracts.
 * Small exact Float64 semantics, not an aliasing GPU view / KV paging implementation.
 * Each operation is an actual calculation on TensorIR.data (rank 1..5).
 */
(function(root,factory){"use strict";
 const api=factory(typeof module==="object"&&module.exports?require("./tensor_ir.js"):root.SeetheTruthTensorIR);
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.SeetheTruthSemanticState=api;
})(typeof window!=="undefined"?window:null,function(IR){"use strict";
 if(!IR)throw Error("SemanticState requires TensorIR");
 const fail=m=>{throw Error("SemanticState: "+m);};
 const int=(n,lo,hi,name)=>{if(!Number.isSafeInteger(n)||n<lo||n>hi)fail(name+" out of bounds");return n;};
 const prod=sh=>sh.reduce((p,d)=>p*d,1);
 const axesValid=axes=>Array.isArray(axes)&&axes.length>=1&&axes.length<=5&&
   new Set(axes).size===axes.length&&axes.every(x=>typeof x==="string"&&/^[A-Za-z][A-Za-z0-9_]{0,15}$/.test(x));
 function verifyTensor(t){
  if(!t||t.dtype!=="float64")fail("expected Float64 semantic tensor");
  // IR.make validates numerical payload, rank, dimensions, axis names and size.
  return IR.make(t.shape,t.axes,t.data);
 }
 function each(shape,fn){
  const i=Array(shape.length).fill(0);
  function loop(k){if(k===shape.length){fn(i.slice());return;}
   for(let j=0;j<shape[k];j++){i[k]=j;loop(k+1);}}
  loop(0);
 }
 function reorder(t,axes){
  t=verifyTensor(t);
  if(!axesValid(axes)||axes.length!==t.axes.length||axes.some(a=>!t.axes.includes(a)))
   fail("reorder requires the exact named-axis permutation");
  return IR.transpose(t,axes.map(a=>t.axes.indexOf(a)));
 }
 function align(t,axes){
  t=verifyTensor(t);
  if(!axesValid(axes)||t.axes.some(a=>!axes.includes(a)))
   fail("align must preserve every original named axis; renaming is not implicit");
  const shape=axes.map(a=>t.axes.includes(a)?t.shape[t.axes.indexOf(a)]:1);
  const data=[];
  each(shape,i=>data.push(IR.get(t,t.axes.map(a=>i[axes.indexOf(a)]))));
  return IR.make(shape,axes,data);
 }
 function broadcast(t,axes,shape){
  const a=align(t,axes);
  if(!Array.isArray(shape)||shape.length!==axes.length)fail("broadcast shape/axes length mismatch");
  for(let i=0;i<shape.length;i++){
   int(shape[i],1,32,"broadcast dimension");
   if(a.shape[i]!==shape[i]&&a.shape[i]!==1)fail("broadcast requires matching or singleton dimension on "+axes[i]);
  }
  if(prod(shape)>IR.MAX_ELEMENTS)fail("broadcast exceeds bounded element limit");
  const data=[];
  each(shape,i=>data.push(IR.get(a,i.map((v,j)=>a.shape[j]===1?0:v))));
  return IR.make(shape,axes,data);
 }
 function binary(a,b,operation,axes){
  if(!["add","subtract","multiply"].includes(operation))fail("binary operation must be add/subtract/multiply");
  a=verifyTensor(a);b=verifyTensor(b);
  const set=new Set([...a.axes,...b.axes]);
  if(!axesValid(axes)||axes.length!==set.size||axes.some(x=>!set.has(x)))
   fail("binary axes must enumerate exact named-axis union; no silent axis reinterpretation");
  const aa=align(a,axes),bb=align(b,axes);
  const shape=axes.map((ax,i)=>{
   const x=aa.shape[i],y=bb.shape[i];
   if(x!==y&&x!==1&&y!==1)fail("named-axis mismatch for "+ax+": "+x+" vs "+y);
   return Math.max(x,y);
  });
  if(prod(shape)>IR.MAX_ELEMENTS)fail("binary result exceeds bounded element limit");
  const data=[];
  each(shape,index=>{
   const x=IR.get(aa,index.map((v,j)=>aa.shape[j]===1?0:v));
   const y=IR.get(bb,index.map((v,j)=>bb.shape[j]===1?0:v));
   data.push(operation==="add"?x+y:operation==="subtract"?x-y:x*y);
  });
  return IR.make(shape,axes,data);
 }
 function slice(t,axis,start,length){
  t=verifyTensor(t);const idx=t.axes.indexOf(axis);
  if(idx<0)fail("unknown slice axis "+axis);
  int(start,0,t.shape[idx]-1,"slice start");
  int(length,1,t.shape[idx],"slice length");
  if(start+length>t.shape[idx])fail("slice extends outside axis "+axis);
  const shape=t.shape.slice();shape[idx]=length;const out=[];
  each(shape,index=>{index[idx]+=start;out.push(IR.get(t,index));});
  return IR.make(shape,t.axes,out);
 }
 function concat(tensors,axis){
  if(!Array.isArray(tensors)||tensors.length<2||tensors.length>8)fail("concat needs 2..8 tensors");
  const values=tensors.map(verifyTensor),head=values[0],where=head.axes.indexOf(axis);
  if(where<0)fail("unknown concatenation axis "+axis);
  for(const t of values.slice(1)){
   if(t.axes.join("|")!==head.axes.join("|"))fail("concat requires identical named-axis order");
   for(let k=0;k<t.shape.length;k++)if(k!==where&&t.shape[k]!==head.shape[k])
    fail("concat non-time axis mismatch at "+head.axes[k]);
  }
  const shape=head.shape.slice();shape[where]=values.reduce((s,x)=>s+x.shape[where],0);
  if(shape[where]>32||prod(shape)>IR.MAX_ELEMENTS)fail("concat exceeds bounded shape/capacity");
  const out=[];
  each(shape,index=>{
   let cursor=index[where],chosen;
   for(const t of values){if(cursor<t.shape[where]){chosen=t;break;}cursor-=t.shape[where];}
   const source=index.slice();source[where]=cursor;out.push(IR.get(chosen,source));
  });
  return IR.make(shape,head.axes,out);
 }
 function freezeTensor(t){
  const x=verifyTensor(t);
  Object.freeze(x.shape);Object.freeze(x.axes);Object.freeze(x.data);
  return Object.freeze(x);
 }
 function freezeState(t,axis,origin,capacity,version){
  t=freezeTensor(t);
  const axisIdx=t.axes.indexOf(axis);
  if(axisIdx<0)fail("state axis absent");
  const nextPosition=origin+t.shape[axisIdx];
  return Object.freeze({schema:"seethetruth-semantic-stream-v1",tensor:t,axis,
    origin,capacity,version,nextPosition,storage:"immutable_materialized_cpu",
    physicalCacheParity:"not_claimed"});
 }
 function validateState(s){
  if(!s||s.schema!=="seethetruth-semantic-stream-v1")fail("not a semantic stream state");
  const t=verifyTensor(s.tensor),axisIdx=t.axes.indexOf(s.axis);
  if(axisIdx<0)fail("state axis absent");
  int(s.origin,0,4096,"state origin");int(s.capacity,1,32,"state capacity");
  int(s.version,0,4096,"state version");
  if(t.shape[axisIdx]>s.capacity)fail("state exceeds capacity");
  if(s.nextPosition!==s.origin+t.shape[axisIdx])fail("corrupted next position");
  if(s.storage!=="immutable_materialized_cpu"||s.physicalCacheParity!=="not_claimed")
   fail("unexpected storage/evidence contract");
  return t;
 }
 function init(t,{axis="T",origin=0,capacity=32}={}){
  t=verifyTensor(t);int(origin,0,4096,"state origin");int(capacity,1,32,"stream capacity");
  const where=t.axes.indexOf(axis);
  if(where<0||t.shape[where]>capacity)fail("initial tensor must fit the declared stream axis/capacity");
  return freezeState(t,axis,origin,capacity,0);
 }
 function append(s,chunk,{position}={}){
  const t=validateState(s),next=verifyTensor(chunk);
  if(position===undefined)fail("append requires explicit absolute position");
  int(position,0,4096,"append position");
  if(position!==s.nextPosition)fail("out-of-order/stale append position");
  const at=s.axis,axisIdx=t.axes.indexOf(at);
  const resultingSize=t.shape[axisIdx]+next.shape[axisIdx];
  if(resultingSize>s.capacity)fail("append exceeds state capacity");
  return freezeState(concat([t,next],at),at,s.origin,s.capacity,s.version+1);
 }
 function read(s,start,length){
  validateState(s);int(start,s.origin,s.nextPosition-1,"read position");
  int(length,1,s.nextPosition-s.origin,"read length");
  if(start+length>s.nextPosition)fail("read beyond populated prefix");
  return Object.freeze({tensor:slice(s.tensor,s.axis,start-s.origin,length),
    logicalStart:start,logicalEnd:start+length,sourceVersion:s.version,
    physicalCacheParity:"not_claimed"});
 }
 function fork(s){
  const t=validateState(s);
  return freezeState(t,s.axis,s.origin,s.capacity,s.version);
 }
 return Object.freeze({verifyTensor,reorder,align,broadcast,binary,slice,concat,
  init,append,read,fork,validateState});
});
