/* Small, explicit CPU N-dimensional tensor kernel for freely composed teaching DAGs.
 * Nested row-major arrays; rank 2–4; no implicit axis-name semantics.
 * Shape-product bounds prevent accidental large numerical execution.
 */
(function(root,factory){const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthND=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
const MAX_ELEMENTS=512,MAX_RANK=4,MAX_DIM=8;
const product=a=>a.reduce((s,x)=>s*x,1);
const fail=s=>{throw Error(s);};
function dims(shape){
 if(!Array.isArray(shape)||shape.length<2||shape.length>MAX_RANK||
    shape.some(d=>!Number.isSafeInteger(d)||d<1||d>MAX_DIM))fail("N-D shape requires rank 2–4, each axis 1–8");
 if(product(shape)>MAX_ELEMENTS)fail("N-D tensor exceeds bounded 512-element CPU allowance");
 return shape.slice();
}
function shape(x){
 const result=[];
 let a=x;
 while(Array.isArray(a)){
  if(!a.length)fail("N-D tensor has an empty axis");
  result.push(a.length);a=a[0];
  if(result.length>MAX_RANK)fail("N-D tensor rank exceeds 4");
 }
 dims(result);const check=(v,level)=>{
  if(level===result.length){if(typeof v!=="number"||!Number.isFinite(v))fail("N-D tensor contains nonfinite/non-numeric values");return;}
  if(!Array.isArray(v)||v.length!==result[level])fail("N-D ragged tensor dimension");
  v.forEach(t=>check(t,level+1));
 };
 check(x,0);return result;
}
const indices=(shape,callback)=>{const index=Array(shape.length).fill(0);function visit(axis){if(axis===shape.length){callback(index.slice());return;}for(let i=0;i<shape[axis];i++){index[axis]=i;visit(axis+1);}}visit(0);};
const at=(x,index)=>index.reduce((v,i)=>v[i],x);
function build(sh,fn){const axes=dims(sh);const iter=(depth,index)=>depth===axes.length?fn(index):Array.from({length:axes[depth]},(_,i)=>iter(depth+1,[...index,i]));const out=iter(0,[]);shape(out);return out;}
function flatten(x){const sh=shape(x),out=[];indices(sh,index=>out.push(at(x,index)));return out;}
function reshape(x,to){const a=shape(x),b=dims(to);if(product(a)!==product(b))fail("N-D reshape element count mismatch: "+a.join("×")+" → "+b.join("×"));
 const data=flatten(x);let cursor=0;return build(b,()=>data[cursor++]);}
function permute(x,order){
 const old=shape(x);if(!Array.isArray(order)||order.length!==old.length||new Set(order).size!==old.length||
  order.some(i=>!Number.isSafeInteger(i)||i<0||i>=old.length))fail("N-D transpose axes require a full permutation");
 return build(order.map(i=>old[i]),index=>{
  const original=Array(old.length);order.forEach((o,i)=>{original[o]=index[i];});return at(x,original);
 });
}
function broadcastShape(a,b){
 const n=Math.max(a.length,b.length),out=[];
 for(let i=0;i<n;i++){const x=a[a.length-1-i]??1,y=b[b.length-1-i]??1;
  if(x!==y&&x!==1&&y!==1)fail("N-D broadcast incompatible axes: "+a.join("×")+" vs "+b.join("×"));
  out.unshift(Math.max(x,y));
 }return dims(out);
}
function broadcastBinary(a,b,op){
 const A=shape(a),B=shape(b),out=broadcastShape(A,B);
 const value=(x,sh,coords)=>at(x,coords.slice(out.length-sh.length).map((v,i)=>sh[i]===1?0:v));
 return build(out,coords=>op(value(a,A,coords),value(b,B,coords)));
}
function reduceSum(x,axis,keepdim=true){
 const src=shape(x);
 if(!Number.isSafeInteger(axis)||axis<0||axis>=src.length)fail("N-D reduce axis out of range");
 const outShape=keepdim?src.map((d,i)=>i===axis?1:d):src.filter((_,i)=>i!==axis);
 dims(outShape);
 return build(outShape,outIdx=>{
   const idx=keepdim?outIdx.slice():[...outIdx.slice(0,axis),0,...outIdx.slice(axis)];
   let sum=0;for(let i=0;i<src[axis];i++){idx[axis]=i;sum+=at(x,idx);}
   return sum;
 });
}
function matmul4(a,b){
 const A=shape(a),B=shape(b);if(A.length!==4||B.length!==4)fail("N-D Batched MatMul requires [B,H,M,K] and [B,H,K,N]");
 if(A[3]!==B[2])fail("N-D MatMul K dimensions mismatch");
 const lead=broadcastShape(A.slice(0,2),B.slice(0,2)),out=[...lead,A[2],B[3]];
 return build(out,([i,j,m,n])=>{
  const ai=A[0]===1?0:i,aj=A[1]===1?0:j,bi=B[0]===1?0:i,bj=B[1]===1?0:j;
  let z=0;for(let k=0;k<A[3];k++)z+=a[ai][aj][m][k]*b[bi][bj][k][n];
  return z;
 });
}
function scale(x,factor){if(!Number.isFinite(factor)||Math.abs(factor)>100)fail("N-D scale factor invalid");
 const sh=shape(x);return build(sh,index=>at(x,index)*factor);}
function softmaxLast(x){
 const sh=shape(x);
 return build(sh,index=>{
   const row=at(x,index.slice(0,-1)),peak=Math.max(...row);
   const terms=row.map(v=>Math.exp(v-peak)),z=terms.reduce((a,b)=>a+b,0);
   if(!Number.isFinite(z)||z<=0)fail("N-D softmax invalid normalization");
   return terms[index.at(-1)]/z;
 });
}
function causalMask4(x,queryStart=0,keyStart=0){
 const sh=shape(x);if(sh.length!==4)fail("N-D causal mask expects [B,H,Tq,Tk]");
 if(!Number.isSafeInteger(queryStart)||queryStart<0||queryStart>64||
    !Number.isSafeInteger(keyStart)||keyStart<0||keyStart>64)fail("causal absolute offsets must be 0–64");
 if(queryStart<keyStart)fail("causal mask would contain a row with no legal key");
 return build(sh,idx=>(keyStart+idx[3]<=queryStart+idx[2]?at(x,idx):-1e9));
}
function seedTensor(sh,seed=1){
 dims(sh);if(!Number.isSafeInteger(seed)||Math.abs(seed)>10000)fail("N-D seed must be bounded integer");
 return build(sh,index=>Math.sin(index.reduce((a,i,j)=>a+(i+1)*(j+2)*0.43,seed*.37))*.8);
}
return Object.freeze({MAX_ELEMENTS,shape,dims,product,flatten,build,reshape,permute,broadcastShape,broadcastBinary,reduceSum,matmul4,seedTensor,scale,softmaxLast,causalMask4});
});
