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
function repeatKVHeads(x,queryHeads){
 const sh=shape(x);
 if(sh.length!==4)fail("GQA KV Repeat requires four axes [B,Hkv,T,D]");
 if(!Number.isSafeInteger(queryHeads)||queryHeads<1||queryHeads>MAX_DIM||
    queryHeads%sh[1]!==0)fail("GQA Query head count must be an integer multiple of KV heads");
 const group=queryHeads/sh[1];
 return build([sh[0],queryHeads,sh[2],sh[3]],([b,h,t,d])=>x[b][Math.floor(h/group)][t][d]);
}
function meanSquaredError(x,y){
 const a=shape(x),b=shape(y);
 if(a.length!==b.length||a.some((d,i)=>d!==b[i]))fail("N-D MSE requires identical tensor shapes");
 const xf=flatten(x),yf=flatten(y);
 return [[xf.reduce((sum,v,i)=>sum+(v-yf[i])**2,0)/xf.length]];
}
/* Adjacent-pair RoPE (explicit convention): positions are absolute token indices.
 * Real models may instead use split-half pairing, scaling, or distinct frequencies.
 */
function rope4(x,offset=0,rotaryDim=4,base=10000){
 const sh=shape(x);if(sh.length!==4)fail("N-D RoPE expects [B,H,T,D]");
 if(!Number.isSafeInteger(offset)||offset<0||offset>64||!Number.isSafeInteger(rotaryDim)||
    rotaryDim<2||rotaryDim>sh[3]||rotaryDim%2!==0||
    !Number.isFinite(base)||base<2||base>1000000)
   fail("N-D RoPE invalid offset, even rotary dimension or frequency base");
 return build(sh,idx=>{
   const channel=idx[3];if(channel>=rotaryDim)return at(x,idx);
   const pair=Math.floor(channel/2),position=offset+idx[2];
   const angle=position/Math.pow(base,2*pair/rotaryDim);
   const other=[...idx.slice(0,3),channel%2===0?channel+1:channel-1];
   const a=at(x,idx),b=at(x,other),cos=Math.cos(angle),sin=Math.sin(angle);
   return channel%2===0?a*cos-b*sin:a*cos+b*sin;
 });
}
/* Wan2.2 pinned video RoPE: wan/modules/model.py@1ea34ff, rope_params
 * and rope_apply. Source layout is [B,T,N,D]; this native learning block
 * consumes the explicit [B,N,T,D] permuted view used by all N-D operators.
 * complex-frequency segments are (time, height, width), with each pair's
 * frequency exponent its GLOBAL complex index 2*pair/D.
 */
function wanVideoRoPE3DCoordinate(sh,idx,frames,height,width,base=10000){
 dims(sh);
 if(sh.length!==4||sh[3]%2!==0||sh[3]<6||
    !Number.isSafeInteger(frames)||frames<1||frames>8||
    !Number.isSafeInteger(height)||height<1||height>8||
    !Number.isSafeInteger(width)||width<1||width>8||
    frames*height*width!==sh[2]||
    typeof base!=="number"||!Number.isFinite(base)||base<2||base>1000000)
   fail("Wan video RoPE requires [B,N,F*H*W,D even >=6], F/H/W <=8 and valid base");
 if(!Array.isArray(idx)||idx.length!==4||idx.some((v,i)=>!Number.isSafeInteger(v)||v<0||v>=sh[i]))
   fail("Wan video RoPE coordinate out of range");
 const t=idx[2],d=idx[3],pair=Math.floor(d/2),complex=sh[3]/2,
  spatial=Math.floor(complex/3),temporal=complex-2*spatial;
 const axis=pair<temporal?"F":pair<temporal+spatial?"H":"W";
 const f=Math.floor(t/(height*width)),h=Math.floor(t/width)%height,w=t%width;
 const position=axis==="F"?f:axis==="H"?h:w;
 const theta=position/Math.pow(base,2*pair/sh[3]);
 return Object.freeze({axis,frame:f,height:h,width:w,position,
  pair,partner:d%2===0?d+1:d-1,angle:theta,cos:Math.cos(theta),
  sin:Math.sin(theta),temporalPairs:temporal,heightPairs:spatial,widthPairs:spatial});
}
function wanVideoRoPE3D4(x,frames,height,width,base=10000){
 const sh=shape(x);
 // Validation is independent of element magnitudes, and performed even
 // for a zero tensor. N-D build keeps the existing small-CPU bounds.
 wanVideoRoPE3DCoordinate(sh,[0,0,0,0],frames,height,width,base);
 return build(sh,idx=>{
  const meta=wanVideoRoPE3DCoordinate(sh,idx,frames,height,width,base),
   value=at(x,idx),other=at(x,[...idx.slice(0,3),meta.partner]);
  return idx[3]%2===0?value*meta.cos-other*meta.sin:
    value*meta.cos+other*meta.sin;
 });
}
/* Pinned Wan2.2 WanRMSNorm + learned channel gamma before heads:
 * model.py@1ea34ff lines 69-85 and 136-143, 169-172.
 * x is [B,1,T,C], representing actual [B,T,C] BEFORE .view(B,T,H,D).
 * Weight is [1,1,1,C]. Do not normalize each head independently.
 */
/* Generic source-compatible non-affine LayerNorm on the last axis.
 * WanLayerNorm uses float() and eps=1e-6 in the pinned source; the
 * bounded Float64 educational interpretation keeps all channel coupling.
 */
function layerNormLast4(x,epsilon=1e-6){
 const sh=shape(x);
 if(sh.length!==4||typeof epsilon!=="number"||!Number.isFinite(epsilon)||epsilon<=0||epsilon>1)
   fail("N-D LayerNorm requires rank four and positive epsilon");
 const C=sh[3];
 return build(sh,([b,h,t,c])=>{
  let mean=0;for(let j=0;j<C;j++)mean+=x[b][h][t][j]/C;
  let variance=0;for(let j=0;j<C;j++)variance+=(x[b][h][t][j]-mean)**2/C;
  return (x[b][h][t][c]-mean)/Math.sqrt(variance+epsilon);
 });
}
function wanQKWeightedRMSNorm4(x,weight,epsilon=1e-6){
 const sh=shape(x),wh=shape(weight);
 if(sh.length!==4||sh[1]!==1||wh.length!==4||
    wh[0]!==1||wh[1]!==1||wh[2]!==1||wh[3]!==sh[3]||
    typeof epsilon!=="number"||!Number.isFinite(epsilon)||epsilon<=0||epsilon>1)
  fail("Wan QK RMSNorm requires full-channel [B,1,T,C], gamma [1,1,1,C] and positive epsilon");
 return build(sh,([b,h,t,c])=>{
  let norm=epsilon;for(let j=0;j<sh[3];j++)norm+=x[b][0][t][j]**2/sh[3];
  return x[b][0][t][c]/Math.sqrt(norm)*weight[0][0][0][c];
 });
}
/* Explicit cache read = materialize a contiguous absolute prefix/window.
 * This is a pure DAG operation, not mutable cross-turn model state.
 */
function cacheRead4(x,start=0,length=1){
 const sh=shape(x);if(sh.length!==4)fail("N-D cache read requires [B,H,T,D]");
 if(!Number.isSafeInteger(start)||start<0||!Number.isSafeInteger(length)||
    length<1||start+length>sh[2])fail("N-D cache read window is outside cached Token axis");
 return build([sh[0],sh[1],length,sh[3]],idx=>x[idx[0]][idx[1]][start+idx[2]][idx[3]]);
}
function cacheAppend4(cache,segment){
 const a=shape(cache),b=shape(segment);
 if(a.length!==4||b.length!==4||a[0]!==b[0]||a[1]!==b[1]||a[3]!==b[3])
   fail("N-D cache append requires matching Batch, Head and Channel axes");
 if(a[2]+b[2]>MAX_DIM)fail("N-D cache append exceeds Token capacity");
 return build([a[0],a[1],a[2]+b[2],a[3]],idx=>{
   const t=idx[2];return t<a[2]?cache[idx[0]][idx[1]][t][idx[3]]:
     segment[idx[0]][idx[1]][t-a[2]][idx[3]];
 });
}
function seedTensor(sh,seed=1){
 dims(sh);if(!Number.isSafeInteger(seed)||Math.abs(seed)>10000)fail("N-D seed must be bounded integer");
 return build(sh,index=>Math.sin(index.reduce((a,i,j)=>a+(i+1)*(j+2)*0.43,seed*.37))*.8);
}
return Object.freeze({MAX_ELEMENTS,shape,dims,product,flatten,build,at,indices,reshape,permute,broadcastShape,broadcastBinary,reduceSum,matmul4,seedTensor,scale,softmaxLast,causalMask4,repeatKVHeads,meanSquaredError,rope4,layerNormLast4,wanQKWeightedRMSNorm4,wanVideoRoPE3DCoordinate,wanVideoRoPE3D4,cacheRead4,cacheAppend4});
});
