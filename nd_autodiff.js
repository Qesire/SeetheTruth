/* Exact local VJPs for bounded 2–4D educational tensor blocks.
 * Each rule is an algebraic adjoint, not an optimizer or GPU kernel.
 */
(function(root,factory){const api=factory(root);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthNDVJP=api;})(typeof window!=="undefined"?window:{},function(root){"use strict";
const ND=typeof module==="object"&&module.exports?require("./nd_tensor.js"):root.SeetheTruthND;
const eq=(a,b)=>a.length===b.length&&a.every((x,i)=>x===b[i]);
const zeros=x=>ND.build(ND.shape(x),()=>0);
const accum=(g,a)=>ND.broadcastBinary(g,a,(x,y)=>x+y);
function vjp(kind,input,params,upstream,output){
 const dshape=ND.shape(upstream),source=k=>input[k],A=source("a"),B=source("b"),X=source("x"),result={};
 if(!eq(dshape,ND.shape(output)))throw Error("N-D VJP upstream gradient shape mismatch");
 if(kind==="nd_input")return result;
 if(kind==="nd_mse"){
   const a=source("prediction"),b=source("target"),sh=ND.shape(a),size=ND.product(sh),delta=upstream[0][0];
   result.prediction=ND.build(sh,idx=>2*(ND.at(a,idx)-ND.at(b,idx))*delta/size);
   result.cuts=["target"];return result;
 }
 if(kind==="nd_repeat_kv"){
   const sh=ND.shape(X),ratio=params.queryHeads/sh[1];
   result.x=ND.build(sh,([batch,head,t,d])=>{
     let total=0;for(let h=0;h<ratio;h++)total+=upstream[batch][head*ratio+h][t][d];
     return total;
   });return result;
 }
 if(kind==="nd_permute"){
   const order=[params.axis0,params.axis1,params.axis2,params.axis3],inverse=[];
   order.forEach((old,newAxis)=>{inverse[old]=newAxis;});
   result.x=ND.permute(upstream,inverse);return result;
 }
 if(kind==="nd_reshape"||kind==="nd_flatten"||kind==="nd_expand4d"){
   result.x=ND.reshape(upstream,ND.shape(X));return result;
 }
 if(kind==="nd_scale"){result.x=ND.scale(upstream,params.factor);return result;}
 if(kind==="nd_causal_mask"){
   const sh=ND.shape(X);
   result.x=ND.build(sh,idx=>(params.keyStart+idx[3]<=params.queryStart+idx[2]?ND.at(upstream,idx):0));
   return result;
 }
 if(kind==="nd_softmax_last"){
   const sh=ND.shape(output);
   result.x=ND.build(sh,idx=>{
     const y=ND.at(output,idx),g=ND.at(upstream,idx),prior=idx.slice(0,-1);
     const outRow=ND.at(output,prior),upRow=ND.at(upstream,prior);
     const dot=outRow.reduce((acc,v,j)=>acc+v*upRow[j],0);
     return y*(g-dot);
   });return result;
 }
 if(kind==="nd_reduce_sum"){
   const sh=ND.shape(X);
   result.x=ND.build(sh,idx=>ND.at(upstream,idx.map((v,i)=>i===params.axis?0:v)));
   return result;
 }
 if(kind==="nd_broadcast_add"){
   const out=ND.shape(output);
   const unbroadcast=(v)=>{
     const shape=ND.shape(v),offset=out.length-shape.length;
     return ND.build(shape,index=>{
       let total=0;
       ND.indices(out,idx=>{
         const sub=idx.slice(offset).map((coord,i)=>shape[i]===1?0:coord);
         if(sub.every((coord,i)=>coord===index[i]))total+=ND.at(upstream,idx);
       });
       return total;
     });
   };
   result.a=unbroadcast(A);result.b=unbroadcast(B);return result;
 }
 if(kind==="nd_batched_matmul"){
   const a=ND.shape(A),b=ND.shape(B),out=ND.shape(output),K=a[3];
   const gradA=ND.build(a,([batch,h,m,k])=>{
     let total=0;
     const bs=a[0]===1?Array.from({length:out[0]},(_,i)=>i):[batch],
           hs=a[1]===1?Array.from({length:out[1]},(_,i)=>i):[h];
     for(const bi of bs)for(const hi of hs)for(let n=0;n<b[3];n++)
       total+=upstream[bi][hi][m][n]*B[b[0]===1?0:bi][b[1]===1?0:hi][k][n];
     return total;
   });
   const gradB=ND.build(b,([batch,h,k,n])=>{
     let total=0;
     const bs=b[0]===1?Array.from({length:out[0]},(_,i)=>i):[batch],
           hs=b[1]===1?Array.from({length:out[1]},(_,i)=>i):[h];
     for(const bi of bs)for(const hi of hs)for(let m=0;m<a[2];m++)
       total+=A[a[0]===1?0:bi][a[1]===1?0:hi][m][k]*upstream[bi][hi][m][n];
     return total;
   });
   result.a=gradA;result.b=gradB;return result;
 }
 throw Error("unsupported N-D local VJP "+kind);
}
return Object.freeze({vjp});
});
