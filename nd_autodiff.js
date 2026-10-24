/* Exact local VJPs for bounded 2–4D educational tensor blocks.
 * Each rule is an algebraic adjoint, not an optimizer or GPU kernel.
 */
(function(root,factory){const api=factory(root);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthNDVJP=api;})(typeof window!=="undefined"?window:{},function(root){"use strict";
const ND=typeof module==="object"&&module.exports?require("./nd_tensor.js"):root.SeetheTruthND;
const Act=typeof module==="object"&&module.exports?require("./activation_math.js"):root.SeetheTruthActivations;
const eq=(a,b)=>a.length===b.length&&a.every((x,i)=>x===b[i]);
const zeros=x=>ND.build(ND.shape(x),()=>0);
const accum=(g,a)=>ND.broadcastBinary(g,a,(x,y)=>x+y);
function vjp(kind,input,params,upstream,output){
 const dshape=ND.shape(upstream),source=k=>input[k],A=source("a"),B=source("b"),X=source("x"),result={};
 if(!eq(dshape,ND.shape(output)))throw Error("N-D VJP upstream gradient shape mismatch");
 if(["nd_gelu_erf","nd_silu","nd_relu_squared","nd_mish","nd_elu","nd_hardswish"].includes(kind)){
   const sh=ND.shape(X);result.x=ND.build(sh,idx=>ND.at(upstream,idx)*Act.derivative(kind.slice(3),ND.at(X,idx),params));return result;
 }
 if(kind==="nd_input")return result;
 if(kind==="nd_mse"){
   const a=source("prediction"),b=source("target"),sh=ND.shape(a),size=ND.product(sh),delta=upstream[0][0];
   result.prediction=ND.build(sh,idx=>2*(ND.at(a,idx)-ND.at(b,idx))*delta/size);
   result.cuts=["target"];return result;
 }
 if(kind==="nd_rope"){
   const sh=ND.shape(X),R=params.rotaryDim,offset=params.offset,base=params.base;
   result.x=ND.build(sh,idx=>{
     const d=idx[3];if(d>=R)return ND.at(upstream,idx);
     const pair=Math.floor(d/2),theta=(offset+idx[2])/Math.pow(base,2*pair/R);
     const cos=Math.cos(theta),sin=Math.sin(theta),pairIdx=[...idx.slice(0,3),d%2===0?d+1:d-1],
           g=ND.at(upstream,idx),other=ND.at(upstream,pairIdx);
     return d%2===0?cos*g+sin*other:cos*g-sin*other;
   });return result;
 }
 if(kind==="nd_layernorm_last"){
   const sh=ND.shape(X),C=sh[3],eps=params.epsilon;
   result.x=ND.build(sh,([b,h,t,c])=>{
     let mean=0;for(let j=0;j<C;j++)mean+=X[b][h][t][j]/C;
     let var0=0;for(let j=0;j<C;j++)var0+=(X[b][h][t][j]-mean)**2/C;
     const inv=1/Math.sqrt(var0+eps),normC=(X[b][h][t][c]-mean)*inv;
     let avgG=0,avgGZ=0;
     for(let j=0;j<C;j++){
       const g=upstream[b][h][t][j];
       avgG+=g/C;avgGZ+=g*(X[b][h][t][j]-mean)*inv/C;
     }
     return inv*(upstream[b][h][t][c]-avgG-normC*avgGZ);
   });
   return result;
 }
 if(kind==="nd_wan_qk_rmsnorm"){
   const sh=ND.shape(X),w=source("weight"),C=sh[3],eps=params.epsilon;
   if(sh.length!==4||sh[1]!==1||!eq(ND.shape(w),[1,1,1,C]))
     throw Error("Wan QK full-width RMSNorm VJP input/gamma shape mismatch");
   const inv=(b,t)=>{
     let sq=eps;for(let j=0;j<C;j++)sq+=X[b][0][t][j]**2/C;
     return 1/Math.sqrt(sq);
   };
   result.x=ND.build(sh,([b,h,t,c])=>{
     const u=inv(b,t);let dot=0;
     for(let j=0;j<C;j++)dot+=upstream[b][0][t][j]*w[0][0][0][j]*X[b][0][t][j];
     return upstream[b][0][t][c]*w[0][0][0][c]*u-
       X[b][0][t][c]*u**3*dot/C;
   });
   result.weight=ND.build([1,1,1,C],([b,h,t,c])=>{
     let total=0;for(let batch=0;batch<sh[0];batch++)for(let token=0;token<sh[2];token++)
       total+=upstream[batch][0][token][c]*X[batch][0][token][c]*inv(batch,token);
     return total;
   });
   return result;
 }
 if(kind==="nd_wan_rope3d"){
   const sh=ND.shape(X),{frames,height,width,base}=params;
   result.x=ND.build(sh,idx=>{
     const d=idx[3],m=ND.wanVideoRoPE3DCoordinate(sh,idx,frames,height,width,base),
      pair=[...idx.slice(0,3),m.partner],g=ND.at(upstream,idx),other=ND.at(upstream,pair);
     return d%2===0?m.cos*g+m.sin*other:m.cos*g-m.sin*other;
   });return result;
 }
 if(kind==="nd_cache_read"){
   const sh=ND.shape(X);
   result.x=ND.build(sh,idx=>{
     const t=idx[2]-params.start;
     return t>=0&&t<params.length?ND.at(upstream,[idx[0],idx[1],t,idx[3]]):0;
   });return result;
 }
 if(kind==="nd_cache_append"){
   const a=ND.shape(input.cache),b=ND.shape(input.segment);
   result.cache=ND.cacheRead4(upstream,0,a[2]);
   result.segment=ND.cacheRead4(upstream,a[2],b[2]);
   return result;
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
 if(kind==="nd_broadcast_mul"){
   const ash=ND.shape(A),bsh=ND.shape(B),out=ND.shape(output);
   const reduce=(src,other)=>{
     const sh=ND.shape(src),os=ND.shape(other);
     return ND.build(sh,idx=>{
       let total=0;ND.indices(out,coord=>{
         const local=coord.map((v,i)=>sh[i]===1?0:v);
         if(!local.every((v,i)=>v===idx[i]))return;
         const paired=coord.map((v,i)=>os[i]===1?0:v);
         total+=ND.at(upstream,coord)*ND.at(other,paired);
       });return total;
     });
   };
   result.a=reduce(A,B);result.b=reduce(B,A);return result;
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
