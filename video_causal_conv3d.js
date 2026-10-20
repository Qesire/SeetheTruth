/* Native 5D causal Conv3D micro-kernel with immutable chunked inference.
 * Unlike patch/token-only models, B,T,H,W,C and O,Kt,Kh,Kw,I are separate
 * semantic axes.  No padding in space, stride 1 in time, zero-left-pad in time.
 * This is a small deterministic mathematical subset, not the full Wan VAE.
 */
(function(root,factory){"use strict";
 const api=factory(typeof module==="object"&&module.exports?require("./tensor_ir.js"):root.SeetheTruthTensorIR);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthVideoConv3D=api;
})(typeof window!=="undefined"?window:{},function(IR){"use strict";
 const fail=s=>{throw Error(s);};
 const integer=(x,name)=>{if(!Number.isSafeInteger(x)||x<1||x>8)fail(name+" must be integer in 1..8");};
 function validate(p){
  for(const k of ["frames","height","width","channels","kernelFrames","kernelHeight","kernelWidth","stride","outChannels"])integer(p[k],k);
  if(p.kernelHeight>p.height||p.kernelWidth>p.width)fail("spatial kernel exceeds video frame");
  if(p.kernelFrames>p.frames)fail("temporal kernel exceeds configured video frames");
 }
 function dimensions(p){
  validate(p);
  return {OH:Math.floor((p.height-p.kernelHeight)/p.stride)+1,
    OW:Math.floor((p.width-p.kernelWidth)/p.stride)+1,
    K:p.kernelFrames*p.kernelHeight*p.kernelWidth*p.channels};
 }
 function bounded2d(s,name){
  if(!Array.isArray(s)||s.length!==2||s.some(v=>!Number.isSafeInteger(v)||v<1||v>16))
   fail(name+" needs a bounded 2D matrix");
 }
 function shape(inputs,p){
  const {OH,OW,K}=dimensions(p);
  bounded2d(inputs.x,"video input");bounded2d(inputs.kernel,"conv3d weights");
  if(inputs.x[0]!==p.frames*p.height||inputs.x[1]!==p.width*p.channels)
   fail("video carrier must be [T*H,W*Cin] in THWC order");
  if(inputs.kernel[0]!==p.outChannels||inputs.kernel[1]!==K)
   fail("kernel must be [Cout,Kt*Kh*Kw*Cin]");
  const out=[p.frames*OH,OW*p.outChannels];
  if(out[0]>8||out[1]>16)fail("Conv3D output exceeds bounded [T<=8,D<=16] teaching matrix");
  return out;
 }
 function matShape(x,label){
  if(!Array.isArray(x)||!x.length||!Array.isArray(x[0])||!x[0].length||
     x.some(row=>!Array.isArray(row)||row.length!==x[0].length||row.some(v=>typeof v!=="number"||!Number.isFinite(v))))
   fail(label+" must be a rectangular finite matrix");
  return [x.length,x[0].length];
 }
 function makeNativeVideo(x,p){
  const raw=matShape(x,"video");
  if(raw[0]!==p.frames*p.height||raw[1]!==p.width*p.channels)
   fail("video carrier must match THWC axes");
  return IR.make([1,p.frames,p.height,p.width,p.channels],["B","T","H","W","C"],x.flat());
 }
 function makeNativeKernel(w,p){
  const raw=matShape(w,"kernel"),K=p.kernelFrames*p.kernelHeight*p.kernelWidth*p.channels;
  if(raw[0]!==p.outChannels||raw[1]!==K)fail("kernel carrier must match O×Kt×Kh×Kw×I");
  return IR.make([p.outChannels,p.kernelFrames,p.kernelHeight,p.kernelWidth,p.channels],
    ["O","Kt","Kh","Kw","I"],w.flat());
 }
 function nativeConv3D(video,kernel,p){
  const {OH,OW}=dimensions(p);
  if(video.axes.join(",")!=="B,T,H,W,C"||kernel.axes.join(",")!=="O,Kt,Kh,Kw,I"||
     video.shape.join(",")!==[1,p.frames,p.height,p.width,p.channels].join(",")||
     kernel.shape.join(",")!==[p.outChannels,p.kernelFrames,p.kernelHeight,p.kernelWidth,p.channels].join(","))
   fail("native rank-5 axes or shape mismatch; channel ordering must be explicit");
  const output=Array(p.outChannels*p.frames*OH*OW).fill(0);
  for(let oc=0;oc<p.outChannels;oc++)for(let t=0;t<p.frames;t++)
   for(let y=0;y<OH;y++)for(let xx=0;xx<OW;xx++){
    let sum=0;
    for(let dt=0;dt<p.kernelFrames;dt++){
     const time=t+dt-(p.kernelFrames-1);
     if(time<0)continue; // zero-left-padding, strict causal receptive field
     for(let dy=0;dy<p.kernelHeight;dy++)for(let dx=0;dx<p.kernelWidth;dx++)
      for(let ci=0;ci<p.channels;ci++)
       sum+=IR.get(video,[0,time,y*p.stride+dy,xx*p.stride+dx,ci])*
        IR.get(kernel,[oc,dt,dy,dx,ci]);
    }
    output[((oc*p.frames+t)*OH+y)*OW+xx]=sum;
   }
  return IR.make([1,p.outChannels,p.frames,OH,OW],["B","O","T","H","W"],output);
 }
 function toCarrier(native,p){
  const {OH,OW}=dimensions(p),out=[];
  for(let t=0;t<p.frames;t++)for(let y=0;y<OH;y++){
   const row=[];
   for(let xx=0;xx<OW;xx++)for(let oc=0;oc<p.outChannels;oc++)
    row.push(IR.get(native,[0,oc,t,y,xx]));
   out.push(row);
  }
  return out;
 }
 function execute(x,kernel,p){
  shape({x:matShape(x,"video"),kernel:matShape(kernel,"kernel")},p);
  return toCarrier(nativeConv3D(makeNativeVideo(x,p),makeNativeKernel(kernel,p),p),p);
 }
 function initialize(p,kernel){
  const {K}=dimensions(p);
  const ws=matShape(kernel,"kernel");
  if(ws[0]!==p.outChannels||ws[1]!==K)fail("stream kernel shape mismatch");
  return {p:{...p},kernel:kernel.map(row=>row.slice()),history:[],processed:0};
 }
 function pushChunk(state,chunk){
  const p=state.p,{OH,OW}=dimensions(p),K=p.kernelFrames;
  const x=matShape(chunk,"chunk"),T=x[0]/p.height;
  if(!Number.isSafeInteger(T)||T<1||x[1]!==p.width*p.channels)
   fail("stream chunk must contain complete THWC frames");
  if(state.processed+T>p.frames)fail("stream chunk exceeds declared total frames");
  if(state.history.length%p.height||state.history.length>Math.min(K-1,state.processed)*p.height)
   fail("corrupt causal stream state");
  const result=[];
  const prevCount=state.history.length/p.height;
  for(let t=0;t<T;t++)for(let y=0;y<OH;y++){
   const row=[];
   for(let xx=0;xx<OW;xx++)for(let oc=0;oc<p.outChannels;oc++){
    let sum=0;
    for(let dt=0;dt<K;dt++){
     const time=t+dt-(K-1);
     if(time < -prevCount)continue; // unavailable prefix = zeros
     for(let dy=0;dy<p.kernelHeight;dy++)for(let dx=0;dx<p.kernelWidth;dx++)
      for(let ci=0;ci<p.channels;ci++){
       const r=time>=0?x[time*p.height+y*p.stride+dy]:
        state.history[(prevCount+time)*p.height+y*p.stride+dy];
       sum+=r[(xx*p.stride+dx)*p.channels+ci]*
        state.kernel[oc][((dt*p.kernelHeight+dy)*p.kernelWidth+dx)*p.channels+ci];
      }
    }
    row.push(sum);
   }
   result.push(row);
  }
  const keep=Math.min(K-1,state.processed+T)*p.height;
  const history=(keep?[...state.history,...chunk].slice(-keep):[]).map(row=>row.slice());
  return {output:result,state:{p:{...p},kernel:state.kernel.map(row=>row.slice()),
    history,processed:state.processed+T}};
 }
 return Object.freeze({validate,dimensions,shape,makeNativeVideo,makeNativeKernel,nativeConv3D,
   toCarrier,execute,initialize,pushChunk});
});
