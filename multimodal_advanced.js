/* Actual bounded Conv2D, HTK-mel projection, deterministic epsilon-DDIM step.
 * 2D carriers have explicit layout: image [H,W*C], spectrogram [frame,freq],
 * latent [T,D]. All computations are real Float64, not model inference.
 */
(function(root,factory){"use strict";const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthMultimodalAdvanced=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
 const KINDS=Object.freeze(["vision_conv2d","audio_mel_filterbank","diffusion_ddim_step"]);
 const fail=s=>{throw Error(s);};
 const integer=(v,name,limit=16)=>{if(!Number.isSafeInteger(v)||v<1||v>limit)fail(name+" must be an integer 1.."+limit);};
 function validate(kind,p){
  if(!KINDS.includes(kind))fail("unknown multimodal advanced operator "+kind);
  if(kind==="vision_conv2d"){
   for(const k of ["height","width","channels","kernelHeight","kernelWidth","stride","outChannels"])integer(p[k],k,8);
   if(p.kernelHeight>p.height||p.kernelWidth>p.width)fail("Conv2D kernel exceeds image");
  }else if(kind==="audio_mel_filterbank"){
   integer(p.fftSize,"fftSize",16);integer(p.bands,"bands",8);
   if(![2,4,8,16].includes(p.fftSize))fail("mel FFT must have power-of-two window");
   integer(p.sampleRate,"sampleRate",32000);
   if(p.sampleRate<1000)fail("sampleRate must be >= 1000 Hz");
  }else{
   for(const field of ["alphaT","alphaPrevious"]){
    const v=p[field];
    if(typeof v!=="number"||!Number.isFinite(v)||v<=0||v>1)fail(field+" must be in (0,1]");
   }
   if(p.alphaT>p.alphaPrevious)fail("DDIM alpha cumulative product must not decrease along denoising step");
  }
 }
 function check2D(shape){
  if(!Array.isArray(shape)||shape.length!==2||shape.some(x=>!Number.isSafeInteger(x)||x<1||x>16))
   fail("bounded 2D tensor required");
 }
 function shape(kind,inputs,p){
  validate(kind,p);let out;
  if(kind==="vision_conv2d"){
   const x=inputs.x,w=inputs.kernel;check2D(x);check2D(w);
   if(x[0]!==p.height||x[1]!==p.width*p.channels)fail("Conv2D image carrier must be [H,W*C]");
   if(w[0]!==p.outChannels||w[1]!==p.kernelHeight*p.kernelWidth*p.channels)
    fail("Conv2D kernel must be [Cout,Kh*Kw*Cin] with HWC flattening");
   const oh=Math.floor((p.height-p.kernelHeight)/p.stride)+1,ow=Math.floor((p.width-p.kernelWidth)/p.stride)+1;
   out=[oh,ow*p.outChannels];
  }else if(kind==="audio_mel_filterbank"){
   check2D(inputs.x);
   if(inputs.x[1]!==p.fftSize/2+1)fail("mel frequency bins must match fftSize/2+1");
   out=[inputs.x[0],p.bands];
  }else{
   check2D(inputs.x);check2D(inputs.epsilon);
   if(inputs.x.some((v,i)=>v!==inputs.epsilon[i]))fail("DDIM latent and epsilon prediction shapes must match");
   out=inputs.x.slice();
  }
  if(out.some(v=>v>16)||out[0]>8)fail("advanced multimodal result exceeds [T<=8,D<=16] budget");
  return out;
 }
 const hzToMel=f=>2595*Math.log10(1+f/700),melToHz=m=>700*(10**(m/2595)-1);
 function melWeights(p){
  validate("audio_mel_filterbank",p);
  const upper=p.sampleRate/2,m0=hzToMel(0),m1=hzToMel(upper);
  const edges=Array.from({length:p.bands+2},(_,i)=>melToHz(m0+(m1-m0)*i/(p.bands+1)));
  return Array.from({length:p.fftSize/2+1},(_,k)=>{
   const hz=k*p.sampleRate/p.fftSize;
   return Array.from({length:p.bands},(_,m)=>{
    const left=(hz-edges[m])/(edges[m+1]-edges[m]),
          right=(edges[m+2]-hz)/(edges[m+2]-edges[m+1]);
    return Math.max(0,Math.min(left,right));
   });
  });
 }
 function execute(kind,inputs,p){
  const dims=Object.fromEntries(Object.entries(inputs).map(([key,v])=>[key,[v.length,v[0].length]]));
  const outShape=shape(kind,dims,p);
  let out;
  if(kind==="vision_conv2d"){
   const {height,width,channels,kernelHeight:kh,kernelWidth:kw,stride,outChannels:oc}=p;
   out=Array.from({length:outShape[0]},(_,i)=>Array.from({length:outShape[1]},(_,j)=>{
    const col=Math.floor(j/oc),co=j%oc;
    let result=0;
    for(let dy=0;dy<kh;dy++)for(let dx=0;dx<kw;dx++)for(let ci=0;ci<channels;ci++){
     result+=inputs.x[i*stride+dy][(col*stride+dx)*channels+ci]*
       inputs.kernel[co][(dy*kw+dx)*channels+ci];
    }
    return result;
   }));
  }else if(kind==="audio_mel_filterbank"){
   const w=melWeights(p);
   if(inputs.x.some(row=>row.some(v=>v<0)))fail("Mel power projection requires nonnegative input");
   out=inputs.x.map(row=>Array.from({length:p.bands},(_,m)=>
     row.reduce((sum,v,k)=>sum+v*w[k][m],0)));
  }else{
   const {alphaT:a,alphaPrevious:b}=p;
   const retain=Math.sqrt(b/a),noise=Math.sqrt(1-b)-Math.sqrt(b*(1-a)/a);
   out=inputs.x.map((row,i)=>row.map((v,j)=>retain*v+noise*inputs.epsilon[i][j]));
  }
  if(out.some(row=>row.some(v=>!Number.isFinite(v))))fail("multimodal advanced output non-finite");
  return out;
 }
 return Object.freeze({KINDS,validate,shape,melWeights,execute});
});
