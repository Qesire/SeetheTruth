/* Bounded, real numerical multimodal tensor adapters.
 * Flat educational contracts, NOT decoded pixels, audio files, video codecs,
 * pretrained embeddings, large-model patch projections or 3D convolutions.
 */
(function(root,factory){"use strict";const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthMultimodal=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
 const KINDS=Object.freeze(["vision_patchify","video_tubelet","audio_stft_power","audio_log_power"]);
 function fail(s){throw Error(s);}
 function integer(v,name,max=16){if(!Number.isSafeInteger(v)||v<1||v>max)fail(name+" must be integer 1.."+max);return v;}
 function validate(kind,p){
  if(!KINDS.includes(kind))fail("unknown multimodal block "+kind);
  if(kind==="vision_patchify"||kind==="video_tubelet"){
   const {height,width,channels,patchHeight,patchWidth}=p;
   [height,width,channels,patchHeight,patchWidth].forEach((v,i)=>integer(v,["height","width","channels","patchHeight","patchWidth"][i],8));
   if(height%patchHeight||width%patchWidth)fail("image dimensions must be divisible by patch sizes");
   if(kind==="video_tubelet"){
    integer(p.frames,"frames",8);integer(p.patchFrames,"patchFrames",8);
    if(p.frames%p.patchFrames)fail("temporal frames must be divisible by patchFrames");
   }
  }else if(kind==="audio_stft_power"){
   integer(p.fftSize,"fftSize",16);integer(p.hop,"hop",16);
   if(![2,4,8,16].includes(p.fftSize))fail("fftSize must be power-of-two among 2,4,8,16");
  }else{
   if(typeof p.floor!=="number"||!Number.isFinite(p.floor)||p.floor<=0||p.floor>1)fail("audio log floor must be in (0,1]");
  }
 }
 function shape(kind,input,p){
  validate(kind,p);
  if(!Array.isArray(input)||input.length!==2||input.some(v=>!Number.isSafeInteger(v)||v<=0||v>16))fail("2D input shape required");
  const [rows,cols]=input;
  let result;
  if(kind==="vision_patchify"){
   if(rows!==p.height||cols!==p.width*p.channels)fail("image layout must be [H,W*C] with channel-last pixels");
   result=[p.height/p.patchHeight*p.width/p.patchWidth,p.patchHeight*p.patchWidth*p.channels];
  }else if(kind==="video_tubelet"){
   if(rows!==p.frames*p.height||cols!==p.width*p.channels)fail("video layout must be [T*H,W*C], frame-major, channel-last");
   result=[p.frames/p.patchFrames*p.height/p.patchHeight*p.width/p.patchWidth,
     p.patchFrames*p.patchHeight*p.patchWidth*p.channels];
  }else if(kind==="audio_stft_power"){
   if(rows!==1||cols<p.fftSize)fail("waveform must be [1,N] with N>=fftSize");
   result=[1+Math.floor((cols-p.fftSize)/p.hop),p.fftSize/2+1];
  }else result=input.slice();
  if(result[0]>8||result[1]>16)fail("result exceeds bounded [T<=8,D<=16] teaching budget");
  return result;
 }
 function execute(kind,x,p){
  const d=shape(kind,[x.length,x[0].length],p),out=[];
  if(kind==="vision_patchify"){
   for(let y=0;y<p.height;y+=p.patchHeight)for(let z=0;z<p.width;z+=p.patchWidth){
    const row=[];
    for(let dy=0;dy<p.patchHeight;dy++)for(let dz=0;dz<p.patchWidth;dz++)
     for(let c=0;c<p.channels;c++)row.push(x[y+dy][(z+dz)*p.channels+c]);
    out.push(row);
   }
  }else if(kind==="video_tubelet"){
   for(let t=0;t<p.frames;t+=p.patchFrames)for(let y=0;y<p.height;y+=p.patchHeight)
    for(let z=0;z<p.width;z+=p.patchWidth){
     const row=[];
     for(let dt=0;dt<p.patchFrames;dt++)for(let dy=0;dy<p.patchHeight;dy++)
      for(let dz=0;dz<p.patchWidth;dz++)for(let c=0;c<p.channels;c++)
       row.push(x[(t+dt)*p.height+y+dy][(z+dz)*p.channels+c]);
     out.push(row);
    }
  }else if(kind==="audio_stft_power"){
   const {fftSize:N,hop}=p;
   const window=Array.from({length:N},(_,n)=>0.5-0.5*Math.cos(2*Math.PI*n/N)); // periodic Hann
   for(let start=0;start+N<=x[0].length;start+=hop){
    const row=[];
    for(let freq=0;freq<=N/2;freq++){
     let re=0,im=0;
     for(let n=0;n<N;n++){
      const a=x[0][start+n]*window[n],phi=2*Math.PI*freq*n/N;
      re+=a*Math.cos(phi);im-=a*Math.sin(phi);
     }
     row.push(re*re+im*im);
    }
    out.push(row);
   }
  }else if(kind==="audio_log_power"){
   for(const row of x){if(row.some(v=>v<0))fail("audio log-power expects nonnegative powers");out.push(row.map(v=>Math.log10(Math.max(v,p.floor))));}
  }
  if(out.length!==d[0]||out.some(row=>row.length!==d[1]||row.some(v=>!Number.isFinite(v))))
   fail("multimodal numerical output mismatches shape or finite contract");
  return out;
 }
 return Object.freeze({KINDS,validate,shape,execute});
});
