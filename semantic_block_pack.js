/* One independently installable semantic-video block family. All operators call
 * TensorIR/SemanticState; axes remain B,T,H,W,C throughout the pack. */
(function(root,factory){'use strict';const api=factory(
 typeof module==='object'&&module.exports?require('./tensor_ir.js'):root.SeetheTruthTensorIR,
 typeof module==='object'&&module.exports?require('./semantic_state.js'):root.SeetheTruthSemanticState);
 if(typeof module==='object'&&module.exports)module.exports=api;
 root.SeetheTruthSemanticBlockPack=api;
})(typeof window!=='undefined'?window:{},function(IR,S){'use strict';
 const AXES=['B','T','H','W','C'],source='https://docs.pytorch.org/docs/stable/tensor_view.html';
 const def=(title,en,ports,params,formula,why,lesson)=>({title,en,category:'video',ports,params,formula,why,lesson,
  portAxes:Object.fromEntries(ports.map(p=>[p,AXES])),outputAxes:AXES,
  caution:'本模块使用实际五维 TensorIR 数值和具名时间轴 T。通过改动计算次序与索引检查语义。',
  source,knowledge:{flow:'显示真实 [B,T,H,W,C] 张量及时间维的读写顺序，区分时间与空间维。',
   practice:'相同数据和形状下，改变 T 轴的范围或切分顺序，直接核对每一个索引。',
   trap:'不能把同样长度的 H、W、C 轴当作时间轴，也不能把未显式执行的 Stream append 当作 KV 写入。',
   exercise:'先生成视频再按 T 切两段，连接时间合并积木；比较重新连接的顺序与输出。',
   refs:'tensor_views,broadcasting'}});
 const defs=Object.freeze({
  semantic_video_source:def('具名视频张量源 [B,T,H,W,C]','Named Video Tensor Source',[],{frames:3,seed:1},
   'X ∈ ℝ^{1×T×2×2×2}, axes=[B,T,H,W,C]','生成真实五维 Float64 张量，定义明确的时间与空间轴。',
   '调节帧数/种子，直接观察五维数值与 Frame 轴切片。'),
  semantic_video_slice:def('具名 T 轴裁切','Named Time Slice',['x'],{start:0,length:1},
   'Y = slice_T(X,start,length)','仅沿具名时间轴取连续帧，不改变 B/H/W/C 的含义。',
   '选择不同起点，比较裁剪结果和原视频中相应的切片。'),
  semantic_video_concat:def('具名 T 轴拼接','Named Time Concat',['a','b'],{},
   'Y = concat_T(A,B)','两个已确认 B/H/W/C 匹配的视频片段沿 T 轴组成新序列。',
   '反转 A/B 连线观察输出帧顺序变化，形状虽匹配却不代表语义相同。')
 });
 function valid(kind,p){
  if(kind==='semantic_video_source'){
   if(!Number.isSafeInteger(p.frames)||p.frames<1||p.frames>4||
      !Number.isSafeInteger(p.seed)||Math.abs(p.seed)>1000)throw Error('video source frames 1..4, seed integer ±1000');
  }else if(kind==='semantic_video_slice'){
   if(!Number.isSafeInteger(p.start)||p.start<0||p.start>4||
      !Number.isSafeInteger(p.length)||p.length<1||p.length>4)throw Error('time slice start/length bounds');
  }
 }
 const check5=(sh,label)=>{if(!Array.isArray(sh)||sh.length!==5||sh[0]!==1||
  sh.slice(2).some((n,i)=>n!==2))throw Error(label+' expects named B,T,H,W,C with [1,T,2,2,2]');};
 function infer(kind,p,inputs){
  if(kind==='semantic_video_source')return [1,p.frames,2,2,2];
  if(kind==='semantic_video_slice'){
   const x=inputs.x;check5(x,'time slice');if(p.start+p.length>x[1])throw Error('time slice exceeds T');
   return [1,p.length,2,2,2];
  }
  if(kind==='semantic_video_concat'){
   const a=inputs.a,b=inputs.b;check5(a,'time concat A');check5(b,'time concat B');
   if(a[1]+b[1]>4)throw Error('combined time length exceeds four-frame teaching budget');
   return [1,a[1]+b[1],2,2,2];
  }
  throw Error('unknown named video block '+kind);
 }
 const pack=(shape,flat)=>{let idx=0;const go=d=>d===shape.length?flat[idx++]:Array.from({length:shape[d]},()=>go(d+1));return go(0);};
 const tensor=v=>IR.make(shape(v),AXES,flatten(v));
 function shape(value){let v=value,ret=[];while(Array.isArray(v)){ret.push(v.length);v=v[0];}return ret;}
 function flatten(value){return Array.isArray(value)?value.flatMap(flatten):[value];}
 function execute(kind,p,inputs){
  if(kind==='semantic_video_source'){
   const shape=[1,p.frames,2,2,2],flat=Array.from({length:shape.reduce((a,b)=>a*b,1)},(_,i)=>Math.sin((i+1)*.31+p.seed*.7));
   const canonical=IR.make(shape,AXES,flat);
   return {values:pack(shape,canonical.data),internals:{sourceAxes:AXES.join(',')}};
  }
  if(kind==='semantic_video_slice'){
   const result=S.slice(tensor(inputs.x),'T',p.start,p.length);
   return {values:pack(result.shape,result.data),internals:{timeStart:p.start,timeEnd:p.start+p.length}};
  }
  const result=S.concat([tensor(inputs.a),tensor(inputs.b)],'T');
  return {values:pack(result.shape,result.data),internals:{timeBoundary:shape(inputs.a)[1]}};
 }
 function trace(kind,p,inputs,run,node){
  const value=run.values,sh=shape(value);
  const out={shape:sh,values:flatten(value)};
  const operands=Object.fromEntries(Object.entries(inputs).map(([k,v])=>[k,{shape:shape(v),values:flatten(v)}]));
  let indexMap;
  if(kind==='semantic_video_slice'){
   const span=sh.slice(2).reduce((a,b)=>a*b,1);
   indexMap=out.values.map((_,i)=>({port:'x',index:(Math.floor(i/span)+p.start)*span+i%span}));
  }else if(kind==='semantic_video_concat'){
   const first=flatten(inputs.a).length;
   indexMap=out.values.map((_,i)=>i<first?{port:'a',index:i}:{port:'b',index:i-first});
  }
  return [{id:node.id+':named_axes',kind:'named_axis_transform',title:kind==='semantic_video_source'?'实际生成 B,T,H,W,C':kind==='semantic_video_slice'?'沿 T 轴严格裁切':'按连线顺序拼接 T 轴',
    formula:defs[kind].formula,why:defs[kind].why+' 输出形状 '+sh.join('×')+'；真实内容来自 TensorIR / SemanticState。',
    out,operands,outputIsParent:true,...(indexMap?{indexMap}:{})}];
 }
 const recipes=Object.freeze([{
  id:'extension_video_time_composition',
  title:'模块包 · 五维视频时序切片与重组',
  summary:'独立 B/T/H/W/C 视频积木：以真实五维 Float64 数值验证时间切分与有序拼接',
  graph:{
   title:'视频时序积木 · 原始视频 → 两段切片 → 有序重组',
   nodes:[
    {id:'video',kind:'semantic_video_source',x:20,y:170,params:{frames:3,seed:1}},
    {id:'early',kind:'semantic_video_slice',x:310,y:65,params:{start:0,length:2}},
    {id:'late',kind:'semantic_video_slice',x:310,y:320,params:{start:2,length:1}},
    {id:'join',kind:'semantic_video_concat',x:650,y:170,params:{}}
   ],
   edges:[
    {from:'video',to:'early',port:'x'},{from:'video',to:'late',port:'x'},
    {from:'early',to:'join',port:'a'},{from:'late',to:'join',port:'b'}
   ]
  }
 }]);
 return Object.freeze({name:'semantic_video',defs,validate:valid,infer,execute,trace,recipes});
});