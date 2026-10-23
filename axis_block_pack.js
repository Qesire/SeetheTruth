/* Semantic named-axis pack. This is a real family of freely composable graph
 * blocks: definitions, inference, numeric execution, trace, and lesson recipe
 * are installed through one generic registry, not page-specific conditions. */
(function(root,factory){'use strict';const api=factory(
 typeof module==='object'&&module.exports?require('./tensor_ir.js'):root.SeetheTruthTensorIR,
 typeof module==='object'&&module.exports?require('./semantic_state.js'):root.SeetheTruthSemanticState);
 if(typeof module==='object'&&module.exports)module.exports=api;
 root.SeetheTruthAxisBlockPack=api;
})(typeof window!=='undefined'?window:{},function(IR,S){'use strict';
 const SRC='https://pytorch.org/docs/stable/notes/broadcasting.html';
 const def=(title,en,ports,params,formula,why,lesson)=>({
  title,en,category:'tensor',ports,params,formula,why,lesson,source:SRC,
  caution:'保留 T/D 轴身份并执行真实 Float64 元素运算；同样的形状不自动代表同样的语义。',
  knowledge:{flow:'按 T（Token）与 D（通道）展开每一个位置，不以 2D 长宽代替轴名。',
   practice:lesson,trap:'不能把特征 Bias 当成 Token 轴上的长度匹配；同型不等于同轴。',
   exercise:'连接 T/D 张量源和特征 Bias，检查逐通道广播、转置、时间均值。',
   refs:'broadcasting,tensor_views'}
 });
 const defs=Object.freeze({
  axis_tensor_source:def('具名 T×D 数据源','Named Token × Feature Source',[],{tokens:3,width:3,seed:1},
   'X[t,d]=sin(0.37(tD+d+1)+0.11·seed)','生成明确有 T/D 轴语义的小张量。','改变 Token 数量或通道数，检查下游 Shape。'),
  axis_feature_bias:def('具名 D 轴偏置','Named Feature Bias',[],{width:3,scale:1},
   'b[d]=scale·(d+1)','Bias 只属于 D 轴，输出保持可与 [T,D] 连线的 [1,D] 形状。','调整 scale，观察所有 Token 的相同 D 通道增量。'),
  axis_named_add:def('按 D 轴广播加法','Named Feature Broadcast Add',['x','bias'],{},
   'Y[t,d]=X[t,d]+b[d]','只沿特征通道广播，对 Token 不作广播语义猜测。','交换输入与 Bias 应被端口尺寸合同识别。'),
  axis_named_transpose:def('显式具名轴转置','Transpose T,D → D,T',['x'],{},
   'Y[d,t]=X[t,d]','实际逐坐标转置，用以与原生 Transpose2D 做数值对照。','对比原生 Transpose2D 是否一项不差。'),
  axis_time_mean:def('沿 Token/时间 T 轴平均','Mean over T',['x'],{},
   'Y[0,d]=(1/T) Σ_t X[t,d]','对每个 D 通道逐 Token 求和，缩减成 [1,D]。','改变一枚 Token 只影响相应通道的平均值。')
 });
 // Contract travels with the graph edge. Two arrays of the same length
 // remain different if their named axes identify different semantics.
 const namedAxes=Object.freeze({
  axis_tensor_source:{output:['T','D'],inputs:{}},
  axis_feature_bias:{output:['One','D'],inputs:{}},
  axis_named_add:{output:['T','D'],inputs:{x:['T','D'],bias:['One','D']}},
  axis_named_transpose:{output:['D','T'],inputs:{x:['T','D']}},
  axis_time_mean:{output:['MeanT','D'],inputs:{x:['T','D']}}
 });
 const annotatedDefs=Object.freeze(Object.fromEntries(Object.entries(defs).map(([kind,d])=>[
  kind,Object.freeze({...d,portAxes:namedAxes[kind].inputs,outputAxes:namedAxes[kind].output})
 ])));
 const count=(n,lo,hi,label)=>{if(!Number.isSafeInteger(n)||n<lo||n>hi)throw Error(label+' must be integer '+lo+'..'+hi);};
 function validate(kind,p){
  if(kind==='axis_tensor_source'){count(p.tokens,1,8,'tokens');count(p.width,1,16,'width');count(p.seed,-100,100,'seed');}
  if(kind==='axis_feature_bias'){count(p.width,1,16,'width');if(typeof p.scale!=='number'||!Number.isFinite(p.scale)||Math.abs(p.scale)>100)throw Error('feature bias scale must be finite ≤100');}
  if(!Object.hasOwn(defs,kind))throw Error('unknown named-axis module '+kind);
 }
 const check2=(a,label)=>{if(!Array.isArray(a)||a.length!==2||a.some(n=>!Number.isSafeInteger(n)||n<1)||a[0]>8||a[1]>16)throw Error(label+' requires [T,D] with T≤8 D≤16');};
 function infer(kind,p,shapes){
  if(kind==='axis_tensor_source')return [p.tokens,p.width];
  if(kind==='axis_feature_bias')return [1,p.width];
  if(kind==='axis_named_add'){
   const x=shapes.x,b=shapes.bias;check2(x,'named add x');check2(b,'named add bias');
   if(b[0]!==1||b[1]!==x[1])throw Error('named feature bias must be exactly [1,D] where D=input D');
   return x.slice();
  }
  const x=shapes.x;check2(x,kind);
  return kind==='axis_named_transpose'?[x[1],x[0]]:[1,x[1]];
 }
 const shp=v=>[v.length,v[0].length];
 const flatten=v=>v.flat();
 function execute(kind,p,inputs){
  if(kind==='axis_tensor_source'){
   const sh=[p.tokens,p.width],data=Array.from({length:p.tokens*p.width},(_,i)=>Math.sin(.37*(i+1)+.11*p.seed));
   const t=IR.make(sh,['T','D'],data);
   return {values:Array.from({length:sh[0]},(_,i)=>t.data.slice(i*sh[1],(i+1)*sh[1])),
    internals:{axes:['T','D'],generatedBy:'TensorIR.make'}};
  }
  if(kind==='axis_feature_bias'){
   const t=IR.make([p.width],['D'],Array.from({length:p.width},(_,i)=>p.scale*(i+1)));
   return {values:[t.data.slice()],internals:{axes:['D'],outputPortShape:[1,p.width]}};
  }
  if(kind==='axis_named_add'){
   const x=IR.make(shp(inputs.x),['T','D'],flatten(inputs.x));
   const b=IR.make([inputs.bias[0].length],['D'],inputs.bias[0]);
   const y=S.binary(x,b,'add',['T','D']);
   return {values:Array.from({length:y.shape[0]},(_,i)=>y.data.slice(i*y.shape[1],(i+1)*y.shape[1])),
    internals:{semanticAxes:['T','D'],broadcastPort:'bias',broadcastAxis:'D'}};
  }
  const x=IR.make(shp(inputs.x),['T','D'],flatten(inputs.x));
  if(kind==='axis_named_transpose'){
   const y=S.reorder(x,['D','T']);
   return {values:Array.from({length:y.shape[0]},(_,i)=>y.data.slice(i*y.shape[1],(i+1)*y.shape[1])),
    internals:{fromAxes:['T','D'],toAxes:['D','T']}};
  }
  // Each channel is independently reduced across the T axis in a real small
  // numeric operator; no fabricated output is copied from a preset.
  const values=x.data,sh=x.shape;
  const mean=Array.from({length:sh[1]},(_,d)=>{
   let sum=0;for(let t=0;t<sh[0];t++)sum+=IR.get(x,[t,d]);return sum/sh[0];
  });
  return {values:[mean],internals:{reducedAxis:'T',inputShape:sh}};
 }
 function trace(kind,p,inputs,run,node){
  const out={shape:[run.values.length,run.values[0].length],values:flatten(run.values)};
  const operands=Object.fromEntries(Object.entries(inputs).map(([port,v])=>[port,{shape:shp(v),values:flatten(v)}]));
  let indexMap;
  if(kind==='axis_named_transpose'){
   const [T,D]=shp(inputs.x);
   indexMap=out.values.map((_,i)=>({port:'x',index:(i%T)*D+Math.floor(i/T)}));
  }
  const op={
   id:node.id+':td',kind:'named_axis_transform',title:defs[kind].title,
   formula:defs[kind].formula,why:defs[kind].why+'。这里执行的正是积木注册引擎里的真实 T/D 张量操作。',
   out,operands,outputIsParent:true,...(indexMap?{indexMap}:{})
  };
  return [op];
 }
 const nodes=[
  {id:'x',kind:'axis_tensor_source',x:20,y:60,params:{tokens:3,width:3,seed:2}},
  {id:'bias',kind:'axis_feature_bias',x:20,y:260,params:{width:3,scale:2}},
  {id:'add',kind:'axis_named_add',x:300,y:120,params:{}},
  {id:'swap',kind:'axis_named_transpose',x:555,y:75,params:{}},
  {id:'native',kind:'transpose2d',x:555,y:305,params:{}},
  {id:'delta',kind:'subtract',x:820,y:190,params:{}},
  {id:'mean',kind:'axis_time_mean',x:540,y:480,params:{}}
 ];
 const recipes=Object.freeze([{
  id:'extension_named_axis_composition',
  title:'模块包 · 具名轴广播、转置、均值',
  summary:'同一输出连接原生算子作数值差分；具名 T/D 语义、真正的广播与转换',
  graph:{title:'通用模块包 · T/D 广播 → 双路径转置等价与逐通道时间均值',
   nodes,edges:[
    {from:'x',to:'add',port:'x'},{from:'bias',to:'add',port:'bias'},
    {from:'add',to:'swap',port:'x'},{from:'add',to:'native',port:'x'},
    {from:'swap',to:'delta',port:'a'},{from:'native',to:'delta',port:'b'},
    {from:'add',to:'mean',port:'x'}
   ]}
 }]);
 return Object.freeze({name:'named_axis',defs:annotatedDefs,validate,infer,execute,trace,recipes});
});