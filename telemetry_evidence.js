/* Portable telemetry validator and clearly labeled synthetic fixture; no DOM. */
(function(root,factory){"use strict";const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;if(root)root.SeetheTruthTelemetryEvidence=api;
})(typeof window!=="undefined"?window:null,function(){"use strict";
function example(){
const mat=Array.from({length:20},(_,i)=>Array.from({length:20},(_,j)=>+(Math.sin(i*.66)*Math.cos(j*.39)+Math.sin((i-j)*.16)*.23).toFixed(4)));
const flat=mat.flat(),lo=Math.min(...flat),hi=Math.max(...flat),bins=Array(24).fill(0);
for(const x of flat)bins[Math.min(23,Math.floor((x-lo)/(hi-lo)*24))]++;
return {schema:'seethetruth-telemetry-v1',name:'Synthetic tutorial (not a measured model)',meta:{measurement:'SYNTHETIC_DEMO_NOT_MEASURED',gpu_activity:false,tensor_values_opt_in:true,host_timing_is_not_gpu_latency:true,notes:['Synthetic numeric values demonstrate the UI only.']},modules:[{name:'layers.0.attn',type:'DemoAttention',calls:1,host_wall_ms:3.8,last_outputs:[{shape:[1,20,20],stride:[400,20,1],dtype:'float32'}]},{name:'layers.0.ffn',type:'DemoFFN',calls:1,host_wall_ms:2.6,last_outputs:[{shape:[1,20,20],stride:[400,20,1],dtype:'float32'}]}],operators:[{name:'aten::mm',count:4,self_cpu_ms:3.0,self_device_ms:null},{name:'aten::softmax',count:2,self_cpu_ms:1.2,self_device_ms:null},{name:'aten::mul',count:3,self_cpu_ms:.6,self_device_ms:null}],snapshots:[{module:'layers.0.attn',tensor:{shape:[1,20,20],stride:[400,20,1],dtype:'float32',device:'cpu',numel:400,logical_bytes:1600,contiguous:true},summary:{population:400,sampled:400,finite_count:400,nonfinite_count:0,sample_zero_fraction:0,statistics:{min:lo,max:hi,mean:0,std:.6},histogram:{bins,domain:[lo,hi]},matrix:mat,matrix_original_shape:[20,20],matrix_stride:[1,1],leading_indices:[0]}}],gpu_memory:null};
}
function validate(d){
 const fail=msg=>{throw Error('Telemetry schema：'+msg)};
 const obj=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
 const finite=x=>typeof x==='number'&&Number.isFinite(x);
 const text=x=>typeof x==='string'&&x.length<=500;
 const ints=(x,max)=>Array.isArray(x)&&x.length<=max&&x.every(y=>Number.isSafeInteger(y)&&y>=0);
 const nullableNumber=x=>x===null||finite(x);
 if(!obj(d)||d.schema!=='seethetruth-telemetry-v1'||!Array.isArray(d.operators)||!Array.isArray(d.modules)||!Array.isArray(d.snapshots))fail('不是 SeetheTruth Telemetry v1');
 if(d.operators.length>5000||d.modules.length>5000||d.snapshots.length>64)fail('数据超过允许的条目数');
 if(d.meta!=null&&(!obj(d.meta)||
   (d.meta.notes!=null&&(!Array.isArray(d.meta.notes)||d.meta.notes.length>100||!d.meta.notes.every(text)))||
   (d.meta.measurement!=null&&!text(d.meta.measurement))))fail('非法采集元数据');
 for(const op of d.operators){
  if(!obj(op)||!text(op.name)||!Number.isSafeInteger(op.count)||op.count<0||
    !finite(op.self_cpu_ms)||op.self_cpu_ms<0||
    !(op.self_device_ms==null||(finite(op.self_device_ms)&&op.self_device_ms>=0)))fail('非法算子耗时或调用次数');
 }
 for(const m of d.modules){
  if(!obj(m)||!text(m.name)||!text(m.type)||!Number.isSafeInteger(m.calls)||m.calls<0||!finite(m.host_wall_ms)||m.host_wall_ms<0)fail('非法模块名称或 Host 时间');
  if(m.last_outputs!=null&&!Array.isArray(m.last_outputs))fail('模块张量元数据需要数组');
  if(m.exclusive_self_cpu_ms!=null&&(!finite(m.exclusive_self_cpu_ms)||m.exclusive_self_cpu_ms<0||
      !finite(m.inclusive_attributed_cpu_ms)||m.inclusive_attributed_cpu_ms<0||
      m.inclusive_attributed_cpu_ms+.0002<m.exclusive_self_cpu_ms))fail('非法模块独占或含下级 ATen CPU 成本');
 }
 if(d.module_calls!==undefined){
  if(!Array.isArray(d.module_calls)||d.module_calls.length>10000)fail('过多模块逐次调用');
  const keys=new Set();
  for(const c of d.module_calls){
   if(!obj(c)||!text(c.name)||!text(c.type)||!text(c.invocation_key)||
       !Number.isSafeInteger(c.invocation_index)||c.invocation_index<1||
       !finite(c.exclusive_self_cpu_ms)||c.exclusive_self_cpu_ms<0||
       !finite(c.inclusive_attributed_cpu_ms)||c.inclusive_attributed_cpu_ms<0||
       c.inclusive_attributed_cpu_ms+.0002<c.exclusive_self_cpu_ms||
       keys.has(c.invocation_key))fail('非法或重复模块调用成本');
   keys.add(c.invocation_key);
  }
 }
 for(const t of d.snapshots){
  if(!obj(t)||!text(t.module)||!obj(t.tensor)||!obj(t.summary))fail('非法 snapshot 格式');
  if(!ints(t.tensor.shape,16)||!Array.isArray(t.tensor.stride)||t.tensor.stride.length!==t.tensor.shape.length||
    !t.tensor.stride.every(Number.isSafeInteger)||!text(t.tensor.dtype||'')||!text(t.tensor.device||''))fail('非法 shape/stride/dtype');
  const z=t.summary, mat=z.matrix, hist=z.histogram;
  if(!Array.isArray(mat)||mat.length>32||mat.some(row=>!Array.isArray(row)||row.length>32||row.some(v=>!nullableNumber(v))))fail('非法热力图矩阵');
  if(!obj(hist)||!Array.isArray(hist.bins)||hist.bins.length<1||hist.bins.length>256||
    !ints(hist.bins,256)||!Array.isArray(hist.domain)||hist.domain.length!==2||!hist.domain.every(nullableNumber))fail('非法采样直方图');
  if(!obj(z.statistics)||Object.values(z.statistics).some(v=>!nullableNumber(v)))fail('非法统计量');
  if(!Number.isSafeInteger(z.sampled)||!Number.isSafeInteger(z.population)||z.sampled<0||z.population<z.sampled||
    !Number.isSafeInteger(z.finite_count)||!Number.isSafeInteger(z.nonfinite_count)||
    z.finite_count+z.nonfinite_count!==z.sampled||
    !finite(z.sample_zero_fraction)||z.sample_zero_fraction<0||z.sample_zero_fraction>1)fail('采样计数或比例不一致');
  if(hist.bins.reduce((a,v)=>a+v,0)!==z.finite_count)fail('直方图计数和有效样本数不匹配');
  if(!ints(z.matrix_original_shape,2)||z.matrix_original_shape.length!==2||
    !ints(z.matrix_stride,2)||z.matrix_stride.length!==2)fail('矩阵切片形状与间隔无效');
 }
}

return Object.freeze({validate,example});
});