/* Core Source→Graph→Operator→Tensor→Coordinate contract.
 * Every preview is explicitly tagged static, teaching-numeric or source-aligned
 * with synthetic numbers. Source metadata NEVER upgrades arithmetic provenance.
 * No HTML generation, IO, evaluation, network or dynamic code.
 */
(function(root,factory){'use strict';const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;root.SeetheTruthComputeMicroscope=api;})(typeof window!=='undefined'?window:{},function(){'use strict';
const SCHEMA='seethetruth-computation-microscope-v1',MAX_STEPS=160,MAX_CELLS=4096;
const TIERS=Object.freeze({STATIC:'STATIC_SHAPES_NO_EXECUTION',MICRO:'EXECUTABLE_DETERMINISTIC_TEACHING_TENSORS',WAN:'PINNED_WAN_SOURCE_SYNTHETIC_FLOAT64'});
function must(x,msg){if(!x)throw Error('microscope: '+msg);}
function product(shape){return shape.reduce((x,y)=>x*y,1);}
function numericTensor(shape,values){
 must(Array.isArray(shape)&&shape.length===2&&shape.every(x=>Number.isSafeInteger(x)&&x>0)&&product(shape)<=MAX_CELLS,'only bounded two-dimensional tensors');
 must(Array.isArray(values)&&values.length===product(shape)&&values.every(x=>typeof x==='number'&&Number.isFinite(x)),'tensor values/shape invalid');
 return {shape:shape.slice(),values:values.slice()};
}
function staticTensor(shape){
 must(Array.isArray(shape)&&shape.length>=1&&shape.length<=5&&shape.every(x=>Number.isSafeInteger(x)&&x>0),'invalid symbolic shape');
 return {shape:shape.slice(),values:null};
}
function fromNested(values){
 must(Array.isArray(values)&&values.length>0&&Array.isArray(values[0])&&values[0].length>0,'only 2D numeric tensors can be drilled into');
 const rows=values.length,cols=values[0].length;
 must(values.every(r=>Array.isArray(r)&&r.length===cols),'ragged matrix');
 return numericTensor([rows,cols],values.flat());
}
function hashWeight(id,r,c){let h=2166136261;for(const ch of id+':'+r+':'+c)h=Math.imul(h^ch.charCodeAt(0),16777619);return ((h>>>0)%2001-1000)/2600;}
function dotSpec(input,weights,bias,orientation='out_in'){return {type:'dot',input,weights,bias,orientation};}
function validate(frame){
 must(frame&&frame.schema===SCHEMA,'wrong schema');
 must(Object.values(TIERS).includes(frame.evidenceTier),'unknown evidence tier');
 must(Array.isArray(frame.steps)&&frame.steps.length>=1&&frame.steps.length<=MAX_STEPS,'invalid step count');
 must(typeof frame.title==='string'&&frame.title.length>0,'missing title');
 const ids=new Set();let hasNumeric=false;
 for(const s of frame.steps){
  must(s&&typeof s.id==='string'&&/^[A-Za-z0-9_:\-]{1,80}$/.test(s.id)&&!ids.has(s.id),'duplicate or invalid step id');
  must(typeof s.kind==='string'&&s.kind.length>0&&typeof s.title==='string'&&s.title.length>0,'missing step metadata');
  must(typeof s.formula==='string'&&typeof s.why==='string','missing math description');
  must(Array.isArray(s.dependencies)&&s.dependencies.every(d=>d&&typeof d.from==='string'&&ids.has(d.from)&&typeof d.port==='string'&&d.port.length>0),'dependencies must point backward to computed nodes');
  must(s.out&&Array.isArray(s.out.shape),'missing output tensor');
  if(s.out.values!==null){numericTensor(s.out.shape,s.out.values);hasNumeric=true;}
  else staticTensor(s.out.shape);
  must(s.operands&&typeof s.operands==='object'&&!Array.isArray(s.operands),'missing operands');
  for(const x of Object.values(s.operands)){if(x.values!==null)numericTensor(x.shape,x.values);else staticTensor(x.shape);}
  if(s.dot){
   must(s.out.values!==null&&s.dot.type==='dot'&&s.dot.orientation==='out_in','unsupported dot contract');
   const {input,weights,bias}=s.dot;
   numericTensor(input.shape,input.values);numericTensor(weights.shape,weights.values);numericTensor(bias.shape,bias.values);
   const [T,D]=input.shape,[H,C]=weights.shape;
   must(D===C&&s.out.shape[0]===T&&s.out.shape[1]===H&&bias.shape[0]===1&&bias.shape[1]===H,'GEMM input/weight/output misalignment');
   for(let t=0;t<T;t++)for(let h=0;h<H;h++){
    let sum=bias.values[h];for(let k=0;k<D;k++)sum+=input.values[t*D+k]*weights.values[h*D+k];
    must(Math.abs(sum-s.out.values[t*H+h])<=1e-9*Math.max(1,Math.abs(sum)),'GEMM reconstruction contradicts displayed output');
   }
  }
  ids.add(s.id);
 }
 must((frame.evidenceTier===TIERS.STATIC)===(hasNumeric===false),'symbolic vs numeric provenance mismatch');
 if(frame.source)must(typeof frame.source.url==='string'&&/^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+\/blob\/[a-f0-9]{40}\//.test(frame.source.url)&&frame.source.tier==='PINNED_CODE_STRUCTURE_ONLY','unverified pinned-source metadata');
 return frame;
}
function dot(frame,stepId,row,col){
 validate(frame);const s=frame.steps.find(x=>x.id===stepId);must(s,'unknown node');
 must(s.out.values!==null,'numeric execution unavailable');
 const [T,H]=s.out.shape;must(Number.isInteger(row)&&row>=0&&row<T&&Number.isInteger(col)&&col>=0&&col<H,'invalid coordinate');
 const observed=s.out.values[row*H+col];
 if(!s.dot)return {nodeId:s.id,row,col,value:observed,formula:s.formula,reconstructable:false};
 const {input,weights,bias}=s.dot,D=input.shape[1];
 const terms=Array.from({length:D},(_,k)=>({k,input:input.values[row*D+k],weight:weights.values[col*D+k],product:input.values[row*D+k]*weights.values[col*D+k]}));
 const recomputed=bias.values[col]+terms.reduce((a,b)=>a+b.product,0);
 return {nodeId:s.id,row,col,value:observed,bias:bias.values[col],terms,recomputed,error:Math.abs(observed-recomputed),formula:'bias[out] + Σ input[token,k] × W[out,k]',reconstructable:true};
}
function learning(graph,run,catalog){
 must(graph&&Array.isArray(graph.nodes)&&Array.isArray(graph.edges),'missing graph');
 must(run&&Array.isArray(run.steps)&&run.steps.length>0,'missing execution');
 must(catalog&&catalog.defs,'missing block dictionary');
 const numeric=run.evidence==='deterministic_browser_teaching_simulation_not_model_execution';
 // concept steps have "transformation" and no tensor values.
 const isConcept=run.mode==='concept';
 must(numeric||isConcept,'unrecognized execution evidence; decline to infer authenticity');
 const nodes=new Map(graph.nodes.map(n=>[n.id,n]));const byId=new Map();
 const steps=run.steps.map((r,i)=>{
  const n=nodes.get(r.id),def=catalog.defs[n?.kind];must(n&&def,'trace node not found in graph');
  must(r.kind===n.kind,'trace kind changed since run');
  const dependencies=graph.edges.filter(e=>e.to===n.id).map(e=>({from:e.from,port:e.port}));
  let out;
  if(isConcept){out=staticTensor(r.shape);}else{
   out=fromNested(r.values);
   must(r.shape.join(',')===out.shape.join(','),'trace tensor shape contradicts row data');
  }
  const operands={},kind=n.kind;let specification=null;
  if(!isConcept){
   for(const dep of dependencies){
    const upstream=byId.get(dep.from);must(upstream&&upstream.out.values,'missing computed upstream');
    operands[dep.port]=upstream.out;
   }
   if(kind==='linear'){
    const x=operands.x;must(x&&x.shape.length===2,'Linear input missing');
    const [_,D]=x.shape,H=out.shape[1];
    const weights=numericTensor([H,D],Array.from({length:H*D},(_,p)=>hashWeight(n.id,Math.floor(p/D),p%D)));
    const bias=numericTensor([1,H],Array.from({length:H},(_,h)=>hashWeight(n.id,97,h)));
    specification=dotSpec(x,weights,bias);operands.weight=weights;operands.bias=bias;
   }
   if(kind==='matmul'){
    const A=operands.a,B=operands.b;must(A&&B&&A.shape[1]===B.shape[0],'matmul operands missing');
    const [D,H]=B.shape,transposed=numericTensor([H,D],Array.from({length:H*D},(_,p)=>B.values[(p%D)*H+Math.floor(p/D)]));
    const bias=numericTensor([1,H],Array(H).fill(0));
    specification=dotSpec(A,transposed,bias);operands.weight=transposed;
   }
  }
  const s={id:r.id,kind,title:def.title,formula:def.formula,why:def.why,
   sourceLines:null,dependencies,out,operands,dot:specification,
   note:isConcept?'仅静态推导，数值不可用':'执行器真实返回的小矩阵；未装载实际模型'};
  byId.set(s.id,s);return s;
 });
 return validate({schema:SCHEMA,title:graph.title||'可执行积木计算图',evidenceTier:isConcept?TIERS.STATIC:TIERS.MICRO,
  source:null,steps,origin:'learning_lab_executor',limits:'no_real_model_weights_or_hardware_profiling'});
}
function wan(data){
 must(data&&data.source&&Array.isArray(data.steps)&&data.steps.length===9,'missing pinned Wan FFN trace');
 const ref=data.source,srcUrl=ref.source;
 must(ref.commit==='1ea34ff48f87168174e12956e200b1d908b1c5ff','unsupported Wan source revision');
 const steps=data.steps.map((s,i)=>{
  const previous=i===0?[]:[{from:data.steps[i-1].id,port:'previous'}];
  const operands=Object.fromEntries(Object.entries(s.aux||{}).filter(([k,v])=>v&&Array.isArray(v.shape)&&Array.isArray(v.values)).map(([k,v])=>[k,numericTensor(v.shape,v.values)]));
  const dot=(i===4||i===6)?dotSpec(operands.input,operands.weights,operands.bias):null;
  const line=s.lines?.match(/\d+/)?.[0]||null;
  return {id:s.id,kind:i===4||i===6?'linear':s.id,title:s.name,formula:s.code,why:s.why,
    sourceLines:line?{start:Number(line),end:Number(s.lines?.match(/\d+/g)?.[1]||line)}:null,
    dependencies:previous,out:numericTensor(s.main.shape,s.main.values),operands,dot,
    note:s.error};
 });
 return validate({schema:SCHEMA,title:'Wan2.2 I2V A14B · FFN 逐步计算',evidenceTier:TIERS.WAN,
   source:{url:srcUrl,tier:'PINNED_CODE_STRUCTURE_ONLY',revision:ref.commit,path:ref.path},
   steps,origin:'source_aligned_micro_numeric_engine',limits:'synthetic_float64_no_actual_wan_weights_no_cuda_profile'});
}
return Object.freeze({SCHEMA,TIERS,validate,learning,wan,dot,numericTensor,staticTensor});
});