/* Core Source→Graph→Operator→Tensor→Coordinate contract.
 * Every preview is explicitly tagged static, teaching-numeric or source-aligned
 * with synthetic numbers. Source metadata NEVER upgrades arithmetic provenance.
 * No HTML generation, IO, evaluation, network or dynamic code.
 */
(function(root,factory){'use strict';const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;root.SeetheTruthComputeMicroscope=api;})(typeof window!=='undefined'?window:{},function(){'use strict';
const SCHEMA='seethetruth-computation-microscope-v1',MAX_STEPS=160,MAX_CELLS=16384;
const TIERS=Object.freeze({STATIC:'STATIC_SHAPES_NO_EXECUTION',MICRO:'EXECUTABLE_DETERMINISTIC_TEACHING_TENSORS',WAN:'PINNED_WAN_SOURCE_SYNTHETIC_FLOAT64'});
function must(x,msg){if(!x)throw Error('microscope: '+msg);}
function product(shape){return shape.reduce((x,y)=>x*y,1);}
function numericTensor(shape,values){
 must(Array.isArray(shape)&&shape.length>=1&&shape.length<=5&&shape.every(x=>Number.isSafeInteger(x)&&x>0)&&product(shape)<=MAX_CELLS,'only bounded rank 1–5 tensors');
 must(Array.isArray(values)&&values.length===product(shape)&&values.every(x=>typeof x==='number'&&Number.isFinite(x)),'tensor values/shape invalid');
 return {shape:shape.slice(),values:values.slice()};
}
function staticTensor(shape){
 must(Array.isArray(shape)&&shape.length>=1&&shape.length<=5&&shape.every(x=>Number.isSafeInteger(x)&&x>0),'invalid symbolic shape');
 return {shape:shape.slice(),values:null};
}
function tagAxes(tensor,axes){
 if(axes==null)return tensor;
 must(Array.isArray(axes)&&axes.length===tensor.shape.length&&new Set(axes).size===axes.length&&
  axes.every(x=>typeof x==='string'&&/^[A-Za-z][A-Za-z0-9_]{0,15}$/.test(x)),
  'invalid named axis metadata');
 return {...tensor,axes:axes.slice()};
}
function checkAxes(tensor){
 if(tensor.axes!==undefined)tagAxes(tensor,tensor.axes);
}
function flatCoords(index,shape){
 const out=Array(shape.length);
 for(let i=shape.length-1;i>=0;i--){out[i]=index%shape[i];index=Math.floor(index/shape[i]);}
 return out;
}
function coordinateLabel(tensor,coords){
 return '['+coords.map((value,i)=>(tensor.axes?tensor.axes[i]+'=':'')+value).join(',')+']';
}
function fromNested(values){
 must(Array.isArray(values)&&values.length>0,'empty tensor');
 const shape=[];let cursor=values;
 while(Array.isArray(cursor)){
  must(cursor.length>0,'empty tensor axis');shape.push(cursor.length);
  must(shape.length<=5,'unsupported rank >5');cursor=cursor[0];
 }
 const flat=[];
 function visit(node,axis){
  if(axis===shape.length){must(typeof node==='number'&&Number.isFinite(node),'nonfinite scalar');flat.push(node);return;}
  must(Array.isArray(node)&&node.length===shape[axis],'ragged tensor');
  for(const child of node)visit(child,axis+1);
 }
 visit(values,0);return numericTensor(shape,flat);
}
function hashWeight(id,r,c){let h=2166136261;for(const ch of id+':'+r+':'+c)h=Math.imul(h^ch.charCodeAt(0),16777619);return ((h>>>0)%2001-1000)/2600;}
function dotSpec(input,weights,bias,orientation='out_in'){return {type:'dot',input,weights,bias,orientation};}
function verifyDot(s){
 if(!s.dot)return;
 must(s.out.values!==null&&s.dot.type==='dot'&&s.dot.orientation==='out_in','unsupported dot contract');
 const {input,weights,bias}=s.dot;
 numericTensor(input.shape,input.values);numericTensor(weights.shape,weights.values);numericTensor(bias.shape,bias.values);
 const [T,D]=input.shape,[H,C]=weights.shape;
 must(input.shape.length===2&&weights.shape.length===2&&s.out.shape.length===2&&bias.shape.length===2&&D===C&&s.out.shape[0]===T&&s.out.shape[1]===H&&bias.shape[0]===1&&bias.shape[1]===H,'GEMM input/weight/output misalignment');
 for(let t=0;t<T;t++)for(let h=0;h<H;h++){
  let sum=bias.values[h];for(let k=0;k<D;k++)sum+=input.values[t*D+k]*weights.values[h*D+k];
  must(Math.abs(sum-s.out.values[t*H+h])<=1e-9*Math.max(1,Math.abs(sum)),'GEMM reconstruction contradicts displayed output');
 }
}
const BINARY_PROOFS=['add','mul','subtract','divide'];
const UNARY_PROOFS=['scale','bias','square','relu','relu_squared','sigmoid','tanh','gelu_tanh','softplus'];
const INDEX_PROOFS=['transpose2d','slice_rows','slice_cols','concat_cols','repeat_rows'];
function unaryValue(op,x,p){
 if(op==='scale')return x*p.factor;
 if(op==='bias')return x+p.bias;
 if(op==='square')return x*x;
 if(op==='relu')return Math.max(0,x);
 if(op==='relu_squared')return Math.max(0,x)**2;
 if(op==='sigmoid')return x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
 if(op==='tanh')return Math.tanh(x);
 if(op==='gelu_tanh')return .5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+.044715*x**3)));
 if(op==='softplus')return Math.max(x,0)+Math.log1p(Math.exp(-Math.abs(x)));
 throw Error('microscope: unknown unary proof');
}
function verifyUnary(s){
 if(s.unary===undefined)return;
 const p=s.unary,op=p?.operator,x=s.operands.x;
 must(s.out.values!==null&&s.out.shape.length===2&&UNARY_PROOFS.includes(op)&&s.kind===op,
  'unsupported scalar numerical proof');
 must(x&&x.values&&x.shape.join(',')===s.out.shape.join(','),'scalar operand and output shapes disagree');
 const key=op==='scale'?'factor':op==='bias'?'bias':null;
 must(key?(Number.isFinite(p[key])&&Object.keys(p).length===2):Object.keys(p).length===1,
  'invalid scalar proof parameter');
 for(let i=0;i<s.out.values.length;i++){
  const value=unaryValue(op,x.values[i],p);
  must(Number.isFinite(value)&&Math.abs(value-s.out.values[i])<=1e-12*Math.max(1,Math.abs(value)),
   'scalar numerical reconstruction contradicts displayed output');
 }
}
function nativeIndexPlan(s){
 const op=s.indexRule?.operator,p=s.indexRule?.params||{},O=s.out.shape;
 must(INDEX_PROOFS.includes(op)&&op===s.kind&&O.length===2,'unsupported native indexed proof');
 const x=s.operands.x,a=s.operands.a,b=s.operands.b,[R,C]=O;
 if(op==='concat_cols')must(a&&b&&a.shape.length===2&&b.shape.length===2&&
  a.shape[0]===R&&b.shape[0]===R&&a.shape[1]+b.shape[1]===C,'native concat shape mismatch');
 else{
  must(x&&x.shape.length===2,'native indexed source missing');
  const [r,c]=x.shape;
  if(op==='transpose2d')must(R===c&&C===r,'native transpose shape mismatch');
  if(op==='slice_rows')must(Number.isSafeInteger(p.start)&&p.start>=0&&R===p.length&&C===c&&p.start+R<=r,'native row slice shape mismatch');
  if(op==='slice_cols')must(Number.isSafeInteger(p.start)&&p.start>=0&&C===p.width&&R===r&&p.start+C<=c,'native col slice shape mismatch');
  if(op==='repeat_rows')must(Number.isSafeInteger(p.times)&&p.times>0&&R===r*p.times&&C===c,'native repeated row shape mismatch');
 }
 return Array.from({length:R*C},(_,n)=>{
  const i=Math.floor(n/C),j=n%C;
  if(op==='transpose2d')return {port:'x',index:j*x.shape[1]+i};
  if(op==='slice_rows')return {port:'x',index:(i+p.start)*x.shape[1]+j};
  if(op==='slice_cols')return {port:'x',index:i*x.shape[1]+j+p.start};
  if(op==='repeat_rows')return {port:'x',index:(i%x.shape[0])*x.shape[1]+j};
  return j<a.shape[1]?{port:'a',index:i*a.shape[1]+j}:{port:'b',index:i*b.shape[1]+j-a.shape[1]};
 });
}
function verifyElementwise(s){
 if(s.elementwise===undefined)return;
 must(s.out.values!==null&&s.out.shape.length===2&&
  s.elementwise&&BINARY_PROOFS.includes(s.elementwise.operator)&&s.elementwise.operator===s.kind,
  'unsupported numerical elementwise reconstruction');
 const a=s.operands.a,b=s.operands.b;
 must(a&&b&&a.values&&b.values&&
  a.shape.join(',')===s.out.shape.join(',')&&b.shape.join(',')===s.out.shape.join(','),
  'elementwise operands and output shapes disagree');
 for(let i=0;i<s.out.values.length;i++){
  const op=s.elementwise.operator,x=a.values[i],y=b.values[i];
  must(op!=='divide'||y!==0,'elementwise division by zero');
  const expected=op==='add'?x+y:op==='mul'?x*y:op==='subtract'?x-y:x/y;
  must(expected===s.out.values[i],'elementwise arithmetic contradicts displayed output');
 }
}
function kernelCoordinate(s,row,col){
 const {operator,inputPort,epsilon}=s.kernel,x=s.operands[inputPort];
 const cols=x.shape[1],start=row*cols,v=x.values[start+col];
 if(operator==='silu'){
  const denominator=1+Math.exp(-v),recomputed=v/denominator;
  return {recomputed,terms:[{k:col,input:v,term:denominator}],statistic:'1+exp(-x)',statisticValue:denominator,
    formula:'SiLU(x)=x/(1+exp(-x))'};
 }
 if(operator==='rmsnorm'){
  const values=x.values.slice(start,start+cols),squares=values.map(z=>z*z),
   meanSquare=squares.reduce((a,b)=>a+b,0)/cols,
   inverse=1/Math.sqrt(meanSquare+epsilon);
  return {recomputed:v*inverse,terms:values.map((a,k)=>({k,input:a,term:squares[k]})),
   statistic:'1/√(mean(x²)+ε)',statisticValue:inverse,
   formula:'x/√(Σ_k x[k]² / D + ε), ε='+epsilon};
 }
 const values=x.values.slice(start,start+cols),maximum=Math.max(...values),
  exps=values.map(z=>Math.exp(z-maximum)),denominator=exps.reduce((a,b)=>a+b,0);
 return {recomputed:exps[col]/denominator,terms:values.map((a,k)=>({k,input:a,term:exps[k]})),
  statistic:'Σ exp(x[k]−max(x))',statisticValue:denominator,
  formula:'exp(x[c]−max(x))/Σ_k exp(x[k]−max(x)), max(x)='+maximum.toPrecision(10)};
}
function verifyKernel(s){
 if(s.kernel===undefined)return;
 const rule=s.kernel;
 must(s.out.values!==null&&s.out.shape.length===2&&rule&&
  ['rmsnorm','silu','softmax'].includes(rule.operator)&&s.kind===rule.operator,
  'unsupported rowwise numerical proof');
 const port=s.id.endsWith(':softmax')?'scores':'x';
 must(rule.inputPort===port,'rowwise input port provenance mismatch');
 if(rule.operator==='rmsnorm')
  must(typeof rule.epsilon==='number'&&Number.isFinite(rule.epsilon)&&rule.epsilon>0&&rule.epsilon<=1,'invalid RMSNorm epsilon');
 else must(rule.epsilon===undefined,'unexpected rowwise parameter');
 const input=s.operands[port];
 must(input&&input.values!==null&&input.shape.join(',')===s.out.shape.join(','),
  'rowwise input/output shapes disagree');
 for(let row=0;row<s.out.shape[0];row++)for(let col=0;col<s.out.shape[1];col++){
  const value=kernelCoordinate(s,row,col).recomputed,actual=s.out.values[row*s.out.shape[1]+col];
  must(Math.abs(value-actual)<=1e-11*Math.max(1,Math.abs(value)),
   rule.operator+' numerical reconstruction contradicts output');
 }
}
function verifyIndexMap(s){
 if(s.indexMap===undefined){must(s.indexRule===undefined,'native index rule without source map');return;}
 must(s.out.values!==null&&Array.isArray(s.indexMap)&&s.indexMap.length===s.out.values.length,'invalid tensor index source map');
 const expectedIndex=s.indexRule?nativeIndexPlan(s):null;
 for(let i=0;i<s.indexMap.length;i++){
  const entry=s.indexMap[i];
  if(expectedIndex)must(entry&&entry.port===expectedIndex[i].port&&entry.index===expectedIndex[i].index,'native source index contract mismatch');
  must(entry&&typeof entry.port==='string'&&Object.hasOwn(s.operands,entry.port),'unresolved tensor source port');
  const input=s.operands[entry.port];
  must(input&&input.values!==null&&Number.isSafeInteger(entry.index)&&entry.index>=0&&entry.index<input.values.length,'source index out of bounds');
  must(s.out.values[i]===input.values[entry.index],'mapped output contradicts input coordinate');
 }
}
function validate(frame){
 must(frame&&frame.schema===SCHEMA,'wrong schema');
 must(Object.values(TIERS).includes(frame.evidenceTier),'unknown evidence tier');
 must(Array.isArray(frame.steps)&&frame.steps.length>=1&&frame.steps.length<=MAX_STEPS,'invalid step count');
 must(typeof frame.title==='string'&&frame.title.length>0,'missing title');
 const ids=new Set(),resolved=new Map();let hasNumeric=false;
 for(const s of frame.steps){
  must(s&&typeof s.id==='string'&&/^[A-Za-z0-9_:\-]{1,80}$/.test(s.id)&&!ids.has(s.id),'duplicate or invalid step id');
  must(typeof s.kind==='string'&&s.kind.length>0&&typeof s.title==='string'&&s.title.length>0,'missing step metadata');
  must(typeof s.formula==='string'&&typeof s.why==='string','missing math description');
  must(Array.isArray(s.dependencies)&&s.dependencies.every(d=>d&&typeof d.from==='string'&&ids.has(d.from)&&typeof d.port==='string'&&d.port.length>0),'dependencies must point backward to computed nodes');
  must(s.out&&Array.isArray(s.out.shape),'missing output tensor');
  if(s.out.values!==null){numericTensor(s.out.shape,s.out.values);hasNumeric=true;}
  else staticTensor(s.out.shape);
  checkAxes(s.out);
  must(s.operands&&typeof s.operands==='object'&&!Array.isArray(s.operands),'missing operands');
  for(const x of Object.values(s.operands)){if(x.values!==null)numericTensor(x.shape,x.values);else staticTensor(x.shape);checkAxes(x);}
  verifyDot(s);verifyIndexMap(s);verifyElementwise(s);verifyUnary(s);verifyKernel(s);
  if(s.substeps!==undefined){
   must(Array.isArray(s.substeps)&&s.substeps.length>=1&&s.substeps.length<=12,'invalid nested substeps');
   const nestedIds=new Set();
   for(const child of s.substeps){
    must(child&&typeof child.id==='string'&&child.id.startsWith(s.id+':')&&/^[A-Za-z0-9_:\-]{1,80}$/.test(child.id)&&!ids.has(child.id)&&!nestedIds.has(child.id),'invalid or duplicate substep');
    must(typeof child.title==='string'&&child.title&&typeof child.kind==='string'&&typeof child.formula==='string'&&typeof child.why==='string','substep missing explanation');
    must(child.out&&child.out.values!==null,'nested substep needs measured teaching numerical data');
    numericTensor(child.out.shape,child.out.values);checkAxes(child.out);
    must(child.operands&&typeof child.operands==='object'&&!Array.isArray(child.operands),'nested inputs invalid');
    for(const tensor of Object.values(child.operands)){numericTensor(tensor.shape,tensor.values);checkAxes(tensor);}
    verifyDot(child);verifyIndexMap(child);verifyElementwise(child);verifyUnary(child);verifyKernel(child);
    if(child.outputIsParent===true){
     for(const [port,tensor] of Object.entries(child.operands)){
      const parentOperand=s.operands[port];
      if(!parentOperand)continue;
      must(parentOperand.shape.join(',')===tensor.shape.join(',')&&
       JSON.stringify(parentOperand.axes||null)===JSON.stringify(tensor.axes||null)&&
       parentOperand.values&&tensor.values&&parentOperand.values.length===tensor.values.length&&
       tensor.values.every((v,i)=>v===parentOperand.values[i]),
       'named-axis operand substep contradicts incoming graph port');
     }
     must(s.out.values!==null&&child.out.shape.join(',')===s.out.shape.join(',')&&
      JSON.stringify(child.out.axes||null)===JSON.stringify(s.out.axes||null)&&
      child.out.values.length===s.out.values.length&&
      child.out.values.every((value,i)=>value===s.out.values[i]),
      'declared final substep contradicts parent output');
    }
    nestedIds.add(child.id);
   }
  }
  if(frame.origin==='learning_lab_executor'&&s.out.values!==null){
   for(const dep of s.dependencies){
    const parent=resolved.get(dep.from),got=s.operands[dep.port];
    must(parent&&got&&got.shape.join(',')===parent.shape.join(',')&&
     JSON.stringify(got.axes||null)===JSON.stringify(parent.axes||null)&&
     got.values&&parent.values&&got.values.length===parent.values.length&&
     got.values.every((v,i)=>v===parent.values[i]),
     'execution dependency operand provenance mismatch');
   }
  }
  resolved.set(s.id,s.out);
  ids.add(s.id);
 }
 must((frame.evidenceTier===TIERS.STATIC)===(hasNumeric===false),'symbolic vs numeric provenance mismatch');
 if(frame.source)must(typeof frame.source.url==='string'&&/^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+\/blob\/[a-f0-9]{40}\//.test(frame.source.url)&&frame.source.tier==='PINNED_CODE_STRUCTURE_ONLY','unverified pinned-source metadata');
 return frame;
}
function dot(frame,stepId,row,col,prefix=[]){
 validate(frame);const s=frame.steps.find(x=>x.id===stepId)||frame.steps.flatMap(x=>x.substeps||[]).find(x=>x.id===stepId);must(s,'unknown node');
 must(s.out.values!==null,'numeric execution unavailable');
 const shape=s.out.shape,dimensions=shape.length;
 const T=dimensions===1?1:shape[dimensions-2],H=shape[dimensions-1];
 must(Array.isArray(prefix)&&prefix.length<=Math.max(0,dimensions-2),'invalid tensor axis slice');
 const leading=dimensions<=2?[]:Array.from({length:dimensions-2},(_,i)=>prefix[i]??0);
 must(leading.every((v,i)=>Number.isInteger(v)&&v>=0&&v<shape[i]),'invalid tensor axis slice');
 must(Number.isInteger(row)&&row>=0&&row<T&&Number.isInteger(col)&&col>=0&&col<H,'invalid coordinate');
 let index=0;const coord=[...leading,...(dimensions===1?[]:[row]),col];
 for(let i=0;i<dimensions;i++)index=index*shape[i]+coord[i];
 const observed=s.out.values[index],coordinate=coordinateLabel(s.out,coord);
 if(s.indexMap){
  const entry=s.indexMap[index],source=s.operands[entry.port].values[entry.index];
  return {nodeId:s.id,row,col,prefix:leading,value:observed,coordinateLabel:coordinate,
   sourceCoordinateLabel:coordinateLabel(s.operands[entry.port],flatCoords(entry.index,s.operands[entry.port].shape)),
   formula:'output'+coordinate+' = '+entry.port+'[flat '+entry.index+']',
   reconstructable:true,mapped:true,sourcePort:entry.port,sourceIndex:entry.index,
   recomputed:source,error:Math.abs(observed-source),terms:[]};
 }
 if(s.elementwise){
  const a=s.operands.a.values[index],b=s.operands.b.values[index];
  const op=s.elementwise.operator,recomputed=op==='add'?a+b:op==='mul'?a*b:op==='subtract'?a-b:a/b;
  return {nodeId:s.id,row,col,prefix:leading,value:observed,coordinateLabel:coordinate,
   formula:'out=a '+({add:'+',mul:'×',subtract:'−',divide:'/'}[op])+' b',
   elementwise:true,operation:s.elementwise.operator,operands:[{port:'a',value:a},{port:'b',value:b}],
   reconstructable:true,recomputed,error:Math.abs(observed-recomputed)};
 }
 if(s.unary){
  const x=s.operands.x.values[index],recomputed=unaryValue(s.unary.operator,x,s.unary);
  return {nodeId:s.id,row,col,prefix:leading,value:observed,coordinateLabel:coordinate,
   formula:s.formula,reconstructable:true,unary:true,operation:s.unary.operator,input:x,
   parameter:s.unary.factor??s.unary.bias??null,recomputed,error:Math.abs(observed-recomputed)};
 }
 if(s.kernel){
  const diagnostics=kernelCoordinate(s,row,col);
  return {nodeId:s.id,row,col,prefix:leading,value:observed,coordinateLabel:coordinate,
   formula:diagnostics.formula,reconstructable:true,kernel:s.kernel.operator,
   input: s.operands[s.kernel.inputPort].values[index],
   kernelTerms:diagnostics.terms,statistic:diagnostics.statistic,statisticValue:diagnostics.statisticValue,
   recomputed:diagnostics.recomputed,error:Math.abs(observed-diagnostics.recomputed)};
 }
 if(!s.dot)return {nodeId:s.id,row,col,prefix:leading,value:observed,coordinateLabel:coordinate,formula:s.formula,reconstructable:false};
 const {input,weights,bias}=s.dot,D=input.shape[1];
 const terms=Array.from({length:D},(_,k)=>({k,input:input.values[row*D+k],weight:weights.values[col*D+k],product:input.values[row*D+k]*weights.values[col*D+k]}));
 const recomputed=bias.values[col]+terms.reduce((a,b)=>a+b.product,0);
 return {nodeId:s.id,row,col,prefix:leading,value:observed,coordinateLabel:coordinate,bias:bias.values[col],terms,recomputed,error:Math.abs(observed-recomputed),formula:'bias[out] + Σ input[token,k] × W[out,k]',reconstructable:true};
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
  const expectedAxes=def.outputAxes||null;
  must(r.axes==null||JSON.stringify(r.axes)===JSON.stringify(expectedAxes),
   'runtime named axes contradict registered contract '+n.kind);
  out=tagAxes(out,expectedAxes);
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
  let substeps;
  if(!isConcept&&kind==='attention'){
   const q=operands.q,k=operands.k,v=operands.v;
   must(q&&k&&v&&q.shape.length===2&&k.shape.length===2&&v.shape.length===2,'Attention operands missing');
   const scores=fromNested(r.internals?.scores?.values),probabilities=fromNested(r.internals?.probabilities?.values);
   const dimension=q.shape[1],qScaled=numericTensor(q.shape,q.values.map(z=>z/Math.sqrt(dimension)));
   const zeroQ=numericTensor([1,k.shape[0]],Array(k.shape[0]).fill(0));
   const valueWeights=numericTensor([v.shape[1],v.shape[0]],Array.from({length:v.shape[0]*v.shape[1]},(_,p)=>v.values[(p%v.shape[0])*v.shape[1]+Math.floor(p/v.shape[0])]));
   const zeroV=numericTensor([1,v.shape[1]],Array(v.shape[1]).fill(0));
   substeps=[
    {id:r.id+':qk_scores',kind:'scaled_matmul',title:'QKᵀ / √d · Scores',formula:'scores[t,j] = Σ_k Q[t,k] K[j,k] / √d',why:'Attention 先对每个 query/token 和 key/token 完整计算相似度；不是逐元素乘同位置 Token。',out:scores,operands:{q,k},dot:dotSpec(qScaled,k,zeroQ)},
    {id:r.id+':softmax',kind:'softmax',title:'Softmax · Attention probabilities',formula:'P[t,j] = exp(scores[t,j]−max)/Σ_j exp(scores[t,j]−max)',why:'沿每个 Query 的所有 Key 做归一化，避免大 logits 导致的数值溢出。',out:probabilities,operands:{scores},kernel:{operator:'softmax',inputPort:'scores'}},
    {id:r.id+':weighted_value',kind:'matmul',title:'Probabilities × V · Context',formula:'out[t,c] = Σ_j P[t,j] V[j,c]',why:'所有 key/value token 的值按概率汇聚到对应 Query，形成 Attention 的最终输出。',out,operands:{probabilities,v},dot:dotSpec(probabilities,valueWeights,zeroV)}
   ];
  }
  if(!isConcept&&r.extensionSubsteps){
   must(!substeps,'node has ambiguous trace providers');
   substeps=r.extensionSubsteps.map(child=>{
    const operandTags=Object.fromEntries(Object.entries(child.operands||{}).map(([port,tensor])=>[
     port,tagAxes(tensor,def.portAxes?.[port]||null)]));
    return {...child,operands:operandTags,
     out:child.outputIsParent?tagAxes(child.out,expectedAxes):child.out};
   });
  }
  const indexed=!isConcept&&INDEX_PROOFS.includes(kind)?{indexRule:{operator:kind,
   params:kind==='slice_rows'?{start:n.params.start,length:n.params.length}:
    kind==='slice_cols'?{start:n.params.start,width:n.params.width}:
    kind==='repeat_rows'?{times:n.params.times}:{}}}:null;
  if(indexed)indexed.indexMap=nativeIndexPlan({kind,operands,out,indexRule:indexed.indexRule});
  const s={id:r.id,kind,title:def.title,formula:def.formula,why:def.why,
   sourceLines:null,dependencies,out,operands,dot:specification,
   ...(numeric&&BINARY_PROOFS.includes(kind)?{elementwise:{operator:kind}}:{}),
   ...(numeric&&UNARY_PROOFS.includes(kind)?{unary:{operator:kind,
    ...(kind==='scale'?{factor:n.params.factor}:{}),...(kind==='bias'?{bias:n.params.bias}:{})}}:{}),
   ...(indexed||{}),
   ...(numeric&&['rmsnorm','silu','softmax'].includes(kind)?{kernel:{operator:kind,inputPort:'x',
    ...(kind==='rmsnorm'?{epsilon:n.params?.epsilon??def.params.epsilon}:{})}}:{}),substeps,
   note:isConcept?'仅静态推导，数值不可用':'执行器真实返回的小矩阵；未装载实际模型'};
  byId.set(s.id,s);return s;
 });
 return validate({schema:SCHEMA,title:graph.title||'可执行积木计算图',evidenceTier:isConcept?TIERS.STATIC:TIERS.MICRO,
  source:null,steps,origin:'learning_lab_executor',limits:'no_real_model_weights_or_hardware_profiling'});
}
// Independent numerical oracle for the pinned Wan attention-block FFN slice.
// Sources: Wan-Video/Wan2.2, commit 1ea34ff, wan/modules/model.py
// norm2 is non-affine LayerNorm; FFN is Linear→GELU(tanh)→Linear;
 // modulation chunk indices 3/4/5 and post-projection e5 gate matter.
function verifyWanArithmetic(data){
 const c=data.config,steps=data.steps,near=(a,b,label)=>{
  must(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(b)),label+' contradicts pinned FFN arithmetic');
 };
 must(c&&Number.isInteger(c.tokens)&&c.tokens>=1&&c.tokens<=8&&Number.isInteger(c.dim)&&c.dim>=2&&c.dim<=12&&
  Number.isInteger(c.hidden)&&c.hidden>=2&&c.hidden<=24&&
  [c.gate,c.scale,c.shift].every(x=>typeof x==='number'&&Number.isFinite(x)&&Math.abs(x)<=2),
  'invalid bounded Wan teaching parameters');
 const T=c.tokens,C=c.dim,H=c.hidden;
 const source=(name,n)=>{const a=c[name];must(Array.isArray(a)&&a.length===n&&a.every(Number.isFinite),'invalid Wan source '+name);return a;};
 const after=source('afterSelf',T*C),cross=source('crossDelta',T*C);
 const e0=source('e0',T*6*C),modulation=source('modulation',6*C);
 const w1=source('w1',H*C),b1=source('b1',H),w2=source('w2',C*H),b2=source('b2',C);
 must(Array.isArray(data.e)&&data.e.length===6,'missing six Wan conditioning branches');
 for(let g=0;g<6;g++){
  const branch=data.e[g];must(Array.isArray(branch)&&branch.length===T*C,'invalid Wan conditioning branch '+g);
  for(let t=0;t<T;t++)for(let d=0;d<C;d++)
   near(branch[t*C+d],e0[t*6*C+g*C+d]+modulation[g*C+d],'e['+g+']');
 }
 const dims=[[T,6*C],[T,C],[T,C],[T,C],[T,H],[T,H],[T,C],[T,C],[T,C]];
 const val=steps.map((step,i)=>{
  must(step&&step.id===['condition','cross','norm','affine','up','gelu','down','gate','residual'][i],
   'unexpected Wan arithmetic step order');
  must(step.main&&step.main.shape.join(',')===dims[i].join(','),'unexpected Wan stage shape');
  numericTensor(step.main.shape,step.main.values);
  return step.main.values;
 });
 for(let t=0;t<T;t++){
  for(let g=0;g<6;g++)for(let d=0;d<C;d++)
   near(val[0][t*6*C+g*C+d],data.e[g][t*C+d],'condition');
  let mean=0;for(let d=0;d<C;d++)mean+=val[1][t*C+d];mean/=C;
  let variance=0;for(let d=0;d<C;d++)variance+=(val[1][t*C+d]-mean)**2;variance/=C;
  const inv=1/Math.sqrt(variance+1e-6);
  for(let d=0;d<C;d++){
   const i=t*C+d;
   near(val[1][i],after[i]+cross[i],'cross');
   near(val[2][i],(val[1][i]-mean)*inv,'LayerNorm');
   near(val[3][i],val[2][i]*(1+c.scale*data.e[4][i])+c.shift*data.e[3][i],'FFN affine');
  }
  for(let h=0;h<H;h++){
   let up=b1[h];for(let d=0;d<C;d++)up+=val[3][t*C+d]*w1[h*C+d];
   const v=val[4][t*H+h];near(v,up,'Linear1');
   const q=Math.sqrt(2/Math.PI)*(v+.044715*v*v*v);
   near(val[5][t*H+h],.5*v*(1+Math.tanh(q)),'GELU tanh');
  }
  for(let d=0;d<C;d++){
   const i=t*C+d;
   let down=b2[d];for(let h=0;h<H;h++)down+=val[5][t*H+h]*w2[d*H+h];
   near(val[6][i],down,'Linear2');
   near(val[7][i],val[6][i]*c.gate*data.e[5][i],'e5 output gate');
   near(val[8][i],val[1][i]+val[7][i],'final residual');
  }
 }
 const matchAux=(stage,port,shape,expected)=>{
  const tensor=steps[stage].aux?.[port];
  must(tensor&&tensor.shape.join(',')===shape.join(',')&&Array.isArray(tensor.values)&&
   tensor.values.length===expected.length,'invalid Wan auxiliary operand '+stage+'.'+port);
  tensor.values.forEach((v,i)=>near(v,expected[i],'Wan auxiliary '+stage+'.'+port));
 };
 matchAux(0,'shift',[T,C],data.e[3]);
 matchAux(0,'scale',[T,C],data.e[4]);
 matchAux(0,'gate',[T,C],data.e[5]);
 matchAux(1,'afterSelf',[T,C],after);
 matchAux(1,'crossDelta',[T,C],cross);
 matchAux(2,'crossInput',[T,C],val[1]);
 matchAux(3,'normalized',[T,C],val[2]);
 matchAux(3,'scale',[T,C],data.e[4]);
 matchAux(3,'shift',[T,C],data.e[3]);
 matchAux(4,'input',[T,C],val[3]);
 matchAux(4,'weights',[H,C],w1);
 matchAux(4,'bias',[1,H],b1);
 matchAux(5,'preActivation',[T,H],val[4]);
 matchAux(6,'input',[T,H],val[5]);
 matchAux(6,'weights',[C,H],w2);
 matchAux(6,'bias',[1,C],b2);
 matchAux(7,'ffn',[T,C],val[6]);
 matchAux(7,'gate',[T,C],data.e[5]);
 matchAux(8,'crossInput',[T,C],val[1]);
 matchAux(8,'delta',[T,C],val[7]);
 must(data.final&&data.final.shape.join(',')===dims[8].join(',')&&Array.isArray(data.final.values)&&
  data.final.values.length===val[8].length,'missing final output contract');
 data.final.values.forEach((v,i)=>near(v,val[8][i],'final observation'));
}
function wan(data){
 must(data&&data.source&&Array.isArray(data.steps)&&data.steps.length===9,'missing pinned Wan FFN trace');
 const ref=data.source,srcUrl=ref.source;
 must(ref.commit==='1ea34ff48f87168174e12956e200b1d908b1c5ff','unsupported Wan source revision');
 must(ref.repo==='Wan-Video/Wan2.2'&&ref.path==='wan/modules/model.py'&&ref.config==='wan/configs/wan_i2v_A14B.py'&&
  ref.dim===5120&&ref.ffnDim===13824&&ref.layers===40&&ref.heads===40&&ref.eps===1e-6&&
  srcUrl==='https://github.com/Wan-Video/Wan2.2/blob/'+ref.commit+'/wan/modules/model.py',
  'pinned Wan source/config identity mismatch');
 must(data.evidence==='source-pinned-structure_plus_synthetic_float64_not_original_wan_weights','Wan evidence label mismatch');
 verifyWanArithmetic(data);
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