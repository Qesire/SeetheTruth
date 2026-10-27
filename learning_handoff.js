/* Pedagogical cross-page handoff of native A/B evidence to an existing graph lesson.
 * Every imported run and state cursor is restored through ObservationSnapshots.
 * No native state is treated as a tensor graph or a mutable engine checkpoint.
 */
(function(root,factory){"use strict";
 const isNode=typeof module==="object"&&module.exports;
 const api=factory(isNode?require("./observation_snapshots.js"):root.SeetheTruthObservationSnapshots);
 if(isNode)module.exports=api;
 if(root)root.SeetheTruthLearningHandoff=api;
})(typeof window!=="undefined"?window:null,function(S){"use strict";
 const SCHEMA="seethetruth-native-graph-handoff-v1";
 const KEY="seethetruth:native-graph-handoff-v1",LIMIT=180000;
 const target=Object.freeze({
  deepinfer_kv:Object.freeze({course:"decode",node:"gqa",question:"解码时读取的 K/V 前缀与物理块生命周期是什么关系？"}),
  deepinfer_scheduler:Object.freeze({course:"decode",node:"gqa",question:"调度准入如何决定哪些 token 可以进入 Attention 计算？"}),
  deepinfer_worker:Object.freeze({course:"attention",node:"attn",question:"依赖 DAG 的先后关系和 Attention 中 Q/K/V 的数据依赖有什么异同？"})
 });
 function fail(m){throw Error("learning handoff: "+m);}
 function bounded(x){
  const json=JSON.stringify(x);
  if(typeof json!=="string"||json.length>LIMIT)fail("cross-page teaching evidence exceeds size limit");
  return json;
 }
 function check(input){
  if(!input||typeof input!=="object"||Array.isArray(input))fail("invalid package");
  if(Object.keys(input).some(k=>!["schema","domain","active","baseline","question"].includes(k)))
   fail("unexpected package member");
  if(input.schema!==SCHEMA||!Object.hasOwn(target,input.domain))fail("unregistered native lab");
  if(typeof input.question!=="string"||input.question.length>300)fail("invalid learner question");
  if(!input.active||!input.active.cursor)fail("source event cursor is required");
  S.validate(input.active);
  if(input.baseline!==null&&input.baseline!==undefined)S.validate(input.baseline);
  bounded(input);
  return input;
 }
 function create(core,active,baseline=null,selectedId=null,question=""){
  if(!active||active.kind!=="specialized_execution"||!Object.hasOwn(target,active.id)||
     !active.stateTrace?.steps?.length)fail("replayable source state required");
  const snapshot=S.create(core,active,"native selected B");
  const stepId=selectedId&&active.stateTrace.steps.some(x=>x.id===selectedId)?
   selectedId:active.stateTrace.steps[0].id;
  const cursor=S.at(core,snapshot,stepId);
  const A=baseline?S.create(core,baseline,"native baseline A"):null;
  const payload={schema:SCHEMA,domain:active.id,active:cursor,baseline:A,
    question:String(question||"").trim().slice(0,300)};
  restore(core,payload);
  return Object.freeze(payload);
 }
 function restore(core,payload){
  const p=check(payload);
  if(p.active.kind!=="specialized_execution"||p.active.id!==p.domain||
     p.active.cursor.kind!=="state_transition")fail("source lab and event mismatch");
  const current=S.restore(core,p.active);
  const baseline=p.baseline?S.restore(core,p.baseline):null;
  if(baseline&&(baseline.record.id!==current.record.id||
      baseline.record.operation!==current.record.operation||
      baseline.record.engine!==current.record.engine||
      baseline.record.evidence!==current.record.evidence))
   fail("A and B use different native execution semantics");
  let comparison=null;
  if(baseline){
   comparison=core.compareRecords(baseline.record,current.record);
   if(!comparison.replayVerified||comparison.inputEqual)fail("independent A/B input replay required");
  }
  const lesson=target[p.domain];
  return Object.freeze({schema:SCHEMA,domain:p.domain,course:lesson.course,
   focusNode:lesson.node,question:p.question||lesson.question,sourceHref:current.sourceHref,
   current,baseline,comparison,selectedStepId:current.checkpoint.stepId,
   selectedStepIndex:current.checkpoint.index,evidence:current.evidence});
 }
 function store(storage,payload){
  const data=bounded(check(payload));
  if(!storage||typeof storage.setItem!=="function")fail("per-tab teaching storage unavailable");
  storage.setItem(KEY,data);
 }
 function load(storage){
  if(!storage||typeof storage.getItem!=="function")fail("per-tab teaching storage unavailable");
  const raw=storage.getItem(KEY);
  if(!raw)fail("no active learner-initiated handoff");
  if(raw.length>LIMIT)fail("cross-page payload exceeds size limit");
  let parsed;
  try{parsed=JSON.parse(raw);}catch(_){fail("invalid cross-page JSON");}
  return check(parsed);
 }

 // The mechanism lens is a read-only join of two independently executed
 // contracts. Native event dependencies are precedence relations, graph ports
 // are data dependencies; neither represents a cross-domain arithmetic edge.
 function explain(core,payload,currentGraph,numericRecord=null){
  const native=restore(core,payload);
  const graph=core.inspectGraph(currentGraph,{numeric:false,id:native.course});
  const node=graph.path.steps.find(step=>step.id===native.focusNode);
  const expectedKind=native.domain==="deepinfer_worker"?"attention":"gqa";
  if(!node||node.kind!==expectedKind)
   fail("current graph lesson no longer contains the intended Attention operation");
  const ports=node.ports.filter(p=>["q","k","v"].includes(p.port)).map(p=>Object.freeze({
   port:p.port,sourceId:p.sourceId,sourceShape:p.sourceShape,
   destinationId:node.id,destinationKind:node.kind
  }));
  if(ports.length!==3||ports.some(p=>!p.sourceId||!Array.isArray(p.sourceShape)))
   fail("the current lesson needs three actual Q/K/V source ports");
  let numeric=null;
  if(numericRecord){
   if(numericRecord.kind!=="graph"||numericRecord.numericOk!==true||
      JSON.stringify(numericRecord.graph)!==JSON.stringify(graph.graph))
    fail("numeric trace belongs to a different or non-executed graph");
   core.replay(numericRecord);
   const computed=numericRecord.path?.steps?.find(step=>step.id===native.focusNode);
   if(!computed||computed.evidence!=="SYNTHETIC_NUMERIC"||
      !numericRecord.session?.trace?.steps?.some(step=>step.id===node.id))
    fail("selected graph node lacks an accepted numeric calculation");
   numeric=Object.freeze({shape:computed.shape,sample:computed.sample,
    evidence:computed.evidence,observedNode:computed.id});
  }
  const step=native.current.record.stateTrace.steps[native.selectedStepIndex];
  if(!step||step.id!==native.selectedStepId)fail("native selected event changed");
  const changed=native.comparison?.changedSteps.some(item=>
   item.kind==="state_transition"&&item.stepId===step.id)??null;
  const details=Object.freeze({
   stepId:step.id,title:step.title,
   dependencies:Object.freeze(step.dependencies.slice()),
   changes:Object.freeze(step.changes.map(item=>item.key)),
   timing:step.timing?Object.freeze({...step.timing}):null,
   selectedStepChangedInAB:changed,
   totalChangedNativeSteps:native.comparison?.changedSteps.length??null
  });
  const topic=native.domain==="deepinfer_worker"
   ?"Worker 的边是事件先后约束；Q/K/V 的边是张量输入依赖。试比较两者的可并行条件。"
   :native.domain==="deepinfer_kv"
   ?"物理块队列与引用计数管理 K/V 存储所有权；图中的 k/v 是逻辑张量数据输入。试区分存储地址与内容。"
   :"Scheduler 的准入决定请求能否进入 Forward；图的 Q/K/V 端口说明已准入的一次 Attention 如何计算。试区分调度与算术。";
  return Object.freeze({
   schema:"seethetruth-native-graph-mechanism-v1",domain:native.domain,
   course:native.course,question:native.question,sourceEvidence:native.evidence,
   native:details,graph:Object.freeze({id:graph.id,kind:node.kind,
    nodeId:node.id,shape:node.shape,
    ports:Object.freeze(ports),evidence:numeric?"SYNTHETIC_NUMERIC":"STATIC_ONLY",
    numeric}),
   contrast:topic,
   status:numeric?"REPLAYED_NATIVE_AND_GRAPH_NUMERIC":"REPLAYED_NATIVE_AND_GRAPH_STATIC"
  });
 }

 // Per-coordinate evidence comes from a *replayed* original graph execution.
 // We expose P[t,k]*V[k,c] as an output identity check, not another attention
 // engine. Native scheduler/KV events remain distinct from tensor arithmetic.
 function coordinate(core,packet,graph,numericRecord,selection={}){
  if(!numericRecord||numericRecord.kind!=="graph"||numericRecord.numericOk!==true)
   fail("accepted numeric run required");
  const lens=explain(core,packet,graph,numericRecord);
  const nodeId=lens.graph.nodeId,kind=lens.graph.kind;
  const trace=numericRecord.session?.trace?.steps?.find(x=>x.id===nodeId);
  if(!trace||!Array.isArray(trace.values)||!trace.internals)
   fail("original attention intermediate tensors unavailable");
  const outputShape=lens.graph.numeric.shape;
  if(!Array.isArray(outputShape)||outputShape.length!==2)fail("rank-2 output expected");
  const bound=(value,max,label)=>{
   if(!Number.isSafeInteger(value)||value<0||value>=max)fail(label+" out of bounds");
   return value;
  };
  const get2=(matrix,row,col,label)=>{
   const value=matrix?.[row]?.[col];
   if(!Number.isFinite(value))fail(label+" source coordinate unavailable");
   return value;
  };
  let heads=1,kvHeads=1,valueWidth=outputShape[1],keyCount=0,group=1,
      scoreAt,probAt,valueAt,excluded=()=>false,contextAt=null;
  if(kind==="attention"){
   const probs=trace.internals.probabilities?.values;
   const scores=trace.internals.scores?.values;
   keyCount=probs?.[0]?.length||0;
   const valueSource=lens.graph.ports.find(p=>p.port==="v").sourceId;
   const values=numericRecord.session.trace.steps.find(x=>x.id===valueSource)?.values;
   if(!Array.isArray(values)||values.length!==keyCount||!Array.isArray(scores)||
      values[0]?.length!==valueWidth)fail("attention P/V/output shape mismatch");
   scoreAt=(t,h,k)=>get2(scores,t,k,"score");
   probAt=(t,h,k)=>get2(probs,t,k,"probability");
   valueAt=(h,k,c)=>get2(values,k,c,"value");
  }else if(kind==="gqa"){
   const probs=trace.internals.probabilities?.tensor;
   const scores=trace.internals.scores?.tensor;
   const values=trace.internals.v_heads?.tensor;
   const context=trace.internals.context_heads?.tensor;
   const mask=trace.internals.causal_mask?.values||null;
   if(!probs||!scores||!values||!context||
      probs.shape?.length!==4||values.shape?.length!==4||
      context.shape?.length!==4||probs.shape[0]!==1)
    fail("native GQA head intermediates unavailable");
   heads=probs.shape[1];keyCount=probs.shape[3];
   kvHeads=values.shape[1];valueWidth=context.shape[3];group=heads/kvHeads;
   if(!Number.isSafeInteger(group)||group<1||values.shape[2]!==keyCount||
      scores.shape.join(",")!==probs.shape.join(",")||
      context.shape[1]!==heads||context.shape[2]!==outputShape[0]||
      outputShape[1]!==heads*valueWidth)fail("native GQA grouped-head shape mismatch");
   const get4=(tensor,h,t,c,label)=>{
    const idx=((h*tensor.shape[2]+t)*tensor.shape[3])+c;
    const value=tensor.data?.[idx];
    if(!Number.isFinite(value))fail(label+" native GQA coordinate unavailable");
    return value;
   };
   scoreAt=(t,h,k)=>get4(scores,h,t,k,"score");
   probAt=(t,h,k)=>get4(probs,h,t,k,"probability");
   valueAt=(h,k,c)=>get4(values,Math.floor(h/group),k,c,"Value");
   contextAt=(t,h,c)=>get4(context,h,t,c,"Context");
   excluded=(t,k)=>mask?get2(mask,t,k,"causal mask")===0:false;
  }else fail("unsupported Attention kind");
  const query=bound(selection.queryToken??0,outputShape[0],"query token");
  const head=bound(selection.head??0,heads,"query head");
  const channel=bound(selection.channel??0,valueWidth,"head output channel");
  if(keyCount<1||keyCount>64||keyCount*valueWidth>4096)
   fail("bounded original key-token limit exceeded");
  const outputChannel=head*valueWidth+channel;
  const observed=get2(trace.values,query,outputChannel,"Attention output");
  const terms=[];
  for(let key=0;key<keyCount;key++){
   const score=scoreAt(query,head,key);
   const probability=probAt(query,head,key);
   const value=valueAt(head,key,channel);
   const masked=excluded(query,key);
   if(probability<0||probability>1+1e-12||(masked&&probability!==0))
    fail("source probability / mask mismatch");
   terms.push(Object.freeze({key,score,probability,value,
    contribution:probability*value,masked}));
  }
  const sum=terms.reduce((x,y)=>x+y.contribution,0);
  const probabilitySum=terms.reduce((x,y)=>x+y.probability,0);
  const error=Math.abs(sum-observed);
  if(Math.abs(probabilitySum-1)>1e-10||
     error>1e-10*Math.max(1,Math.abs(observed))||
     (contextAt&&Math.abs(contextAt(query,head,channel)-observed)>1e-10))
   fail("source P×V decomposition diverges from accepted output");
  // Independently request the selected *parent node* from the existing
  // coordinate microscope. Substeps are consumed internally by that UI; the
  // shared Core inspector accepts real graph node identities only.
  const proof=core.inspectObservationStep(numericRecord,{
   kind:"tensor_computation",stepId:nodeId,
   coordinate:[query,outputChannel]});
  if(!proof.replayVerified||!Number.isFinite(proof.detail?.value)||
     Math.abs(proof.detail.value-observed)>1e-12)
   fail("original graph coordinate microscope mismatch");
  const originalSubstep=Object.freeze({id:nodeId,replayVerified:true,
   coordinateValue:proof.detail.value});
  return Object.freeze({
   schema:"seethetruth-attention-coordinate-witness-v1",domain:lens.domain,
   nativeStepId:lens.native.stepId,nodeId,queryToken:query,queryHead:head,
   kvHead:Math.floor(head/group),headCount:heads,kvHeadCount:kvHeads,
   outputChannel,headChannel:channel,outputShape:Object.freeze([...outputShape]),
   keyCount,terms:Object.freeze(terms),observed,recomputed:sum,error,
   probabilitySum,originalSubstep,sourceEvidence:lens.sourceEvidence,
   numericalEvidence:"SYNTHETIC_NUMERIC_REPLAY_VERIFIED",
   note:"Output coordinate decomposed from accepted teaching P and V; physical cache addresses and GPU timing are separate."});
 }

 // Score microscope: use original unpositioned/positioned Q/K evidence from
 // the same accepted run. All arithmetic below verifies recorded stages,
 // including RoPE pairs, scaled dot products, mask, and stable Softmax.
 function score(core,packet,graph,numericRecord,selection={}){
  const witness=coordinate(core,packet,graph,numericRecord,selection);
  const key=selection.keyToken??0;
  if(!Number.isSafeInteger(key)||key<0||key>=witness.keyCount)
   fail("key token out of bounds");
  const trace=numericRecord.session.trace.steps;
  const stage=trace.find(s=>s.id===witness.nodeId);
  const kind=stage.kind,query=witness.queryToken,head=witness.queryHead;
  const kvHead=witness.kvHead,params=numericRecord.graph.nodes.find(
   n=>n.id===witness.nodeId)?.params||{};
  const originalValue=(matrix,t,d,which)=>{
   const n=matrix?.[t]?.[d];
   if(!Number.isFinite(n))fail(which+" original value unavailable");
   return n;
  };
  let dim,qOriginal,kOriginal,qPositioned,kPositioned,rotary=false,qOffset=0;
  if(kind==="attention"){
   const ports=Object.fromEntries(graph.edges.filter(e=>e.to===witness.nodeId)
    .map(e=>[e.port,e.from]));
   const getSource=(port,t,d)=>originalValue(
    trace.find(s=>s.id===ports[port])?.values,t,d,port);
   const q=trace.find(s=>s.id===ports.q)?.values;
   dim=q?.[0]?.length;
   if(!Number.isSafeInteger(dim)||dim<1||dim>64)fail("Attention score head width unavailable");
   const k=trace.find(s=>s.id===ports.k)?.values;
   if(k?.[0]?.length!==dim)fail("Attention Q/K score width mismatch");
   qOriginal=d=>getSource("q",query,d);kOriginal=d=>getSource("k",key,d);
   qPositioned=qOriginal;kPositioned=kOriginal;
  }else if(kind==="gqa"){
   const internal=stage.internals;
   const fetch=name=>{
    const t=internal?.[name]?.tensor;
    if(!t||t.shape?.length!==4||t.shape[0]!==1||
       !Array.isArray(t.data))fail("missing original GQA stage "+name);
    return t;
   };
   const qo=fetch("q_heads"),ko=fetch("k_heads");
   const qp=fetch("q_positioned"),kp=fetch("k_positioned");
   dim=qp.shape[3];rotary=params.rope===true;
   qOffset=params.causal===true?witness.keyCount-witness.outputShape[0]:0;
   if(!Number.isSafeInteger(dim)||dim<1||dim>64||
      qo.shape.join(",")!==qp.shape.join(",")||
      ko.shape.join(",")!==kp.shape.join(",")||
      qo.shape[1]!==witness.headCount||
      ko.shape[1]!==witness.kvHeadCount||
      ko.shape[2]!==witness.keyCount||dim!==kp.shape[3]||
      qOffset<0)fail("GQA source staged score shape mismatch");
   const get=(tensor,h,t,d,label)=>{
    const idx=(h*tensor.shape[2]+t)*dim+d;
    const val=tensor.data[idx];
    if(!Number.isFinite(val))fail("GQA "+label+" value unavailable");
    return val;
   };
   qOriginal=d=>get(qo,head,query,d,"original Q");
   kOriginal=d=>get(ko,kvHead,key,d,"original K");
   qPositioned=d=>get(qp,head,query,d,"positioned Q");
   kPositioned=d=>get(kp,kvHead,key,d,"positioned K");
  }else fail("unsupported score microscope kind");
  const near=(actual,expected,label)=>{
   if(!Number.isFinite(actual)||!Number.isFinite(expected)||
      Math.abs(actual-expected)>1e-10*Math.max(1,Math.abs(expected)))
    fail("replayed "+label+" differs from source stage");
  };
  const scale=1/Math.sqrt(dim),features=[];
  if(rotary&&dim%2)fail("RoPE requires even head dimension");
  for(let d=0;d<dim;d++){
   const rawQ=qOriginal(d),rawK=kOriginal(d);
   const positionedQ=qPositioned(d),positionedK=kPositioned(d);
   const pair=d-d%2;
   const qAngle=rotary?(query+qOffset)/Math.pow(10000,pair/dim):0;
   const kAngle=rotary?key/Math.pow(10000,pair/dim):0;
   if(rotary){
    const q0=qOriginal(pair),q1=qOriginal(pair+1);
    const k0=kOriginal(pair),k1=kOriginal(pair+1);
    near(positionedQ,d%2===0?
     q0*Math.cos(qAngle)-q1*Math.sin(qAngle):
     q0*Math.sin(qAngle)+q1*Math.cos(qAngle),"Q RoPE pair");
    near(positionedK,d%2===0?
     k0*Math.cos(kAngle)-k1*Math.sin(kAngle):
     k0*Math.sin(kAngle)+k1*Math.cos(kAngle),"K RoPE pair");
   }else{
    near(positionedQ,rawQ,"unrotated Q");
    near(positionedK,rawK,"unrotated K");
   }
   features.push(Object.freeze({dimension:d,rawQ,rawK,positionedQ,
    positionedK,qAngle,kAngle,product:positionedQ*positionedK,
    scaledContribution:positionedQ*positionedK*scale}));
  }
  const recomputed=features.reduce((sum,t)=>sum+t.scaledContribution,0);
  const selected=witness.terms[key];
  near(recomputed,selected.score,"QK scaled score");
  const visible=witness.terms.filter(t=>!t.masked);
  if(!visible.length)fail("all keys masked");
  const rowMax=Math.max(...visible.map(t=>t.score));
  const denominator=visible.reduce((sum,t)=>sum+Math.exp(t.score-rowMax),0);
  const numerator=selected.masked?0:Math.exp(selected.score-rowMax);
  const softmaxProbability=numerator/denominator;
  near(softmaxProbability,selected.probability,"stable Softmax");
  return Object.freeze({schema:"seethetruth-attention-score-witness-v1",
   nodeId:witness.nodeId,queryToken:query,keyToken:key,queryHead:head,
   kvHead,headWidth:dim,queryPositionOffset:qOffset,rotary,scale,
   sourceScore:selected.score,recomputedScore:recomputed,
   scoreError:Math.abs(recomputed-selected.score),
   features:Object.freeze(features),masked:selected.masked,
   probability:selected.probability,softmax:Object.freeze({rowMax,
    numerator,denominator,recomputed:softmaxProbability,
    error:Math.abs(softmaxProbability-selected.probability)}),
   numericalEvidence:"SYNTHETIC_NUMERIC_REPLAY_VERIFIED",
   nativeStepId:witness.nativeStepId,
   note:"Q/K source stages and RoPE transformation verified against original numerical run; not GPU or physical KV cache."});
 }

 // Controlled Q/K intervention. A and B execute identical graph topology with
 // an explicit Parameter at precisely one Attention port. A is required to
 // reproduce the untouched source graph's Attention output. B changes exactly
 // one Parameter coordinate. This is not a trained projection-weight update.
 function intervene(core,packet,currentGraph,numericRecord,options={}){
  if(!numericRecord?.numericOk||numericRecord.kind!=="graph")
   fail("a replayed numeric graph is required before intervention");
  const relation=explain(core,packet,currentGraph,numericRecord);
  const port=options.port;
  if(port!=="q"&&port!=="k")fail("intervention port must be Q or K");
  const source=relation.graph.ports.find(p=>p.port===port);
  const trace=numericRecord.session.trace.steps;
  const values=trace.find(step=>step.id===source.sourceId)?.values;
  if(!Array.isArray(values)||!values.length||!Array.isArray(values[0])||
     values.length>8||values[0].length>16||
     values.some(row=>!Array.isArray(row)||row.length!==values[0].length||
      row.some(v=>!Number.isFinite(v)||Math.abs(v)>1000)))
   fail("Q/K projection output cannot be frozen as a bounded Parameter");
  const token=options.sourceToken??0,feature=options.feature??0,delta=options.delta;
  if(!Number.isSafeInteger(token)||token<0||token>=values.length)
   fail("source token coordinate out of bounds");
  if(!Number.isSafeInteger(feature)||feature<0||feature>=values[0].length)
   fail("source feature coordinate out of bounds");
  if(typeof delta!=="number"||!Number.isFinite(delta)||Math.abs(delta)>1||delta===0)
   fail("intervention delta must be finite, nonzero, magnitude <= 1");
  if(Math.abs(values[token][feature]+delta)>1000||
     values[token][feature]+delta===values[token][feature])
   fail("intervention cannot be represented in teaching Float64 bounds");
  const observed=core.replay(numericRecord);
  if(!observed.reproduced)fail("source graph replay failed");
  const id="ab_"+port+"_probe";
  if(currentGraph.nodes.some(n=>n.id===id))
   fail("reserved intervention Parameter node ID already exists");
  const originalPortEdges=currentGraph.edges.filter(e=>e.to===relation.graph.nodeId&&e.port===port);
  if(originalPortEdges.length!==1||originalPortEdges[0].from!==source.sourceId)
   fail("Q/K incoming source changed");
  const matrix=values.map(row=>row.slice());
  const zero=matrix.map(row=>row.map(()=>0));
  const aGraph=JSON.parse(JSON.stringify(numericRecord.graph));
  aGraph.title=(aGraph.title||"Attention")+" · 单端口定值 A";
  aGraph.nodes.push({id,kind:"parameter",x:210,y:750,
   params:{values:matrix,optimizer:{t:0,m:zero,v:zero.map(row=>row.slice())}}});
  aGraph.edges=aGraph.edges.map(e=>e.to===relation.graph.nodeId&&e.port===port?
   {from:id,to:e.to,port:e.port}:e);
  const bGraph=JSON.parse(JSON.stringify(aGraph));
  bGraph.title=aGraph.title; // same A/B title to keep changes structural-free
  const bMatrix=bGraph.nodes.find(n=>n.id===id).params.values;
  const previous=bMatrix[token][feature];
  bMatrix[token][feature]=previous+delta;
  const baseline=core.inspectGraph(aGraph,{numeric:true,id:relation.course});
  const changed=core.inspectGraph(bGraph,{numeric:true,id:relation.course});
  if(!baseline.numericOk||!changed.numericOk)
   fail("intervention A/B graph cannot execute in accepted teaching runtime");
  const aTrace=baseline.session.trace.steps,bTrace=changed.session.trace.steps;
  const original=trace.find(step=>step.id===relation.graph.nodeId);
  const a=aTrace.find(step=>step.id===relation.graph.nodeId);
  const b=bTrace.find(step=>step.id===relation.graph.nodeId);
  const distance=(left,right)=>{
   if(!Array.isArray(left)||!Array.isArray(right)||left.length!==right.length)
    fail("intervention output shape changed");
   let max=0,changedCoords=0;
   for(let t=0;t<left.length;t++){
    if(!Array.isArray(left[t])||!Array.isArray(right[t])||
       left[t].length!==right[t].length)fail("intervention output width changed");
    for(let c=0;c<left[t].length;c++){
     const diff=Math.abs(left[t][c]-right[t][c]);
     if(!Number.isFinite(diff))fail("intervention output is nonfinite");
     max=Math.max(max,diff);if(diff>1e-12)changedCoords++;
    }
   }
   return Object.freeze({maxAbsolute:max,changedCoordinates:changedCoords});
  };
  // A freezes a port yet reproduces every existing graph step's output
  // and original intermediate stages, not merely the selected Attention node.
  for(const oldStep of trace){
   const now=aTrace.find(step=>step.id===oldStep.id);
   if(!now||JSON.stringify(now.values)!==JSON.stringify(oldStep.values)||
      JSON.stringify(now.internals??null)!==JSON.stringify(oldStep.internals??null))
    fail("frozen A differs from original source execution: "+oldStep.id);
  }
  const restoredB=JSON.parse(JSON.stringify(bGraph));
  restoredB.nodes.find(n=>n.id===id).params.values[token][feature]=previous;
  if(JSON.stringify(restoredB)!==JSON.stringify(aGraph))
   fail("A/B differ beyond the specified single Parameter coordinate");
  const equivalence=distance(original.values,a.values);
  if(equivalence.maxAbsolute>1e-10)
   fail("frozen A input does not reproduce source Attention output");
  const comparison=core.compareRecords(baseline,changed);
  if(!comparison.replayVerified||comparison.inputEqual||
     !comparison.numericCompared)fail("Q/K A/B must be independent replayed numeric graphs");
  const readOnly=Object.freeze({
   baselineGraph:baseline,changedGraph:changed,
   graphChange:comparison,baselineEquivalent:equivalence,
   outputChange:distance(a.values,b.values),
   source:Object.freeze({port,from:source.sourceId,parameter:id,
    token,feature,before:previous,after:previous+delta,delta}),
   nativeEvent:relation.native.stepId
  });
  const selected={
   queryToken:options.queryToken??0,head:options.head??0,
   channel:options.channel??0,keyToken:options.keyToken??0
  };
  const attentionA=coordinate(core,packet,aGraph,baseline,selected);
  const attentionB=coordinate(core,packet,bGraph,changed,selected);
  const scoreA=score(core,packet,aGraph,baseline,selected);
  const scoreB=score(core,packet,bGraph,changed,selected);

  // Evidence-only propagation: use the two accepted graph traces and the
  // already replayed original score/coordinate witnesses. No inferred
  // cross-domain event-to-tensor edge and no second Attention execution.
  const tolerance=1e-12,near=(lhs,rhs,label)=>{
   if(!Number.isFinite(lhs)||!Number.isFinite(rhs)||
      Math.abs(lhs-rhs)>1e-10*Math.max(1,Math.abs(rhs)))
    fail("causal propagation witness mismatch: "+label);
  };
  const graphFlow=[];
  if(aTrace.length!==bTrace.length)fail("A/B trace topology mismatch");
  const BById=new Map(bTrace.map(step=>[step.id,step]));
  for(const step of aTrace){
   const other=BById.get(step.id);
   if(!other||other.kind!==step.kind||
      JSON.stringify(step.shape)!==JSON.stringify(other.shape))
    fail("A/B graph step identity/shape changed");
   const aValues=step.values,bValues=other.values;
   if(!Array.isArray(aValues)||!Array.isArray(bValues)||aValues.length!==bValues.length)
    fail("A/B execution stage has incompatible values");
   let maximum=0,changedCount=0;
   const examples=[];
   for(let t=0;t<aValues.length;t++){
    if(!Array.isArray(aValues[t])||!Array.isArray(bValues[t])||
       aValues[t].length!==bValues[t].length)
     fail("A/B stage tensor rank mismatch");
    for(let c=0;c<aValues[t].length;c++){
     const before=aValues[t][c],after=bValues[t][c],diff=after-before;
     if(!Number.isFinite(diff))fail("nonfinite A/B propagation delta");
     maximum=Math.max(maximum,Math.abs(diff));
     if(Math.abs(diff)>tolerance){
      changedCount++;
      if(examples.length<6)examples.push(Object.freeze({token:t,feature:c,before,after,delta:diff}));
     }
    }
   }
   graphFlow.push(Object.freeze({id:step.id,kind:step.kind,
    changedCoordinates:changedCount,maxAbsolute:maximum,
    changed:changedCount>0,examples:Object.freeze(examples)}));
  }

  // Convert observed deltas into a grounded causal *subgraph*: nothing
  // outside a directed descendant of the one changed Parameter may change.
  // Include actual port-labelled DAG edges so each explanation is navigable.
  const reachable=new Set([id]),byStep=new Map(graphFlow.map(step=>[step.id,step]));
  for(const step of graphFlow){
   if(aGraph.edges.some(edge=>edge.to===step.id&&reachable.has(edge.from)))
    reachable.add(step.id);
  }
  for(const step of graphFlow){
   if(step.changed&&!reachable.has(step.id))
    fail("an unrelated non-descendant graph node changed during intervention: "+step.id);
  }
  const graphRoutes=graphFlow.map(step=>Object.freeze({...step,
   reachable:reachable.has(step.id),
   incoming:Object.freeze(aGraph.edges.filter(edge=>edge.to===step.id).map(edge=>
    Object.freeze({from:edge.from,port:edge.port,
     sourceChanged:byStep.get(edge.from)?.changed===true})))
  }));
  const nativeNode=graphFlow.find(step=>step.id===relation.graph.nodeId);
  if(!nativeNode||nativeNode.changedCoordinates!==readOnly.outputChange.changedCoordinates)
   fail("Attention node difference disagrees with full graph trace");
  const sourceNode=graphFlow.find(step=>step.id===id);
  if(!sourceNode||sourceNode.changedCoordinates!==1)
   fail("intervention source must differ at exactly one coordinate");
  const originalNodes=new Set(trace.map(step=>step.id));
  if(graphFlow.some(step=>step.id!==id&&!originalNodes.has(step.id)))
   fail("unexpected extra execution stage");
  const stage=(label,before,after)=>Object.freeze({
   label,before,after,delta:after-before,changed:Math.abs(after-before)>tolerance});
  const allKeys=attentionA.terms.map((term,i)=>{
   const other=attentionB.terms[i];
   if(!other||other.key!==term.key||term.masked!==other.masked)
    fail("A/B changed key ordering or causal visibility");
   if(term.masked&&(term.probability!==0||other.probability!==0))
    fail("masked future key acquired probability");
   const scoreDelta=other.score-term.score,
    probabilityDelta=other.probability-term.probability,
    valueDelta=other.value-term.value,
    contributionDelta=other.contribution-term.contribution;
   if(![scoreDelta,probabilityDelta,valueDelta,contributionDelta].every(Number.isFinite))
    fail("nonfinite per-key A/B contribution");
   return Object.freeze({key:term.key,masked:term.masked,
    score:stage("Score",term.score,other.score),
    probability:stage("Softmax P",term.probability,other.probability),
    value:stage("Value",term.value,other.value),
    contribution:stage("P×V",term.contribution,other.contribution)});
  });
  const features=scoreA.features.map((item,i)=>{
   const other=scoreB.features[i];
   if(!other||other.dimension!==item.dimension)
    fail("A/B score feature index changed");
   return Object.freeze({dimension:item.dimension,
    q:stage("Q positioned",item.positionedQ,other.positionedQ),
    k:stage("K positioned",item.positionedK,other.positionedK),
    dot:stage("QK/√Dk",item.scaledContribution,other.scaledContribution)});
  });
  near(features.reduce((sum,item)=>sum+item.dot.delta,0),
   scoreB.sourceScore-scoreA.sourceScore,"feature Score sum");
  near(allKeys.reduce((sum,item)=>sum+item.probability.delta,0),
   0,"Softmax row normalization delta");
  near(allKeys.reduce((sum,item)=>sum+item.contribution.delta,0),
   attentionB.observed-attentionA.observed,"P×V to output delta");
  const scoreStage=stage("QK scaled Score",scoreA.sourceScore,scoreB.sourceScore);
  const softmaxStage=stage("selected P",scoreA.probability,scoreB.probability);
  const outputStage=stage("selected Output",attentionA.observed,attentionB.observed);
  const rowScoreChanged=allKeys.filter(item=>item.score.changed).length;
  const rowProbabilityChanged=allKeys.filter(item=>item.probability.changed).length;
  const rowContributionChanged=allKeys.filter(item=>item.contribution.changed).length;
  const propagation=Object.freeze({
   schema:"seethetruth-attention-causal-propagation-v1",
   tolerance,nativeEventId:relation.native.stepId,
   observed: "REPLAYED_GRAPH_PARAMETER_AB",
   featureTerms:Object.freeze(features),keys:Object.freeze(allKeys),
   score:scoreStage,probability:softmaxStage,output:outputStage,
   row:Object.freeze({scoreChangedKeys:rowScoreChanged,
    probabilityChangedKeys:rowProbabilityChanged,
    contributionChangedKeys:rowContributionChanged,
    probabilitySumDelta:allKeys.reduce((sum,item)=>sum+item.probability.delta,0),
    reconstructedOutputDelta:allKeys.reduce((sum,item)=>sum+item.contribution.delta,0)}),
   graphSteps:Object.freeze(graphRoutes),
   causalReachableCount:reachable.size,
   changedOutsideCausalCone:0,
   maskedSelectedKey:!!attentionA.terms[selected.keyToken].masked,
   selectedKeySuppressed:!!attentionA.terms[selected.keyToken].masked&&
    !softmaxStage.changed,
   selectedHeadIsolated:!scoreStage.changed&&nativeNode.changed,
   note:"Graph stages use accepted A/B DAG traces; QK, Softmax and PV are decompositions of the same traced operation. Unchanged local result does not imply unchanged global output."
  });
  return Object.freeze({schema:"seethetruth-coordinate-intervention-v1",
   evidence:"REPLAYED_GRAPH_PARAMETER_AB",
   source:readOnly.source,sourceEquivalent:equivalence,
   baselineVerifiedSteps:trace.length,changedInputScalars:1,
   propagation,
   graphComparison:Object.freeze({replayVerified:comparison.replayVerified,
    inputEqual:comparison.inputEqual,changedSteps:comparison.changedSteps,
    numericCompared:comparison.numericCompared}),
   outputChange:readOnly.outputChange,
   selected:Object.freeze({queryToken:selected.queryToken,head:selected.head,
    channel:selected.channel,keyToken:selected.keyToken}),
   score:Object.freeze({before:scoreA.sourceScore,after:scoreB.sourceScore,
    delta:scoreB.sourceScore-scoreA.sourceScore}),
   probability:Object.freeze({before:scoreA.probability,after:scoreB.probability,
    delta:scoreB.probability-scoreA.probability}),
   output:Object.freeze({before:attentionA.observed,after:attentionB.observed,
    delta:attentionB.observed-attentionA.observed}),
   sourceEventId:readOnly.nativeEvent,
   explanation:"Parameter freezes one projected port; A reproduces original Q/K input, B changes one value. Independent native state A/B is unchanged."});
 }
 return Object.freeze({SCHEMA,KEY,LIMIT,create,restore,store,load,explain,coordinate,score,intervene});
});
