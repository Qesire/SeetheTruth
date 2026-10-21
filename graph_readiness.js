/* Permissive, incremental graph readiness check.
 * No architecture whitelist: any catalog blocks may be wired into a DAG.
 * Incomplete edits are retained; only executable graphs must satisfy each
 * operator's tensor-shape contract and all required input ports.
 */
(function(root,factory){const api=factory(root);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthGraphReadiness=api;})(typeof window!=="undefined"?window:{},function(root){"use strict";
const L=typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning;
const C=typeof module==="object"&&module.exports?require("./concept_flow.js"):root.SeetheTruthConceptFlow;
const Branch=typeof module==="object"&&module.exports?require("./branch_execution.js"):root.SeetheTruthBranchExecution;
function check(raw,options={}){
 let graph,seq;
 try{graph=options.targetId?Branch.slice(raw,options.targetId):L.validateGraph(raw);seq=L.order(graph);}
 catch(err){return {executable:false,structuralError:err.message,firstError:{nodeId:err.nodeId||null,reason:err.message},nodes:[],readyCount:0,missingCount:0,errorCount:1,blockedCount:0};}
 const results=new Map(),edges=new Map(graph.edges.map(e=>[e.to+":"+e.port,e.from]));
 let missingCount=0,errorCount=0,blockedCount=0,readyCount=0;
 for(const n of seq){
   const inputs={},logicalStates={},missing=[],unavailable=[];
   for(const port of L.defs[n.kind].ports){
     const source=edges.get(n.id+":"+port);
     if(!source){missing.push(port);continue;}
     const prior=results.get(source);
     if(prior?.status!=="ready"){unavailable.push({port,source});continue;}
     inputs[port]=prior.shape;
     logicalStates[port]=prior.logicalState||null;
   }
   let result;
   if(missing.length){
     result={id:n.id,kind:n.kind,status:"missing",shape:null,reason:"缺少必要输入："+missing.join(", "),ports:missing,unavailable};
     missingCount+=missing.length;
   }else if(unavailable.length){
     result={id:n.id,kind:n.kind,status:"blocked",shape:null,
       reason:"等待上游："+unavailable.map(x=>x.source+" → "+x.port).join("；"),ports:[],unavailable};
     blockedCount++;
   }else{
     try{
       const out=C.analyze(n,inputs,{logicalStates});
       result={id:n.id,kind:n.kind,status:"ready",shape:out.shape.slice(),reason:null,ports:[],unavailable:[],rule:out.rule};
       if(out.logicalState)result.logicalState=out.logicalState;
       readyCount++;
     }catch(err){
       result={id:n.id,kind:n.kind,status:"shape-error",shape:null,reason:err.message,ports:[],unavailable:[]};
       errorCount++;
     }
   }
   results.set(n.id,result);
 }
 const list=seq.map(n=>results.get(n.id));
 const firstError=list.find(x=>x.status==="shape-error")||list.find(x=>x.status==="missing")||list.find(x=>x.status==="blocked")||null;
 const executable=list.length===readyCount;
 let numeric=null;
 if(executable&&options.numeric===true){
   try{const trace=L.evaluate(graph);numeric={ok:true,steps:trace.steps.length};}
   catch(err){numeric={ok:false,reason:err.message,nodeId:err.nodeId||null};}
 }
 return {executable:executable&&numeric?.ok!==false,structuralError:null,firstError:numeric?.ok===false?{nodeId:numeric.nodeId,reason:numeric.reason}:firstError,
    nodes:list,readyCount,missingCount,errorCount,blockedCount,numeric,
    targetId:options.targetId||null,scope:options.targetId?"selected_upstream_dependency_slice":"full_graph",
    note:"只要求图无环、输入完整且每块积木的真实张量/数值合同成立；不要求符合任何预设模型结构"};
}

/* Advisory only. Even an incompatible preview must never silently refuse wiring. */
function preview(raw,from,to,port){
 let before;
 try{before=check(raw);}catch(err){return {status:"invalid",reason:err.message};}
 const source=before.nodes.find(n=>n.id===from),previous=before.nodes.find(n=>n.id===to);
 if(!source||!previous)return {status:"invalid",reason:"source or destination not found"};
 if(source.status!=="ready")return {status:"pending",reason:"上游积木尚未得到合法形状",sourceShape:null,targetShape:null};
 let candidate;
 try{candidate=L.connect(raw,from,to,port);}
 catch(err){return {status:"invalid",reason:err.message,sourceShape:source.shape};}
 const checked=check(candidate),target=checked.nodes.find(n=>n.id===to);
 if(target?.status==="shape-error")return {status:"incompatible",reason:target.reason,sourceShape:source.shape,targetShape:null};
 // Existing downstream consumers may become incompatible after rewiring.
 const affected=new Set([to]),queue=[to];
 for(let i=0;i<queue.length;i++)for(const edge of candidate.edges){
   if(edge.from===queue[i]&&!affected.has(edge.to)){affected.add(edge.to);queue.push(edge.to);}
 }
 const descendants=checked.nodes.filter(n=>affected.has(n.id)&&n.status==="shape-error");
 if(descendants.length)return {status:"incompatible",reason:descendants[0].id+": "+descendants[0].reason,sourceShape:source.shape,targetShape:target?.shape||null};
 return {status:target?.status==="ready"?"compatible":"pending",reason:target?.reason||"已知维度匹配",
   sourceShape:source.shape,targetShape:target?.shape||null};
}
return Object.freeze({check,preview});
});
