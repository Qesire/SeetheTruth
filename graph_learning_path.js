/* Graph-derived learning route. No per-model assumptions or synthetic evidence promotion. */
(function(root,factory){"use strict";
 const node=typeof module==="object"&&module.exports;
 const api=factory(node?{
  L:require("./learning_lab.js"),S:require("./graph_session.js"),
  O:require("./operator_contracts.js"),K:require("./block_knowledge.js")
 }:{L:root.SeetheTruthLearning,S:root.SeetheTruthGraphSession,
  O:root.SeetheTruthOperatorContracts,K:root.SeetheTruthBlockKnowledge});
 if(node)module.exports=api;if(root)root.SeetheTruthGraphLearningPath=api;
})(typeof window!=="undefined"?window:null,function(d){"use strict";
 function preview(value){const out=[];
  function visit(x){if(out.length>=6)return;if(Array.isArray(x)){for(const y of x)visit(y);}
   else if(typeof x==="number"&&Number.isFinite(x))out.push(x);}
  visit(value);return Object.freeze(out);
 }
 function build(raw,options={}){
  const s=options.session||d.S.inspect(raw,{numeric:options.numeric===true});
  if(s.schema!=="seethetruth-graph-session-v1")throw Error("invalid session");
  let g,index;
  try{g=d.L.validateGraph(raw);index=d.L.indexGraph(g);}
  catch(e){return Object.freeze({schema:"seethetruth-graph-learning-path-v1",
   executable:false,numericOk:false,firstError:s.readiness?.structuralError||e.message,
   evidence:"NO_EXECUTION",steps:Object.freeze([])});}
  const nodes=new Map(s.nodes.map(n=>[n.id,n]));
  const trace=new Map((s.trace?.steps||[]).map(n=>[n.id,n]));
  const ir=new Map((s.ir?.nodes||[]).map(n=>[n.id,n]));
  const numericOk=!!(s.ir&&s.trace&&!s.error);
  const steps=s.nodes.map((n,i)=>{
   const op=d.O.resolve(n.kind),card=d.K.get(n.kind);
   if(!card)throw Error("missing knowledge "+n.kind);
   const ports=op.ports.map(port=>{
    const sourceId=index.incomingByPort.get(n.id+":"+port)||null,up=nodes.get(sourceId);
    return Object.freeze({port,sourceId,sourceShape:up?.shape?[...up.shape]:null,
     sourceStatus:up?.status||"missing"});
   });
   const seen=ir.get(n.id),t=trace.get(n.id);
   const observed=!!(numericOk&&n.numericObserved&&seen?.valueAvailable&&t?.values!==undefined);
   if(observed&&JSON.stringify(t.shape)!==JSON.stringify(n.shape))
    throw Error("static/numeric mismatch "+n.id);
   return Object.freeze({ordinal:i+1,id:n.id,kind:n.kind,title:op.title,
    status:n.status,reason:n.reason||null,ports:Object.freeze(ports),
    shape:n.shape?[...n.shape]:null,formula:op.formula,
    why:card.intuition,flow:card.flow,practice:card.practice,trap:card.trap,exercise:card.exercise,
    params:Object.freeze({...index.ordered[i].params}),
    evidence:observed?"SYNTHETIC_NUMERIC":"STATIC_ONLY",
    sample:observed?preview(t.values):null,
    numpyCandidate:op.backends.numpyEligible,torchCandidate:op.backends.torchEligible});
  });
  return Object.freeze({schema:"seethetruth-graph-learning-path-v1",
   executable:s.executable,numericOk,evidence:numericOk?"SYNTHETIC_NUMERIC":"STATIC_ONLY",
   firstError:s.error?.reason||s.readiness.firstError?.reason||null,
   steps:Object.freeze(steps)});
 }
 return Object.freeze({build});
});
