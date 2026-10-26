/* Shared export preparation for NumPy and Torch: DAG, ports, support and numeric trace.
 * Backend arithmetic and generated code remain intentionally independent.
 * Eligibility is not an executable-graph guarantee.
 */
(function(root,factory){"use strict";
 const isNode=typeof module==="object"&&module.exports;
 const api=factory(isNode?{
   learning:require("./learning_lab.js"),branch:require("./branch_execution.js"),
   execution:require("./execution_ir.js")
 }:{learning:root.SeetheTruthLearning,branch:root.SeetheTruthBranchExecution,
   execution:root.SeetheTruthExecutionIR});
 if(isNode)module.exports=api;
 if(root)root.SeetheTruthLoweringPlan=api;
})(typeof window!=="undefined"?window:null,function(deps){"use strict";
function prepare(raw,options={}){
 const {learning:L,branch:Branch,execution:Execution}=deps;
 if(!L||!Branch||!Execution)throw Error("missing graph lowering dependencies");
 const {targetId,supportedKinds,unsupportedMessage}=options;
 if(!supportedKinds||typeof supportedKinds[Symbol.iterator]!=="function")
   throw Error("graph lowering requires declared backend supported kinds");
 const kinds=new Set(supportedKinds);
 const graph=targetId?Branch.slice(raw,targetId):L.validateGraph(raw);
 const index=L.indexGraph(graph);
 const unsupported=index.ordered.filter(n=>!kinds.has(n.kind));
 if(unsupported.length){
   const description=typeof unsupportedMessage==="function"?
     unsupportedMessage(unsupported):"unsupported graph operators: "+unsupported.map(n=>n.id+"("+n.kind+")").join(", ");
   throw Error(description);
 }
 // Numerical validation happens only after unsupported-kind validation, matching
 // the exporters' original fail-closed precedence.
 const forward=L.evaluate(graph);
 // The same validated execution identity is consumed by GraphSession and
 // backend lowering. IR verifies full tensor nesting, every step's identity,
 // semantic shape transformations and canonical port source lineage.
 const executionIR=Execution.snapshot(graph,{trace:forward});
 const steps=new Map(forward.steps.map(s=>[s.id,s]));
 if(steps.size!==index.ordered.length||
    executionIR.nodes.length!==index.ordered.length||
    index.ordered.some((n,i)=>executionIR.nodes[i].id!==n.id||
      executionIR.nodes[i].kind!==n.kind||steps.get(n.id)?.kind!==n.kind||
      L.defs[n.kind].ports.some(p=>
       executionIR.nodes[i].inputSources[p]!==(
         index.incomingByPort.get(n.id+":"+p)||null))))
   throw Error("graph lowering IR/trace/port identity mismatch");
 return Object.freeze({graph,sorted:index.ordered,edges:index.incomingByPort,
   forward,steps,executionIR});
}
return Object.freeze({prepare});
});
