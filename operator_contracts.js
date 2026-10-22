/* Single read-only operator contract projection; never confuses eligibility with graph execution. */
(function(root,factory){
"use strict";
const isNode=typeof module==="object"&&module.exports;
const api=factory(isNode?{
 learning:require("./learning_lab.js"),
 capabilities:require("./block_capabilities.js"),
 knowledge:require("./block_knowledge.js")
}:{
 learning:root.SeetheTruthLearning,
 capabilities:root.SeetheTruthCapabilities,
 knowledge:root.SeetheTruthBlockKnowledge
});
if(isNode)module.exports=api;
if(root)root.SeetheTruthOperatorContracts=api;
})(typeof window!=="undefined"?window:null,function(deps){
"use strict";
function inventory(){
 const {learning,capabilities,knowledge}=deps;
 if(!learning||!capabilities||!knowledge)throw Error("missing operator contract dependencies");
 const audit=capabilities.audit();
 const byKind=new Map(audit.rows.map(row=>[row.kind,row]));
 const operators=Object.create(null);
 for(const kind of learning.KINDS){
  const def=learning.defs[kind],cap=byKind.get(kind);
  if(!cap||!def||!Array.isArray(def.ports)||!def.params||!knowledge.get(kind))
   throw Error("incomplete operator contract "+kind);
  operators[kind]=Object.freeze({
   kind,title:def.title,category:def.category,
   ports:Object.freeze([...def.ports]),
   defaults:Object.freeze({...def.params}),
   formula:def.formula||null,
   semantics:Object.freeze({teaching:true,knowledgeCard:true,
    numericalGraph:"requires_graph_validation",
    graphAutodiff:cap.graphAutodiff,
    modelParity:cap.modelParity}),
   backends:Object.freeze({numpyEligible:cap.numpyEligible,torchEligible:cap.torchEligible})
  });
 }
 return Object.freeze({schema:"seethetruth-operator-contracts-v1",
  count:learning.KINDS.length,operators:Object.freeze(operators)});
}
function resolve(kind){const contract=inventory().operators[kind];if(!contract)throw Error("unknown operator "+kind);return contract;}
return Object.freeze({inventory,resolve});
});
