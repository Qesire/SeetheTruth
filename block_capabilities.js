/* Source-driven eligibility audit. Backend allowlists are NOT full-graph guarantees. */
(function(root,factory){"use strict";
const isNode=typeof module==="object"&&module.exports;
const api=factory(isNode?{
 learning:require("./learning_lab.js"),numpy:require("./python_codegen.js"),
 torch:require("./torch_codegen.js"),knowledge:require("./block_knowledge.js")
}:{
 learning:root.SeetheTruthLearning,numpy:root.SeetheTruthPythonExport,
 torch:root.SeetheTruthTorchExport,knowledge:root.SeetheTruthBlockKnowledge
});
if(isNode)module.exports=api;if(root)root.SeetheTruthCapabilities=api;
})(typeof window!=="undefined"?window:null,function(s){"use strict";
function audit(){
 const {learning,numpy,torch,knowledge}=s;
 if(!learning||!numpy||!torch||!knowledge)throw Error("missing capability inventory dependency");
 const kinds=[...learning.KINDS].sort(),known=new Set(kinds);
 const np=new Set(numpy.supportedKinds),pt=new Set(torch.supportedKinds);
 if(kinds.length!==known.size)throw Error("duplicate kind");
 for(const k of [...np,...pt])if(!known.has(k))throw Error("unknown exporter kind "+k);
 const rows=kinds.map(kind=>{
  const def=learning.defs[kind],card=knowledge.get(kind);
  if(!def||!def.title||!def.category||!card)throw Error("incomplete kind "+kind);
  return Object.freeze({kind,title:def.title,category:def.category,
   registered:true,knowledgeCard:true,boundedTeachingContract:true,
   numpyEligible:np.has(kind),torchEligible:pt.has(kind),
   graphAutodiff:"requires_graph_test",modelParity:"not_claimed"});
 });
 return Object.freeze({schema:"seethetruth-capabilities-v1",
  counts:Object.freeze({registered:rows.length,knowledgeCards:rows.length,
   numpyEligible:rows.filter(r=>r.numpyEligible).length,
   torchEligible:rows.filter(r=>r.torchEligible).length,
   numpyMissing:rows.filter(r=>!r.numpyEligible).length,
   torchMissing:rows.filter(r=>!r.torchEligible).length}),
  rows:Object.freeze(rows)});
}
return Object.freeze({audit});
});
