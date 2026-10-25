/* Source-driven eligibility audit. Backend allowlists are NOT full-graph guarantees. */
(function(root,factory){"use strict";
const isNode=typeof module==="object"&&module.exports;
const api=factory(isNode?{
 learning:require("./learning_lab.js"),numpy:require("./python_codegen.js"),
 torch:require("./torch_codegen.js"),knowledge:require("./block_knowledge.js"),
 certificates:()=>require("./operator_certificate_registry.js")
}:{
 learning:root.SeetheTruthLearning,numpy:root.SeetheTruthPythonExport,
 torch:root.SeetheTruthTorchExport,knowledge:root.SeetheTruthBlockKnowledge,
 certificates:()=>root.SeetheTruthOperatorCertificates
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
// Explicit capability states: a backend allowlist is not an independent run.
function matrix(){
 const existing=audit();
 const certificates=s.certificates?.();
 if(!certificates||typeof certificates.forKind!=="function")
  throw Error("missing source-pinned independent certificate registry");
 const summary=certificates.summary();
 const rows=existing.rows.map(row=>{
  const statuses={
   registered:{availability:"available",verification:"registration_checked"},
   staticShape:{availability:"requires_valid_graph",verification:"not_executed"},
   boundedForward:{availability:"requires_valid_graph",verification:"not_executed"},
   coordinateTrace:{availability:"requires_numeric_trace",verification:"not_executed"},
   localVjp:{availability:"requires_per_graph_oracle",verification:"not_attested"},
   numpyExport:{availability:row.numpyEligible?"eligible":"unsupported",verification:"not_attested"},
   torchExport:{availability:row.torchEligible?"eligible":"unsupported",verification:"not_attested"},
   modelParity:{availability:"not_claimed",verification:"not_attested"},
   gpuKernel:{availability:"not_claimed",verification:"not_attested"}
  };
  const nativeCase=certificates.forKind(row.kind);
  return Object.freeze({kind:row.kind,title:row.title,
   certifiedCases:nativeCase?Object.freeze([nativeCase]):Object.freeze([]),
   capability:Object.freeze(Object.fromEntries(Object.entries(statuses).map(([key,v])=>[key,Object.freeze(v)])))});
 });
 return Object.freeze({schema:"seethetruth-operator-capability-matrix-v1",
  source:existing.schema,evidence:"DECLARATIVE_NOT_EXECUTED",
  certificateEvidence:summary.evidence,
  certificateSummary:summary,
  counts:existing.counts,rows:Object.freeze(rows)});
}
function describe(kind){
 const row=matrix().rows.find(x=>x.kind===kind);
 if(!row)throw Error("unknown operator capability kind "+kind);
 return row;
}
return Object.freeze({audit,matrix,describe});
});
