/* Source, imported-model and telemetry evidence are observational, never executable. */
(function(root,factory){"use strict";
 const node=typeof module==="object"&&module.exports;
 const api=factory(node?require("./source_evidence_levels.js"):root.SeetheTruthSourceView,
                   node?require("./model_trace.js"):root.SeetheTruthTrace,
                   node?require("./telemetry_evidence.js"):root.SeetheTruthTelemetryEvidence);
 if(node)module.exports=api;
 if(root)root.SeetheTruthEvidenceCore=api;
})(typeof window!=="undefined"?window:null,function(Source,Trace,Telemetry){"use strict";
 const SCHEMA="seethetruth-evidence-inspection-v1",fail=x=>{throw Error("evidence: "+x);};
 function wrap(domain,op,tier,data){return Object.freeze({schema:SCHEMA,kind:"evidence_inspection",
  domain,operation:op,evidenceTier:tier,executed:false,independentlyReplayed:false,data});}
 function inspect(domain,op,input={}){
  if(!input||typeof input!=="object"||Array.isArray(input))fail("invalid input object");
  if(domain==="source_atlas"&&op==="project"){
   if(!Source)fail("source projector unavailable");
   const {graph,scope=null,level="macro",capacity=190}=input;
   if(graph?.meta?.analysis!=="static-python-ast")fail("not a static Python AST source graph");
   if(!Number.isSafeInteger(capacity)||capacity<1||capacity>190)fail("invalid projection capacity");
   const result=Source.project(graph,scope,level,capacity);
   const projection=JSON.parse(JSON.stringify({...result,matches:[...result.matches]}));
   // Existing 3D atlas uses matches.has(id); retain Set behavior while exposing
   // a transport-safe matchIds list for JSON serialization and evidence tests.
   projection.matchIds=projection.matches;
   projection.matches=new Set(projection.matchIds);
   return wrap(domain,op,"STATIC_SOURCE_ONLY",Object.freeze({projection}));
  }
  if(domain==="real_model"&&op==="validate_import"){
   if(!Trace)fail("model trace validator unavailable");
   return wrap(domain,op,"IMPORTED_MODEL_PROVENANCE_CLAIM_UNREPLAYED",
     Object.freeze({summary:Trace.validate(input.trace)}));
  }
  if(domain==="telemetry"&&op==="validate_import"){
   if(!Telemetry)fail("telemetry validator unavailable");
   const data=input.data;Telemetry.validate(data);
   const synthetic=input.synthetic===true||String(data.meta?.measurement||"").startsWith("SYNTHETIC");
   return wrap(domain,op,synthetic?"SYNTHETIC_TELEMETRY_DEMO":"IMPORTED_TELEMETRY_UNATTESTED",
    Object.freeze({schema:data.schema,synthetic,
      counts:Object.freeze({operators:data.operators.length,modules:data.modules.length,
       snapshots:data.snapshots.length,moduleCalls:data.module_calls?.length||0}),
      measurement:String(data.meta?.measurement||"unspecified")}));
  }
  fail("unregistered evidence operation "+domain+"/"+op);
 }
 return Object.freeze({SCHEMA,inspect});
});