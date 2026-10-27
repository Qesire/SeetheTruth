/* Coordinate-journey selection mapped to authenticated free-graph wires.
 * The native VJP timeline supplies coordinate keys; this adapter checks that
 * each suggested wire exists in the CURRENT graph and addresses one port.
 */
(function(root,factory){"use strict";
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 if(root)root.SeetheTruthCoordinateJourneyCanvas=api;
})(typeof window!=="undefined"?window:null,function(){"use strict";
 const SCHEMA="seethetruth-coordinate-journey-canvas-v1";
 const fail=m=>{throw Error("coordinate journey canvas: "+m);};
 const coord=x=>Array.isArray(x)&&x.length>=1&&x.length<=5&&
  x.every(n=>Number.isSafeInteger(n)&&n>=0);
 function resolve(report,step,graph){
  if(!report||report.schema!=="seethetruth-coordinate-journey-v1"||
    !Array.isArray(report.steps)||!report.steps.includes(step))
   fail("step must belong to current accepted coordinate journey");
  if(!graph||!Array.isArray(graph.nodes)||!Array.isArray(graph.edges))
   fail("current executable graph required");
  const ids=new Set(graph.nodes.map(n=>n.id));
  if(report.scope==="mechanism_correspondence"||report.scope==="native_training_transaction")
   return Object.freeze({schema:SCHEMA,kind:"separate_evidence",nodeId:null,wire:null,coordinate:null});
  if(!["native_structural_graph","native_numeric_graph"].includes(report.scope))fail("unsupported scope");
  if(step.kind==="edge"){
   // Journey's sourceKey denotes the downstream output coordinate; targetKey
   // denotes the dependency coordinate that receives a VJP contribution.
   if(!coord(step.index)||!coord(step.sourceIndex)||
      typeof step.sourceNodeId!=="string"||!ids.has(step.sourceNodeId)||
      typeof step.nodeId!=="string"||!ids.has(step.nodeId)||
      typeof step.port!=="string"||!step.port)fail("invalid native scalar edge coordinates");
   const wire=graph.edges.find(e=>e.from===step.nodeId&&e.to===step.sourceNodeId&&e.port===step.port);
   if(!wire)fail("selected scalar dependency is absent from current graph wiring");
   if(graph.edges.filter(e=>e.from===wire.from&&e.to===wire.to&&e.port===wire.port).length!==1)
    fail("ambiguous graph dependency port");
   return Object.freeze({schema:SCHEMA,kind:"edge",nodeId:step.nodeId,
    coordinate:Object.freeze([...step.index]),downstreamNodeId:step.sourceNodeId,
    downstreamCoordinate:Object.freeze([...step.sourceIndex]),
    wire:Object.freeze({from:wire.from,to:wire.to,port:wire.port})});
  }
  if((step.kind==="seed"||step.kind==="merge")&&
     typeof step.nodeId==="string"&&ids.has(step.nodeId)&&coord(step.index))
   return Object.freeze({schema:SCHEMA,kind:"node",nodeId:step.nodeId,
    coordinate:Object.freeze([...step.index]),wire:null});
  fail("selected timeline step is not a graph coordinate");
 }
 function wireEquals(a,b){return !!a&&!!b&&a.from===b.from&&a.to===b.to&&a.port===b.port;}
 return Object.freeze({SCHEMA,resolve,wireEquals});
});
