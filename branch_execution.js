/* Execute any selected node's complete upstream dependency slice.
 * Every unrelated incomplete side-branch is retained on the canvas.
 * The complete graph must still be a well-formed DAG; no architectural whitelist.
 */
(function(root,factory){const api=factory(root);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthBranchExecution=api;})(typeof window!=="undefined"?window:{},function(root){"use strict";
const common=typeof module==="object"&&module.exports;
const L=common?require("./learning_lab.js"):root.SeetheTruthLearning;
const C=common?require("./concept_flow.js"):root.SeetheTruthConceptFlow;
function slice(raw,targetId){
 const graph=L.validateGraph(raw);
 const index=L.indexGraph(graph);
 if(!graph.nodes.some(n=>n.id===targetId))throw Error("selected output block does not exist");

 const included=new Set(),stack=[targetId];
 while(stack.length){
   const id=stack.pop();if(included.has(id))continue;included.add(id);
   stack.push(...index.parents.get(id));
 }
 return L.validateGraph({schema:graph.schema,title:graph.title+" · branch "+targetId,
   nodes:graph.nodes.filter(n=>included.has(n.id)),
   edges:graph.edges.filter(e=>included.has(e.to)&&included.has(e.from))});
}
function execute(raw,targetId,mode="numeric"){
 const g=slice(raw,targetId);
 const trace=mode==="concept"?C.evaluate(g):mode==="numeric"?L.evaluate(g):
   (()=>{throw Error("unknown execution mode "+mode);})();
 return {graph:g,trace,targetId,omittedCount:raw.nodes.length-g.nodes.length,
   scope:"selected_upstream_dependency_slice"};
}
return Object.freeze({slice,execute});
});
