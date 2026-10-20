/* SeetheTruth scalar objective lineage:
 * CE/MSE/MAE can be mixed using scalar Add and nonnegative scalar Scale.
 * Never treat an arbitrary scalar, detached number or output activation as loss.
 */
(function(root,factory){
 const api=factory(root);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthTrainingObjective=api;
})(typeof window!=="undefined"?window:{},function(root){"use strict";
const L=typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning;
const BASE=new Set(["mse","nd_mse","mae_loss","cross_entropy"]);
function analyze(raw){
 const graph=L.validateGraph(raw),byId=new Map(graph.nodes.map(n=>[n.id,n]));
 const edge=new Map(graph.edges.map(e=>[e.to+":"+e.port,e.from]));
 const memo=new Map(),guard=new Set();
 function valid(id){
  if(memo.has(id))return memo.get(id);
  if(guard.has(id))return false;
  const n=byId.get(id);if(!n)return false;
  guard.add(id);
  let yes=BASE.has(n.kind);
  if(n.kind==="scale"){
   const x=edge.get(id+":x"),factor=n.params.factor;
   yes=typeof factor==="number"&&Number.isFinite(factor)&&factor>=0&&factor<=100&&valid(x);
  }
  if(n.kind==="add")yes=valid(edge.get(id+":a"))&&valid(edge.get(id+":b"));
  memo.set(id,yes);guard.delete(id);return yes;
 }
 const choices=graph.nodes.filter(n=>valid(n.id)).map(n=>n.id);
 const preferred=choices.includes("loss")?"loss":
   [...graph.nodes].reverse().find(n=>choices.includes(n.id))?.id||null;
 return {choices,preferred,isLoss:id=>memo.has(id)?memo.get(id):valid(id),
   evidence:"objective must derive entirely from CE/MSE/MAE leaves through scalar nonnegative Scale/Add"};
}
return Object.freeze({analyze,BASE:Object.freeze([...BASE])});
});
