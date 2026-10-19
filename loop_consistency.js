/* LoopFormer-inspired short/long depth consistency ablation.
 * This deliberately DOES NOT implement LoopFormer time/step-size conditioned
 * short trajectories. It tests only the loss-design hypothesis with ordinary
 * user-editable blocks and genuine stop-gradient.
 *
 * Official source: https://github.com/armenjeddi/loopformer/blob/master/train.py
 * long/short NTP + 0.1 * mse_loss(long.detach(), short)
 */
(function(root,factory){
 const api=factory(root);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthLoopConsistency=api;
})(typeof window!=="undefined"?window:{},function(root){"use strict";
const L=typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning;
const Loop=typeof module==="object"&&module.exports?require("./loop_transformer.js"):root.SeetheTruthLoopTransformer;
function build(opts={}){
 const coefficient=opts.coefficient===undefined?0.1:opts.coefficient;
 if(typeof coefficient!=="number"||!Number.isFinite(coefficient)||coefficient<0||coefficient>1)
  throw Error("consistency coefficient must be a finite number within [0,1]");
 const graph=Loop.build({loops:2});
 const prior=graph.nodes.find(n=>n.id==="loss");
 if(!prior||prior.kind!=="scale"||graph.edges.some(e=>e.from==="loss"))throw Error("expected terminal normalized CE loss graph");
 prior.id="base_loss";
 for(const edge of graph.edges){if(edge.to==="loss")edge.to="base_loss";}
 const N=(id,kind,x,y,params={})=>graph.nodes.push({id,kind,x,y,params});
 const E=(from,to,port)=>graph.edges.push({from,to,port});
 N("deep_teacher","detach_demo",5580,1950);E("r1_state","deep_teacher","x");
 N("short_consistency","mse",5750,2040);
 E("r0_state","short_consistency","prediction");E("deep_teacher","short_consistency","target");
 N("cons_weight","scale",5950,2040,{factor:coefficient});E("short_consistency","cons_weight","x");
 N("loss","add",6230,1990);E("base_loss","loss","a");E("cons_weight","loss","b");
 graph.title="任务 42 · LoopFormer 动机实验 · 长深度教师/短深度一致性";
 return L.validateGraph(graph);
}
function inspect(raw){
 const graph=L.validateGraph(raw),by=new Map(graph.nodes.map(n=>[n.id,n]));
 const linked=(id,kind)=>by.get(id)?.kind===kind;
 const port=new Map(graph.edges.map(e=>[e.to+":"+e.port,e.from]));
 const valid=linked("base_loss","scale")&&linked("deep_teacher","detach_demo")&&
  linked("short_consistency","mse")&&linked("cons_weight","scale")&&linked("loss","add")&&
  port.get("deep_teacher:x")==="r1_state"&&
  port.get("short_consistency:prediction")==="r0_state"&&
  port.get("short_consistency:target")==="deep_teacher"&&
  port.get("cons_weight:x")==="short_consistency"&&
  port.get("loss:a")==="base_loss"&&port.get("loss:b")==="cons_weight";
 if(!valid)throw Error("not a valid short/long stop-gradient consistency ablation");
 return Object.freeze({base:"base_loss",short:"r0_state",long:"r1_state",
   teacher:"deep_teacher",consistency:"short_consistency",coefficient:by.get("cons_weight").params.factor,
   proof:"distinct short/deep states, explicit deep stop-gradient, actual MSE and scalar loss addition",
   limitation:"no time/step-size conditioning, no sampled shortcut trajectory; NOT a LoopFormer reproduction"});
}
return Object.freeze({build,inspect,
 source:"https://github.com/armenjeddi/loopformer/blob/master/train.py"});
});