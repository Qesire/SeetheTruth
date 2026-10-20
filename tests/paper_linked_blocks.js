const assert=require("node:assert/strict");const C=require("../block_catalog.js"), L=require("../learning_lab.js"),K=require("../block_knowledge.js");
const run=(kind,inputs={},params={})=>C.execute({id:"test",kind,params},inputs);
const close=(x,y)=>assert.ok(Math.abs(x-y)<1e-9);
assert.equal(K.validate().cards,L.KINDS.length);
assert.deepEqual(run("sliding_window_mask",{x:[[0,0,0],[0,0,0],[0,0,0]]},{window:2}).values,[[1,0,0],[1,1,0],[0,1,1]]);
assert.deepEqual(run("router_topk",{x:[[1,3,2]]},{k:2}).values,[[1,2]]);
close(run("qk_norm",{x:[[3,4]]},{epsilon:1e-5}).values[0][0],3/Math.sqrt(12.5+1e-5));
close(run("logit_softcap",{x:[[100]]},{cap:5}).values[0][0],5*Math.tanh(20));
assert.deepEqual(run("weighted_sum",{weights:[[.25,.75]],values:[[2,4],[6,8]]}).values,[[5,7]]);
assert.deepEqual(run("detach_demo",{x:[[1,-2]]}).values,[[1,-2]]);
assert.deepEqual(run("residual_scale",{x:[[2,4]]},{factor:.5}).values,[[1,2]]);
for(const id of ["qk_norm","sliding_window_mask","logit_softcap","router_topk","weighted_sum","detach_demo","residual_scale"]){assert.ok(K.get(id)?.refs.length);assert.ok(C.defs[id]);}
console.log("paper-linked block contracts PASS");
