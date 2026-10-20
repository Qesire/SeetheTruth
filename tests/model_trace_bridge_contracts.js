'use strict';
const assert=require('node:assert/strict');
const Bridge=require('../model_trace_bridge.js'),L=require('../learning_lab.js');
const {validate}=require('../model_trace.js');
const summary=data=>({shape:[2,2],count:4,mean:data.reduce((a,b)=>a+b,0)/4,
  rms:Math.hypot(...data)/2,l2:Math.hypot(...data),
  min:Math.min(...data),max:Math.max(...data),
  zeroFraction:data.filter(x=>x===0).length/4,sample:data});
const trace={
 schema:"seethetruth-real-model-trace-v1",
 provenance:{kind:"real_model_execution",modelId:"TEST_NOT_REAL/<img src=x onerror=alert(1)>",
 revision:"a".repeat(40),modulePath:"decoder.layers.0.mlp.down_proj",moduleType:"torch.nn.Linear",
 device:"cpu",framework:"pytorch",dtype:"float32",objective:"unit_test",
 promptSha256:"b".repeat(64),tokenIds:[1,2,3]},
 loss:.75,parameter:{identity:"decoder.layers.0.mlp.down_proj.weight",
   weight:summary([1,-2,3,0]),gradient:summary([.1,0,-.5,.3])},
 activation:{input:[.1,-.3,.7],output:[-.2,.5,-.1]}
};
validate(trace);
const before=JSON.stringify(trace),graph=L.preset('trainable_attention'),raw=JSON.stringify(graph);
for(const kind of Bridge.supportedKinds){
 const node=kind==='linear'?{id:'L1',kind}:{id:'wq',kind};
 const ref=Bridge.describe(trace,node),html=Bridge.render(trace,node);
 assert.equal(ref.nodeId,node.id);
 assert.equal(ref.nodeKind,kind);
 assert.equal(ref.relation,'read_only_real_model_reference_not_a_graph_operand');
 assert.equal(ref.evidenceTier,'REPLAY_REQUIRES_PINNED_MODEL_EXECUTION');
 assert.deepEqual(ref.observedWeightShape,[2,2]);
 assert.match(html,/待复跑来源/);
 assert.match(html,/不修改执行图、参数、优化器、导出代码/);
 assert.match(html,/扁平元素/);
 assert.match(html,/&lt;img src=x onerror=alert\(1\)&gt;/);
 assert.ok(!html.includes('<img src=x onerror=alert(1)>'),'hostile labels must be escaped');
 assert.equal((html.match(/real-sample-values/g)||[]).length,4,'weight/gradient/input/output sample sets');
}
assert.throws(()=>Bridge.describe(trace,{id:'x',kind:'silu'}),/选择 Linear、MatMul 或 Parameter/);
assert.throws(()=>Bridge.describe(trace,{id:'bad node!',kind:'matmul'}),/invalid graph node identity/);
const damaged=JSON.parse(before);damaged.parameter.weight.l2=0;
assert.throws(()=>Bridge.render(damaged,{id:'q_proj',kind:'matmul'}),/tensor L2/);
assert.equal(JSON.stringify(trace),before,'trace must remain immutable');
assert.equal(JSON.stringify(graph),raw,'educational graph must never be modified by evidence bridge');
console.log('REAL_TRACE_BRIDGE_PASS 3 appropriate block types, 4 bounded samples, exact graph immutability, HTML escaping, false evidence rejection');
