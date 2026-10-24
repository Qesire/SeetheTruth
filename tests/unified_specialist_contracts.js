'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..'),read=p=>fs.readFileSync(path.join(root,p),'utf8');
const Core=require('../learning_experience_core.js');
const Paper=require('../paper_patches.js'),Rewrite=require('../paper_graph_rewrites.js');
const L=require('../learning_lab.js'),Micro=require('../micro_training_flows.js');
const Cases=require('../micro_training_scenarios.js'),Semantic=require('../semantic_state_examples.js');
const Capability=require('../block_capabilities.js');
const eq=(a,b)=>assert.deepEqual(a,b);
const x=[[1,2]],w=[[1,0],[0,1]],a=[[1],[2]],b=[[3,4]];
for(const [op,input,expected] of [
 ['lora',{x,w,a,b,alpha:2},Paper.lora(x,w,a,b,2)],
 ['dora',{x,w,a,b,m:[[2,3]]},Paper.dora(x,w,a,b,[[2,3]])],
 ['online_attention',{q:[[1,0],[0,1]],k:[[1,0],[0,1]],v:[[3,4],[5,6]],chunk:1},
   Paper.onlineAttention([[1,0],[0,1]],[[1,0],[0,1]],[[3,4],[5,6]],1)]
]){
 const answer=Core.executeLab('paper',op,input);
 eq(answer.value,expected);assert.equal(answer.runtime,'graph_rewrite');
 assert.equal(answer.stateTrace,null);assert.equal(answer.computationTrace,null);
}
assert.throws(()=>Core.executeLab('paper','switch_top1',{}),/no registered executable adapter/);
assert.throws(()=>Core.executeLab('paper','lora',{x,w,a:[[1]],b}),/invalid LoRA dimensions/);
assert.throws(()=>Core.executeLab('paper','lora',{x,w,a,b,alpha:Infinity}),/non-finite numeric input/);
assert.throws(()=>Core.executeLab('paper','dora',{x,w,a,b,m:[[NaN,3]]}),/non-finite numeric input/);
assert.throws(()=>Core.executeLab('paper','online_attention',{q:[[1,0]],k:[[1,0]],v:[[1]],chunk:0}),/attention dimensions/);
const graph=L.preset('autodiff_core'),snapshot=JSON.stringify(graph);
const rew=Core.executeLab('paper','rewrite_lora',{graph,outputId:'project',options:{rank:2,alpha:2}});
eq(rew.value,Rewrite.lora(graph,'project',{rank:2,alpha:2}));
assert.equal(rew.value.delta,0);
assert.equal(rew.value.graph.nodes.length,graph.nodes.length+6);
assert.equal(rew.value.graph.nodes.find(n=>n.id==='project').kind,'add');
assert.equal(JSON.stringify(graph),snapshot,'paper rewrite cannot mutate original canvas');
assert.equal(Core.inspectGraph(rew.value.graph,{numeric:true}).numericOk,true);
assert.throws(()=>Core.executeLab('paper','rewrite_lora',{graph,outputId:'normalized'}),/selected 2D MatMul/);
for(const name of Semantic.names){
 const result=Core.executeLab('semantic_state','scenario',{name});
 eq(result.value,Semantic.run(name));assert.equal(result.value.passed,true);
 assert.equal(result.evidence,'SYNTHETIC_NUMERIC');
}
assert.throws(()=>Core.executeLab('semantic_state','scenario',{name:'not-a-case'}),/unknown example/);
const microSamples={
 shared:{x1:1,x2:2,w1:2,w2:3,target:0},
 moe:{r:5,capacity:2},
 ppo:{logit:0,adv:1,eps:.2}
};
for(const [name,parameters] of Object.entries(microSamples)){
 const answer=Core.executeLab('micro_training','scenario',{name,parameters});
 eq(answer.value,Cases.execute(name,parameters));
 assert.equal(answer.evidence,'SYNTHETIC_NUMERIC');
 assert.equal(answer.value.contract,'bounded_cpu_numeric_not_gpu_trace');
}
const shared=Core.executeLab('micro_training','scenario',{name:'shared',parameters:microSamples.shared}).value;
assert.equal(shared.summary.loss,256);assert.equal(shared.summary.dW1,64);
assert.equal(shared.summary.dW2,128);assert.equal(shared.summary.sharedParameters,1);
const noRouter=Core.executeLab('micro_training','scenario',{name:'moe',parameters:microSamples.moe}).value;
assert.equal(noRouter.result.gradRouter,null,'undefined discrete routing VJP is not zero');
const clipped=Core.executeLab('micro_training','scenario',{name:'ppo',parameters:{logit:10,adv:1,eps:.2}}).value;
assert.equal(clipped.result.active,false);assert.ok(clipped.result.gradLogits.every(x=>x===0));
assert.throws(()=>Core.executeLab('micro_training','scenario',{name:'moe',parameters:{r:5,capacity:3}}),/容量/);
assert.throws(()=>Core.executeLab('micro_training','scenario',{name:'shared',parameters:{...microSamples.shared,x1:null}}),/finite numerical/);
assert.throws(()=>Core.executeLab('micro_training','scenario',{name:'wrong',parameters:{}}),/unknown micro training scenario/);
const audit=Core.executeLab('capabilities','audit',{});
eq(audit.value,Capability.audit());
assert.equal(audit.evidence,'STATIC_ONLY');assert.equal(audit.value.rows.length,L.KINDS.length);
assert.ok(audit.value.rows.every(r=>r.graphAutodiff==='requires_graph_test'&&r.modelParity==='not_claimed'));
assert.throws(()=>Core.executeLab('capabilities','unsupported',{}),/no registered executable adapter/);
const routeScripts={
 'paper_lab.html':['paper_patches.js','learning_experience_core.js','specialized_learning_adapter.js'],
 'micro_training.html':['micro_training_flows.js','micro_training_scenarios.js','learning_experience_core.js','specialized_learning_adapter.js'],
 'semantic_state.html':['semantic_state.js','semantic_state_examples.js','learning_experience_core.js','specialized_learning_adapter.js','semantic_state_lab.js'],
 'capabilities.html':['block_capabilities.js','learning_experience_core.js','specialized_learning_adapter.js','capabilities_page.js']
};
for(const [p,names] of Object.entries(routeScripts)){
 const html=read(p);let last=-1;
 for(const name of names){const at=html.indexOf('src="'+name+'"');assert.ok(at>last,'script dependency order: '+p+' '+name);last=at;}
}
for(const [p,pattern] of [
 ['paper_lab.html','SeetheTruthSpecializedLesson.execute'],
 ['micro_training.html','SeetheTruthSpecializedLesson'],
 ['semantic_state_lab.js','Core.executeLab("semantic_state","scenario"'],
 ['capabilities_page.js','SeetheTruthSpecializedLesson.execute'],
 ['learn.html','UnifiedCore.executeLab("paper","rewrite_lora"']
])assert.ok(read(p).includes(pattern),'independent lab arithmetic left in page '+p);
for(const asset of ['micro_training_scenarios.js','semantic_state_examples.js']){
 assert.ok(read('android/teaching-assets.txt').split(/\r?\n/).includes(asset),asset+' unavailable on Android');
 assert.ok(read('atlas_server.py').includes('"/'+asset+'":"'+asset+'"'),asset+' not served');
}
console.log('UNIFIED_SPECIALIST_PASS source parity + invalid actions + real graph rewrite + micro VJP + named axes + 146 capability eligibility, PC/Android resources');
