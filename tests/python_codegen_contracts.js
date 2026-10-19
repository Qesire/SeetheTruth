'use strict';
const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Py=require('../python_codegen.js');
const compare=(name,fromPython,source,eps=3e-9)=>{
 const a=source.flat(Infinity),b=fromPython.flat(Infinity);
 assert.equal(a.length,b.length,name+' flat element count');
 for(let i=0;i<a.length;i++){
  const tolerance=eps+Math.abs(a[i])*3e-12;
  assert.ok(Number.isFinite(b[i])&&Math.abs(b[i]-a[i])<=tolerance,
    name+'['+i+']: Python '+b[i]+' ≠ JS '+a[i]);
 }
};
function execute(compiled,setup=''){
 const code=compiled.source+(setup||'');
 return JSON.parse(execFileSync('python3',['-c',code],{encoding:'utf8',maxBuffer:3*1024*1024,timeout:25000}).trim().split('\n').at(-1));
}
let comparisons=0;
for(const name of [
 'nd_cached_gqa_decode','nd_causal_attention','nd_gqa_attention',
 'nd_mqa_attention','nd_trainable_gqa','nd_trainable_mqa',
 'trainable_core','trainable_swiglu','trainable_attention',
 'activation_gallery','activation_extension','optimizer_compare','optimizer_decay_compare'
]){
 const graph=L.preset(name),program=Py.compile(graph),output=execute(program),
       trace=L.evaluate(graph);
 assert.equal(Object.keys(output).length,graph.nodes.length,name+' exported all steps');
 for(const step of trace.steps){
  assert.ok(Object.hasOwn(output,step.id),name+" missing exported node "+step.id+"; keys="+Object.keys(output));
  assert.ok(Array.isArray(step.values),name+" missing JS value at "+step.id);
  compare(name+': '+step.id,output[step.id],step.values);
  comparisons++;
 }
 assert.ok(program.source.includes('def run_graph(overrides=None):'));
 assert.ok(program.source.includes("if __name__ == '__main__':"));
 assert.ok(!program.source.includes('SeetheTruthLearning'),'output depends on SeetheTruth code');
 console.log('Standalone NumPy graph PASS: '+name+' ('+trace.steps.length+' nodes)');
}
// Source override must ACTUALLY alter downstream computation, not replay baked intermediate outputs.
const graph=L.preset('nd_causal_attention'),program=Py.compile(graph);
const baseline=execute(program),src=graph.nodes.find(n=>n.id==='query');
const x=L.evaluate(graph).steps.find(s=>s.id===src.id).values;
const modified=JSON.parse(JSON.stringify(x));modified[0][0][0][0]+=.25;
const more='\nchanged=json.loads('+JSON.stringify(JSON.stringify(modified))+')\n'
 +'out=run_graph({"query":changed})\n'
 +'print(json.dumps({k:v.tolist() for k,v in out.items()}))\n';
const updated=execute(program,more);
compare('override source',updated.query,modified);
assert.notDeepEqual(updated.context,baseline.context,'exported program must recalculate attention');
assert.deepEqual(baseline.key,updated.key,'unmodified source must not change');
const withIncomplete=L.preset('blank');
withIncomplete.nodes.push({id:'silu_orphan',kind:'silu',x:430,y:300,params:{}});
const sliced=Py.compile(withIncomplete,{targetId:'x'});
assert.equal(Object.keys(execute(sliced)).length,1);
assert.throws(()=>Py.compile(withIncomplete),/没有连接|还没有连接/);
const bad=L.preset('grpo_objective');
assert.throws(()=>Py.compile(bad),/不支持/);
const tampered=JSON.parse(JSON.stringify(L.preset('nd_causal_attention')));
tampered.edges=tampered.edges.filter(e=>!(e.to==='scores'&&e.port==='a'));
assert.throws(()=>Py.compile(tampered),/还没有连接/);
assert.equal(new Set(Py.supportedKinds).size,Py.supportedKinds.length);
console.log('Real executable Python export PASS: '+comparisons+' JS/Python tensor parity checks, override, scoped export, unsupported fail-closed');
