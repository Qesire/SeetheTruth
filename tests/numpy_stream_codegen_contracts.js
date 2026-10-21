'use strict';
/* Independent full Python interpreter proof of generated streaming-state graphs. */
const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),NP=require('../python_codegen.js'),PT=require('../torch_codegen.js');
const kinds=require('../stream_graph_blocks.js').kinds;
const graph=L.preset('stream_state_branches'),oracle=L.evaluate(graph);
const compiled=NP.compile(graph),source=compiled.source;
assert.ok(source.includes('def _state_append(')&&source.includes('def _state_check('));
assert.ok(source.includes('class _StreamOutputs'));
assert.equal(compiled.nodes.length,graph.nodes.length);
assert.deepEqual(compiled.nodes.map(x=>x.id),oracle.steps.map(x=>x.id));
function execute(extra='',program=source){
 const out=execFileSync('python3',['-c',program+'\n'+extra],{
  encoding:'utf8',timeout:30000,maxBuffer:4*1024*1024});
 return JSON.parse(out.trim().split('\n').at(-1));
}
const script=(overrides={})=>[
 'overrides=json.loads('+JSON.stringify(JSON.stringify(overrides))+')',
 'results=run_graph(overrides)',
 'print(json.dumps({"values":{key:data.tolist() for key,data in results.items()},',
 '                  "states":results.state_metadata},allow_nan=False))'
].join('\n');
const result=execute(script());
const flatten=x=>x.flat(Infinity);
function close(a,b,label){
 const x=flatten(a),y=flatten(b);
 assert.equal(x.length,y.length,label+' element count');
 x.forEach((v,i)=>assert.ok(Number.isFinite(y[i])&&Math.abs(v-y[i])<=2e-10+Math.abs(v)*2e-12,
  label+'['+i+'] '+v+' != '+y[i]));
}
assert.equal(Object.keys(result.values).length,10);
for(const step of oracle.steps)close(step.values,result.values[step.id],step.id);
for(const state of oracle.steps.filter(x=>x.logicalState))
 assert.deepEqual(result.states[state.id],state.logicalState,state.id+' full logical state provenance');
assert.deepEqual(Object.keys(result.states).sort(),['append_a','append_b','fork_a','fork_b','init']);
assert.notDeepEqual(result.values.append_a.at(-1),result.values.append_b.at(-1),'fork continuation must differ');
assert.deepEqual(result.values.init,result.values.fork_a.slice(0,2));
const override={
 prefix:[[6,7,8],[9,10,11]],chunk_a:[[12,13,14]],chunk_b:[[15,16,17]]
};
const changed=execute(script(override));
close(changed.values.init,override.prefix,'source override prefix');
close(changed.values.append_a,[...override.prefix,...override.chunk_a],'override recomputes branch A');
close(changed.values.append_b,[...override.prefix,...override.chunk_b],'override recomputes branch B');
close(changed.values.read_a,[override.prefix[1],override.chunk_a[0]],'override recomputes read A');
close(changed.values.read_b,[override.prefix[1],override.chunk_b[0]],'override recomputes read B');
assert.deepEqual(changed.states,result.states,'overrides cannot change graph static version/position');
assert.notDeepEqual(result.values.read_a,changed.values.read_a,'no baked intermediate arrays');
function rejected(extra,pattern){
 const output=execute([
  'try:',
  ...extra.split('\n').map(line=>'    '+line),
  'except ValueError as e:',
  '    print(json.dumps({"rejected":str(e)}))',
  'else:',
  '    print(json.dumps({"rejected":"ACCEPTED"}))'
 ].join('\n'));
 assert.match(output.rejected,pattern,'invalid state path must reject');
}
rejected('run_graph({"chunk_a":[[1,2]]})',/source chunk_a expected/);
rejected('run_graph({"prefix":[[1,2,3],[1,float("nan"),3]]})',/NaN or Infinity/);
rejected('s=_state_init([[1,2],[3,4]],10,2)\n_state_append(s,[[5,6]],11)',/out-of-order\/stale/);
rejected('s=_state_init([[1,2],[3,4]],10,2)\n_state_append(s,[[5,6]],12)',/capacity/);
rejected('s=_state_init([[1,2]],10,3)\n_state_append(s,[[1,2,3]],11)',/feature axis/);
rejected('s=_state_init([[1,2]],10,3)\n_state_read(s,10,2)',/exceeds known/);
rejected('s=_state_init([[1,2]],10,3)\ns["nextPosition"]=12\n_state_fork(s)',/corrupt/);
rejected('s=_state_init([[1,2]],10,3)\ns["axes"]=["Tk","D"]\n_state_read(s,10,1)',/expected logical-stream-state/);
const forks=execute([
 'parent=_state_init([[1,2],[3,4]],10,4)',
 'copy=_state_fork(parent)',
 'copy["tensor"][0,0]=999.0',
 'read=_state_read(parent,10,1)',
 'read[0,0]=-20.',
 'print(json.dumps({"original":parent["tensor"].tolist(),"fork":copy["tensor"].tolist(),"read":read.tolist()}))'
].join('\n'));
assert.deepEqual(forks.original,[[1,2],[3,4]]);
assert.deepEqual(forks.fork,[[999,2],[3,4]]);
assert.deepEqual(forks.read,[[-20,2]]);
const only=NP.compile(graph,{targetId:'read_a'}),partial=execute(script().replace('run_graph(overrides)','run_graph(overrides)'),only.source);
assert.ok(!('append_b' in partial.values),'scoped branch must not execute unrelated fork');
assert.ok('append_a' in partial.values&&'read_a' in partial.values);
const corrupted=JSON.parse(JSON.stringify(graph));
corrupted.nodes.find(n=>n.id==='append_a').params.position=11;
assert.throws(()=>NP.compile(corrupted),/out-of-order\/stale/);
const invalidState=JSON.parse(JSON.stringify(graph));
invalidState.edges.find(e=>e.to==='read_a'&&e.port==='state').from='prefix';
assert.throws(()=>NP.compile(invalidState),/expected stream-state/);
for(const kind of kinds){
 assert.ok(NP.supportedKinds.includes(kind),kind+' exposed in NumPy whitelist');
 assert.ok(PT.supportedKinds.includes(kind),kind+' Torch autograd is covered by torch_stream_autograd_contracts.js');
}
const legacy=NP.compile(L.preset('ffn'));
assert.ok(!legacy.source.includes('class _StreamOutputs'),'legacy non-state exports are untouched');
console.log('STREAM_NUMPY_PARITY_PASS '+oracle.steps.length+' JS-Python nodes, '+Object.keys(result.states).length+
 ' immutable versions, overrides, fork isolation, error paths and scoped codegen');
