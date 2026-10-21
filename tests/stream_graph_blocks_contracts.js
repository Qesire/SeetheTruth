'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),C=require('../concept_flow.js'),R=require('../graph_readiness.js');
const S=require('../stream_graph_blocks.js'),NP=require('../python_codegen.js'),PT=require('../torch_codegen.js');
const clone=x=>JSON.parse(JSON.stringify(x));
const n=(id,kind,params,x=20,y=50)=>({id,kind,x,y,params});
const e=(from,to,port)=>({from,to,port});
const graph={schema:L.SCHEMA,title:'typed state fork for speculative hypotheses',nodes:[
 n('prefix','input',{rows:2,width:3,seed:1}),
 n('chunk_a','input',{rows:1,width:3,seed:2},20,320),
 n('chunk_b','input',{rows:1,width:3,seed:3},20,600),
 n('init','stream_state_init',{origin:10,capacity:5},300,50),
 n('fork_a','stream_state_fork',{},550,75),
 n('fork_b','stream_state_fork',{},550,350),
 n('append_a','stream_state_append',{position:12},795,70),
 n('append_b','stream_state_append',{position:12},795,390),
 n('read_a','stream_state_read',{start:11,length:2},1080,55),
 n('read_b','stream_state_read',{start:11,length:2},1080,405),
 n('last','relu',{},1300,55)
],edges:[
 e('prefix','init','x'),e('init','fork_a','state'),e('init','fork_b','state'),
 e('fork_a','append_a','state'),e('chunk_a','append_a','chunk'),
 e('fork_b','append_b','state'),e('chunk_b','append_b','chunk'),
 e('append_a','read_a','state'),e('append_b','read_b','state'),
 e('read_a','last','x')
]};
assert.equal(S.kinds.length,4);
assert.equal(L.KINDS.length,138);
for(const k of S.kinds)assert.ok(L.KINDS.includes(k));
const validated=L.validateGraph(graph),saved=JSON.stringify(graph);
const concept=C.evaluate(validated),readiness=R.check(validated,{numeric:true}),numeric=L.evaluate(validated);
assert.equal(readiness.executable,true);assert.equal(readiness.readyCount,11);
assert.equal(concept.steps.length,11);assert.equal(numeric.steps.length,11);
assert.deepEqual(concept.steps.map(x=>x.shape),numeric.steps.map(x=>x.shape));
assert.deepEqual(readiness.nodes.map(x=>x.shape),numeric.steps.map(x=>x.shape));
const steps=Object.fromEntries(numeric.steps.map(x=>[x.id,x]));
const original=steps.prefix.values;
assert.deepEqual(steps.init.values,original);
assert.deepEqual(steps.init.logicalState,{type:'logical-stream-state',axes:['T','D'],origin:10,capacity:5,nextPosition:12,version:0});
assert.deepEqual(steps.fork_a.logicalState,steps.init.logicalState);
assert.deepEqual(steps.fork_b.logicalState,steps.init.logicalState);
assert.deepEqual(steps.append_a.logicalState,{type:'logical-stream-state',axes:['T','D'],origin:10,capacity:5,nextPosition:13,version:1});
assert.deepEqual(steps.append_a.values.slice(0,2),original);
assert.deepEqual(steps.append_b.values.slice(0,2),original);
assert.deepEqual(steps.append_a.values[2],steps.chunk_a.values[0]);
assert.deepEqual(steps.append_b.values[2],steps.chunk_b.values[0]);
assert.notDeepEqual(steps.append_a.values[2],steps.append_b.values[2]);
assert.deepEqual(steps.read_a.values,[original[1],steps.chunk_a.values[0]]);
assert.deepEqual(steps.read_b.values,[original[1],steps.chunk_b.values[0]]);
assert.deepEqual(steps.last.values,steps.read_a.values.map(row=>row.map(v=>Math.max(v,0))));
assert.deepEqual(steps.append_a.internals.logical_clock.values,[[10,13,1,5]]);
assert.deepEqual(steps.init.internals.logical_clock.values,[[10,12,0,5]]);
assert.ok(!('streamState' in steps.init),'trace must not publish mutable hidden runtime state');
assert.equal(JSON.stringify(graph),saved,'runtime must preserve graph source JSON');
assert.deepEqual(L.evaluate(L.validateGraph(clone(graph))),numeric,'JSON reload must reproduce CPU values and states');
assert.deepEqual(C.evaluate(graph),concept,'concept trace deterministic');
const broken=(modify)=>{const copy=clone(graph);modify(copy);return copy;};
const failures=[
 [g=>g.edges.find(x=>x.to==='append_a'&&x.port==='state').from='chunk_a',/expected stream-state input/,'numeric plugged into state port'],
 [g=>g.edges.find(x=>x.to==='last'&&x.port==='x').from='append_a',/expected numeric-tensor input/,'state plugged into a generic tensor port'],
 [g=>g.nodes.find(x=>x.id==='append_a').params.position=11,/out-of-order\/stale/,'stale append'],
 [g=>g.nodes.find(x=>x.id==='append_a').params.position=13,/out-of-order\/stale/,'gapped append'],
 [g=>g.nodes.find(x=>x.id==='init').params.capacity=2,/append exceeds stream capacity/,'capacity overload after init'],
 [g=>g.nodes.find(x=>x.id==='read_a').params.start=12,/stream read exceeds known absolute positions/,'unpopulated range'],
 [g=>g.nodes.find(x=>x.id==='read_a').params.start=9,/stream read exceeds known absolute positions/,'read before origin'],
 [g=>g.nodes.find(x=>x.id==='chunk_a').params.width=4,/feature axis D mismatch/,'feature mismatch'],
 [g=>g.nodes.find(x=>x.id==='append_a').params.position=-1,/position must be integer/,'negative position'],
 [g=>g.nodes.find(x=>x.id==='init').params.origin=1.5,/origin must be integer/,'fractional origin'],
];
for(const [fn,why,label] of failures){
 const x=broken(fn),ready=R.check(x),positive=ready.nodes.find(t=>t.status==='shape-error');
 assert.equal(ready.executable,false,label);
 assert.ok(ready.structuralError||positive,label+' must fail readiness before execution');
 assert.throws(()=>C.evaluate(x),why,label+' concept check');
 assert.throws(()=>L.evaluate(x),why,label+' numerical check');
}
const incomplete=broken(g=>g.edges=g.edges.filter(x=>!(x.to==='append_a'&&x.port==='chunk')));
const pending=R.check(incomplete);
assert.equal(pending.executable,false);
assert.equal(pending.nodes.find(x=>x.id==='append_a').status,'missing');
assert.equal(pending.nodes.find(x=>x.id==='read_a').status,'blocked');
const badCycle=broken(g=>g.edges.push(e('append_a','init','x')));
assert.equal(R.check(badCycle).executable,false);
assert.match(R.check(badCycle).structuralError,/循环|端口/);
for(const k of S.kinds){
 assert.equal(NP.supportedKinds.includes(k),false,'NumPy export eligibility is not verified');
 assert.equal(PT.supportedKinds.includes(k),false,'PyTorch export eligibility is not verified');
}
console.log('GRAPH_STREAM_BLOCKS_PASS four typed free-graph operators, forked exact CPU streams, graph invariants, fail-closed ports, v1 JSON compatibility');
