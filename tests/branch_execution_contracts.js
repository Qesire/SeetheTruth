'use strict';
const assert=require('node:assert/strict'),L=require('../learning_lab.js'),R=require('../graph_readiness.js'),B=require('../branch_execution.js');
const N=(id,kind,params={})=>({id,kind,x:30,y:30,params}),E=(from,to,port)=>({from,to,port});
const g={schema:L.SCHEMA,title:"free-branch",nodes:[
 N("x","input",{rows:3,width:4,seed:1}),
 N("trans","transpose2d"),
 N("mat","matmul"),
 N("result","softmax"),
 N("stray","silu"),
 N("bad","matmul")
 ],edges:[E("x","trans","x"),E("x","mat","a"),E("trans","mat","b"),E("mat","result","x"),E("x","bad","a"),E("x","bad","b")]};
const whole=R.check(g);
assert.equal(whole.executable,false);
assert.equal(whole.nodes.find(n=>n.id==="stray").status,"missing");
assert.equal(whole.nodes.find(n=>n.id==="bad").status,"shape-error");
const branch=B.slice(g,"result");
assert.deepEqual(branch.nodes.map(n=>n.id),["x","trans","mat","result"]);
assert.equal(branch.edges.length,4);
assert.notStrictEqual(branch.nodes[0].params,g.nodes[0].params);
assert.equal(g.nodes.length,6,"slicing must not mutate the editable canvas");
const info=R.check(g,{targetId:"result",numeric:true});
assert.equal(info.executable,true);assert.equal(info.scope,"selected_upstream_dependency_slice");
assert.equal(info.readyCount,4);assert.equal(info.numeric.steps,4);
const numeric=B.execute(g,"result","numeric");
const concept=B.execute(g,"result","concept");
assert.deepEqual(numeric.trace.steps.map(x=>x.shape),concept.trace.steps.map(x=>x.shape));
assert.deepEqual(numeric.trace.steps.at(-1).shape,[3,3]);
for(const row of numeric.trace.steps.at(-1).values)assert.ok(Math.abs(row.reduce((s,x)=>s+x,0)-1)<1e-12);
const incomplete=B.slice(g,"stray");
assert.equal(R.check(incomplete).executable,false);
assert.throws(()=>B.execute(g,"stray","numeric"),/还没有连接|未连接/);
assert.throws(()=>B.slice(g,"absent"),/does not exist/);
assert.throws(()=>B.execute(g,"result","unknown"),/unknown execution mode/);
// Connection preview remains advisory. A wrong shape is not forbidden by the editor.
const pWrong=R.preview(g,"x","mat","b");
assert.equal(pWrong.status,"incompatible");
assert.match(pWrong.reason,/MatMul|收缩|不匹配/);
const pCorrect=R.preview(g,"trans","mat","b");
assert.equal(pCorrect.status,"compatible");
assert.deepEqual(pCorrect.targetShape,[3,3]);
const pStray=R.preview(g,"x","stray","x");
assert.equal(pStray.status,"compatible","other bad branches must not poison local preview");
const pCycle=R.preview(g,"result","trans","x");
assert.equal(pCycle.status,"invalid");
assert.match(pCycle.reason,/循环|cycle/i);
const pMissing=R.preview(g,"stray","result","x");
assert.equal(pMissing.status,"pending");
assert.equal(g.edges.find(e=>e.to==="mat"&&e.port==="b").from,"trans");
const Train=require('../train_state.js');
const train=L.preset("trainable_core");
train.nodes.push(N("unfinishedActivation","relu"));
train.nodes.push(N("unfinishedMatMul","matmul"));
train.edges.push(E("features","unfinishedMatMul","a"));
const original=JSON.stringify(train);
assert.equal(R.check(train).executable,false);
assert.throws(()=>Train.steps(train,{count:1}),/还没有连接|未连接/);
const updated=Train.steps(train,{targetId:"loss",count:2,lr:0.004});
assert.equal(updated.history.length,2);
assert.ok(Number.isFinite(updated.finalLoss));
assert.equal(updated.graph.nodes.length,train.nodes.length);
assert.equal(updated.graph.nodes.find(n=>n.id==="shared_weight").params.optimizer.t,2);
assert.equal(updated.graph.nodes.find(n=>n.id==="unfinishedMatMul").kind,"matmul");
assert.equal(updated.graph.nodes.find(n=>n.id==="unfinishedActivation").kind,"relu");
assert.deepEqual(updated.graph.edges,train.edges);
assert.equal(JSON.stringify(train),original,"training must not edit the in-progress canvas");
assert.deepEqual(updated.next.steps.at(-1).shape,[1,1]);
const resume=Train.steps(L.validateGraph(JSON.parse(JSON.stringify(updated.graph))),{targetId:"loss",count:1,lr:0.004});
const direct=Train.steps(updated.graph,{targetId:"loss",count:1,lr:0.004});
assert.deepEqual(resume.graph,direct.graph,"resumed branch optimizer state must match uninterrupted training");
assert.throws(()=>Train.steps(train,{targetId:"unfinishedMatMul",count:1}),/trainable Parameter|还没有连接|未连接/);
console.log('Arbitrary branch execution & advisory port preview PASS: partial canvas, target slice, shapes, softmax and cycles');
