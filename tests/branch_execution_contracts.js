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
console.log('Arbitrary branch execution & advisory port preview PASS: partial canvas, target slice, shapes, softmax and cycles');
