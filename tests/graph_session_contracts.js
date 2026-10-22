'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),S=require('../graph_session.js');
for(const name of ['residual','swiglu','stream_state_branches']){
 const g=L.preset(name),staticResult=S.inspect(g),numeric=S.inspect(g,{numeric:true});
 assert.equal(staticResult.executable,true);
 assert.equal(staticResult.ir,null);
 assert.equal(numeric.executable,true);
 assert.equal(numeric.ir.nodes.length,g.nodes.length);
 assert.deepEqual(numeric.nodes.map(n=>n.id),numeric.ir.nodes.map(n=>n.id));
 assert.ok(numeric.nodes.every(n=>n.numericObserved));
 assert.deepEqual(numeric.nodes.map(n=>n.shape),numeric.ir.nodes.map(n=>n.shape));
}
const graph=L.preset('residual');
const partial=L.validateGraph({...graph,edges:graph.edges.filter(e=>!(e.to==='sum'&&e.port==='b'))});
const pending=S.inspect(partial,{numeric:true});
assert.equal(pending.executable,false);
assert.equal(pending.ir,null);
assert.ok(pending.nodes.some(n=>n.id==='sum'&&n.status==='missing'));
const selected=S.inspect(partial,{numeric:true,targetId:'act'});
assert.equal(selected.executable,true);
assert.equal(selected.nodes.length,3);
assert.equal(selected.ir.nodes.length,3);
console.log('GRAPH_SESSION_PASS readiness / branch / IR / operator identity');
