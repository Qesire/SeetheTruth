'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const L=require('../learning_lab.js'),S=require('../graph_session.js'),P=require('../graph_learning_path.js');
const g=L.preset('ffn'),idx=L.indexGraph(L.validateGraph(g)),route=P.build(g);
assert.equal(route.executable,true);assert.equal(route.numericOk,false);
assert.deepEqual(route.steps.map(x=>x.id),idx.ordered.map(x=>x.id));
for(const n of route.steps){
 assert.equal(n.evidence,'STATIC_ONLY');assert.equal(n.sample,null);
 assert.ok(n.formula&&n.why&&n.flow&&n.exercise&&n.trap);
 for(const p of n.ports)assert.equal(p.sourceId,idx.incomingByPort.get(n.id+':'+p.port)||null);
}
const numeric=P.build(g,{numeric:true}),trace=S.inspect(g,{numeric:true}).trace;
assert.equal(numeric.numericOk,true);
for(let i=0;i<numeric.steps.length;i++){
 const n=numeric.steps[i],step=trace.steps[i];
 assert.equal(n.evidence,'SYNTHETIC_NUMERIC');assert.ok(n.sample.length>0);
 assert.deepEqual(n.shape,step.shape);
 let first=step.values;while(Array.isArray(first))first=first[0];
 assert.equal(n.sample[0],first);
}
const cut=JSON.parse(JSON.stringify(g)),edge=cut.edges.pop();const incomplete=P.build(cut);
assert.equal(incomplete.executable,false);
assert.ok(incomplete.steps.find(n=>n.id===edge.to).ports.some(p=>p.port===edge.port&&p.sourceId===null));
const invalid={...g,edges:[...g.edges,{from:'unknown',to:g.nodes[0].id,port:'x'}]};
assert.equal(P.build(invalid).evidence,'NO_EXECUTION');
const t={schema:L.SCHEMA,title:'Runtime-only error',nodes:[
 {id:'ids',kind:'token_ids',x:10,y:20,params:{rows:3,vocab:16,seed:1}},
 {id:'emb',kind:'embedding',x:250,y:20,params:{vocab:2,width:4}}],
 edges:[{from:'ids',to:'emb',port:'ids'}]};
const bad=P.build(t,{numeric:true});
assert.equal(bad.numericOk,false);
assert.ok(bad.firstError);
assert.ok(bad.steps.every(n=>n.sample===null));
const root=path.resolve(__dirname,'..'),read=p=>fs.readFileSync(path.join(root,p),'utf8');
const html=read('learn.html');
assert.ok(html.includes('id="learningPathPanel"')&&html.includes('UnifiedCore.inspectGraph(graph,{numeric:numerical}).path'));
assert.ok(html.includes('src="graph_learning_path.js"'));
assert.ok(read('atlas_server.py').includes('"/graph_learning_path.js":"graph_learning_path.js"'));
assert.ok(read('android/teaching-assets.txt').split(/\r?\n/).includes('graph_learning_path.js'));
console.log('GRAPH_LEARNING_PATH_PASS');
