'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const X=require('../block_extension_registry.js'),L=require('../learning_lab.js');
const C=require('../concept_flow.js'),R=require('../graph_readiness.js'),P=require('../graph_learning_path.js');
assert.deepEqual(X.packs,['semantic_video','named_axis']);
assert.equal(X.kinds.length,8);
for(const kind of X.kinds)assert.ok(L.KINDS.includes(kind),'registered in actual free graph: '+kind);
const named=L.preset('extension_named_axis_composition');
const trace=L.evaluate(named),concept=C.evaluate(named),check=R.check(named,{numeric:true}),learning=P.build(named,{numeric:true});
assert.equal(check.executable,true);assert.equal(learning.numericOk,true);
assert.deepEqual(trace.steps.map(s=>s.shape),concept.steps.map(s=>s.shape));
const v=Object.fromEntries(trace.steps.map(s=>[s.id,s.values]));
for(let t=0;t<3;t++)for(let d=0;d<3;d++)assert.equal(v.add[t][d],v.x[t][d]+v.bias[0][d]);
assert.deepEqual(v.swap,v.native);assert.ok(v.delta.flat().every(x=>x===0));
assert.deepEqual(trace.steps.find(s=>s.id==='swap').axes,['D','T']);
const copied=X.recipe('extension_named_axis_composition');copied.nodes[0].params.seed=99;
assert.equal(X.recipe('extension_named_axis_composition').nodes[0].params.seed,2);
const wrongAxes=L.preset('extension_named_axis_composition');wrongAxes.edges.find(e=>e.to==='mean').from='swap';
assert.throws(()=>C.evaluate(wrongAxes),/named-axis port mismatch/);
assert.throws(()=>L.evaluate(wrongAxes),/named-axis port mismatch/);
assert.equal(R.check(wrongAxes).executable,false);
const wrongWidth=JSON.parse(JSON.stringify(named));wrongWidth.nodes.find(n=>n.id==='bias').params.width=4;
assert.throws(()=>L.evaluate(wrongWidth),/named feature bias/);
const graph={schema:L.SCHEMA,title:'Video and named axis packs with native arithmetic',nodes:[
 {id:'v',kind:'semantic_video_source',params:{frames:3,seed:1},x:10,y:40},
 {id:'a',kind:'semantic_video_slice',params:{start:0,length:2},x:270,y:40},
 {id:'b',kind:'semantic_video_slice',params:{start:2,length:1},x:270,y:260},
 {id:'join',kind:'semantic_video_concat',params:{},x:550,y:140},
 ...named.nodes.map(n=>({...n,id:'axis_'+n.id,y:n.y+450}))],
 edges:[{from:'v',to:'a',port:'x'},{from:'v',to:'b',port:'x'},
 {from:'a',to:'join',port:'a'},{from:'b',to:'join',port:'b'},
 ...named.edges.map(e=>({from:'axis_'+e.from,to:'axis_'+e.to,port:e.port}))]};
const run=L.evaluate(graph);
assert.equal(R.check(graph,{numeric:true}).executable,true);
const vals=Object.fromEntries(run.steps.map(s=>[s.id,s.values]));
assert.deepEqual(vals.join,vals.v,'all 5D values exactly reconstructed across temporal chunks');
assert.deepEqual(run.steps.find(s=>s.id==='join').shape,[1,3,2,2,2]);
assert.equal(P.build(graph,{numeric:true}).steps.length,graph.nodes.length);
const mismatch=JSON.parse(JSON.stringify(graph));mismatch.edges.find(e=>e.to==='a').from='axis_add';
assert.equal(R.check(mismatch).executable,false);
assert.throws(()=>L.evaluate(mismatch),/named-axis|expects named/);
assert.throws(()=>X.tensorShape([[1],[2,3]]),/ragged/);
assert.throws(()=>X.tensorShape([[[[NaN]]]]),/nonfinite/);
const badPack={name:'rogue',defs:{x_a:{title:'X',en:'X',category:'tensor',ports:[],params:{},formula:'x',why:'x',lesson:'x',caution:'x',source:'http://untrusted'}},validate(){},infer(){return [1,1]},execute(){return {values:[[1]]}}};
assert.throws(()=>X.createRegistry([badPack]),/source reference/);
assert.throws(()=>X.createRegistry([require('../axis_block_pack.js'),require('../axis_block_pack.js')]),/duplicate pack id/);
const root=path.resolve(__dirname,'..'),read=p=>fs.readFileSync(path.join(root,p),'utf8');
for(const f of ['semantic_block_pack.js','axis_block_pack.js','block_extension_registry.js']){
 assert.ok(read('learn.html').includes('src="'+f+'"'));
 assert.ok(read('android/teaching-assets.txt').split(/\r?\n/).includes(f));
 assert.ok(read('atlas_server.py').includes('"/'+f+'":"'+f+'"'));
}
console.log('UNIFIED_PACK_RUNTIME_PASS two independent packs, 8 kinds, 5D/2D, named axes, mixed native graph, graph learning route, negative contracts');