'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const L=require('../learning_lab.js'),R=require('../graph_readiness.js'),ND=require('../nd_tensor.js');
const Shapes=require('../shape_index_rules.js'),IR=require('../execution_ir.js');
const N=(id,kind,params={},x=25,y=30)=>({id,kind,params,x,y});
const custom=(kind,params,rows,width)=>L.validateGraph({schema:L.SCHEMA,title:'shape '+kind,
 nodes:[N('x','input',{rows,width,seed:3}),N('y',kind,params,330,30)],
 edges:[{from:'x',to:'y',port:'x'}]});
const examples=[
 {name:'transpose2d',graph:custom('transpose2d',{},2,3),target:'y'},
 {name:'nd_reshape',graph:custom('nd_reshape',{b:1,h:2,t:3,d:4},6,4),target:'y'},
 ...['biased','summed','flat','expand','perm'].map(id=>({name:id,graph:L.preset('nd_attention_lesson'),target:id})),
 {name:'repeat_kv',graph:L.preset('nd_cached_gqa_decode'),target:'shared_k'}
];
const kinds=new Set();
for(const item of examples){
 const g=item.graph,trace=L.evaluate(g),nodes=new Map(trace.steps.map(x=>[x.id,x]));
 const indexed=L.indexGraph(L.validateGraph(g)),n=g.nodes.find(x=>x.id===item.target);
 assert.ok(n,'missing '+item.target);kinds.add(n.kind);
 const inputShapes=Object.fromEntries(L.defs[n.kind].ports.map(port=>[port,nodes.get(indexed.incomingByPort.get(n.id+':'+port)).shape]));
 const mapping=Shapes.analyze(n,inputShapes),actual=nodes.get(n.id);
 assert.deepEqual(mapping.outputShape,actual.shape,item.name+' output');
 const value=ND.at(actual.values,mapping.outputIndex);
 const sourceValues=mapping.sourceIndices.map(p=>ND.at(nodes.get(indexed.incomingByPort.get(n.id+':'+p.port)).values,p.index));
 const expected=n.kind==='nd_broadcast_add'?sourceValues[0]+sourceValues[1]:
  n.kind==='nd_reduce_sum'?sourceValues.reduce((a,b)=>a+b,0):sourceValues[0];
 assert.ok(Math.abs(value-expected)<1e-11,item.name+' index value '+value+' vs '+expected);
 const ready=R.check(g);
 assert.ok(ready.executable,item.name+' readiness');
 assert.deepEqual(ready.nodes.find(x=>x.id===n.id).shape,mapping.outputShape);
 const ir=IR.snapshot(g,{trace}),envelope=ir.nodes.find(x=>x.id===n.id);
 assert.deepEqual(envelope.shapeTransform,mapping);
 assert.ok(Object.isFrozen(envelope.shapeTransform));
 assert.ok(mapping.gradientRule&&mapping.rule);
 assert.deepEqual(mapping.outputIndex.length,mapping.outputShape.length);
 const a=Shapes.emit(n,'numpy',()=> 'v["x"]');
 const b=Shapes.emit(n,'torch',()=> 'v["x"]');
 if(n.kind!=='nd_broadcast_add')assert.ok(a&&b,item.name+' backend emitter');
 console.log('SHAPE_INDEX_PASS '+item.name+' '+mapping.outputShape.join('x'));
}
assert.deepEqual([...kinds].sort(),Shapes.kinds);
assert.equal(Shapes.analyze(N('x','input'),{}),null);
assert.throws(()=>Shapes.analyze(N('y','nd_reshape',{b:1,h:1,t:2,d:2}),{x:[2,3]}),/element count/);
assert.throws(()=>Shapes.analyze(N('y','nd_permute',{axis0:0,axis1:0,axis2:2,axis3:3}),{x:[1,2,3,4]}),/permutation/);
assert.throws(()=>Shapes.analyze(N('y','nd_broadcast_add'),{a:[1,2,3,4],b:[1,4,3,4]}),/incompatible/);
const root=path.join(__dirname,'..'),read=p=>fs.readFileSync(path.join(root,p),'utf8');
const scripts=new Set(read('android/teaching-assets.txt').trim().split(/\r?\n/));
assert.ok(scripts.has('shape_index_rules.js'));
assert.ok(read('atlas_server.py').includes('"/shape_index_rules.js":"shape_index_rules.js"'));
for(const page of ['learn.html','capabilities.html']){
 const scripts=[...read(page).matchAll(/<script src="([^"]+)"><\/script>/g)].map(x=>x[1]);
 assert.ok(scripts.indexOf('shape_index_rules.js')<scripts.indexOf('python_codegen.js'));
 assert.ok(scripts.indexOf('shape_index_rules.js')<scripts.indexOf('torch_codegen.js'));
 if(page==='learn.html')assert.ok(scripts.indexOf('shape_index_rules.js')<scripts.indexOf('execution_ir.js'));
}
const html=read('learn.html');
assert.ok(html.includes('id="shapeIndexInspector"')&&html.includes('ShapeRules.analyze(n,portShapes)'));
assert.ok(html.includes('反向梯度'));
console.log('SHAPE_INDEX_CONTRACTS_PASS exact index lineage and forward/IR/static parity');
