'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),E=require('../element_lineage.js'),ND=require('../nd_tensor.js');
const node=(id,kind,params={},x=0,y=0)=>({id,kind,params,x,y});
const wire=(from,to,port)=>({from,to,port});
const values=[[0.2,0.5],[0.7,-0.9]],zeros=[[0,0],[0,0]];
const param=()=>node('x','parameter',{values:values.map(r=>[...r]),optimizer:{kind:'adamw',t:0,m:zeros,v:zeros}});
const graph=L.validateGraph({schema:L.SCHEMA,title:'convergent scalar paths',
 nodes:[param(),node('a','scale',{factor:2}),node('b','scale',{factor:3}),
  node('sum','add'),node('sq','square')],
 edges:[wire('x','a','x'),wire('x','b','x'),wire('a','sum','a'),
  wire('b','sum','b'),wire('sum','sq','x')]});
const original=JSON.stringify(graph),trace=L.evaluate(graph);
const report=E.inspect(graph,{targetId:'sq',outputIndex:[0,1],trace});
assert.equal(report.schema,'seethetruth-element-lineage-v1');
assert.equal(report.mode,'bounded-numeric');assert.equal(report.complete,true);
assert.equal(report.truncated,false);assert.deepEqual(report.outputIndex,[0,1]);
assert.equal(JSON.stringify(graph),original);
const x=report.vertices.find(v=>v.key==='x@0,1');
assert.ok(x);assert.equal(x.status,'source');assert.ok(Math.abs(x.sensitivity-25)<1e-10);
assert.ok(Math.abs(report.vertices.find(v=>v.id==='sq').value-6.25)<1e-10);
assert.equal(report.edges.filter(e=>e.to==='x@0,1').length,2);
const lossAt=(g,offset,id,atIndex)=>{
 const copy=JSON.parse(JSON.stringify(g));
 const p=copy.nodes.find(n=>n.id===id);
 let dest=p.params.values;
 for(let i=0;i<atIndex.length-1;i++)dest=dest[atIndex[i]];
 dest[atIndex.at(-1)]+=offset;
 const t=L.evaluate(copy);
 return ND.at(t.steps.at(-1).values,report.outputIndex);
};
const h=1e-6,finite=(lossAt(graph,h,'x',[0,1])-lossAt(graph,-h,'x',[0,1]))/(2*h);
assert.ok(Math.abs(finite-x.sensitivity)<2e-6);
const stat=E.inspect(graph,{targetId:'sq',outputIndex:[0,1]});
assert.equal(stat.mode,'structural');
assert.equal(stat.vertices.find(v=>v.id==='x').sensitivity,null);
assert.equal(stat.vertices.find(v=>v.id==='sq').sensitivity,1);
assert.throws(()=>E.inspect(graph,{targetId:'sq',outputIndex:[0,2]}),/out of bounds/);
assert.throws(()=>E.inspect(graph,{targetId:'sq',outputIndex:[1]}),/out of bounds/);
assert.throws(()=>E.inspect(graph,{targetId:'absent'}),/missing/);
assert.throws(()=>E.inspect(graph,{targetId:'sq',trace:{steps:trace.steps.slice(1)}}),/trace does not match/);
const limited=E.inspect(graph,{targetId:'sq',trace,outputIndex:[0,1],maxVertices:2});
assert.equal(limited.complete,false);assert.equal(limited.truncated,true);
const attention=L.preset('nd_attention_lesson'),attTrace=L.evaluate(attention);
const q=E.inspect(attention,{targetId:'perm',outputIndex:[0,1,0,2],trace:attTrace});
assert.equal(q.complete,true);
assert.ok(q.vertices.some(v=>v.id==='queries'&&v.status==='source'));
assert.ok(q.vertices.some(v=>v.id==='keys'&&v.status==='source'));
assert.ok(q.vertices.some(v=>v.id==='bias'&&v.status==='source'));
assert.ok(q.vertices.some(v=>v.id==='batch_scores'&&v.kind==='nd_batched_matmul'));
assert.ok(q.edges.some(e=>e.port==='a'&&e.term.startsWith('k=')));
const source=q.vertices.find(v=>v.id==='queries'&&v.sensitivity!==0&&v.sensitivity!==null);
assert.ok(source,'query source derivative must propagate');
const replay=JSON.parse(JSON.stringify(attention)),srcNode=replay.nodes.find(n=>n.id==='queries');
const queryValue=attTrace.steps.find(s=>s.id==='queries').values;
srcNode.kind='nd_parameter';
srcNode.params={values:queryValue,optimizer:{kind:'adamw',t:0,m:ND.build(ND.shape(queryValue),()=>0),v:ND.build(ND.shape(queryValue),()=>0)}};
const perturbed=offset=>{
 const copy=JSON.parse(JSON.stringify(replay));
 const array=copy.nodes.find(n=>n.id==='queries').params.values;
 let target=array;for(const v of source.index.slice(0,-1))target=target[v];
 target[source.index.at(-1)]+=offset;
 return ND.at(L.evaluate(copy).steps.find(s=>s.id==='perm').values,[0,1,0,2]);
};
const finiteND=(perturbed(h)-perturbed(-h))/(2*h);
assert.ok(Math.abs(finiteND-source.sensitivity)<3e-5,'4D graph gradient: '+finiteND+' vs '+source.sensitivity);
const noTrace=E.inspect(attention,{targetId:'perm',outputIndex:[0,1,0,2]});
assert.equal(noTrace.complete,true);
assert.equal(noTrace.vertices.find(v=>v.id==='queries').sensitivity,null);
const divergent=E.inspect(L.preset('ffn'),{targetId:'act',outputIndex:[0,1]});
assert.equal(divergent.complete,false);assert.equal(divergent.frontiers.length,1);
assert.equal(divergent.vertices.find(v=>v.id==='act').status,'computed');
assert.equal(divergent.vertices.find(v=>v.id==='up').status,'boundary');
const soft=L.preset('sampling');
const y=E.inspect(soft,{targetId:'prob',outputIndex:[0,0],trace:L.evaluate(soft)});
assert.equal(y.complete,true);
assert.equal(y.edges.filter(e=>e.from==='prob@0,0').length,8);
const sum=y.edges.filter(e=>e.from==='prob@0,0').reduce((a,e)=>a+e.localDerivative,0);
assert.ok(Math.abs(sum)<1e-10,'softmax Jacobian row sums to zero');
const fs=require('node:fs'),path=require('node:path'),root=require('node:path').resolve(__dirname,'..');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const resources=new Set(read('android/teaching-assets.txt').split(/\r?\n/).map(s=>s.trim()));
assert.ok(resources.has('element_lineage.js'));
assert.ok(read('atlas_server.py').includes('"/element_lineage.js":"element_lineage.js"'));
const html=read('learn.html');
const scripts=[...html.matchAll(/<script src="([^"]+)"><\/script>/g)].map(x=>x[1]);
assert.ok(scripts.indexOf('element_lineage.js')>scripts.indexOf('shape_index_rules.js'));
assert.ok(scripts.indexOf('element_lineage.js')>scripts.indexOf('graph_readiness.js'));
assert.ok(scripts.indexOf('element_lineage.js')>scripts.indexOf('nd_tensor.js'));

assert.ok(scripts.indexOf('element_lineage.js')<scripts.indexOf('training_session.js'));
assert.ok(html.includes('id="graphLineageSection"')&&html.includes('ElementLineage.inspect(graph,'));
assert.ok(html.includes('id="traceGraphElement"'));

/* AutoResearcher M3/CR: reconstruct the exact last-axis Jacobian, not only a
 * same-shaped tensor's scalar chain. Independent central differences perturb
 * the actual upstream parameter through the complete LearningLab forward. */
const scalarForward=(g,target,sourceIdx,offset,outputIdx=[0,1])=>{
 const clone=JSON.parse(JSON.stringify(g));
 let values=clone.nodes.find(n=>n.id==='x').params.values;
 for(const v of sourceIdx.slice(0,-1))values=values[v];
 values[sourceIdx.at(-1)]+=offset;
 return ND.at(L.evaluate(clone).steps.find(step=>step.id===target).values,outputIdx);
};
const numericalDerivative=(g,target,sourceIdx)=>
 (scalarForward(g,target,sourceIdx,h)-scalarForward(g,target,sourceIdx,-h))/(2*h);
const oneOp=(kind,params={})=>L.validateGraph({schema:L.SCHEMA,title:'lineage '+kind,
 nodes:[param(),node('act',kind,params)],edges:[wire('x','act','x')]});
for(const kind of ['silu','gelu_erf','gelu_tanh','mish','relu_squared','leaky_relu','elu','hardswish']){
 const g=oneOp(kind,kind==='leaky_relu'?{slope:.12}:{});
 const audited=E.inspect(g,{targetId:'act',outputIndex:[0,1],trace:L.evaluate(g)});
 assert.equal(audited.complete,true,kind+' full dependency closure');
 const source=audited.vertices.find(v=>v.key==='x@0,1');
 assert.ok(source&&Number.isFinite(source.sensitivity),kind+' analytic derivative');
 assert.ok(Math.abs(source.sensitivity-numericalDerivative(g,'act',[0,1]))<3e-5,
  kind+' finite difference: '+source.sensitivity);
 const symbolic=E.inspect(g,{targetId:'act',outputIndex:[0,1]});
 assert.equal(symbolic.vertices.find(v=>v.key==='x@0,1').sensitivity,null,
  kind+' must not invent a numerical derivative in structural mode');
}
// A genuine rank-four activation exercises ND tensor indexing, not a 2D alias.
for(const kind of ['nd_silu','nd_gelu_erf','nd_mish']){
 const tensor=[[[[.13,-.7]],[[.4,1.1]]]];
 const initial={values:tensor,optimizer:{kind:'adamw',t:0,
  m:ND.build([1,2,1,2],()=>0),v:ND.build([1,2,1,2],()=>0)}};
 const g=L.validateGraph({schema:L.SCHEMA,title:'4D '+kind,nodes:[
  node('x','nd_parameter',initial),node('act',kind)
 ],edges:[wire('x','act','x')]});
 const outputIndex=[0,1,0,1],trace4=L.evaluate(g);
 const audited=E.inspect(g,{targetId:'act',outputIndex,trace:trace4});
 assert.equal(audited.complete,true,kind+' rank-four lineage');
 const source=audited.vertices.find(v=>v.key==='x@0,1,0,1');
 assert.ok(source&&Number.isFinite(source.sensitivity));
 const forward=delta=>{
  const copy=JSON.parse(JSON.stringify(g));
  copy.nodes.find(n=>n.id==='x').params.values[0][1][0][1]+=delta;
  return ND.at(L.evaluate(copy).steps.find(step=>step.id==='act').values,outputIndex);
 };
 assert.ok(Math.abs(source.sensitivity-(forward(h)-forward(-h))/(2*h))<3e-5,
  kind+' four-axis finite difference');
}
for(const kind of ['rmsnorm','qk_norm','layernorm']){
 const g=oneOp(kind,{epsilon:.08});
 const audited=E.inspect(g,{targetId:'act',outputIndex:[0,1],trace:L.evaluate(g)});
 assert.equal(audited.complete,true,kind+' full row coupling');
 assert.equal(audited.edges.length,2,kind+' uses both same-row features');
 for(const sourceIdx of [[0,0],[0,1]]){
  const source=audited.vertices.find(v=>v.key==='x@'+sourceIdx.join(','));
  assert.ok(source&&Number.isFinite(source.sensitivity),kind+' row sensitivity');
  assert.ok(Math.abs(source.sensitivity-numericalDerivative(g,'act',sourceIdx))<3e-5,
   kind+' finite difference: '+sourceIdx.join(',')+' / '+source.sensitivity);
 }
 const symbolic=E.inspect(g,{targetId:'act',outputIndex:[0,1]});
 assert.equal(symbolic.edges.length,2);
 assert.equal(symbolic.vertices.find(v=>v.key==='x@0,0').sensitivity,null);
}
const staleGraph=JSON.parse(JSON.stringify(graph));
staleGraph.nodes.find(n=>n.id==='a').params.factor=4;
assert.throws(()=>E.inspect(staleGraph,{targetId:'sq',outputIndex:[0,1],trace}),
 /trace does not match current graph/,'same-shaped parameter edits invalidate numeric evidence');
const poisonedTrace=JSON.parse(JSON.stringify(trace));
poisonedTrace.steps.find(step=>step.id==='x').values[0][1]+=0.125;
assert.throws(()=>E.inspect(graph,{targetId:'sq',outputIndex:[0,1],trace:poisonedTrace}),
 /trace does not match current graph/,'altered trace values cannot carry bounded-numeric status');
const incompleteTrace=JSON.parse(JSON.stringify(trace));
delete incompleteTrace.steps.find(step=>step.id==='x').values;
assert.throws(()=>E.inspect(graph,{targetId:'sq',outputIndex:[0,1],trace:incompleteTrace}),
 /trace does not match current graph/,'partial traces must not silently downgrade their evidence');
console.log('ELEMENT_LINEAGE_PASS analytic activations, row normalization Jacobians, stale numeric trace rejection, shared-path VJP, 4D GEMM, softmax');
