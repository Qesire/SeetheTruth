'use strict';
// VLM acceptance: actual installed DAGs, finite numerics, exact token provenance,
// causal nonleakage, compressed prefill and strict rejection of invalid contracts.
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const Pack=require('../vlm_block_pack.js'), X=require('../block_extension_registry.js');
const L=require('../learning_lab.js'),R=require('../graph_readiness.js');
const C=require('../concept_flow.js'),Core=require('../learning_experience_core.js');
const Numpy=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
const Inference=require('../inference_ops.js');
const flatten=a=>a.flat(Infinity), vals=run=>Object.fromEntries(run.steps.map(s=>[s.id,s.values]));
assert.equal(Pack.name,'vlm');
assert.equal(Pack.recipes.length,4);
for(const kind of Object.keys(Pack.defs)){
 assert.ok(X.isKind(kind),'VLM kind must be installed in common extension registry');
 assert.ok(L.KINDS.includes(kind),'VLM kind must participate in unrestricted graph composition');
 assert.ok(Pack.defs[kind].source.startsWith('https://'));
}
const names=['extension_vlm_early_fusion','extension_vlm_compressed_prefix','extension_vlm_cross_attention','extension_vlm_query_patch_selection'];
const graphs=names.map(id=>L.preset(id));
// Guided graphs must start entirely inside the visible coordinate plane;
 // negative positions used to hide the visual pool/dense comparison at y<0.
for(const graph of graphs)for(const node of graph.nodes){
 assert.ok(Number.isFinite(node.x)&&Number.isFinite(node.y)&&node.x>=0&&node.y>=0,
  'guided VLM node outside initial visible graph plane: '+graph.title+'/'+node.id);
}
const runs=graphs.map((graph,i)=>{
 const readiness=R.check(graph,{numeric:true});
 assert.equal(readiness.executable,true,names[i]+' not a connected runnable graph');
 const numeric=L.evaluate(graph),symbolic=C.evaluate(graph);
 assert.equal(numeric.steps.length,graph.nodes.length);
 for(const step of numeric.steps){
  assert.deepEqual(step.shape,symbolic.steps.find(s=>s.id===step.id).shape,
   names[i]+'/'+step.id+' numeric vs shape');
  assert.ok(flatten(step.values).every(Number.isFinite),names[i]+'/'+step.id+' finite');
 }
 return numeric;
});
const base=vals(runs[0]),short=vals(runs[1]),cross=vals(runs[2]);
assert.deepEqual(base.prefix,base.vision_projector.concat(base.text_embedding),
 'visual-first concat must preserve every source coordinate');
assert.deepEqual(base.text_context,base.context.slice(4),
 'text readout indexes after visual prefix, not before');
assert.equal(base.prefix.length,6);
assert.equal(base.context.length,6);
assert.deepEqual([base.language_head.length,base.language_head[0].length],[2,8]);
assert.deepEqual(short.visual_pool,[
 base.vision_projector[0].map((v,j)=>(v+base.vision_projector[1][j])/2),
 base.vision_projector[2].map((v,j)=>(v+base.vision_projector[3][j])/2)
]);
assert.deepEqual(short.prefix,short.visual_pool.concat(short.text_embedding));
assert.equal(short.prefix.length,4);
assert.deepEqual(short.text_context,short.context.slice(2));
assert.deepEqual([short.language_head.length,short.language_head[0].length],[2,8]);
assert.deepEqual([cross.cross_attention.length,cross.cross_attention[0].length],[2,4]);
assert.deepEqual([cross.language_head.length,cross.language_head[0].length],[2,8]);
for(const [number,run] of [[6,runs[0]],[4,runs[1]]]){
 const value=vals(run);
 assert.deepEqual(value.mask.map(row=>row.length),Array(number).fill(number));
 value.mask.forEach((row,i)=>row.forEach((v,j)=>assert.equal(v,j<=i?1:0)));
 value.probabilities.forEach((row,i)=>{
  row.forEach((w,j)=>{if(j>i)assert.equal(w,0,'future token must be exactly blocked');});
  assert.ok(Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-12);
 });
 const maxCausalPositions=number*(number+1)/2;
 assert.equal(value.mask.flat().reduce((a,b)=>a+b,0),maxCausalPositions);
 // An arbitrary change to any future Value row must not alter earlier context:
 // the native weighted-sum result is exactly the causal weight-vector dot V.
 const v=value.value;
 const future=number-1;
 for(let row=0;row<future;row++){
  const changed=v.map(r=>r.slice());changed[future][0]+=123;
  for(let col=0;col<v[0].length;col++){
   const before=value.probabilities[row].reduce((acc,w,j)=>acc+w*v[j][col],0);
   const after=value.probabilities[row].reduce((acc,w,j)=>acc+w*changed[j][col],0);
   assert.ok(Math.abs(before-after)<1e-12,'future Value leaks to preceding query');
  }
 }
}
// Independent dense-vs-TopK visual readout oracle, spatial indices and a bound.
const routed=vals(runs[3]);
assert.deepEqual([routed.top_patches.length,routed.top_patches[0].length],[2,2]);
assert.deepEqual(routed.patch_coordinates,routed.top_patches.flat().map(id=>
 [Math.floor(id/2),id%2]));
const scoreOracle=(q,k)=>q.map(query=>k.map(key=>
 query.reduce((sum,x,d)=>sum+x*key[d],0)/Math.sqrt(query.length)));
const rawScores=scoreOracle(routed.text_embedding,routed.visual_projector);
for(let t=0;t<routed.text_embedding.length;t++){
 const scores=rawScores[t],indices=scores.map((value,index)=>({value,index}))
  .sort((a,b)=>b.value-a.value||a.index-b.index).slice(0,2).map(x=>x.index);
 assert.deepEqual(routed.top_patches[t],indices,'Top-K must select the original visual Patch IDs');
 for(let j=0;j<scores.length;j++)
  assert.ok(Math.abs(routed.scale_scores[t][j]-scores[j])<1e-12,'real QK score should agree with independent scalar dot product');
 const max=Math.max(...scores),exp=scores.map(x=>Math.exp(x-max)),Z=exp.reduce((a,b)=>a+b,0);
 const weights=exp.map(x=>x/Z);
 const value=routed.visual_projector;
 const dense=value[0].map((_,d)=>weights.reduce((sum,w,j)=>sum+w*value[j][d],0));
 const included=new Set(indices),keptMass=weights.reduce((sum,w,j)=>sum+(included.has(j)?w:0),0);
 const dropped=1-keptMass;
 assert.ok(dropped>0&&dropped<1);
 const sparse=value[0].map((_,d)=>indices.reduce((sum,j)=>sum+weights[j]*value[j][d],0)/keptMass);
 const complement=value[0].map((_,d)=>weights.reduce((sum,w,j)=>
  sum+(included.has(j)?0:w*value[j][d]),0)/dropped);
 let squaredError=0;
 for(let d=0;d<dense.length;d++){
  assert.ok(Math.abs(routed.dense_visual[t][d]-dense[d])<1e-12,'dense operator must match independent oracle');
  assert.ok(Math.abs(routed.sparse_visual[t][d]-sparse[d])<1e-12,'native sparse operator must match independent oracle');
  assert.ok(Math.abs(routed.readout_delta[t][d]-(dense[d]-sparse[d]))<1e-12);
  assert.ok(Math.abs((dense[d]-sparse[d])-dropped*(complement[d]-sparse[d]))<1e-12,
   'exact discarded-attention-mass decomposition must hold');
  squaredError+=(dense[d]-sparse[d])**2;
 }
 const R=Math.max(...value.map(v=>Math.hypot(...v)));
 assert.ok(Math.sqrt(squaredError)<=2*dropped*R+1e-12,'bounded value norm implies rigorous readout error bound');
}
// Source-aligned Cross-Attention: Q=text Embedding, K=V=visual projector.
// There are no separate trainable Wq/Wk/Wv nodes on this lesson graph.
const crossGraph=graphs[2];
assert.equal(crossGraph.nodes.some(n=>["query","key","value"].includes(n.id)),false);
for(let t=0;t<cross.text_embedding.length;t++){
 const q=cross.text_embedding[t],vision=cross.visual_projector;
 const score=vision.map(v=>q.reduce((acc,x,d)=>acc+x*v[d],0)/Math.sqrt(q.length));
 const m=Math.max(...score),exp=score.map(v=>Math.exp(v-m)),z=exp.reduce((a,b)=>a+b,0);
 const probabilities=exp.map(v=>v/z);
 for(let d=0;d<q.length;d++){
  const context=vision.reduce((acc,v,j)=>acc+probabilities[j]*v[d],0);
  assert.ok(Math.abs(cross.cross_attention[t][d]-context)<1e-12,
   "actual Cross-Attention must use text Query and shared visual K/V");
  assert.ok(Math.abs(cross.text_residual[t][d]-q[d]-context)<1e-12,
   "Cross-Attention residual must add the original text embedding");
 }
}
const coordinates=runs[3].steps.find(step=>step.id==="patch_coordinates").extensionSubsteps?.[0];
assert.ok(coordinates&&Array.isArray(coordinates.selectedRows),
 "Patch coordinate trace must declare its actual selected-Query origin");
assert.deepEqual(coordinates.selectedRows,
 routed.top_patches.flatMap((ids,query)=>ids.map((patchId,rank)=>({
  query,rank,patchId,y:Math.floor(patchId/2),x:patchId%2
 }))));
const localMse=routed.readout_delta.flat().reduce((sum,v)=>sum+v*v,0)/routed.readout_delta.flat().length;
assert.ok(Math.abs(routed.readout_error[0][0]-localMse)<1e-12);
// The complete selected set must reproduce the dense attention result.
const fullSelection=JSON.parse(JSON.stringify(graphs[3]));
fullSelection.nodes.find(n=>n.id==='top_patches').params.topk=4;
const fullValues=vals(L.evaluate(fullSelection));
assert.deepEqual(fullValues.top_patches.map(r=>r.slice().sort((a,b)=>a-b)),[[0,1,2,3],[0,1,2,3]]);
assert.ok(fullValues.readout_delta.flat().every(x=>Math.abs(x)<1e-12));
assert.ok(Math.abs(fullValues.readout_error[0][0])<1e-20);
// Wrong patch-grid indices / overbudget routing are independently rejected.
const gridVisual=Array.from({length:4},()=>[1,2,3,4]);
assert.throws(()=>Pack.execute('vlm_patch_grid_lookup',{gridHeight:2,gridWidth:2},
 {indices:[[0,4]],visual:gridVisual}),/outside selected image Patch grid/);
assert.throws(()=>Pack.execute('vlm_patch_grid_lookup',{gridHeight:2,gridWidth:2},
 {indices:[[0,1.5]],visual:gridVisual}),/Patch index/);
assert.throws(()=>Pack.infer('vlm_patch_grid_lookup',{gridHeight:2,gridWidth:2},
 {indices:[5,2],visual:[4,4]}),/eight teaching rows/);
assert.throws(()=>Pack.infer('vlm_patch_grid_lookup',{gridHeight:1,gridWidth:3},
 {indices:[2,2],visual:[4,4]}),/match actual visual token count/);
const badTopK=JSON.parse(JSON.stringify(graphs[3]));
badTopK.nodes.find(n=>n.id==='top_patches').params.topk=5;
assert.equal(R.check(badTopK,{numeric:true}).executable,false);
assert.throws(()=>L.evaluate(badTopK),/topk|vocab width/);
function verifyTrace(run){
 for(const step of run.steps.filter(s=>X.isKind(s.kind))){
  const item=step.extensionSubsteps?.[0];
  assert.ok(item&&item.out&&item.operands,'all installed pack operators emit actual numeric microscope data');
  if(!item.indexMap)continue;
  item.indexMap.forEach((point,i)=>
   assert.equal(item.out.values[i],item.operands[point.port].values[point.index],
    'wrong coordinate lineage '+step.id+'/'+i));
 }
}
runs.forEach(verifyTrace);
assert.throws(()=>Pack.infer('vlm_token_concat',{}, {visual:[3,4],text:[2,3]}),/feature widths/);
assert.throws(()=>Pack.infer('vlm_token_concat',{}, {visual:[7,4],text:[2,4]}),/length/);
assert.throws(()=>Pack.infer('vlm_visual_pair_pool',{}, {visual:[3,4]}),/even/);
assert.throws(()=>Pack.infer('vlm_text_tail',{prefixTokens:4},
 {context:[4,4],visual:[4,4]}),/leave/);
assert.throws(()=>Pack.infer('vlm_text_tail',{prefixTokens:4},
 {context:[5,4],visual:[3,4]}),/must match connected visual token count/);
assert.throws(()=>Pack.validate('vlm_text_tail',{prefixTokens:0}),/prefixTokens/);
const mutated=JSON.parse(JSON.stringify(graphs[0]));
mutated.nodes.find(n=>n.id==='text_context').params.prefixTokens=7;
assert.equal(R.check(mutated,{numeric:true}).executable,false);
assert.throws(()=>L.evaluate(mutated),/prefixTokens|text token/);
const wrongPrefix=JSON.parse(JSON.stringify(graphs[0]));
wrongPrefix.nodes.find(n=>n.id==='text_context').params.prefixTokens=3;
assert.equal(R.check(wrongPrefix,{numeric:true}).executable,false);
assert.throws(()=>L.evaluate(wrongPrefix),/must match connected visual token count/);
const wrongGrid=JSON.parse(JSON.stringify(graphs[3]));
wrongGrid.nodes.find(n=>n.id==='patch_coordinates').params.gridWidth=3;
assert.equal(R.check(wrongGrid,{numeric:true}).executable,false);
assert.throws(()=>L.evaluate(wrongGrid),/match actual visual token count/);
const badMerge=JSON.parse(JSON.stringify(graphs[0]));
badMerge.edges.find(e=>e.to==='prefix'&&e.port==='text').from='prompt_ids';
assert.equal(R.check(badMerge,{numeric:true}).executable,false,'nonembedded text ids cannot silently align with visual width');
// Release contract: VLM extension operators remain fail-closed in arbitrary Python exporters.
for(const kind of Object.keys(Pack.defs)){
 assert.equal(Numpy.supportedKinds.includes(kind),false,kind+' must not claim NumPy export');
 assert.equal(Torch.supportedKinds.includes(kind),false,kind+' must not claim Torch export');
}
assert.throws(()=>Numpy.compile(graphs[0]),/vlm_token_concat/,
 'unsupported VLM graph must not be silently compiled to NumPy');
assert.throws(()=>Torch.compile(graphs[0]),/vlm_token_concat/,
 'unsupported VLM graph must not be silently compiled to PyTorch');

// Counterexample: retaining one more high-probability visual Patch can increase readout error.
const weights=[0.5,0.3,0.2],values=[[0],[100],[-100]],k=weights.map(w=>[Math.log(w)]);
const dense=weights.reduce((sum,p,j)=>sum+p*values[j][0],0);
const sparse1=Inference.execute('inference_sparse_attention',
 {q:[[1]],k,v:values,indices:[[0]]},{causal:false,queryOffset:0})[0][0];
const sparse2=Inference.execute('inference_sparse_attention',
 {q:[[1]],k,v:values,indices:[[0,1]]},{causal:false,queryOffset:0})[0][0];
assert.ok(Math.abs(dense-10)<1e-12 && Math.abs(sparse1)<1e-12);
assert.ok(Math.abs(sparse2-37.5)<1e-12);
assert.ok(Math.abs(dense-sparse2)>Math.abs(dense-sparse1),
 'Top-K cardinality does not guarantee monotonic readout error under normalized Softmax');

const route=Core.inspectCourse('vlm');
assert.equal(route.target.kind,'guided_graph');
assert.equal(route.target.id,names[0]);
assert.equal(Core.openPreset(names[0],{numeric:true}).graph.nodes.length,graphs[0].nodes.length);
const root=path.resolve(__dirname,'..'),read=p=>fs.readFileSync(path.join(root,p),'utf8');
for(const page of ['learn.html','capabilities.html'])
 assert.ok(read(page).includes('src="vlm_block_pack.js"'),page+' missing trusted VLM pack script');
assert.ok(read('android/teaching-assets.txt').split(/\r?\n/).includes('vlm_block_pack.js'));
assert.ok(read('atlas_server.py').includes('"/vlm_block_pack.js":"vlm_block_pack.js"'));
assert.ok(read('teaching_hub.js').includes('l.target.id')&&read('learn.html').includes('requestedMission'),
 'curriculum route must deep-link to its actual graph preset');
console.log('VLM_LEARNING_PASS four source-bound graph recipes, query-selected Patch indices and dense/sparse error bound, visual/text alignment, prefix causal mask, compressed tokens, negative contracts and PC/Android routes');
