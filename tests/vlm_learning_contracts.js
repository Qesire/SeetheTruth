'use strict';
// VLM acceptance: actual installed DAGs, finite numerics, exact token provenance,
// causal nonleakage, compressed prefill and strict rejection of invalid contracts.
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const Pack=require('../vlm_block_pack.js'), X=require('../block_extension_registry.js');
const L=require('../learning_lab.js'),R=require('../graph_readiness.js');
const C=require('../concept_flow.js'),Core=require('../learning_experience_core.js');
const flatten=a=>a.flat(Infinity), vals=run=>Object.fromEntries(run.steps.map(s=>[s.id,s.values]));
assert.equal(Pack.name,'vlm');
assert.equal(Pack.recipes.length,3);
for(const kind of Object.keys(Pack.defs)){
 assert.ok(X.isKind(kind),'VLM kind must be installed in common extension registry');
 assert.ok(L.KINDS.includes(kind),'VLM kind must participate in unrestricted graph composition');
 assert.ok(Pack.defs[kind].source.startsWith('https://'));
}
const names=['extension_vlm_early_fusion','extension_vlm_compressed_prefix','extension_vlm_cross_attention'];
const graphs=names.map(id=>L.preset(id));
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
assert.throws(()=>Pack.infer('vlm_text_tail',{prefixTokens:4}, {context:[4,4]}),/leave/);
assert.throws(()=>Pack.validate('vlm_text_tail',{prefixTokens:0}),/prefixTokens/);
const mutated=JSON.parse(JSON.stringify(graphs[0]));
mutated.nodes.find(n=>n.id==='text_context').params.prefixTokens=7;
assert.equal(R.check(mutated,{numeric:true}).executable,false);
assert.throws(()=>L.evaluate(mutated),/prefixTokens|text token/);
const badMerge=JSON.parse(JSON.stringify(graphs[0]));
badMerge.edges.find(e=>e.to==='prefix'&&e.port==='text').from='prompt_ids';
assert.equal(R.check(badMerge,{numeric:true}).executable,false,'nonembedded text ids cannot silently align with visual width');
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
console.log('VLM_LEARNING_PASS three source-bound graph recipes, visual/text alignment, prefix causal mask, compressed tokens, negative contracts and PC/Android routes');
