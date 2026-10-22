'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const R=require('../block_extension_registry.js'),L=require('../learning_lab.js'),IR=require('../tensor_ir.js'),S=require('../semantic_state.js'),C=require('../compute_microscope.js');
assert.ok(R.packs.includes('named_axis'));assert.ok(R.recipes.some(x=>x.id==='extension_named_axis_composition'));
for(const kind of ['axis_tensor_source','axis_feature_bias','axis_named_add','axis_named_transpose','axis_time_mean'])
 assert.ok(R.kinds.includes(kind)&&L.KINDS.includes(kind));
const preset=L.preset('extension_named_axis_composition'),run=L.evaluate(preset),frame=C.learning(preset,run,L);
assert.equal(run.steps.length,7);assert.equal(C.validate(frame),frame);
const v=Object.fromEntries(run.steps.map(s=>[s.id,s.values]));
for(let t=0;t<3;t++)for(let d=0;d<3;d++)assert.equal(v.add[t][d],v.x[t][d]+v.bias[0][d]);
for(let t=0;t<3;t++)for(let d=0;d<3;d++)assert.equal(v.swap[d][t],v.add[t][d]);
assert.deepEqual(v.swap,v.native);
assert.ok(v.delta.flat().every(x=>x===0));
for(let d=0;d<3;d++)assert.ok(Math.abs(v.mean[0][d]-v.add.reduce((sum,row)=>sum+row[d],0)/3)<1e-13);
const sub=frame.steps.find(x=>x.id==='swap').substeps[0];
assert.equal(sub.kind,'named_axis_transform');
for(let t=0;t<3;t++)for(let d=0;d<3;d++){
 const hit=C.dot(frame,sub.id,d,t);assert.ok(hit.reconstructable&&hit.error<=1e-12);
}
const originalRecipe=R.recipe('extension_named_axis_composition');
originalRecipe.nodes[0].params.seed=89;
assert.equal(R.recipe('extension_named_axis_composition').nodes[0].params.seed,2,'recipes must be cloned and immutable to the caller');
assert.equal(R.recipe('bogus'),null);
const mismatch=L.preset('extension_named_axis_composition');
mismatch.nodes.find(x=>x.id==='bias').params.width=4;
assert.throws(()=>L.evaluate(mismatch),/named feature bias/);
const altered=L.preset('extension_named_axis_composition');
altered.nodes.find(x=>x.id==='bias').params.scale=3;
const now=L.evaluate(altered),values=Object.fromEntries(now.steps.map(z=>[z.id,z.values]));
for(let t=0;t<3;t++)for(let d=0;d<3;d++)
 assert.ok(Math.abs(values.add[t][d]-v.add[t][d]-(d+1))<=1e-14,'bias scale changes one named feature channel across all tokens');
const invalid=JSON.parse(JSON.stringify(altered));invalid.nodes.find(x=>x.id==='x').params.tokens=9;
assert.throws(()=>L.evaluate(invalid),/tokens/);
const meta=R.recipes.find(x=>x.id==='extension_named_axis_composition');
assert.ok(meta.title&&meta.summary&&meta.pack==='named_axis');
const src=fs.readFileSync(path.join(__dirname,'../learn.html'),'utf8');
assert.ok(src.includes('installedRecipes=window.SeetheTruthBlockExtensions?.recipes')&&src.includes('insertBefore(b,blankRecipeButton)'));
const allow=fs.readFileSync(path.join(__dirname,'../android/teaching-assets.txt'),'utf8').split(/\r?\n/);
assert.ok(allow.includes('axis_block_pack.js'));
const page=fs.readFileSync(path.join(__dirname,'../capabilities.html'),'utf8');
assert.ok(page.includes('src="axis_block_pack.js"'),'capability matrix must load exact same module registry dependencies');
console.log('NAMED_AXIS_PACK_PASS 5 free-composition blocks, dynamic recipe, exact broadcast/transpose/mean, changed parameters and negative port shape checks');
