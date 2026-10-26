'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const Rules=require('../portable_operator_rules.js');
const NP=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
const expected=['matmul','nd_batched_matmul','add','mul','subtract','divide','nd_broadcast_add','nd_broadcast_mul',
 'scale','nd_scale','residual_scale','bias','square','temperature','low_rank_project',
 'row_bias_add','weighted_sum','slice_cols','slice_rows','nd_cache_read','diffusion_ddim_step'];
assert.deepEqual(Rules.kinds,[...expected].sort());
const params={factor:0.25,bias:-0.3,temperature:0.8,start:1,width:2,length:2,alphaT:0.36,alphaPrevious:0.64};
for(const kind of Rules.kinds){
 assert.ok(NP.supportedKinds.includes(kind),kind+' NumPy supported');
 assert.ok(Torch.supportedKinds.includes(kind),kind+' Torch supported');
 const seen=[];
 const expr=Rules.emit({kind,params},port=>{seen.push(port);return 'v["'+port+'"]';});
 assert.equal(typeof expr,'string');assert.ok(expr.includes('v['),kind);
 assert.ok(!expr.includes('undefined')&&!expr.includes('NaN'),kind);
 assert.ok(seen.length>0);
 if(kind==='diffusion_ddim_step'){
  const c=Math.sqrt(params.alphaPrevious/params.alphaT);
  const d=Math.sqrt(1-params.alphaPrevious)-Math.sqrt(params.alphaPrevious*(1-params.alphaT)/params.alphaT);
  assert.equal(expr,'('+JSON.stringify(c)+'*v["x"] + '+JSON.stringify(d)+'*v["epsilon"])');
 }
 if(kind==='low_rank_project')assert.deepEqual(seen,['x','down','up']);
 if(kind==='nd_cache_read')assert.equal(expr,'v["x"][:,:, 1:3,:]');
}
assert.equal(Rules.emit({kind:'unsupported',params:{}},()=>{throw Error('bad call');}),null);
assert.throws(()=>Rules.emit(null,()=>0),/invalid/);
assert.throws(()=>Rules.emit({kind:'add',params:{}},null),/invalid/);
const root=path.join(__dirname,'..'),read=x=>fs.readFileSync(path.join(root,x),'utf8');
const assets=new Set(read('android/teaching-assets.txt').split(/\r?\n/).map(s=>s.trim()));
assert.ok(assets.has('portable_operator_rules.js'));
assert.ok(read('atlas_server.py').includes('"/portable_operator_rules.js":"portable_operator_rules.js"'));
for(const file of ['learn.html','capabilities.html']){
 const scripts=[...read(file).matchAll(/<script src="([^"]+)"><\/script>/g)].map(m=>m[1]);
 const idx=scripts.indexOf('portable_operator_rules.js');
 assert.ok(idx>=0&&idx<scripts.indexOf('python_codegen.js'),file+' NumPy loading order');
 assert.ok(idx<scripts.indexOf('torch_codegen.js'),file+' Torch loading order');
}
for(const file of ['python_codegen.js','torch_codegen.js'])assert.ok(read(file).includes('Portable.emit(n,'),file+' actual integration');
console.log('PORTABLE_OPERATOR_RULES_PASS '+Rules.kinds.length+' common math expressions and web/Android packaging');
