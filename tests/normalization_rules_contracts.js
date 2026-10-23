'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const S=require('../normalization_rules.js'),NP=require('../python_codegen.js'),PT=require('../torch_codegen.js');
const expected=['softmax','nd_softmax_last','rmsnorm','qk_norm','layernorm','attention','log_softmax','row_logsumexp'];
assert.equal(S.schema,'seethetruth-normalization-semantics-v1');
assert.deepEqual(S.kinds,[...expected].sort());
const sample={kind:'layernorm',params:{epsilon:1e-5}};
for(const kind of expected){
 assert.ok(NP.supportedKinds.includes(kind),'NumPy eligibility '+kind);
 assert.ok(PT.supportedKinds.includes(kind),'Torch eligibility '+kind);
 const n={kind,params:{epsilon:1e-5}},seen=[];
 const a=S.emit(n,'numpy',port=>{seen.push(port);return 'v["'+port+'"]';});
 const b=S.emit(n,'torch',port=>'v["'+port+'"]');
 assert.ok(a&&b,kind+' lowering expressions');
 for(const e of [a,b])assert.ok(!e.includes('undefined')&&!e.includes('NaN'),kind+' valid expression');
 assert.equal(S.descriptions[kind].axis,-1);
 assert.ok(seen.length>=1);
}
assert.equal(S.descriptions.layernorm.epsilonPlacement,'inside_sqrt');
assert.equal(S.descriptions.rmsnorm.epsilonPlacement,'inside_sqrt');
assert.equal(S.descriptions.qk_norm.epsilonPlacement,'inside_sqrt');
assert.equal(S.descriptions.attention.scaleAxis,'q.last');
assert.equal(S.descriptions.softmax.stable,true);
assert.equal(S.emit({kind:'input'},'numpy',()=>{throw Error('called');}),null);
assert.throws(()=>S.emit(sample,'bad',()=>0),/invalid normalization/);
assert.throws(()=>S.emit(sample,'torch',undefined),/invalid normalization/);
const root=path.resolve(__dirname,'..'),read=x=>fs.readFileSync(path.join(root,x),'utf8');
const assets=new Set(read('android/teaching-assets.txt').split(/\r?\n/).map(x=>x.trim()));
assert.ok(assets.has('normalization_rules.js'));
assert.ok(read('atlas_server.py').includes('"/normalization_rules.js":"normalization_rules.js"'));
for(const page of ['learn.html','capabilities.html']){
 const scripts=[...read(page).matchAll(/<script src="([^"]+)"><\/script>/g)].map(m=>m[1]);
 const n=scripts.indexOf('normalization_rules.js');
 assert.ok(n>scripts.indexOf('learning_lab.js'));
 assert.ok(n<scripts.indexOf('python_codegen.js')&&n<scripts.indexOf('torch_codegen.js'));
}
for(const filename of ['python_codegen.js','torch_codegen.js']){
 const code=read(filename);
 assert.ok(code.includes('Norm.emit(n,'),filename+' must use actual shared lowering');
}
console.log('NORMALIZATION_RULES_PASS shared math contracts, shape axes, exporter wiring and offline bundle');
