'use strict';
// Synthetic contract fixtures only. They MUST NOT be shipped as REAL_CAPTURE samples.
const assert=require('node:assert/strict');
const V=require('../model_trace.js');
const values=[1,2,3,4],g=[-1,0,1,0];
const summary=(data)=>({shape:[2,2],count:4,mean:data.reduce((a,b)=>a+b,0)/4,
 rms:Math.sqrt(data.reduce((a,b)=>a+b*b,0)/4),min:Math.min(...data),
 max:Math.max(...data),l2:Math.hypot(...data),
 zeroFraction:data.filter(x=>x===0).length/4,sample:data.slice()});
const trace={
 schema:"seethetruth-real-model-trace-v1",
 provenance:{
  kind:"real_model_execution",modelId:"TEST_FIXTURE_NOT_CAPTURED/TinyStories",revision:"a".repeat(40),
  framework:"pytorch",device:"cpu",dtype:"float32",modulePath:"transformer.h.0.mlp.c_fc",
  moduleType:"torch.nn.Linear",objective:"causal_language_modeling_shifted_cross_entropy",
  promptSha256:"b".repeat(64),tokenIds:[1,2,3]
 },
 loss:.5,
 parameter:{identity:"transformer.h.0.mlp.c_fc.weight",weight:summary(values),gradient:summary(g)},
 activation:{input:[1,2,3],output:[2,-1,0]}
};
const frozen=JSON.stringify(trace),result=V.validate(trace);
assert.equal(result.status,"provenance_claim_structurally_valid_not_independently_verified");
assert.ok(result.source.includes("TEST_FIXTURE_NOT_CAPTURED"));
assert.equal(JSON.stringify(trace),frozen);
const fail=(mutate,re)=>{const t=JSON.parse(frozen);mutate(t);assert.throws(()=>V.validate(t),re);};
fail(t=>t.provenance.revision="deadbeef",/immutable|provenance/);
fail(t=>t.provenance.promptSha256="bad",/hashed input/);
fail(t=>t.parameter.weight.shape=[3,2],/weight and gradient|count/);
fail(t=>t.parameter.gradient.count=3,/tensor element count/);
fail(t=>t.parameter.weight.l2=1,/tensor L2/);
fail(t=>t.parameter.weight.sample[0]=100,/sample contradicts/);
fail(t=>t.parameter.gradient.zeroFraction=2,/inconsistent tensor summary/);
fail(t=>t.parameter.identity="another.weight",/invalid model trace/);
fail(t=>t.provenance.tokenIds=[1.2,3],/bounded real token/);
fail(t=>t.activation.input=[NaN],/invalid activation/);
console.log('Model trace contract PASS: honest structurally-only provenance; tensor shape, RMS/L2, range, prompt hash, token IDs and false evidence rejected');
