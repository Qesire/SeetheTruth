'use strict';
const assert=require('node:assert/strict');
const Lab=require('../kv_cache_lab.js');
const IR=require('../tensor_ir.js');
function close(actual,expected,tol=1e-12){
 assert.equal(actual.length,expected.length);
 for(let i=0;i<actual.length;i++)assert.ok(Math.abs(actual[i]-expected[i])<tol,
  'Mismatch '+i+': '+actual[i]+' != '+expected[i]);
}
function run(config){
 const s=Lab.Session(config),start=s.snapshot();
 assert.equal(start.cache.length,0);
 assert.deepEqual(start.cache.key,[]);
 const settings=s.config,states=[],history=[];
 for(let i=0;i<=settings.decode;i++){
   const after=s.advance();states.push(after);history.push(after.last);
   assert.equal(after.cache.length,after.cache.key.length);
   assert.equal(after.cache.length,after.cache.value.length);
   assert.equal(after.history.length,i+1);
   assert.equal(after.last.cacheAfter,after.cache.length);
   assert.equal(after.last.keyLength,after.cache.length);
   assert.equal(after.last.queryLength,i===0?settings.prefix:1);
   assert.equal(after.last.phase,i===0?'prefill':'decode');
   assert.deepEqual(after.last.probabilitiesShape,
     [1,settings.heads,i===0?settings.prefix:1,settings.prefix+i]);
   assert.equal(after.last.appended,i===0?settings.prefix:1);
   assert.equal(after.last.mathMatmulFlops,
     2*settings.heads*after.last.queryLength*after.cache.length*
     (settings.headDim+settings.valueDim));
   assert.deepEqual(after.cache.key.at(-1),s.project('k',after.cache.length-1));
   assert.deepEqual(after.cache.value.at(-1),s.project('v',after.cache.length-1));
   if(i){
     // The old prefix has actually been stored; it must not be reprojected.
     assert.deepEqual(after.cache.key.slice(0,-1),states[i-1].cache.key);
     assert.deepEqual(after.cache.value.slice(0,-1),states[i-1].cache.value);
     assert.equal(after.last.avoidedOldProjectionTokens,after.cache.length-1);
   }
   const p=IR.plane(after.last.attention,{Hq:0}).view;
   for(let row=0;row<p.length;row++){
     close([p[row].reduce((a,b)=>a+b,0)],[1]);
     if(i===0)for(let j=row+1;j<p[row].length;j++)assert.equal(p[row][j],0);
   }
 }
 const full=Lab.referenceFull(settings);
 for(let i=0;i<history.length;i++){
   const item=history[i],positions=item.queryPositions;
   for(let k=0;k<positions.length;k++)close(item.output[k],full.output[positions[k]]);
 }
 const state=s.snapshot();
 assert.equal(state.cursor,settings.total);
 assert.throws(()=>s.advance(),/already processed/);
 assert.equal(state.history.length,settings.decode+1);
 // A saved snapshot is structurally immutable relative to subsequent mutation.
 const old=JSON.stringify(states[0]);
 const again=s.reset();
 assert.equal(again.cache.length,0);assert.equal(again.cursor,0);
 assert.equal(JSON.stringify(states[0]),old);
 assert.deepEqual(s.advance(),states[0]);
 const costs=Lab.countPlan(settings);
 assert.equal(costs.projectedPositionsWithCache,settings.total);
 assert.equal(costs.attributedMatmulFlops,history.reduce((a,b)=>a+b.mathMatmulFlops,0));
 assert.ok(costs.fullPrefixRecomputeMatmulFlops>=costs.attributedMatmulFlops);
 assert.ok(costs.fullPrefixRecomputeProjectedPositions>=settings.total);
}
run({prefix:3,decode:4,heads:2,kvHeads:1,headDim:4,valueDim:4,rope:true});
run({prefix:2,decode:5,heads:4,kvHeads:2,headDim:2,valueDim:3,rope:false});
run({prefix:4,decode:2,heads:1,kvHeads:1,headDim:6,valueDim:2,rope:true});
for(const opts of [{prefix:8,decode:8},{prefix:0},{decode:0},{heads:3,kvHeads:2},
 {headDim:3},{valueDim:0},{rope:'true'}]){
 assert.throws(()=>Lab.Session(opts),'Invalid config must fail closed: '+JSON.stringify(opts));
}
const a=Lab.Session({prefix:2,decode:1});
assert.throws(()=>a.project('k',99),/invalid projection/);
assert.throws(()=>a.project('bad',1),/invalid projection/);
console.log('KV Cache real append, causal decode equivalence, immutable snapshots, reset, GQA and cost invariants PASS');
