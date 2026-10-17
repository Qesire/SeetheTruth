'use strict';
/* Independent tiny-number oracles for the semantic multi-rank Tensor IR. */
const assert=require('node:assert/strict');
const IR=require('../tensor_ir.js');
const L=require('../learning_lab.js');
function eq(a,b,eps=1e-12){assert.ok(Math.abs(a-b)<=eps,String(a)+' != '+String(b));}
function fails(fn,pattern){assert.throws(fn,pattern);}
const a=IR.fromMatrix([[1,2,3],[4,5,6]],['T','D']);
assert.deepEqual(a.shape,[2,3]);assert.deepEqual(a.axes,['T','D']);
assert.deepEqual(IR.reshape(a,[1,2,3],['B','T','D']).data,a.data);
const t=IR.transpose(IR.reshape(a,[2,3],['T','D']),[1,0]);
assert.deepEqual(t.shape,[3,2]);assert.deepEqual(IR.plane(t).view,[[1,4],[2,5],[3,6]]);
fails(()=>IR.reshape(a,[3,3],['B','T']),/reshape changes element count/);
fails(()=>IR.transpose(a,[0,0]),/bijection/);
fails(()=>IR.make([2,2],['T','T'],[1,2,3,4]),/distinct/);
fails(()=>IR.make([32,32,8],['A','B','D'],Array(8192).fill(1)),/element limit/);
fails(()=>IR.fromMatrix([[1,2],[3]]),/rectangular/);
fails(()=>IR.make([2,2],['T','D'],[1,2,3,Infinity]),/finite/);
const packed=IR.headPack([[1,2,3,4],[5,6,7,8]],2,'Tq','Hq','Dk');
assert.deepEqual(packed.shape,[1,2,2,2]);assert.deepEqual(packed.axes,['B','Hq','Tq','Dk']);
assert.deepEqual(IR.plane(packed,{Hq:0}).view,[[1,2],[5,6]]);
assert.deepEqual(IR.plane(packed,{Hq:1}).view,[[3,4],[7,8]]);
assert.deepEqual(IR.fromHeads(packed),[[1,2,3,4],[5,6,7,8]]);
const rotated=IR.rotary(packed,1);
for(let h=0;h<2;h++)for(let token=0;token<2;token++){
 const before=IR.plane(packed,{Hq:h}).view[token],after=IR.plane(rotated,{Hq:h}).view[token];
 eq(before[0]**2+before[1]**2,after[0]**2+after[1]**2);
}
fails(()=>IR.rotary(IR.headPack([[1,2,3]],1,'Tq','Hq','Dk')),/even Dk/);
{
 const q=[[0,0],[0,0]],k=[[1,0],[0,1]],v=[[2,4],[6,8]];
 const r=IR.gqa(q,k,v,{heads:1,kvHeads:1,causal:false,rope:false});
 assert.deepEqual(r.output,[[4,6],[4,6]]);
 assert.deepEqual(IR.plane(r.stages.probabilities).view,[[.5,.5],[.5,.5]]);
 assert.deepEqual(r.stages.scores.axes,['B','Hq','Tq','Tk']);
 assert.deepEqual(r.shape,[2,2]);
 eq(r.knownMatmulFlops,2*1*2*2*(2+2));
}
{
 const q=[[0,0],[0,0],[0,0]],k=[[0,0],[0,0],[0,0]],v=[[1,3],[5,7],[9,11]];
 const r=IR.gqa(q,k,v,{heads:1,kvHeads:1,causal:true,rope:false});
 assert.deepEqual(r.masks.allowed,[[true,false,false],[true,true,false],[true,true,true]]);
 assert.deepEqual(r.output,[[1,3],[3,5],[5,7]]);
 assert.deepEqual(IR.plane(r.stages.probabilities).view,
   [[1,0,0],[.5,.5,0],[1/3,1/3,1/3]]);
 const short=IR.gqa([[0,0]],k,v,{heads:1,kvHeads:1,causal:true,rope:false});
 assert.equal(short.teachingConfig.queryPositionOffset,2);
 assert.equal(short.teachingConfig.cachedPrefixProvidedAsInput,true);
 assert.equal(short.teachingConfig.createsOrUpdatesKVCache,false);
 assert.deepEqual(short.output,[[5,7]]);
}
{
 const q=[[1,0, 1,0, 1,0, 1,0]];
 const k=[[1,0, 0,1],[0,1, 1,0]];
 const v=[[3,1, 10,20],[7,5, 30,40]];
 const r=IR.gqa(q,k,v,{heads:4,kvHeads:2,causal:false,rope:false});
 const prob=IR.plane(r.stages.probabilities,{Hq:0}).view[0][0];
 const prob2=IR.plane(r.stages.probabilities,{Hq:2}).view[0][0];
 assert.ok(prob>prob2,'different KV heads must affect attention distributions');
 for(let i=0;i<2;i++){
   assert.deepEqual(IR.plane(r.stages.probabilities,{Hq:i}).view,
                    IR.plane(r.stages.probabilities,{Hq:0}).view);
 }
 assert.deepEqual(r.shape,[1,8]);assert.equal(r.teachingConfig.groupSize,2);
}
fails(()=>IR.gqa([[1,2,3,4]],[[1,2]],[[1,2]],{heads:3,kvHeads:1}),/divisible/);
fails(()=>IR.gqa([[1,2]],[[1,2],[2,3]],[[1,2]],{heads:1,kvHeads:1}),/K\/V token/);
fails(()=>IR.gqa([[1,2],[3,4]],[[1,2]],[[1,2]],{heads:1,kvHeads:1,causal:true}),/cannot exceed/);
fails(()=>IR.gqa([[1,2]],[[1]],[[1]],{heads:2,kvHeads:1,rope:true}),/even Dk/);
// Feature switches must actually change the numeric computation.
{
 const q=[[1,2,3,4],[2,-1,0,3],[4,0,-2,1]];
 const k=[[2,0,1,-3],[1,2,-2,0],[0,-2,1,2]];
 const v=[[1,2,-3,4],[4,2,1,0],[-1,3,2,5]];
 const common={heads:2,kvHeads:1};
 const noRope=IR.gqa(q,k,v,{...common,causal:false,rope:false});
 const withRope=IR.gqa(q,k,v,{...common,causal:false,rope:true});
 const masked=IR.gqa(q,k,v,{...common,causal:true,rope:false});
 assert.ok(noRope.output.flat().some((x,i)=>Math.abs(x-withRope.output.flat()[i])>1e-6),
           'RoPE switch must alter positional attention math');
 assert.ok(masked.output.flat().some((x,i)=>Math.abs(x-noRope.output.flat()[i])>1e-6),
           'Causal switch must alter which keys contribute');
 const p=IR.plane(masked.stages.probabilities,{Hq:1}).view;
 for(const row of p)eq(row.reduce((x,y)=>x+y,0),1);
 assert.deepEqual(p[0].slice(1),[0,0]);
}
for(const lesson of ['gqa','decode']){
 const g=L.preset(lesson), result=L.evaluate(g),step=result.steps.find(x=>x.kind==='gqa');
 assert.ok(step,'GQA executable learning block');
 assert.deepEqual(step.shape,[lesson==='decode'?1:3,8]);
 assert.deepEqual(step.internals.q_heads.shape,[1,2,lesson==='decode'?1:3,4]);
 assert.deepEqual(step.internals.probabilities.shape,[1,2,lesson==='decode'?1:3,3]);
 assert.equal(step.internals.probabilities.axes[1],'Hq');
 assert.deepEqual(result,L.evaluate(g),'teaching pass must be deterministic');
 if(lesson==='decode')assert.deepEqual(step.internals.causal_mask.values,[[1,1,1]]);
 else assert.deepEqual(step.internals.causal_mask.values,[[1,0,0],[1,1,0],[1,1,1]]);
}
{
 const g=L.preset('gqa'),n=g.nodes.find(x=>x.id==='gqa');
 n.params.heads=3;n.params.kvHeads=2;fails(()=>L.evaluate(g),/整除/);
 n.params.heads=2;n.params.rope=1;fails(()=>L.evaluate(g),/布尔/);
 n.params.rope=false;n.params.kvHeads=2;
 g.nodes.find(x=>x.id==='k').params.out=8;
 g.nodes.find(x=>x.id==='v').params.out=8;
 assert.deepEqual(L.evaluate(g).steps.find(x=>x.kind==='gqa').shape,[3,8]);
}
console.log('Semantic tensor IR: named axes, transpose, GQA, causal, RoPE, decode and error oracles PASS');
