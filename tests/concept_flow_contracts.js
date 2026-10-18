'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),C=require('../concept_flow.js'),B=require('../block_catalog.js');
assert.equal(C.coveredKinds.length,L.KINDS.length);
assert.deepEqual(new Set(C.coveredKinds),new Set(L.KINDS));
const mock=(shape,fn=(i,j)=>(i+1)*.1+j*.12)=>
 Array.from({length:shape[0]},(_,i)=>Array.from({length:shape[1]},(_,j)=>fn(i,j)));
const presetNames=['ffn','residual','attention','loss','gqa','decode','swiglu',
 'norm','embedding','classifier','mask','sampling','mla_lite','moe_route','grpo_signals','blank'];
for(const name of presetNames){
 const g=L.preset(name),con=C.evaluate(g),num=L.evaluate(g);
 assert.equal(con.evidence,C.EVIDENCE);
 assert.equal(con.noNumericalExecution,true);
 assert.equal(con.totalKnownFlops,null);
 assert.equal(con.steps.length,num.steps.length,name);
 for(let i=0;i<con.steps.length;i++){
   assert.deepEqual(con.steps[i].shape,num.steps[i].shape,name+' step '+i);
   assert.equal(con.steps[i].values,null);
   assert.equal(con.steps[i].flops,null);
   assert.ok(con.steps[i].transformation.length>9);
   assert.equal(con.steps[i].shapeEvidence,'declared_parameter_and_graph_shape_inference');
 }
}
function perKind(k){
 const d=L.defs[k],n={id:k,kind:k,params:{...d.params}},inputs={};
 for(const port of d.ports){
   if(port==='ids'||port==='labels')inputs[port]=[2,1];
   else if(k==='gqa')inputs[port]=[2,port==='q'?8:4];
   else if(k==='masked_softmax')inputs[port]=[2,2];
   else if(k==='matmul')inputs[port]=port==='a'?[2,4]:[4,4];
   else if(k==='low_rank_project')inputs[port]=port==='x'?[2,4]:port==='down'?[4,2]:[2,4];
   else if(k==='row_bias_add')inputs[port]=port==='bias'?[1,4]:[2,4];
   else if(k==='weighted_sum')inputs[port]=port==='weights'?[2,2]:[2,4];
   else if(['gather_rows','scatter_add','router_gate_weights','expert_bank_mix'].includes(k)&&port==='indices')inputs[port]=[2,1];
   else if(k==='expert_bank_mix'&&port==='gates')inputs[port]=[2,1];
   else if(k==='sliding_window_mask')inputs[port]=[2,2];
   else inputs[port]=[2,4];
 }
 const result=C.analyze(n,inputs);
 assert.ok(result.shape.every(x=>Number.isInteger(x)&&x>0),k);
 assert.ok(result.rule.length>7,k);
 const values={};
 for(const [port,sh] of Object.entries(inputs)){
   if(port==='ids'||port==='labels')values[port]=mock(sh,()=>0);
   else if(k==='masked_softmax'&&port==='mask')values[port]=[[1,0],[1,1]];
   else if(k==='masked_router_topk'&&port==='mask')values[port]=mock(sh,()=>1);
   else if(['gather_rows','scatter_add','router_gate_weights','expert_bank_mix'].includes(k)&&port==='indices')values[port]=[[0],[1]];
   else if(k==='kl_prob')values[port]=mock(sh,()=>1/sh[1]);
   else values[port]=mock(sh);
 }
 let actual;
 if(B.KINDS.includes(k))actual=B.execute(n,values).shape;
 else{
   const nodes=[],edges=[];
   for(const port of d.ports){
     const source='src_'+port;
     let sourceKind='input',params={rows:inputs[port][0],width:inputs[port][1],seed:3};
     if(port==='labels'){sourceKind='token_ids';params={rows:2,vocab:4,seed:1};}
     nodes.push({id:source,kind:sourceKind,x:15,y:nodes.length*100,params});
     edges.push({from:source,to:k,port});
   }
   if(!d.ports.length)nodes.push({id:k,kind:k,x:10,y:10,params:d.params});
   else nodes.push({id:k,kind:k,x:550,y:150,params:d.params});
   const graph={schema:L.SCHEMA,title:'test',nodes,edges};
   actual=L.evaluate(graph).steps.at(-1).shape;
 }
 assert.deepEqual(result.shape,actual,k);
}
for(const k of L.KINDS)perKind(k);
assert.deepEqual(C.analyze({id:'test',kind:'linear',params:{out:7}},{x:[2048,4096]}).shape,[2048,7],
 'concept inference on abstract large dimensions must not allocate the actual matrix');
assert.throws(()=>C.analyze({id:'bad',kind:'matmul',params:{}},{a:[8,7],b:[9,4]}),/收缩维/);
assert.throws(()=>C.analyze({id:'bad',kind:'masked_softmax',params:{}},
 {logits:[3,3],mask:[2,3]}),/相同/);
assert.throws(()=>C.analyze({id:'bad',kind:'gqa',params:{heads:3,kvHeads:2,causal:false,rope:false}},
 {q:[2,9],k:[2,6],v:[2,6]}),/整除/);
const broken=L.preset('ffn');broken.edges.pop();assert.throws(()=>C.evaluate(broken),/未连接/);
assert.ok(C.evaluate(L.preset('classifier')).steps.find(s=>s.kind==='cross_entropy')
 .conditions.some(x=>x.includes('标签')));
console.log('QUALITATIVE / all '+L.KINDS.length+' block contracts, presets + free build, large abstract shape PASS');
