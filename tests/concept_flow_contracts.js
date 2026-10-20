'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),C=require('../concept_flow.js'),B=require('../block_catalog.js');
assert.equal(C.coveredKinds.length,L.KINDS.length);
assert.deepEqual(new Set(C.coveredKinds),new Set(L.KINDS));
const mock=(shape,fn=(i,j)=>(i+1)*.1+j*.12)=>
 Array.from({length:shape[0]},(_,i)=>Array.from({length:shape[1]},(_,j)=>fn(i,j)));
const presetNames=['ffn','residual','attention','loss','gqa','decode','swiglu',
 'norm','embedding','classifier','mask','sampling','mla_lite','moe_route','grpo_signals','moe_expert_ffn','mla_cache','grpo_objective','autodiff_core','moe_backward','trainable_core','nd_attention_lesson','nd_causal_attention','nd_gqa_attention','nd_mqa_attention','nd_trainable_gqa','nd_trainable_mqa','nd_cached_gqa_decode','activation_gallery','optimizer_compare','activation_extension','optimizer_decay_compare','blank'];
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
 if(k.startsWith('nd_')){
  if(k==='nd_rope'){
    const ND=require('../nd_tensor.js'),n={id:k,kind:k,params:L.defs[k].params};
    const input=ND.seedTensor([1,2,3,4],5),actual=B.execute(n,{x:input});
    assert.deepEqual(C.analyze(n,{x:[1,2,3,4]}).shape,actual.shape);return;
  }
  if(k==='nd_parameter'){
   const ND=require('../nd_tensor.js'),n={id:k,kind:k,params:L.defs[k].params};
   const expected=ND.shape(n.params.values),sym=C.analyze(n,{}),num=B.execute(n,{});
   assert.deepEqual(sym.shape,expected);assert.deepEqual(num.shape,expected);return;
  }
  const ND=require('../nd_tensor.js'),d=L.defs[k],n={id:k,kind:k,params:d.params};
  const sample=(sh,seed)=>ND.seedTensor(sh,seed);
  let inputs={};
  if(k==='nd_permute'||k==='nd_flatten'||k==='nd_reduce_sum'||k==='nd_scale'||k==='nd_softmax_last'||k==='nd_causal_mask')inputs={x:sample([1,2,3,4],2)};
  else if(['nd_gelu_erf','nd_silu','nd_relu_squared','nd_mish','nd_elu','nd_hardswish'].includes(k))inputs={x:sample([1,2,3,4],2)};
  else if(k==='nd_repeat_kv')inputs={x:sample([1,2,3,4],2)};
  else if(k==='nd_cache_read')inputs={x:sample([1,2,3,4],2)};
  else if(k==='nd_cache_append')inputs={cache:sample([1,2,2,4],2),segment:sample([1,2,1,4],3)};
  else if(k==='nd_mse')inputs={prediction:sample([1,2,3,4],2),target:sample([1,2,3,4],3)};
  else if(k==='nd_reshape')inputs={x:sample([6,4],2)};
  else if(k==='nd_expand4d')inputs={x:sample([3,4],2)};
  else if(k==='nd_broadcast_add')inputs={a:sample([1,2,3,4],2),b:sample([1,1,3,4],3)};
  else if(k==='nd_batched_matmul')inputs={a:sample([1,2,3,4],2),b:sample([1,2,4,3],3)};
  const shapes=Object.fromEntries(Object.entries(inputs).map(([port,value])=>[port,ND.shape(value)]));
  const inferred=C.analyze(n,shapes),actual=B.execute(n,inputs);
  assert.deepEqual(inferred.shape,actual.shape,k+' multidimensional shape parity');
  assert.deepEqual(actual.shape,ND.shape(actual.values),k+' runtime shape validity');
  return;
 }

 if(['deepinfer_mtp_confidence_schedule','deepinfer_yoco_swa_mask','deepinfer_yoco_swa_compact'].includes(k)){
  const n={id:k,kind:k,params:{...L.defs[k].params}};
  const shapes=k==='deepinfer_mtp_confidence_schedule'?{remap:[3,1],prev_confidence:[2,2]}:
    {q:[4,3],k:[4,3],v:[4,2]};
  const values=k==='deepinfer_mtp_confidence_schedule'?{
    remap:[[0],[1],[-1]],prev_confidence:[[.9,.8],[.5,.5]]
  }:{q:mock([4,3]),k:mock([4,3]),v:mock([4,2])};
  const inferred=C.analyze(n,shapes),actual=B.execute(n,values);
  assert.deepEqual(inferred.shape,actual.shape,k+' shape contract');
  assert.ok(actual.values.flat(Infinity).every(Number.isFinite));
  return;
 }
 if(['inference_topk','inference_sparse_attention','inference_paged_kv_read','inference_greedy_verify'].includes(k)){
  const n={id:k,kind:k,params:{...L.defs[k].params}},shapes=
   k==='inference_topk'?{scores:[2,4]}:
   k==='inference_sparse_attention'?{q:[2,2],k:[3,2],v:[3,2],indices:[2,2]}:
   k==='inference_paged_kv_read'?{cache:[4,2],blocks:[2,1],positions:[2,1]}:
   {draft:[2,1],target:[3,1]};
  const values=Object.fromEntries(Object.entries(shapes).map(([key,sh])=>[key,mock(sh,(i,j)=>
   key==='indices'?(j%3):key==='blocks'?i:key==='positions'?i:
   key==='draft'||key==='target'?i:0.2+i*.1+j*.05)]));
  const inferred=C.analyze(n,shapes),actual=B.execute(n,values);
  assert.deepEqual(inferred.shape,actual.shape,k+' inference shape parity');
  assert.ok(actual.values.flat(Infinity).every(Number.isFinite));
  return;
 }
 if(k==='video_causal_conv3d'){
  const p=L.defs[k].params,n={id:k,kind:k,params:{...p}},
   inputShapes={x:[8,2],kernel:[2,4]},
   values={x:mock([8,2]),kernel:mock([2,4])};
  const inferred=C.analyze(n,inputShapes),actual=B.execute(n,values);
  assert.deepEqual(inferred.shape,actual.shape,'native 5D causal convolution contracts');
  assert.ok(actual.values.flat(Infinity).every(Number.isFinite));
  return;
 }
 if(['vision_conv2d','audio_mel_filterbank','diffusion_ddim_step'].includes(k)){
  const p=L.defs[k].params,n={id:k,kind:k,params:{...p}};
  const dims=k==='vision_conv2d'?{x:[4,4],kernel:[2,4]}:k==='audio_mel_filterbank'?{x:[3,3]}:{x:[2,4],epsilon:[2,4]};
  const values=Object.fromEntries(Object.entries(dims).map(([key,sh])=>[key,mock(sh,(i,j)=>
    k==='audio_mel_filterbank'?0.1+(i+j)/4:0.2+i*.1+j*.05)]));
  const inferred=C.analyze(n,dims),actual=B.execute(n,values);
  assert.deepEqual(inferred.shape,actual.shape,k+' named HWC/Mel/DDIM shape parity');
  assert.ok(actual.values.flat(Infinity).every(Number.isFinite));
  return;
 }
 if(['vision_patchify','video_tubelet','audio_stft_power','audio_log_power'].includes(k)){
  const p=L.defs[k].params,n={id:k,kind:k,params:{...p}};
  const dims=k==='vision_patchify'?[4,4]:k==='video_tubelet'?[8,2]:k==='audio_stft_power'?[1,8]:[3,3];
  const x=mock(dims,(i,j)=>k==='audio_log_power'?.1+(i+j)/4:i+j);
  const inferred=C.analyze(n,{x:dims}),actual=B.execute(n,{x});
  assert.deepEqual(inferred.shape,actual.shape,k+' named modality shape parity');
  assert.ok(actual.values.flat(Infinity).every(Number.isFinite));
  return;
 }
 const d=L.defs[k],n={id:k,kind:k,params:{...d.params}},inputs={};
 for(const port of d.ports){
   if(port==='ids'||port==='labels')inputs[port]=[2,1];
   else if(k==='gqa')inputs[port]=[2,port==='q'?8:4];
   else if(k==='masked_softmax')inputs[port]=[2,2];
   else if(k==='token_depth_mask'&&port==='depths')inputs[port]=[2,1];
   else if(k==='matmul')inputs[port]=port==='a'?[2,4]:[4,4];
   else if(k==='response_token_reshape'&&port==='x')inputs[port]=[4,1];
   else if(k==='response_advantage_broadcast'&&port==='advantage')inputs[port]=[2,1];
   else if(k==='cache_read_prefix'&&port==='cache')inputs[port]=[3,4];
   else if(k==='cache_write_segment'&&port==='cache')inputs[port]=[2,4];
   else if(k==='cache_write_segment'&&port==='segment')inputs[port]=[1,4];
   else if(k==='low_rank_project')inputs[port]=port==='x'?[2,4]:port==='down'?[4,2]:[2,4];
   else if(k==='row_bias_add')inputs[port]=port==='bias'?[1,4]:[2,4];
   else if(k==='weighted_sum')inputs[port]=port==='weights'?[2,2]:[2,4];
   else if(['gather_rows','scatter_add','router_gate_weights','expert_bank_mix'].includes(k)&&port==='indices')inputs[port]=[2,1];
   else if(['dispatch_routed_tokens','expert_linear_bank'].includes(k)&&port==='indices')inputs[port]=[2,2];
   else if(k==='dispatch_routed_tokens'&&port==='x')inputs[port]=[2,2];
   else if(k==='expert_linear_bank'&&port==='x')inputs[port]=[4,2];
   else if(k==='expert_linear_bank'&&port==='weights')inputs[port]=[4,2];
   else if(k==='routed_weighted_reduce'&&port==='values')inputs[port]=[4,2];
   else if(k==='routed_weighted_reduce'&&port==='gates')inputs[port]=[2,2];
   else if(k==='gather_routed_bank'&&port==='indices')inputs[port]=[2,2];
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
   else if(k==='token_depth_mask'&&port==='depths')values[port]=[[0],[1]];
   else if(k==='state_mask_blend'&&port==='mask')values[port]=[[1,1,1,1],[0,0,0,0]];
   else if(k==='masked_router_topk'&&port==='mask')values[port]=mock(sh,()=>1);
   else if(k==='masked_grpo_objective'&&port==='mask')values[port]=mock(sh,()=>1);
   else if(['gather_rows','scatter_add','router_gate_weights','expert_bank_mix'].includes(k)&&port==='indices')values[port]=[[0],[1]];
   else if(['dispatch_routed_tokens','expert_linear_bank'].includes(k)&&port==='indices')values[port]=[[0,1],[1,0]];
   else if(k==='gather_routed_bank'&&port==='indices')values[port]=[[0,1],[1,0]];
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
