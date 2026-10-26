'use strict';
/* Bridge independent FFN CPU witness to the SAME full native Wan graph.
 * Self and Cross remain connected, with their gradients already checked by
 * the complete-loss finite differences in wan_full_block_contracts.js.
 */
const path=require('node:path'),{spawnSync}=require('node:child_process');
const ND=require('../nd_tensor.js'),G=require('../graph_autodiff.js');
const {graph,trace}=require('./wan_full_block_contracts.js');
const steps=new Map(trace.steps.map(s=>[s.id,s.values]));
const gradients=G.backward(graph,trace);
const operandNames=['after_cross','e3_sum','e4_sum','e5_sum',
 'ffn_w1','ffn_w2','ffn_b1','ffn_b2','target'];
const stageNames=['norm2_ffn','ffn_conditioned','ffn_up_bias',
 'ffn_gelu_tanh','ffn_down_bias','ffn_gated','after_ffn','loss'];
const payload={
 schema:'seethetruth-wan-full-block-boundary-oracle-v1',
 operands:Object.fromEntries(operandNames.map(x=>[x,steps.get(x)])),
 native:Object.fromEntries(stageNames.map(x=>[x,steps.get(x)])),
 nativeGradients:Object.fromEntries(operandNames.filter(x=>x!=='target').map(x=>
  [x,gradients.gradients[x]]))
};
if(Object.values(payload.nativeGradients).some(x=>!x||!ND.flatten(x).every(Number.isFinite)))
 throw Error('full-loss graph VJP unavailable/nonfinite');
const result=spawnSync('python3',[path.join(__dirname,'python_wan_full_block_ffn_oracle.py')],{
 input:JSON.stringify(payload),encoding:'utf8',timeout:230000,maxBuffer:10*1024*1024
});
if(result.error||result.status!==0)
 throw Error('Wan FFN independent CPU oracle failed: '+(result.error?.message||result.stderr||result.stdout).slice(-9000));
if(!result.stdout.includes('WAN_FULL_BLOCK_BOUNDARY_CPU_PASS'))
 throw Error('missing CPU execution certificate');
process.stdout.write(result.stdout);
