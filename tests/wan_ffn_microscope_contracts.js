'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const A=require('../wan_ffn_trace.js');
const R=A.SOURCE,close=(a,b,eps=1e-12)=>{assert.ok(Math.abs(a-b)<eps,'mismatch: '+a+' vs '+b);};
assert.equal(R.repo,'Wan-Video/Wan2.2');
assert.equal(R.commit,'1ea34ff48f87168174e12956e200b1d908b1c5ff');
assert.equal(R.path,'wan/modules/model.py');assert.equal(R.config,'wan/configs/wan_i2v_A14B.py');
assert.deepEqual([R.dim,R.ffnDim,R.layers,R.heads],[5120,13824,40,40]);
assert.deepEqual(A.STEPS.map(x=>x.id),['condition','cross','norm','affine','up','gelu','down','gate','residual']);
assert.equal(A.geluTanh(0),0);assert.ok(A.geluTanh(-2)<0,'GELU is not ReLU');
const independentGelu=x=>x/2*(1+(Math.exp(2*Math.sqrt(2/Math.PI)*(x+0.044715*x*x*x))-1)/(Math.exp(2*Math.sqrt(2/Math.PI)*(x+0.044715*x*x*x))+1));
for(const x of [-4,-1,-.1,0,.5,2])close(A.geluTanh(x),independentGelu(x),1e-12);
assert.throws(()=>A.trace({tokens:0}),/tokens/);
assert.throws(()=>A.trace({dim:1}),/dim/);
assert.throws(()=>A.trace({hidden:25}),/hidden/);
assert.throws(()=>A.trace({gate:Infinity}),/modulation/);
assert.throws(()=>A.trace({seed:0}),/seed/);
function reference(c){
 const {tokens:T,dim:C,hidden:H}=c,output=[];
 const shift=Array.from({length:T*C},(_,i)=>c.e0[Math.floor(i/C)*6*C+3*C+i%C]+c.modulation[3*C+i%C]);
 const scale=Array.from({length:T*C},(_,i)=>c.e0[Math.floor(i/C)*6*C+4*C+i%C]+c.modulation[4*C+i%C]);
 const gate=Array.from({length:T*C},(_,i)=>c.e0[Math.floor(i/C)*6*C+5*C+i%C]+c.modulation[5*C+i%C]);
 for(let t=0;t<T;t++){
  const row=c.afterSelf.slice(t*C,(t+1)*C).map((x,j)=>x+c.crossDelta[t*C+j]);
  const mean=row.reduce((a,b)=>a+b,0)/C;
  const variance=row.reduce((a,b)=>a+(b-mean)**2,0)/C;
  const norm=row.map(v=>(v-mean)/Math.sqrt(variance+1e-6));
  const x=norm.map((v,j)=>v*(1+c.scale*scale[t*C+j])+c.shift*shift[t*C+j]);
  const hid=[];
  for(let h=0;h<H;h++){
    let dot=c.b1[h];
    for(let k=0;k<C;k++)dot+=x[k]*c.w1[h*C+k];
    hid.push(.5*dot*(1+Math.tanh(Math.sqrt(2/Math.PI)*(dot+.044715*dot**3))));
  }
  for(let j=0;j<C;j++){
   let d=c.b2[j];for(let h=0;h<H;h++)d+=c.w2[j*H+h]*hid[h];
   output.push(row[j]+d*c.gate*gate[t*C+j]);
  }
 }
 return output;
}
for(const [seed,T,C,H] of [[17,3,4,8],[29,1,2,3],[93,4,6,11],[120,2,9,5]]){
 const t=A.trace({seed,tokens:T,dim:C,hidden:H});
 assert.equal(t.steps.length,9);
 assert.deepEqual(t.steps.map(x=>x.main.shape),[[T,6*C],[T,C],[T,C],[T,C],[T,H],[T,H],[T,C],[T,C],[T,C]]);
 const r=reference(t.config);
 for(let i=0;i<T*C;i++)close(t.final.values[i],r[i],1e-12);
 assert.equal(t.flopsDemo,4*T*C*H);
 for(const s of [4,6])for(let token=0;token<T;token++){
  const width=t.steps[s].main.shape[1];
  for(let channel=0;channel<width;channel++){
   const inspect=A.explainDot(t,s,token,channel);
   close(inspect.reconstructed,inspect.value,1e-12);
   assert.equal(inspect.terms.length,s===4?C:H);
   assert.equal(inspect.terms[0].x,t.steps[s].aux.input.values[token*inspect.terms.length]);
  }
 }
 assert.ok(t.steps[2].main.values.every(Number.isFinite));
 for(let token=0;token<T;token++){
  const norm=t.steps[2].main.values.slice(token*C,(token+1)*C);
  close(norm.reduce((a,b)=>a+b,0),0,1e-11);
 }
 const sensitivity=A.hiddenChannelEffect(t,0,0);
 assert.equal(sensitivity.effects.length,C);
 assert.equal(sensitivity.hiddenChannel,0);
 for(const effect of sensitivity.effects){
  const reference=t.steps[8].main.values[effect.outputChannel]-t.steps[5].main.values[0]*t.config.w2[effect.outputChannel*H]*t.config.gate*t.e[5][effect.outputChannel];
  close(effect.ablated,reference);
  close(effect.dense-effect.ablated,effect.gatedDelta);
 }
 const noGate=A.trace({seed,tokens:T,dim:C,hidden:H,gate:0});
 assert.equal(A.hiddenChannelEffect(noGate,0,0).deltaL2,0);
 assert.deepEqual(noGate.steps[8].main.values,noGate.steps[1].main.values);
 const zeroShift=A.trace({seed,tokens:T,dim:C,hidden:H,shift:0});
 assert.notDeepEqual(zeroShift.final.values,t.final.values);
 const neg=A.trace({seed,tokens:T,dim:C,hidden:H,gate:-1});
 for(let k=0;k<T*C;k++)close(neg.final.values[k],2*t.steps[1].main.values[k]-t.final.values[k]);
 assert.throws(()=>A.explainDot(t,4,-1,0),/token/);
 assert.throws(()=>A.explainDot(t,4,0,H),/channel/);
}
const Common=require('../compute_microscope.js');
for(const [seed,T,C,H]of [[17,3,4,8],[29,1,2,3],[93,4,6,11],[120,2,9,5]]){
 const source=A.trace({seed,tokens:T,dim:C,hidden:H});
 const shared=Common.wan(source);
 assert.equal(shared.evidenceTier,Common.TIERS.WAN);
 assert.equal(shared.origin,'source_aligned_micro_numeric_engine');
 assert.equal(Common.validate(shared),shared);
 assert.equal(shared.steps.length,9);
 assert.deepEqual(shared.steps.map(x=>x.id),A.STEPS.map(x=>x.id));
 for(const idx of [4,6])for(let t=0;t<T;t++)for(let channel=0;channel<shared.steps[idx].out.shape[1];channel++){
  const d=Common.dot(shared,shared.steps[idx].id,t,channel);
  assert.equal(d.reconstructable,true);assert.ok(d.error<1e-9);
 }
 for(const [which,mutate,expected]of [
  ['cross',d=>{d.steps[1].main.values[0]+=.33;},/cross contradicts/],
  ['norm',d=>{d.steps[2].main.values[0]+=.33;},/LayerNorm contradicts/],
  ['affine',d=>{d.steps[3].main.values[0]+=.33;},/FFN affine contradicts/],
  ['gelu',d=>{d.steps[5].main.values[0]+=.33;},/GELU tanh contradicts/],
  ['gate',d=>{d.steps[7].main.values[0]+=.33;},/e5 output gate contradicts/],
  ['residual',d=>{d.steps[8].main.values[0]+=.33;},/final residual contradicts/],
  ['condition',d=>{d.e[4][0]+=.33;},/e\[4\] contradicts/],
  ['forged_GELU_input',d=>{d.steps[5].aux.preActivation.values[0]+=.33;},/Wan auxiliary 5.preActivation contradicts/],
  ['forged_gate_input',d=>{d.steps[7].aux.gate.values[0]+=.33;},/Wan auxiliary 7.gate contradicts/],
  ['forged_w1',d=>{d.steps[4].aux.weights.values[0]+=.33;},/Wan auxiliary 4.weights contradicts/],
  ['forged_cross_delta',d=>{d.steps[1].aux.crossDelta.values[0]+=.33;},/Wan auxiliary 1.crossDelta contradicts/],
  ['source',d=>{d.source.source='https://github.com/Wan-Video/Wan2.2/blob/fake/wan/modules/model.py';},/pinned Wan source\/config identity mismatch/],
  ['claim',d=>{d.evidence='model_checkpoint_measured';},/Wan evidence label mismatch/]
 ]){
  const corrupt=structuredClone(source);mutate(corrupt);
  assert.throws(()=>Common.wan(corrupt),expected,which+' must be refused');
 }
}
const real=A.trace().real;
assert.equal(real.params,141576704);assert.equal(real.flopsPerToken,283115520);
const root=path.resolve(__dirname,'..'),html=fs.readFileSync(path.join(root,'wan_ffn_lab.html'),'utf8');
const js=fs.readFileSync(path.join(root,'wan_ffn_lab.js'),'utf8');new vm.Script(js);
for(const id of ['steps','diagram','mainMatrix','extraMatrices','selected','dotBreakdown','dependencies','ablationPanel','ablateChannel','ablateToken','sourceCode','openSource','recompute','zeroGate','numerics'])assert.ok(html.includes('id="'+id+'"'),id);
for(const file of ['wan_ffn_trace.js','wan_ffn_lab.js','compute_microscope.js','compute_microscope_ui.js'])assert.ok(html.includes('src="'+file+'"'),file);
assert.ok(html.includes('id="wanSharedMicroscope"'));
assert.ok(js.includes('commonMicroscope.wan(d)')&&js.includes('sharedMicroscope.show(commonFrame,step.id)'));
assert.ok(fs.readFileSync(path.join(root,'teaching.html'),'utf8').includes('wan_ffn_lab.html'));
const allow=fs.readFileSync(path.join(root,'android/teaching-assets.txt'),'utf8').split(/\r?\n/);
for(const f of ['wan_ffn_lab.html','wan_ffn_lab.js','wan_ffn_trace.js'])assert.ok(allow.includes(f),f+' not staged');
console.log('WAN_FFN_MICROSCOPE_PASS pinned official C/H/FFN computation order; independent Float64 reference; all 9 stage shapes; GEMM dot reconstruct; LayerNorm; GELU tanh; gate/residual interventions; mobile asset contracts');