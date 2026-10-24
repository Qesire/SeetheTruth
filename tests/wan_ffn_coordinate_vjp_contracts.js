'use strict';
/* Independent finite differences of the pinned *math*, not the VJP adapter
 * or the upstream trace evaluator. Tests all source fields + intervening
 * offset/scale/gate controls for actual selected stage coordinates. */
const assert=require('node:assert/strict');
const Trace=require('../wan_ffn_trace.js'),Common=require('../compute_microscope.js'),
 Bridge=require('../wan_ffn_coordinate_vjp.js'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const opts={tokens:3,dim:4,hidden:8,seed:17,gate:1,scale:.75,shift:1.2};
const equal=(a,b,tol=1e-5,context='')=>assert.ok(Math.abs(a-b)<tol,
 context+' analytic '+a+' finite '+b);
const reference=c=>{
 const T=c.tokens,C=c.dim,H=c.hidden,
  matrix=(rows,cols,fn)=>Array.from({length:rows*cols},(_,i)=>fn(Math.floor(i/cols),i%cols));
 const condition=matrix(T,6*C,(t,j)=>c.e0[t*6*C+j]+c.modulation[j]);
 const e=g=>matrix(T,C,(t,j)=>condition[t*6*C+g*C+j]);
 const e3=e(3),e4=e(4),e5=e(5);
 const cross=matrix(T,C,(t,j)=>c.afterSelf[t*C+j]+c.crossDelta[t*C+j]);
 const norm=matrix(T,C,(t,j)=>{
  const row=cross.slice(t*C,(t+1)*C),mean=row.reduce((s,x)=>s+x,0)/C,
   v=row.reduce((s,x)=>s+(x-mean)*(x-mean),0)/C;
  return (row[j]-mean)/Math.sqrt(v+1e-6);
 });
 const affine=matrix(T,C,(t,j)=>norm[t*C+j]*(1+c.scale*e4[t*C+j])+c.shift*e3[t*C+j]);
 const up=matrix(T,H,(t,h)=>{
  let acc=c.b1[h];for(let j=0;j<C;j++)acc+=affine[t*C+j]*c.w1[h*C+j];
  return acc;
 });
 const gelu=up.map(x=>{
  const u=Math.sqrt(2/Math.PI)*(x+0.044715*x*x*x);
  return .5*x*(1+Math.tanh(u));
 });
 const down=matrix(T,C,(t,j)=>{
  let acc=c.b2[j];for(let h=0;h<H;h++)acc+=c.w2[j*H+h]*gelu[t*H+h];
  return acc;
 });
 const gate=matrix(T,C,(t,j)=>down[t*C+j]*c.gate*e5[t*C+j]);
 const residual=cross.map((v,i)=>v+gate[i]);
 return [condition,cross,norm,affine,up,gelu,down,gate,residual];
};
const sample=(c,stage,t,k)=>reference(c)[stage][t*(stage===0?6*c.dim:stage===4||stage===5?c.hidden:c.dim)+k];
const derivative=(base,key,i,stage,t,k)=>{
 const h=1e-6,one=sign=>{
  const c=structuredClone(base);
  if(Array.isArray(c[key]))c[key][i]+=sign*h;
  else c[key]+=sign*h;
  return sample(c,stage,t,k);
 };
 return (one(1)-one(-1))/(2*h);
};
for(const options of [
 opts,{tokens:2,dim:2,hidden:3,seed:23,gate:-.8,scale:1.25,shift:.25},
 {tokens:1,dim:7,hidden:5,seed:53,gate:0,scale:0,shift:0}]){
 const trace=Trace.trace(options),frame=Common.wan(trace),C=trace.config.dim,H=trace.config.hidden;
 for(let stage=0;stage<9;stage++){
  const w=stage===0?6*C:stage===4||stage===5?H:C;
  for(const col of [...new Set([0,Math.min(w-1,Math.floor(w/2))])]){
   const t=options.tokens-1,report=Bridge.inspect(trace,frame,{stepId:Trace.STEPS[stage].id,coordinate:[t,col],options});
   assert.equal(report.schema,Bridge.SCHEMA);
   assert.equal(report.backward.complete,true);
   assert.equal(report.loss.status,'NO_LOSS_OBJECTIVE');
   assert.equal(report.evidence,'PINNED_WAN_SOURCE_SYNTHETIC_FLOAT64_NOT_MODEL_GRADIENT');
   assert.equal(report.source.revision,Trace.SOURCE.commit);
   assert.equal(report.codegen.vjpExport,'NOT_CLAIMED');
   equal(report.selected.value,sample(trace.config,stage,t,col),1e-12,'stage forward '+stage);
   let checked=0;
   for(const [key,values]of Object.entries(report.sensitivities))
    for(let i=0;i<values.length;i++){
     const observed=derivative(trace.config,key.startsWith('control')?
       ({controlGate:'gate',controlScale:'scale',controlShift:'shift'}[key]):key,
       i,stage,t,col);
     equal(values[i],observed,7e-6,stage+':'+key+'['+i+']');
     checked++;
    }
   assert.ok(checked>=2*C,'must validate all scalar source derivatives');
   if(stage===2){
    assert.equal(report.backward.internalStages[0].stage,'wan_layernorm_not_rmsnorm');
    assert.equal(report.backward.localEdges.length,C);
    assert.ok(report.backward.localEdges.some((e,j)=>j!==col&&Math.abs(e.localDerivative)>1e-9));
   }
   if(stage===4||stage===6){
    assert.equal(report.backward.localEdges.length,(stage===4?C:H)*2+1);
    assert.ok(report.selected.forwardReconstructable);
    assert.ok(report.selected.forwardError<1e-10);
   }
   if(stage===7)assert.ok(report.backward.localEdges.some(e=>e.port==='e5'));
   if(stage===8)assert.ok(report.backward.localEdges.some(e=>e.port==='cross'));
  }
 }
}
// For zero e5 multiplier the entire FFN correction is cut, but
// the x_after_cross residual path remains nonzero and LayerNorm is bypassed.
const zeroOpts={...opts,gate:0},zeroData=Trace.trace(zeroOpts),zeroFrame=Common.wan(zeroData);
const zero=Bridge.inspect(zeroData,zeroFrame,{stepId:'residual',coordinate:[0,1],options:zeroOpts});
assert.ok(zero.sensitivities.w1.every(x=>x===0));
assert.ok(zero.sensitivities.w2.every(x=>x===0));
assert.ok(zero.sensitivities.afterSelf[1]===1);
assert.ok(zero.sensitivities.crossDelta[1]===1);
assert.ok(zero.sensitivities.controlGate[0]!==0);
assert.ok(zero.backward.externalBoundary.includes('cross-attention'));
// Shared modulation m[g,c] is the SUM over tokens. Independent FD on the
// scalar selected stage makes this exact, unlike a copied per-token surrogate.
const t=Trace.trace(opts),frame=Common.wan(t),mod=Bridge.inspect(t,frame,
 {stepId:'residual',coordinate:[2,1],options:opts});
const flatten=2*6*opts.dim+5*opts.dim+1;
equal(mod.sensitivities.e0[flatten],mod.sensitivities.modulation[5*opts.dim+1],1e-12);
// Every observed report must bind the exact accepted controller input AND
// canonical frame to its source revision and synthetic evidence tier.
assert.throws(()=>Bridge.inspect(t,frame,{stepId:'residual',coordinate:[2,1],
 options:{...opts,scale:1}}),/controller replay/);
assert.throws(()=>Bridge.inspect(t,frame,{stepId:'absent',coordinate:[0,0],options:opts}),/invalid source stage/);
assert.throws(()=>Bridge.inspect(t,frame,{stepId:'up',coordinate:[1,opts.hidden],options:opts}),/out of bounds/);
const forgedData=structuredClone(t);forgedData.config.w1[0]+=.2;
assert.throws(()=>Bridge.inspect(forgedData,frame,{stepId:'up',coordinate:[0,0],options:opts}),/controller replay/);
const forgedFrame=structuredClone(frame);forgedFrame.steps[8].out.values[0]+=.25;
assert.throws(()=>Bridge.inspect(t,forgedFrame,{stepId:'residual',coordinate:[0,0],options:opts}),/microscope does not match/);
const badPin=structuredClone(t);badPin.source.commit='0000000000000000000000000000000000000000';
assert.throws(()=>Bridge.inspect(badPin,frame,{stepId:'residual',coordinate:[0,0],options:opts}),/controller replay/);
// UI integration must not break established browser contracts; Android stays
// a purely offline read-only teaching surface.
const root=path.resolve(__dirname,'..'),html=fs.readFileSync(path.join(root,'wan_ffn_lab.html'),'utf8'),
 script=fs.readFileSync(path.join(root,'wan_ffn_lab.js'),'utf8');
new vm.Script(script);
for(const file of ['wan_ffn_trace.js','compute_microscope.js','activation_math.js',
 'wan_ffn_coordinate_vjp.js','compute_microscope_ui.js'])
 assert.ok(html.includes('src="'+file+'"'),'source UI script missing '+file);
const scripts=[...html.matchAll(/<script src="([^"]+)"><\/script>/g)].map(m=>m[1]);
assert.ok(scripts.indexOf('activation_math.js')<scripts.indexOf('wan_ffn_coordinate_vjp.js'));
assert.ok(scripts.indexOf('wan_ffn_coordinate_vjp.js')<scripts.indexOf('wan_ffn_lab.js'));
const assets=fs.readFileSync(path.join(root,'android/teaching-assets.txt'),'utf8').split(/\r?\n/);
assert.ok(assets.includes('wan_ffn_coordinate_vjp.js'));
assert.ok(script.includes("state.acceptedOptions=accepted")&&script.includes("SpecializedLesson.execute('trace'"));
console.log('WAN_FFN_COORDINATE_VJP_PASS: nine stages, LayerNorm cross-channel, e3/e4/e5, W1/W2, gate-zero, all source finite differences, stale controller and source pin');
