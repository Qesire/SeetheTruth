'use strict';
const a=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const C=require('../compute_microscope.js'),P=require('../paper_patches.js'),T=require('../paper_compute_trace.js');
const near=(x,y,tol=1e-9)=>a.ok(Math.abs(x-y)<tol, x+' != '+y);
const X=[[1,2],[2,-1]],W=[[1,0],[0,1]],A=[[1],[2]],B=[[3,4]];
for(const alpha of [-3,0,1,2]){
 const inp={x:X,w:W,a:A,b:B,alpha},result=P.lora(X,W,A,B,alpha),frame=T.lora(inp,result);
 a.equal(C.validate(frame),frame);a.equal(frame.steps.length,9);
 for(const step of ['base','down','up']){const node=frame.steps.find(z=>z.id===step);for(let i=0;i<node.out.shape[0];i++)for(let j=0;j<node.out.shape[1];j++)near(C.dot(frame,step,i,j).error,0);}
 for(let i=0;i<X.length;i++)for(let j=0;j<2;j++)near(result.output[i][j],result.base[i][j]+result.update[i][j]);
 const tampered=JSON.parse(JSON.stringify(result));tampered.internals.down[0][0]+=.05;
 a.throws(()=>T.lora(inp,tampered),/numerical result differs/);
}
for(const scale of [.5,1,4]){
 const inp={x:X,w:W,a:A,b:B,m:[[2,3]],alpha:scale},res=P.dora(X,W,A,B,inp.m,inp.alpha),frame=T.dora(inp,res);
 a.equal(frame.steps.length,11);a.equal(C.validate(frame),frame);
 for(const id of ['ab','output']){const s=frame.steps.find(x=>x.id===id);for(let i=0;i<s.out.shape[0];i++)for(let j=0;j<s.out.shape[1];j++)near(C.dot(frame,id,i,j).error,0);}
 for(let j=0;j<2;j++){
  near(Math.hypot(...res.effectiveWeight.map(r=>r[j])),inp.m[0][j]);
  near(res.internals.norms[0][j],Math.hypot(...res.directions.map(r=>r[j])));
 }
 const corrupted=JSON.parse(JSON.stringify(res));corrupted.internals.norms[0][0]+=.25;
 a.throws(()=>T.dora(inp,corrupted),/numerical result differs/);
}
const q=[[1,0],[0,1]],k=[[1,0],[0,1],[1,1]],v=[[3,4],[5,6],[-2,3]];
function dense(q,k,v){
 const d=q[0].length;
 return q.map(query=>{
  const logits=k.map(key=>query.reduce((z,x,j)=>z+x*key[j],0)/Math.sqrt(d)),max=Math.max(...logits),weights=logits.map(z=>Math.exp(z-max)),sum=weights.reduce((a,b)=>a+b,0);
  return v[0].map((_,j)=>weights.reduce((a,b,t)=>a+b*v[t][j],0)/sum);
 });
}
const truth=dense(q,k,v);
for(const chunk of [1,2,3]){
 const inp={q,k,v,chunk},res=P.onlineAttention(q,k,v,chunk),frame=T.online(inp,res);
 a.equal(C.validate(frame),frame);
 a.equal(frame.steps.length,4+2*Math.ceil(k.length/chunk));
 a.equal(frame.steps.filter(s=>s.kind==='online_softmax').length,2*Math.ceil(k.length/chunk));
 for(const tile of frame.steps.filter(s=>s.substeps)){
  a.equal(tile.substeps.length,6);
  const dotId=tile.substeps[0].id;
  for(let j=0;j<tile.substeps[0].out.shape[1];j++)near(C.dot(frame,dotId,0,j).error,0);
 }
 for(let i=0;i<2;i++)for(let j=0;j<2;j++)near(res.output[i][j],truth[i][j],1e-12);
 const tampered=JSON.parse(JSON.stringify(res));tampered.tileTrace[0].normalizer+=.1;
 a.throws(()=>T.online(inp,tampered),/numerical result differs/);
}
a.throws(()=>P.lora([[1,2]],W,A,B,Infinity),/invalid LoRA dimensions/);
a.throws(()=>P.dora([[1,2]],W,A,B,[[2,3]],Infinity),/invalid DoRA shapes/);
a.throws(()=>T.online({q,k,v,chunk:0},{output:[]}),/attention dimensions/);
const paper=fs.readFileSync(path.join(__dirname,'../paper_lab.html'),'utf8');
for(const dep of ['paper_compute_trace.js','compute_microscope.js','compute_microscope_ui.js'])a.ok(paper.includes('src="'+dep+'"'));
a.ok(paper.includes('Trace.lora(args,result)')&&paper.includes('Trace.dora(args,result)')&&paper.includes('Trace.online(args,result)'));
a.ok(paper.includes('id="computeMicroscope"')&&paper.includes('microscope.show(frame)'));
const assets=fs.readFileSync(path.join(__dirname,'../android/teaching-assets.txt'),'utf8').split(/\r?\n/);
a.ok(assets.includes('paper_compute_trace.js'));
new vm.Script(fs.readFileSync(path.join(__dirname,'../paper_compute_trace.js'),'utf8'));
console.log('PAPER_TRACE_PASS LoRA 4 alphas, DoRA 3 normalization variants, Online Softmax 3 tile widths, independent dense oracle + numerical tamper negative cases');
