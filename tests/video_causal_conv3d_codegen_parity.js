'use strict';
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Py=require('../python_codegen.js'),Torch=require('../torch_codegen.js');
function run(source){
 return JSON.parse(execFileSync('python3',['-c',source],{encoding:'utf8',timeout:120000,maxBuffer:8*1024*1024}).trim().split('\n').at(-1));
}
function match(a,b,label){
 const x=a.flat(Infinity),y=b.flat(Infinity);assert.equal(x.length,y.length,label+' count');
 x.forEach((v,i)=>assert.ok(Number.isFinite(y[i])&&Math.abs(v-y[i])<2e-8+Math.abs(v)*1e-10,label+' '+i+' '+v+' vs '+y[i]));
}
const graph=L.preset('video_causal_conv3d'),steps=L.evaluate(graph).steps;
const np=Py.compile(graph),pt=Torch.compile(graph);
const n=run(np.source),t=run(pt.source+'\nprint(json.dumps({k:v.detach().tolist() for k,v in GraphRuntime().forward().items()}))\n');
for(const item of steps){
 match(item.values,n[item.id],'NumPy '+item.id);
 match(item.values,t[item.id],'PyTorch '+item.id);
}
assert.ok(pt.source.includes('F.conv3d(')&&pt.source.includes('F.pad('));
const gradient=run(pt.source+'\nr=GraphRuntime()\nout=r.forward()\nout["causal_conv"].sum().backward()\nprint(json.dumps(r.parameters["kernel3d"].grad.detach().tolist()))\n');
const pix=steps.find(z=>z.id==='frames').values,parameter=graph.nodes.find(z=>z.id==='causal_conv').params;
const {frames:T,height:H,width:W,channels:C,kernelFrames:KT,kernelHeight:KH,kernelWidth:KW,outChannels:O,stride:S}=parameter;
const OH=Math.floor((H-KH)/S)+1,OW=Math.floor((W-KW)/S)+1;
const expect=Array.from({length:O},()=>Array.from({length:KT*KH*KW*C},(_,idx)=>{
 const dt=Math.floor(idx/(KH*KW*C)),rem=idx%(KH*KW*C);
 const dy=Math.floor(rem/(KW*C)),dx=Math.floor(rem/C)%KW,ci=rem%C;
 let total=0;
 for(let tt=0;tt<T;tt++){const inputT=tt+dt-(KT-1);if(inputT<0)continue;
  for(let y=0;y<OH;y++)for(let xx=0;xx<OW;xx++)
   total+=pix[inputT*H+y*S+dy][(xx*S+dx)*C+ci];
 }
 return total;
}));
match(expect,gradient,'PyTorch real Conv3D persistent kernel autograd');
console.log('VIDEO_CAUSAL_CONV3D_PARITY_PASS JS/NumPy/PyTorch and Conv3D kernel VJP');
