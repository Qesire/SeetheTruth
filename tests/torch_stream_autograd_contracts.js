'use strict';
/* Generated PyTorch CPU executed in independent interpreter; real gradients,
   branch VJP, invariant rejection, AdamW, and no aliasing. */
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Torch=require('../torch_codegen.js'),NumPy=require('../python_codegen.js');
const kinds=require('../stream_graph_blocks.js').kinds;
const clone=x=>JSON.parse(JSON.stringify(x));
const graph=L.preset('stream_state_branches');
const js=L.evaluate(graph),num=NumPy.compile(graph),pt=Torch.compile(graph);
const zero=x=>x.map(row=>row.map(()=>0));
const param=(n,values,frozen=false)=>{
 n.kind="parameter";n.params={values,frozen,optimizer:{kind:"adamw",t:0,m:zero(values),v:zero(values)}};
};
function run(python,extra=''){
 const output=execFileSync('python3',['-c',python.source+'\n'+extra],{encoding:'utf8',maxBuffer:4*1024*1024,timeout:60000});
 return JSON.parse(output.trim().split('\n').at(-1));
}
const numeric=(a,b,name)=>{
 assert.equal(a.flat(Infinity).length,b.flat(Infinity).length,name+' size');
 for(let i=0;i<a.flat(Infinity).length;i++){
  const x=a.flat(Infinity)[i],y=b.flat(Infinity)[i];
  assert.ok(Number.isFinite(y)&&Math.abs(x-y)<2e-10+Math.abs(x)*2e-12,name+'['+i+'] '+x+'!='+y);
 }
};
assert.deepEqual(Torch.supportedKinds.filter(k=>k.startsWith('stream_state_')).sort(),[...kinds].sort());
assert.ok(pt.source.includes('def _stream_append(')&&pt.source.includes('class _StreamOutputs'));
assert.ok(pt.source.includes('torch.cat(')&&pt.source.includes('self.last_state_metadata'));
assert.ok(!pt.source.includes('import seethetruth'));
const basic=run(pt);
for(const step of js.steps)numeric(basic[step.id],step.values,step.id+' Torch↔JS');
const metadata=run(pt,[
 'm=GraphRuntime()', 'v=m.forward()',
 'print(json.dumps({"state":v.state_metadata,"shape":list(v["append_a"].shape)}))'
].join('\n'));
assert.deepEqual(metadata.state.init,js.steps.find(s=>s.id==='init').logicalState);
assert.deepEqual(metadata.state.append_a,js.steps.find(s=>s.id==='append_a').logicalState);
assert.deepEqual(metadata.state.append_b,js.steps.find(s=>s.id==='append_b').logicalState);
assert.deepEqual(metadata.shape,[3,3]);
const opt=clone(graph);
param(opt.nodes.find(n=>n.id==='prefix'),[[.2,-.3,.4],[.8,.6,-.2]]);
param(opt.nodes.find(n=>n.id==='chunk_a'),[[.3,-.4,.5]]);
param(opt.nodes.find(n=>n.id==='chunk_b'),[[-.7,.2,.9]]);
function add(id,kind,params,x,y){opt.nodes.push({id,kind,params,x,y});}
function edge(from,to,port){opt.edges.push({from,to,port});}
add('target_a','parameter',{},1460,50);
param(opt.nodes.at(-1),[[0,0,0],[.1,.25,-.1]],true);
add('target_b','parameter',{},1460,360);
param(opt.nodes.at(-1),[[.3,-.1,.2],[-.2,.4,.6]],true);
add('loss_a','mse',{},1740,60);add('loss_b','mse',{},1740,360);
edge('read_a','loss_a','prediction');edge('target_a','loss_a','target');
edge('read_b','loss_b','prediction');edge('target_b','loss_b','target');
add('total','add',{},1950,175);edge('loss_a','total','a');edge('loss_b','total','b');
const good=L.validateGraph(opt),reference=L.evaluate(good),trained=Torch.compile(good,{lossId:'total'});
assert.equal(trained.lossId,'total','loss must include both read branches');
const checked=run(trained,[
 'm=GraphRuntime()',
 'out=m.forward()',
 'loss=out["total"]',
 'loss.backward()',
 'p=m.parameters["prefix"].detach().clone().requires_grad_(True)',
 'ca=m.parameters["chunk_a"].detach().clone().requires_grad_(True)',
 'cb=m.parameters["chunk_b"].detach().clone().requires_grad_(True)',
 'tar_a=m.inputs["target_a"]',
 'tar_b=m.inputs["target_b"]',
 'expect=F.mse_loss(torch.cat((p[1:,:],ca),dim=0),tar_a)+F.mse_loss(torch.cat((p[1:,:],cb),dim=0),tar_b)',
 'expect.backward()',
 'print(json.dumps({"loss":loss.detach().item(),"expectedLoss":expect.detach().item(),',
 '"grads":{k:m.parameters[k].grad.tolist() for k in ("prefix","chunk_a","chunk_b")},',
 '"oracle":{"prefix":p.grad.tolist(),"chunk_a":ca.grad.tolist(),"chunk_b":cb.grad.tolist()},',
 '"metadata":out.state_metadata},allow_nan=False))'
].join('\n'));
assert.ok(Math.abs(checked.loss-checked.expectedLoss)<1e-12);
numeric([[checked.loss]],reference.steps.find(x=>x.id==='total').values,'aggregate JS MSE');
for(const key of ['prefix','chunk_a','chunk_b'])numeric(checked.grads[key],checked.oracle[key],key+' branch gradient');
assert.deepEqual(checked.grads.prefix[0],[0,0,0],'unread prefix row has no gradient');
assert.ok(checked.grads.prefix[1].some(x=>Math.abs(x)>1e-6),'shared prefix receives nonzero dual-path gradient');
assert.ok(checked.grads.chunk_a[0].some(x=>Math.abs(x)>1e-6),'A chunk gradient');
assert.ok(checked.grads.chunk_b[0].some(x=>Math.abs(x)>1e-6),'B chunk gradient');
assert.deepEqual(checked.metadata.append_a,{type:'logical-stream-state',axes:['T','D'],origin:10,capacity:5,nextPosition:13,version:1});
const stepped=run(trained,[
 'm=GraphRuntime()',
 'before=m.forward()["total"].item()',
 'history=m.train(steps=2,lr=.01,weight_decay=0)',
 'after=m.forward()["total"].item()',
 'import tempfile,os',
 'fd,path=tempfile.mkstemp(suffix=".pt");os.close(fd)',
 'm.save_checkpoint(path)',
 'restored=GraphRuntime();restored.load_checkpoint(path);os.unlink(path)',
 'check=restored.forward()["total"].item()',
 'print(json.dumps({"before":before,"after":after,"restored":check,"history":history,',
 '"steps":[int(s["step"].item()) if isinstance(s["step"],torch.Tensor) else int(s["step"]) for s in [',
 ' m._optimizer.state[m.parameters["prefix"]],restored._optimizer.state[restored.parameters["prefix"]]]],',
 '"original":m.parameters["prefix"].detach().tolist(),"reloaded":restored.parameters["prefix"].detach().tolist()},allow_nan=False))'
].join('\n'));
assert.ok(stepped.after<stepped.before,'two AdamW steps lower the combined loss');
assert.ok(Math.abs(stepped.restored-stepped.after)<1e-12,'checkpoint restores parameters and optimizer');
assert.deepEqual(stepped.original,stepped.reloaded);
assert.deepEqual(stepped.steps,[2,2]);
assert.equal(stepped.history.length,2);
const overrides=run(trained,[
 'm=GraphRuntime({"prefix":[[2,2,2],[3,3,3]],"chunk_a":[[4,4,4]],"chunk_b":[[5,5,5]]})',
 'v=m.forward()',
 'print(json.dumps({"A":v["read_a"].detach().tolist(),"B":v["read_b"].detach().tolist()}))'
].join('\n'));
assert.deepEqual(overrides.A,[[3,3,3],[4,4,4]]);
assert.deepEqual(overrides.B,[[3,3,3],[5,5,5]]);
function invalid(command,pattern){
 const output=run(pt,[
 'try:',...command.split('\n').map(l=>'    '+l),
 'except ValueError as exc:', '    print(json.dumps({"error":str(exc)}))',
 'else:', '    print(json.dumps({"error":"ACCEPTED"}))'
 ].join('\n'));
 assert.match(output.error,pattern,command);
}
invalid('s=_stream_init(torch.ones((2,3),dtype=DTYPE),10,2)\n_stream_append(s,torch.ones((1,3),dtype=DTYPE),12)',/capacity/);
invalid('s=_stream_init(torch.ones((2,3),dtype=DTYPE),10,4)\n_stream_append(s,torch.ones((1,3),dtype=DTYPE),11)',/stale/);
invalid('s=_stream_init(torch.ones((2,3),dtype=DTYPE),10,4)\n_stream_read(s,12,1)',/known absolute/);
invalid('s=_stream_init(torch.ones((2,3),dtype=DTYPE),10,4)\ns["axes"]=["Tk","D"]\n_stream_fork(s)',/expected logical-stream-state/);
invalid('s=_stream_init(torch.ones((2,3),dtype=DTYPE),10,4)\ns["nextPosition"]=13\n_stream_fork(s)',/corrupt/);
invalid('s=_stream_init(torch.ones((2,3),dtype=DTYPE),10,4)\n_stream_append(s,torch.ones((1,4),dtype=DTYPE),12)',/feature axis/);
const alias=run(pt,[
 's=_stream_init(torch.ones((2,3),dtype=DTYPE),10,4)',
 'fork=_stream_fork(s)',
 'fork["tensor"][0,0]=999',
 'read=_stream_read(s,10,1)',
 'read[0,0]=-100',
 'print(json.dumps({"parent":s["tensor"].tolist(),"fork":fork["tensor"].tolist()}))'
].join('\n'));
assert.deepEqual(alias.parent,[[1,1,1],[1,1,1]]);
assert.equal(alias.fork[0][0],999);
const broken=clone(good);broken.nodes.find(n=>n.id==='append_a').params.position=11;
assert.throws(()=>Torch.compile(broken),/out-of-order|stale/);
const mismatched=clone(good);mismatched.edges.find(e=>e.to==='append_a'&&e.port==='state').from='prefix';
assert.throws(()=>Torch.compile(mismatched),/expected stream-state/);
const legacy=Torch.compile(L.preset('ffn'));
assert.ok(!legacy.source.includes('class _StreamOutputs'),'legacy code should not include stream runtime');
console.log('STREAM_TORCH_AUTOGRAD_PASS typed Torch CPU forward JS parity, independent gradient oracle, two-branch sharing, override, bounded guards, 2 AdamW and checkpoint');
