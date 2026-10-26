'use strict';
/* Independent CPU multi-fixture operator certificates. Every verified outcome
 * is a real LearningLab/GraphAutodiff vs standalone PyTorch CPU execution.
 * A case never proves universal correctness for its operator kind.
 */
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Torch=require('../torch_codegen.js');
const GraphAutodiff=require('../graph_autodiff.js');
const ND=require('../nd_tensor.js');
const first=require('./fixtures/coordinate_training_graph.js');
const more=require('./fixtures/operator_coverage_graphs.js');
const ndCases=require('./fixtures/nd_broadcast_reduction_graphs.js');
const root=path.resolve(__dirname,'..');
const sourceFiles=[
 'block_catalog.js','graph_autodiff.js','learning_lab.js','torch_codegen.js',
 'training_objective.js','training_coordinate_proof.js',
 'nd_tensor.js','nd_autodiff.js','portable_operator_rules.js','python_codegen.js',
 'tests/fixtures/coordinate_training_graph.js',
 'tests/fixtures/operator_coverage_graphs.js',
 'tests/fixtures/nd_broadcast_reduction_graphs.js',
 'tests/torch_operator_certificate_ledger.js'
];
const sha256=x=>crypto.createHash('sha256').update(x).digest('hex');
const fingerprints=Object.fromEntries(sourceFiles.map(p=>[p,sha256(fs.readFileSync(path.join(root,p)))]));
const flatten=x=>Array.isArray(x)?x.flat(Infinity):[x];
const tol={forward:2e-10,backward:2e-9,finiteDifference:3e-6};
const upper=(observed,limit)=>observed<=1e-12?1e-12:limit; // proven, repeatable bucket
const specs=[
 {id:'coordinate_training_graph',make:first.create,
  source:'tests/fixtures/coordinate_training_graph.js',
  tags:['2x2','two_path_single_projection','frozen_target'],skipFD:[]},
 {id:'rectangular_relu_fanout',make:more.rectangularRelu,
  source:'tests/fixtures/operator_coverage_graphs.js',
  tags:['matmul_2x3_3x2','negative_relu_slope','mixed_sign','fanout'],skipFD:[]},
 {id:'shared_weight_two_matmuls',make:more.sharedWeight,
  source:'tests/fixtures/operator_coverage_graphs.js',
  tags:['shared_parameter','two_matmuls','branch_accumulation','rectangular'],skipFD:[]},
 {id:'relu_exact_zero',make:more.reluKink,
  source:'tests/fixtures/operator_coverage_graphs.js',
  tags:['nondifferentiable','relu_kink_zero','torch_zero_subgradient'],
  skipFD:[{nodeId:'x',index:[0,1],reason:'ReLU at x=0 has no two-sided derivative; compare declared PyTorch zero-subgradient, not centered difference'}]}
 ,{id:'nd_token_reduce_head_broadcast',make:()=>ndCases.create('nd_token_reduce_head_broadcast'),
  source:'tests/fixtures/nd_broadcast_reduction_graphs.js',
  tags:['four_axis','head_token_broadcast','keepdim_reduce','channel_gradient_aggregation'],skipFD:[]}
 ,{id:'nd_batch_reduce_channel_broadcast',make:()=>ndCases.create('nd_batch_reduce_channel_broadcast'),
  source:'tests/fixtures/nd_broadcast_reduction_graphs.js',
  tags:['four_axis','batch_broadcast','keepdim_reduce','channel_gradient_aggregation'],skipFD:[]}

];
const fixtures=[],nodes=[],cases=[];
let torchVersion=null,totalFD=0;
for(const spec of specs){
 const graph=spec.make(),forward=L.evaluate(graph);
 const back=GraphAutodiff.backward(graph,forward,{targetId:'loss'});
 const compiled=Torch.compile(graph,{lossId:'loss'});
 assert.equal(compiled.lossId,'loss',spec.id);
 assert(!compiled.source.includes('import seethetruth'));
 const extra=[
  'torch.set_num_threads(1)',
  'm=GraphRuntime()',
  'out=m.forward()',
  'for v in out.values():',
  '    if v.requires_grad: v.retain_grad()',
  'out["loss"].backward()',
  'proof={"torch":torch.__version__,"dtype":str(out["loss"].dtype),',
  ' "forward":{k:v.detach().tolist() for k,v in out.items()},',
  ' "gradients":{k:v.grad.detach().tolist() for k,v in out.items() if v.grad is not None}}',
  'print(json.dumps(proof,allow_nan=False,sort_keys=True))'
 ].join('\n');
 const stdout=execFileSync('python3',['-c',compiled.source+'\n'+extra],{
  encoding:'utf8',maxBuffer:6*1024*1024,timeout:90000});
 const pt=JSON.parse(stdout.trim().split('\n').at(-1));
 assert.match(pt.torch,/^2\.6\.0\+cpu$/,'CPU torch version must be pinned');
 assert.equal(pt.dtype,'torch.float64');
 if(torchVersion!==null)assert.equal(pt.torch,torchVersion);
 torchVersion=pt.torch;
 const kinds=new Map();
 for(const step of forward.steps){
  const js=flatten(step.values),cpu=flatten(pt.forward[step.id]);
  assert.equal(cpu.length,js.length,spec.id+'/'+step.id+' forward length');
  let maxF=0;
  for(let i=0;i<js.length;i++){
   const diff=Math.abs(cpu[i]-js[i]);
   assert(Number.isFinite(diff)&&diff<=tol.forward,
    spec.id+'/'+step.id+' forward['+i+'] '+js[i]+' != '+cpu[i]);
   maxF=Math.max(maxF,diff);
  }
  const eligible=back.pathIds.includes(step.id);
  const jsGrad=eligible?flatten(back.gradients[step.id]):[];
  const cpuGrad=eligible?flatten(pt.gradients[step.id]):[];
  if(eligible)assert.equal(cpuGrad.length,jsGrad.length,
   spec.id+'/'+step.id+' missing native PyTorch loss adjoint');
  let maxB=0;
  for(let i=0;i<jsGrad.length;i++){
   const diff=Math.abs(jsGrad[i]-cpuGrad[i]);
   assert(Number.isFinite(diff)&&diff<=tol.backward,
    spec.id+'/'+step.id+' backward['+i+'] '+jsGrad[i]+' != '+cpuGrad[i]);
   maxB=Math.max(maxB,diff);
  }
  nodes.push({fixtureId:spec.id,id:step.id,kind:step.kind,
   shape:step.shape,forwardCoordinates:js.length,
   wholeLossAdjointCoordinates:jsGrad.length,
   maxForwardErrorUpperBound:upper(maxF,tol.forward),
   maxBackwardErrorUpperBound:upper(maxB,tol.backward)});
  const r=kinds.get(step.kind)||{kind:step.kind,caseIds:[],
   forwardCoordinates:0,wholeLossAdjointCoordinates:0};
  r.caseIds.push(step.id);r.forwardCoordinates+=js.length;
  r.wholeLossAdjointCoordinates+=jsGrad.length;kinds.set(step.kind,r);
 }
 let fdCount=0;
 const explicitSkips=new Set(spec.skipFD.map(x=>x.nodeId+'['+x.index.join(',')+']'));
 for(const node of graph.nodes){
  if(!['parameter','nd_parameter'].includes(node.kind)||node.params.frozen)continue;
  const values=node.params.values,sh=ND.shape(values),sourceGradient=back.gradients[node.id];
  assert.deepEqual(ND.shape(sourceGradient),sh,spec.id+' trainable N-D gradient shape');
  ND.indices(sh,idx=>{
   if(explicitSkips.has(node.id+'['+idx.join(',')+']'))return;
   const raw=ND.at(sourceGradient,idx);
   const value=sign=>{
    const copy=structuredClone(graph);
    let tensor=copy.nodes.find(n=>n.id===node.id).params.values;
    for(let axis=0;axis<idx.length-1;axis++)tensor=tensor[idx[axis]];
    tensor[idx.at(-1)]+=sign*1e-6;
    const out=L.evaluate(copy).steps.find(x=>x.id==='loss');
    return out.values[0][0];
   };
   const derivative=(value(1)-value(-1))/(2e-6);
   assert(Number.isFinite(derivative)&&Math.abs(derivative-raw)<=tol.finiteDifference,
    spec.id+'/'+node.id+' finite difference['+idx.join(',')+'] '+derivative+' vs '+raw);
   fdCount++;
  });
 }
 if(spec.id==='relu_exact_zero'){
  const xgrad=back.gradients.x[0][1];
  assert.equal(xgrad,0,'JS ReLU kink zero subgradient');
  assert.equal(pt.gradients.x[0][1],0,'PyTorch ReLU kink zero subgradient');
 }
 const meta={id:spec.id,source:spec.source,graphDigest:sha256(JSON.stringify(graph)),
  targetId:'loss',tags:spec.tags,parameterCoordinatesChecked:fdCount,
  excludedFiniteDifferences:spec.skipFD,
  nodeCount:forward.steps.length};
 fixtures.push(meta);totalFD+=fdCount;
 for(const row of kinds.values())cases.push({
  ...row,fixtureId:spec.id,backend:'pytorch_cpu',
  forward:'passed_on_fixed_fixture',
  backward:row.wholeLossAdjointCoordinates?'passed_on_fixed_fixture':'not_checked',
  notGeneralized:true
 });
}
const ledger={
 schema:'seethetruth-executed-operator-case-certificates-v2',
 evidence:'EXECUTED_PINNED_PYTORCH_CPU_GRAPH_CASE',
 sourceDigest:sha256(JSON.stringify(fingerprints)),sourceFiles:fingerprints,
 environment:{torch:torchVersion,device:'cpu',dtype:'torch.float64',python:'3.11',runner:'ubuntu-latest'},
 fixtures, tolerance:tol,
 cases:cases.sort((a,b)=>a.kind.localeCompare(b.kind)||a.fixtureId.localeCompare(b.fixtureId)),
 nodes
};
const serial=JSON.stringify(ledger,null,2)+'\n';
const browser='/* Generated from real standalone PyTorch CPU; do not hand edit. */\n'+
 '(function(root){"use strict";\nconst data='+JSON.stringify(ledger)+';\n'+
 'if(typeof module==="object"&&module.exports)module.exports=data;\n'+
 'if(root)root.SeetheTruthOperatorCertificateData=data;\n'+
 '})(typeof window!=="undefined"?window:null);\n';
const jsonFile=path.join(root,'operator_certificates_data.json');
const browserFile=path.join(root,'operator_certificates_data.js');
if(process.argv.includes('--check')){
 assert.equal(fs.readFileSync(jsonFile,'utf8'),serial,'independent CPU evidence differs from checked-in JSON');
 assert.equal(fs.readFileSync(browserFile,'utf8'),browser,'independent CPU evidence differs from checked-in browser mirror');
}else if(process.argv.includes('--write')){
 fs.writeFileSync(jsonFile,serial);fs.writeFileSync(browserFile,browser);
}else if(!process.argv.includes('--emit'))throw Error('Use --check, --write or --emit');
if(process.argv.includes('--emit'))console.log('OPERATOR_CERTIFICATE_JSON='+JSON.stringify(ledger));
console.log('OPERATOR_CERTIFICATE_EXECUTED_PASS '+JSON.stringify({
 sourceDigest:ledger.sourceDigest,fixtures:fixtures.length,caseKinds:new Set(cases.map(x=>x.kind)).size,
 cases:cases.length,nodes:nodes.length,
 forwardCoordinates:nodes.reduce((n,x)=>n+x.forwardCoordinates,0),
 backwardCoordinates:nodes.reduce((n,x)=>n+x.wholeLossAdjointCoordinates,0),
 finiteDifference:totalFD,skippedKink:specs.flatMap(x=>x.skipFD).length,
 environment:ledger.environment
}));
