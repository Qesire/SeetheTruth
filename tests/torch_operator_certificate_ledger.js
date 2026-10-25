'use strict';
/* Independent CPU operator-case certificate generator.
 * Only a fully executed original graph AND standalone PyTorch autograd
 * may produce the fixed-shape bounded certificates. No allowlist promotion.
 */
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const {execFileSync}=require('node:child_process');
const L=require('../learning_lab.js'),Torch=require('../torch_codegen.js');
const GraphAutodiff=require('../graph_autodiff.js');
const {create}=require('./fixtures/coordinate_training_graph.js');
const root=path.resolve(__dirname,'..');
const sourceFiles=[
 'block_catalog.js','graph_autodiff.js','learning_lab.js','torch_codegen.js',
 'training_objective.js','training_coordinate_proof.js',
 'tests/fixtures/coordinate_training_graph.js',
 'tests/torch_operator_certificate_ledger.js'
];
const sha256=x=>crypto.createHash('sha256').update(x).digest('hex');
const fingerprints=Object.fromEntries(sourceFiles.map(p=>[p,sha256(fs.readFileSync(path.join(root,p)))]));
const graph=create(),forward=L.evaluate(graph),back=GraphAutodiff.backward(graph,forward,{targetId:'loss'});
const compiled=Torch.compile(graph,{lossId:'loss'});
assert.equal(compiled.lossId,'loss');
assert(!compiled.source.includes('import seethetruth'));
const extra=[
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
 encoding:'utf8',maxBuffer:4*1024*1024,timeout:90000});
const independent=JSON.parse(stdout.trim().split('\n').at(-1));
assert.match(independent.torch,/^2\.6\.0\+cpu$/,'must run the pinned independent CPU PyTorch');
assert.equal(independent.dtype,'torch.float64');
const flatten=x=>Array.isArray(x)?x.flat(Infinity):[x];
const tolerance={forward:2e-10,backward:2e-9,finiteDifference:3e-6};
const byKind=new Map(),nodeMeta=[];
for(const step of forward.steps){
 const pt=independent.forward[step.id],expected=flatten(step.values),actual=flatten(pt);
 assert.equal(actual.length,expected.length,'forward coordinates '+step.id);
 let max=0;
 for(let i=0;i<actual.length;i++){
  const diff=Math.abs(actual[i]-expected[i]);
  assert(Number.isFinite(diff)&&diff<=tolerance.forward,
   'independent forward mismatch '+step.id+'['+i+'] '+expected[i]+' '+actual[i]);
  max=Math.max(max,diff);
 }
 const grad=back.pathIds.includes(step.id),local=grad?flatten(back.gradients[step.id]):[],
       pytorch=grad?flatten(independent.gradients[step.id]):[];
 if(grad)assert.equal(pytorch.length,local.length,'PyTorch has no gradient for '+step.id);
 let maxBackward=0;
 for(let i=0;i<local.length;i++){
  const diff=Math.abs(local[i]-pytorch[i]);
  assert(Number.isFinite(diff)&&diff<=tolerance.backward,
   'independent backward mismatch '+step.id+'['+i+'] '+local[i]+' '+pytorch[i]);
  maxBackward=Math.max(maxBackward,diff);
 }
 nodeMeta.push({id:step.id,kind:step.kind,shape:step.shape,
  forwardCoordinates:expected.length,wholeLossAdjointCoordinates:grad?local.length:0,
  maxForwardError:max,maxBackwardError:maxBackward});
 const record=byKind.get(step.kind)||{kind:step.kind,caseIds:[],
  forwardCoordinates:0,wholeLossAdjointCoordinates:0};
 record.caseIds.push(step.id);
 record.forwardCoordinates+=expected.length;
 record.wholeLossAdjointCoordinates+=grad?local.length:0;
 byKind.set(step.kind,record);
}
let paramFD=0;
for(const id of ['x','w'])for(let i=0;i<2;i++)for(let j=0;j<2;j++){
 const grad=back.gradients[id][i][j];
 const value=sign=>{
  const copy=structuredClone(graph);
  copy.nodes.find(n=>n.id===id).params.values[i][j]+=sign*1e-6;
  return L.evaluate(copy).steps.find(s=>s.id==='loss').values[0][0];
 };
 const fd=(value(1)-value(-1))/(2e-6);
 assert(Math.abs(fd-grad)<=tolerance.finiteDifference,
  'independent central difference mismatch '+id+'['+i+','+j+']');
 paramFD++;
}
const sourceDigest=sha256(JSON.stringify(fingerprints));
const ledger={
 schema:'seethetruth-executed-operator-case-certificates-v1',
 evidence:'EXECUTED_PINNED_PYTORCH_CPU_GRAPH_CASE',
 sourceDigest,sourceFiles:fingerprints,
 environment:{torch:independent.torch,device:'cpu',dtype:independent.dtype,
  python:'3.11',runner:'ubuntu-latest'},
 fixture:{id:'coordinate_training_graph',source:'tests/fixtures/coordinate_training_graph.js',
  graphDigest:sha256(JSON.stringify(graph)),targetId:'loss',parameterCoordinatesChecked:paramFD,
  requirements:'original JS forward + full-loss graph VJP vs independent standalone generated PyTorch CPU autograd; parameter central differences'},
 tolerance,
 // Claims refer only to the exact observed graph, not universal per-kind support.
 cases:[...byKind.values()].sort((a,b)=>a.kind.localeCompare(b.kind)).map(x=>({
  ...x,fixtureId:'coordinate_training_graph',backend:'pytorch_cpu',
  forward:'passed_on_fixed_fixture',
  backward:x.wholeLossAdjointCoordinates?'passed_on_fixed_fixture':'not_checked',
  notGeneralized:true
 })),
 nodes:nodeMeta
};
const text=JSON.stringify(ledger,null,2)+'\n';
const output=path.join(root,'operator_certificates_data.json');
if(process.argv.includes('--check')){
 assert(fs.existsSync(output),'certificate data absent');
 assert.equal(fs.readFileSync(output,'utf8'),text,'re-executed independent PyTorch certificate differs from published source-pinned artifact');
}else if(process.argv.includes('--write')){
 fs.writeFileSync(output,text);
}else if(!process.argv.includes('--emit')){
 throw Error('Use --emit, --write, or --check');
}
if(process.argv.includes('--emit'))console.log('OPERATOR_CERTIFICATE_JSON='+JSON.stringify(ledger));
console.log('OPERATOR_CERTIFICATE_EXECUTED_PASS '+JSON.stringify({
 sourceDigest,kinds:ledger.cases.length,nodes:nodeMeta.length,
 forwardCoordinates:ledger.cases.reduce((n,x)=>n+x.forwardCoordinates,0),
 backwardCoordinates:ledger.cases.reduce((n,x)=>n+x.wholeLossAdjointCoordinates,0),
 finiteDifference:paramFD,environment:ledger.environment
}));
