/* Source-pinned, fixed-fixture operator evidence. The sample case is not a
 * proof of generic forward/VJP correctness for every possible graph/shape.
 */
(function(root,factory){'use strict';
 const node=typeof module==='object'&&module.exports;
 const api=factory(node?require('./operator_certificates_data.js'):root?.SeetheTruthOperatorCertificateData,
  node?require('./learning_lab.js'):root?.SeetheTruthLearning,
  node?require('./torch_codegen.js'):root?.SeetheTruthTorchExport);
 if(node)module.exports=api;
 if(root)root.SeetheTruthOperatorCertificates=api;
})(typeof window!=='undefined'?window:null,function(data,L,T){'use strict';
 const SCHEMA='seethetruth-executed-operator-case-certificates-v1';
 const fail=m=>{throw Error('operator certificates: '+m);};
 const plain=x=>x&&typeof x==='object'&&!Array.isArray(x);
 const hex=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
 const num=x=>Number.isSafeInteger(x)&&x>=0&&x<=8192;
 const clone=x=>JSON.parse(JSON.stringify(x));
 const freeze=x=>{
  if(x&&typeof x==='object'){
   Object.values(x).forEach(freeze);Object.freeze(x);
  }
  return x;
 };
 function validate(raw){
  if(!plain(raw)||raw.schema!==SCHEMA||raw.evidence!=='EXECUTED_PINNED_PYTORCH_CPU_GRAPH_CASE'||
     !hex(raw.sourceDigest)||!plain(raw.sourceFiles)||Object.keys(raw.sourceFiles).length<5||
     !Object.values(raw.sourceFiles).every(hex)||
     !plain(raw.environment)||raw.environment.device!=='cpu'||
     raw.environment.dtype!=='torch.float64'||
     !/^2\.6\.0\+cpu$/.test(raw.environment.torch)||
     !plain(raw.fixture)||raw.fixture.id!=='coordinate_training_graph'||
     !hex(raw.fixture.graphDigest)||!num(raw.fixture.parameterCoordinatesChecked)||
     !plain(raw.tolerance)||
     !Array.isArray(raw.cases)||!raw.cases.length||raw.cases.length>150||
     !Array.isArray(raw.nodes)||!raw.nodes.length||raw.nodes.length>150)
   fail('invalid pinned executed evidence header');
  for(const k of ['forward','backward','finiteDifference']){
   if(!Number.isFinite(raw.tolerance[k])||raw.tolerance[k]<=0||raw.tolerance[k]>1e-3)
    fail('invalid numerical tolerance');
  }
  if(!L||!T)fail('missing operator identity/codegen dependency');
  const validKinds=new Set(L.KINDS),torch=new Set(T.supportedKinds);
  const nodeMap=new Map();
  for(const n of raw.nodes){
   if(!plain(n)||typeof n.id!=='string'||!/^[-a-zA-Z0-9_]{1,72}$/.test(n.id)||
      !validKinds.has(n.kind)||!torch.has(n.kind)||
      !Array.isArray(n.shape)||n.shape.length<1||n.shape.length>4||
      !n.shape.every(v=>Number.isSafeInteger(v)&&v>0&&v<=16)||
      !num(n.forwardCoordinates)||n.forwardCoordinates===0||
      !num(n.wholeLossAdjointCoordinates)||n.wholeLossAdjointCoordinates>n.forwardCoordinates||
      !Number.isFinite(n.maxForwardErrorUpperBound)||n.maxForwardErrorUpperBound<0||
      n.maxForwardErrorUpperBound>raw.tolerance.forward||
      !Number.isFinite(n.maxBackwardErrorUpperBound)||n.maxBackwardErrorUpperBound<0||
      n.maxBackwardErrorUpperBound>raw.tolerance.backward||nodeMap.has(n.id))
    fail('invalid independent operator node witness');
   const size=n.shape.reduce((v,w)=>v*w,1);
   if(size!==n.forwardCoordinates)fail('node shape/coordinate count mismatch');
   nodeMap.set(n.id,n);
  }
  const seen=new Set(),covered=new Set();
  for(const c of raw.cases){
   if(!plain(c)||!validKinds.has(c.kind)||!torch.has(c.kind)||seen.has(c.kind)||
      c.fixtureId!=='coordinate_training_graph'||c.backend!=='pytorch_cpu'||
      c.forward!=='passed_on_fixed_fixture'||c.notGeneralized!==true||
      !num(c.forwardCoordinates)||!num(c.wholeLossAdjointCoordinates)||
      !Array.isArray(c.caseIds)||!c.caseIds.length||
      !['passed_on_fixed_fixture','not_checked'].includes(c.backward)||
      (c.wholeLossAdjointCoordinates>0)!==(c.backward==='passed_on_fixed_fixture'))
    fail('invalid operator case record');
   let forwards=0,backs=0;
   for(const id of c.caseIds){
    const n=nodeMap.get(id);
    if(!n||n.kind!==c.kind||covered.has(id))fail('case points to wrong or duplicate native node');
    covered.add(id);forwards+=n.forwardCoordinates;backs+=n.wholeLossAdjointCoordinates;
   }
   if(forwards!==c.forwardCoordinates||backs!==c.wholeLossAdjointCoordinates)
    fail('forged coordinate count');
   seen.add(c.kind);
  }
  if(covered.size!==nodeMap.size)fail('orphan native case witness');
  return freeze(clone(raw));
 }
 const value=validate(data);
 const lookup=Object.freeze(Object.fromEntries(value.cases.map(c=>[c.kind,c])));
 function forKind(kind){return Object.hasOwn(lookup,kind)?lookup[kind]:null;}
 function summary(){return Object.freeze({schema:value.schema,evidence:value.evidence,
  sourceDigest:value.sourceDigest,fixtureId:value.fixture.id,
  certifiedKinds:value.cases.length,certifiedNodes:value.nodes.length,
  forwardCoordinates:value.cases.reduce((n,c)=>n+c.forwardCoordinates,0),
  lossAdjointCoordinates:value.cases.reduce((n,c)=>n+c.wholeLossAdjointCoordinates,0),
  parameterCoordinates:value.fixture.parameterCoordinatesChecked,
  environment:value.environment});}
 return Object.freeze({SCHEMA,validate,summary,forKind,sourceDigest:value.sourceDigest,certificate:value});
});
