/* Executed PyTorch CPU fixture-specific evidence; no universal kind proof.
 * Browser consumes reviewed, checked-in evidence. Independent execution is CI.
 */
(function(root,factory){'use strict';
 const node=typeof module==='object'&&module.exports;
 const api=factory(node?require('./operator_certificates_data.js'):root?.SeetheTruthOperatorCertificateData,
  node?require('./learning_lab.js'):root?.SeetheTruthLearning,
  node?require('./torch_codegen.js'):root?.SeetheTruthTorchExport);
 if(node)module.exports=api;
 if(root)root.SeetheTruthOperatorCertificates=api;
})(typeof window!=='undefined'?window:null,function(data,L,T){'use strict';
 const SCHEMA='seethetruth-executed-operator-case-certificates-v2';
 const fail=x=>{throw Error('operator certificates: '+x);};
 const plain=x=>!!x&&typeof x==='object'&&!Array.isArray(x);
 const hex=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
 const nat=x=>Number.isSafeInteger(x)&&x>=0&&x<=16384;
 const nodeId=x=>typeof x==='string'&&/^[a-zA-Z0-9_-]{1,72}$/.test(x);
 const deepFreeze=x=>{if(x&&typeof x==='object'){Object.values(x).forEach(deepFreeze);Object.freeze(x);}return x;};
 const clone=x=>JSON.parse(JSON.stringify(x));
 function validate(raw){
  if(!plain(raw)||raw.schema!==SCHEMA||
   raw.evidence!=='EXECUTED_PINNED_PYTORCH_CPU_GRAPH_CASE'||
   !hex(raw.sourceDigest)||!plain(raw.sourceFiles)||
   Object.keys(raw.sourceFiles).length<7||!Object.values(raw.sourceFiles).every(hex)||
   !plain(raw.environment)||raw.environment.device!=='cpu'||
   raw.environment.dtype!=='torch.float64'||!/^2\.6\.0\+cpu$/.test(raw.environment.torch)||
   !Array.isArray(raw.fixtures)||raw.fixtures.length<2||raw.fixtures.length>24||
   !Array.isArray(raw.cases)||!raw.cases.length||raw.cases.length>500||
   !Array.isArray(raw.nodes)||!raw.nodes.length||raw.nodes.length>500||
   !plain(raw.tolerance))fail('invalid pinned multi-fixture evidence header');
  for(const key of ['forward','backward','finiteDifference']){
   if(!Number.isFinite(raw.tolerance[key])||raw.tolerance[key]<=0||raw.tolerance[key]>1e-3)
    fail('invalid numerical tolerance');
  }
  if(!L||!T)fail('missing operator identity/codegen dependency');
  const known=new Set(L.KINDS),torch=new Set(T.supportedKinds);
  const fixtureMap=new Map();
  for(const fixture of raw.fixtures){
   if(!plain(fixture)||!nodeId(fixture.id)||fixtureMap.has(fixture.id)||
     !hex(fixture.graphDigest)||!nodeId(fixture.targetId)||
     typeof fixture.source!=='string'||!/^[a-zA-Z0-9_./-]+\.js$/.test(fixture.source)||
     !fixture.source.startsWith('tests/fixtures/')||
     !Array.isArray(fixture.tags)||fixture.tags.length<1||
     fixture.tags.some(t=>!nodeId(t))||
     !nat(fixture.parameterCoordinatesChecked)||!nat(fixture.nodeCount)||
     fixture.nodeCount===0||
     !Array.isArray(fixture.excludedFiniteDifferences)||
     fixture.excludedFiniteDifferences.some(x=>!plain(x)||!nodeId(x.nodeId)||
      !Array.isArray(x.index)||x.index.some(i=>!nat(i))||
      typeof x.reason!=='string'||x.reason.length<12))
    fail('invalid fixture identity or nondifferentiability policy');
   fixtureMap.set(fixture.id,fixture);
  }
  const nodeMap=new Map(),nodesPerFixture=new Map();
  for(const n of raw.nodes){
   const key=n.fixtureId+'::'+n.id;
   if(!plain(n)||!fixtureMap.has(n.fixtureId)||!nodeId(n.id)||
     !known.has(n.kind)||!torch.has(n.kind)||
     !Array.isArray(n.shape)||!n.shape.length||n.shape.length>4||
     !n.shape.every(x=>Number.isSafeInteger(x)&&x>0&&x<=16)||
     !nat(n.forwardCoordinates)||!n.forwardCoordinates||
     n.shape.reduce((p,v)=>p*v,1)!==n.forwardCoordinates||
     !nat(n.wholeLossAdjointCoordinates)||
     n.wholeLossAdjointCoordinates>n.forwardCoordinates||
     !Number.isFinite(n.maxForwardErrorUpperBound)||n.maxForwardErrorUpperBound<0||
     n.maxForwardErrorUpperBound>raw.tolerance.forward||
     !Number.isFinite(n.maxBackwardErrorUpperBound)||n.maxBackwardErrorUpperBound<0||
     n.maxBackwardErrorUpperBound>raw.tolerance.backward||
     nodeMap.has(key))fail('invalid independent per-fixture node witness');
   nodeMap.set(key,n);
   nodesPerFixture.set(n.fixtureId,(nodesPerFixture.get(n.fixtureId)||0)+1);
  }
  for(const fixture of raw.fixtures)if(nodesPerFixture.get(fixture.id)!==fixture.nodeCount)
   fail('missing or extra nodes in a fixture');
  const unique=new Set(),used=new Set();
  for(const c of raw.cases){
   const pair=c.fixtureId+'::'+c.kind;
   if(!plain(c)||!fixtureMap.has(c.fixtureId)||
     !known.has(c.kind)||!torch.has(c.kind)||unique.has(pair)||
     c.backend!=='pytorch_cpu'||c.notGeneralized!==true||
     c.forward!=='passed_on_fixed_fixture'||
     !['passed_on_fixed_fixture','not_checked'].includes(c.backward)||
     !nat(c.forwardCoordinates)||!c.forwardCoordinates||
     !nat(c.wholeLossAdjointCoordinates)||
     (c.wholeLossAdjointCoordinates>0)!==(c.backward==='passed_on_fixed_fixture')||
     !Array.isArray(c.caseIds)||!c.caseIds.length)
    fail('invalid source-scoped kind case');
   let f=0,b=0;
   for(const id of c.caseIds){
    const key=c.fixtureId+'::'+id,n=nodeMap.get(key);
    if(!n||n.kind!==c.kind||used.has(key))fail('unbound, wrong-kind or reused source node');
    used.add(key);f+=n.forwardCoordinates;b+=n.wholeLossAdjointCoordinates;
   }
   if(f!==c.forwardCoordinates||b!==c.wholeLossAdjointCoordinates)
    fail('forged coordinate count');
   unique.add(pair);
  }
  if(used.size!==nodeMap.size)fail('orphaned measured node');
  return deepFreeze(clone(raw));
 }
 const value=validate(data);
 const byKind=Object.freeze(Object.fromEntries([...new Set(value.cases.map(c=>c.kind))].map(kind=>[
  kind,Object.freeze(value.cases.filter(c=>c.kind===kind))
 ])));
 function forKind(kind){return Object.hasOwn(byKind,kind)?byKind[kind]:Object.freeze([]);}
 function summary(){return Object.freeze({
  schema:value.schema,evidence:value.evidence,sourceDigest:value.sourceDigest,
  fixtureIds:Object.freeze(value.fixtures.map(f=>f.id)),fixtures:value.fixtures.length,
  certifiedKinds:Object.keys(byKind).length,certifiedCases:value.cases.length,
  certifiedNodes:value.nodes.length,
  forwardCoordinates:value.nodes.reduce((n,c)=>n+c.forwardCoordinates,0),
  lossAdjointCoordinates:value.nodes.reduce((n,c)=>n+c.wholeLossAdjointCoordinates,0),
  parameterCoordinates:value.fixtures.reduce((n,f)=>n+f.parameterCoordinatesChecked,0),
  excludedFiniteDifferences:value.fixtures.reduce((n,f)=>n+f.excludedFiniteDifferences.length,0),
  environment:value.environment
 });}
 return Object.freeze({SCHEMA,validate,summary,forKind,sourceDigest:value.sourceDigest,certificate:value});
});
