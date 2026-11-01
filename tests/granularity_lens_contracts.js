'use strict';
/* Independent architecture × granularity axes; canonical provenance and non-equivalence. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..'),read=p=>fs.readFileSync(path.join(root,p),'utf8');
const G=require('../granularity_lens.js');
const A=require('../architecture_perspective.js');
const C=require('../teaching_curriculum.js');
const Core=require('../learning_experience_core.js');
const L=require('../learning_lab.js');
const Catalog=require('../block_catalog.js');
const Composites=require('../composite_atlas.js');
const extras={operators:Catalog,composites:Composites};
assert.equal(G.GRAINS.length,5);
const courseIndex=G.curriculumIndex(C);
assert.equal(courseIndex.length,C.lessons.length);
assert.deepEqual(new Set(courseIndex.map(x=>x.id)),new Set(C.lessons.map(x=>x.id)));
assert.equal(courseIndex.find(x=>x.id==='engineering_pipeline').grain,'subsystem');
assert.equal(courseIndex.find(x=>x.id==='building').grain,'operation');
assert.equal(courseIndex.find(x=>x.id==='research_agent_arch').grain,'system');
assert.deepEqual(G.GRAINS.map(g=>g.id),['system','subsystem','component','operation','atomic']);
assert.equal(new Set(G.GRAINS.map(g=>g.id)).size,5);
assert.equal(G.rootTracks.length,A.paths(C).length);
assert.equal(L.KINDS.includes('opt_adamw'),true);
assert.equal(Catalog.KINDS.includes('opt_adamw'),true);
const allowedRelations=new Set(['ROOT','PEDAGOGICAL_ZOOM','SAME_LAB_REFOCUS','SAME_SIMULATION_REFOCUS']);
const allowedCoverages=new Set(['NOT_ESTABLISHED','PARTIAL_SUBSYSTEM','SOURCE_ALIGNED_LOCAL','ANALOGY_ONLY','UNPROVEN_EQUIVALENCE','SCRIPTED_MODEL']);
for(const track of G.rootTracks){
 const grains=G.inspectTrack(track,C,Core,extras);
 assert.equal(grains.length,5);
 assert.deepEqual(grains.map(item=>item.code),G.GRAINS.map(item=>item.code));
 for(const [i,item] of grains.entries()){
  assert.equal(item.grain,G.GRAINS[i].id);
  assert.equal(item.track,track);
  assert.ok(allowedRelations.has(item.relation),item.relation);
  assert.ok(allowedCoverages.has(item.containment),item.containment);
  assert.ok(item.sourceId.startsWith(item.targetKind+':'),item.sourceId);
  assert.ok(item.href.length>5);
  assert.ok(fs.existsSync(path.join(root,item.href.split('?')[0].split('#')[0])),item.href);
  if(item.targetKind==='lab'){
    const lab=Core.inspectLab(item.sourceId.slice(4));
    assert.equal(lab.href,item.href);
    assert.equal(lab.evidence,'NOT_EXECUTED');
  }else if(item.targetKind==='course')assert.equal(item.href,'teaching.html?focus='+item.sourceId.slice(7)+'#cards');
  else if(item.targetKind==='operator')assert.ok(L.defs[item.sourceId.slice(9)]);
  else assert.equal(item.targetKind,'research');
  const nav=G.navigation(track,item.grain,C,Core,extras);
  assert.deepEqual(nav.current,item);
  assert.equal(nav.parent?.code,grains[i-1]?.code);
  assert.equal(nav.child?.code,grains[i+1]?.code);
  assert.equal(nav.path.length,i+1);
  assert.equal(nav.numericallyEquivalent,'NOT_CLAIMED');
 }
}
assert.equal(G.resolve('agent','atomic',C,Core,extras).sourceId,'research:agent');
assert.equal(G.resolve('video','component',C,Core,extras).sourceId,'lab:wan_ffn');
assert.equal(G.resolve('posttrain','operation',C,Core,extras).sourceId,'operator:opt_adamw');
assert.match(G.resolve('posttrain','operation',C,Core,extras).href,/\?block=opt_adamw$/);
assert.throws(()=>G.resolve('no','system',C,Core,extras),/unknown granularity track/);
assert.throws(()=>G.resolve('video','no',C,Core,extras),/unknown granularity/);
assert.throws(()=>G.resolve('posttrain','operation',C,Core),/unknown registered operator/);
assert.throws(()=>G.resolve('posttrain','system',null,Core,extras),/canonical curriculum/);
assert.throws(()=>G.resolve('posttrain','system',C,null,extras),/unified learning core/);
const html=read('teaching.html'),hub=read('teaching_hub.js'),studio=read('learn.html');
for(const id of ['granularityExplorer','granularitySteps','granularityLaunch','granularityProof','granularityIdentity'])
 assert.ok(html.includes('id="'+id+'"'));
assert.ok(html.indexOf('src="block_catalog.js"')<html.indexOf('src="granularity_lens.js"'));
assert.ok(html.indexOf('src="granularity_lens.js"')<html.indexOf('src="teaching_hub.js"'));
assert.ok(hub.includes('Grain.inspectTrack(architectureTrack.id,C,Core,{operators:BlockCatalog})'));
assert.ok(hub.includes('window.SeetheTruthGranularityExplorer'));
assert.ok(studio.includes('grainFocus.get("block")')&&studio.includes('grainFocus.get("composite")'));
assert.ok(studio.includes('A.get(focusComposite)')&&studio.includes('L.KINDS.includes(focusBlock)'));
const policy=JSON.parse(read('static_asset_policy.json'));
assert.ok(policy.androidAssets.includes('granularity_lens.js'));
assert.ok(read('android/teaching-assets.txt').split(/\r?\n/).includes('granularity_lens.js'));
assert.ok(read('atlas_server.py').includes('"/granularity_lens.js":"granularity_lens.js"'));
console.log('GRANULARITY_CONTRACT_PASS orthogonal 3×5, real source bindings, no equivalent-function claims');
