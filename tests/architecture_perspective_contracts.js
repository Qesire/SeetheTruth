'use strict';
/* F-tier architecture presentation contract: no new executor claims. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const A=require('../architecture_perspective.js');
const C=require('../teaching_curriculum.js');
const L=require('../learning_lab.js');
const names=A.LEVELS.map(x=>x.id);
assert.deepEqual(names,['system','architecture','runtime','operator','evidence']);
assert.equal(new Set(names).size,5);
assert.equal(A.paths(C).length,3);
for(const track of A.paths(C)){
 assert.equal(track.steps.length,5);
 assert.deepEqual(track.steps.map(x=>x.level),names);
 assert.ok(C.lookup(track.course));
 assert.match(track.goal,/./);
 assert.match(track.metrics,/./);
 for(const stage of track.steps){
  const lesson=C.lookup(stage.course);
  assert.equal(stage.href,lesson.href,'all architecture links resolve via canonical curriculum');
  assert.ok(read(stage.href).length>50);
  assert.ok(stage.evidence);
 }
}
assert.deepEqual(A.paths(C),A.paths(C),'paths must be deterministic and pure');
assert.throws(()=>A.paths({lookup:()=>null}),/unknown shared course/);
assert.throws(()=>A.paths({}),/shared curriculum/);
assert.deepEqual(A.roles.map(r=>r.id),['all','operator','representation','runtime','training']);
const roles=new Set(A.roles.map(x=>x.id));
const seen=new Set();
for(const kind of L.KINDS){
 const d=L.defs[kind];
 const role=A.roleFor(kind,d);
 assert.ok(roles.has(role));
 seen.add(role);
}
for(const role of ['operator','representation','runtime','training'])assert.ok(seen.has(role),role);
assert.equal(A.roleFor('inference_paged_kv_read',L.defs.inference_paged_kv_read),'runtime');
assert.equal(A.roleFor('nd_gelu_erf',L.defs.nd_gelu_erf),'operator');
assert.throws(()=>A.roleFor('bad',null),/registered operator/);
const html=read('teaching.html'),hub=read('teaching_hub.js'),studio=read('learn.html');
for(const id of ['architectureAtlas','architectureTracks','architectureSteps','architectureLaunch'])assert.ok(html.includes('id="'+id+'"'));
assert.ok(html.indexOf('src="architecture_perspective.js"')<html.indexOf('src="teaching_hub.js"'));
assert.ok(hub.includes('Architecture.paths(C)'));
assert.ok(hub.includes('window.SeetheTruthArchitectureAtlas'));
assert.ok(studio.includes('id="architectureBlockRoles"')&&studio.includes('id="architectureRoleHint"'));
assert.ok(studio.includes('ArchitecturePerspective.roleFor(kind,def)'));
assert.ok(studio.includes('roleFilter==="all"'));
assert.ok(studio.includes('data-role-filter'));
assert.ok(studio.includes('data-add='));
assert.ok(studio.includes('这些入口连接已有状态机与专家实验室'));
assert.ok(studio.indexOf('src="architecture_perspective.js"')<studio.indexOf('(()=>{"use strict";'));
const policy=JSON.parse(read('static_asset_policy.json'));
const assets=read('android/teaching-assets.txt');
const server=read('atlas_server.py');
assert.ok(policy.androidAssets.includes('architecture_perspective.js'));
assert.ok(assets.split(/\r?\n/).includes('architecture_perspective.js'));
assert.ok(server.includes('"/architecture_perspective.js":"architecture_perspective.js"'));
console.log('ARCHITECTURE_PERSPECTIVE_PASS 3 tracks × 5 layers, role partition, shared PC/Android route, real graph identity');
