'use strict';
/* Real source-policy projection with adversarial validation. A third installed
 * pack is included by the same generator; no hidden server/Android lists.
 */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const root=path.resolve(__dirname,'..');
const Policy=require('../static_asset_policy.json');
const Packs=require('../installed_block_packs.js');
const Sync=require('../scripts/sync_runtime_assets.js');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const modules=Packs.entries.map(e=>e.module);
const result=Sync.render();
assert.equal(result.data.android.length,125);
assert.equal(result.data.published.length,133);
assert.equal(read('android/teaching-assets.txt'),result.assetList,
 'Android list must equal canonical generated publication');
assert.ok(read('atlas_server.py').includes(result.mapping),
 'HTTP whitelist must exactly mirror the canonical generated publication');
assert.deepEqual(result.data.android.filter(n=>n.endsWith('_block_pack.js')),
 modules.slice().sort(),'trusted module count is source-driven');
assert.deepEqual(result.data.published.filter(n=>n.endsWith('_block_pack.js')),
 modules.slice().sort(),'HTTP pack module count is source-driven');
assert.ok(result.data.android.every(n=>result.data.published.includes(n)));
const command=execFileSync(process.execPath,
 [path.join(root,'scripts/sync_runtime_assets.js'),'--check'],
 {cwd:root,encoding:'utf8'});
assert.match(command,/STATIC_ASSET_CHECK_PASS/);
const policyClone=()=>structuredClone(Policy);
const fails=(mutate,pattern)=>{const clone=policyClone();mutate(clone);assert.throws(
 ()=>Sync.validate(clone,modules),pattern);};
fails(p=>p.webOnly.push(p.androidAssets[0]),/duplicate/);
fails(p=>p.androidAssets.push('../secrets.py'),/invalid publication filename/);
fails(p=>p.androidAssets.push('new_unreviewed.js'),/missing static file/);
fails(p=>p.androidAssets=p.androidAssets.filter(x=>x!=='learn.html'),
 /missing offline teaching entry pages/);
fails(p=>p.androidAssets.push('index.html'),/duplicate\/multiply owned static file/);
const fakeName='research_extension_block_pack.js';
const simulated=Sync.render(Policy,[...modules,fakeName],
 file=>file===fakeName||fs.existsSync(path.join(root,file)),
 file=>read(file));
assert.ok(simulated.data.android.includes(fakeName)&&simulated.data.published.includes(fakeName),
 'third installed pack must reach BOTH destinations automatically');
const unavailable=Sync.validate.bind(null,Policy,[...modules,fakeName]);
assert.throws(()=>unavailable(),/missing static file/);
const scriptInjected=filename=>filename==='learn.html'
 ?read(filename).replace('</body>','<script src="unreviewed_new_runtime.js"></script></body>')
 :read(filename);
assert.throws(()=>Sync.validate(Policy,modules,undefined,scriptInjected),
 /missing HTTP dependency unreviewed_new_runtime\.js/);
const androidInjected=filename=>filename==='learn.html'
 ?read(filename).replace('</body>','<script src="tensor_subgraph.js"></script></body>')
 :read(filename);
assert.throws(()=>Sync.validate(Policy,modules,undefined,androidInjected),
 /missing Android dependency tensor_subgraph\.js/);
console.log('STATIC_ASSET_POLICY_PASS '+JSON.stringify({
 offline:result.data.android.length,published:result.data.published.length,
 installedPacks:modules.length,tests:['duplicate','traversal','unknown','entry-policy',
 'offline-web-boundary','third-pack','missing-pack','unreviewed-import','web-only-leak']
}));
