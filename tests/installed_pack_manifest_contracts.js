'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {execFileSync}=require('node:child_process');
const ROOT=path.resolve(__dirname,'..');
const Manifest=require('../installed_block_packs.js');
const X=require('../block_extension_registry.js');
const L=require('../learning_lab.js');
assert.equal(Manifest.SCHEMA,'seethetruth-installed-block-packs-v1');
assert.equal(Manifest.entries.length,X.packs.length);
assert.deepEqual(Manifest.entries.map(x=>x.id),X.packs);
assert.equal(new Set(Manifest.entries.map(x=>x.module)).size,Manifest.entries.length);
assert.equal(Manifest.load().length,Manifest.entries.length);
for(const entry of Manifest.entries){
 const impl=require(path.join(ROOT,entry.module));
 assert.equal(impl.name,entry.id);
 assert.ok(Object.keys(impl.defs).length>0);
 for(const kind of Object.keys(impl.defs)){
  assert.ok(L.KINDS.includes(kind),'manifest must reach active graph execution '+kind);
  assert.ok(X.isKind(kind));
 }
}
assert.throws(()=>Manifest.validate([]),/1..32/);
assert.throws(()=>Manifest.validate([{id:'x',module:'https://malicious.invalid/evil.js',global:'window.alert'}]),/invalid trusted/);
assert.throws(()=>Manifest.validate([...Manifest.entries,Manifest.entries[0]]),/duplicate/);
assert.throws(()=>Manifest.resolve([{id:'untrusted_local',module:'safe_block_pack.js',global:'SeetheTruthSafeBlockPack'}],()=>null),/source\/identity mismatch/);
const synthetic={
 name:'isolated_pack',
 defs:{isolated_source:{title:'Independent',en:'Independent',category:'data',ports:[],params:{n:3},
  formula:'x[i] = i',why:'proof of general registry',lesson:'follow indices',
  caution:'synthetic',source:'https://docs.pytorch.org/docs/stable/generated/torch.arange.html'}},
 validate:(_,p)=>{if(!Number.isSafeInteger(p.n)||p.n<1||p.n>6)throw Error('invalid n')},
 infer:(_,p)=>[p.n],
 execute:(_,p)=>({values:Array.from({length:p.n},(_,i)=>i)})
};
const p=Manifest.resolve([{id:'isolated_pack',module:'isolated_block_pack.js',
 global:'SeetheTruthIsolatedBlockPack'}],()=>synthetic);
const isolated=X.createRegistry(p);
assert.deepEqual(isolated.infer({kind:'isolated_source',params:{n:4}},{}).shape,[4]);
assert.deepEqual(isolated.execute({kind:'isolated_source',params:{n:4}},{}).values,[0,1,2,3]);
assert.equal(L.KINDS.includes('isolated_source'),false,'build-time-only probe may not silently mutate active runtime');
const run=execFileSync(process.execPath,[path.join(ROOT,'scripts/sync_installed_block_packs.js'),'--check'],{
 cwd:ROOT,encoding:'utf8'});
assert.match(run,/TRUSTED_PACK_INSTALL_CHECK_PASS/);
console.log('TRUSTED_INSTALLED_PACK_CONTRACT_PASS: ordered trusted install, actual graph binding, independently creatable third pack, deterministic offline asset sync');
