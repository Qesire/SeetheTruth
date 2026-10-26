#!/usr/bin/env node
'use strict';
// Deterministic build-time sync from the trusted installed-block-pack manifest.
// Usage: node scripts/sync_installed_block_packs.js --write | --check
// No downloading, eval, dynamic browser script injection or hidden runtime assets.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const ROOT=path.resolve(__dirname,'..');
const Manifest=require('../installed_block_packs.js');
const names=Manifest.entries.map(e=>e.module);
const SCRIPTS=[...names,'installed_block_packs.js'];
const mod=(name,fn)=>{
 const file=path.join(ROOT,name),was=fs.readFileSync(file,'utf8'),next=fn(was);
 if(next===was)return false;
 if(process.argv.includes('--check'))throw Error(name+' differs from trusted pack manifest (run --write)');
 if(!process.argv.includes('--write'))throw Error('use --check or --write');
 fs.writeFileSync(file,next);return true;
};
const tag=name=>'<script src="'+name+'"></script>';
const htmlFiles=['learn.html','capabilities.html'];
const marker=tag('block_extension_registry.js');
const htmlRe=/<script src="(?:[a-z][a-z0-9_]*_block_pack|installed_block_packs)\.js"><\/script>\s*/g;
const routeRe=/"\/(?:[a-z][a-z0-9_]*_block_pack|installed_block_packs)\.js":"(?:[a-z][a-z0-9_]*_block_pack|installed_block_packs)\.js",/g;
const assetRe=/^(?:[a-z][a-z0-9_]*_block_pack|installed_block_packs)\.js\r?\n/gm;
for(const e of Manifest.entries){
 assert.ok(fs.existsSync(path.join(ROOT,e.module)),'missing trusted pack source '+e.module);
 const actual=require(path.join(ROOT,e.module));
 assert.equal(actual.name,e.id,'source identity differs '+e.id);
}
const normalize=(content,drop,anchor,prepend,name)=>{
 const pure=content.replace(drop,'');
 if(!pure.includes(anchor))throw Error('missing install anchor in '+name);
 return pure.replace(anchor,prepend+anchor);
};
let changed=[];
for(const file of htmlFiles)if(mod(file,s=>normalize(s,htmlRe,marker,
 SCRIPTS.map(tag).join('\n')+'\n',file)))changed.push(file);
// The static publication policy owns HTTP and Android outputs. The pack
// manifest contributes its modules to that one generator, not another writer.
require('./sync_runtime_assets.js').sync(process.argv.includes('--write')?'--write':'--check');
if(!process.argv.includes('--check')&&!process.argv.includes('--write'))
 throw Error('supply --check or --write');
console.log('TRUSTED_PACK_INSTALL_'+(process.argv.includes('--check')?'CHECK':'SYNC')+'_PASS '+JSON.stringify({
 installed:Manifest.entries.map(e=>e.id),changes:changed
}));
