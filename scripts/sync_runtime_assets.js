#!/usr/bin/env node
'use strict';
/* Deterministic, reviewed static publication policy.
 * One policy for Android assets and local-only HTTP routes; pack entries are
 * derived from installed_block_packs.js. HTML load order remains authored in
 * the actual pages, and checked against their dependency DAG.
 */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const ROOT=path.resolve(__dirname,'..');
const PACKS=require('../installed_block_packs.js');
const POLICY_PATH=path.join(ROOT,'static_asset_policy.json');
const POLICY=JSON.parse(fs.readFileSync(POLICY_PATH,'utf8'));
const fileRx=/^[a-zA-Z0-9][a-zA-Z0-9._-]*\.(?:js|css|html|json|svg|png|webp)$/;
function validate(policy,packModules,exists=file=>fs.existsSync(path.join(ROOT,file)),
  read=file=>fs.readFileSync(path.join(ROOT,file),'utf8')){
 assert.equal(policy?.schema,'seethetruth-static-asset-policy-v1','static policy schema');
 assert.ok(Array.isArray(policy.androidAssets)&&Array.isArray(policy.webOnly));
 const packs=[...packModules,'installed_block_packs.js'];
 const all=[...policy.androidAssets,...policy.webOnly,...packs];
 for(const file of all){
  assert.match(file,fileRx,'invalid publication filename '+file);
  assert.ok(exists(file),'missing static file '+file);
 }
 assert.equal(new Set(all).size,all.length,'duplicate/multiply owned static file');
 const android=[...policy.androidAssets,...packs].sort();
 const published=[...new Set([...all])].sort();
 const setAndroid=new Set(android),setWeb=new Set(published);
 for(const filename of published.filter(file=>file.endsWith('.html'))){
  const source=read(filename);
  for(const match of source.matchAll(/<(?:script|link)\b[^>]+(?:src|href)=["']([^"']+)["']/g)){
   const dependency=match[1].split(/[?#]/)[0];
   if(!fileRx.test(dependency))continue; // external URLs and non-file hrefs
   assert.ok(setWeb.has(dependency),filename+' missing HTTP dependency '+dependency);
   if(setAndroid.has(filename))
    assert.ok(setAndroid.has(dependency),filename+' missing Android dependency '+dependency);
  }
 }
 assert.ok(setWeb.has('index.html'),'missing web root');
 assert.ok(setAndroid.has('teaching.html')&&setAndroid.has('learn.html'),
   'missing offline teaching entry pages');
 assert.ok(!setAndroid.has('index.html')&&!setAndroid.has('real_model_lab.html'),
   'desktop-only experience leaked into Android package');
 return Object.freeze({android:Object.freeze(android),published:Object.freeze(published)});
}
const oldRoute=/^        mapping=\{[^\r\n]*\}$/m;
function render(policy=POLICY,packModules=PACKS.entries.map(x=>x.module),
  exists,read){
 const data=validate(policy,packModules,exists,read);
 const map={'/':'index.html'};
 for(const asset of data.published)map['/'+asset]=asset;
 const mapping='        mapping='+JSON.stringify(map);
 const assetList=data.android.join('\n')+'\n';
 return Object.freeze({data,mapping,assetList});
}
function sync(mode){
 assert.ok(['--check','--write'].includes(mode),'use --check or --write');
 const {mapping,assetList,data}=render();
 const targets=[
  ['atlas_server.py',before=>{
   assert.ok(oldRoute.test(before),'missing exact static route anchor');
   return before.replace(oldRoute,mapping);
  }],
  ['android/teaching-assets.txt',()=>assetList]
 ];
 const changed=[];
 for(const [filename,transform] of targets){
  const target=path.join(ROOT,filename),before=fs.readFileSync(target,'utf8');
  const after=transform(before);
  if(after===before)continue;
  changed.push(filename);
  if(mode==='--check')throw Error(filename+' differs from single static asset policy (run node scripts/sync_runtime_assets.js --write)');
  fs.writeFileSync(target,after);
 }
 console.log('STATIC_ASSET_'+(mode==='--check'?'CHECK':'SYNC')+'_PASS '+JSON.stringify({
  android:data.android.length,served:data.published.length+1,
  packModules:PACKS.entries.length,changed
 }));
}
if(require.main===module)sync(process.argv[2]);
module.exports=Object.freeze({validate,render,sync});
