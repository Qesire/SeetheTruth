'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const pages=["teaching.html","android_teaching.html","learn.html","tradeoffs.html","paper_lab.html","deepinfer_worker_lab.html","deepinfer_kv_lifecycle_lab.html","deepinfer_scheduler_lab.html","deepinfer_prefill_lab.html","cache.html","training.html","micro_training.html","capabilities.html","semantic_state.html","engineering_journey.html","engineering_pipeline.html"];
const allow=read('android/teaching-assets.txt').split(/\r?\n/).map(s=>s.trim()).filter(s=>s&&!s.startsWith('#'));
assert.equal(new Set(allow).size,allow.length,'duplicate entries');
const set=new Set(allow);
for(const filename of allow){
 assert.match(filename,/^[a-zA-Z0-9._-]+\.(?:js|html|css|svg|png|webp)$/);
 assert.ok(fs.existsSync(path.join(root,filename)),'missing allowlist resource '+filename);
 if(filename.endsWith('.html'))assert.ok(pages.includes(filename),'unexpected HTML screen '+filename);
}
for(const page of pages){
 assert.ok(set.has(page),'missing teaching page '+page);
 const html=read(page);
 for(const m of html.matchAll(/<(?:script|link)\b[^>]+(?:src|href)=["']([^"']+)["']/g)){
  const dep=m[1].split(/[?#]/)[0];
  if(/^[\w.-]+\.(?:js|css|png|svg|webp)$/.test(dep)){
   assert.ok(set.has(dep),'unpackaged page dependency '+page+' -> '+dep);
  }
 }
}
for(const file of ['android_teaching_mode.js','teaching_curriculum.js','block_capabilities.js'])assert.ok(set.has(file));
for(const excluded of ['index.html','real_model_lab.html','telemetry.html','atlas_server.py'])assert.ok(!set.has(excluded));
const gradle=read('android/app/build.gradle');
assert.ok(gradle.includes("teaching-assets.txt"));
assert.ok(!gradle.includes("include '*.js'")&&!gradle.includes("include '*.html'"));
console.log('ANDROID_ASSETS_PASS '+allow.length+' reviewed files and complete direct page dependencies');
