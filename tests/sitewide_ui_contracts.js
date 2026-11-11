'use strict';
/* One source-bound acceptance for all SeetheTruth human-facing routes.
 * This is structural, not a substitute for Playwright, real Android, or user trials.
 */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..'),read=p=>fs.readFileSync(path.join(root,p),'utf8');
const Core=require('../learning_experience_core.js');
const PAGES=[
 'index.html','learn.html','teaching.html','android_teaching.html',
 'cache.html','capabilities.html','tradeoffs.html','training.html',
 'micro_training.html','paper_lab.html','real_model_lab.html',
 'semantic_state.html','telemetry.html','wan_ffn_lab.html',
 'deepinfer_prefill_lab.html','deepinfer_scheduler_lab.html',
 'deepinfer_kv_lifecycle_lab.html','deepinfer_worker_lab.html',
 'engineering_journey.html','engineering_pipeline.html'
];
assert.equal(PAGES.length,20);
assert.equal(new Set(PAGES).size,20);
const policy=require('../static_asset_policy.json');
assert.ok(policy.androidAssets.includes('ui_accessibility.css'));
assert.ok(read('android/teaching-assets.txt').split(/\r?\n/).includes('ui_accessibility.css'));
assert.ok(read('atlas_server.py').includes('"/ui_accessibility.css":"ui_accessibility.css"'));
for(const p of PAGES){
 const html=read(p);
 assert.match(html,/name=["']viewport["']/i,p+' responsive viewport absent');
 assert.ok(html.includes('href="ui_accessibility.css"'),p+' lacks shared keyboard surface');
 assert.ok(html.includes('class="stt-skip" href="#stt-main"'),p+' lacks keyboard skip link');
 assert.ok(html.includes('<main id="stt-main"'),p+' lacks primary landmark');
 assert.ok(html.includes('href="teaching.html'),p+' cannot navigate to shared curriculum');
 assert.ok(/<h1\b/i.test(html),p+' lacks page title');
}
const studio=read('learn.html'),source=read('teaching_hub.js');
assert.ok(studio.includes('id="jumpToGraph"')&&studio.includes('href="#stt-main"'));
assert.ok(studio.includes('class="stt-learning-studio"'));
assert.ok(studio.includes('id="jumpToCatalog"')&&studio.includes('id="stt-catalog"'));
assert.ok(read('ui_accessibility.css').includes('body.stt-learning-studio .app>.stage{order:0}'));
assert.ok(studio.includes('class="stt-lab-menu"')&&studio.includes('其他实验室'));
assert.ok(read('index.html').includes('数据与工具'));
assert.ok(read('android_teaching.html').includes('href="teaching.html?edition=teaching"'));
assert.ok(source.includes('for(const lab of Core.labs)'),
 'specialist discovery must derive from canonical Core.labs not a second URL registry');
assert.equal(Core.labs.length,17);
const excluded=new Set(["index.html","real_model_lab.html","telemetry.html"]);
for(const lab of Core.labs){
 assert.ok(PAGES.includes(lab.href),'unknown specialist route '+lab.href);
 if(excluded.has(lab.href))assert.ok(!policy.androidAssets.includes(lab.href),
   'offline package must exclude web-only lab '+lab.href);
 else assert.ok(policy.androidAssets.includes(lab.href),
   'specialist Android source route missing '+lab.href);
}
const telemetry=read('telemetry.html');
for(const name of ['aria-label="热力图编码方式"','aria-label="采样值统计视图"',
 'aria-label="张量数值热力图"','aria-label="张量采样值的统计图"'])
 assert.ok(telemetry.includes(name),'inaccessible telemetry control '+name);
for(const p of ['engineering_journey.html','engineering_pipeline.html'])
 assert.ok(read(p).includes('无法撤销'),p+' reset consequences omitted');
console.log('SITEWIDE_UI_CONTRACT_PASS '+JSON.stringify({
 pages:PAGES.length,specialistDomains:Core.labs.length,androidExcluded:[...excluded],
 claims:'structural only; browser/mobile screenshots and task trials are separate'
}));
