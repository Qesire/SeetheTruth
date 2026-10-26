'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..'),read=p=>fs.readFileSync(path.join(root,p),'utf8');
const css=read('command_deck.css'),policy=JSON.parse(read('static_asset_policy.json'));
const assets=read('android/teaching-assets.txt').trim().split(/\r?\n/);
const pages=['index.html','learn.html','teaching.html','android_teaching.html',
 'deepinfer_kv_lifecycle_lab.html','deepinfer_scheduler_lab.html','deepinfer_worker_lab.html'];
assert.ok(policy.androidAssets.includes('command_deck.css'));
assert.ok(assets.includes('command_deck.css'));
assert.ok(read('atlas_server.py').includes('"/command_deck.css":"command_deck.css"'));
for(const page of pages){const html=read(page);
 assert.ok(html.includes('href="command_deck.css"'),page+' missing command deck');
 assert.ok(html.includes('data-theme="command-deck"'),page+' theme unscoped');
}
for(const x of ['.board-viewport','.block.selected','.command-contrast',
 'prefers-reduced-motion','@media(max-width:680px)','#stateLearningGuide']){
 assert.ok(css.includes(x),'missing source-visible style '+x);
}
assert.doesNotMatch(css,/@import|url\s*\(\s*['"]?https?:|blizzard|starcraft/i,
 'theme uses original CSS, with no copied external game artwork');
assert.ok(read('unified_state_lab_ui.js').includes('lab.compare(contrastBase,b)'));
assert.ok(read('unified_state_lab_ui.js').includes('lab.inspectStep(b,'));
console.log('COMMAND_DECK_CONTRACT_PASS '+pages.length+' original offline UI surfaces');
