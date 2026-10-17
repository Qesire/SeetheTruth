/* Compile actual inline Learning Studio script with a source filename. */
'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const page=fs.readFileSync('learn.html','utf8');
const scripts=[...page.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
assert.ok(scripts.length>=1);
for(const [i,script] of scripts.entries()){
 if(script[1].trim())new vm.Script(script[1],{filename:'learn.html:inline:'+i});
}
assert.match(page,/learning_lab\.js/);
assert.match(page,/block_catalog\.js/);
assert.match(page,/block_knowledge\.js/);
assert.match(page,/composite_atlas\.js/);
assert.match(page,/id="compositeLaunch"/);
assert.match(page,/id="compositeNavigator"/);
assert.match(page,/id="blockKnowledge"/);
assert.match(page,/concept_flow\.js/);
assert.match(page,/id="studyMode"/);
assert.match(page,/id="blockSearch"/);
assert.match(page,/id="catalogCategories"/);
assert.match(page,/tensor_ir\.js/);
const training=fs.readFileSync('training.html','utf8');
const trainingScripts=[...training.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
for(const [i,script] of trainingScripts.entries()){
 if(script[1].trim())new vm.Script(script[1],{filename:'training.html:inline:'+i});
}
assert.match(training,/training_flow\.js/);
assert.match(training,/id="sequence"/);
assert.match(page,/training\.html/);
const server=fs.readFileSync('atlas_server.py','utf8');
const cache=fs.readFileSync('cache.html','utf8');
const cacheScripts=[...cache.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
for(const [i,script] of cacheScripts.entries()){
 if(script[1].trim())new vm.Script(script[1],{filename:'cache.html:inline:'+i});
}
assert.match(cache,/kv_cache_lab\.js/);
assert.match(cache,/id="cacheSlots"/);
assert.match(cache,/id="headSlice"/);
assert.match(server,/cache\.html/);
assert.match(server,/training\.html/);
assert.match(server,/training_flow\.js/);
assert.match(server,/kv_cache_lab\.js/);
assert.match(page,/id="blocks"/);
assert.match(page,/id="wires"/);
assert.match(page,/id="challenge"/);
assert.match(page,/TEACHING SIMULATION/);
assert.match(server,/learn\.html/);
assert.match(server,/learning_lab\.js/);
assert.match(server,/block_catalog\.js/);
console.log('Learning Studio client syntax and local-only route PASS');
