'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const read=p=>fs.readFileSync(path.join(__dirname,'..',p),'utf8');
const html=read('learn.html'),server=read('atlas_server.py');
const resources=new Set(read('android/teaching-assets.txt').split(/\r?\n/).map(x=>x.trim()).filter(Boolean));
const scripts=[...html.matchAll(/<script src="([^"]+)"\s*><\/script>/g)].map(m=>m[1]);
for(const src of ['training_session.js','gradient_inspection.js','node_timeline.js']){
 assert.ok(scripts.includes(src),'Learning Studio missing '+src);
 assert.ok(server.includes('"/'+src+'":"'+src+'"'),'HTTP server missing '+src);
 assert.ok(resources.has(src),'Android offline asset missing '+src);
}
assert.ok(scripts.indexOf('training_session.js')<scripts.indexOf('node_timeline.js'));
assert.ok(scripts.indexOf('gradient_inspection.js')<scripts.indexOf('node_timeline.js'));
assert.ok(html.includes('NodeTimeline.fromSession('));
assert.ok(html.includes('data-timeline-phase'));
assert.ok(html.includes('id="timelineStep"'));
assert.ok(html.includes('renderNodeTimeline(trainStatus)'));
console.log('STUDIO_NODE_TIMELINE_PASS: script order, data source, HTML controls, HTTP and Android packaging');
