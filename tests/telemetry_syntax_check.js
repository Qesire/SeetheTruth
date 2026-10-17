/* Validate inline telemetry client syntax, schema and first-party integration anchors. */
'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const text=fs.readFileSync('telemetry.html','utf8');
const scripts=[...text.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
assert(scripts.length>0,'must contain client script');
for(const s of scripts)if(s[1].trim())new vm.Script(s[1]);
assert(text.includes('seethetruth-telemetry-v1'));
assert(text.includes('data-snap'));
assert(text.includes('drawHistogram'));
assert(text.includes('drawMatrix'));
assert(text.includes('SYNTHETIC_DEMO_NOT_MEASURED'));
assert(text.includes('capture_values'));
const main=fs.readFileSync('index.html','utf8');
assert(main.includes('href="telemetry.html"'));
const server=fs.readFileSync('atlas_server.py','utf8');
assert(server.includes('"/telemetry.html":"telemetry.html"'));
console.log('Telemetry Observatory client and routing anchors PASS');
