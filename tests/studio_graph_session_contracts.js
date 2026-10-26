'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../learn.html'),'utf8');
const scripts=[...html.matchAll(/<script src="([^"]+)"><\/script>/g)].map(m=>m[1]);
const required=['learning_lab.js','branch_execution.js','graph_readiness.js','block_capabilities.js','operator_contracts.js','execution_ir.js','graph_session.js'];
for(const p of required)assert.ok(scripts.includes(p),'missing studio runtime '+p);
const server=fs.readFileSync(path.join(__dirname,'../atlas_server.py'),'utf8');
const assets=new Set(fs.readFileSync(path.join(__dirname,'../android/teaching-assets.txt'),'utf8').split(/\r?\n/).map(x=>x.trim()));
for(const p of required){
 assert.ok(server.includes('"/'+p+'":"'+p+'"'),'server does not serve '+p);
 assert.ok(assets.has(p),'Android offline manifest misses '+p);
}

// Assert the actual dependency DAG; a single total order would incorrectly
// require capability presentation to initialize before source ExecutionIR.
const dependencies=[
 ['learning_lab.js','branch_execution.js'],
 ['branch_execution.js','graph_lowering_plan.js'],
 ['shape_index_rules.js','execution_ir.js'],
 ['execution_ir.js','graph_lowering_plan.js'],
 ['graph_lowering_plan.js','python_codegen.js'],
 ['graph_lowering_plan.js','torch_codegen.js'],
 ['python_codegen.js','block_capabilities.js'],
 ['torch_codegen.js','block_capabilities.js'],
 ['block_capabilities.js','operator_contracts.js'],
 ['operator_contracts.js','graph_session.js'],
 ['execution_ir.js','graph_session.js']
];
for(const [source,consumer] of dependencies)
 assert.ok(scripts.indexOf(source)>=0&&scripts.indexOf(source)<scripts.indexOf(consumer),
   'load order '+source+' -> '+consumer);
for(const exporter of ['python_codegen.js','torch_codegen.js'])assert.ok(scripts.indexOf(exporter)<scripts.indexOf('block_capabilities.js'),'capability audit requires '+exporter);
assert.ok(scripts.indexOf('block_knowledge.js')<scripts.indexOf('block_capabilities.js'));

assert.ok(html.includes('readinessSnapshot=Session.inspect(graph).readiness;'));
assert.ok(html.includes('const session=Session.inspect(graph,{numeric:studyMode()==="numeric"});'));
assert.ok(html.includes('Session.inspect(graph,{numeric:true})'));
const L=require('../learning_lab.js'),Session=require('../graph_session.js');
const g=L.preset('swiglu'),s=Session.inspect(g,{numeric:true});
assert.equal(s.executable,true);assert.ok(s.trace&&s.ir);
assert.deepEqual(s.ir.nodes.map(n=>n.id),s.trace.steps.map(n=>n.id));
console.log('STUDIO_GRAPH_SESSION_PASS');
