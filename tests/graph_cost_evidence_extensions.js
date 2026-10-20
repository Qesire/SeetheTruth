'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js'),Cost=require('../graph_cost_report.js');
const node=(id,kind,params={})=>({id,kind,params,x:0,y:0});
const graph=(nodes,edges)=>L.validateGraph({schema:L.SCHEMA,title:'shape-grounded oracle',nodes,edges:edges.map(([from,to,port])=>({from,to,port}))});
const attention=graph([
 node('q','input',{rows:2,width:4,seed:1}),
 node('k','input',{rows:3,width:4,seed:2}),
 node('v','input',{rows:3,width:6,seed:3}),
 node('attn','attention',{})
],[['q','attn','q'],['k','attn','k'],['v','attn','v']]);
const a=Cost.profile(attention);
assert.equal(a.attentionNodes,1);
assert.equal(a.attentionMAC,2*3*(4+6));
assert.equal(a.countedMatmulMAC,0,'fused attention MAC must not masquerade as explicit MatMul nodes');
assert.equal(a.knownArithmeticMAC,60);
assert.ok(a.rows.find(n=>n.id==='attn').attentionMAC===60);
const gqa=graph([
 node('q','input',{rows:2,width:4,seed:1}),
 node('k','input',{rows:3,width:2,seed:2}),
 node('v','input',{rows:3,width:2,seed:3}),
 node('g','gqa',{heads:2,kvHeads:1,rope:false,causal:false})
],[['q','g','q'],['k','g','k'],['v','g','v']]);
assert.equal(Cost.profile(gqa).attentionMAC,2*2*3*(2+2));
assert.equal(Cost.profile(gqa).knownArithmeticFlopTerms,96);
const expert=graph([
 node('x','input',{rows:4,width:2,seed:1}),
 node('idx','input',{rows:2,width:2,seed:2}),
 node('weights','input',{rows:8,width:3,seed:3}),
 node('e','expert_linear_bank',{})
],[['x','e','x'],['idx','e','indices'],['weights','e','weights']]);
const e=Cost.profile(expert);
assert.equal(e.expertLinearNodes,1);
assert.equal(e.expertLinearMAC,4*2*3);
assert.equal(e.physicalParameterElements,0,'synthetic input weight bank is not trainable Parameter');
const cache=graph([
 node('old','nd_input',{b:1,h:2,t:2,d:4,seed:1}),
 node('new','nd_input',{b:1,h:2,t:1,d:4,seed:2}),
 node('append','nd_cache_append',{})
],[['old','append','cache'],['new','append','segment']]);
const c=Cost.profile(cache);
assert.equal(c.cacheSnapshotElements,1*2*3*4);
assert.equal(c.cacheSnapshots[0].logicalElements,24);
assert.equal(c.knownArithmeticMAC,0);
const diff=Cost.compare(attention,gqa);
assert.equal(diff.deltas.attentionMAC.baseline,60);
assert.equal(diff.deltas.attentionMAC.candidate,48);
assert.equal(diff.deltas.attentionMAC.difference,-12);
assert.equal(diff.deltas.attentionMAC.ratio,0.8);
assert.equal(Cost.compare(cache,cache).deltas.cacheSnapshotElements.difference,0);
const invalid=JSON.parse(JSON.stringify(attention));
invalid.edges=invalid.edges.filter(x=>x.port!=='k');
assert.throws(()=>Cost.compare(invalid,attention),/Incomplete or invalid graph/);
const gqaBad=JSON.parse(JSON.stringify(gqa));
gqaBad.nodes.find(x=>x.id==='g').params.kvHeads=3;
assert.throws(()=>Cost.profile(gqaBad),/invalid|divis|整除/i);
console.log('GRAPH_EVIDENCE_EXTENSIONS_PASS: 2D attention, GQA, routed expert, 4D cache, comparable graph deltas');
