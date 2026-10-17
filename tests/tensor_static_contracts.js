/* Tensor graph micro-contracts using deterministic dimensions and exact edges. */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const context={window:{}};
vm.runInNewContext(fs.readFileSync('tensor_subgraph.js','utf8'),context);
const graphEngine=context.window.SeetheTruthTensor;

function make(kind,focus,mode='prefill'){
 const st={cfg:{kind,b:2,s:6,h:64,a:8,kv:kind==='gqa'?2:8,f:192,e:4,k:2},mode,focus};
 const palette={input:'#a',norm:'#b',cond:'#c',ffn:'#d',attn:'#e',head:'#f',gate:'#9',output:'#8',router:'#7',residual:'#6'};
 const node=(id,label)=>({id,label});
 const edge=(a,b)=>({a,b});
 const graph=graphEngine.build({st,palette,node,edge});
 const ids=graph.nodes.map(n=>n.id);
 assert.equal(new Set(ids).size,ids.length,'duplicate scene node IDs');
 assert.ok(graph.edges.every(e=>ids.includes(e.a)&&ids.includes(e.b)),'dangling edge');
 return {st,graph};
}
function detail(x,key){return graphEngine.detail('tx_'+key,x.st)}
function contains(x,a,b){return x.graph.edges.some(e=>e.a==='tx_'+a&&e.b==='tx_'+b)}
function check(x,key,expected){const v=detail(x,key);assert.ok(v,key);assert.equal(v.out,expected,key)}
const g=make('gqa','attention');
check(g,'qview','[2,6,8,8]');
check(g,'kview','[2,6,2,8]');
check(g,'qk','[2,8,6,6]');
check(g,'av','[2,8,6,8]');
check(g,'wo','[2,6,64]');
for(const [a,b] of [['q','qview'],['qview','qperm'],['qperm','qrope'],['qrope','qk'],
 ['krope','kcache'],['kcache','kexpand'],['kexpand','kt'],['qk','mask'],['mask','softmax'],
 ['softmax','av'],['vexpand','av'],['av','layout'],['layout','contiguous'],['contiguous','merge'],['merge','wo']]){
 assert.ok(contains(g,a,b),a+'→'+b);
}
const decode=make('gqa','attention','decode');
check(decode,'qview','[2,1,8,8]');
check(decode,'kview','[2,1,2,8]');
check(decode,'qk','[2,8,1,6]');
const dit=make('dit','attention');
assert.ok(!dit.graph.nodes.some(n=>n.id==='tx_kcache'));
check(dit,'qk','[2,8,6,6]');
const ffn=make('gqa','ffn');
check(ffn,'gmm','[2,6,192]');
check(ffn,'down','[2,6,64]');
assert.ok(contains(ffn,'silu','mul')&&contains(ffn,'umm','mul'));
const gpt=make('gpt','ffn');
check(gpt,'w1','[2,6,192]');
check(gpt,'w2','[2,6,64]');
const moe=make('moe','ffn');
assert.ok(contains(moe,'routermm','topk')&&contains(moe,'topk','dispatch'));
for(const id of ['e1g','e1u','e1act','e1mul','e1down','e2g','e2u','e2act','e2mul','e2down'])assert.ok(moe.graph.nodes.some(n=>n.id==='tx_'+id));
console.log('Deterministic tensor dimensions, routing branches and 3D edge contracts PASS');
