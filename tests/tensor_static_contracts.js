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

/* Cross-check the teaching DAG against the six-slot, two-gate DiTBlock
 * structure present in pinned Meta DiT (see test_pinned_llama_mixtral_dit.py).
 * A zero residual gate must preserve the input, not add modulated features.
 */
const ada=make('dit','adaln');
for(const slot of ['shift_msa','scale_msa','gate_msa_param','shift_mlp_param','scale_mlp_param','gate_mlp_param'])
 assert.ok(contains(ada,'split',slot),'missing distinct DiT parameter path: '+slot);
for(const [from,to] of [
 ['c','csilu'],['csilu','cw'],['cw','split'],
 ['shift_msa','shift'],['scale_msa','shift'],['norm','shift'],
 ['shift','attn_f'],['attn_f','gate_msa'],['gate_msa_param','gate_msa'],
 ['x','msa_res'],['gate_msa','msa_res'],['msa_res','norm2'],
 ['norm2','shift_mlp'],['shift_mlp_param','shift_mlp'],['scale_mlp_param','shift_mlp'],
 ['shift_mlp','mlp_f'],['mlp_f','gate_mlp'],['gate_mlp_param','gate_mlp'],
 ['msa_res','adaln_out'],['gate_mlp','adaln_out']
]) assert.ok(contains(ada,from,to),'DiT gated-residual dependency omitted: '+from+' -> '+to);
assert.ok(!contains(ada,'shift','msa_res')&&!contains(ada,'shift','adaln_out'),
          'modulated input must pass through attention before residual addition');
assert.match(graphEngine.ops.gate_msa[1],/gate_msa/);
assert.match(graphEngine.ops.gate_mlp[1],/gate_mlp/);
check(ada,'shift','[2,6,64]');
check(ada,'adaln_out','[2,6,64]');

/* Deterministic mathematical oracle: both zero gates must make a DiTBlock
 * act as identity, regardless of conditioning/modulated branch outputs. */
function gatedResidual(x,attn,mlp,gateAttn,gateMlp){
 const afterAttn=x.map((v,i)=>v+gateAttn[i]*attn[i]);
 return afterAttn.map((v,i)=>v+gateMlp[i]*mlp[i]);
}
const original=[1.25,-3.0,4.5], attn=[30,-50,70],mlp=[-90,80,60];
assert.deepEqual(gatedResidual(original,attn,mlp,[0,0,0],[0,0,0]),original);
assert.deepEqual(gatedResidual(original,attn,mlp,[1,1,1],[0,0,0]),original.map((v,i)=>v+attn[i]));
assert.deepEqual(gatedResidual(original,attn,mlp,[0,0,0],[1,1,1]),original.map((v,i)=>v+mlp[i]));
assert.ok(contains(moe,'x','moeout'),'MoE(x) residual has to depend on original input x');
assert.ok(contains(moe,'combine','moeout'),'MoE(x) residual must also depend on expert combine');

console.log('Deterministic tensor dimensions, routing branches and 3D edge contracts PASS');
