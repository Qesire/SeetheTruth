const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const content=fs.readFileSync("tensor_subgraph.js","utf8");
const sandbox={window:{}};
vm.runInNewContext(content,sandbox,{timeout:1000});
const module=sandbox.window.SeetheTruthTensor;
assert.ok(module);
function scene(kind,focus,mode="prefill"){
 const st={cfg:{kind,h:768,f:3072,a:12,kv:kind==="gqa"?3:12,b:1,s:64},focus,mode};
 const palette={input:"#a",norm:"#b",cond:"#c",ffn:"#d",attn:"#e",head:"#f",gate:"#9",output:"#8",router:"#7",residual:"#6"};
 const node=(id,label,x,y,z,w,h,d,color,tag)=>({id,label,x,y,z,w,h,d,color,tag});
 const edge=(a,b)=>({a,b});
 const graph=module.build({st,palette,node,edge});
 const ids=new Set(graph.nodes.map(x=>x.id));
 assert.ok(ids.size===graph.nodes.length);
 assert.ok(graph.edges.every(e=>ids.has(e.a)&&ids.has(e.b)));
 return {ids,graph,st};
}
function has(x,id){return x.ids.has("tx_"+id)}
const gqa=scene("gqa","attention");
for(const op of ["q","k","v","qview","qperm","qrope","krope","kcache","vcache","kexpand","vexpand","kt","qk","mask","softmax","av","layout","contiguous","merge","wo"])assert.ok(has(gqa,op),op);
const gpt=scene("gpt","attention");
assert.ok(!has(gpt,"qrope")&&!has(gpt,"krope"));
assert.ok(has(gpt,"kcache"));
const dit=scene("dit","attention");
assert.ok(!has(dit,"qrope")&&!has(dit,"kcache")&&!has(dit,"kexpand"));
assert.ok(has(dit,"qk")&&has(dit,"av"));
const moe=scene("moe","ffn");
for(const op of ["routermm","topk","dispatch","e1g","e1u","e1down","e2g","e2u","e2down","combine"])assert.ok(has(moe,op),op);
const ffn=scene("gqa","ffn");
for(const op of ["gmm","umm","silu","mul","down"])assert.ok(has(ffn,op));
const mlp=scene("gpt","ffn");
assert.ok(has(mlp,"w1")&&has(mlp,"gelu")&&has(mlp,"w2")&&!has(mlp,"gmm"));
const example=module.detail("tx_qview",gqa.st);
assert.equal(example.out,"[1,64,12,64]");
console.log("Tensor micrographs: GQA, GPT, DiT, MoE, SwiGLU and GELU PASS");
