"use strict";
/* Real source graph + math contract for state ↔ A/B ↔ guided-block navigation. */
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const vm=require("node:vm");
const ROOT=path.resolve(__dirname,"..");
const Lab=require("../learning_lab.js");
const Examples=require("../semantic_state_examples.js");
const Bridge=require("../state_learning_handoff.js");
const stateHtml=fs.readFileSync(path.join(ROOT,"semantic_state.html"),"utf8");
const studioHtml=fs.readFileSync(path.join(ROOT,"learn.html"),"utf8");
const stateSource=fs.readFileSync(path.join(ROOT,"semantic_state_lab.js"),"utf8");
const handoffSource=fs.readFileSync(path.join(ROOT,"state_learning_handoff.js"),"utf8");
const policy=JSON.parse(fs.readFileSync(path.join(ROOT,"static_asset_policy.json"),"utf8"));
const names=["stream","video","axis","reject"];
for(const id of ["aTitle","bTitle","aValues","bValues","aGraphLink","bGraphLink","difference","lessonLink","trace"])
 assert.match(stateHtml,new RegExp('id="'+id+'"'),"missing experiment UI "+id);
for(const html of [stateHtml,studioHtml])
 assert.ok(html.includes('<script src="state_learning_handoff.js"></script>'),"missing navigation bootstrap");
assert.ok(policy.androidAssets.includes("state_learning_handoff.js"),"shared offline asset must be reviewed");
assert.match(stateHtml,/prefers-reduced-motion/);
assert.match(studioHtml,/state-handoff-anchor/);
const expectedPresets={
 stream:"stream_state_branches",video:"extension_video_time_composition",
 axis:"extension_named_axis_composition",reject:"stream_state_branches"
};
for(const name of names){
 const spec=Bridge.caseOf(name);
 assert.equal(spec.preset,expectedPresets[name]);
 const graph=Lab.preset(spec.preset);
 Bridge.validateGraph(name,graph);
 const result=Examples.run(name);assert.equal(result.passed,true);
 const proofs=Bridge.evidence(name,result);
 assert.equal(proofs.length,spec.anchors.length);
 for(const point of proofs){
  const node=graph.nodes.find(n=>n.id===point.id);
  assert.equal(node.kind,point.kind);
  assert.ok(Object.prototype.hasOwnProperty.call(point,"sample"));
  assert.match(Bridge.href(name,point.id),new RegExp("state_case="+name+"&state_node="+point.id+"$"));
 }
 const tampered={...graph,nodes:graph.nodes.map(n=>({...n}))};
 tampered.nodes.find(n=>n.id===spec.anchors[0].id).kind="input";
 assert.throws(()=>Bridge.validateGraph(name,tampered),/签名/);
 const broken={...graph,edges:graph.edges.filter(e=>!(e.from===spec.edges[0][0]&&e.to===spec.edges[0][1]&&e.port===spec.edges[0][2]))};
 assert.throws(()=>Bridge.validateGraph(name,broken),/依赖/);
}
assert.equal(Bridge.caseOf("malicious"),null);
assert.throws(()=>Bridge.href("stream","not-a-node"),/未知/);
function domElement(tag){
 return {tag,id:"",href:"",className:"",style:{},children:[],handlers:{},attributes:{},
  textContent:"",dataset:{},
  setAttribute(k,v){this.attributes[k]=String(v);},
  addEventListener(k,v){this.handlers[k]=v;},
  append(...parts){this.children.push(...parts);},
  click(){if(this.handlers.click)this.handlers.click();},
  remove(){this.removed=true;}};
}
function evaluateStatePage(search){
 const elements=new Map(),buttons=names.map(name=>({dataset:{scenario:name},attributes:{},
  setAttribute(k,v){this.attributes[k]=String(v);},addEventListener(){}}));
 const get=id=>{if(!elements.has(id))elements.set(id,{textContent:"",href:""});return elements.get(id);};
 const core={executeLab(domain,op,input){
  assert.equal(domain,"semantic_state");assert.equal(op,"scenario");
  return {value:Examples.run(input.name)};
 }};
 const window={location:{search},SeetheTruthStateBridge:Bridge,
  SeetheTruthUnifiedCore:core,SeetheTruthSemanticExamples:{names}};
 const document={getElementById:get,querySelectorAll:selector=>selector==="[data-scenario]"?buttons:[]};
 vm.runInNewContext(stateSource,{window,document,URLSearchParams,JSON,Object,String,Error});
 return {window,get,buttons};
}
for(const name of names){
 const {window,get}=evaluateStatePage("?state_case="+name);
 assert.equal(get("caseTitle").textContent,Examples.run(name).title);
 assert.ok(get("trace").textContent.length>20);
 assert.ok(get("aValues").textContent.length);
 assert.ok(get("bValues").textContent.length);
 const spec=Bridge.caseOf(name);
 assert.equal(get("aGraphLink").href,Bridge.href(name,spec.primary.a));
 assert.equal(get("bGraphLink").href,Bridge.href(name,spec.primary.b));
 assert.match(get("lessonLink").href,/state_case=/);
 assert.equal(window.SeetheTruthStateLab.run(name).passed,true);
}
function evaluateHandoff(name,node){
 const selection=[],focus=[],panels=[],studio={};
 const spec=Bridge.caseOf(name),graph=Lab.preset(spec.preset);
 studio.inspect=()=>({graph});
 studio.focusNode=id=>{focus.push(id);return id;};
 const document={
  createElement:domElement,body:{append:el=>panels.push(el)},
  getElementById:id=>id==="stateGraphExample"?{click:()=>selection.push("stream_state_branches")}:null,
  querySelector:selector=>selector==='[data-preset="'+spec.preset+'"]'
   ?{click:()=>selection.push(spec.preset)}:null
 };
 const core={executeLab(domain,op,input){
  assert.equal(domain,"semantic_state");assert.equal(op,"scenario");return {value:Examples.run(input.name)};
 },replay:accepted=>{assert.equal(accepted.value.passed,true);}};
 const window={location:{search:"?state_case="+name+"&state_node="+node},
  SeetheTruthUnifiedCore:core,SeetheTruthUnifiedWorkbench:studio};
 vm.runInNewContext(handoffSource,{window,document,URLSearchParams,JSON,Object,String,Error});
 const panel=panels.at(-1);
 assert.ok(panel&&panel.id==="stateHandoff");
 assert.equal(selection.at(-1),spec.preset);
 assert.equal(focus.at(-1),node);
 assert.equal(panel.children[2].textContent.includes("课程关联待检查"),false);
 const links=panel.children[3];
 assert.equal(links.children.length,spec.anchors.length);
 for(let i=0;i<links.children.length;i++){
  links.children[i].click();
  assert.equal(focus.at(-1),spec.anchors[i].id);
 }
 assert.ok(panel.children[4].textContent.includes("独立示例输入"));
 assert.equal(panel.children[6].href,"semantic_state.html?state_case="+name);
}
for(const name of names){
 const spec=Bridge.caseOf(name);
 evaluateHandoff(name,spec.primary.a);
 evaluateHandoff(name,spec.primary.b);
}
console.log("STATE_AB_GRAPH_BRIDGE_PASS: real scenario arithmetic, installed recipe signatures, per-branch focus, provenance and offline publication");
