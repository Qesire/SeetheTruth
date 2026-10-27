"use strict";
/* Contract: A/B panels project real unified execution, with no second fake arithmetic engine. */
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const path=require("node:path");
const root=path.resolve(__dirname,"..");
const html=fs.readFileSync(path.join(root,"semantic_state.html"),"utf8");
const source=fs.readFileSync(path.join(root,"semantic_state_lab.js"),"utf8");
for(const id of ["aTitle","bTitle","aValues","bValues","difference","lessonLink","lessonReason","trace"])
 assert.match(html,new RegExp('id="'+id+'"'),"missing accessible experiment surface "+id);
assert.match(html,/prefers-reduced-motion/);
assert.match(html,/aria-live="polite"/);
const values={
 stream:{advanced:{next:103,values:[1,2,3,4,5,6]},fork:{next:103,values:[1,2,3,4,50,60]}},
 video:{firstFrame:[1,2],lastFrame:[17,18],shape:[1,3,2,2,2],joined:Array.from({length:24},(_,i)=>i+1),exactEqual:true},
 axis:{axes:["T","D"],shape:[2,3],values:[11,22,33,14,25,36],wrongAxisRejected:true},
 reject:{origin:4,nextPosition:6,original:[1,2,3,4],attemptedPosition:5,error:"out-of-order/stale append position"}
};
const elements=new Map(),buttons=["stream","video","axis","reject"].map(name=>({
 dataset:{scenario:name},pressed:null,handlers:{},
 setAttribute(key,v){if(key==="aria-pressed")this.pressed=v;},
 addEventListener(event,callback){this.handlers[event]=callback;}
}));
const get=id=>{if(!elements.has(id))elements.set(id,{textContent:"",href:""});return elements.get(id);};
const calls=[];
const window={
 location:{search:""},
 SeetheTruthSemanticExamples:{names:Object.keys(values)},
 SeetheTruthUnifiedCore:{executeLab(lab,action,args){
  calls.push([lab,action,args.name]);
  assert.equal(lab,"semantic_state");assert.equal(action,"scenario");
  return {value:{title:args.name,explanation:"mechanism",prediction:"condition",values:values[args.name]}};
 }}
};
const document={getElementById:get,querySelectorAll:selector=>selector==="[data-scenario]"?buttons:[]};
vm.runInNewContext(source,{window,document,Error,Object,JSON,String,URLSearchParams});
for(const name of Object.keys(values)){
 const result=window.SeetheTruthStateLab.run(name);
 assert.equal(result.title,name);
 assert.equal(buttons.find(b=>b.dataset.scenario===name).pressed,"true");
 assert.ok(get("aValues").textContent.length);
 assert.ok(get("bValues").textContent.length);
 assert.match(get("lessonLink").href,/learn\.html/);
 assert.ok(get("lessonReason").textContent.length);
}
assert.equal(calls.length,5);
assert.equal(window.SeetheTruthStateLab.comparison("video",{values:values.video}).b[2].exactEqual,true);
console.log("State A/B learning bridge DOM and canonical-execution contract passed");

const handoffSource=fs.readFileSync(path.join(root,"state_learning_handoff.js"),"utf8");
const selected=[];
const hrefs=[];
function element(tag){return {tag,style:{},children:[],dataset:{},handlers:{},textContent:"",href:"",setAttribute(){},addEventListener(k,v){this.handlers[k]=v;},append(...xs){this.children.push(...xs);},remove(){this.removed=true;}};}
const panelNodes=[];
const routeDocument={
 createElement:element,
 body:{append(node){panelNodes.push(node);}},
 getElementById:id=>id==="stateGraphExample"?{click(){selected.push("stream_state_branches");}}:null,
 querySelector:selector=>selector==='[data-preset="video_causal_conv3d"]'?{click(){selected.push("video_causal_conv3d");}}:null
};
for(const scenario of ["stream","video","axis","reject"]){
 const expected=scenario==="video"?"video_causal_conv3d":"stream_state_branches";
 const kind=scenario==="video"?"video_causal_conv3d":scenario==="reject"?"stream_state_append":scenario==="axis"?"stream_state_read":"stream_state_fork";
 const focused=[];
 const studio={inspect(){return {graph:{nodes:[{id:"target",kind}]}};},focusNode:id=>focused.push(id)};
 const w={location:{search:"?state_case="+scenario},SeetheTruthUnifiedWorkbench:studio};
 vm.runInNewContext(handoffSource,{window:w,document:routeDocument,URLSearchParams,Object,Error});
 assert.equal(selected.at(-1),expected);
 assert.deepEqual(focused,["target"]);
 assert.equal(panelNodes.at(-1).id,"stateHandoff");
}
console.log("State-to-graph handoff selection and focus contracts passed");
