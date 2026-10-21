/* Universal, bounded EVENT/STATE microscope. Provenance is always a scoped
 * simulation contract; events and transitions are obtained from validated
 * engines, never from a fabricated tensor/GPU telemetry stream. */
(function(root,make){'use strict';const api=make();if(typeof module==='object'&&module.exports)module.exports=api;root.SeetheTruthStateMicroscope=api;})(typeof window!=='undefined'?window:{},function(){'use strict';
const SCHEMA='seethetruth-state-microscope-v1';
const TIERS=Object.freeze({KV:'SOURCE_GROUNDED_KV_CPU_STATE_SIMULATION',WORKER:'SOURCE_GROUNDED_WORKER_ASSUMED_DURATIONS'});
const fail=m=>{throw Error('state microscope: '+m);};
const need=(cond,m)=>{if(!cond)fail(m);};
const plain=x=>x!==null&&typeof x==='object'&&!Array.isArray(x);
const finite=v=>typeof v==='number'&&Number.isFinite(v);
function values(x,depth=0){
 need(depth<=5,'state field nesting too deep');
 if(x===null||typeof x==='string'||typeof x==='boolean')return true;
 if(typeof x==='number'){need(finite(x),'nonfinite state value');return true;}
 if(Array.isArray(x)){need(x.length<=160,'oversized state array');for(const v of x)values(v,depth+1);return true;}
 need(plain(x)&&Object.keys(x).length<=48,'invalid state structure');
 for(const [k,v]of Object.entries(x)){need(/^[A-Za-z0-9_\-]{1,72}$/.test(k),'invalid state key');values(v,depth+1);}
 return true;
}
function delta(a,b){
 const keys=[...new Set([...Object.keys(a),...Object.keys(b)])].sort();
 return keys.filter(k=>JSON.stringify(a[k])!==JSON.stringify(b[k])).map(key=>({key,before:Object.hasOwn(a,key)?a[key]:null,after:Object.hasOwn(b,key)?b[key]:null}));
}
function validate(frame){
 need(frame?.schema===SCHEMA,'wrong schema');
 need(Object.values(TIERS).includes(frame.evidenceTier),'unknown evidence tier');
 need(typeof frame.title==='string'&&frame.title.length>0&&frame.title.length<140,'invalid title');
 need(Array.isArray(frame.steps)&&frame.steps.length>0&&frame.steps.length<=250,'invalid steps');
 need(typeof frame.disclaimer==='string'&&frame.disclaimer.length>0,'missing epistemic boundary');
 let prev=-1;const ids=new Set();
 for(const [index,e]of frame.steps.entries()){
  need(e&&typeof e.id==='string'&&/^[A-Za-z0-9_:-]{1,80}$/.test(e.id)&&!ids.has(e.id),'invalid/duplicate event ID');
  need(e.index===index&&e.index>prev,'steps out of order');
  need(typeof e.title==='string'&&e.title.length>0&&typeof e.why==='string'&&e.why.length>0,'missing event explanation');
  need(plain(e.before)&&plain(e.after)&&Array.isArray(e.changes),'state before/after missing');
  values(e.before);values(e.after);
  need(JSON.stringify(e.changes)===JSON.stringify(delta(e.before,e.after)),'inconsistent state changes');
  need(Array.isArray(e.dependencies)&&e.dependencies.every(id=>typeof id==='string'&&ids.has(id)),'invalid causal dependencies');
  if(e.timing){need(finite(e.timing.start)&&finite(e.timing.end)&&e.timing.start>=0&&e.timing.end>=e.timing.start,'invalid simulated timestamp');}
  ids.add(e.id);prev=e.index;
 }
 return frame;
}
function kv(replay,KV){
 need(replay&&Array.isArray(replay.timeline)&&replay.timeline.length>=2&&replay.timeline.length<=81&&KV&&typeof KV.verify==='function','invalid KV replay');
 const steps=[];for(let i=0;i<replay.timeline.length;i++){
  const s=replay.timeline[i];KV.verify(s);
  const b=i?replay.timeline[i-1]:s;
  need(i===0||s.revision===b.revision+1,'KV revisions must advance by one');
  const op=s.events.at(-1);
  const project=x=>({revision:x.revision,freeQueue:x.free.slice(),refs:x.refs.slice(),
   owners:Object.fromEntries(Object.entries(x.requests).map(([id,v])=>[id,{status:v.status,blocks:v.blocks.slice(),cachedTokens:v.cachedTokens,recomputes:v.recomputes}]))});
  const before=project(b),after=project(s);
  steps.push({id:'kv:'+i,index:i,title:i?'操作 '+op?.op+' · '+(op?.id??''):'初始化 Block Pool',
   why:i?'每次操作都由原引擎校验物理块引用计数和 FIFO 空闲列表，展示的是 CPU 参考逻辑，不是设备显存实际分配。':'物理块 0 永远保留；其它块按 FIFO 维护引用计数。',
   dependencies:i?['kv:'+(i-1)]:[],before,after,changes:delta(before,after),timing:null});
 }
 return validate({schema:SCHEMA,title:'KV BlockManager · 状态转换的通用显微镜',evidenceTier:TIERS.KV,steps,
   disclaimer:'Verified small CPU block/refcount transitions, NOT GPU blocks, runtime Scheduler or measured time'});
}
function worker(run){
 need(run&&Array.isArray(run.events)&&run.events.length>=1&&run.events.length<=250,'invalid worker event run');
 const completed=new Map(),seen=new Set(),steps=[],busy={CPU:0,GPU:0};
 for(let i=0;i<run.events.length;i++){
  const e=run.events[i];
  need(e&&(e.resource==='CPU'||e.resource==='GPU')&&typeof e.stage==='string'&&Number.isSafeInteger(e.step),'invalid worker event');
  need(finite(e.start)&&finite(e.end)&&finite(e.duration)&&e.start>=0&&e.end>=e.start&&Math.abs(e.end-e.start-e.duration)<1e-9,'invalid assumed event timing');
  need(!seen.has(e.id),'duplicate worker event');
  need(Array.isArray(e.dependsOn)&&e.dependsOn.every(dep=>completed.has(dep)&&completed.get(dep)<=e.start+1e-9),'worker dependency order violated');
  need(e.start+1e-9>=busy[e.resource],'same resource overlaps');
  const before={completedEvents:i,resourceBusy:{...busy}},after={completedEvents:i+1,resourceBusy:{...busy,[e.resource]:e.end}};
  steps.push({id:e.id,index:i,title:'Step '+e.step+' · '+e.stage,
    why:'演示 CPU/GPU 资源排队和跨步因果依赖。Duration 来自可修改的假设输入，不能解释为实测硬件延迟。',
    dependencies:e.dependsOn.slice(),before,after,changes:delta(before,after),timing:{start:e.start,end:e.end,resource:e.resource}});
  busy[e.resource]=e.end;seen.add(e.id);completed.set(e.id,e.end);
 }
 return validate({schema:SCHEMA,title:'DeepInfer Worker · 跨 Step 假设时序',evidenceTier:TIERS.WORKER,steps,
   disclaimer:'Durations are assumed logical units, NOT profiler data, GPU utilization or production throughput'});
}
return Object.freeze({SCHEMA,TIERS,validate,kv,worker,delta});
});