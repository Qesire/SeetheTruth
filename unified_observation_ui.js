/* Shared readonly observation UI: graph and specialist domains consume one v1 envelope.
 * Never computes alternative numerical values, invents event edges, or upgrades provenance.
 */
(function(root,factory){'use strict';const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;if(root)root.SeetheTruthUnifiedObservationUI=api;})(typeof window!=='undefined'?window:null,function(){'use strict';
const tag=(name,value)=>{const n=document.createElement(name);if(value!==undefined)n.textContent=String(value);return n;};
const freezeClone=value=>JSON.parse(JSON.stringify(value));
function mount(options={}){
 const core=options.core,readAccepted=options.readAccepted,snapshots=options.snapshots;
 if(!core||typeof core.observe!=='function'||typeof core.replay!=='function'||
    typeof core.inspectObservationStep!=='function'||typeof core.compareRecords!=='function'||
    typeof readAccepted!=='function'||!snapshots||typeof snapshots.create!=='function'||typeof snapshots.restore!=='function')throw Error('unified observation needs core, snapshot codec and accepted-record provider');
 const host=options.host||document.createElement('section');
 if(!options.host)document.body.append(host);
 host.id=host.id||'unifiedObservationPanel';
 host.setAttribute('aria-label','统一计算与状态观测协议');
 host.style.cssText='max-width:1180px;margin:20px auto 36px;padding:16px;border:1px solid #426279;border-radius:14px;background:#102438;color:#e9f5ff;font:13px/1.6 system-ui,sans-serif;overflow-wrap:anywhere';
 const h=tag('h2',options.title||'统一观测 / Observation v1');h.style.margin='0 0 8px';
 const hint=tag('p','真实计算仍由原引擎执行；本面板只核验、选择和比较原生轨迹。回放相等不是实际 GPU 或模型权重证明。');
 const tools=tag('div');tools.style.cssText='display:flex;gap:8px;flex-wrap:wrap';
 const button=(id,label)=>{const b=tag('button',label);b.id=id;b.type='button';b.style.cssText='padding:8px 12px;border:1px solid #5e869a;background:#173e53;color:#f1faff;border-radius:7px;cursor:pointer';tools.append(b);return b;};
 const reload=button('unifiedObservationRefresh','核验当前执行'),pin=button('unifiedObservationPin','固定此执行为基线'),diff=button('unifiedObservationCompare','与基线比较'),clear=button('unifiedObservationClear','清除基线');
 const download=button('unifiedObservationExport','导出已核验快照');
 const branchButton=button('unifiedObservationFork','将当前实验另存为分支');
 const checkpointButton=button('unifiedObservationCheckpoint','导出当前状态步骤');
 const importer=tag('input');importer.id='unifiedObservationImport';importer.type='file';
 importer.accept='.json,application/json';importer.setAttribute('aria-label','导入已保存的观测快照');
 importer.style.cssText='max-width:240px;color:#d5eef6;font-size:12px';
 const sourceLink=tag('a','前往原实验室验证快照');
 sourceLink.id='unifiedObservationOpenSource';sourceLink.hidden=true;
 sourceLink.style.cssText='color:#a5e9d9;text-decoration:underline';
 const status=tag('p','尚未核验执行记录');status.id='unifiedObservationStatus';status.setAttribute('aria-live','polite');
 const menu=tag('div');menu.style.cssText='display:flex;flex-wrap:wrap;gap:10px;align-items:center';
 const source=tag('select');source.id='unifiedObservationSource';const step=tag('select');step.id='unifiedObservationStep';
 for(const select of [source,step])select.style.cssText='background:#14344a;color:#effaff;border:1px solid #66879a;padding:7px;border-radius:6px;max-width:100%';
 menu.append(tag('label','轨迹类型'),source,tag('label','实际步骤'),step);
 const coords=tag('div');coords.id='unifiedObservationCoordinates';coords.style.cssText='display:flex;gap:7px;flex-wrap:wrap;margin:8px 0';
 const check=button('unifiedObservationInspect','检验选中步骤');
 const detail=tag('pre');detail.id='unifiedObservationDetail';
 detail.style.cssText='white-space:pre-wrap;word-break:break-word;padding:12px;border:1px solid #36566a;border-radius:8px;max-height:440px;overflow:auto';
 const comparison=tag('pre');comparison.id='unifiedObservationDiff';
 comparison.style.cssText=detail.style.cssText;
 host.replaceChildren(h,hint,tools,importer,sourceLink,status,menu,coords,check,detail,comparison);
 let current=null,inventory=null,baseline=null,foreignSnapshot=null,sourceIndex=0,stepIndex=0;
 const HANDOFF_KEY='SeetheTruthObservationHandoffV1';
 function setStatus(message,failed=false){status.textContent=message;status.style.color=failed?'#ffb1a7':'#a5e9d9';}
 function chosen(){
  const group=inventory?.sources[sourceIndex],point=group?.steps[stepIndex];
  return group&&point?{group,point}:null;
 }
 const item=(value,title)=>{const opt=tag('option',title);opt.value=String(value);return opt;};
 function renderAxes(){
  coords.replaceChildren();
  const selected=chosen();
  if(!selected||selected.group.kind!=='tensor_computation')return;
  const point=selected.point;
  if(!point.hasNumericValues){coords.append(tag('span','该节点只有静态形状，不提供数值坐标。'));return;}
  for(let i=0;i<point.shape.length;i++){
   const label=tag('label','轴 '+i+' · [0,'+(point.shape[i]-1)+'] '),box=tag('input');
   box.type='number';box.min='0';box.max=String(point.shape[i]-1);box.step='1';box.value='0';
   box.style.cssText='width:65px;background:#14344a;color:#fff;border:1px solid #66879a';
   box.setAttribute('data-observation-axis',String(i));label.append(box);coords.append(label);
  }
 }
 function renderSources(){
  source.replaceChildren();step.replaceChildren();coords.replaceChildren();
  if(!inventory?.sources.length){menu.hidden=true;check.disabled=true;detail.textContent='无已执行原生轨迹；静态项目或分析结果不会被伪装成数值执行。';return;}
  menu.hidden=false;check.disabled=false;
  sourceIndex=Math.min(sourceIndex,inventory.sources.length-1);
  inventory.sources.forEach((s,i)=>source.append(item(i,s.kind+' · '+s.evidence)));
  source.value=String(sourceIndex);
  renderSteps();
 }
 function renderSteps(){
  step.replaceChildren();const group=inventory.sources[sourceIndex];
  stepIndex=Math.min(stepIndex,group.steps.length-1);
  group.steps.forEach((s,i)=>step.append(item(i,(i+1)+'. '+s.id)));
  step.value=String(stepIndex);renderAxes();detail.textContent='选择步骤后执行坐标/状态核验。';
 }
 function refresh(){
  const record=readAccepted();if(!record)throw Error('当前页面没有已接受的执行记录');
  core.replay(record);
  const frame=core.observe(record);
  current=freezeClone(record);inventory=frame;sourceIndex=0;stepIndex=0;
  renderSources();comparison.textContent='';
  setStatus('已核验 '+(record.id||'execution')+' · '+frame.evidence+' · '+frame.sources.reduce((n,s)=>n+s.steps.length,0)+' 个原生步骤'+(baseline?' · 已固定基线':''));
  return frame;
 }
 function inspect(){
  if(!current||!inventory)throw Error('请先核验当前执行');
  const p=chosen();if(!p)throw Error('当前执行没有可检索步骤');
  if(p.group.kind==='tensor_computation'&&!p.point.hasNumericValues)
   throw Error('该步骤没有数值观测');
  const axes=[...coords.querySelectorAll('[data-observation-axis]')];
  const coordinate=p.group.kind==='tensor_computation'?axes.map(a=>Number(a.value)):undefined;
  const checked=core.inspectObservationStep(current,{kind:p.group.kind,stepId:p.point.id,
   ...(coordinate?{coordinate}:{})});
  const text=JSON.stringify(checked,null,2);
  detail.textContent=text.length<=15000?text:text.slice(0,15000)+'\n… 仅截断显示，原始执行结果未修改。';
  return checked;
 }
 function pinBaseline(){
  if(!current)refresh();core.replay(current);baseline=freezeClone(current);
  comparison.textContent='基线已固定；修改当前实验并重新执行后，可比较真实步骤差异。';
  setStatus('固定基线：'+(baseline.id||'execution')+' · '+baseline.evidence);
 }
 function compare(){
  if(!baseline)throw Error('尚未固定基线');
  const record=readAccepted();if(!record)throw Error('当前页面没有可比较执行');
  const result=core.compareRecords(baseline,record);
  comparison.textContent=JSON.stringify({
   replayVerified:result.replayVerified,inputEqual:result.inputEqual,valueEqual:result.valueEqual,
   numericCompared:result.numericCompared??null,evidence:result.evidence??result.evidencePair,
   changedStepCount:result.changedSteps.length,changedSteps:result.changedSteps.slice(0,80)
  },null,2);
  return result;
 }

 function createPortable(){
  if(!baseline)pinBaseline();
  const portable=snapshots.create(core,baseline);
  return portable;
 }
 function downloadFile(portable,suffix=""){
  const body=JSON.stringify(portable,null,2),blob=new Blob([body],{type:'application/json'});
  const url=URL.createObjectURL(blob),a=tag('a');
  a.href=url;a.download='seethetruth-observation-'+portable.id+suffix+'.json';
  document.body.append(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
 }
 function exportSnapshot(){
  const portable=createPortable();
  downloadFile(portable);
  setStatus('已导出可重放执行快照 · '+portable.evidence);
  return portable;
 }
 function forkSnapshot(){
  if(!baseline)throw Error('请先固定或导入已核验基线');
  const original=snapshots.create(core,baseline);
  const live=readAccepted();if(!live)throw Error('当前页面无可执行分支');
  core.replay(live);
  const candidate=snapshots.fork(core,original,
   live.kind==='graph'?live.graph:live.inputSnapshot,'experiment branch');
  const reproduced=snapshots.restore(core,candidate);
  const check=core.compareRecords(live,reproduced.record);
  if(!check.replayVerified||check.inputEqual!==true||
     (check.valueEqual!==true&&check.valueEqual!==null))
   throw Error('分支结果与当前实验不一致');
  downloadFile(candidate,'-branch');
  comparison.textContent='分支快照已导出 · 父快照：'+candidate.parent+' · 原始数值引擎已验证。';
  setStatus('已导出独立可重放实验分支 · '+candidate.evidence);
  return candidate;
 }
 function exportCheckpoint(){
  if(!current||!inventory)refresh();
  const selected=chosen();
  if(!selected||selected.group.kind!=="state_transition")
   throw Error("仅原生状态转换步骤支持状态检查点");
  const portable=snapshots.at(core,snapshots.create(core,current),selected.point.id);
  const confirmed=snapshots.restore(core,portable);
  if(!confirmed.checkpoint?.replayVerified)throw Error("状态步骤未通过原引擎恢复");
  downloadFile(portable,'-checkpoint');
  setStatus('已导出只读状态步骤 · '+selected.point.id+' · '+confirmed.evidence);
  return portable;
 }
 function importSnapshot(input){
  // A file is a replay recipe, never an authenticated execution artifact.
  const parsed=snapshots.validate(input),about=snapshots.describe(core,parsed);
  const local=(location.pathname.split('/').at(-1)||'index.html')===about.sourceHref;
  foreignSnapshot=null;sourceLink.hidden=true;
  if(local){
   const restored=snapshots.restore(core,parsed);
   baseline=freezeClone(restored.record);
   if(restored.checkpoint){
    // A verified cursor restores the *educational view*; no claim of
    // in-memory GPU checkpoint, scheduler mutation or external side effects.
    current=freezeClone(restored.record);inventory=restored.observation;
    sourceIndex=inventory.sources.findIndex(x=>x.kind==="state_transition");
    if(sourceIndex<0)throw Error("状态检查点缺少原生状态轨迹");
    stepIndex=restored.checkpoint.index;
    renderSources();
    const shown=inspect();
    if(!shown?.replayVerified)throw Error("恢复的教学状态步骤未经检验");
    if(typeof options.onCheckpoint==="function")
     options.onCheckpoint(restored);
   }
   setStatus('已恢复并重放验证 '+about.id+' · '+about.evidence+
    (restored.checkpoint?' · 步骤 '+restored.checkpoint.stepId:''));
   comparison.textContent=restored.checkpoint
    ?'已回到原生状态步骤 '+restored.checkpoint.stepId+'；仅恢复教学视图，不修改设备或外部系统。'
    :'导入快照已核验，可以与当前实验比较。';
   return restored;
  }
  foreignSnapshot=parsed;
  sourceLink.href=about.sourceHref;sourceLink.hidden=false;
  comparison.textContent='来自 '+about.id+' / '+about.operation+' 的快照已读取，但当前页面未运行其原始引擎。请前往原实验室验证。';
  setStatus('跨页面快照已解析（尚未重放核验） · '+about.evidence);
  return about;
 }
 function sessionStore(){
  try{return window.sessionStorage||null;}catch(_){return null;}
 }
 function restoreHandoff(){
  const store=sessionStore();
  if(!store)return; // Offline WebViews may disable session storage.
  try{
   const text=store.getItem(HANDOFF_KEY);
   if(!text)return;
   const parsed=snapshots.validate(JSON.parse(text));
   const about=snapshots.describe(core,parsed);
   if((location.pathname.split('/').at(-1)||'index.html')!==about.sourceHref)return;
   const restored=importSnapshot(parsed);
   store.removeItem(HANDOFF_KEY);
   return restored;
  }catch(error){
   try{store.removeItem(HANDOFF_KEY);}catch(_){}
   setStatus('跨页面快照恢复被拒绝：'+error.message,true);
  }
 }
 function guard(fn){try{return fn();}catch(error){setStatus('观测拒绝：'+error.message,true);detail.textContent=error.message;return null;}}
 reload.addEventListener('click',()=>guard(refresh));
 check.addEventListener('click',()=>guard(inspect));
 pin.addEventListener('click',()=>guard(pinBaseline));
 diff.addEventListener('click',()=>guard(compare));
 clear.addEventListener('click',()=>{baseline=null;foreignSnapshot=null;sourceLink.hidden=true;comparison.textContent='基线已清除';setStatus('已清除基线');});
 download.addEventListener('click',()=>guard(exportSnapshot));
 branchButton.addEventListener('click',()=>guard(forkSnapshot));
 checkpointButton.addEventListener('click',()=>guard(exportCheckpoint));
 importer.addEventListener('change',async()=>{
  const file=importer.files?.[0];if(!file)return;
  try{if(file.size>snapshots.MAX_BYTES)throw Error('快照文件超过大小限制');
   importSnapshot(JSON.parse(await file.text()));
  }catch(error){setStatus('快照导入被拒绝：'+error.message,true);}
  finally{importer.value='';}
 });
 sourceLink.addEventListener('click',event=>{
  if(!foreignSnapshot){event.preventDefault();return;}
  try{const store=sessionStore();if(!store)throw Error('当前 WebView 不支持会话存储');
   store.setItem(HANDOFF_KEY,JSON.stringify(foreignSnapshot));}
  catch(error){event.preventDefault();setStatus('跨页面交接失败：'+error.message,true);}
 });
 source.addEventListener('change',()=>{sourceIndex=Number(source.value);stepIndex=0;renderSteps();});
 step.addEventListener('change',()=>{stepIndex=Number(step.value);renderAxes();});
 menu.hidden=true;check.disabled=true;restoreHandoff();
 return Object.freeze({refresh,inspect,pin:pinBaseline,compare,exportSnapshot,forkSnapshot,exportCheckpoint,importSnapshot,
  clear:()=>clear.click(),
  get observation(){return inventory;},get baseline(){return baseline;},get current(){return current;},host});
}
return Object.freeze({mount});
});
