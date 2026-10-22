/* Shared read-only event/state microscope UI. Never inserts untrusted HTML. */
(function(root,factory){'use strict';const api=factory(root);if(typeof module==='object'&&module.exports)module.exports=api;root.SeetheTruthStateMicroscopeUI=api;})(typeof window!=='undefined'?window:{},function(root){'use strict';
const Engine=typeof module==='object'&&module.exports?require('./state_microscope.js'):root.SeetheTruthStateMicroscope;
const el=(tag,txt,cls)=>{const x=document.createElement(tag);if(txt!==undefined)x.textContent=String(txt);if(cls)x.className=cls;return x;};
function mount(container){if(!container||!container.replaceChildren)throw Error('state microscope mount needs container');
 let frame=null,selected=0;
 function view(){
  container.replaceChildren();if(!frame){container.append(el('p','运行状态系统后在这里查看每一步的前置状态、变化和因果依赖。','sm-hint'));return;}
  const step=frame.steps[selected],info=el('div',undefined,'sm-meta');info.append(el('b',frame.title),el('small',frame.evidenceTier));container.append(info);
  const nav=el('div',undefined,'sm-nav');nav.setAttribute('aria-label','通用事件状态转换次序');
  for(let i=0;i<frame.steps.length;i++){const e=frame.steps[i],b=el('button',(i+1)+'. '+e.title,'sm-step');b.type='button';if(i===selected)b.setAttribute('aria-current','step');b.addEventListener('click',()=>{selected=i;view()});nav.append(b);}
  container.append(nav);
  container.append(el('h3',step.title),el('p',step.why,'sm-hint'));
  if(step.timing)container.append(el('p','假设时序：'+step.timing.resource+' · '+step.timing.start+' → '+step.timing.end+'（非实测）','sm-hint'));
  const deps=el('p','实际前置依赖：'+(step.dependencies.join('、')||'无'),'sm-hint');container.append(deps);
  const changes=el('div',undefined,'sm-changes');changes.append(el('b','逐项状态变化 / BEFORE → AFTER'));
  if(!step.changes.length)changes.append(el('p','此步只初始化或验证状态，暂无变化。','sm-hint'));
  for(const x of step.changes.slice(0,20)){
   const line=el('div',undefined,'sm-row');line.append(el('strong',x.key),el('pre',JSON.stringify(x.before)),el('span','→'),el('pre',JSON.stringify(x.after)));changes.append(line);
  }
  container.append(changes);
  const detail=el('details');detail.append(el('summary','完整的本步前后状态（只读）'),el('pre',JSON.stringify({before:step.before,after:step.after},null,2)));container.append(detail);
  container.append(el('p','EVIDENCE · '+frame.disclaimer,'sm-warning'));
 }
 function show(f,index=0){Engine.validate(f);frame=f;selected=Math.max(0,Math.min(f.steps.length-1,index));view();}
 view();return Object.freeze({show,clear(){frame=null;selected=0;view();},get frame(){return frame;},get selected(){return selected;}});
}
return Object.freeze({mount});
});