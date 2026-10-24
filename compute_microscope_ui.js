/* Reusable read-only computation microscope surface for workbench and model pages.
 * DOM constructed with textContent; imported source labels are not executable HTML. */
(function(root,factory){'use strict';const api=factory(root);if(typeof module==='object'&&module.exports)module.exports=api;root.SeetheTruthMicroscopeUI=api;})(typeof window!=='undefined'?window:{},function(root){'use strict';
const CORE=typeof module==='object'&&module.exports?require('./compute_microscope.js'):root.SeetheTruthComputeMicroscope;
function el(tag,text,cls){const a=document.createElement(tag);if(text!==undefined)a.textContent=String(text);if(cls)a.className=cls;return a;}
function mini(t,label,click,selected){
 const axes=t.axes||null;
 const wrapper=el('div',undefined,'cmp-mini'),heading=el('div',label+' · ['+t.shape.join('×')+']'+(axes?' · '+axes.map((name,i)=>name+'='+t.shape[i]).join(' · '):''),'cmp-label');wrapper.append(heading);
 if(t.values===null){wrapper.append(el('p','静态形状推断：无数值矩阵。','cmp-hint'));return wrapper;}
 const shape=t.shape,rank=shape.length,R=rank===1?1:shape[rank-2],C=shape[rank-1];
 const prefix=Array.from({length:Math.max(0,rank-2)},(_,i)=>selected?.prefix?.[i]??0);
 function updatePrefix(axis,index){
  const next=prefix.slice();next[axis]=index;
  if(click){click(0,0,next);return;}
  const copy=mini(t,label,null,{prefix:next});wrapper.replaceWith(copy);
 }
 if(prefix.length){
  const controls=el('div',undefined,'cmp-axes');
  for(let axis=0;axis<prefix.length;axis++){
   const label=el('label',(axes?axes[axis]:'Axis '+axis)+' · ','cmp-axis'),sel=el('select');sel.setAttribute('aria-label',label.textContent+' index');
   for(let index=0;index<shape[axis];index++){const option=el('option',index);option.value=index;sel.append(option);}
   sel.value=String(prefix[axis]);sel.addEventListener('change',()=>updatePrefix(axis,Number(sel.value)));label.append(sel);controls.append(label);
  }
  wrapper.append(controls);
 }
 let offset=0;for(let i=0;i<prefix.length;i++)offset=offset*shape[i]+prefix[i];
 offset*=R*C;
 const g=el('div',undefined,'cmp-matrix'),peak=Math.max(1e-12,...t.values.map(Math.abs));
 g.style.gridTemplateColumns='repeat('+Math.min(C,24)+',minmax(37px,1fr))';
 for(let i=0;i<Math.min(R,20);i++)for(let j=0;j<Math.min(C,24);j++){
  const v=t.values[offset+i*C+j],b=el('button',Number(v).toPrecision(3),'cmp-cell');
  b.type='button';b.style.background='rgba('+(v>=0?'41,145,134,':'178,108,92,')+(.17+.69*Math.abs(v)/peak)+')';
  const coords=[...prefix,...(rank===1?[]:[i]),j],coordLabel='['+coords.map((value,axis)=>(axes?axes[axis]+'=':'')+value).join(',')+']';
  b.title=coordLabel+' '+v.toPrecision(15);
  b.setAttribute('aria-label',label+' index '+coordLabel+' = '+v.toPrecision(9));
  if(selected&&selected.row===i&&selected.col===j)b.classList.add('cmp-selected');
  if(click)b.addEventListener('click',()=>click(i,j,prefix));
  else b.disabled=true;
  g.append(b);
 }
 wrapper.append(g);
 if(R>20||C>24)wrapper.append(el('p','仅显示前 '+Math.min(R,20)+' 行、'+Math.min(C,24)+' 列；计算仍使用完整的有界张量。','cmp-hint'));
 return wrapper;
}
function mount(rootElement,onSelect){
 if(!rootElement||typeof rootElement.replaceChildren!=='function')throw Error('microscope UI mount required');
 let frame=null,active=null,subSelected=null,row=0,col=0,prefix=[];
 function clear(msg='运行数值图或进行静态形状分析后，选择积木即可展开；当前没有执行证据。'){
  frame=null;active=null;rootElement.replaceChildren(el('p',msg,'cmp-hint'));
 }
 function present(){
  if(!frame)return clear();
  const parent=frame.steps.find(x=>x.id===active)||frame.steps[0];
  const s=parent.substeps?.find(x=>x.id===subSelected)||parent;
  rootElement.replaceChildren();
  const summary=el('div',undefined,'cmp-top');summary.append(el('div',frame.evidenceTier,'cmp-tier'),el('div',frame.title,'cmp-title'));
  rootElement.append(summary);
  const nav=el('div',undefined,'cmp-steps');nav.setAttribute('role','group');nav.setAttribute('aria-label','已验证执行顺序');
  for(let i=0;i<frame.steps.length;i++){
   const item=frame.steps[i],b=el('button',(i+1)+'. '+item.title,'cmp-step');
   b.type='button';if(item.id===parent.id)b.setAttribute('aria-current','step');
   b.title=item.kind+' · '+item.out.shape.join('×');
   b.addEventListener('click',()=>{active=item.id;subSelected=null;row=0;col=0;prefix=[];present();onSelect?.(item.id);});nav.append(b);
  }
  rootElement.append(nav);
  if(parent.substeps?.length){
   const stages=el('div',undefined,'cmp-substeps');stages.append(el('div','COMPOSITE OPERATOR · 点击查看内部真实计算顺序','cmp-label'));
   const steps=el('div',undefined,'cmp-sublist');
   const back=el('button','聚合模块输出','cmp-substep');if(!subSelected)back.setAttribute('aria-current','step');
   back.addEventListener('click',()=>{subSelected=null;row=0;col=0;prefix=[];present();});steps.append(back);
   for(const item of parent.substeps){
    const button=el('button',item.title,'cmp-substep');button.type='button';
    if(subSelected===item.id)button.setAttribute('aria-current','step');
    button.addEventListener('click',()=>{subSelected=item.id;row=0;col=0;prefix=[];present();});steps.append(button);
   }
   stages.append(steps);rootElement.append(stages);
  }
  const meta=el('div',undefined,'cmp-meta');
  meta.append(el('strong',s.title+' · '+s.kind),el('span','Output '+s.out.shape.join(' × ')+(s.out.axes?' · '+s.out.axes.join(' / '):'')));
  rootElement.append(meta);
  const operands=el('div',undefined,'cmp-deps');
  operands.append(el('div','实际依赖 / DATAFLOW EDGES','cmp-label'));
  const activeDependencies=s.dependencies||[{from:parent.id,port:'inside'}];
  if(activeDependencies.length)for(const dep of activeDependencies)operands.append(el('span',dep.from+' → '+dep.port,'cmp-chip'));
  else operands.append(el('span','起始阶段 / 无上游积木','cmp-chip'));
  rootElement.append(operands);
  const math=el('pre',s.formula,'cmp-formula');rootElement.append(math);
  rootElement.append(el('p',s.why,'cmp-hint'));
  if(frame.source?.url&&s.sourceLines){
   const link=el('a','查看固定源码 '+frame.source.path+':'+s.sourceLines.start,'cmp-link');
   link.href=frame.source.url+'#L'+s.sourceLines.start+'-L'+s.sourceLines.end;link.target='_blank';link.rel='noopener noreferrer';rootElement.append(link);
  }
  const output=el('div',undefined,'cmp-out');
  output.append(mini(s.out,'输出张量 / '+s.id,(r,c,p)=>{row=r;col=c;prefix=p||[];present();}, {row,col,prefix}));
  rootElement.append(output);
  if(s.out.values!==null){
   const detail=CORE.dot(frame,s.id,Math.min(row,s.out.shape.length===1?0:s.out.shape[s.out.shape.length-2]-1),Math.min(col,s.out.shape[s.out.shape.length-1]-1),prefix);
   const box=el('div',undefined,'cmp-breakdown');
   box.append(el('div','坐标 '+(detail.coordinateLabel||'['+detail.row+','+detail.col+']')+' = '+detail.value.toPrecision(12),'cmp-label'));
   box.append(el('pre',detail.formula,'cmp-formula'));
   if(detail.reconstructable){
    box.append(el('p','逐项重建值 '+detail.recomputed.toPrecision(12)+' · 误差 '+detail.error.toExponential(2),'cmp-hint'));
    if(detail.mapped)box.append(el('p','精确来源：'+detail.sourcePort+' '+detail.sourceCoordinateLabel+' [flat '+detail.sourceIndex+'] → 当前输出；这是数据索引/搬运，不是矩阵乘法。','cmp-hint'));
    if(detail.elementwise)box.append(el('p','逐坐标原生运算：'+detail.operands.map(x=>x.port+'='+Number(x.value).toPrecision(9)).join('，')+'；'+({add:'加法',mul:'乘法',subtract:'减法',divide:'除法'}[detail.operation])+' → '+detail.recomputed.toPrecision(12),'cmp-hint'));
    if(detail.unary)box.append(el('p','逐坐标单输入计算 '+detail.operation+'：x='+detail.input.toPrecision(9)+(detail.parameter===null?'':'，参数='+detail.parameter)+' → '+detail.recomputed.toPrecision(12),'cmp-hint'));
    if(detail.kernel){
     box.append(el('p','逐行/逐坐标运算 '+detail.kernel+' · 输入 x='+Number(detail.input).toPrecision(10)+
      ' · '+detail.statistic+'='+detail.statisticValue.toPrecision(10),'cmp-hint'));
     const table=el('table',undefined,'cmp-terms'),head=el('tr');
     for(const label of ['行内坐标 k','输入 x[k]',detail.kernel==='softmax'?'exp(x[k]−max)':detail.kernel==='rmsnorm'?'x[k]²':'1+exp(−x)'])head.append(el('th',label));
     table.append(head);
     for(const term of detail.kernelTerms.slice(0,64)){
      const tr=el('tr');for(const value of [term.k,term.input.toPrecision(7),term.term.toPrecision(7)])tr.append(el('td',value));
      table.append(tr);
     }
     if(detail.kernelTerms.length>64)box.append(el('p','行内数据仅展示前 64 项；校验使用全部通道。','cmp-hint'));
     box.append(table);
    }
    if(!detail.mapped&&!detail.elementwise&&!detail.unary&&!detail.kernel){
    const table=el('table',undefined,'cmp-terms'),thead=el('tr');
    for(const head of ['输入通道 k','x[k]','W[out,k]','乘积'])thead.append(el('th',head));
    table.append(thead);
    for(const t of detail.terms.slice(0,64)){
      const tr=el('tr');for(const v of [t.k,t.input.toPrecision(5),t.weight.toPrecision(5),t.product.toPrecision(5)])tr.append(el('td',v));table.append(tr);
    }
    if(detail.terms.length>64)box.append(el('p','仅展示前 64 项，但闭合误差对全部 '+detail.terms.length+' 项计算。','cmp-hint'));
    box.append(table);
    }
   }else box.append(el('p','此运算不是单一 GEMM 点积；仍可查看输入/输出与数学公式，不伪造逐项分解。','cmp-hint'));
   rootElement.append(box);
  }
  if(Object.keys(s.operands).length){
   const details=el('details'),head=el('summary','展开本步骤全部输入、权重与中间值');details.append(head);
   for(const [name,value] of Object.entries(s.operands))details.append(mini(value,name));
   rootElement.append(details);
  }
  const foot=el('p','EVIDENCE · '+frame.origin+'；'+frame.limits,'cmp-footer');rootElement.append(foot);
 }
 function show(nextFrame,nodeId){
  CORE.validate(nextFrame);
  const target=nextFrame.steps.some(s=>s.id===nodeId)?nodeId:nextFrame.steps[0].id;
  if(nextFrame===frame&&target===active)return;
  const changed=frame!==nextFrame||target!==active;
  frame=nextFrame;active=target;if(changed){row=0;col=0;prefix=[];subSelected=null;}present();
 }
 clear();
 return Object.freeze({show,clear,get selected(){return active;},get frame(){return frame;}});
}
return Object.freeze({mount});
});