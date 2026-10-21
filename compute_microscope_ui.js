/* Reusable read-only computation microscope surface for workbench and model pages.
 * DOM constructed with textContent; imported source labels are not executable HTML. */
(function(root,factory){'use strict';const api=factory(root);if(typeof module==='object'&&module.exports)module.exports=api;root.SeetheTruthMicroscopeUI=api;})(typeof window!=='undefined'?window:{},function(root){'use strict';
const CORE=typeof module==='object'&&module.exports?require('./compute_microscope.js'):root.SeetheTruthComputeMicroscope;
function el(tag,text,cls){const a=document.createElement(tag);if(text!==undefined)a.textContent=String(text);if(cls)a.className=cls;return a;}
function mini(t,label,click,selected){
 const wrapper=el('div',undefined,'cmp-mini'),heading=el('div',label+' · ['+t.shape.join('×')+']','cmp-label');wrapper.append(heading);
 if(t.values===null){wrapper.append(el('p','静态形状推断：无数值矩阵。','cmp-hint'));return wrapper;}
 const [R,C]=t.shape,g=el('div',undefined,'cmp-matrix'),peak=Math.max(1e-12,...t.values.map(Math.abs));
 g.style.gridTemplateColumns='repeat('+Math.min(C,24)+',minmax(37px,1fr))';
 for(let i=0;i<Math.min(R,20);i++)for(let j=0;j<Math.min(C,24);j++){
  const v=t.values[i*C+j],b=el('button',Number(v).toPrecision(3),'cmp-cell');
  b.type='button';b.style.background='rgba('+(v>=0?'41,145,134,':'178,108,92,')+(.17+.69*Math.abs(v)/peak)+')';
  b.title='['+i+','+j+'] '+v.toPrecision(15);b.setAttribute('aria-label',label+' row '+i+' channel '+j+' = '+v.toPrecision(9));
  if(selected&&selected.row===i&&selected.col===j)b.classList.add('cmp-selected');
  if(click)b.addEventListener('click',()=>click(i,j));
  else b.disabled=true;
  g.append(b);
 }
 wrapper.append(g);
 if(R>20||C>24)wrapper.append(el('p','仅显示前 '+Math.min(R,20)+' 行、'+Math.min(C,24)+' 列；计算仍使用完整的有界张量。','cmp-hint'));
 return wrapper;
}
function mount(rootElement,onSelect){
 if(!rootElement||typeof rootElement.replaceChildren!=='function')throw Error('microscope UI mount required');
 let frame=null,active=null,row=0,col=0;
 function clear(msg='运行数值图或进行静态形状分析后，选择积木即可展开；当前没有执行证据。'){
  frame=null;active=null;rootElement.replaceChildren(el('p',msg,'cmp-hint'));
 }
 function present(){
  if(!frame)return clear();
  const s=frame.steps.find(x=>x.id===active)||frame.steps[0];
  rootElement.replaceChildren();
  const summary=el('div',undefined,'cmp-top');summary.append(el('div',frame.evidenceTier,'cmp-tier'),el('div',frame.title,'cmp-title'));
  rootElement.append(summary);
  const nav=el('div',undefined,'cmp-steps');nav.setAttribute('role','group');nav.setAttribute('aria-label','已验证执行顺序');
  for(let i=0;i<frame.steps.length;i++){
   const item=frame.steps[i],b=el('button',(i+1)+'. '+item.title,'cmp-step');
   b.type='button';if(item.id===s.id)b.setAttribute('aria-current','step');
   b.title=item.kind+' · '+item.out.shape.join('×');
   b.addEventListener('click',()=>{active=item.id;row=0;col=0;present();onSelect?.(item.id);});nav.append(b);
  }
  rootElement.append(nav);
  const meta=el('div',undefined,'cmp-meta');
  meta.append(el('strong',s.title+' · '+s.kind),el('span','Output '+s.out.shape.join(' × ')));
  rootElement.append(meta);
  const operands=el('div',undefined,'cmp-deps');
  operands.append(el('div','实际依赖 / DATAFLOW EDGES','cmp-label'));
  if(s.dependencies.length)for(const dep of s.dependencies)operands.append(el('span',dep.from+' → '+dep.port,'cmp-chip'));
  else operands.append(el('span','起始阶段 / 无上游积木','cmp-chip'));
  rootElement.append(operands);
  const math=el('pre',s.formula,'cmp-formula');rootElement.append(math);
  rootElement.append(el('p',s.why,'cmp-hint'));
  if(frame.source?.url&&s.sourceLines){
   const link=el('a','查看固定源码 '+frame.source.path+':'+s.sourceLines.start,'cmp-link');
   link.href=frame.source.url+'#L'+s.sourceLines.start+'-L'+s.sourceLines.end;link.target='_blank';link.rel='noopener noreferrer';rootElement.append(link);
  }
  const output=el('div',undefined,'cmp-out');
  output.append(mini(s.out,'输出张量 / '+s.id,(r,c)=>{row=r;col=c;present();}, {row,col}));
  rootElement.append(output);
  if(s.out.values!==null){
   const detail=CORE.dot(frame,s.id,Math.min(row,s.out.shape[0]-1),Math.min(col,s.out.shape[1]-1));
   const box=el('div',undefined,'cmp-breakdown');
   box.append(el('div','坐标 ['+detail.row+','+detail.col+'] = '+detail.value.toPrecision(12),'cmp-label'));
   box.append(el('pre',detail.formula,'cmp-formula'));
   if(detail.reconstructable){
    box.append(el('p','逐项重建值 '+detail.recomputed.toPrecision(12)+' · 误差 '+detail.error.toExponential(2),'cmp-hint'));
    const table=el('table',undefined,'cmp-terms'),thead=el('tr');
    for(const head of ['输入通道 k','x[k]','W[out,k]','乘积'])thead.append(el('th',head));
    table.append(thead);
    for(const t of detail.terms.slice(0,64)){
      const tr=el('tr');for(const v of [t.k,t.input.toPrecision(5),t.weight.toPrecision(5),t.product.toPrecision(5)])tr.append(el('td',v));table.append(tr);
    }
    if(detail.terms.length>64)box.append(el('p','仅展示前 64 项，但闭合误差对全部 '+detail.terms.length+' 项计算。','cmp-hint'));
    box.append(table);
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
  frame=nextFrame;active=target;if(changed){row=0;col=0;}present();
 }
 clear();
 return Object.freeze({show,clear,get selected(){return active;},get frame(){return frame;}});
}
return Object.freeze({mount});
});