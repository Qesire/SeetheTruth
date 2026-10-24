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
function mount(rootElement,onSelect,onCoordinate){
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
  // The same selected graph node and coordinate drive forward and backward
  // observations; nested substeps are not standalone graph nodes.
  if(s===parent&&s.out.values!==null&&(frame.evidenceTier===CORE.TIERS.MICRO||frame.evidenceTier===CORE.TIERS.WAN)&&typeof onCoordinate==='function'){
   const vjp=el('div',undefined,'cmp-vjp'),action=el('button','追踪此坐标的局部 VJP / 梯度汇合','cmp-vjp-trigger');
   action.type='button';
   const detail=el('div','选择此输出坐标的反向贡献；不等于对整个训练损失求导。','cmp-vjp-result cmp-hint');
   action.addEventListener('click',()=>{
    try{
     const rank=s.out.shape.length;
     const leading=Array.from({length:Math.max(0,rank-2)},(_,i)=>prefix[i]??0);
     const index=[...leading,...(rank===1?[]:[row]),col];
     const report=onCoordinate({stepId:s.id,coordinate:index,frame});
     if(report?.schema!=='seethetruth-coordinate-vjp-bridge-v1')throw Error('unregistered VJP observation');
     detail.replaceChildren();
     detail.append(el('strong','已绑定输出 '+report.selected.nodeId+' ['+report.selected.index.join(',')+'] · 值 '+report.selected.value.toPrecision(9)));
     if(report.evidence==='PINNED_WAN_SOURCE_SYNTHETIC_FLOAT64_NOT_MODEL_GRADIENT')
      detail.append(el('p','WAN SOURCE-PINNED · '+report.source?.path+' @ '+
       report.source?.revision?.slice(0,12)+'；这只是官方结构对齐的合成小张量局部 VJP，非官方权重/真实模型梯度。','cmp-hint'));
     detail.append(el('p',report.backward.nodeCount+' 个元素坐标、'+report.backward.edgeCount+' 条局部 VJP 边；'+
      (report.backward.complete?'已覆盖全部可验证路径':'存在规则边界或探索预算截断：仅展示已核验路径'),'cmp-hint'));
     const local=report.backward.localEdges.slice(0,12);
     if(local.length){
      detail.append(el('div','首层局部导数 ∂当前输出/∂上游坐标','cmp-label'));
      for(const edge of local)detail.append(el('p',edge.port+' → '+edge.source+
       '：'+(edge.localDerivative===null?'尚无数值':Number(edge.localDerivative).toPrecision(9)),'cmp-hint'));
     }
     if(report.backward.internalStages?.length){
      detail.append(el('div','算子内部计算 / FORWARD–JACOBIAN STAGES','cmp-label'));
      for(const stage of report.backward.internalStages.slice(0,24)){
       const fields=Object.entries(stage).map(([name,value])=>name+'='+
        (value===null?'静态未求值':typeof value==='number'?Number(value).toPrecision(7):String(value)));
       detail.append(el('p',fields.join(' · '),'cmp-hint'));
      }
      if(report.backward.internalStages.length>24)
       detail.append(el('p','只展示前 24 项；完整的 bounded 逐项导数仍已计算。','cmp-hint'));
     }
     if(report.backward.discreteConditional)
      detail.append(el('p','DISCRETE ROUTING / 固定离散选择：'+
       report.backward.conditionalCuts.slice(0,8).map(c=>c.nodeId+'.'+c.port+
       (c.selection===null?'':' → expert '+c.selection)).join('；')+
       '。仅对这次运行所选择的连续分支求导；没有穿过 TopK / expert ID。','cmp-hint'));
     detail.append(el('div','源张量坐标的累计灵敏度（相同来源的多条路径已求和）','cmp-label'));
     for(const source of report.backward.sources.slice(0,24))
      detail.append(el('p',source.nodeId+' ['+source.index.join(',')+'] · ∂selected/∂source = '+
       (source.sensitivity===null?'未知，非零值未获证明':Number(source.sensitivity).toPrecision(9)),'cmp-hint'));
     if(report.backward.sources.length>24)detail.append(el('p','仅展示按绝对灵敏度排序的前 24 个来源，计算覆盖完整源坐标。','cmp-hint'));
     if(report.backward.externalBoundary)
      detail.append(el('p','计算边界：'+report.backward.externalBoundary,'cmp-hint'));
     const loss=report.loss;
     if(loss?.status==='VERIFIED_GRAPH_LOSS_VJP'){
      detail.append(el('div','TRAINING LOSS / 独立完整反向传播','cmp-label'));
      detail.append(el('p','目标 '+loss.targetId+' · L='+Number(loss.loss).toPrecision(9)+
       ' · 选定坐标 ∂L/∂yᵢ='+loss.selectedAdjoint.toPrecision(9)+
       (loss.selectedOnLossPath?' · 位于损失路径':' · 不在损失路径'),'cmp-hint'));
      detail.append(el('p','对每个来源：经本坐标贡献 = (∂L/∂yᵢ)(∂yᵢ/∂source)，'+
       '完整梯度还包括经过其他输出坐标或其他分支的贡献。','cmp-hint'));
      for(const source of loss.sourceContributions.slice(0,24)){
       detail.append(el('p',source.nodeId+' ['+source.index.join(',')+'] · 经此输出 = '+
        (source.throughSelected===null?'未验证/不适用':source.throughSelected.toPrecision(8))+
        '；完整 dL/dSource = '+source.wholeLossGradient.toPrecision(8)+
        (source.stopped?' · 本图中有停梯度边':'') ,'cmp-hint'));
      }
      if(loss.sourceContributions.length>24)detail.append(el('p','只展示前 24 个源坐标，完整梯度仍由全图反向传播计算。','cmp-hint'));
      if(loss.cutEdges.length)detail.append(el('p','停止梯度：'+loss.cutEdges.slice(0,12).map(c=>
       c.nodeId+'.'+c.port+' ← '+c.from+'：'+c.reason).join('；'),'cmp-hint'));
     }else if(loss?.status==='NO_LOSS_OBJECTIVE'){
      detail.append(el('p','TRAINING LOSS · 当前图无符合规则的标量损失；以上只是所选输出坐标的 VJP。','cmp-hint'));
     }else if(loss?.status==='UNAVAILABLE_GRAPH_LOSS_VJP'){
      detail.append(el('p','TRAINING LOSS · '+loss.targetId+' 的完整反向传播未通过：'+loss.reason+
       '；不以零梯度代替缺失算子。','cmp-hint'));
     }
     detail.append(el('p','NumPy 单算子资格 '+(report.codegen.numpyOperatorEligible?'有':'无')+
      ' · PyTorch 单算子资格 '+(report.codegen.torchOperatorEligible?'有':'无')+
      '；整图导出及 VJP 代码导出均未在此证明。','cmp-hint'));
    }catch(error){detail.replaceChildren(el('p','反向路径拒绝：'+error.message,'cmp-hint'));}
   });
   vjp.append(action,detail);rootElement.append(vjp);
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