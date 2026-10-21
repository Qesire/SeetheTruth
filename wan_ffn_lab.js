(function(){'use strict';
const A=window.SeetheTruthWanFFNTrace;if(!A)throw Error('Wan FFN trace unavailable');
const $=id=>document.getElementById(id);
const names={shift:'e3 / FFN shift',scale:'e4 / FFN scale',gate:'e5 / output gate',afterSelf:'x after self-attn',crossDelta:'externally supplied cross-attn delta',crossInput:'cross-attn residual x',normalized:'norm2(x)',input:'GEMM input',weights:'nn.Linear weights, [out,in]',bias:'nn.Linear bias',preActivation:'Linear1 logits',ffn:'Linear2 output',delta:'FFN branch change'};
const state={step:0,row:0,col:0,data:null,error:null};
function options(){return Object.fromEntries(['tokens','dim','hidden','seed','gate','scale','shift'].map(id=>[id,Number($(id).value)]));}
const fmt=v=>Number(v).toFixed(5);
const signColor=(value,max)=>{const strength=.13+.64*Math.min(1,Math.abs(value)/(max||1));return value>=0?'rgba(45,160,139,'+strength+')':'rgba(172,99,85,'+strength+')';};
function element(tag,textValue,cls){const e=document.createElement(tag);if(textValue!==undefined)e.textContent=String(textValue);if(cls)e.className=cls;return e;}
function matrixView(m,label,onCell,compact=false){
 const container=element('div',undefined,compact?'micro-matrix':'');
 container.append(element('div',label+' · ['+m.shape.join(' × ')+']','mat-label'));
 const grid=element('div',undefined,'mat');const [rows,cols]=m.shape,max=Math.max(...m.values.map(Math.abs),1e-12);
 grid.style.gridTemplateColumns='repeat('+cols+',minmax(0,1fr))';
 for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){
  const val=m.values[r*cols+c],b=element('button',fmt(val));b.type='button';
  b.style.backgroundColor=signColor(val,max);
  b.title='['+r+','+c+'] = '+val.toPrecision(15);
  b.setAttribute('aria-label',label+' token '+r+' channel '+c+' = '+val.toPrecision(12));
  if(onCell){if(r===state.row&&c===state.col)b.classList.add('selected');b.addEventListener('click',()=>{state.row=r;state.col=c;render();});}
  else b.addEventListener('click',()=>{b.title+=' · '+val.toString();});
  grid.append(b);
 }container.append(grid);return container;
}
function flow(){
 const d=$('diagram');d.replaceChildren();
 const ids=['condition','cross','norm','affine','up','gelu','down','gate','residual'];
 for(let i=0;i<ids.length;i++){
  const step=A.STEPS[i],button=element('button',undefined,'flow-node'+(state.step===i?' active':''));
  button.type='button';button.append(element('b',step.name),element('small',step.shape));
  button.addEventListener('click',()=>{state.step=i;state.row=0;state.col=0;render();});d.append(button);
  if(i<ids.length-1)d.append(element('span','→','arrow'));
 }
 const selected=d.querySelector('.flow-node.active');if(selected)selected.scrollIntoView({block:'nearest',inline:'nearest'});
}
function breakdown(){
 const info=A.explainDot(state.data,state.step,state.row,state.col);
 const area=$('dotBreakdown');area.replaceChildren();
 $('selected').textContent='选中位置：token '+state.row+' / feature '+state.col+'；数值 '+info.value.toPrecision(12);
 area.append(element('pre',info.formula,'code'));
 if(info.terms){
  const table=element('table',undefined,'dot'),head=element('tr');
  for(const field of ['输入通道','input[t,k]','W[h,k]','乘积'])head.append(element('th',field));
  const thead=element('thead');thead.append(head);table.append(thead);
  const body=element('tbody');for(const t of info.terms){
   const row=element('tr');for(const v of [t.k,fmt(t.x),fmt(t.w),fmt(t.product)])row.append(element('td',v));body.append(row);
  }
  const summary=element('tr');
  for(const v of ['bias + Σ','bias='+fmt(info.bias),'Σ乘积='+fmt(info.terms.reduce((a,b)=>a+b.product,0)),'='+fmt(info.reconstructed)])summary.append(element('td',v));
  body.append(summary);table.append(body);area.append(table);
  const abs=Math.abs(info.reconstructed-info.value);
  area.append(element('p','点积数值闭合误差 = '+abs.toExponential(3),'tiny muted'));
 }else if(info.base!==undefined){
  area.append(element('p','x=' +fmt(info.base)+'，FFN delta='+fmt(info.delta)+'，相加='+fmt(info.reconstructed),'tiny'));
 }else if(info.input!==undefined){
  area.append(element('p','GELU 输入 '+fmt(info.input)+' → 输出 '+fmt(info.value),'tiny'));
 }
}
function integrity(){
 const t=state.data,{tokens:T,dim:C,hidden:H}=t.config;
 const final=t.steps[8].main.values,cross=t.steps[1].main.values,gate=t.steps[7].main.values;
 const residualErr=Math.max(...final.map((v,i)=>Math.abs(v-(cross[i]+gate[i]))));
 const norm=t.steps[2].main.values,means=Array.from({length:T},(_,row)=>norm.slice(row*C,(row+1)*C).reduce((a,b)=>a+b,0)/C);
 const dotIndex=state.step===4||state.step===6?state.step:4;
 const dot=A.explainDot(t,dotIndex,0,0),dotErr=Math.abs(dot.value-dot.reconstructed);
 const zeroError=t.config.gate===0?Math.max(...final.map((v,i)=>Math.abs(v-cross[i]))):null;
 const text=['source topology: Wan2.2 I2V FFN / 9 stages','demo: T='+T+', C='+C+', H='+H+' · Float64','first+second GEMM ≈ '+t.flopsDemo.toLocaleString()+' arithmetic FLOPs',
  'W1[H,C]='+H+'×'+C+', W2[C,H]='+C+'×'+H,'residual identity max error: '+residualErr.toExponential(2),
  'Linear1 dot reconstruction: '+dotErr.toExponential(2),
  'LayerNorm per-token |mean| max: '+Math.max(...means.map(Math.abs)).toExponential(2)];
 if(zeroError!==null)text.push('counterfactual e5=0 → x_out=x_after_cross, max error: '+zeroError.toExponential(2));
 $('numerics').textContent=text.join('\n');
}
function render(){
 const d=state.data,step=d.steps[state.step],m=step.main;
 state.row=Math.min(state.row,m.shape[0]-1);state.col=Math.min(state.col,m.shape[1]-1);
 $('ordinal').textContent='EXECUTION STAGE '+(state.step+1)+' / '+d.steps.length;
 $('stepTitle').textContent=step.name;$('shape').textContent='张量形状 '+step.shape+' · 源码行 '+step.lines;
 $('sourceCode').textContent=step.code;
 $('why').textContent=step.why;$('failure').textContent=step.error;
 $('lineRef').textContent=A.SOURCE.path+' · lines '+step.lines+' · commit '+A.SOURCE.commit.slice(0,12);
 const firstLine=step.lines.match(/\d+/)?.[0]||211;
 $('openSource').href=A.SOURCE.source+'#L'+firstLine;
 $('configUrl').href=A.SOURCE.configUrl;
 $('prev').disabled=state.step===0;$('next').disabled=state.step===d.steps.length-1;
 $('mainLabel').textContent='OUTPUT '+step.id+' · shape '+m.shape.join(' × ');
 $('mainMatrix').replaceChildren(...Array.from(matrixView(m,'main output',true).children).slice(1));
 $('extraMatrices').replaceChildren(...Object.entries(step.aux).map(([name,value])=>matrixView(value,names[name]||name,false,true)));
 $('stateLabel').textContent='当前重新计算版本：seed '+d.config.seed+'；Gate倍率 '+d.config.gate+'；Scale倍率 '+d.config.scale+'；Shift倍率 '+d.config.shift+
 '。Gate倍率是演示反事实控制，不等于修改官方固定参数。';
 $('realFlops').textContent=d.real.flopsPerToken.toLocaleString();
 $('cost').replaceChildren(element('p','官方单 block 配置：C=5120、FFN H=13824；两次投影 weight+bias 约 '+d.real.params.toLocaleString()+' 个参数。'),
  element('p','本地演示：T='+d.config.tokens+'，C='+d.config.dim+'，H='+d.config.hidden+'；并非真实规模的权重执行。'),
  element('p','在序列 Token 数 N 下，两次 Dense Linear 理论约 4×N×C×H FLOPs。'));
 $('steps').replaceChildren(...d.steps.map((s,i)=>{const b=element('button',undefined,'step');b.type='button';
  b.setAttribute('aria-current',i===state.step?'step':'false');b.append(element('span',String(i).padStart(2,'0'),'idx'),
  element('span',s.name));b.addEventListener('click',()=>{state.step=i;state.row=0;state.col=0;render();});return b;}));
 flow();breakdown();integrity();
}
function recompute(){
 try{state.data=A.trace(options());state.error=null;state.row=0;state.col=0;render();}
 catch(e){state.error=e.message;$('numerics').textContent='拒绝不合法输入：'+e.message;}
}
$('recompute').addEventListener('click',recompute);
$('zeroGate').addEventListener('click',()=>{$('gate').value='0';state.step=8;recompute();});
$('reset').addEventListener('click',()=>{for(const [key,value] of Object.entries({tokens:3,dim:4,hidden:8,seed:17,gate:1,scale:1,shift:1}))$(key).value=value;state.step=0;recompute();});
$('prev').addEventListener('click',()=>{state.step=Math.max(0,state.step-1);state.row=0;state.col=0;render();});
$('next').addEventListener('click',()=>{state.step=Math.min(A.STEPS.length-1,state.step+1);state.row=0;state.col=0;render();});
recompute();
window.SeetheTruthWanFFNLearning={get current(){return state.step;},get stages(){return A.STEPS.length;},get evidence(){return state.data?.evidence;},get lastError(){return state.error;},get lastOutput(){return state.data?.final.values.slice();}};
})();