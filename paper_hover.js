/* Paper-aware hover inspector: augments, never replaces, Scratch-style composition. */
(function(root,factory){const api=factory(root);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthPaperHover=api;})(typeof window!=="undefined"?window:globalThis,function(root){"use strict";
const K=root.SeetheTruthBlockKnowledge;
function safe(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function resolve(kind){if(!K||!K.get)return null;const card=K.get(kind);if(!card)return null;return {...card,papers:card.refs.map(id=>({id,title:K.sources[id][0],url:K.sources[id][1]}))};}
function install(doc=root.document){if(!doc||!K)return null;let timer=null,active=null,pinned=false;
const panel=doc.createElement("aside");panel.id="paperHoverInspector";panel.setAttribute("role","complementary");panel.setAttribute("aria-label","积木论文解析");panel.hidden=true;doc.body.append(panel);
const show=(kind,anchor)=>{const card=anchor?.getAttribute("data-paper-patch")==="lora"?{title:"LoRA · 低秩参数改造",formula:"Y=XW₀+(α/r)(XA)B",flow:"将选中 MatMul 替换为原稠密分支 XW₀ 与 XA→B→scale→Add 低秩分支，原节点输出 ID 不变；B=0 保证改造时前向数值等价。",practice:"W₀ 保持只读 Input 或将原 Parameter 显式冻结，A/B 是两个可训练 Parameter。梯度经过正常计算图求和；不得把离散 Expert 路由或权重量化混为 LoRA。",trap:"LoRA 本身并不会减少基础稠密投影的前向乘法量；低秩更新不等于稀疏推理加速。继续编辑连线后应重新测试前向等价。",exercise:"分别扰动低秩 B 和 A，执行两步优化器更新，比较梯度和完整图输出。",epistemic:"REAL GRAPH REWRITE · bounded math; no model checkpoint or hardware measurement",papers:[{id:"lora",title:"LoRA · Hu et al.",url:"https://arxiv.org/abs/2106.09685"}]}:resolve(kind);if(!card)return;active=kind;const papers=card.papers.map(p=>'<a href="'+safe(p.url)+'" target="_blank" rel="noopener noreferrer">'+safe(p.title)+' ↗</a>').join("");panel.innerHTML='<div class="phi-heading"><span>RESEARCH / BLOCK ANATOMY</span><button type="button" data-phi-close aria-label="关闭论文解析">×</button></div><h3>'+safe(card.title)+'</h3><code>'+safe(card.formula)+'</code><h4>计算流与数学功能</h4><p>'+safe(card.flow)+'</p><h4>论文与业界实现</h4><p>'+safe(card.practice)+'</p><div class="phi-references">'+papers+'</div><h4>易错认知与边界</h4><p>'+safe(card.trap)+'</p><h4>动手验证</h4><p>'+safe(card.exercise)+'</p><small>'+safe(card.epistemic)+'</small>';
panel.hidden=false;const vw=root.innerWidth||doc.documentElement.clientWidth,width=Math.min(390,Math.max(240,vw-24));panel.style.width=width+"px";panel.style.left=Math.max(12,vw-width-14)+"px";panel.style.top="90px";panel.style.maxHeight=Math.max(180,(root.innerHeight||720)-106)+"px";
};
const hide=()=>{if(!pinned){panel.hidden=true;active=null;}};
const kindFrom=el=>{const b=el.closest("[data-add]");if(b)return b.dataset.add;const n=el.closest("[data-node]");if(n){const h=n.querySelector(".block-kind");const name=h?.textContent?.split(" · ").slice(1).join(" · ");const L=root.SeetheTruthLearning;return Object.keys(L?.defs||{}).find(k=>L.defs[k].en===name)||null;}return null;};
doc.addEventListener("pointerover",e=>{if(panel.contains(e.target))return;const anchor=e.target.closest?.("[data-add],[data-node]");if(!anchor)return;const kind=kindFrom(anchor);if(!kind||kind===active)return;clearTimeout(timer);timer=setTimeout(()=>{if(!pinned)show(kind,anchor);},180);});
doc.addEventListener("pointerout",e=>{const anchor=e.target.closest?.("[data-add],[data-node]");if(anchor&&panel.hidden)clearTimeout(timer);});
panel.addEventListener("pointerenter",()=>clearTimeout(timer));panel.addEventListener("pointerleave",()=>clearTimeout(timer));
panel.addEventListener("click",e=>{if(e.target.closest("[data-phi-close]")){pinned=false;panel.hidden=true;active=null;}});
doc.addEventListener("keydown",e=>{if(e.key==="Escape"){pinned=false;panel.hidden=true;active=null;}});
doc.addEventListener("focusin",e=>{const el=e.target.closest?.("[data-add],[data-node]");if(el){const kind=kindFrom(el);if(kind&&!pinned)show(kind,el);}});
doc.addEventListener("click",e=>{if(!panel.contains(e.target))hide();});
doc.addEventListener("dragstart",hide);
return {show,hide,resolve};}
return Object.freeze({install,resolve});
});
