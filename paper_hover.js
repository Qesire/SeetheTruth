/* Paper-aware hover inspector: augments, never replaces, Scratch-style composition. */
(function(root,factory){const api=factory(root);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthPaperHover=api;})(typeof window!=="undefined"?window:globalThis,function(root){"use strict";
const K=root.SeetheTruthBlockKnowledge;
function safe(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function resolve(kind){if(!K||!K.get)return null;const card=K.get(kind);if(!card)return null;return {...card,papers:card.refs.map(id=>({id,title:K.sources[id][0],url:K.sources[id][1]}))};}
function install(doc=root.document){if(!doc||!K)return null;let timer=null,active=null,pinned=false;
const panel=doc.createElement("aside");panel.id="paperHoverInspector";panel.setAttribute("role","complementary");panel.setAttribute("aria-label","积木论文解析");panel.hidden=true;doc.body.append(panel);
const show=(kind,anchor)=>{const card=resolve(kind);if(!card)return;active=kind;const papers=card.papers.map(p=>'<a href="'+safe(p.url)+'" target="_blank" rel="noopener noreferrer">'+safe(p.title)+' ↗</a>').join("");panel.innerHTML='<div class="phi-heading"><span>RESEARCH / BLOCK ANATOMY</span><button type="button" data-phi-close aria-label="关闭论文解析">×</button></div><h3>'+safe(card.title)+'</h3><code>'+safe(card.formula)+'</code><h4>计算流与数学功能</h4><p>'+safe(card.flow)+'</p><h4>论文与业界实现</h4><p>'+safe(card.practice)+'</p><div class="phi-references">'+papers+'</div><h4>易错认知与边界</h4><p>'+safe(card.trap)+'</p><h4>动手验证</h4><p>'+safe(card.exercise)+'</p><small>'+safe(card.epistemic)+'</small>';
panel.hidden=false;const rect=anchor.getBoundingClientRect(),width=Math.min(390,Math.max(280,root.innerWidth-24));panel.style.width=width+"px";const x=rect.right+12+width<root.innerWidth?rect.right+12:Math.max(12,rect.left-width-12);panel.style.left=Math.min(root.innerWidth-width-12,Math.max(12,x))+"px";panel.style.top=Math.min(Math.max(12,rect.top),Math.max(12,root.innerHeight-Math.min(panel.scrollHeight,root.innerHeight-24)-12))+"px";
};
const hide=()=>{if(!pinned){panel.hidden=true;active=null;}};
const kindFrom=el=>{const b=el.closest("[data-add]");if(b)return b.dataset.add;const n=el.closest("[data-node]");if(n){const h=n.querySelector(".block-kind");const name=h?.textContent?.split(" · ").slice(1).join(" · ");const L=root.SeetheTruthLearning;return Object.keys(L?.defs||{}).find(k=>L.defs[k].en===name)||null;}return null;};
doc.addEventListener("pointerover",e=>{if(panel.contains(e.target))return;const anchor=e.target.closest?.("[data-add],[data-node]");if(!anchor)return;const kind=kindFrom(anchor);if(!kind||kind===active)return;clearTimeout(timer);timer=setTimeout(()=>{if(!pinned)show(kind,anchor);},180);});
doc.addEventListener("pointerout",e=>{const anchor=e.target.closest?.("[data-add],[data-node]");if(!anchor)return;if(e.relatedTarget&&(anchor.contains(e.relatedTarget)||panel.contains(e.relatedTarget)))return;clearTimeout(timer);timer=setTimeout(hide,250);});
panel.addEventListener("pointerenter",()=>clearTimeout(timer));panel.addEventListener("pointerleave",()=>{timer=setTimeout(hide,200);});
panel.addEventListener("click",e=>{if(e.target.closest("[data-phi-close]")){pinned=false;panel.hidden=true;active=null;}});
doc.addEventListener("keydown",e=>{if(e.key==="Escape"){pinned=false;panel.hidden=true;active=null;}});
doc.addEventListener("focusin",e=>{const el=e.target.closest?.("[data-add],[data-node]");if(el){const kind=kindFrom(el);if(kind&&!pinned)show(kind,el);}});
doc.addEventListener("click",e=>{if(!e.target.closest("[data-add],[data-node]")&&!panel.contains(e.target))hide();});
return {show,hide,resolve};}
return Object.freeze({install,resolve});
});
