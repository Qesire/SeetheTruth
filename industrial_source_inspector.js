/* Source-backed architecture notes for a known educational recipe.
 * NEVER asserts that a reconnected arbitrary graph or imported checkpoint
 * implements exactly the cited model. Static, non-executable paper references.
 */
(function(root,factory){
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthIndustrialSources=api;
})(typeof window!=="undefined"?window:{},function(){"use strict";
const QWEN="https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py";
const CONFIG="https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/configuration_qwen3.py";
const DEEPSEEK="https://github.com/deepseek-ai/DeepSeek-V3";
const ENTRIES=Object.freeze({
 qgain:Object.freeze({title:"Qwen3 · 逐头 Q/K Norm 与共享可训练 γ",formula:"head_RMS(x)·γ; γ_Q∈ℝ², γ_K∈ℝ²",flow:"q0、q1 分别先按 head_dim=2 归一化；两条 Q 路径共用一个 qgain Parameter，K 使用另一 kgain。经过 Repeat 和 Multiply 后才应用 RoPE。",practice:"Qwen3Attention 中 q_norm/k_norm 在每个 Head 的 head_dim 上执行，且有独立的可训练 RMSNorm.weight，Q head 间共享 q_norm.weight。",trap:"本例采用二维 slice+repeat 展示各头，不是 Qwen3 的四轴 Tensor shape 或真实模型权重。",exercise:"修改 qgain[0,0]，比较 q0/q1 对同一参数的敏感度；运行反传查看双路梯度求和。",epistemic:"SOURCE_REFERENCED_BOUNDED_RECIPE; not an actual Qwen checkpoint",papers:[{id:"qwen3",title:"Qwen3Attention · Transformers source",url:QWEN}]}),
 gqa:Object.freeze({title:"Qwen3 · 2Q:1KV Grouped Query Attention",formula:"Q_h K_gᵀ / √d → causal softmax → P_h V_g",flow:"两个查询头由同一 Q Projection 切为 Q0/Q1；只有一组 K 和 V 投影，两个 query head 复用同一个 kr→kt 和 v，输出 concat 后再 o_proj。",practice:"工业 GQA 使用 num_attention_heads / num_key_value_heads 决定 Query 对共享 KV 的组大小，可降低 KV Cache 的头数和内存流量。",trap:"本例 2:1、head_dim=2 是缩尺计算图，没有 Paged KV Cache、FlashAttention、TensorParallel 或批量维，不能声称测得推理加速。",exercise:"固定 K/V 改变 Q1 路径，检查 Q0 的初始输出是否独立；对训练时共享 K/V 梯度做 VJP 数值比对。",epistemic:"SOURCE_REFERENCED_BOUNDED_RECIPE",papers:[{id:"qwen3",title:"Qwen3Attention · Transformers source",url:QWEN}]}),
 rope:Object.freeze({title:"RoPE · 头内坐标旋转",formula:"R_t(a,b)=(a cosθ_t−b sinθ_t, a sinθ_t+b cosθ_t)",flow:"Q/K 各自先完成 QK Norm 和可训练 γ 缩放，随后在头特征对上执行正交二维旋转；反传使用 R_tᵀ。",practice:"Qwen3 源码先 q_norm/k_norm，后 apply_rotary_pos_emb；旋转频率从 RoPE config 与 position_ids 派生。",trap:"本例采用相邻偶奇坐标布局、默认 base=10000，而 Qwen 的 rotate_half、RoPE base 和 Scaling 根据源码/配置可能不同，不能把两者按元素逐坐标直接视为同一实现。",exercise:"检查相邻通道对 L2 范数在旋转前后严格不变；有限差分验证 RoPE VJP。",epistemic:"MATH_VERIFIED_LAYOUT_VARIANT; source sequence matched, coordinate convention differs",papers:[{id:"qwen3",title:"Qwen3 rotary application source",url:QWEN}]}),
 tie:Object.freeze({title:"Embedding ↔ LM Head · 真正的参数别名",formula:"logits = H Eᵀ; dE = dE_gather + (dLogits)ᵀH",flow:"词表输出层从同一 embed Parameter 计算 transpose2d；没有独立 lm_head Parameter。两个语义分支在 Autograd 的同一存储身份上累加梯度。",practice:"权重绑定是部分开源模型的可选配置；Qwen3Config 中 tie_word_embeddings 默认 false，本课程显式选择 true 用于学习身份与梯度复用。",trap:"共享参数并不表示两个独立副本的数据恰好相等；只有单一参数身份、两路梯度累加且优化器状态只有一份才是权重绑定。",exercise:"保存一次 PyTorch checkpoint，检查仅有 embed 参数和一个对应 AdamW 状态，没有名为 lm_head 的第二套状态。",epistemic:"OPTIONAL_INDUSTRIAL_VARIANT_NOT_QWEN3_DEFAULT",papers:[{id:"config",title:"Qwen3Config · tie_word_embeddings",url:CONFIG}]}),
 mask:Object.freeze({title:"Causal Mask · 无未来信息流",formula:"mask[t,j]=1 iff j≤t; P[t,j]=0 if j>t",flow:"独立 mask node 构造下三角可见性；两个 Query heads 均通过 masked_softmax 使用它。梯度通过概率及 Q/K/V 路径传播，不通过离散 mask 结构传播。",practice:"工业 Attention 在逻辑语义上必须禁止未来 Token，具体实现可采用 kernel 内置 causal 配置，而不显式物化完整方阵。",trap:"本例真的生成 [T,T] 二维 mask 以便教学；这不等于 FlashAttention 的内核执行成本。",exercise:"只改动 t=2 的输入 Token，确认 t=0、t=1 的 logits 和 context 不变。",epistemic:"MATH_VERIFIED_SOURCE_REFERENCED",papers:[{id:"qwen3",title:"Qwen3Attention source",url:QWEN}]}),
 comparison:Object.freeze({title:"GQA 与 MLA 的根本区别",formula:"GQA: Hq>Hkv; MLA: compressed latent KV representation",flow:"本课 Q0/Q1 共享一组 K/V heads。DeepSeek-V3 的 MLA 从低维 latent 恢复/吸收 K/V，与 GQA 的直接共享 KV heads 不等价。",practice:"架构判断必须重建实际 Q/K/V 路径、RoPE 子空间、cache 状态和参数维数，不能凭模型名或‘低 KV’概括。",trap:"本课程不是 DeepSeek-V3 MLA 复刻；其关键缓存和权重吸收机制未实现。",exercise:"对比工业 GQA 的 K/V 宽度和 MLA 的 latent 缓存状态及从 latent 恢复的线性代数。",epistemic:"DISTINCT_ARCHITECTURE_FAMILY",papers:[{id:"deepseek",title:"DeepSeek-V3 · official architecture",url:DEEPSEEK}]})
});
const IDS=Object.freeze({
 qgain:"qgain",kgain:"qgain",q0n:"qgain",q1n:"qgain",kn:"qgain",
 qgainrows:"qgain",kgainrows:"qgain",q0aff:"qgain",q1aff:"qgain",kaff:"qgain",
 q:"gqa",k:"gqa",v:"gqa",q0:"gqa",q1:"gqa",kt:"gqa",score0:"gqa",score1:"gqa",out0:"gqa",out1:"gqa",heads:"gqa",out_proj:"gqa",
 q0r:"rope",q1r:"rope",kr:"rope",
 embed:"tie",tied_head:"tie",logits:"tie",
 mask:"mask",scale0:"mask",scale1:"mask",prob0:"mask",prob1:"mask",
});
function kind(graph,id){
 if(!graph||typeof graph.title!=="string"||!graph.title.includes("任务 38")||
    !graph.nodes?.some(n=>n.id===id))return null;
 return IDS[id]||null;
}
function card(key){return ENTRIES[key]||null;}
function render(key){
 const c=card(key);if(!c)return "";
 const esc=t=>String(t).replace(/[&<>"']/g,x=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[x]));
 return '<details class="knowledge-part" open><summary>模型源码拆解 · '+esc(c.title)+'</summary>'+
 '<p><strong>数学表示：</strong>'+esc(c.formula)+'</p>'+
 '<p><strong>实际组合链：</strong>'+esc(c.flow)+'</p>'+
 '<p><strong>工业共识与源码：</strong>'+esc(c.practice)+'</p>'+
 '<p><strong>适用边界：</strong>'+esc(c.trap)+'</p>'+
 '<p><strong>最小可证伪实验：</strong>'+esc(c.exercise)+'</p>'+
 '<p><strong>证据等级：</strong>'+esc(c.epistemic)+'</p>'+
 c.papers.map(p=>'<a target="_blank" rel="noopener noreferrer" href="'+esc(p.url)+'">↗ '+esc(p.title)+'</a>').join(" · ")+
 '</details>';
}
return Object.freeze({kind,card,render,available:Object.freeze(Object.keys(ENTRIES))});
});
