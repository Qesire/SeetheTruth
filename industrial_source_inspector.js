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
const OURO="https://github.com/Ouro-LLM/Ouro";
const UT="https://arxiv.org/abs/1807.03819";
const HUGINN="https://github.com/seal-rg/recurrent-pretraining";
const MOR="https://github.com/raymin0223/mixture_of_recursions";
const ENTRIES=Object.freeze({
 loop_residual:Object.freeze({title:"Looped residual scale · 为什么 1/N 可能优于 1/√N？",formula:"H←H+εFθ(H); N=2, ε∈{1,1/√2,1/2}",flow:"两轮共享同一套 Attention/SwiGLU 参数，每轮 Attention 输出和 FFN 输出都先经过一个真实 Scale，再与残差主路 Add。四个 Scale 当前数值必须匹配；参数不复制。",practice:"2026 年对 Looped Transformers 的残差缩放研究强调共享访问导致相关更新，因此讨论 1/N 而非单纯非共享网络的 1/√N。DeepLoop 另在 Post-LN DeepNorm 里用 α/β 定义稳定更新。",trap:"本例是 Pre-Norm 的严格控制消融，不是 DeepLoop 的 Post-LN 结构、不等同已证明 1/N 必然提升真实 LLM 收敛；更小的更新幅度也可能妨碍有效表征更新。",exercise:"改动四个 Scale 参数并比较损失、相对输入状态变化 RMS、Wq 梯度；固定权重做差分，观察 1/√N 与 1/N 的梯度差别。",epistemic:"CONTROLLED_CPU_MICRO_GRAPH_MECHANISM; NOT_LARGE_MODEL_GENERALIZATION",papers:[{id:"residual",title:"On the Residual Scaling of Looped Transformers (2026)",url:"https://arxiv.org/abs/2606.18524"},{id:"deeploop",title:"DeepLoop · Post-LN α/β is a different parameterization",url:"https://arxiv.org/abs/2607.13491"}]}),
 loop_multidepth:Object.freeze({title:"Ouro 启发 · 每轮深度监督与权重共享",formula:"H_(r+1)=F_θ(H_r); L=Σᵣ αᵣ CE(logits_r,labels), αᵣ≥0, Σαᵣ=1",flow:"每轮独立经过共享 Q/K/V、Causal Attention、Sandwich RMSNorm、SwiGLU 和同一个 tied_head，产出每轮 CE；用 Scale + Add 汇成一个真正的反向目标。改深度时不复制参数及 AdamW 状态。",practice:"Ouro 的循环语言模型确实使用共享层组、多步输出和学习式深度分配。这个实验仅以固定非负系数验证多深度梯度，尚未实现 learned exit gate、entropy regularization 或 per-token depth routing。",trap:"有多轮 CE 并不等于动态退出：当前全部轮次均执行，不能声称推理省算力、达到 Ouro 精度或重现其训练配方。切换循环数会改变目标权重，但保留全部参数/优化器状态。",exercise:"比较第 0、1、2 轮的 CE，验证最终 loss 是每轮加权和；扰动 Wq 做有限差分；训练后从 2 轮切至 3 轮，参数和优化器动量应逐项不变。",epistemic:"EXECUTABLE_FIXED_DEPTH_MULTI_STEP_SUPERVISION; NOT ADAPTIVE_HALTING",papers:[{id:"ouro",title:"Ouro · Scaling Latent Reasoning via Looped Language Models",url:"https://arxiv.org/abs/2510.25741"},{id:"recipes",title:"Closing the Loop · practical recipes",url:"https://arxiv.org/abs/2610.00673"}]}),
 loop_shared:Object.freeze({title:"Looped Transformer · 跨深度共享真实参数",formula:"H^(r+1)=F_θ(H^r); dL/dθ=Σ_r (∂L/∂H^(r+1)) ∂F_θ(H^r)/∂θ",flow:"完整 Attention+FFN 数学流在第 r 轮和 r+1 轮重复，所有 Q/K/V、输出和 SwiGLU 权重回到同一 Parameter 节点。为了与自由 DAG 计算引擎兼容，循环被显式展开为两轮或三轮，不建立非法图循环。",practice:"Universal Transformer 采用深度方向的迭代计算；Huginn / Ouro 将 recurrent depth 扩展到语言模型。增加有效深度可以不增加核心 Block 的存储参数，但激活与计算次数会增长。",trap:"这不是 Ouro 完整 checkpoint、真实隐藏状态初始注入、随机深度训练、ACT 学习式退出或 GPU kernel；本课固定深度、无省算力声明。不要误将同值但不同参数 ID 当成权重共享。",exercise:"比较两轮与三轮模型的 Parameter ID 集合完全相同，逐轮 state 不同。单独修改 wq 一处，必须影响两个 loop 的输出梯度。",epistemic:"EXECUTABLE_SHARED_WEIGHT_RECURRENT_DEPTH_SUBSET",papers:[{id:"ouro",title:"Ouro official implementation",url:OURO},{id:"ut",title:"Universal Transformers 2018",url:UT},{id:"huginn",title:"Huginn recurrent pretraining",url:HUGINN}]}),
 loop_causal:Object.freeze({title:"Looped Causal Attention · 循环层级与 Token 时间轴独立",formula:"P_r[t,j]=softmax_j(Q_r[t]K_r[j]^T/√d), j≤t",flow:"每轮 state_r 经过共享 Q/K/V 投影、RoPE、Causal Mask、Softmax、PV、输出投影与残差。不同轮计算同一 Token 序列的更高阶表征；不会新增输出 Token 或将未来 Token 暴露给过去。",practice:"真实 recurrent-decoder 需要同时追踪 token/time 和 recurrence/depth 两轴。Serving 时按 loop/layer 定位 KV Cache 或采取复用策略，否则会错误地把不同深度的缓存混为一谈。",trap:"这里只模拟 prefill 型完整三 Token 窗口。没有递归深度对应的独立 KV Cache、跨 loop 的缓存共享优化、Prefill/Decode 调度或 FlashAttention 实测。",exercise:"改变输入序列最后一个上下文 Token，检查每轮早期位置的概率、state 与最终 logits 严格不变。",epistemic:"CAUSAL_RECURRENT_DEPTH_MATH_VERIFIED_NO_CACHE",papers:[{id:"ouro",title:"Ouro reference",url:OURO},{id:"huginn",title:"Huginn official code",url:HUGINN}]}),
 loop_ffn:Object.freeze({title:"Recurrent SwiGLU FFN · 每次重用同一套门控矩阵",formula:"U_r=RMSNorm(H_r); F_r=(SiLU(U_r W_g)⊙(U_r W_u)) W_d; H_(r+1)=H_r+F_r",flow:"每一次递归都会重新计算 Norm→Gate MatMul→Up MatMul→SwiGLU→Down MatMul→Residual，但 wgate/wup/wdown 的底层节点和优化器状态始终唯一。",practice:"循环共享参数≠复制参数。反向传播从后向前穿过展开的计算步骤，同一权重接收所有轮次贡献。这正是 recurrent-depth Transformer 与独立堆叠 Transformer 在参数规模上的区别。",trap:"训练时的完整 BPTT、截断 BPTT、梯度 checkpointing 和动态深度采样有不同内存/梯度定义。本例只实现完整两至三轮 BPTT；不将运行时 loop 次数误报成模型参数数。",exercise:"对 wgate 做中心有限差分，再和 JS VJP / PyTorch Autograd 比较；断点恢复后的第三步参数必须完全一致。",epistemic:"FULL_BPTT_SHARED_PARAMETERS_NUMERICALLY_VERIFIABLE",papers:[{id:"ouro",title:"Ouro looped LM",url:OURO},{id:"ut",title:"Universal Transformers",url:UT}]}),
 loop_adaptive:Object.freeze({title:"Adaptive depth / MoR · 本课未实现的前沿分支",formula:"fixed r=2,3 (this lesson); ACT/MoR: learned/token-specific stopping",flow:"本课只改变全序列固定循环次数。Ouro 还有学习式退出分布，Mixture-of-Recursions 则以路由器为 Token 分配深度，甚至在更高循环轮次采用有效 Token 的注意力图。",practice:"固定权重递归是后续 ACT、熵正则退出和逐 Token 动态路由的前提，但它们是不同的状态转移和优化目标，需要单独的循环权重、归一化与推理终止规则。",trap:"没有实际实现退出分布、动态 Token 筛选、预期计算成本项、路由梯度或 KV 复用，绝不可据此宣称 Ouro/ MoR 等价计算或加速。",exercise:"以同一输入对比 2/3 轮 logits；新增 ACT 方案前须定义每 Token 退出概率和补足到 1 的残余概率。",epistemic:"RESEARCH_NEXT_STEPS_NOT_IMPLEMENTED",papers:[{id:"ouro",title:"Ouro reference",url:OURO},{id:"mor",title:"MoR official repository",url:MOR}]}),

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
 if(!graph||typeof graph.title!=="string"||!graph.nodes?.some(n=>n.id===id))return null;
 if(graph.title.startsWith("任务 43 · Loop Residual Scaling")){
   if(/^r[01]_(attn_delta|ffn_delta|attn_res|state)$/.test(id)||id==="loss")return "loop_residual";
   if(["Wq","Wk","Wv","Wo","Wup","Wgate","Wdown"].includes(id))return "loop_residual";
   return null;
 }
 if(graph.title.includes("任务 41")&&graph.title.includes("Looped Transformer")){
   if(id==="loss"||/^r[0-2]_(ce|logits)$/.test(id)||/^weight[0-2]$/.test(id)||/^total[12]$/.test(id))return "loop_multidepth";
   if(/^r[0-2]_(attn_norm|q|k|v|kt|scores|scaled|mask|probs|context|proj|attn_res)$/.test(id))return "loop_causal";
   if(/^r[0-2]_(ffn_norm|gate|up|swiglu|down|state)$/.test(id))return "loop_ffn";
   if(["embedding","Wq","Wk","Wv","Wo","Wup","Wgate","Wdown","tied_head","lookup"].includes(id))return "loop_shared";
   return null;
 }
 if(graph.title.includes("Looped Transformer")){
   if(/^r[1-3]_(?:gate|up|swi|down|ffnorm|state)$/.test(id)||["wgate","wup","wdown"].includes(id))return "loop_ffn";
   if(/^r[1-3]_(?:q|k|v|qr|kr|kt|score|scale|prob|ctx|attn|attn_res)$/.test(id)||id==="mask")return "loop_causal";
   if(["loss","logits","tied_head"].includes(id))return "loop_adaptive";
   if(["embed","wq","wk","wv","wo"].includes(id)||/^r[1-3]_norm$/.test(id))return "loop_shared";
   return null;
 }
 if(!graph.title.includes("任务 38"))return null;
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
