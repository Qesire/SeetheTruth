/* Source-bounded why/why-not architectural design choices for actual block kinds.
 * An architecture paper supplies a hypothesis and concrete tradeoffs, not proof
 * that an arbitrary edited graph is identical to a released checkpoint.
 */
(function(root,factory){
 const api=factory();
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthArchitectureWhy=api;
})(typeof window!=="undefined"?window:{},function(){"use strict";
const Q="https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py";
const NEXT="https://huggingface.co/docs/transformers/main/model_doc/qwen3_next";
const DS="https://arxiv.org/abs/2412.19437";
const GQA="https://arxiv.org/abs/2305.13245";
const ROPE="https://arxiv.org/abs/2104.09864";
const SWI="https://arxiv.org/abs/2002.05202";
const FLASH="https://arxiv.org/abs/2205.14135";
const ADAMW="https://arxiv.org/abs/1711.05101";
const CASES=Object.freeze({
 qk_norm:Object.freeze({
 title:"为什么在 Q/K 上做 Norm，而不是只对 Layer 输入 RMSNorm？",
 goal:"独立控制注意力分数的 Query/Key 向量尺度，让 token-token 相关性不完全受投影范数漂移主导",
 math:"q̂=q/√(mean(q²)+ε)·γ_Q; k̂=k/√(mean(k²)+ε)·γ_K",
 alternative:"仅做输入 RMSNorm（不稳定性可能留在 Q/K 投影之后）；不使用 QK Norm；使用注意力 logit softcap 限幅",
 cost:"额外逐 Head 归一化与可训练 gamma（依架构而定）；过度控制尺度可能损伤某些幅度信号",
 failure:"注意力分数尺度和训练稳定性不存在问题、或者训练中幅度信息至关重要时，不一定有净收益",
 test:"在同一初始投影上开关 Q/K Norm；记录 attention-logit 分布、Softmax 熵、梯度范数、loss 曲线",
 sourceName:"Qwen3Attention · official Transformers implementation",source:Q,
 status:"QK Norm 使用事实来自源码；训练稳定性属于可检验的工程解释，而非保证收益"
 }),
 gqa:Object.freeze({
 title:"为什么让多个 Query Head 共用较少 KV Head？",
 goal:"减小自回归解码时每 Token 的 KV Cache 宽度与 HBM 读取需求",
 math:"H_q>H_kv, groups=H_q/H_kv; KV bytes/token ∝ 2·H_kv·head_dim·bytes",
 alternative:"MHA：更多独立 KV Head，表达更自由；MQA：全部 Q Head 共用一组 KV，缓存更小但约束更强；MLA：压缩 latent KV，数学路径不同",
 cost:"KV Head 共享限制不同 Q Head 的独立 K/V 表达；训练和 fused kernel 调度仍需适配",
 failure:"纯 Prefill 的计算瓶颈不一定来自 KV Cache；Token 长度小或内存带宽不是瓶颈时，GQA 不必然提高速度",
 test:"同 D、Hq 下对比 Hkv=Hq、Hq/2、1：真实 KV 字节数、attention 偏差、decode TPS；预填充/解码分开统计",
 sourceName:"GQA: Training Generalized Multi-Query Transformer Models",source:GQA,
 status:"头数/缓存关系为形状推导；模型质量/实际延迟必须实验"
 }),
 rope2d:Object.freeze({
 title:"为什么用旋转位置而不是直接把绝对位置加到表示中？",
 goal:"把 token 相对位移编码进 Query 与 Key 的内积相位，保留每个旋转对的 L2 范数",
 math:"〈R_i q,R_j k〉=〈q,R_(j−i) k〉 (相容配对/频率下)",
 alternative:"绝对正弦位置加法；ALiBi 对 logit 加偏置；不编码显式位置的策略",
 cost:"频率外推、长上下文尺度需要专项设计；不同布局（interleaved/rotate_half）不可只看 shape 当作相同",
 failure:"位置远外推引入相位混叠、频率和缓存 offset 不一致、或源代码维度配对被误读",
 test:"旋转前后范数应守恒；平移所有 position_id 后 QK 相对点积是否保持；有无 KV Cache 位置偏移对照",
 sourceName:"RoFormer / Rotary Position Embedding",source:ROPE,
 status:"相对相位为数学性质；长上下文质量取决于训练与具体缩放方法"
 }),
 swiglu:Object.freeze({
 title:"为什么从 ReLU/GELU FFN 改成 SwiGLU 门控 FFN？",
 goal:"让一个可学习分支调制另一分支的中间通道，提升逐 token 非线性表达的灵活度",
 math:"FFN(X)=(SiLU(XW_g)⊙XW_u)W_d",
 alternative:"ReLU/GELU 两矩阵 FFN；GeGLU；ReLU²；MoE 专家门控",
 cost:"多一组输入投影，需相应缩小中间维度才有相近 FLOPs/参数；核融合/带宽影响实际速度",
 failure:"若 width 没有按等预算公平匹配，比较混淆更大模型与门控机制；SiLU 门的饱和区和输入相关性可改变梯度",
 test:"控制相同参数/MACs 比较 ReLU FFN 与 SwiGLU 的训练质量；观察 Gate 分布、零/小幅通道和权重梯度",
 sourceName:"GLU Variants Improve Transformer",source:SWI,
 status:"数学流与门控机制可计算；优于替代激活不是全部数据/预算下的定律"
 }),
 moe:Object.freeze({
 title:"为什么用稀疏专家 MoE，而不是每 Token 通过全部 FFN？",
 goal:"让参数容量增长比每 Token 实际 FFN 算力增长更快",
 math:"Y_t=Σ_(e∈TopK(router(x_t))) gate_te·Expert_e(x_t)",
 alternative:"完全稠密 FFN；共享专家+路由专家；静态分组稀疏或 Hash Routing",
 cost:"Router、Dispatch、All-to-All、专家负载均衡以及小批量 GEMM；选择离散性可能使梯度/质量不稳定",
 failure:"专家负载严重倾斜或 Token/专家批次太小导致效率劣化；活跃参数/FLOPs 降低不一定对应 GPU 端到端加速",
 test:"同数据/FLOPs 下记录专家负载、激活比、all-to-all、kernel 形状、Token 质量并与稠密基线比较",
 sourceName:"DeepSeek-V3 technical report",source:DS,
 status:"稀疏容量扩展是研究动机；模型/硬件收益必须联合验证"
 }),
 mla:Object.freeze({
 title:"为什么用压缩 KV latent（MLA），而不是直接缓存全部 Head K/V？",
 goal:"减少长上下文解码的 KV Cache 宽度与存储带宽，同时尽量保留表达能力",
 math:"c_t=W_down h_t; K,V 由 c_t 与位置相关子空间按实际权重路径恢复/吸收",
 alternative:"标准 MHA；GQA/MQA 减少 KV Head 数；KV 量化／分页缓存",
 cost:"额外压缩/恢复投影、RoPE 分离维度与权重吸收约束；算法等价不等于任意量化等价",
 failure:"把压缩 latent 当成普通 GQA KV 导致 cache/layout 语义错误；未对照原代码就声称 MatMul 可以任意融合",
 test:"源码构建完整 cache shape 与数值路径；分别验证未吸收/已吸收前向等价及 prefill/decode 一致性",
 sourceName:"DeepSeek-V3 technical report · MLA",source:DS,
 status:"MLA 与 GQA 不可互换；实际恢复/吸收模式应依对应检查点与源码确认"
 }),
 flash:Object.freeze({
 title:"为什么 FlashAttention 不需要物化完整 T×T attention 矩阵？",
 goal:"减少 HBM 读写，让 Attention 的中间概率分块在片上保留并以在线 Softmax 精确归一化",
 math:"blockwise online Softmax maintains running max m and denominator ℓ",
 alternative:"naive QKᵀ→Softmax→PV 显式两次 GEMM；分块重算；低秩/线性注意力（改数学语义）",
 cost:"需要复杂 kernel、Tile/MMA 与因果边界管理，性能受具体 GPU、HeadDim 和 Batch 形状影响",
 failure:"缩尺教学 NumPy 分块并不证明吞吐量；错误在线归一化会导致数值偏差，显存节省也不一定改善短序列速度",
 test:"针对同 Q/K/V/Mask 比较 Flash 与 PyTorch SDPA 数值、反向、峰值 HBM/显存、不同形状吞吐量",
 sourceName:"FlashAttention: Fast and Memory-Efficient Exact Attention",source:FLASH,
 status:"IO 算法原理来源于论文；本工作台无实际 CUDA kernel 计时"
 }),
 adamw:Object.freeze({
 title:"为什么权重衰减要与梯度更新解耦？",
 goal:"避免把 Adam 的自适应梯度缩放错误地同时施加到 L2 惩罚上",
 math:"θ'=(1−ηλ)θ−η·AdamDirection(g)",
 alternative:"Adam + 耦合 L2；SGD+momentum；Lion；RMSprop",
 cost:"一个权重乘法和动量状态；衰减设置仍需与 LR、权重分组和 schedule 联合调试",
 failure:"Bias/Norm 等参数不加选择地衰减、或把 Adam L2 当作 AdamW，导致实验解释错误",
 test:"控制同一初始 θ、g、η、λ，按标量一步计算验证 Adam L2 vs AdamW 差异；再比较跨步训练",
 sourceName:"Decoupled Weight Decay Regularization",source:ADAMW,
 status:"优化器数学定义可逐步复现；具体任务性能需要真实数据"
 })
});
const KIND=Object.freeze({
 qk_norm:"qk_norm",nd_repeat_kv:"gqa",nd_rope:"rope2d",rope2d:"rope2d",swiglu:"swiglu",
 expert_linear_bank:"moe",expert_bank_mix:"moe",router_topk:"moe",masked_router_topk:"moe",
 router_gate_weights:"moe",group_topk_mask:"moe",gather_routed_bank:"moe",
 nd_cache_read:"mla",nd_cache_append:"mla",masked_softmax:"flash",nd_softmax_last:"flash",
 opt_adamw:"adamw"
});
function classify(graph,node){
 if(!node||!graph||!graph.nodes?.some(n=>n.id===node.id))return null;
 if(graph.title?.includes("Loop"))return null; // loop-specific exact evidence is in the other panel
 if(graph.title?.includes("任务 38")){
  if(["qgain","kgain","q0n","q1n","kn","q0aff","q1aff","kaff"].includes(node.id))return "qk_norm";
  if(["q0","q1","k","v","score0","score1","out0","out1","heads"].includes(node.id))return "gqa";
 }
 const k=KIND[node.kind]||null;
 if(k==="mla"&& !/mla/i.test(graph.title||"") )return null; // KV slice/append ≠ MLA
 return k;
}
const esc=x=>String(x??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function render(graph,node){
 const key=classify(graph,node),d=CASES[key];if(!d)return "";
 let s='<section class="knowledge-part" data-industry-why="'+esc(key)+'"><h3>INDUSTRY WHY / 设计权衡</h3>'+
 '<strong>'+esc(d.title)+'</strong><p><b>研究路线：</b>先识别实际计算机制和敏感变量，执行下面的对照实验，再决定是否替换模块；不要直接跳到局部参数微调。</p>';
 for(const [title,k] of [["设计目标","goal"],["等价数学操作","math"],["低成本鉴别实验","test"],
  ["业界替代方案","alternative"],["失效机制","failure"],["代价及系统权衡","cost"]])
  s+='<p><b>'+esc(title)+'：</b>'+esc(d[k])+'</p>';
 return s+'<p><strong>证据边界：</strong>'+esc(d.status)+'</p>'+
 '<a href="'+esc(d.source)+'" target="_blank" rel="noopener noreferrer">↗ '+esc(d.sourceName)+'</a></section>';
}
return Object.freeze({classify,render,cases:CASES,supported:Object.freeze(Object.keys(CASES))});
});