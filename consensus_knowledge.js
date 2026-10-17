/* SeetheTruth mechanism atlas v1.
 * Reference-grounded *general* design knowledge. Never substitutes for source,
 * config, runtime, tensor shape or quality evidence from a specific model.
 * No external data is fetched by this script.
 */
(function(root, factory) {
  const atlas = factory();
  if (typeof module === "object" && module.exports) module.exports = atlas;
  if (root) root.SeetheTruthConsensus = atlas;
})(typeof window !== "undefined" ? window : null, function() {
"use strict";
const references = Object.freeze({
  vaswani: {title:"Attention Is All You Need", url:"https://arxiv.org/abs/1706.03762", year:2017},
  gqa: {title:"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", url:"https://aclanthology.org/2023.emnlp-main.298/", year:2023},
  rope: {title:"RoFormer: Enhanced Transformer with Rotary Position Embedding", url:"https://arxiv.org/abs/2104.09864", year:2021},
  glu: {title:"GLU Variants Improve Transformer", url:"https://arxiv.org/abs/2002.05202", year:2020},
  ln: {title:"Layer Normalization", url:"https://arxiv.org/abs/1607.06450", year:2016},
  rms: {title:"Root Mean Square Layer Normalization", url:"https://arxiv.org/abs/1910.07467", year:2019},
  moe: {title:"Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", url:"https://www.jmlr.org/papers/v23/21-0998.html", year:2022},
  dit: {title:"Scalable Diffusion Models with Transformers", url:"https://arxiv.org/abs/2212.09748", year:2023},
  flash: {title:"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", url:"https://arxiv.org/abs/2205.14135", year:2022},
  ppo: {title:"Proximal Policy Optimization Algorithms", url:"https://arxiv.org/abs/1707.06347", year:2017},
  grpo: {title:"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", url:"https://arxiv.org/abs/2402.03300", year:2024},
  adamw: {title:"Decoupled Weight Decay Regularization", url:"https://arxiv.org/abs/1711.05101", year:2019}
});
/*
 * Layering:
 * mechanism = algebraic effect of this operation
 * objective = intended downstream effect (often conditional, NOT guaranteed)
 * chain = hypothesized mechanism-to-effect pathway
 * assumptions = necessary conditions
 * measure = discriminating observations (not "quality guaranteed")
 * contrast = distinguish from similarly named variants
 * epistemic = definition / empirical-in-literature / engineering-practice
 */
const entries = [
{id:"linear", title:"线性投影 / Linear", family:"映射", formula:"Y = X W + b (按行向量约定)", mechanism:"将输入通道重新组合到可学习的输出子空间；单独的线性层不引入非线性。", objective:"根据后续任务生成查询、内容特征、路由分数、分类 logits 或模型中间状态。", chain:["通道加权求和","得到任务所需坐标表示","交给下游非线性或损失"], assumptions:"W 的方向/转置取决于实际库；投影用途必须由后续计算流确认。", measure:"核对 W 的形状、输入输出维度、偏置及其直接使用者；不能从 Linear 名字推定 Q/K/V。", contrast:"Linear 只是数学原语，不等于 Attention，也不意味着模型精度必然上升。", epistemic:"mechanism", refs:["vaswani"]},
{id:"scaled_attention", title:"Scaled Dot-Product Attention", family:"Attention", formula:"A = softmax(QKᵀ/√dₖ + M); O = A V", mechanism:"先由 Q、K 得到位置间的匹配权重，再用该权重对 V 内容进行加权聚合。", objective:"让 token/patch 的表示条件性地读取其他位置的信息，建立跨位置依赖。", chain:["QKᵀ 生成匹配分数","Softmax 归一化路由权重","AV 汇聚跨位置内容","输出进入残差流影响预测"], assumptions:"遮罩 M、缩放因子、头布局以及 attention dropout 因实现和模式而变化；attention 权重不应被直接当成因果解释。", measure:"分别对 Q/K 路由与 V 内容做扰动；比较分数、输出和最终任务指标，验证影响链。", contrast:"源代码可能使用 SDPA/FlashAttention 一个融合算子，不意味着运行了三个独立 CUDA kernel。", epistemic:"mechanism", refs:["vaswani","flash"]},
{id:"gqa", title:"Grouped-Query Attention / GQA", family:"推理效率", formula:"H_q > H_kv; 每组多个 Q 头共享同一组 K/V 头", mechanism:"用较少的 KV 头服务较多的 Query 头，使 KV cache 存储量和解码期的 KV 读写需求通常下降。", objective:"在降低解码期 KV 访存和缓存占用的同时，尽量保持多头注意力质量。", chain:["降低 KV 头数","减少 KV cache 字节与读写","可能缓解 decode 的内存带宽压力","实际吞吐由 kernel、序列长度与硬件决定"], assumptions:"要求实现中确有 Q/KV 头分组映射；不能仅因调用 repeat_kv 就证明系统延迟下降。", measure:"记录 Q/KV 头数、每 token cache bytes、decode latency 与任务质量，匹配相同 batch/seq/device。", contrast:"GQA 不等于按 Token 剪枝；从 MHA 换成 GQA 的质量/速度权衡不是数学上必然。", epistemic:"empirical-in-literature", refs:["gqa"]},
{id:"rope", title:"Rotary Position Embedding / RoPE", family:"位置编码", formula:"qᵢ′ = R(i)qᵢ; kⱼ′ = R(j)kⱼ; (qᵢ′)ᵀkⱼ′ = qᵢᵀR(j−i)kⱼ", mechanism:"通过成对坐标旋转把位置信息注入 Q/K 内积，分数显式依赖相对位置差。", objective:"让注意力能够区分相对位置信息，支持位置相关的序列建模。", chain:["按位置旋转 Q/K","改变 QKᵀ 的相位关系","使注意力匹配携带相对位置信息","影响序列依赖建模"], assumptions:"该等式针对配对正交旋转约定；不同 RoPE scaling、partial rotary、轴排列可能改变具体形式。", measure:"固定 Q/K 内容、改变相对位移，检查内积模式；确认旋转轴、频率、偏移及 cache position。", contrast:"RoPE 不自动保证无限长度外推，也不是直接给 V 加位置向量。", epistemic:"mechanism", refs:["rope"]},
{id:"swiglu", title:"SwiGLU / Gated FFN", family:"FFN", formula:"FFN(X) = (SiLU(XW_g) ⊙ (XW_u)) W_d", mechanism:"对同一输入形成门控和内容两条投影，逐通道非线性调节中间特征再映射回残差维。", objective:"增强每个 token 的非线性通道变换和可学习特征选择能力。", chain:["两个上投影产生门控与内容","SiLU×内容形成输入依赖门控","下投影混合中间通道","为后续层提供表达能力"], assumptions:"只有源码确有双分支、SiLU 和 Hadamard 乘法时才是 SwiGLU；普通 GELU MLP 不应被标成 SwiGLU。", measure:"对比 gate、up 与 down 的通道统计和消融质量；分离门控支路误差与投影支路误差。", contrast:"门控值并非严格二值稀疏掩码，不能据此推断可无损跳过非零通道计算。", epistemic:"empirical-in-literature", refs:["glu"]},
{id:"gelu_mlp", title:"GELU MLP / FFN", family:"FFN", formula:"FFN(X) = GELU(XW₁+b₁) W₂+b₂", mechanism:"第一次投影扩展或重排中间特征，GELU 引入逐元素非线性，再由第二次投影汇聚回隐藏空间。", objective:"提供注意力之后的 token 内非线性通道混合。", chain:["投影至中间特征","非线性激活","投影回主残差维","更新各 token 表示"], assumptions:"适用两层 GELU MLP；不同中间宽度、残差位置或 dropout 影响实际表现。", measure:"检查激活分布、两次 GEMM 维度、输出范数及失真向后传播。", contrast:"与 SwiGLU 的三矩阵双上投影结构不同。", epistemic:"mechanism", refs:["vaswani","glu"]},
{id:"layernorm", title:"LayerNorm", family:"归一化", formula:"y = γ ⊙ (x−μ)/√(σ²+ε) + β", mechanism:"对每个样本的指定归一化维计算均值/方差，再做可学习仿射变换。", objective:"控制隐藏表示的尺度与数值分布，支持稳定训练。", chain:["中心化/缩放隐藏状态","改变后续运算数值条件","影响梯度和优化稳定性"], assumptions:"归一化轴、epsilon、仿射参数、Pre/Post-Norm 位置必须与真实代码一致。", measure:"对比归一化前后均值、RMS、梯度尺度、训练曲线。", contrast:"RMSNorm 不执行均值中心化；二者不能混为同一公式。", epistemic:"empirical-in-literature", refs:["ln"]},
{id:"rmsnorm", title:"RMSNorm", family:"归一化", formula:"y = γ ⊙ x / √(mean(x²)+ε)", mechanism:"只使用均方根控制激活幅值，不消去输入均值。", objective:"减少归一化运算复杂度，同时维持对激活缩放的鲁棒性。", chain:["RMS 估计","逐通道缩放","控制进入 Attention/FFN 的激活幅值"], assumptions:"不同框架会在 epsilon 的放置及 dtype 上有细节差异。", measure:"核对是否减均值、epsilon、计算 dtype、归一化前后 RMS 与下游质量。", contrast:"“RMSNorm 更快”是实现和硬件相关的经验结论，不是任何部署下必然加速。", epistemic:"empirical-in-literature", refs:["rms"]},
{id:"moe", title:"Mixture-of-Experts Top-k Routing", family:"条件计算", formula:"p = softmax(XW_r); S = TopK(p,k); y = Σ_{e∈S} p̃_e Expert_e(X)", mechanism:"由 Router 为每个 token 选择部分专家，以条件计算代替全部专家同时激活。", objective:"在每个 token 只激活一部分参数时提高模型总参数容量。", chain:["路由分数→Top-k 专家","Dispatch token 到所选专家","专家非线性变换","加权聚合回残差流"], assumptions:"容量限制、负载均衡、路由归一化、专家共享及通信方案会改变行为。", measure:"专家负载分布、路由熵、token 丢弃率、通讯量、质量与端到端吞吐。", contrast:"稀疏激活并不保证 wall-clock 加速；专家失衡和 all-to-all 通信可能成为瓶颈。", epistemic:"engineering-practice", refs:["moe"]},
{id:"adaln_zero", title:"AdaLN-Zero / DiT Conditioning", family:"条件生成", formula:"m(x,c) = LN(x) ⊙ (1+s(c)) + b(c); x′ = x + g(c) ⊙ F(m(x,c))", mechanism:"由 timestep/class 等条件产生 shift、scale 和门控，控制子层变换及其注入残差的强度。", objective:"使生成网络的去噪/预测行为依赖时间步及外部条件，并通过零初始化门控形成接近恒等的初始残差路径。", chain:["条件 c 经 MLP 生成调制参数","调制规范化的隐藏状态","门控 Attention/FFN 残差","改变连续预测场"], assumptions:"这是经典 DiT adaLN-Zero 形式；Wan/其他视频 DiT 的精确条件构造、门控和初始化需看源码。", measure:"跨 timestep 追踪 shift/scale/gate 的分布；消融调制分量，观察预测残差和生成质量。", contrast:"条件依赖不意味着模型一定更稳定或更好；需比较一致数据/训练设定。", epistemic:"empirical-in-literature", refs:["dit"]},
{id:"flash_attention", title:"FlashAttention / IO-aware Attention", family:"算子效率", formula:"softmax(QKᵀ/√d + M)V（数学目标不变，以分块在线归一化实现）", mechanism:"分块计算并通过在线 softmax 减少中间 N×N 注意力矩阵的 HBM 读写和显式物化。", objective:"在合适输入与硬件上降低 Attention 的内存 IO 和运行时间。", chain:["分块数据复用","减少 HBM 流量","减小 attention 中间激活存储需求","可能提升实际训练/推理吞吐"], assumptions:"具体速度依赖 kernel 后端、head dimension、mask、序列长度、GPU 和工作负载。", measure:"比较相同语义与数值容差下的 kernel 时间、HBM bytes、峰值分配与吞吐。", contrast:"数学分解里仍有 QKᵀ 和 AV，不代表 CUDA 实际执行为两个独立 kernel。", epistemic:"empirical-in-literature", refs:["flash"]},
{id:"cross_entropy", title:"Token Cross-Entropy / NLL", family:"训练目标", formula:"L = − Σ_t log pθ(y_t | context_t)（按 mask / reduction 定义）", mechanism:"最大化正确目标 token 的条件对数概率，将输出 logits 与监督标签建立梯度联系。", objective:"学习条件 token 分布；用于语言模型预训练或监督微调等任务。", chain:["模型 logits→log-softmax","计算目标 token NLL","梯度传播到可训练参数","优化器迭代改善训练目标"], assumptions:"teacher forcing、shift、padding mask、label smoothing、reduction 均非所有实现相同。", measure:"核对 shift/label mask、逐 token NLL、grad finite 与 held-out perplexity。", contrast:"降低训练交叉熵不自动保证指令遵循、事实正确或人类偏好对齐。", epistemic:"mechanism", refs:["vaswani"]},
{id:"adamw", title:"AdamW / Decoupled Weight Decay", family:"优化", formula:"θ ← (1−ηλ)θ − η·m̂/(√v̂+ε)", mechanism:"对梯度一阶/二阶动量做自适应更新，并将权重衰减从梯度损失项解耦。", objective:"推进损失优化并通过独立权重衰减控制参数规模。", chain:["Loss.backward 生成梯度","动量状态估计更新方向","独立 weight decay + 参数更新","改变下次模型预测"], assumptions:"bias correction、epsilon 位置、parameter groups、学习率调度、梯度裁剪会改变精确更新。", measure:"检查参数组 decay 配置、更新范数、梯度状态和学习率轨迹。", contrast:"不能仅从 optimizer.step() 证明用的是 AdamW。", epistemic:"empirical-in-literature", refs:["adamw"]},
{id:"ppo", title:"PPO Clipped Surrogate", family:"RL 优化", formula:"L_clip = E[min(r_t A_t, clip(r_t,1−ε,1+ε) A_t)]", mechanism:"用新旧策略概率比与优势函数构造替代目标，裁剪部分过大策略更新的收益。", objective:"在策略改进与单次更新幅度之间取得经验上的稳定性折中。", chain:["旧/新策略 logprob","比率 r 和优势 A","剪切替代目标","梯度更新策略"], assumptions:"优势估计、价值头、KL/熵项、clip 变体和样本来源依具体 PPO 实现。", measure:"监控 approximate KL、clip fraction、优势分布、奖励、策略熵与有效样本量。", contrast:"Clipping 不是严格保证策略不会偏移；经典 PPO 和 LLM PPO 实现可能不同。", epistemic:"empirical-in-literature", refs:["ppo"]},
{id:"grpo", title:"GRPO / Group-relative Advantage", family:"RL 优化", formula:"A_i ≈ (R_i−mean_group(R))/(std_group(R)+ε)；策略目标含重要性比率与裁剪项", mechanism:"对同一问题采样一组候选，以组内奖励构造相对优势，更新高相对奖励响应的概率。", objective:"在序列/语言模型 RL 中减少对显式价值网络的依赖并利用组内比较信号。", chain:["同一 prompt 采样多候选","获取组内奖励","标准化相对优势","策略比率与 KL/裁剪约束","优化行为分布"], assumptions:"奖励粒度、组内标准差为零、KL 计算方式、采样策略及长度偏差需要单独确认。", measure:"组内奖励方差、zero-variance groups、响应长度、KL、clip fraction、最终任务指标。", contrast:"这里是 DeepSeekMath 的代表性方法描述，不能覆盖全部 GRPO 变种。", epistemic:"empirical-in-literature", refs:["grpo","ppo"]}
];
const byId = Object.freeze(Object.fromEntries(entries.map(x=>[x.id,Object.freeze(x)])));
function resolve({id="",node=null,workflow="model",preset="",focus="",rlAlgo="grpo",graphMeta=null}={}) {
  const suffix=String(id).toLowerCase();
  const op=(node&&typeof node==="object"?String(node.callee||node.aten_op||"").toLowerCase():"");
  const ctor=(node&&typeof node==="object"?String(node.module_constructor||"").toLowerCase():"");
  const analysis=String(graphMeta?.analysis||"");
  const sourceGraph=Boolean(graphMeta)&&analysis!=="";
  const entry=String(graphMeta?.entry||"");
  let match=null, basis=null;
  function assign(key,why){if(byId[key]){match=byId[key];basis=why;}}
  if(workflow!=="model") {
    if(workflow==="rl"&&(suffix.includes("advantage")||suffix.includes("clip")||suffix.includes("ratio")||suffix==="rl_policy")) assign(rlAlgo==="ppo"?"ppo":"grpo","workflow_template_only");
    else if(workflow==="train"&&suffix==="wf_loss"&&preset!=="dit")assign("cross_entropy","workflow_template_only");
    else if(workflow==="train"&&(suffix.includes("optimizer")||suffix==="wf_update"))assign("adamw","workflow_example_optimizer");
  } else if(sourceGraph) {
    // AST or runtime graph: never infer a semantic role from arbitrary node label alone.
    if(/(^|\.)scaled_dot_product_attention$/.test(op)) assign("scaled_attention","explicit_function");
    else if(/(^|\.)flash_attn_func$/.test(op)) assign("flash_attention","explicit_function");
    else if(/(^|\.)cross_entropy$/.test(op)) assign("cross_entropy","explicit_function");
    else if(/(^|\.)rms_norm$/.test(op)||ctor==="rmsnorm")assign("rmsnorm","explicit_function_or_constructor");
    else if(ctor==="layernorm")assign("layernorm","explicit_constructor");
    else if(/(^|\.)apply_rotary_pos_emb$/.test(op))assign("rope","explicit_function");
    else if(/(^|\.)topk$/.test(op)&&/moe/i.test(entry))assign("moe","source_entry_context");
    else if(ctor==="linear"||/(^|\.)linear$/.test(op))assign("linear","explicit_function_or_constructor");
    // Unknown call / method / dynamic dispatch: no conjectural design role.
  } else {
    // Deliberate teaching presets: canonical structures are allowed, with label TEMPLATE.
    const known={
      "attention":"scaled_attention","qk":"scaled_attention","av":"scaled_attention","mx_score":"scaled_attention","mx_av":"scaled_attention",
      "ro pe":"rope","rope":"rope","qrope":"rope","krope":"rope",
      "ffn":preset==="gpt"?"gelu_mlp":preset==="moe"?"moe":"swiglu",
      "gate":"swiglu","mx_gmm":"swiglu","mx_umm":"swiglu","mx_dmm":"swiglu",
      "mx_w1mm":"gelu_mlp","mx_w2mm":"gelu_mlp",
      "norm":"layernorm","norm1":preset==="gpt"||preset==="dit"?"layernorm":"rmsnorm",
      "norm2":preset==="gpt"||preset==="dit"?"layernorm":"rmsnorm",
      "adaln":"adaln_zero","mod":"adaln_zero","mx_mod":"adaln_zero","condition":"adaln_zero",
      "router":"moe","topk":"moe","mx_topk":"moe","mx_routermm":"moe",
      "mx_qmm":"linear","mx_kmm":"linear","mx_vmm":"linear","mx_wo":"linear",
      "cache":"gqa","mx_cache":"gqa","kexpand":"gqa","vexpand":"gqa"
    };
    const lookup=known[suffix] || (suffix.startsWith("mx_")&&suffix.endsWith("mm")?"linear":null);
    if(lookup)assign(lookup,"teaching_template");
    if(preset==="dit"&&suffix==="ffn")assign("gelu_mlp","teaching_template");
    if(preset==="gqa"&&suffix==="attention")assign("gqa","teaching_template");
    if(preset==="moe"&&suffix==="ffn")assign("moe","teaching_template");
    if((suffix==="logits"||suffix==="mx_scale")&&preset!=="dit")assign("scaled_attention","teaching_template");
  }
  if(!match)return null;
  return {entry:match, matchBasis:basis,
          applicationEvidence:sourceGraph?"literature_only_not_model_validated":"template_only_not_model_validated"};
}
function validate() {
  const ids=new Set();
  for(const e of entries){
    if(ids.has(e.id))throw Error("duplicate consensus ID: "+e.id);
    ids.add(e.id);
    for(const f of ["title","formula","mechanism","objective","chain","assumptions","measure","contrast","epistemic","refs"]){
      if(!e[f] || (typeof e[f]==="string"&&!e[f].trim()))throw Error("missing "+e.id+"."+f);
    }
    if(!Array.isArray(e.chain)||e.chain.length<2||!Array.isArray(e.refs)||!e.refs.length)throw Error("missing chains / references: "+e.id);
    for(const r of e.refs){if(!references[r]||!references[r].url.startsWith("https://"))throw Error("invalid reference "+r);}
  }
  return {entries:entries.length,refs:Object.keys(references).length};
}
validate();
return Object.freeze({entries:Object.freeze(entries),references,byId,resolve,validate});
});
