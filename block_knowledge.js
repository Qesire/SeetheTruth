/* Detailed explanations for learning blocks. All content is general literature/
 * framework knowledge; none implies observation of a specific model instance. */
(function(root,make){"use strict";const api=make(
 typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthBlockKnowledge=api;
})(typeof window!=="undefined"?window:{},function(L){"use strict";
const sources=Object.freeze({
 gqa_paper:["GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints","https://aclanthology.org/2023.emnlp-main.298/"],
 tensor_views:["PyTorch tensor views / reshape / permute","https://docs.pytorch.org/docs/stable/tensor_view.html"],
 broadcasting:["PyTorch broadcasting semantics","https://docs.pytorch.org/docs/stable/notes/broadcasting.html"],
 torch_matmul:["PyTorch torch.matmul batched dimensions","https://docs.pytorch.org/docs/stable/generated/torch.matmul.html"],
 adamw:["Decoupled Weight Decay Regularization","https://arxiv.org/abs/1711.05101"],
 pytorch:["PyTorch Parameter","https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html"],
 qwen3:["Qwen3 Technical Report","https://arxiv.org/abs/2505.09388"],
 gemma3:["Gemma 3 Technical Report","https://arxiv.org/abs/2503.19786"],
 deepseek3:["DeepSeek-V3 Technical Report","https://arxiv.org/abs/2412.19437"],
 deepseek3code:["DeepSeek-V3 official Gate.forward","https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py"],
 mixtral:["Mixtral of Experts","https://arxiv.org/abs/2401.04088"],
 deepseekmath:["DeepSeekMath: GRPO","https://arxiv.org/abs/2402.03300"],
 ppo:["Proximal Policy Optimization Algorithms","https://arxiv.org/abs/1707.06347"],
 deepseekr1:["DeepSeek-R1: Incentivizing Reasoning","https://arxiv.org/abs/2501.12948"],
 deepseekv2:["DeepSeek-V2: MLA / DeepSeekMoE","https://arxiv.org/abs/2405.04434"],
 softcap:["Gemma 2 Technical Report","https://arxiv.org/abs/2408.00118"],
 switch:["Switch Transformers","https://arxiv.org/abs/2101.03961"],
 stopgrad:["PyTorch detach API","https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html"],

 transformer:["Attention Is All You Need","https://arxiv.org/abs/1706.03762"],
 glu:["GLU Variants Improve Transformer","https://arxiv.org/abs/2002.05202"],
 rms:["RMSNorm paper","https://arxiv.org/abs/1910.07467"],
 ln:["PyTorch LayerNorm","https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html"],
 gelu:["PyTorch GELU, exact versus tanh","https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU"],
 sdpa:["PyTorch scaled_dot_product_attention","https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"],
 gqa:["Grouped-Query Attention (EMNLP 2023)","https://aclanthology.org/2023.emnlp-main.298/"],
 rope:["RoFormer (RoPE)","https://arxiv.org/abs/2104.09864"],
 flash:["FlashAttention and IO awareness","https://arxiv.org/abs/2205.14135"],
 cache:["Hugging Face KV cache strategies","https://huggingface.co/docs/transformers/main/kv_cache"],
 dropout:["PyTorch Dropout","https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout"],
 ce:["PyTorch CrossEntropyLoss","https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"],
 kl:["PyTorch KLDivLoss","https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html"],
 softmax:["PyTorch Softmax","https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html"],
 linear:["PyTorch Linear","https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html"],
 embed:["PyTorch Embedding","https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html"],
 matmul:["PyTorch matmul","https://docs.pytorch.org/docs/stable/generated/torch.matmul.html"],
 topk:["PyTorch topk","https://docs.pytorch.org/docs/stable/generated/torch.topk.html"],
 argmax:["PyTorch argmax","https://docs.pytorch.org/docs/stable/generated/torch.argmax.html"],
 norm:["PyTorch F.normalize","https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html"],
 softplus:["PyTorch Softplus","https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html"],
 mse:["PyTorch MSELoss","https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html"]
});
/* Each entry: expanded compute flow; industry-known nuance; bad inference; check; source ids. */
const notes={
 nd_input:["显式生成轴次序 [B,H,T,D] 的 4D 合成张量，按行主序储存。形状如 [1,2,3,4] 表示一个批次、两个头、三个时间坐标、每个四维通道。","常见 Transformer 计算会把批次、头、Token 与通道作为独立轴；不同实现实际轴顺序可能不同。","轴名仅是当前课程约定，形状数值不意味着真实模型参数。全部数值有 512 元素上限。","独立改变 H 与 T，看 MatMul 是否受内维约束。","transformer,tensor_views"],
 nd_permute:["Permute 按 π 指定的新轴索引取旧轴位置；例如 [B,H,T,D] → [B,T,H,D] 对应 π=(0,2,1,3)。数据地址意义改变，而不是随意改标签。","PyTorch permute/view 的 stride 与可能的连续化区别非常重要；本教学内核执行真实值重排，未模拟真实 stride 或物理显存。","轴值必须构成完整的 0、1、2、3 置换；reshape 不等于 permute。","设置 0,2,1,3，再反向设置 0,2,1,3，检查输出与输入逐元素一致。","tensor_views"],
 nd_reshape:["Reshape 在行主序下保留每个元素的线性顺序，把数量不变的元素重新组织到 [B,H,T,D]。","真实模型中常把合并特征维重新解释为 Head × HeadDim，但此前必须保证维度积可整除。","reshape 不移动 Head/Token 的逻辑轴；和 transpose 不可交换。","把 [6,4] 改为 [1,2,3,4]，再 Flatten，比较一维展开次序。","tensor_views"],
 nd_flatten:["把 [B,H,T,D] 依次展开为 [B·H·T,D]，各元素仍保持行主序。","常见批处理会 flatten leading dimensions 交给二维矩阵算子；之后要恢复维度边界。","不能推断被合并的 B/H/T 原始边界，它们需要作为显式元数据或由另一个 reshape 提供。","先 Flatten 接传统 2D MatMul，验证从四轴模块进入旧积木是否成功。","tensor_views"],
 nd_expand4d:["为 [T,D] 的二维矩阵增添 B=1,H=1 的前导轴，得到 [1,1,T,D]，内容没有重复。","广播和批处理需要理解长度为 1 的轴；singleton 轴与真正的多头数据不同。","Expand 此处实际上只是有界 reshape 插入两个 singleton 轴，不是任意重复数据。","把旧 2D Input 接 Expand，之后接四轴 Broadcast Add。","tensor_views,broadcasting"],
 nd_broadcast_add:["两个张量先右对齐维度，相等或为 1 的轴可以广播，逐元素求和；例如 [1,2,3,4]+[1,1,3,4]。","广播是 PyTorch/NumPy 张量运算常识，在多头 Attention 的 Head 轴添加共享 bias 时尤为常见。","不匹配的非 singleton 维度不能相加；不会自动转置或推断正确的轴语义。","用 B=1,H=1 的 Bias 加到 B=1,H=2 的 Score，检查两个 Head 都收到正确值。","broadcasting,transformer"],
 nd_reduce_sum:["给定轴 axis 的元素求和，输出 keepdim=1 的张量。例：沿 T 求和 [1,2,3,4]→[1,2,1,4]。","按 Token 聚合、按 Head 聚合、按通道统计的科学意义不同；必须明确 reduce 哪一轴。","Sum 不等于 Mean；若要平均须再显式 Scale。本积木固定 keepdim 以方便继续广播。","在 Token 轴求和，把输入每个 Token 的同通道数值手算相加。","tensor_views"],
 nd_repeat_kv:["GQA 让多个 Query Head 共享同一个 KV Head，设 Hq/Hkv=r，教学张量复制遵守 out[b,h,t,d]=in[b,floor(h/r),t,d]。MQA 是 Hkv=1 的特例。","Ainslie 等人的 GQA (EMNLP 2023) 证明在 Query Heads 与 KV Heads 数量之间设置分组，可用较少 KV Head 实现接近多头 Attention 的模型质量。PyTorch SDPA 的 GQA 条件是 Hq%Hkv==0 且 K/V Head 数相同。","这是一种可微的显式教学展开；实际高效 GQA 内核无需物理复制 KV Cache。反向时一个 KV Head 的梯度必须累加全部共享的 Query Head。","先用 Hq=4,Hkv=2 检查 Head 0/1 共享 KV0，再改成 Hkv=1 比较 MQA；使用反向模式观察 K/V 梯度汇合。","gqa_paper,sdpa"],
 nd_mse:["接受形状完全相同的四轴预测与固定目标，计算全部元素的 mean((prediction-target)^2) 并返回 [1,1] 标量，给多维计算图提供精确损失起点。","训练中损失的选择影响梯度数学含义；该教学标量只验证自动微分，不是自回归语言模型的 token Cross-Entropy。","不隐式广播目标，也不自动引入可训练参数；nd_input 来源只能接收局部 VJP，不相当于完成参数训练。","将 GQA Context 接 prediction，把独立四轴教学源接 target，然后逐个扰动 Q/K/V 值检查有限差分。","gqa_paper,tensor_views"],
 nd_scale:["在每个张量坐标执行 y=factor·x。标准 Attention 常采用 factor=1/√Dk，将 QKᵀ 的点积尺度归一化。","Transformer Scaled Dot-Product Attention 的缩放来自 Key 通道维大小，缩放的位置在 Softmax 之前；数值内核不替用户自动推断 Dk。","factor 是当前积木参数，不代表实际模型配置或测得的 GPU 性能；其他训练网络可以合法使用不同缩放。","先用 Dk=4、factor=0.5，再用 factor=1 对比注意力分布的尖锐程度。","transformer,torch_matmul"],
 nd_softmax_last:["给定任意 [B,H,Tq,Tk] 四轴分数张量，在固定 B、H 和 Tq 后，仅对最后 Tk 轴稳定 Softmax：exp(x-max)/Σexp(x-max)。","PyTorch Scaled Dot-Product Attention 同样沿 Key 维归一化；每个 Head、每条 Query 独立形成注意力概率行。","Softmax 既不是逐元素 sigmoid，也不能跨不同 Head 归一化；对所有 Mask 位置无效的行必须特殊处理。","查看 Head0 与 Head1 两种切片，检查每条 Query 的所有 Key 概率之和为 1。","transformer,sdpa"],
 nd_causal_mask:["将未来 Key 的 score 设为有限大负值 -1e9；未来定义为 keyStart+k > queryStart+q。该输出之后才接 Softmax。","自回归生成采用因果可见性；缓存解码中 Tq 可与 Tk 不同，因此要分清 Query 与 Key 的绝对位置起点。","采用有限大负值是 Float64 教学近似，并非存储负无穷；若第一条 Query 没有有效 Key 则拒绝运行，不能返回误导性的均匀概率。","Tq=Tk=3、offset=0 时第一条 Query 只有 Key0 可见；更改绝对偏移模拟缓存解码。","transformer,sdpa,cache"],
 nd_batched_matmul:["计算 A[B,H,M,K]·B[B,H,K,N]，沿 K 轴累加，仅对 B/H 的 singleton 轴允许广播。","Scaled dot-product Attention 的 QKᵀ 和 PV 可以分解为两个批量 MatMul，中间还需缩放、Mask 与 Softmax。","该积木不自动执行 Kᵀ，也不冒充完整 Attention；Q/K 轴如何排列由使用者显式决定。","对两组 Head 独立手算一次点积，检查不能把不同 Head 错混在一起。","torch_matmul,transformer"],

 parameter:["这块独立矩阵的节点 ID 是参数身份，多个 MatMul 可以从其同一个输出端口取值并累加权重梯度；params.values、optimizer.{t,m,v} 随学习工程 JSON 一起保存。","对应常见 PyTorch Parameter 与 AdamW optimizer.state 的关键区别：数值张量和一阶/二阶动量属于训练状态，节点位置或渲染顺序不是参数身份。","本课程仅用微型 Float64 CPU 张量表达优化的状态转换；不能把单步 Loss 下降等同于模型泛化或真实 GPU 训练。","先用两个 MatMul 共用该节点，改变一条连线后检查梯度变化；执行两步 AdamW，导出 JSON 再导入检查参数与动量保留。","adamw,pytorch"],
 response_token_reshape:["把 G×T 条展平 token 记录 [G*T,1] 恢复成回答-Token 矩阵 [G,T]。输出行 g、列 t 与原第 g*T+t 行严格同位。","PPO/GRPO 按采样回答及生成 token 组织旧/新策略 log probabilities；训练代码通常还需 attention/completion masks 和 padding 对齐。","这只是把布局显式化，不会在 Token 之间执行求和，不会恢复隐藏的对话或采样信息。","用 [0,1,2,3] 四个标量重排为两回答各两 Token，验证原次序。","deepseekmath,ppo"],
 response_advantage_broadcast:["把 [G,1] 的组内回答优势 A[g] 复制到每个有效 token 的位置；完整目标的 Mask 在后续积木处理。","DeepSeekMath GRPO 把同题多回答的 reward 相对优势用于采样回答的 token 级 surrogate，无独立 critic value baseline。","这里 A 按回答常量，不是 PPO 的 token-level GAE，更不能把不同 prompt 的 group 混为同一组。","使两个回答优势一正一负，观察广播后每行相同而行间不同。","deepseekmath,deepseekr1"],
 policy_importance_ratio:["输入同一 token 的新策略与采样时旧策略 logprob，计算 exp(logπθ−logπold)。比率定义依赖固定的已生成 token 动作。","PPO/GRPO 以旧策略的生成样本为条件，对新策略做重要性校正；不能把参考模型概率误用成 old policy 分母。","old/reference 是角色不同的两套策略，ratio 本身不等于优势目标或 KL 惩罚。数值示例拒绝过大 logp 差异。","令 current=old 验证 r=1，并分别施加正负偏移检查 r 的变化方向。","ppo,deepseekmath"],
 ppo_clipped_surrogate:["逐位置分别计算 rA 与 clip(r,1−epsilon,1+epsilon)A，取 min 构成 PPO clip 保守策略目标；负优势不能直接采用 max 替代。","PPO 原论文提出 clipped surrogate 限制偏离旧策略的过大改进；GRPO 继承类似形式，但优势来源是组内奖励而非 value model。","这是要最大化的 surrogate，优化器通常最小化其负号。剪裁不保证输出分布必然满足单一 KL 界。","同一 r=1.5 分别测试 A=+1 与 A=−1 的目标，检查 clip 后的分段行为。","ppo,deepseekmath"],
 sampled_reference_kl:["d=logπref(a)−logπθ(a)，逐采样 Token 计算 exp(d)−d−1；在真实有限值下非负，当前等于参考时为零。","DeepSeekMath/DeepSeek-R1 的 GRPO 可使用参考策略 KL 惩罚估计控制偏离基础模型；不同后训练变体可能省略或改变系数。","这是在被采样 Token 上的 KL 惩罚估计，不是严格对完整词表的 Σp log(p/q)。不能把 KL(ref||current) 和估计器逐值画等号。","令 d=0、±0.5，观察 KL 非负并比较不同符号的不对称幅度。","deepseekmath,deepseekr1"],
 completion_length_mask:["输出 [G,T] 二值有效标记，每行在实际生成长度内为 1，超出部分为 0，使 padding token 在 loss 里不贡献。","LLM PPO/GRPO 的 response token mask 决定归约口径；长度不同的回答若不处理，会改变优化目标的隐式长度权重。","本教学只参数化至多两个回答，不是 tokenizer/attention_mask，也不会在真实 rollout 自动识别 EOS。","分别设置第一、第二条回答的有效 token 长度为 2 与 1，观察 Mask 和最终 loss。","deepseekmath,ppo"],
 masked_grpo_objective:["对每个回答只对有效 token 计算 (surrogate−beta·KL) 均值，然后对同组回答均值，输出一个最大化的策略目标标量。","原始 GRPO 一种标准表达按回答内有效 token 归约，再在一组回答之间归约；近年的框架可能采用不同 normalization 或无 KL 变体。","这是最大化 J 而不是最小化损失；假如训练框架要最小化 loss，应显式接负号积木。Mask 不能整行为零。","构造长短两回答同值 token surrogate，改变 padding 数但不改变 valid 平均，验证结果不变。","deepseekmath,deepseekr1,ppo"],

 cache_write_segment:["前一个 cache 快照 C[P,D] 与新片段 S[N,D] 组成新的不可变快照：仅在 start_pos..start_pos+N 写入新行，其余已缓存行原样保存。模拟真实缓存写入的前后状态，但不产生隐式跨图生命周期。","DeepSeek-V3 官方 MLA optimized 分支在当前 start_pos:end_pos 写入 kv_cache 与 pe_cache，分别保存 normalized KV latent 和已旋转的位置 Key；两个缓存宽度不同。","严格区分 KV cache 的容量、已写入长度、当前合法 end_pos。不能误以为缓存的是完整 K/V，也不能把 raw position Key 当作已经施加 RoPE。","分别设置 start_pos=2 与 start_pos=1，尝试追加和合法覆盖；确认输入缓存不变，旧 Token 数据未被破坏。","deepseek3,deepseek3code,deepseekv2"],
 cache_read_prefix:["从当前快照读取前 end_pos 行以供本步 Attention 使用；当包含当前 Token 的 K/V 后，读出行数是历史 Token 加当前步新写入 Token。","DeepSeek-V3 MLA optimized 实现从 kv_cache[:end_pos] 与 pe_cache[:end_pos] 计算注意力分数，不等于直接访问全容量缓冲区；因果 Mask 处理未完成片段的未来位。","缓存前缀本身不附带因果 Mask；在单次批量 Prefill 中仅 prefix 读取不足以禁止查询看到同一片段里的未来 Key。","尝试 length>end_pos，确认不允许读取；改变 prefix 长度观察 Attention 的 Score 维。","deepseek3,deepseek3code"],
 query_weight_absorb:["若 Key 的内容部分 K_nope=C·W_K，其中 C[S,R]、W_K[R,Dk]，则 Q[T,Dk]·K_nope^T=(Q·W_K^T)·C^T。权重吸收后的 Q_abs[T,R] 可直接和 cached C 计算内容 Score。","DeepSeek-V3 MLA optimized 分支先将 W_K 转移到 Q_nope 一侧计算低秩 Query，避免每个 decode 步对全部历史 latent 重建非位置 K。","吸收矩阵乘法是精确实数代数等价，但不代表低精度累计顺序完全相等；已旋转的位置分支必须单独处理，不能把 RoPE 随意移过 W_K。","使用 Q、C、W_K 先走显式 K 重建路径，再走 Query 吸收路径，要求两套 Content Score 每项差异在浮点容差内。","deepseek3,deepseek3code,deepseekv2"],

 dispatch_routed_tokens:["把 X[T,D] 的每行按路由宽度 K 复制为 X_selected[T·K,D]；新行索引 n=t·K+k。专家 ID 由独立的 indices[T,K] 给出。","Mixtral 与 DeepSeek 等稀疏 MoE 都是先选专家再把相应 Token 送到专家网络；真实实现往往还包含按 expert 排序、负载限制或 All-to-All 通信。","复制 Token 不等于执行专家；不要把被选择的专家参数与门控系数混在这个节点里。","对 T=3,K=2，检查原始第 1 行是否成为新矩阵第 2、3 行；修改 indices 不应改变复制的 X 数值。","deepseek3,deepseek3code,mixtral"],
 expert_linear_bank:["将多个专家的权重 W_e[D,H] 按 e·D+d 展平成 W[E·D,H]；对 route n=t·K+k 使用 indices[t,k]=e 指向的独立子矩阵，计算 Y[n,h]=Σ_d X[n,d] W[e·D+d,h]。","DeepSeek-V3 官方 Expert 含 w1/w2/w3 三套 Linear，经过 SiLU(w1(x))⊙w3(x) 与 w2 输出；Mixtral 专家也有 token-dependent 的 gated FFN。","权重按专家区分但同一专家跨 Token 共享；其梯度必须把全部命中它的 Token/路由贡献相加。这里没有 bias 与设备通信，也没有模型真实权重。","把所有专家的 W 设不同，将 indices 从 0 改成 1，输入相同时输出应不同；对重复命中专家的两次路线推导权重梯度求和。","deepseek3,deepseek3code,mixtral"],
 routed_weighted_reduce:["将 V[T·K,D] 按 token-major 顺序恢复为 T 组，每组 K 行，用门控 G[T,K] 加权求和：Y[t,d]=Σ_k G[t,k]·V[t·K+k,d]。","DeepSeek-V3 和 Mixtral 的 routed expert 汇合都需要 per-token gated reduction；DeepSeek 还叠加不经 Router 的 shared-expert 支路。","权重不必自动等于概率，其合法性来自门控上游；这种归约必须与 Dispatch 排列一致，不会重排任意打包后的专家输出。","把权重设为 [1,0]、[0,1] 或 [0.2,0.8]，检查导数 dY/dV、dY/dG 和重排敏感性。","deepseek3,deepseek3code,switch"],

 gather_routed_bank:["输入专家池 E[Experts,D] 与每个 token 的路由索引 I[T,K]，沿 token-major 顺序生成 Y[t·K+k,:]=E[I[t,k],:]。","对应 MoE 的选择/分派前置操作；实际专家通常对路由 token 做不同的可训练 FFN，此处仅展示静态专家向量的 Gather。","这个算子不等同执行真实专家，输出 [T·K,D] 而不是 [T,D]，必须另用门控加权和规约恢复 token 维度。","令三个 token 分别选两个专家，手算第 4 行来自哪个专家索引。","deepseek3,deepseek3code"],

 router_gate_weights:["先按离散专家 indices[T,K] 从原始 scores[T,E] 收集每个被选中的 score；identity 模式接收上游 sigmoid affinity；sigmoid 模式接收未归一化 logits，softmax 模式仅在选中集合归一化；可选择归一化开关。","DeepSeek-V3 用附加 bias 参与专家选择，但门控权重由原始 affinity 提取；它和传统 selected-softmax 是不同的门控配方。","排序索引不可微；原始分数与加入路由 bias 的选择分数不能无条件混同，也不能假设所有 MoE 使用 sigmoid。","用高 bias 改变选中专家，同时保持原始 logits 不变，检查相应 gate 是从哪一路计算的。","deepseek3,deepseek3code,switch"],
 expert_bank_mix:["令 I[T,K] 为专家索引，G[T,K] 为每个 token 的门控值，E[Experts,D] 为演示用静态专家向量；输出 Y[T,D]=sum_k G[T,K]×E[I,D]。","对应 MoE combine 的加权回填结构；真实 Experts 必须是逐 token 的 FFN 或其他子网络，对所选 token 分别执行。","这不是 DeepSeek/Mixtral 的完整专家计算；Bank 可表达路由与加权数学，但无法反映专家参数、MLP 激活、容量和通信。","设两个专家向量互相正交，改变 gates 并手算输出；再加上 Shared Expert 的 Add 分支。","deepseek3,switch"],


 low_rank_project:["先执行 X[T,D]·Wdown[D,R] 得 C[T,R]，再执行 C[T,R]·Wup[R,O] 得 Y[T,O]，中间低秩激活仍可展开。","DeepSeek MLA 的 KV latent 压缩与多个 Up Projection 共享同一缓存 latent；LoRA 也是低秩路径，但训练身份和位置不同。","一个低秩投影不是完整 MLA；必须再表达 K/V、独立带位置 RoPE 的 K 支路及 Attention。","画出压缩 latent 分别通向 K 与 V 两条矩阵计算的分叉，验证维度和共享。","deepseek3,deepseekv2"],
 partial_rope:["输入每行仅前 r 个通道以相邻偶数对执行位置旋转，后续 D−r 个通道逐元素保持不变。","MLA 将内容向量与解耦的 RoPE 位置向量分开；并非将部分 RoPE 积木放到任何网络都自动获得缓存压缩。","RoPE 保持每对坐标的欧氏范数，但不同 token 位置的点积可能改变；实际配对布局依实现。","在两行、六通道输入上令 rotaryDim=2，检验其余四维保持原样。","deepseek3,rope"],
 row_bias_add:["将路由分数 S[T,E] 与专家偏置 B[1,E] 沿 token 轴广播相加，输出形状仍为 [T,E]。","DeepSeek-V3 的负载均衡采用路由选择的附加偏置，并根据训练时负载状态进行更新；不把 bias 混进原始门控权重。","加偏置改变排序，但不代表真实模型中 sigmoid 门控直接替换成 softmax；也不等于 optimizer bias。","将第二个专家的选择 bias 提高，比较原始门控 logits 和最终 Top-k 选择。","deepseek3"],
 group_topk_mask:["按专家索引连续等分为组；组评分是组内最高的两个分数之和（小组可能只有一个），保留得分最高的若干组。","DeepSeek-V3 采用 group-limited routing 约束路由集合；具体组评分和 bias 使用应对照真实实现检查。","这里只做组过滤；后面需要 masked Top-k 才完成专家选择，输出 0/1 不代表专家概率。","对四个专家分两组，构造一组单一极高分但另一组综合分更高的情况。","deepseek3"],
 masked_router_topk:["先使用二值 mask 删除不允许的专家，然后稳定按原始分数排序选出 k 个专家下标。","可拼接 Group-Limited Routing 与其他稀疏路由器；真实模型中的 gate weight 可能为 sigmoid 而非演示的 selected softmax。","被 Mask 的专家不参与排序，且每行允许的候选不足 k 时必须报错而不是偷偷回退全局 Top-k。","令所有高分专家被 mask，只保留低分专家并检查输出索引。","deepseek3,switch"],
 log_softmax:["沿类别轴以最大值平移计算 logsumexp，然后逐元素做 logits 减法，避免在极小概率处数值下溢。","语言模型 CE、DPO 及 PPO token-level policy log probability 常以 log-domain 计算，并另外处理 mask 与 reduction。","LogSoftmax 不是采样器，不携带选中 token id，也不等于整个序列的概率。","用正负千量级 logits，验证逐行 exp(logprob) 的和在浮点精度下约为 1。","ce,softmax"],
 selected_logprob:["对每个 token 行先做 logsumexp，再按整数 label 选定单个动作的 log probability，输出 [T,1]。","偏好优化和策略梯度需要新旧/参考策略对指定动作的 log probability；序列聚合还要独立加 token mask。","此积木不生成动作，不隐含 rollout，也不执行 PPO clipping 或 DPO 的偏好对数比目标。","比较同一 logits 的 CrossEntropy 单样本损失，验证 selected log-prob 与其为相反数。","deepseekmath,ce"],
 group_advantage:["每行是一组同问题候选结果的 reward，先减组均值，再除以 sqrt(组总体方差+epsilon)；常量行得到全零。","DeepSeekMath GRPO 使用组内相对奖励替代单独价值网络的 baseline；本教学公式固定一套方差/epsilon 约定。","组内标准化不等于完整 GRPO；还需要策略比率、截断、参考策略 KL 与 token reduction。","设置两组不同奖励尺度，比较标准化后正负符号、均值和同奖退化。","deepseekmath"],

 gather_rows:["依据每个整数索引选择源矩阵的一整行；重复索引形成多个下游读者，不修改原张量。","Embedding 查表与稀疏专家 token dispatch 都会用到 gather 类操作，但其轴与批量组织不同。","Gather 不等于自动赋予概率权重，也不能用输出 shape 猜测源轴。","用 indices=[2,0,2] 收集三行，检查首尾两行是否相等。","embed,topk"],
 scatter_add:["对每个源行按整数目标位置执行加法；相同目标索引的贡献必须累加。","Embedding 稀疏梯度、MoE expert outputs 回填和一些并行规约都依赖 scatter/add 的数学语义。","重复目的索引时不能覆盖旧值；并行浮点加法的顺序可能影响最低位。","把两行送到相同目标位置，检验输出是否为向量和。","embed,matmul"],
 row_logsumexp:["计算 m=max(x)，随后得到 m+log(sum(exp(x-m)))，避免直接 exp(很大值) 上溢。","Softmax 和 CrossEntropy 在数值稳定实现中常使用 log-sum-exp；论文与框架可能融合多个操作。","它不是单独 Softmax：输出是逐行标量而非概率矩阵。","设置 [1000,1001] 与 [-1000,-999]，验证结果仍有限。","ce,softmax"],
 repeat_kv_rows:["把每个 KV head 对应向量重复到同组的多个 Query Head；在二维切片里以行复制演示。","GQA 采用更少 KV head 服务更多 query heads；在真实实现中既可能物理复制也可能通过索引视图广播。","不能将 head 复制当作 token 复制；反向若有多个读者，KV 梯度需沿重复分支求和。","两个 KV head、每个复制两次，预测前向输出行数与反向汇合位置。","gqa"],
 qk_norm:["沿每个 head 的特征维计算均方根并归一化，输入输出维度不变；这是 Q/K 投影后的计算而不是 attention score 的归一化。","Qwen3 等模型使用 QK normalization；参数缩放、RoPE 顺序以及应用轴应以具体配置和源码为准。","不能从 QK Norm 推断整个模型使用 LayerNorm，也不能推断所有 Attention 头共享一个 gamma。","把它放在 MatMul(Q,K^T) 前，与仅对 logits 缩放的温度积木对比。","qwen3,gemma3"],
 sliding_window_mask:["逐个查询位置 i 建立可见集合 {j|max(0,i-w+1)≤j≤i}，再由 Masked Softmax 把不可见项置零。","Gemma 3 使用局部/全局注意力交错；窗口长度和 cache 位置语义与模型实现相关。","滑动窗口不是硬裁剪掉整张 KV Cache，也不表示模型不存在全局层。","将窗口设为 1、2 与 T，观察允许的键数与 Masked Softmax 的效果。","gemma3"],
 logit_softcap:["对每个 score 计算 c·tanh(score/c)，在大幅值区域压缩分数而保留平滑性。","Gemma 2 描述过 attention logits soft-capping；它与 QK Norm 是不同的机制，也不能推断 Gemma 3 一样采用该形式。","Softcap 不等于 clamp；也不保证损失或注意力熵总是单调改善。","构造极大正负 logits，比对原始 Softmax、softcap 后 Softmax。","softcap,gemma3"],
 router_topk:["按每个 token 的专家 score 降序选择 k 个索引，将其输出为可用于 Gather/Dispatch 的整数张量。","DeepSeek-V3 的路由还涉及动态 bias、分组限制和分数归一化；演示的 selected-softmax 是教学选项，非完整复刻。","离散 index 不直接对 logits 求导；路由选择与被选专家的连续权重梯度要分开。","修改 k 和 router logits，观察专家选择、并列 tie-break、权重分配的变化。","deepseek3,switch"],
 weighted_sum:["按查询行 t 的权重向量与 values 的行逐项乘积再沿 K 聚合；输出 [T,D]。","可表达 Attention 的 P·V；真实 MoE 组合则需要先为不同 token Gather 对应专家结果。","权重不自动归一化；此积木不说明概率、路由或者因果性。","用两行 weights 连接三行 values，手算一个输出坐标。","transformer,sdpa"],
 detach_demo:["前向保持张量值不变，反向时把该边的局部 VJP 定义为零，从而隔离一条梯度路径。","现代训练的 target network、teacher distillation 和策略基线经常涉及显式 detach 或 no_grad，具体边界由算法确定。","Stop Gradient 并不停止其他仍可微分路径的梯度，也不必然冻结这个参数在所有路径上的更新。","建立双分支图，只在一支使用 detach，追踪共享参数的梯度贡献。","stopgrad"],
 residual_scale:["在支路上计算 αX 后再与主支路 Add，从而将支路幅度显式可组合地控制。","残差缩放可能与初始化、深度以及规范化位置联动，不能仅凭一个标量积木声称复现完整稳定化方法。","Residual Scale 是可微常数乘法，不等同可训练门控或 stochastic depth。","设置 α 为 0、1 和 -1 并连接 Add，比较表示与局部梯度。","transformer"],
 input:["先区分 token 轴与通道轴，再沿图检查哪个模块真正混合 token，哪个只混合通道。","真实表征通常还包括 Batch、Head、position、mask、dtype；二维仅是教学切片。","不能由同一 shape 推断张量内容语义，也不能把演示数值称为真实 Embedding。","更改输入 D，预测下游哪些算子会因为维度不匹配失效。","transformer,embed"],
 linear:["每个 token 共享同一组权重，沿输入通道求和形成新的输出特征。","QKV 可将多个 Linear 融合为一次投影后分拆；是否有 bias、权重转置和融合由源码决定。","单独看到 Linear 名称，不能断言它是 Q/K/V 或非线性 FFN。","分别构建三条投影支路，核对不同的消费者。","linear,transformer"],
 rmsnorm:["每个 token 先计算 mean(x²)，再取平方根调尺度；不减去均值。","RMSNorm 可少一次中心化，但实际速度仍取决于融合、数据类型与硬件。","RMSNorm 不强制均值为零，当前教学也没有可训练 gamma。","给所有通道同时加 10，对比 LayerNorm 的差异。","rms,ln"],
 silu:["对每个值先 Sigmoid 再乘自身，输出可为负且不是门控概率。","实际 SwiGLU 可能将 SiLU 与乘法融合成同一计算路径，不应按节点数计算 kernel 数量。","连续门值和结构化稀疏不同，不能因为接近零而推定有加速。","对 x=-2,0,2 比较 SiLU、ReLU 与 Sigmoid。","glu"],
 add:["将两条形状相同的数据流按坐标相加；只有输入旁路返回本身才是残差。","Pre-Norm 与 Post-Norm 的位置影响主残差路径与训练稳定性。","普通 Add 并不天然拥有残差的全部优化性质。","从输入另拉旁路接到 Add，解释恒等梯度路径。","transformer,ln"],
 mul:["每个坐标的两条信息逐项相乘，区别于矩阵乘法的通道求和。","门控通常为连续非零值，普通密集 GEMM 不会自动跳过小 gate。","逐元素乘法不等于真正结构化剪枝，也不会天然减少模型计算。","用 Sigmoid 的输出当作软 gate，再思考真稀疏还需要什么。","glu,matmul"],
 matmul:["A[T,K] 与 B[K,D] 沿 K 收缩，保留的 T、D 构成输出；多维情况另有批次轴。","转置可能仅改变 stride，数学运算一次也未必对应一个独立 CUDA kernel。","共享维数字相等不证明两边的轴有相同物理含义。","手算 2×3 与 3×5 两个矩阵，指出收缩轴。","matmul"],
 softmax:["每行先减最大值，再 exp 并除以总和，使概率非负且和为 1。","FlashAttention 在线归一化，可能避免完整注意力概率矩阵的显式物化。","Softmax 权重不是严格因果解释，也不会自动屏蔽未来。","整体给一行 logits 加 100，再比较输出。","softmax,flash"],
 attention:["先 QKᵀ/√Dk 得匹配分数，再在 Key 轴归一化，最后以概率对 V 加权。","FlashAttention 是 IO-aware 精确 Attention；PyTorch SDPA 的 mask、dropout 和后端选择需要看实际配置。","此积木仅是无 mask 的单头版本，不等于所有 Transformer Attention。","固定 V 改 Q/K，再固定 Q/K 改 V，比较路由与内容的角色。","transformer,sdpa,flash"],
 gqa:["多个 Q head 按组复用少量 K/V head，每个 head 计算 QK 与 PV 并拼接。","减少 KV head 通常减小解码 cache 存储与访存，但并不保证任意硬件延迟下降。","GQA 不是 Token 剪枝，也不意味着 KV cache 状态已自动持久更新。","改变 Hq/Hkv 并保持单 head 宽度，推导 cache 字节数趋势。","gqa,cache,sdpa"],
 mse:["prediction 与 target 逐元素作差，平方之后再聚合为标量。","扩散目标可能是 epsilon、v 或 x0，必须看实际 target 构造。","一次 Loss 前向不是梯度反传，更不证明训练收敛或感知质量。","比较误差 1 与 10 在 MSE 和 MAE 中的不同权重。","mse"],
 token_ids:["词表把文本切成离散 token，每个 token 再编码为整数 ID，之后才查表。","不同 tokenizer 对空格、汉字、特殊符号可能有不同分段。","ID 差 1 不代表语义相近，token 也未必就是一个字或一个词。","讨论同一句话换 tokenizer 后为什么 token 个数可能变化。","embed,transformer"],
 embedding:["用 Token ID 对表 E[V,D] 进行按行索引，输出 [T,D]，不是执行全部词表 GEMM。","一些架构共享输入和输出词表权重，但必须由实际配置确认。","连续的 ID 不保证对应的特征向量在空间中相近。","挑两个 ID 比较读取哪些表行，区分查表与训练更新。","embed"],
 pos_encoding:["生成不同频率的 sin/cos 位置向量，再加到输入隐藏特征。","真实模型可能采用可学习位置编码、相对偏置或 RoPE 等不同机制。","加性 PE 与 Q/K 上的 RoPE 旋转不是同一个算子。","输入内容不变仅改变位置，解释变化出现在何处。","transformer,rope"],
 transpose2d:["X[T,D] 交换为 Xᵀ[D,T]，索引重新排列供下一步矩阵收缩。","视图形式的 transpose 可不复制物理数据，contiguous 约束可能另行造成拷贝。","变换后的第一个轴未必仍代表 token，不能只按轴序号推测含义。","解释 Q[T,D] 乘 Kᵀ[D,T] 的两个输出轴。","matmul,sdpa"],
 slice_cols:["选取连续特征通道，保持 token 轴、减少可见的特征宽度。","合并 QKV Linear 后常通过切片得到分支；连续切片与任意索引在访存上不同。","单个通道不一定对应一个可解释语义神经元。","将宽度 8 的输出切成两组，说明分支用途依赖哪里。","linear"],
 concat_cols:["Token 对齐后沿特征轴拼接，数据重新排列但未执行可学习混合。","多头合并常涉及 Head 轴 reshape/permute，顺序必须与输出权重匹配。","Concat 本身不是输出投影，也不保证不同 token 已正确对齐。","拼接两条 4 通道支路再接 Linear，指出哪步才发生融合。","transformer,matmul"],
 repeat_rows:["重复已有 token 行以增大序列长度，值没有新来源。","GQA repeat_kv 是 KV Head 轴重用，不是重复 Token 轴，后端可以不实际拷贝。","复制输入不能等同新 token 生成或 KV cache 追加。","对比 Token repeat 与 KV Head 复用在数学轴上的区别。","gqa,cache"],
 row_mean:["沿每个 token 的特征维求平均，其他轴保留；Reduce 的轴语义决定功能。","Loss 和统计常需忽略 padding，实际 reduction 的分母应按有效元素数量确认。","RowMean 不是 Batch Mean，更不会令结果自动归一化到概率。","在固定均值下改变特征宽度，比较 Sum 与 Mean。","ln"],
 row_sum:["沿一个维度累加，不除元素数；可能是进一步做归一化的中间步骤。","mask 后 Sum 需区分有效 token 和 padding；实际框架有多种 reduction 选项。","求和本身不是概率，输出也未必落在固定范围内。","把每行元素翻倍，观察 RowSum 如何变化。","softmax,ln"],
 row_var:["先消去行均值，平方差后对 D 求平均；本课程 correction=0。","PyTorch LayerNorm 使用 biased population variance，不应误用 correction=1。","常数向量的方差为零，但 RMS 可以是非零。","对 [2,2,2] 分别计算 Var 与 RMS²。","ln,rms"],
 layernorm:["逐 token 做减均值、除标准差，真实层通常带 gamma 和 beta。","Pre-Norm/Post-Norm 放置与可训练仿射参数都影响梯度路径和模型行为。","当前教学的无仿射版本不代表所有框架默认配置，实际 gamma/beta 必须核对模型参数。","把同一个常数加到每个通道，对比 RMSNorm 的变化。","ln,rms"],
 l2norm:["沿特征维计算向量长度，再用 max(norm,epsilon) 稳定缩放。","检索和对比学习常对两侧特征先 L2 Normalization 再点积。","零向量不会突然变成长度 1，且与 RMSNorm 的 sqrt(D) 约定不同。","将非零向量乘 3，再观察归一化后的方向。","norm,rms"],
 relu:["对负值输出严格 0、正值直通，在零处按框架约定处理导数。","即便出现数值 0，普通密集 GEMM 也不会自动跳过对应权重乘加。","大量零激活不等于可无损永久删除通道。","对 [-1,0,1] 比较 ReLU 和 SiLU。","glu"],
 gelu_tanh:["用 tanh 近似高斯 CDF 的平滑门控，与 ReLU 的硬阈值不同。","PyTorch GELU 默认 approximate=none；本积木明确采用 tanh 近似。","精确 GELU 和 tanh GELU 不能被称为完全同一数值运算。","选择相同 x 比较 GELU(tanh) 与 SiLU。","gelu,glu"],
 sigmoid:["对各元素独立映射到 0~1，形成一个软门控强度。","Sigmoid gate 的饱和可能影响梯度，但并不自动造成真实跳算。","它不像 Softmax 那样要求一行多个分量和为 1。","给 [2,2] 同时做 Sigmoid 和 Softmax，比较和。","glu,softmax"],
 tanh:["平滑映射到 (-1,1)，满足奇函数对称性。","较大绝对值会进入饱和区，训练影响要结合前后层判断。","Tanh 不是类别概率计算，输出和没有约束。","比较输入 1 和 8 时增量变化，并解释数值饱和。","gelu"],
 softplus:["对大正数近似线性，对负数平滑接近零。","稳定式可用 max(x,0)+log1p(exp(-abs(x))) 避免大正数 exp 溢出。","Softplus 结果可以大于 1，不是 Sigmoid。","比较 exp(1000) 与稳定 Softplus 的数值行为。","softplus"],
 subtract:["对齐 A、B 后按坐标相减，可构造预测残差或差分。","真实框架可能广播不同 shape；教学合同故意拒绝隐式广播。","形状一致不代表 token 次序或物理语义一致。","交换 A、B，解释结果改变而误差绝对值可不变。","matmul"],
 divide:["按坐标相除，值域前提要求分母不为零。","真实框架可能传播 Inf/NaN，本课程拒绝零分母以便定位。","仅形状合法不能确定除法数值一定合法。","设计一个包含零的分母，区分推演与数值实验结果。","norm"],
 scale:["对所有坐标乘固定系数；放在 Softmax 前后语义不同。","Attention 的缩放通常融合进 score kernel，而非独立设备操作。","非线性一般不允许任意交换缩放次序，交换后 Softmax 输出分布也可能不同。","把 Scale 从 Softmax 前移到后，检查概率和。","sdpa,softmax"],
 bias:["给所有坐标加常数；此教学不是每通道独立的可训练 bias 向量。","有些架构省去 Linear bias；偏置的实际语义需结合参数维度。","对全部 logits 加同一数不改变 Softmax；对各类加不同数则可能改变。","给一行 logits 每项加 100，比较 Softmax。","linear,softmax"],
 square:["每元素平方会放大大数贡献，但尚未执行 reduction 或均方误差。","Norm/Loss 的低精度累计可能需要更高精度来避免溢出。","Square 不是 MSE，必须先构造差再平均。","搭 Subtract→Square→Mean，看每一步的数学职责。","mse,rms"],
 swiglu:["双上投影 gate/up，SiLU(gate) 逐元素调制 up，下投影再混合通道。","gate/up 有时被融合为单 GEMM；中间宽度与普通 FFN 的比较需匹配参数量。","门控连续值不是硬稀疏 mask，也不保证自动加速。","拼出 Linear×2→SwiGLU→Linear，分析两路为何不能合并成一个任意 Linear。","glu,transformer"],
 geglu:["双上投影 gate/up，使用 GELU(gate) 而非 SiLU 进行逐元素门控。","GELU exact 与 tanh 近似需通过实际模型配置判断。","GEGLU 不是只有一个投影的 GELU MLP。","固定 gate/up，比较 GEGLU 与 SwiGLU 的输出。","glu,gelu"],
 causal_mask:["给时间序列设定过去可见、未来不可见的约束，是任务定义。","PyTorch SDPA 非方阵 is_causal 有自己的对齐偏置；KV decode 还需考虑 cache_position。","不能把 Tq=1、Tk=4 的 decode 直接画成左上角方阵规则。","在单 Query 读取三条缓存 Key 的场景推导允许位置。","sdpa,cache"],
 masked_softmax:["只在 mask=true 的 Key 中归一化概率，其他位置严格为 0。","PyTorch SDPA 中布尔 mask 的 True 表示允许参与，但其他 API 可能不同。","把被屏蔽的 logits 简单乘零再 Softmax 仍会把零分数纳入分母，因此不是严格遮罩。","给最高分位置设置 mask=0，检查其概率是否为零。","sdpa,softmax"],
 rope2d:["对 Q/K 特征成对正交旋转，位置差影响其相互内积。","真实 RoPE 可能是 adjacent-pair 或 split-half 编排，也有 position scaling。","范数保持不代表不同位置内积保持，更不代表无限长度外推。","对同一个向量采用两个 position，比较范数与内积。","rope,cache"],
 temperature:["把 logits 除以正温度，随后 Softmax 的集中度发生变化。","实际解码还可能有重复惩罚、Top-k/p、EOS 和采样，执行顺序不一定相同。","温度本身不会产生 token，也不会改变正比例缩放下的 ArgMax 顺序。","比较温度 0.5 和 2 的 Softmax 与 ArgMax。","softmax,ce"],
 argmax:["沿类别轴选取最大分数所在下标，属于逐步贪心。","局部最优 token 不保证生成整段文本的全局最高联合概率。","并列情况必须明确 tie-break，不能声称所有 API 稳定一致。","构造并列最高分，核对该教学规则的下标选择。","argmax"],
 topk:["先选出 k 个高分候选；采样或 Router dispatch 是后续动作。","PyTorch Top-k 对并列分数下标可能不稳定；教学稳定 tie-break 只是契约。","Top-k 本身没有在概率中抽样，也没有执行 MoE 专家计算。","Top-k 返回三个值后，描述还缺少哪些步骤才能生成 token。","topk,gqa"],
 dropout_demo:["训练期随机遮挡元素并按 1/(1-p) 调尺度，推理评估期为恒等。","PyTorch nn.Dropout 训练时每次前向重新抽样；SDPA 的 dropout_p 需显式归零。","固定 seed 的课程 mask 不等于真实连续训练随机过程。","对比 training/eval，并解释保持期望的缩放系数。","dropout,sdpa"],
 cross_entropy:["从目标类别对应 logits 的 log 概率计算损失，而不是先 ArgMax。","语言模型训练常需要 logits/labels 时间 shift、ignore_index 和有效 token 归一化。","教学只支持离散 ID，PyTorch CE 也可以处理 soft targets；前向标量不等于梯度更新。","故意让标签错位一 token，分析模型优化到的错误任务。","ce,transformer"],
 mae_loss:["逐元素取绝对误差，再对所有有效元素求平均。","MAE 相比 MSE 对离群值放大较弱，但何者更好取决于目标分布。","一次 MAE 前向不会更新参数，也不必然代表感知质量。","比较误差为 1 和 10 的贡献比例。","mse"],
 kl_prob:["KL(P||Q) 按 P 加权对数概率比，通常有明确方向且不对称。","PyTorch KLDivLoss 使用 logprob 输入，默认 mean 与严格 KL 的 batchmean 语义不同。","不能交换 P/Q 也不能把未归一化 logits 当概率直接代入。","对 P=[1,0] 和 Q=[.5,.5] 正反计算 KL。","kl,ce"]
};
function validate(){
 const keys=Object.keys(L.defs);if(Object.keys(notes).length!==keys.length)throw Error("knowledge coverage mismatch");
 for(const id of keys){
  const card=notes[id];if(!card||card.length!==5)throw Error("missing fields: "+id);
  for(let i=0;i<4;i++)if(typeof card[i]!=="string"||card[i].length<18)throw Error("shallow card "+id);
  for(const ref of card[4].split(","))if(!sources[ref])throw Error("unsupported source "+id+":"+ref);
 }
 return {cards:keys.length,references:Object.keys(sources).length};
}
validate();
const get=id=>{if(!Object.hasOwn(notes,id))return null;
 const n=notes[id],d=L.defs[id];
 return Object.freeze({kind:id,title:d.title,formula:d.formula,why:d.why,
  intuition:d.why+(d.why.length<18?" 需要检查这个操作实际改变了哪一条张量语义轴。":""),flow:n[0],practice:n[1],trap:n[2],exercise:n[3],
  refs:n[4].split(","),epistemic:"公开论文/官方实现约定，不是指定模型已验证事实"});
};
return Object.freeze({get,sources,validate});
});
