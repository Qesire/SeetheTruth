/* Detailed explanations for learning blocks. All content is general literature/
 * framework knowledge; none implies observation of a specific model instance. */
(function(root,make){"use strict";const api=make(
 typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthBlockKnowledge=api;
})(typeof window!=="undefined"?window:{},function(L){"use strict";
const sources=Object.freeze({
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
 layernorm:["逐 token 做减均值、除标准差，真实层通常带 gamma 和 beta。","Pre-Norm/Post-Norm 放置与可训练仿射参数都影响梯度路径和模型行为。","当前教学的无仿射版本不代表所有框架默认配置。","把同一个常数加到每个通道，对比 RMSNorm 的变化。","ln,rms"],
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
 masked_softmax:["只在 mask=true 的 Key 中归一化概率，其他位置严格为 0。","PyTorch SDPA 中布尔 mask 的 True 表示允许参与，但其他 API 可能不同。","对不允许位置的 logits 乘 0 再 Softmax 不是严格遮罩。","给最高分位置设置 mask=0，检查其概率是否为零。","sdpa,softmax"],
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
  intuition:n[0],flow:n[1],practice:n[2],trap:n[3],
  // Questions and source IDs are derived from a dedicated map below.
  refs:n[4].split(","),epistemic:"公开论文/官方实现约定，不是指定模型已验证事实"});
};
return Object.freeze({get,sources,validate});
});
