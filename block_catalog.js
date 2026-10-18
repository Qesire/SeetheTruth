/* Executable, bounded, deterministic two-dimensional teaching blocks.
 * A block is in the catalogue ONLY if validate() and execute() exist.
 * All matrices are explicit Float64 JS arrays, not real LLM inference.
 */
(function(root,create){
 "use strict";
 const result=create();
 if(typeof module==="object"&&module.exports)module.exports=result;
 root.SeetheTruthBlockCatalog=result;
})(typeof window!=="undefined"?window:{},function(){
 "use strict";
 const block=(title,en,category,ports,params,formula,why,lesson,caution)=>
   ({title,en,category,ports,params,formula,why,lesson,caution});
 const defs={
  token_ids:block("Token ID 序列","Token IDs","data",[],{rows:3,vocab:8,seed:1},
   "ids[t] ∈ {0,…,V−1}","生成离散 token 下标而非连续隐藏向量。","连接 Embedding 并检查索引范围。","仅生成教学用整数，不执行真实 tokenizer。"),
  embedding:block("Embedding 查表","Embedding","data",["ids"],{width:4,vocab:8},
   "X[t,d] = E[ids[t],d]","通过索引选择词表参数行，不是 One-hot 与 Linear 的真实大规模实现。","改变 vocab 验证越界检测，区分索引与特征。","词向量表是确定性教学矩阵；没有训练与真实 tokenizer。"),
  pos_encoding:block("正弦位置编码","Sinusoidal PE","position",["x"],{},
   "PE(pos,2i)=sin(pos/10000^(2i/D)); PE(pos,2i+1)=cos(…)","把绝对 token 位置信息加到连续表示上。","和 RoPE 比较：此处加位置向量；RoPE 旋转 Q/K。","使用起始位置 0，不含可学习位置参数，要求 D 为偶数。"),
  transpose2d:block("转置","Transpose","shape",["x"],{},
   "Y[j,i]=X[i,j]","交换二维矩阵轴，改变后续矩阵乘法的收缩方向。","用 MatMul 验证内维要匹配。","只支持二维，输出必须满足教学矩阵 8×16 上限。"),
  slice_cols:block("特征切片","Slice cols","shape",["x"],{start:0,width:2},
   "Y[:,j]=X[:,start+j]","从隐藏通道轴选择连续区间。","观察裁剪后残差分支是否仍形状相容。","仅支持连续列切片，起始下标从零计数。"),
  concat_cols:block("特征拼接","Concatenate","shape",["a","b"],{},
   "Y=concat(A,B,axis=D)","将同一 token 位置的两组特征合并。","和切片、门控组成更复杂 FFN。","要求行数一致，特征宽度之和最多 16。"),
  repeat_rows:block("重复 Token 维","Repeat rows","shape",["x"],{times:2},
   "Y=repeat(X,axis=T,times=r)","在 token 维复制行，帮助理解广播和重复的区别。","观察总元素数如何随 r 增长。","真实 Tensor repeat 会分配重复元素；这里仅二维矩阵。"),
  row_mean:block("每行均值","Mean last axis","reduce",["x"],{},
   "Y[t,0]=(1/D)Σ_d X[t,d]","沿特征维聚合成一个标量，不跨 token 混合。","用 Sum 对照除以 D 的效果。","轴限定为特征轴 D。"),
  row_sum:block("每行求和","Sum last axis","reduce",["x"],{},
   "Y[t,0]=Σ_d X[t,d]","展示一次沿隐藏通道的 Reduction。","结合 Mean 比较归一化因子。","输出每个 token 一列，而非整个张量单个标量。"),
  row_var:block("每行方差","Variance last axis","reduce",["x"],{},
   "v[t]=(1/D)Σ_d(x[t,d]−mean[t])²","理解 LayerNorm 使用有偏总体方差。","检查常量行的方差为零。","总体方差 correction=0；不是样本无偏估计。"),
  layernorm:block("LayerNorm","LayerNorm","norm",["x"],{epsilon:0.00001},
   "y=(x−μ)/sqrt(Var(x)+ε)","每个 token 沿全部特征去中心化再按方差标准化。","与 RMSNorm 比较：LayerNorm 会减均值。","γ=1、β=0 的无仿射版本；方差 correction=0。"),
  l2norm:block("L2 特征归一化","L2 Normalize","norm",["x"],{epsilon:0.000001},
   "y=x/max(||x||₂,ε)","把每个 token 的特征向量映射到近似单位球面。","对比 RMSNorm 的 sqrt(D) 因子。","零向量仍为零；使用 max(norm,ε) 而不是 sqrt(norm²+ε)。"),
  relu:block("ReLU","ReLU","activate",["x"],{},
   "f(x)=max(0,x)","硬阈值激活：全部负值被裁到零。","对比 SiLU/GELU 负半轴的信息传递。","在零点不可微，这里仅执行前向。"),
  gelu_tanh:block("GELU · tanh 近似","GELU tanh","activate",["x"],{},
   "0.5x[1+tanh(√(2/π)(x+0.044715x³))]","近似 Gaussian CDF 门控的平滑激活。","将负值测试与 ReLU/SiLU 对比。","显式采用 PyTorch approximate='tanh'，并非默认 erf/CDF 精确版。"),
  sigmoid:block("Sigmoid","Sigmoid","activate",["x"],{},
   "σ(x)=1/(1+exp(−x))","在 0 到 1 间形成逐元素门控系数。","测试 σ(x)+σ(−x)=1。","这不是 Softmax；各维不会归一化为和为 1。"),
  tanh:block("Tanh","Tanh","activate",["x"],{},
   "tanh(x)","对称、平滑、输出在 −1 到 +1 之间。","比较 tanh(−x) 与 −tanh(x)。","绝对值较大时梯度容易饱和，未实现反传。"),
  softplus:block("Softplus","Softplus","activate",["x"],{},
   "log(1+exp(x))","ReLU 的平滑版本；大正值接近恒等。","观察大数条件下稳定计算公式。","用稳定 log1p/exp 防止正数溢出。"),
  subtract:block("逐元素减法","Subtract","combine",["a","b"],{},
   "Y=A−B","从两个同形张量构建误差与分支差异。","检查逐 token、逐特征对齐。","本课程不自动广播，形状必须完全一致。"),
  divide:block("逐元素除法","Divide","combine",["a","b"],{},
   "Y=A/B","表达比率与元素级缩放。","构造除零反例；观察数值不稳定的根源。","零分母严格拒绝；不模拟浮点 Inf 的传播。"),
  scale:block("常数缩放","Multiply scalar","compute",["x"],{factor:0.5},
   "Y=αX","可观察 attention 的温度和残差分支的幅值变化。","令 factor=0 观察信号归零。","α 是常数参数，不是可训练权重。"),
  bias:block("常数偏置","Add scalar","compute",["x"],{bias:0.2},
   "Y=X+β","模拟偏置平移如何影响后续激活。","试图令正负激活跨越零点。","所有通道共享 β；不等于完整可训练 bias 向量。"),
  square:block("元素平方","Square","compute",["x"],{},
   "Y=X⊙X","构成能量、方差与均方误差的基础步骤。","和 Mean / Subtract 组合恢复 MSE。","不宣称平方是矩阵 GEMM。"),
  swiglu:block("SwiGLU 门控","SwiGLU","gated",["gate","up"],{},
   "Y=SiLU(G)⊙U","现代 Transformer FFN 的常见门控非线性组合。","先用两个独立 Linear 生成 G 与 U，再接下行投影。","本积木不包含 Linear 权重；输入需已经是匹配形状的投影结果。"),
  geglu:block("GEGLU 门控","GEGLU tanh","gated",["gate","up"],{},
   "Y=GELU_tanh(G)⊙U","在上投影的值上施加 GELU 门控。","比较不同门控函数产生的激活差异。","GELU 采用 tanh 近似，和默认精确形式不同。"),
  causal_mask:block("生成因果掩码","Causal Mask","attention",["x"],{},
   "M[i,j]=1 ⇔ j≤i","将当前 token 数转成严格下三角布尔可见性。","接 Masked Softmax 观察不可访问位置变成精确零。","限定自注意力、相同 Q/K 序列长度。"),
  masked_softmax:block("带 Mask Softmax","Masked Softmax","attention",["logits","mask"],{},
   "P[i,j]=softmax_{j∈allowed(i)} logits[i,j]","排除不允许的注意力位置，归一化仅在剩余 Key 上进行。","确认 Mask=0 的概率严格为零。","Mask 只能含 0/1；全屏蔽行被拒绝，避免 NaN。"),
  rope2d:block("RoPE 旋转","RoPE · 2D","position",["x"],{offset:0},
   "R_{pos}(x[2i:2i+2])","按 token 位置对每对相邻通道做正交旋转。","比较旋转前后每对通道的平方范数。","沿相邻坐标配对，需偶数特征维；模型实际 RoPE 布局须查源码。"),
  temperature:block("Logits 温度","Temperature","sampling",["x"],{temperature:1},
   "Y=logits/τ","温度改变 Softmax 分布的集中程度。","τ<1 更尖锐；结合 Softmax 观察。","不是 Sampling，也不产生随机 token。"),
  argmax:block("贪心索引","ArgMax","sampling",["x"],{},
   "id=argmax_d logits[d]","每一行选出分数最高的类别索引。","对比 Top-k 与 Softmax 的含义。","遇到相等最大值取最小列下标；不是随机采样。"),
  topk:block("Top-k 取值","Top-k values","sampling",["x"],{k:2},
   "topk(logits,k)","返回每行得分最高的 k 个值；可展开实际索引。","改变 k 看保留候选的变化。","只返回排序后的值；索引在详情展示，不产生 token 样本。"),
  dropout_demo:block("可复现 Dropout","Dropout seeded","regularize",["x"],{p:0.25,seed:1,training:true},
   "training: y=x·m/(1−p), m∼Bernoulli(1−p)","显式观测元素屏蔽和 inverted dropout 缩放。","切换训练/评估模式查看是否为恒等变换。","采用固定种子的伪随机数供教学；训练模式下同一配置每次执行相同 Mask，非真实独立调用采样。"),
  cross_entropy:block("类别交叉熵","CrossEntropy","loss",["logits","labels"],{},
   "L=mean_t[log Σ_c exp(logits[t,c])−logits[t,y_t]]","给定离散类别 ID，对每个 token 的 logits 做分类损失。","比较全对类别和错误类别的 Loss。","labels 是 [T,1] 的整数 ID，不是概率标签；无 ignore_index/label smoothing。"),
  mae_loss:block("平均绝对误差","MAE Loss","loss",["prediction","target"],{},
   "L=mean(|prediction−target|)","通过绝对误差提供对异常值不同于 MSE 的敏感度。","和 MSE 做相同输入的数值对比。","只前向，不包含反向 subgradient。"),
  kl_prob:block("KL(P ∥ Q)","KL Divergence","loss",["p","q"],{},
   "KL(P||Q)=mean_t Σ_c P[t,c] log(P[t,c]/Q[t,c])","比较两组离散概率分布，可以用于蒸馏机制学习。","利用两个 Softmax 输出构造有效 P、Q。","输入必须都是逐行和为 1 的概率，Q 不能在 P>0 处为 0；不等于 PyTorch 默认 kl_div reduction='mean'。")
  ,
  qk_norm:block("Q/K 向量归一化","QK Norm","attention",["x"],{epsilon:0.00001},"y=x/sqrt(mean(x²)+ε)","对 query/key 的每个头特征向量按 RMS 缩放；真实模型是否有可训练增益须查源码。","组合于 QKᵀ 前，与普通 RMSNorm 比较。","教学把每行视为一个 head；γ=1，没有模拟不同头的参数。"),
  sliding_window_mask:block("滑动窗口因果 Mask","Sliding Window","attention",["x"],{window:2},"M[i,j]=1 iff 0≤i−j<w","局部注意力只允许最近 w 个 token；可接 Masked Softmax。","在每行比较可访问 key 数量；改变窗口宽度。","仅单序列等长自注意力，无 sink token、global token 或 cache offset。"),
  logit_softcap:block("Logits Softcap","Tanh Softcap","attention",["x"],{cap:5},"y=c·tanh(x/c)","平滑限制 logits 幅值，改变 Softmax 概率尖锐程度。","对比 Softmax 前后数值和梯度饱和区间。","不是裁剪到固定阈值的 hard clamp。"),
  router_topk:block("Router Top-k 索引","Router Dispatch IDs","routing",["x"],{k:2},"I=TopK(logits,k); w=softmax(selected logits)","按 token 选择专家索引，显式返回稀疏路由权重。","连接专家分派；分析负载与 selected-gate normalization。","输出只有索引供教学；权重保存在内部信息中，没有执行专家计算。"),
  weighted_sum:block("加权特征汇聚","Weighted Sum","combine",["weights","values"],{},"y_t=Σ_j weights[t,j]·values[j,:]","把加权汇聚拆成独立积木，可用于 Attention 或专家输出的可视化简化。","测试权重归一化是否由上游保障。","无 batch/head 维；不是真实 MoE 的 per-token dynamic gather。"),
  detach_demo:block("停止梯度","Stop Gradient","training",["x"],{},"y=stop_gradient(x)","前向输出与输入完全一致，但反向不通过此边传播梯度。","与残差路径配合，追踪哪些边仍接收梯度。","当前计算引擎仅执行前向；此处反向边界属于符号契约。"),
  residual_scale:block("残差支路缩放","Residual Branch Scale","combine",["x"],{factor:0.5},"y=αx","单独配置某条残差支路的系数，可与 Add 组成可视化残差。","改变 α 观察输出与梯度比例。","只改变支路幅度，不代表 DeepNorm 等完整网络初始化配方。"),
  gather_routed_bank:block("多路专家向量 Gather","Top-k Expert Gather","routing",["x","indices"],{},"Y[t·K+k,:]=X[I[t,k],:]","按 Token 顺序把每个 token 的 K 个所选专家向量展平成 [T·K,D]，适配多专家路由。","连接 Top-k Router 和专家向量池，观察输出顺序和重复专家行。","这里只 Gather 静态演示向量，不执行每 token 的专家 FFN；T·K 不得超过教学张量上限。"),
  gather_rows:block("按索引收集行","Gather Rows","shape",["x","indices"],{},"Y[t,:]=X[indices[t,0],:]","按 token/专家索引收集输入矩阵行，供离散路由或检索组合。","观察重复索引如何复用同一行。","索引必须是 [T,1] 整数，输出行数受教学上限约束。"),
  scatter_add:block("按索引累加行","Scatter Add","combine",["indices","values"],{rows:3},"Y[index[t],:]+=values[t,:]","把多路贡献按目标行索引聚合，是专家输出回填和共享梯度汇合的基础。","用重复 index 检查真实加法聚合而非覆盖。","输出初始化为零，未实现指定 base tensor 上的 scatter_add。"),
  row_logsumexp:block("逐行 LogSumExp","LogSumExp","reduce",["x"],{},"y[t]=m+log Σ_d exp(x[t,d]−m)","稳定计算归一化常数，用于交叉熵、Softmax 和策略 log probability。","比较大幅 logits 的稳定性。","只沿最后的特征维归约，不包含 mask。"),

  low_rank_project:block("低秩压缩与重建","Low-rank Projection","compute",["x","down","up"],{},"C=X·W_down; Y=C·W_up","沿低秩中间维保存压缩张量，可用于 MLA KV/Q 压缩或 LoRA 参数化路径。","观察中间 C 的形状与重建结果，并和两次 MatMul 对照。","仅两次显式矩阵乘法，不含 MLA 的位置分支、KV cache、权重训练或具体模型权重。"),
  partial_rope:block("部分维度 RoPE","Partial Rotary Embedding","position",["x"],{rotaryDim:2,offset:0},"Y[:,0:r]=RoPE(X[:,0:r]); Y[:,r:]=X[:,r:]","只旋转指定偶数维子空间，其他特征不变；支持 MLA 中的 decoupled positional branch 教学。","观察旋转部分范数保持、未旋转部分严格不变。","二维切片采用相邻通道配对，不代表所有模型的真实 RoPE 轴布局。"),
  row_bias_add:block("逐通道路由 Bias","Row Broadcast Bias","routing",["scores","bias"],{},"Y[t,e]=S[t,e]+b[e]","把每个专家 bias 独立加到路由分数上，明确区分选择分数和原始门控分数。","设置大正/负 bias 观察专家排序和是否改变原始 logits。","这里仅前向；DeepSeek-V3 负载均衡的 bias 动态更新另有规则。"),
  group_topk_mask:block("分组 Top-k Mask","Group-Limited Routing","routing",["scores"],{groups:2,keepGroups:1},"group score=sum(top2 scores); allow top groups","把专家分组，每个 token 先选组，再限制可路由专家集合。","对比不分组 Top-k 与分组筛选，检查不能选到未保留组。","只展示先选组阶段；具体模型可能使用不同组评分/偏置定义。"),
  masked_router_topk:block("带 Mask 的专家 Top-k","Masked Router Top-k","routing",["scores","mask"],{k:2},"I=TopK(scores restricted by mask)","将分组约束与专家候选排序实际接起来；输出离散专家 ID。","连接 Group Top-k Mask，观察被选专家全部属于允许的组。","教学权重仅在选中 logits 上做 softmax；不直接复现 DeepSeek-V3 sigmoid gate。"),
  log_softmax:block("稳定 LogSoftmax","Log Softmax","activate",["x"],{},"log p_j=x_j−logsumexp(x)","在 log 域计算归一化概率，训练 CE、PPO/DPO log-prob 均可复用。","比较与 Softmax 后再取 log 的数值稳定性。","只沿二维矩阵最后一个维度归一化，不带 mask 或跨 token loss reduction。"),
  selected_logprob:block("目标 Token 对数概率","Selected Log Probability","training",["logits","labels"],{},"log π(y|x)=logits_y−logsumexp(logits)","用逐位置 token 的真实类别索引取 log probability，作为偏好/RL 目标的前置积木。","与 CrossEntropy 每 token 值比较，应当符号相反。","仅计算已给定动作 ID 的前向 log-prob；未进行采样、序列 mask 或奖励计算。"),
  group_advantage:block("组内奖励标准化","Group-relative Advantage","training",["rewards"],{epsilon:0.00001},"A=(R−mean_group(R))/sqrt(var_group(R)+ε)","沿每行候选回答组对奖励中心化和缩放，展示 GRPO 所用的相对基线思想。","比较奖励同值、离散值、数值平移与正尺度变化。","使用平滑 epsilon 的特定教学约定，不包含 PPO clipping、KL 项或完整 GRPO 目标。"),
  router_gate_weights:block("被选专家门控权重","Selected Router Gates","routing",["scores","indices"],{mode:"sigmoid",normalize:true},"g_t,k=f(score_t,index_t,k) / Σ_k f(score_t,index_t,k)","先由离散路由选专家，再从未加偏置的 affinity 计算实际 gate。可选 identity（已计算 affinity）、sigmoid 或 selected-softmax。","刻意分别连接 biased 选择分数与 raw gate 分数，观察权重路径差异。","使用连续门控值；专家索引不可微，且不是完整 DeepSeek-V3 端到端路由器。"),
  dispatch_routed_tokens:block("Top-k Token Dispatch","Route Token Copies","routing",["x","indices"],{},"Z[t·K+k,d]=X[t,d]","对每个 Token 按路由选中数量复制输入，输出 [T·K,D]；此阶段尚不计算专家网络。","与独立专家投影、SwiGLU 和加权 Combine 组成完整小型 MoE 计算路径。","仅将复制位置对应到选中专家，不进行容量裁剪或分布式 All-to-All；T·K≤8。"),
  expert_linear_bank:block("按专家权重投影","Expert-specific Linear","routing",["x","indices","weights"],{},"Y[tK+k,h]=Σ_d X[tK+k,d] W[expert[t,k]·D+d,h]","参数矩阵按 Expert 分块，每个专家维护独立投影权重；同一专家处理多个 Token 时重复读取同一块权重。","分别作为 W1/W3/W2 投影，与 SwiGLU 组合成 token-dependent Expert FFN。","权重行为教学中显式输入；无 Bias、量化、跨设备通信，训练反向由单独 VJP 合同验证。"),
  routed_weighted_reduce:block("专家贡献加权归约","Weighted Route Reduce","routing",["values","gates"],{},"Y[t,d]=Σ_k G[t,k]·V[t·K+k,d]","把 token-major 的专家输出按原 Token 的 Top-k 门控重新归约为 [T,D]。","比较 Shared Expert 的独立支路及路由门控是否改变输出。","严格依赖 Dispatch 的 token-major 排列，不隐含专家容量或任意排序置换。"),
  expert_bank_mix:block("专家贡献加权回填","Expert Bank Mix","routing",["indices","gates","experts"],{},"Y_t=Σ_k g_t,k · ExpertBank[index_t,k,:]","把选中专家的教学向量按 gate 加权求和，清楚展示多专家聚合。","与 Top-k、Selected Gates、共享专家残差 Add 组合。","experts 是静态向量矩阵，不是 token 条件 Expert FFN；尚无 GPU dispatch 或完整反向传播。"),
  repeat_kv_rows:block("分组 KV 复制","KV Group Repeat","attention",["x"],{groups:2},"Y=repeat_interleave(X,groups,axis=rows)","以二维教学切片展示 GQA 中 KV head 向更多 query head 对齐的重复映射。","观察共享 KV 的多条消费路径以及反向应求和的语义。","此积木把行视为 Head，而非 Token；必须在真实多轴模型中校准轴含义。")
 };
 const KINDS=Object.keys(defs);
 const fail=s=>{throw Error(s);};
 const ensure=(m,name)=>{
   if(!Array.isArray(m)||!m.length||m.length>8||!Array.isArray(m[0])||!m[0].length||m[0].length>16||
      m.some(row=>!Array.isArray(row)||row.length!==m[0].length||row.some(v=>typeof v!=='number'||!Number.isFinite(v))))
       fail(name+' 必须是 T≤8、D≤16 的有限二维张量');
   return m;
 };
 const dims=(m)=>[m.length,m[0].length];
 const int=(x,lo,hi,name)=>{if(!Number.isSafeInteger(x)||x<lo||x>hi)fail(name+' 必须在 '+lo+'–'+hi+' 的整数范围');};
 const float=(x,lo,hi,name,loInclusive=true)=>{
   if(typeof x!=='number'||!Number.isFinite(x)||x>hi||(loInclusive?x<lo:x<=lo))
     fail(name+' 参数不合法');
 };
 const same=(a,b)=>{if(a.length!==b.length||a[0].length!==b[0].length)fail('输入张量形状必须完全一致');};
 const map=(m,fn)=>m.map((row,i)=>row.map((v,j)=>fn(v,i,j)));
 const pair=(a,b,fn)=>{same(a,b);return map(a,(v,i,j)=>fn(v,b[i][j]));};
 const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):(Math.exp(x)/(1+Math.exp(x)));
 const gelu=x=>0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*x**3)));
 const softmax=row=>{const max=Math.max(...row),e=row.map(x=>Math.exp(x-max)),sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum);};
 const requireBinary=m=>{if(m.some(r=>r.some(x=>x!==0&&x!==1)))fail('Mask 只接受 0/1');};
 const table=(index,col,seed)=>Math.sin((index+1)*.331+(col+1)*.73+seed*.11)*.52+
   Math.cos((col+1)*(index+1)*.17)*.1;
 function validate(kind,params){
   const n={...defs[kind].params,...params};
   if(kind==='token_ids'){int(n.rows,1,8,'rows');int(n.vocab,2,16,'vocab');int(n.seed,-10000,10000,'seed');}
   if(kind==='embedding'){int(n.width,1,16,'width');int(n.vocab,2,16,'vocab');}
   if(kind==='slice_cols'){int(n.start,0,15,'start');int(n.width,1,16,'width');}
   if(kind==='repeat_rows')int(n.times,1,8,'times');
   if(kind==='scale')float(n.factor,-100,100,'factor');
   if(kind==='bias')float(n.bias,-100,100,'bias');
   if(kind==='router_gate_weights'&&(!['identity','sigmoid','softmax'].includes(n.mode)||typeof n.normalize!=='boolean'))fail('router gate mode or normalize invalid');
   if(kind==='partial_rope'){int(n.rotaryDim,2,16,'rotary dimension');int(n.offset,0,64,'offset');if(n.rotaryDim%2)fail('rotary dimension must be even');}
   if(kind==='group_topk_mask'){int(n.groups,1,8,'groups');int(n.keepGroups,1,8,'keepGroups');if(n.keepGroups>n.groups)fail('keepGroups cannot exceed groups');}
   if(kind==='masked_router_topk')int(n.k,1,16,'k');
   if(kind==='group_advantage')float(n.epsilon,0,1,'epsilon',false);
   if(kind==='scatter_add')int(n.rows,1,8,'scatter rows');
   if(kind==='repeat_kv_rows')int(n.groups,1,8,'KV groups');
   if(kind==='sliding_window_mask')int(n.window,1,8,'window');
   if(kind==='router_topk')int(n.k,1,8,'routing k');
   if(kind==='qk_norm')float(n.epsilon,0,1,'epsilon',false);
   if(kind==='logit_softcap')float(n.cap,0,100,'cap',false);
   if(kind==='residual_scale')float(n.factor,-100,100,'factor');
   if(kind==='temperature')float(n.temperature,0,100,'temperature',false);
   if(kind==='topk')int(n.k,1,16,'k');
   if(kind==='rope2d')int(n.offset,0,64,'offset');
   if(kind==='l2norm'||kind==='layernorm')float(n.epsilon,0,1,'epsilon',false);
   if(kind==='dropout_demo'){
     float(n.p,0,1,'dropout p',true);if(n.p===1)fail('dropout p 必须小于 1');
     int(n.seed,-10000,10000,'seed');if(typeof n.training!=='boolean')fail('training 必须是布尔值');
   }
 }
 function execute(node,inputs){
   const kind=node.kind,params={...defs[kind].params,...node.params};
   validate(kind,params);
   for(const [port,value] of Object.entries(inputs))ensure(value,port);
   let value,internals=null;
   const x=inputs.x,a=inputs.a,b=inputs.b;
   switch(kind){
    case 'token_ids':
      value=Array.from({length:params.rows},(_,i)=>[(Math.abs((Math.imul(i+1,1103515245) + params.seed*12345)|0))%params.vocab]);break;
    case 'embedding':
      value=inputs.ids.map(row=>{
        const id=row[0];
        if(row.length!==1||!Number.isSafeInteger(id)||id<0||id>=params.vocab)fail('Embedding ID 必须是 0..vocab−1 范围内的整数 [T,1]');
        return Array.from({length:params.width},(_,d)=>table(id,d,11));
      });break;
    case 'pos_encoding':
      if(x[0].length%2)fail('Sinusoidal PE 要求偶数特征维');
      value=map(x,(v,t,d)=>v+(d%2?Math.cos(t/10000**(2*Math.floor(d/2)/x[0].length)):Math.sin(t/10000**(2*Math.floor(d/2)/x[0].length))));break;
    case 'transpose2d':
      if(x[0].length>8)fail('Transpose 输出 token 轴不能超过 8');
      value=Array.from({length:x[0].length},(_,j)=>x.map(row=>row[j]));break;
    case 'slice_cols':
      if(params.start+params.width>x[0].length)fail('切片范围超出特征维');
      value=x.map(row=>row.slice(params.start,params.start+params.width));break;
    case 'concat_cols':
      if(a.length!==b.length)fail('Concat 两路 Token 数必须一致');
      if(a[0].length+b[0].length>16)fail('Concat 特征宽度不得超过 16');
      value=a.map((row,i)=>row.concat(b[i]));break;
    case 'repeat_rows':
      if(x.length*params.times>8)fail('Repeat 后 Token 数不得超过 8');
      value=Array.from({length:params.times},()=>x.map(row=>row.slice())).flat();break;
    case 'row_mean': value=x.map(row=>[row.reduce((u,v)=>u+v,0)/row.length]);break;
    case 'row_sum':value=x.map(row=>[row.reduce((u,v)=>u+v,0)]);break;
    case 'row_var':value=x.map(row=>{const mean=row.reduce((u,v)=>u+v,0)/row.length;return [row.reduce((u,v)=>u+(v-mean)**2,0)/row.length];});break;
    case 'layernorm':
      value=x.map(row=>{const mean=row.reduce((u,v)=>u+v,0)/row.length;
       const variance=row.reduce((u,v)=>u+(v-mean)**2,0)/row.length;
       return row.map(v=>(v-mean)/Math.sqrt(variance+params.epsilon));});break;
    case 'l2norm':
      value=x.map(row=>{const den=Math.max(Math.hypot(...row),params.epsilon);return row.map(v=>v/den);});break;
    case 'relu':value=map(x,v=>Math.max(0,v));break;
    case 'gelu_tanh':value=map(x,gelu);break;
    case 'sigmoid':value=map(x,sigmoid);break;
    case 'tanh':value=map(x,Math.tanh);break;
    case 'softplus':value=map(x,v=>Math.max(v,0)+Math.log1p(Math.exp(-Math.abs(v))));break;
    case 'subtract':value=pair(a,b,(u,v)=>u-v);break;
    case 'divide':value=pair(a,b,(u,v)=>{if(v===0)fail('Divide 分母不得为 0');return u/v;});break;
    case 'scale':value=map(x,v=>v*params.factor);break;
    case 'bias':value=map(x,v=>v+params.bias);break;
    case 'square':value=map(x,v=>v*v);break;
    case 'swiglu':value=pair(inputs.gate,inputs.up,(u,v)=>u*sigmoid(u)*v);break;
    case 'geglu':value=pair(inputs.gate,inputs.up,(u,v)=>gelu(u)*v);break;
    case 'causal_mask':
      value=Array.from({length:x.length},(_,i)=>Array.from({length:x.length},(_,j)=>j<=i?1:0));break;
    case 'masked_softmax':
      same(inputs.logits,inputs.mask);requireBinary(inputs.mask);
      value=inputs.logits.map((row,i)=>{
        const mask=inputs.mask[i];if(!mask.includes(1))fail('Masked Softmax 某行被全部屏蔽');
        const max=Math.max(...row.filter((_,j)=>mask[j]===1));
        const e=row.map((z,j)=>mask[j]?Math.exp(z-max):0),den=e.reduce((u,v)=>u+v,0);
        return e.map(z=>z/den);
      });break;
    case 'rope2d':
      if(x[0].length%2)fail('RoPE 要求偶数特征维');
      value=x.map((row,t)=>row.map((v,j)=>{
        const i=j-j%2,angle=(t+params.offset)/10000**(i/row.length),a=row[i],b=row[i+1];
        return j%2?a*Math.sin(angle)+b*Math.cos(angle):a*Math.cos(angle)-b*Math.sin(angle);
      }));break;

    case 'low_rank_project':{const A=inputs.down,B=inputs.up;if(x[0].length!==A.length||A[0].length!==B.length)fail('low rank projection dimension mismatch');
      const C=x.map(row=>A[0].map((_,j)=>row.reduce((a,v,i)=>a+v*A[i][j],0)));
      value=C.map(row=>B[0].map((_,j)=>row.reduce((a,v,i)=>a+v*B[i][j],0)));
      internals={latent:{shape:dims(C),values:C,note:'low-rank latent: not full MLA cache by itself'}};break;}
    case 'partial_rope':{if(params.rotaryDim>x[0].length)fail('rotary dimension exceeds features');
      value=x.map((row,t)=>row.map((v,j)=>{if(j>=params.rotaryDim)return v;const i=j-j%2,theta=(t+params.offset)/10000**(i/params.rotaryDim);
        return j%2?row[i]*Math.sin(theta)+row[i+1]*Math.cos(theta):row[i]*Math.cos(theta)-row[i+1]*Math.sin(theta);}));break;}
    case 'row_bias_add':{const S=inputs.scores,B=inputs.bias;if(B.length!==1||B[0].length!==S[0].length)fail('row bias requires [1,E]');
      value=S.map(row=>row.map((v,j)=>v+B[0][j]));break;}
    case 'group_topk_mask':{const S=inputs.scores,E=S[0].length;if(E%params.groups)fail('expert count must be divisible by groups');
      const size=E/params.groups;value=[];const selected=[];
      S.forEach(row=>{const ranked=Array.from({length:params.groups},(_,g)=>{
        const top=row.slice(g*size,(g+1)*size).sort((a,b)=>b-a).slice(0,2);
        return {g,score:top.reduce((a,b)=>a+b,0)};}).sort((a,b)=>b.score-a.score||a.g-b.g).slice(0,params.keepGroups);
        const chosen=new Set(ranked.map(v=>v.g));selected.push(ranked.map(v=>v.g));
        value.push(row.map((_,j)=>chosen.has(Math.floor(j/size))?1:0));});
      internals={groups:{shape:[S.length,params.keepGroups],values:selected,note:'group score = sum of highest up to 2 scores in each group; toy routing contract'}};break;}
    case 'masked_router_topk':{const S=inputs.scores,M=inputs.mask;same(S,M);requireBinary(M);
      const selected=S.map((row,t)=>row.map((v,j)=>({v,j})).filter(z=>M[t][z.j]===1).sort((a,b)=>b.v-a.v||a.j-b.j).slice(0,params.k));
      if(selected.some(row=>row.length!==params.k))fail('not enough unmasked experts for k');
      value=selected.map(row=>row.map(z=>z.j));internals={selected_weights:{shape:[S.length,params.k],values:selected.map(row=>softmax(row.map(z=>z.v))),note:'selected-softmax is illustrative; not a universal MoE gate'}};break;}
    case 'log_softmax':value=x.map(row=>{const m=Math.max(...row),logZ=m+Math.log(row.reduce((a,v)=>a+Math.exp(v-m),0));return row.map(v=>v-logZ);});break;
    case 'selected_logprob':{const S=inputs.logits,I=inputs.labels;if(S.length!==I.length||I[0].length!==1)fail('selected logprob labels must be [T,1]');
      value=S.map((row,t)=>{const id=I[t][0];if(!Number.isSafeInteger(id)||id<0||id>=row.length)fail('selected logprob label out of range');
        const m=Math.max(...row),z=m+Math.log(row.reduce((a,v)=>a+Math.exp(v-m),0));return [row[id]-z];});break;}
    case 'group_advantage':{value=inputs.rewards.map(row=>{const mu=row.reduce((a,v)=>a+v,0)/row.length;
      const variance=row.reduce((a,v)=>a+(v-mu)**2,0)/row.length;
      return row.map(v=>(v-mu)/Math.sqrt(variance+params.epsilon));});break;}
    case 'router_gate_weights':{const S=inputs.scores,I=inputs.indices;
      if(S.length!==I.length)fail('router gates require same token rows');
      value=I.map((row,t)=>{const picked=row.map(j=>{
        if(!Number.isSafeInteger(j)||j<0||j>=S[t].length)fail('router gate expert index out of range');
        return S[t][j];});
        const vals=params.mode==='identity'?picked:params.mode==='sigmoid'?picked.map(sigmoid):softmax(picked);
        const sum=vals.reduce((a,b)=>a+b,0);if(params.mode==='identity'&&vals.some(v=>v<0))fail('identity gate affinities must be nonnegative');if(params.normalize&&sum<=0)fail('router gates all zero');
        return params.normalize?vals.map(v=>v/sum):vals;});
      break;}
    case 'dispatch_routed_tokens':{const I=inputs.indices;if(I.length!==x.length)fail('dispatch tokens and index rows mismatch');
      if(I.length*I[0].length>8)fail('dispatch exceeds 8 routed rows');
      value=I.flatMap((row,t)=>row.map(id=>{if(!Number.isSafeInteger(id)||id<0)fail('dispatch expert id must be nonnegative integer');return x[t].slice();}));break;}
    case 'expert_linear_bank':{const I=inputs.indices,W=inputs.weights,Z=x;
      const T=I.length,K=I[0].length,D=Z[0].length;if(T*K!==Z.length)fail('expert linear selected rows must equal T*K');
      if(W.length%D!==0)fail('expert weight rows must equal Experts*D');
      const E=W.length/D;
      const counts=Array(E).fill(0);
      value=Z.map((row,n)=>{const t=Math.floor(n/K),k=n%K,id=I[t][k];
        if(!Number.isSafeInteger(id)||id<0||id>=E)fail('expert linear index out of range');counts[id]++;
        return W[0].map((_,h)=>row.reduce((v,x,d)=>v+x*W[id*D+d][h],0));});
      internals={expert_load:{shape:[E,1],values:counts.map(v=>[v]),note:'每个专家收到的 token 贡献数；重复使用同一组专家参数'}};break;}
    case 'routed_weighted_reduce':{const V=inputs.values,G=inputs.gates,T=G.length,K=G[0].length;
      if(V.length!==T*K)fail('weighted route reduce expects values [T*K,D]');
      value=G.map((g,t)=>V[0].map((_,d)=>g.reduce((sum,w,k)=>sum+w*V[t*K+k][d],0)));break;}
    case 'expert_bank_mix':{const I=inputs.indices,G=inputs.gates,E=inputs.experts;same(I,G);
      value=I.map((row,t)=>E[0].map((_,d)=>row.reduce((sum,id,k)=>{
        if(!Number.isSafeInteger(id)||id<0||id>=E.length)fail('expert bank index out of range');
        return sum+G[t][k]*E[id][d];},0)));break;}
    case 'gather_routed_bank':{const ids=inputs.indices;if(ids.length*ids[0].length>8)fail('routed gather exceeds eight rows');
      value=ids.flatMap(row=>row.map(i=>{if(!Number.isSafeInteger(i)||i<0||i>=x.length)fail('routed gather index out of bounds');return x[i].slice();}));break;}
    case 'gather_rows':{const ids=inputs.indices;if(ids[0].length!==1)fail('indices must be [T,1]');value=ids.map(([i])=>{if(!Number.isSafeInteger(i)||i<0||i>=x.length)fail('gather index out of bounds');return x[i].slice();});break;}
    case 'scatter_add':{const ids=inputs.indices,v=inputs.values;if(ids.length!==v.length||ids[0].length!==1)fail('scatter index/values mismatch');value=Array.from({length:params.rows},()=>Array(v[0].length).fill(0));ids.forEach(([i],t)=>{if(!Number.isSafeInteger(i)||i<0||i>=params.rows)fail('scatter index out of bounds');v[t].forEach((z,j)=>{value[i][j]+=z;});});break;}
    case 'row_logsumexp':value=x.map(row=>{const m=Math.max(...row);return [m+Math.log(row.reduce((a,v)=>a+Math.exp(v-m),0))];});break;
    case 'repeat_kv_rows':if(x.length*params.groups>8)fail('KV repeat rows exceed 8');value=x.flatMap(row=>Array.from({length:params.groups},()=>row.slice()));break;
    case 'qk_norm':value=x.map(row=>{const d=Math.sqrt(row.reduce((a,v)=>a+v*v,0)/row.length+params.epsilon);return row.map(v=>v/d);});break;
    case 'sliding_window_mask':value=x.map((row,i)=>row.map((_,j)=>Number(j<=i&&i-j<params.window)));if(x.length!==x[0].length)fail('sliding window mask expects square matrix');break;
    case 'logit_softcap':value=map(x,v=>params.cap*Math.tanh(v/params.cap));break;
    case 'router_topk':{if(params.k>x[0].length)fail('k exceeds experts');const picks=x.map(row=>row.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v||a.i-b.i).slice(0,params.k));value=picks.map(row=>row.map(z=>z.i));internals={selected_weights:{shape:[x.length,params.k],values:picks.map(row=>softmax(row.map(z=>z.v))),note:'仅对已选专家 logits 重归一化，具体模型的门控方式可能不同'}};break;}
    case 'weighted_sum':{const w=inputs.weights,v=inputs.values;if(w[0].length!==v.length)fail('weights K must equal values rows');value=w.map(row=>v[0].map((_,j)=>row.reduce((acc,z,k)=>acc+z*v[k][j],0)));break;}
    case 'detach_demo':value=x.map(row=>row.slice());internals={gradient_boundary:{note:'forward identity; backward VJP = 0 (symbolic only)'}};break;
    case 'residual_scale':value=map(x,v=>v*params.factor);break;
    case 'temperature':value=map(x,v=>v/params.temperature);break;
    case 'argmax':value=x.map(row=>[row.reduce((best,v,j)=>v>row[best]?j:best,0)]);break;
    case 'topk':
      if(params.k>x[0].length)fail('Top-k 的 k 不能超过特征宽度');
      const pairs=x.map(row=>row.map((v,j)=>({v,j})).sort((u,v)=>v.v-u.v||u.j-v.j).slice(0,params.k));
      value=pairs.map(row=>row.map(z=>z.v));
      internals={indices:{shape:[x.length,params.k],values:pairs.map(row=>row.map(z=>z.j)),note:'并列分数按原下标升序排列'}};break;
    case 'dropout_demo':{
      // deterministic PRNG hashed by node id, seed, element position. Same run reproduces same mask.
      const mask=map(x,(_v,i,j)=>{
        let h=2166136261;
        for(const ch of node.id+':'+params.seed+':'+i+':'+j)h=Math.imul(h^ch.charCodeAt(0),16777619);
        return ((h>>>0)/4294967296)>=params.p?1:0;
      });
      value=params.training?pair(x,mask,(v,m)=>v*m/(1-params.p)):x.map(row=>row.slice());
      internals={mask:{shape:[x.length,x[0].length],values:params.training?mask:mask.map(r=>r.map(()=>1)),
         note:params.training?'固定种子教学 mask（同配置每次相同），不是训练中的独立随机采样':'评估模式不屏蔽元素'}};break;
    }
    case 'cross_entropy':{
      const logits=inputs.logits,labels=inputs.labels;
      if(labels.length!==logits.length||labels[0].length!==1)fail('类别标签 shape 需为 [T,1]');
      const losses=logits.map((row,i)=>{
        const cls=labels[i][0];
        if(!Number.isSafeInteger(cls)||cls<0||cls>=row.length)fail('CrossEntropy 标签必须为 [0,C) 整数 ID');
        const max=Math.max(...row);
        return max+Math.log(row.reduce((s,v)=>s+Math.exp(v-max),0))-row[cls];
      });
      value=[[losses.reduce((s,v)=>s+v,0)/losses.length]];
      internals={per_token_loss:{shape:[losses.length,1],values:losses.map(v=>[v])}};break;
    }
    case 'mae_loss':
      same(inputs.prediction,inputs.target);
      value=[[inputs.prediction.flat().reduce((acc,v,i)=>acc+Math.abs(v-inputs.target.flat()[i]),0)/(inputs.prediction.length*inputs.prediction[0].length)]];break;
    case 'kl_prob':{
      const p=inputs.p,q=inputs.q;same(p,q);
      const rows=p.map((row,i)=>{
        const qi=q[i],sumP=row.reduce((a,b)=>a+b,0),sumQ=qi.reduce((a,b)=>a+b,0);
        if(Math.abs(sumP-1)>1e-6||Math.abs(sumQ-1)>1e-6||row.some(v=>v<0)||qi.some(v=>v<0))fail('KL 输入必须是非负且逐行和为 1 的概率');
        return row.reduce((acc,prob,j)=>{
          if(prob===0)return acc;
          if(qi[j]===0)fail('KL 当 P>0 时 Q 不能等于零');
          return acc+prob*Math.log(prob/qi[j]);
        },0);
      });
      value=[[rows.reduce((u,v)=>u+v,0)/rows.length]];break;
    }
    default:fail('积木未实现数值计算：'+kind);
   }
   ensure(value,'output');
   return {values:value,shape:dims(value),flops:null,internals};
 }
 return Object.freeze({defs: Object.freeze(defs),KINDS,validate,execute});
});
