/* SeetheTruth conceptual teaching execution.
 * Propagates declared tensor shapes, flow and operator semantics ONLY.
 * NEVER allocates tensor matrices, invokes a numerical kernel or measures FLOPs.
 */
(function(root,factory){
  "use strict";
  const api=factory(typeof module==="object"&&module.exports
    ?require("./learning_lab.js"):root.SeetheTruthLearning);
  if(typeof module==="object"&&module.exports)module.exports=api;
  root.SeetheTruthConceptFlow=api;
})(typeof window!=="undefined"?window:{},function(L){
  "use strict";
  const CATALOG=Object.keys(L.defs);
  const EVIDENCE="declared_semantic_and_shape_contract_not_numerical_execution";
  const fail=(message,id)=>{const e=new Error(message);e.nodeId=id;throw e;};
  const equal=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
  const by=(s,what,id)=>{if(!s)fail(what,id);};
  const pretty=s=>s.join("×");
  const groups=Object.freeze({
    data:"数据来源与表征",compute:"特征映射与算术",norm:"归一化：尺度或中心调整",
    vision:"图像空间与 Patch Token",audio:"音频时频表示",video:"视频时空 Tubelet",
    generative:"扩散模型的噪声状态更新",
    inference:"推理服务的采样、验收与调度",
    activate:"非线性与权重分布",combine:"信息融合与残差候选",
    attention:"跨 token 信息汇聚",loss:"训练目标（前向定义）",
    shape:"张量轴与布局改变",reduce:"跨特征聚合",position:"位置信息注入",
    gated:"门控 FFN 的逐通道信息控制",sampling:"输出决策或候选筛选",
    regularize:"训练/推理行为差异",routing:"离散路由与专家集合",training:"策略概率/奖励与梯度语义"
  });
  const unary=new Set(["rmsnorm","silu","softmax","layernorm","l2norm",
    "relu","gelu_tanh","gelu_erf","relu_squared","mish","leaky_relu","elu","hardswish","sigmoid","tanh","softplus","scale","bias","square",
    "dropout_demo","pos_encoding","rope2d","temperature","qk_norm","logit_softcap","detach_demo","residual_scale","partial_rope","log_softmax"]);
  const binary=new Set(["add","mul","subtract","divide","swiglu","geglu"]);
  const reduced=new Set(["row_mean","row_sum","row_var","row_logsumexp"]);
  const losses=new Set(["mse","mae_loss","kl_prob"]);
  const ports=(n,inputs,...keys)=>keys.map(k=>{by(inputs[k],n.kind+" 缺少 "+k+" 输入",n.id);return inputs[k];});
  function analyze(n,inputs){
    const p={...L.defs[n.kind].params,...n.params},kind=n.kind;
    const get=k=>ports(n,inputs,k)[0];
    let output,rule,conditions=[],stages=null;
    if(["inference_topk","inference_sparse_attention","inference_paged_kv_read","inference_greedy_verify"].includes(kind)){
      const Op=typeof module==="object"&&module.exports?require("./inference_ops.js"):globalThis.SeetheTruthInference;
      const inputShapes=Object.fromEntries(L.defs[kind].ports.map(port=>[port,get(port)]));
      output=Op.shape(kind,inputShapes,p);
      rule=kind==="inference_topk"?"数值 Top-K 与稳定 tie-break；返回真正整数索引":
        kind==="inference_sparse_attention"?"按明确 Key 索引做 QK、稳定 Softmax、PV；不模拟 Lightning Indexer":
        kind==="inference_paged_kv_read"?"从逻辑位置经 Block Table 映射到物理 KV 页和页内槽位":
        "贪心 speculative 验证仅接收自第 0 步起的连续正确 Draft 前缀";
      conditions.push("精确的 CPU 教学算术或控制流，并非 DeepSeek GPU Kernel、完整 Serving Engine 或测速");
    }else if(kind==="video_causal_conv3d"){
      const Video=typeof module==="object"&&module.exports?require("./video_causal_conv3d.js"):globalThis.SeetheTruthVideoConv3D;
      output=Video.shape({x:get("x"),kernel:get("kernel")},p);
      rule="真实 [1,T,H,W,C] 输入与 [O,Kt,Kh,Kw,I] 卷积核；左侧补零、时间步长为 1 的因果 3D 互相关，然后转为教育用二维载体";
      conditions.push("未来帧不参与当前输出；流式缓存独立保留历史 Kt-1 帧，非 GPU 持久 KV Cache");
    }else if(["vision_conv2d","audio_mel_filterbank","diffusion_ddim_step"].includes(kind)){
      const Adv=typeof module==="object"&&module.exports?require("./multimodal_advanced.js"):globalThis.SeetheTruthMultimodalAdvanced;
      const tensorShapes=Object.fromEntries(L.defs[kind].ports.map(key=>[key,get(key)]));
      output=Adv.shape(kind,tensorShapes,p);
      rule=kind==="vision_conv2d"?"给定 HWC 和明示 2D 卷积参数矩阵，执行互相关，输出 Hout×(Wout·Cout)":
       kind==="audio_mel_filterbank"?"对真正非负功率频谱应用参数指定的 HTK Mel 三角频带权重投影":
       "根据两组累积 alpha 参数对 xt 与 epsilon 执行确定性 DDIM 逆扩散一步";
      conditions.push("仅微型真实数学计算；未推断 pretrained 模型、音频解码或 GPU 性能");
    }else if(["vision_patchify","video_tubelet","audio_stft_power","audio_log_power"].includes(kind)){
      const Multi=typeof module==="object"&&module.exports?require("./multimodal_ops.js"):globalThis.SeetheTruthMultimodal;
      const x=get("x");output=Multi.shape(kind,x,p);
      rule=kind==="vision_patchify"?"明确 HWC 网格轴并按非重叠空间 Patch 展平，未执行学习投影":
       kind==="video_tubelet"?"显式 THWC 时空块并按时间、空间顺序组成 Tubelet，未执行 3D 卷积":
       kind==="audio_stft_power"?"周期 Hann 窗 + 未中心补零的真实离散 STFT 功率，频率轴为 0..N/2":
       "对非负功率谱按固定 floor 执行 log10；不是标准化的 Mel 谱";
      conditions.push("仅有界二维教学载体；布局来自积木参数，未解码真实媒体文件");
    }else if(kind==="nd_parameter"){const ND=typeof module==="object"&&module.exports?require("./nd_tensor.js"):globalThis.SeetheTruthND;output=ND.shape(p.values);rule="节点 ID 是真实小张量参数身份，JSON 存放 θ 及 AdamW 的 m/v/t，多个下游共享同一可求导来源";
    }else if(["nd_gelu_erf","nd_silu","nd_relu_squared","nd_mish","nd_elu","nd_hardswish"].includes(kind)){
      const x=get("x");output=x.slice();rule="按四轴张量每一数值单独执行 "+L.defs[kind].en+"；轴意义与 Tensor 形状不改变，可通过真实 VJP 求梯度";
    }else if(kind==="nd_input"){
      output=[p.b,p.h,p.t,p.d];rule="生成显式 [B,H,T,D] 的小型四轴张量，实际数值仍是确定性 CPU 合成数据";
    }else if(kind==="nd_permute"){
      const x=get("x"),order=[p.axis0,p.axis1,p.axis2,p.axis3];
      by(x.length===4,"N-D permute requires rank 4",n.id);
      output=order.map(i=>x[i]);rule="按指定轴的完整置换重排每个元素；保留元素总量但不保留原轴顺序";
    }else if(kind==="nd_reshape"){
      const x=get("x"),out=[p.b,p.h,p.t,p.d];
      by(x.reduce((a,b)=>a*b,1)===out.reduce((a,b)=>a*b,1),"N-D reshape element count mismatch",n.id);
      output=out;rule="行主序顺次保留所有元素，只重新解释四轴布局";
    }else if(kind==="nd_flatten"){
      const x=get("x");by(x.length===4,"N-D flatten requires 4 axes",n.id);
      output=[x[0]*x[1]*x[2],x[3]];rule="合并前三轴为二维矩阵的行轴，最后通道轴不变";
    }else if(kind==="nd_expand4d"){
      const x=get("x");by(x.length===2,"N-D expand requires 2 axes",n.id);
      output=[1,1,x[0],x[1]];rule="给二维矩阵的两个前导维增加 singleton axes，不复制内容";
    }else if(kind==="nd_broadcast_add"){
      const [a,b]=ports(n,inputs,"a","b");
      const k=Math.max(a.length,b.length),out=[];
      for(let i=0;i<k;i++){const x=a[a.length-1-i]??1,y=b[b.length-1-i]??1;
       by(x===y||x===1||y===1,"N-D broadcast axes incompatible",n.id);out.unshift(Math.max(x,y));}
      output=out;rule="末尾轴右对齐的 singleton 广播加法；不自动转置轴";
    }else if(kind==="nd_reduce_sum"){
      const x=get("x");by(p.axis<x.length,"N-D reduce axis out of range",n.id);
      output=x.map((d,i)=>i===p.axis?1:d);rule="在指定轴求和而非取平均，keepdim 保留长度为 1 的轴";
    }else if(kind==="nd_rope"){
      const x=get("x");by(x.length===4,"N-D RoPE requires [B,H,T,D]",n.id);
      by(p.rotaryDim<=x[3]&&p.rotaryDim%2===0,"N-D RoPE rotary dimension must be even and <= channel dimension",n.id);
      output=x.slice();rule="对旋转维内的相邻通道二元组按 offset+Token 位置执行正交旋转，其余通道保持原值";
    }else if(kind==="nd_cache_read"){
      const x=get("x");by(x.length===4,"N-D KV cache read requires four axes",n.id);
      by(p.start+p.length<=x[2],"N-D cache read window exceeds cached Token axis",n.id);
      output=[x[0],x[1],p.length,x[3]];
      rule="保持 B/H/D，在 Token 轴显式读取 [start,start+length) 快照，不修改原缓存";
    }else if(kind==="nd_cache_append"){
      const [a,b]=ports(n,inputs,"cache","segment");
      by(a.length===4&&b.length===4,"N-D KV cache append requires four axes",n.id);
      by(a[0]===b[0]&&a[1]===b[1]&&a[3]===b[3],"N-D KV cache append Batch, Head, Channel mismatch",n.id);
      by(a[2]+b[2]<=8,"N-D cache append exceeds Token capacity",n.id);
      output=[a[0],a[1],a[2]+b[2],a[3]];
      rule="沿 Token 轴追加当前片段；B、KV Head 与 D 不允许隐式改变，源输入保持不可变";
    }else if(kind==="nd_repeat_kv"){
      const x=get("x");by(x.length===4,"GQA KV Repeat requires [B,Hkv,T,D]",n.id);
      by(p.queryHeads%x[1]===0,"GQA Query head count must divide KV head count exactly",n.id);
      output=[x[0],p.queryHeads,x[2],x[3]];
      rule="将每组 Query Heads 映射到一个共享的 KV Head，Hq/Hkv 整数比；实际 4D 输出是逐头复用数据";
    }else if(kind==="nd_mse"){
      const [a,b]=ports(n,inputs,"prediction","target");
      by(a.length===b.length&&a.every((d,i)=>d===b[i]),"N-D MSE requires identical tensor shapes",n.id);
      output=[1,1];rule="对所有真实多轴元素执行平均平方误差，target 是固定监督，输出可求导的标量 Loss";
    }else if(kind==="nd_scale"){
      output=get("x").slice();rule="对所有轴中的每个浮点元素按显式 factor 缩放；不混淆独立 Head";
    }else if(kind==="nd_softmax_last"){
      output=get("x").slice();rule="每个前导轴坐标下独立沿最后的 Key 轴做稳定 Softmax";
      conditions.push("输入包含有限数值，归一化结果每行和约等于 1");
    }else if(kind==="nd_causal_mask"){
      const x=get("x");by(x.length===4,"Causal scores need [B,H,Tq,Tk] 4D",n.id);
      by(p.queryStart>=p.keyStart,"Causal first query has no legal key",n.id);
      output=x.slice();rule="只把未来 Key 位置置为有限的大负数；Query 和 Key 使用绝对起点偏移";
    }else if(kind==="nd_batched_matmul"){
      const [a,b]=ports(n,inputs,"a","b");
      by(a.length===4&&b.length===4,"N-D Batched MatMul requires rank 4",n.id);
      by(a[3]===b[2],"N-D Batched MatMul contracted K dimensions mismatch",n.id);
      by((a[0]===b[0]||a[0]===1||b[0]===1)&&(a[1]===b[1]||a[1]===1||b[1]===1),
        "N-D MatMul batch/head axes cannot broadcast",n.id);
      output=[Math.max(a[0],b[0]),Math.max(a[1],b[1]),a[2],b[3]];
      rule="在每个批次和头中只沿 K 轴收缩；B/H 轴允许 singleton 广播";
    }else if(["opt_adamw","opt_adam","opt_sgd_momentum","opt_rmsprop","opt_adagrad","opt_lion"].includes(kind)){
      const [a,b]=ports(n,inputs,"parameter","gradient");
      by(equal(a,b),kind+" parameter and gradient shape mismatch",n.id);
      output=a.slice();rule="对给定 Parameter 和 Gradient 从零历史状态计算一步 "+L.defs[kind].en+"，仅返回下一参数矩阵，不修改外部图或维护多步状态";
    }else if(kind==="input"){output=[p.rows,p.width];rule="输入连续表征，建立 token 轴 T 和特征轴 D";}
    else if(kind==="parameter"){output=[p.values.length,p.values[0].length];rule="节点 ID 为可训练参数身份；值与 AdamW 状态属于 JSON 工程，每次训练更新产生新版本快照";}
    else if(kind==="token_ids"){output=[p.rows,1];rule="产生每个 token 的离散类别索引 [T,1]";}
    else if(kind==="embedding"){
      const ids=get("ids");by(ids[1]===1,"Embedding ID 必须是 [T,1]",n.id);
      output=[ids[0],p.width];rule="每个 token 的 ID 查表成为一个宽度为 D 的向量";
      conditions.push("类别下标必须在词表范围内；定性模式不检查具体 ID");
    }else if(kind==="linear"){
      const x=get("x");output=[x[0],p.out];
      rule="沿特征轴 D 收缩；Token 数不变，输出宽度变为 D_out";
    }else if(kind==="transpose2d"){
      const x=get("x");by(x[1]<=8,"Transpose 输出 Token 轴超过界限",n.id);
      output=[x[1],x[0]];rule="二维轴交换 [T,D] → [D,T]，数学语义必须重新解释";
    }else if(kind==="slice_cols"){
      const x=get("x");by(p.start+p.width<=x[1],"Slice 特征切片范围超出输入宽度",n.id);
      output=[x[0],p.width];rule="保留相同 token 行，仅取连续隐藏特征通道";
    }else if(kind==="concat_cols"){
      const [a,b]=ports(n,inputs,"a","b");
      by(a[0]===b[0],"Concat 要求相同 Token 数",n.id);
      by(a[1]+b[1]<=16,"Concat 特征宽度超过微型合同",n.id);
      output=[a[0],a[1]+b[1]];rule="Token 轴对齐，沿特征轴拼接两路";
    }else if(kind==="repeat_rows"){
      const x=get("x");by(x[0]*p.times<=8,"Repeat 输出 Token 数超过微型合同",n.id);
      output=[x[0]*p.times,x[1]];rule="重复 token 行，增加序列长度，不产生新参数";
    }else if(reduced.has(kind)){
      const x=get("x");output=[x[0],1];rule="沿每个 token 的特征轴 D 做 Reduction，保留 token 轴";
    }else if(unary.has(kind)){
      const x=get("x");output=x.slice();
      rule={
        rmsnorm:"各 token 按均方根调尺度，不减去特征均值",
        layernorm:"各 token 沿特征轴去中心化并按总体方差标准化",
        l2norm:"将每个非零特征向量按 L2 范数调整",
        softmax:"沿每行最后特征轴归一化为概率分布，不跨 token",
        pos_encoding:"以加法将 token 位置信息注入特征表征",
        rope2d:"将每个 token 的相邻通道对按位置旋转",
        dropout_demo:"训练模式按概率屏蔽元素并缩放，评估模式恒等",
        temperature:"logits 除以温度改变 Softmax 集中度，不执行采样"
      }[kind]||"保持输入输出形状不变，在元素或各行上应用 "+L.defs[kind].en;
      if(["pos_encoding","rope2d"].includes(kind))
        by(x[1]%2===0,"位置编码在本教学实现中要求偶数 D",n.id);
      if(kind==="partial_rope")by(p.rotaryDim<=x[1]&&p.rotaryDim%2===0,"Partial RoPE 维度必须不超过 D 且为偶数",n.id);
      if(kind==="detach_demo")conditions.push("Stop Gradient 前向恒等；反向 VJP=0，不是冻结参数所有路径");
      if(kind==="dropout_demo")conditions.push("本概念推演不生成随机 Mask，也不报告实际屏蔽位置");
    }else if(binary.has(kind)){
      const names=["swiglu","geglu"].includes(kind)?["gate","up"]:["a","b"];
      const [a,b]=ports(n,inputs,...names);
      by(equal(a,b),kind+" 两路输入形状必须完全一致",n.id);
      output=a.slice();
      rule=["swiglu","geglu"].includes(kind)?
        "两路已投影特征分别充当门控与信息，再做逐元素相乘":
        "两路位置逐元素对齐后融合；只有旁路回接才可称为残差";
      if(kind==="divide")conditions.push("除法需要逐项非零分母；本模式不检查实际数值");
    }else if(kind==="matmul"){
      const [a,b]=ports(n,inputs,"a","b");
      by(a[1]===b[0],"MatMul 内部收缩维不相同："+pretty(a)+" × "+pretty(b),n.id);
      by(b[1]<=16,"MatMul 输出特征宽度超出界限",n.id);
      output=[a[0],b[1]];rule="A[T,K] × B[K,D] 在 K 上求和，输出 [T,D]";
    }else if(kind==="router_gate_weights"){
      const [scores,indices]=ports(n,inputs,"scores","indices");by(scores[0]===indices[0],"Router weights token count mismatch",n.id);
      output=indices.slice();rule="从原始 score 中按选中 index 提取亲和度，sigmoid 或 softmax 门控；normalize 可选";
      conditions.push("每个路由下标必须指向有效专家；数学模式不检查整数取值");
    }else if(kind==="cache_write_segment"){
      const [cache,segment]=ports(n,inputs,"cache","segment");
      by(cache[1]===segment[1],"Cache write feature width mismatch",n.id);
      by(p.start<=cache[0],"Cache write cannot skip unwritten positions",n.id);
      by(p.start+segment[0]<=p.capacity&&cache[0]<=p.capacity,"Cache write capacity overflow",n.id);
      output=[Math.max(cache[0],p.start+segment[0]),cache[1]];
      rule="纯函数缓存快照写入 C'[start:start+N]=new，其他行保留；历史源输入不发生修改";
      conditions.push("KV/PE cache 分别使用不同宽度的积木，start_pos/end_pos 必须匹配；此处不跨执行周期保存状态");
    }else if(kind==="cache_read_prefix"){
      const cache=get("cache");by(p.length<=cache[0],"Cache read length exceeds committed positions",n.id);
      output=[p.length,cache[1]];rule="读取有效位置区间 [0,end_pos) 的 Token 表示，不读取未来或尚未写入的行";
    }else if(kind==="query_weight_absorb"){
      const [q,w]=ports(n,inputs,"query","keyWeight");
      by(q[1]===w[1],"MLA absorbed query expects projected K channels",n.id);
      output=[q[0],w[0]];
      rule="Q_nope[T,Dk]·W_K[R,Dk]^T → Q_abs[T,R]，随后才能直接和 latent cache[S,R] 缩并";
      conditions.push("这是线性代数交换；真实低精度 Kernel 的舍入可能导致极小数值差异");
    }else if(kind==="dispatch_routed_tokens"){
      const [x,idx]=ports(n,inputs,"x","indices");
      by(x[0]===idx[0],"Dispatch token rows and route rows mismatch",n.id);
      by(idx[0]*idx[1]<=8,"Dispatch exceeds bounded token count",n.id);
      output=[idx[0]*idx[1],x[1]];
      rule="将 X[T,D] 按路由重复 K 次为 [T·K,D]，后续 Expert Linear 会使用对应专家的独立参数";
      conditions.push("展平顺序为 t·K+k，不重排实际计算批次，不计算分布式交换");
    }else if(kind==="expert_linear_bank"){
      const [x,idx,w]=ports(n,inputs,"x","indices","weights");
      by(x[0]===idx[0]*idx[1],"Expert Linear token-major rows mismatch",n.id);
      by(w[0]%x[1]===0,"Expert Linear weights need E·D rows",n.id);
      output=[x[0],w[1]];rule="从展平 E·D×H 权重中按专家 ID 选择 D×H 子矩阵后逐路 MatMul";
      conditions.push("同一专家共享一套参数，反向时多 Token 对其梯度必须汇合；数学模式不验证 index 的整数范围");
    }else if(kind==="routed_weighted_reduce"){
      const [v,g]=ports(n,inputs,"values","gates");
      by(v[0]===g[0]*g[1],"Route Reduce requires [T·K,D] and [T,K]",n.id);
      output=[g[0],v[1]];rule="每个 Token 的 K 个专家输出按其 Gate 权重相加，恢复 [T,D]";
      conditions.push("仅沿固定 Token-major 次序归约，后续可与 Shared Expert 输出做 Add");
    }else if(kind==="expert_bank_mix"){
      const [indices,gates,experts]=ports(n,inputs,"indices","gates","experts");
      by(equal(indices,gates),"Expert gate/index shapes must match",n.id);
      output=[indices[0],experts[1]];rule="用每个 token 的离散专家索引和对应门控权重从演示专家向量池中加权汇聚";
      conditions.push("这是静态专家向量池，不是 token 条件 FFN；数值模式检查 index 有效");
    }else if(kind==="gather_routed_bank"){
      const [x,idx]=ports(n,inputs,"x","indices");
      by(idx[0]*idx[1]<=8,"Top-k Gather teaching rows exceed limit",n.id);
      output=[idx[0]*idx[1],x[1]];
      rule="按 token-major 顺序展开每个 token 的 K 个离散专家索引，输出 [T·K,D] 的示例专家向量";
      conditions.push("概念模式不检查每个索引实际是否有效；不执行真正的专家 FFN");
    }else if(kind==="slice_rows"){
      const x=get("x");
      by(x.length===2&&p.start+p.length<=x[0],"Token slice requires 2D source with start+length <= T",n.id);
      output=[p.length,x[1]];
      rule="沿时间 Token 轴从 start 开始选择 length 行，特征轴 D 完整保留；前缀 [0,T−1) 与后继 [1,T) 可组成 next-token 训练配对";
      conditions.push("此为静态行切片，不执行 tokenizer、随机采样或 attention；超出序列范围的切片必须拒绝");
    }else if(kind==="gather_rows"){
      const [x,idx]=ports(n,inputs,"x","indices");by(idx[1]===1,"Gather indices must be [T,1]",n.id);
      output=[idx[0],x[1]];rule="整数行索引收集输入行；重复索引意味着重复读取相同张量";
      conditions.push("概念模式不检查每个整数 index 是否越界");
    }else if(kind==="scatter_add"){
      const [idx,v]=ports(n,inputs,"indices","values");by(idx[1]===1&&idx[0]===v[0],"Scatter index and values rows mismatch",n.id);
      output=[p.rows,v[1]];rule="按目标 index 将多个源行加和，重复索引不能覆盖之前的贡献";
      conditions.push("概念模式不检查 index 合法性");
    }else if(kind==="repeat_kv_rows"){
      const x=get("x");by(x[0]*p.groups<=8,"KV repeat exceeds bounded heads",n.id);
      output=[x[0]*p.groups,x[1]];rule="示例将行视作 KV Head，复制每行给多个 Query Head；真实张量应保持 Head/Token 轴区分";
    }else if(kind==="sliding_window_mask"){
      const x=get("x");by(x[0]===x[1],"Sliding window mask needs square logits",n.id);
      output=x.slice();rule="窗口 w 内因果可见性布尔矩阵，不进行概率归一化";
    }else if(kind==="router_topk"){
      const x=get("x");by(p.k<=x[1],"Router TopK k exceeds number of experts",n.id);
      output=[x[0],p.k];rule="按每行专家 logits 取最大 k 个 index；选中权重在额外内部张量中";
    }else if(kind==="weighted_sum"){
      const [w,v]=ports(n,inputs,"weights","values");by(w[1]===v[0],"Weighted sum reduce axis mismatch",n.id);
      output=[w[0],v[1]];rule="沿键/来源索引维加权聚合，注意权重不自动归一化";
    }else if(kind==="low_rank_project"){
      const [x,a,b]=ports(n,inputs,"x","down","up");
      by(x[1]===a[0]&&a[1]===b[0],"Low-rank projection contracted dimensions mismatch",n.id);
      output=[x[0],b[1]];rule="X·Down => latent [T,R] => Up => [T,Dout]";
      stages={latent:[x[0],a[1]]};
    }else if(kind==="row_bias_add"){
      const [score,bias]=ports(n,inputs,"scores","bias");
      by(bias[0]===1&&bias[1]===score[1],"Row Bias requires [1,E]",n.id);
      output=score.slice();rule="广播专家选择 bias，不改变原始 gate 分数的数学定义";
    }else if(kind==="group_topk_mask"){
      const score=get("scores");by(score[1]%p.groups===0&&p.keepGroups<=p.groups,"Group topk partition invalid",n.id);
      output=score.slice();rule="组内两最高分聚合后选组，输出专家维 0/1 允许集合";
    }else if(kind==="masked_router_topk"){
      const [score,mask]=ports(n,inputs,"scores","mask");by(equal(score,mask),"Masked Router shape mismatch",n.id);
      by(p.k<=score[1],"Masked Router k exceeds experts",n.id);
      output=[score[0],p.k];rule="在布尔 mask 允许的专家中按分数排序选择 k 个离散索引";
      conditions.push("每行允许专家的数量不得少于 k；概念模式不检查实际 mask 值");
    }else if(kind==="selected_logprob"){
      const [logits,labels]=ports(n,inputs,"logits","labels");
      by(labels[0]===logits[0]&&labels[1]===1,"Selected Logprob labels must be [T,1]",n.id);
      output=[logits[0],1];rule="对每个 token 的指定动作计算 logits_y − logsumexp(logits)";
      conditions.push("labels 必须是合法整数类别；概念模式不检查实际类别");
    }else if(kind==="response_token_reshape"){
      const x=get("x");by(x[1]===1&&x[0]%p.responses===0,"Response Token Reshape expects [G*T,1]",n.id);
      output=[p.responses,x[0]/p.responses];rule="按照原 Token 顺序把展平 G*T 条动作对数概率组织成 G 个回答，每个 T 个 Token";
    }else if(kind==="response_advantage_broadcast"){
      const [a,x]=ports(n,inputs,"advantage","tokenGrid");
      by(a[0]===x[0]&&a[1]===1,"Advantage broadcast expects [G,1]",n.id);
      output=x.slice();rule="每个回答内对各 Token 重复同一个组相对 Advantage，不跨回答混合";
    }else if(["policy_importance_ratio","ppo_clipped_surrogate","sampled_reference_kl"].includes(kind)){
      const pairNames=kind==="policy_importance_ratio"?["current","old"]:kind==="ppo_clipped_surrogate"?["ratio","advantage"]:["current","reference"];
      const [a,b]=ports(n,inputs,...pairNames);
      by(equal(a,b),"Policy ratio / surrogate / KL inputs must match",n.id);
      output=a.slice();
      rule=kind==="policy_importance_ratio"?"同一已选 Token 动作的新旧 logprob 差取指数":kind==="ppo_clipped_surrogate"?"对带正负号 Advantage 的策略贡献取两个备选表达式的 min":"以 exp(ref-current)−(ref-current)−1 计算采样 KL 惩罚估计";
      conditions.push("数值模式另行检查指数范围、比率合法性与数值有限性");
    }else if(kind==="completion_length_mask"){
      output=[p.responses,p.tokens];rule="每个回答依实际有效长度产生二值 completion Token Mask，禁止 padding 参与目标归约";
    }else if(kind==="masked_grpo_objective"){
      const [a,b,m]=ports(n,inputs,"surrogate","kl","mask");
      by(equal(a,b)&&equal(a,m),"GRPO surrogate/KL/mask shapes must match",n.id);
      output=[1,1];rule="逐回答按有效 Token 数求均值，再对 G 个回答取均值，减去 β 倍采样参考 KL";
      stages={per_response:[a[0],1]};
      conditions.push("Mask 每行必须含至少一个有效 Token；数值模式检查实际二值取值");
    }else if(kind==="group_advantage"){
      const rewards=get("rewards");output=rewards.slice();
      rule="每行 reward 组分别中心化并按组方差尺度标准化；等值组的优势全部为零";
      conditions.push("教学 epsilon 定义；不是完整 GRPO 的 clipped objective");
    }else if(kind==="attention"){
      const [q,k,v]=ports(n,inputs,"q","k","v");
      by(q[1]===k[1],"Attention Q/K 特征宽度 d_k 不一致",n.id);
      by(k[0]===v[0],"Attention K/V Token 数不相等",n.id);
      output=[q[0],v[1]];
      rule="QKᵀ/√d → Softmax(K 轴) → PV；输出 Query 长度与 Value 通道数";
      stages={score:[q[0],k[0]],probabilities:[q[0],k[0]]};
    }else if(kind==="gqa"){
      const [q,k,v]=ports(n,inputs,"q","k","v");
      by(k[0]===v[0],"GQA K/V Token 数不相等",n.id);
      by(p.heads%p.kvHeads===0,"GQA Query heads 必须能被 KV heads 整除",n.id);
      by(q[1]%p.heads===0 && k[1]%p.kvHeads===0 && v[1]%p.kvHeads===0,
        "GQA 投影宽度与 Head 数不整除",n.id);
      const dk=q[1]/p.heads,dk2=k[1]/p.kvHeads,dv=v[1]/p.kvHeads;
      by(dk===dk2,"GQA 每个 Q/K Head 的 Dk 必须相同",n.id);
      if(p.rope)by(dk%2===0,"RoPE 要求偶数 Dk",n.id);
      if(p.causal)by(k[0]>=q[0],"Causal GQA Key 前缀不能短于 Query",n.id);
      output=[q[0],p.heads*dv];by(output[1]<=16,"GQA 输出宽度超出微型合同",n.id);
      rule="Hq/Hkv 分组共享 K/V → RoPE(可选) → QKᵀ/√Dk → Mask/Softmax → PV；不更新 Cache";
      stages={q_heads:[1,p.heads,q[0],dk],k_heads:[1,p.kvHeads,k[0],dk],
        v_heads:[1,p.kvHeads,k[0],dv],score:[1,p.heads,q[0],k[0]],
        probabilities:[1,p.heads,q[0],k[0]],context:[1,p.heads,q[0],dv]};
    }else if(["mse","mae_loss"].includes(kind)){
      const [a,b]=ports(n,inputs,"prediction","target");
      by(equal(a,b),"Loss prediction 和 target 的形状必须相同",n.id);
      output=[1,1];rule="在全部对应元素上定义误差，再聚合为一个标量 Loss";
      conditions.push("仅描述前向损失；没有反向梯度和参数更新");
    }else if(kind==="kl_prob"){
      const [a,b]=ports(n,inputs,"p","q");
      by(equal(a,b),"KL(P∥Q) 两个概率分布的形状必须相同",n.id);
      output=[1,1];rule="按类别求 KL(P∥Q)，再按 token 均值聚合";
      conditions.push("P、Q 必须是合法概率分布，非负、逐行和为 1；概念模式不校验具体数值");
    }else if(kind==="cross_entropy"){
      const [l,y]=ports(n,inputs,"logits","labels");
      by(y[0]===l[0]&&y[1]===1,"CrossEntropy labels 必须为 [T,1]",n.id);
      output=[1,1];rule="每 token 按目标整数类别计算 logsumexp，再按 token 平均";
      conditions.push("标签必须是 [0,C) 整数，定性模式无法判断具体标签是否越界");
    }else if(kind==="causal_mask"){
      const x=get("x");by(x[0]<=16,"Causal Mask 长度超出界限",n.id);
      output=[x[0],x[0]];rule="由 Query/Key 时间先后关系产生 T×T 下三角布尔约束";
    }else if(kind==="token_depth_mask"){
      const d=get("depths");by(d[1]===1,"Token budgets must use [T,1] integer ID shape",n.id);
      output=[d[0],p.width];
      rule="逐 Token 离散预算产生 [T,D] 二进制持续迭代 Mask；只改变隐藏状态更新语义，不减少稠密矩阵乘";
      conditions.push("depths 数值必须为 0..3 的整数；静态模式不检查具体取值");
    }else if(kind==="state_mask_blend"){
      const [old,candidate,mask]=ports(n,inputs,"previous","candidate","mask");
      by(equal(old,candidate)&&equal(old,mask),"state blend requires same [T,D] shapes",n.id);
      output=old.slice();rule="M=1 采用本轮候选状态，M=0 严格保留之前已退出的 Token 表征";
      conditions.push("mask 必须为 0/1；即使 M=0，稠密教学后端仍会执行全部候选分支");
    }else if(kind==="masked_softmax"){
      const [logits,mask]=ports(n,inputs,"logits","mask");
      by(equal(logits,mask),"Masked Softmax logits/mask 形状必须相同",n.id);
      output=logits.slice();rule="只对 Mask 允许的位置归一化；被禁止位置的概率严格为零";
      conditions.push("每行至少有一个有效位置且 Mask 为 0/1；概念模式不检查数值");
    }else if(kind==="argmax"){
      const x=get("x");output=[x[0],1];rule="每行选最大类别下标；相等时选最左位置";
    }else if(kind==="topk"){
      const x=get("x");by(p.k<=x[1],"Top-k 的 k 不能超过特征宽度",n.id);
      output=[x[0],p.k];rule="保留每行分数最高的 k 个候选；不进行随机采样";
    }else{
      fail("尚无显式的定性形状合同："+kind,n.id);
    }
    if(kind.startsWith("nd_"))by(output.length>=2&&output.length<=4&&output.every(x=>Number.isSafeInteger(x)&&x>=1&&x<=8)&&output.reduce((a,b)=>a*b,1)<=512,"N-D inferred tensor exceeds rank/axis/512-element bound",n.id);
    return {shape:output,rule,conditions,stages,meaning:groups[L.defs[kind].category]||"数学计算"};
  }
  function evaluate(raw){
    const g=L.validateGraph(raw),seq=L.order(g),saved=new Map(),steps=[];
    for(const n of seq){
      const inputs={};
      for(const port of L.defs[n.kind].ports){
        const e=g.edges.find(e=>e.to===n.id&&e.port===port);
        by(e,L.defs[n.kind].title+" 的 "+port+" 端口未连接",n.id);
        const prev=saved.get(e.from);by(prev,"计算依赖失效："+e.from,n.id);
        inputs[port]=prev.shape.slice();
      }
      const summary=analyze(n,inputs);
      saved.set(n.id,summary);
      steps.push({id:n.id,kind:n.kind,inputs,shape:summary.shape,values:null,flops:null,
        mode:"concept",stages:summary.stages,conditions:summary.conditions,
        shapeEvidence:"declared_parameter_and_graph_shape_inference",
        meaning:summary.meaning,transformation:summary.rule,
        formula:L.defs[n.kind].formula,why:L.defs[n.kind].why,
        training:"概念推演：不执行反向传播，不生成或更新模型参数"});
    }
    return {mode:"concept",steps,totalKnownFlops:null,exactFlopCoverage:null,
      evidence:EVIDENCE,noNumericalExecution:true};
  }
  if(CATALOG.length!==new Set(CATALOG).size)throw Error("Concept flow duplicate block ids");
  return Object.freeze({evaluate,analyze,EVIDENCE,coveredKinds:CATALOG.slice()});
});
