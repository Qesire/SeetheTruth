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
    activate:"非线性与权重分布",combine:"信息融合与残差候选",
    attention:"跨 token 信息汇聚",loss:"训练目标（前向定义）",
    shape:"张量轴与布局改变",reduce:"跨特征聚合",position:"位置信息注入",
    gated:"门控 FFN 的逐通道信息控制",sampling:"输出决策或候选筛选",
    regularize:"训练/推理行为差异"
  });
  const unary=new Set(["rmsnorm","silu","softmax","layernorm","l2norm",
    "relu","gelu_tanh","sigmoid","tanh","softplus","scale","bias","square",
    "dropout_demo","pos_encoding","rope2d","temperature"]);
  const binary=new Set(["add","mul","subtract","divide","swiglu","geglu"]);
  const reduced=new Set(["row_mean","row_sum","row_var"]);
  const losses=new Set(["mse","mae_loss","kl_prob"]);
  const ports=(n,inputs,...keys)=>keys.map(k=>{by(inputs[k],n.kind+" 缺少 "+k+" 输入",n.id);return inputs[k];});
  function analyze(n,inputs){
    const p={...L.defs[n.kind].params,...n.params},kind=n.kind;
    const get=k=>ports(n,inputs,k)[0];
    let output,rule,conditions=[],stages=null;
    if(kind==="input"){output=[p.rows,p.width];rule="输入连续表征，建立 token 轴 T 和特征轴 D";}
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
  if(CATALOG.length!==45)throw Error("Concept flow must be checked against complete 45-block catalog");
  return Object.freeze({evaluate,analyze,EVIDENCE,coveredKinds:CATALOG.slice()});
});
