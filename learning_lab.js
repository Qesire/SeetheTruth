/* SeetheTruth Learning Lab: deterministic, browser-only tensor micro-execution.
 * This is a teaching interpreter for bounded 2-D matrices, never PyTorch,
 * never an imported model, and never a claim about training or GPU kernels.
 */
(function(root, make){
  "use strict";
  const api=make();
  if(typeof module==="object"&&module.exports)module.exports=api;
  root.SeetheTruthLearning=api;
})(typeof window!=="undefined"?window:{},function(){
  "use strict";
  const SCHEMA="seethetruth-learning-lab-v1";
  const TensorIR=(typeof module==="object"&&module.exports)?require("./tensor_ir.js"):globalThis.SeetheTruthTensorIR;
  const BlockCatalog=(typeof module==="object"&&module.exports)?require("./block_catalog.js"):globalThis.SeetheTruthBlockCatalog;
  const MAX_NODES=48,MAX_EDGES=96;
  const defs=Object.freeze({
    input:{title:"输入张量",en:"Input",category:"data",ports:[],params:{rows:3,width:4,seed:1},
      formula:"X ∈ ℝ^{T×D}",why:"建立具有确定形状和数值的微型张量，作为后续计算的数据来源。",
      caution:"这些数值由正弦函数生成，不是真实 tokenizer、embedding 或训练数据。",
      lesson:"观察 token 轴 T 和特征轴 D。改变 D 后，看下游是否还能通过形状检查。"},
    linear:{title:"线性投影",en:"Linear",category:"compute",ports:["x"],params:{out:4},
      formula:"Y = XWᵀ + b",why:"在最后一个特征轴做可学习的仿射变换，改变表示维度。",
      caution:"权重与偏置是固定的教学数值，没有优化或训练。",
      lesson:"对比 in/out 维度，识别 GEMM 的收缩维，以及理论乘加量。"},
    rmsnorm:{title:"RMSNorm",en:"RMSNorm",category:"norm",ports:["x"],params:{epsilon:0.00001},
      formula:"yᵢ = xᵢ / √(meanⱼ xⱼ² + ε)",why:"按每个 token 的特征均方根调整尺度。",
      caution:"此处 γ 固定为 1，未模拟可训练缩放参数。",
      lesson:"比较归一化前后的每行平方均值；RMSNorm 不减去行均值。"},
    silu:{title:"SiLU",en:"SiLU",category:"activate",ports:["x"],params:{},
      formula:"SiLU(x)=xσ(x)",why:"非线性激活，使复合层能表示不止线性变换。",
      caution:"非线性算子的硬件代价依赖具体内核，这里不臆造 FLOPs。",
      lesson:"观察负数区域仍有连续的非零输出，与 ReLU 有何不同？"},
    add:{title:"残差加法",en:"Add",category:"combine",ports:["a","b"],params:{},
      formula:"Y = A + B",why:"让两路张量信息合流；当一支是原输入时构成残差路径。",
      caution:"只有显式连回原输入才是残差，不把所有 Add 都叫残差。",
      lesson:"连接旁路跳过中间变换的 Input，检查两路形状是否完全一致。"},
    mul:{title:"逐元素门控",en:"Multiply",category:"combine",ports:["a","b"],params:{},
      formula:"Y = A ⊙ B",why:"用逐元素乘法表示激活与门控信号的相互作用。",
      caution:"这个算子本身不说明门控信号如何获得。",
      lesson:"如果 B 接近 0，对 A 信息的传递会发生什么？"},
    matmul:{title:"矩阵乘法",en:"MatMul",category:"compute",ports:["a","b"],params:{},
      formula:"Cᵢⱼ = Σₖ AᵢₖBₖⱼ",why:"沿共享维度求和；Attention 和投影都依赖矩阵收缩。",
      caution:"输入轴代表什么语义必须根据实际上下文决定。",
      lesson:"观察 A[T,K] × B[K,D] → C[T,D]；故意改变 K 看错误提示。"},
    softmax:{title:"Softmax",en:"Softmax",category:"activate",ports:["x"],params:{},
      formula:"pⱼ = exp(xⱼ−max(x)) / Σₖ exp(xₖ−max(x))",
      why:"在每一行形成和为 1 的权重分布，数值上使用最大值平移。",
      caution:"它只是行归一化；mask、缩放和注意力轴要单独说明。",
      lesson:"检查每行概率之和、缩放 logits 对权重集中度的影响。"},
    attention:{title:"缩放点积注意力",en:"SDP Attention",category:"attention",ports:["q","k","v"],params:{},
      formula:"S=QKᵀ/√dₖ; P=softmax(S); O=PV",
      why:"通过 query 与 key 的相关性，为每个 query 加权汇聚 value。",
      caution:"只模拟单头、无 mask、无 dropout 的显式数学形式；没有 FlashAttention、KV Cache 或 GQA。",
      lesson:"展开分数矩阵与概率矩阵：Q 长度与 K/V 长度可以不同，但 K 和 V 的 token 数必须一致。"},
    gqa:{title:"多头 GQA / Causal / RoPE",en:"GQA Attention",category:"attention",ports:["q","k","v"],
      params:{heads:2,kvHeads:1,causal:true,rope:true},
      formula:"head(Q)·head(K)ᵀ/√d → causal mask → softmax → P·head(V)",
      why:"多个 query head 分组共享少量 KV head，可分别观察 heads、位置旋转、Mask 与注意力矩阵。",
      caution:"输入为已有的投影矩阵；K/V 全量输入可模拟缓存前缀读取，但不会真实分配或更新 KV Cache。单批次、无 dropout、无训练。",
      lesson:"改变 Hq/Hkv、causal 和 rope；按 head 展开 Q/K/V、scores、probabilities、context，观察轴语义以及错误维度。",
    },
    mse:{title:"均方误差",en:"MSE Loss",category:"loss",ports:["prediction","target"],params:{},
      formula:"L = mean((prediction−target)²)",
      why:"将预测和目标张量的差异化为标量优化目标。",
      caution:"这里只计算前向 Loss；梯度、反向传播和优化器尚未实现。",
      lesson:"Loss 标量并不等于完成一次训练；下一步还需导数与参数更新。"},
    ...BlockCatalog.defs
  });
  const KINDS=Object.keys(defs);
  function fail(message,node){const e=new Error(message);e.nodeId=node||null;throw e;}
  function finite(v){return typeof v==="number"&&Number.isFinite(v);}
  function shape(m){return [m.length,m[0].length];}
  function matrix(rows,cols,fn){return Array.from({length:rows},(_,r)=>Array.from({length:cols},(_,c)=>fn(r,c)));}
  function safeDim(x,node,label,max=16){
    if(!Number.isSafeInteger(x)||x<1||x>max)fail(label+" 必须是 1–"+max+" 的整数",node);
    return x;
  }
  function validateGraph(raw){
    if(!raw||raw.schema!==SCHEMA||!Array.isArray(raw.nodes)||!Array.isArray(raw.edges))fail("无效课程工程 JSON/schema");
    if(!raw.nodes.length||raw.nodes.length>MAX_NODES||raw.edges.length>MAX_EDGES)fail("积木或连线超出限制");
    const ids=new Set(), nodes=[];
    for(const n of raw.nodes){
      if(!n||typeof n!=="object"||typeof n.id!=="string"||!/^[A-Za-z0-9_-]{1,36}$/.test(n.id)||ids.has(n.id))fail("积木 ID 非法或重复");
      if(!Object.hasOwn(defs,n.kind))fail("未知积木类型",n.id);
      if(!finite(n.x)||!finite(n.y)||Math.abs(n.x)>10000||Math.abs(n.y)>10000)fail("积木坐标必须有限",n.id);
      const p=n.params||{}, type=n.kind;
      if(!p||typeof p!=="object"||Array.isArray(p))fail("无效参数",n.id);
      const params={...defs[type].params};
      for(const key of Object.keys(params)){
        if(Object.hasOwn(p,key))params[key]=p[key];
      }
      if(type==="input"){
        safeDim(params.rows,n.id,"token 数",8);
        safeDim(params.width,n.id,"特征维度");
        if(!Number.isSafeInteger(params.seed)||Math.abs(params.seed)>10000)fail("seed 必须是 [-10000,10000] 整数",n.id);
      }
      if(type==="linear")safeDim(params.out,n.id,"输出维度");
      if(type==="gqa"){
        safeDim(params.heads,n.id,"query heads",8);
        safeDim(params.kvHeads,n.id,"KV heads",8);
        if(params.heads%params.kvHeads)fail("Query head 数必须能被 KV head 数整除",n.id);
        for(const flag of ["causal","rope"])if(typeof params[flag]!=="boolean")fail(flag+" 必须是布尔值",n.id);
      }
      if(type==="rmsnorm"&&(!finite(params.epsilon)||params.epsilon<=0||params.epsilon>1))fail("epsilon 必须在 (0,1] 内",n.id);
      if(Object.hasOwn(BlockCatalog.defs,type)){try{BlockCatalog.validate(type,params)}catch(err){fail(err.message,n.id);}}
      ids.add(n.id);nodes.push({id:n.id,kind:n.kind,x:n.x,y:n.y,params});
    }
    const targetPorts=new Set(),edges=[];
    for(const e of raw.edges){
      if(!e||!ids.has(e.from)||!ids.has(e.to)||e.from===e.to)fail("连线端点无效");
      const n=nodes.find(x=>x.id===e.to);
      if(!defs[n.kind].ports.includes(e.port))fail("目标输入端口无效",e.to);
      const key=e.to+":"+e.port;
      if(targetPorts.has(key))fail("同一输入端口不可连接两次",e.to);
      targetPorts.add(key);edges.push({from:e.from,to:e.to,port:e.port});
    }
    return {schema:SCHEMA,title:typeof raw.title==="string"?raw.title.slice(0,80):"未命名实验",
      nodes,edges};
  }
  function order(graph){
    const byId=new Map(graph.nodes.map(n=>[n.id,n]));
    const incoming=new Map(graph.nodes.map(n=>[n.id,0]));
    const adj=new Map(graph.nodes.map(n=>[n.id,[]]));
    for(const e of graph.edges){incoming.set(e.to,incoming.get(e.to)+1);adj.get(e.from).push(e.to);}
    const q=graph.nodes.filter(n=>!incoming.get(n.id)).map(n=>n.id),sorted=[];
    for(let i=0;i<q.length;i++){
      const id=q[i];sorted.push(byId.get(id));
      for(const to of adj.get(id)){incoming.set(to,incoming.get(to)-1);if(incoming.get(to)===0)q.push(to);}
    }
    if(sorted.length!==graph.nodes.length)fail("检测到循环计算依赖：必须是有向无环图");
    return sorted;
  }
  function hashWeight(id,r,c){let h=2166136261;for(const ch of id+":"+r+":"+c){h=Math.imul(h^ch.charCodeAt(0),16777619);}return ((h>>>0)%2001-1000)/2600;}
  function map1(a,fn){return a.map(row=>row.map(fn));}
  function map2(a,b,fn,id){
    if(shape(a).some((v,i)=>v!==shape(b)[i]))fail("逐元素操作要求两路形状完全一致",id);
    return a.map((row,i)=>row.map((v,j)=>fn(v,b[i][j])));
  }
  function dot(a,b,id){
    const [m,k]=shape(a),[kk,n]=shape(b);
    if(k!==kk)fail("矩阵乘法内维不匹配：A="+m+"×"+k+"，B="+kk+"×"+n,id);
    return matrix(m,n,(i,j)=>{let s=0;for(let t=0;t<k;t++)s+=a[i][t]*b[t][j];return s;});
  }
  function transpose(a){return matrix(a[0].length,a.length,(i,j)=>a[j][i]);}
  function rowSoftmax(a){
    return a.map(row=>{const max=Math.max(...row),exp=row.map(x=>Math.exp(x-max));
      const sum=exp.reduce((x,y)=>x+y,0);return exp.map(v=>v/sum);});
  }
  function calculate(n,inputs){
    let value,flops=null,internals=null;
    const x=inputs.x,a=inputs.a,b=inputs.b;
    switch(n.kind){
      case "input":value=matrix(n.params.rows,n.params.width,(i,j)=>
        Math.sin((i+1)*(j+1)*0.71+n.params.seed*0.37)*0.8);break;
      case "linear":{
        const [rows,di]=shape(x),dout=n.params.out;
        value=matrix(rows,dout,(i,j)=>{let sum=hashWeight(n.id,97,j);
          for(let k=0;k<di;k++)sum+=x[i][k]*hashWeight(n.id,j,k);return sum;});
        flops=rows*di*dout*2;break;
      }
      case "rmsnorm":value=x.map(row=>{const inv=1/Math.sqrt(row.reduce((s,v)=>s+v*v,0)/row.length+n.params.epsilon);return row.map(v=>v*inv);});break;
      case "silu":value=map1(x,v=>v/(1+Math.exp(-v)));break;
      case "add":value=map2(a,b,(u,v)=>u+v,n.id);break;
      case "mul":value=map2(a,b,(u,v)=>u*v,n.id);break;
      case "matmul":value=dot(a,b,n.id);flops=2*a.length*a[0].length*b[0].length;break;
      case "softmax":value=rowSoftmax(x);break;
      case "attention":{
        const q=inputs.q,k=inputs.k,v=inputs.v;
        if(q[0].length!==k[0].length)fail("Attention Q/K 的特征维度 dₖ 必须相同",n.id);
        if(k.length!==v.length)fail("Attention K/V 的 token 数必须相同",n.id);
        const d=q[0].length, scores=dot(q,transpose(k),n.id).map(row=>row.map(z=>z/Math.sqrt(d)));
        const weights=rowSoftmax(scores);
        value=dot(weights,v,n.id);
        flops=2*q.length*k.length*(d+v[0].length);
        internals={scores:{shape:shape(scores),values:scores},
                   probabilities:{shape:shape(weights),values:weights}};
        break;
      }
      case "gqa":{
        if(!TensorIR)fail("Tensor IR 尚未加载，请确认 tensor_ir.js 已正确加载",n.id);
        let r;
        try{r=TensorIR.gqa(inputs.q,inputs.k,inputs.v,n.params);}
        catch(e){fail("GQA: "+e.message,n.id);}
        value=r.output;flops=r.knownMatmulFlops;
        internals=Object.fromEntries(Object.entries(r.stages).map(([label,tensor])=>{
          const preview=TensorIR.plane(tensor);
          return [label,{shape:preview.shape,axes:preview.axes,values:preview.view,
                         selected:preview.selected,tensor}];
        }));
        internals.causal_mask=r.masks?{
          shape:r.masks.shape,axes:["Tq","Tk"],
          values:r.masks.allowed.map(row=>row.map(x=>x?1:0)),
          note:"1=allowed, 0=masked (excluded exactly in Softmax)"}:undefined;
        if(!r.masks)delete internals.causal_mask;
        internals.explanation={
          shape:[1,1],values:[[r.teachingConfig.groupSize]],
          note:"GQA group size Hq / Hkv; Q offset = "+r.teachingConfig.queryPositionOffset+
               "; K/V 全量已提供，不代表有可更新的 KV Cache"};
        break;
      }
      case "mse":{
        const sq=map2(inputs.prediction,inputs.target,(u,v)=>(u-v)**2,n.id);
        value=[[sq.flat().reduce((u,v)=>u+v,0)/(sq.length*sq[0].length)]];
        break;
      }
      default:{
        if(!Object.hasOwn(BlockCatalog.defs,n.kind))fail("尚不支持该积木",n.id);
        let r;try{r=BlockCatalog.execute(n,inputs)}catch(err){fail(err.message,n.id);}
        value=r.values;flops=r.flops;internals=r.internals;
        break;
      }
    }
    if(!value.every(row=>row.every(Number.isFinite)))fail("计算产生了 NaN 或 Inf",n.id);
    return {shape:shape(value),values:value,flops,internals};
  }
  function evaluate(raw){
    const graph=validateGraph(raw), sequence=order(graph);
    const out=new Map(),steps=[];
    for(const n of sequence){
      const ins={};
      for(const port of defs[n.kind].ports){
        const edge=graph.edges.find(e=>e.to===n.id&&e.port===port);
        if(!edge)fail(defs[n.kind].title+" 的 "+port+" 端口还没有连接",n.id);
        ins[port]=out.get(edge.from)?.values;
        if(!ins[port])fail("上游计算结果不存在",n.id);
      }
      const value=calculate(n,ins);out.set(n.id,value);
      steps.push({id:n.id,kind:n.kind,inputs:Object.fromEntries(Object.entries(ins).map(([k,v])=>[k,shape(v)])),
        ...value,formula:defs[n.kind].formula,why:defs[n.kind].why,
        training:"仅前向微型数值解释；尚未执行反向传播与权重更新。"});
    }
    return {steps,totalKnownFlops:steps.reduce((s,x)=>s+(x.flops||0),0),
      exactFlopCoverage:"only_linear_matmul_attention_gemm",
      evidence:"deterministic_browser_teaching_simulation_not_model_execution"};
  }
  function connect(raw,from,to,port){
    const g=validateGraph(raw);
    g.edges=g.edges.filter(e=>!(e.to===to&&e.port===port));
    g.edges.push({from,to,port});
    const validated=validateGraph(g);order(validated);return validated;
  }
  function node(id,kind,x,y,params={}){return{id,kind,x,y,params};}
  function preset(name){
    const N=node;
    let title,nodes,edges;
    if(name==="mla_cache"){
      title="任务 17 · DeepSeek MLA · 压缩 KV Cache 与权重吸收的两路等价对照";
      nodes=[
        N("c_old","input",20,30,{rows:2,width:2,seed:1}),
        N("c_new","input",20,230,{rows:1,width:2,seed:2}),
        N("c_write","cache_write_segment",270,75,{start:2,capacity:8}),
        N("c_read","cache_read_prefix",530,75,{length:3}),
        N("pe_old_raw","input",20,440,{rows:2,width:2,seed:3}),
        N("pe_old_rope","rope2d",260,440,{offset:0}),
        N("pe_new_raw","input",20,650,{rows:1,width:2,seed:4}),
        N("pe_new_rope","rope2d",270,650,{offset:2}),
        N("pe_write","cache_write_segment",530,470,{start:2,capacity:8}),
        N("pe_read","cache_read_prefix",800,470,{length:3}),
        N("q_nope","input",20,850,{rows:1,width:2,seed:5}),
        N("q_pe_raw","input",20,1050,{rows:1,width:2,seed:6}),
        N("q_pe","rope2d",280,1050,{offset:2}),
        N("w_k","input",520,850,{rows:2,width:2,seed:7}),
        N("w_v","input",520,1050,{rows:2,width:2,seed:8}),
        N("k_explicit","matmul",790,20),
        N("k_transpose","transpose2d",1050,20),
        N("score_explicit","matmul",1310,30),
        N("q_abs","query_weight_absorb",800,250),
        N("c_transpose","transpose2d",1070,250),
        N("score_abs","matmul",1320,260),
        N("pe_transpose","transpose2d",1070,490),
        N("score_position","matmul",1320,490),
        N("sum_explicit","add",1600,40),
        N("sum_abs","add",1600,260),
        N("scaled_explicit","scale",1840,40,{factor:0.5}),
        N("scaled_abs","scale",1840,260,{factor:0.5}),
        N("probs_explicit","softmax",2080,40),
        N("probs_abs","softmax",2080,260),
        N("v_explicit","matmul",1050,730),
        N("out_explicit","matmul",2330,40),
        N("latent_read","matmul",2330,250),
        N("out_abs","matmul",2580,250),
        N("neg_abs","scale",2820,250,{factor:-1}),
        N("difference","add",3060,130)
      ];
      edges=[
        ["c_old","c_write","cache"],["c_new","c_write","segment"],["c_write","c_read","cache"],
        ["pe_old_raw","pe_old_rope","x"],["pe_new_raw","pe_new_rope","x"],
        ["pe_old_rope","pe_write","cache"],["pe_new_rope","pe_write","segment"],["pe_write","pe_read","cache"],
        ["q_pe_raw","q_pe","x"],
        ["c_read","k_explicit","a"],["w_k","k_explicit","b"],["k_explicit","k_transpose","x"],
        ["q_nope","score_explicit","a"],["k_transpose","score_explicit","b"],
        ["q_nope","q_abs","query"],["w_k","q_abs","keyWeight"],
        ["c_read","c_transpose","x"],["q_abs","score_abs","a"],["c_transpose","score_abs","b"],
        ["pe_read","pe_transpose","x"],["q_pe","score_position","a"],["pe_transpose","score_position","b"],
        ["score_explicit","sum_explicit","a"],["score_position","sum_explicit","b"],
        ["score_abs","sum_abs","a"],["score_position","sum_abs","b"],
        ["sum_explicit","scaled_explicit","x"],["sum_abs","scaled_abs","x"],
        ["scaled_explicit","probs_explicit","x"],["scaled_abs","probs_abs","x"],
        ["c_read","v_explicit","a"],["w_v","v_explicit","b"],
        ["probs_explicit","out_explicit","a"],["v_explicit","out_explicit","b"],
        ["probs_abs","latent_read","a"],["c_read","latent_read","b"],
        ["latent_read","out_abs","a"],["w_v","out_abs","b"],
        ["out_abs","neg_abs","x"],["out_explicit","difference","a"],["neg_abs","difference","b"]
      ];
    }else if(name==="moe_expert_ffn"){
      title="任务 16 · MoE · 逐 Token 专家 SwiGLU FFN 与共享专家";
      nodes=[
        N("x","input",25,50,{rows:3,width:2,seed:4}),
        N("router","linear",25,240,{out:4}),
        N("affinity","sigmoid",25,450),
        N("bias","input",25,670,{rows:1,width:4,seed:9}),
        N("biased","row_bias_add",310,60),
        N("groups","group_topk_mask",310,270,{groups:2,keepGroups:1}),
        N("select","masked_router_topk",310,480,{k:2}),
        N("gates","router_gate_weights",310,690,{mode:"identity",normalize:true}),
        N("dispatch","dispatch_routed_tokens",595,70),
        N("w1","input",595,280,{rows:8,width:2,seed:10}),
        N("w3","input",595,490,{rows:8,width:2,seed:11}),
        N("w2","input",595,700,{rows:8,width:2,seed:12}),
        N("expert_gate","expert_linear_bank",880,50),
        N("expert_up","expert_linear_bank",880,260),
        N("expert_swiglu","swiglu",1155,120),
        N("expert_down","expert_linear_bank",1440,140),
        N("routed","routed_weighted_reduce",1735,160),
        N("shared_gate","linear",880,470,{out:2}),
        N("shared_up","linear",880,700,{out:2}),
        N("shared_swiglu","swiglu",1165,610),
        N("shared_down","linear",1450,650,{out:2}),
        N("out","add",2010,370)
      ];
      edges=[
        ["x","router","x"],["router","affinity","x"],
        ["affinity","biased","scores"],["bias","biased","bias"],
        ["biased","groups","scores"],["biased","select","scores"],
        ["groups","select","mask"],["affinity","gates","scores"],["select","gates","indices"],
        ["x","dispatch","x"],["select","dispatch","indices"],
        ["dispatch","expert_gate","x"],["select","expert_gate","indices"],["w1","expert_gate","weights"],
        ["dispatch","expert_up","x"],["select","expert_up","indices"],["w3","expert_up","weights"],
        ["expert_gate","expert_swiglu","gate"],["expert_up","expert_swiglu","up"],
        ["expert_swiglu","expert_down","x"],["select","expert_down","indices"],["w2","expert_down","weights"],
        ["expert_down","routed","values"],["gates","routed","gates"],
        ["x","shared_gate","x"],["x","shared_up","x"],
        ["shared_gate","shared_swiglu","gate"],["shared_up","shared_swiglu","up"],
        ["shared_swiglu","shared_down","x"],["routed","out","a"],["shared_down","out","b"]
      ];
    }else if(name==="mla_lite"){
      title="任务 13 · MLA-inspired · KV/Q 低秩与解耦 RoPE（单头教学）";
      nodes=[
        N("x","input",20,220,{rows:3,width:4,seed:1}),
        N("downkv","input",20,480,{rows:4,width:2,seed:2}),
        N("downq","input",20,610,{rows:4,width:2,seed:5}),
        N("kvlatent","matmul",260,270),
        N("qlatent","matmul",260,560),
        N("uk","input",280,20,{rows:2,width:2,seed:6}),
        N("uv","input",290,690,{rows:2,width:4,seed:7}),
        N("qrweight","input",525,660,{rows:2,width:2,seed:8}),
        N("krweight","input",280,690,{rows:4,width:2,seed:9}),
        N("qu","input",525,20,{rows:2,width:2,seed:10}),
        N("kc","matmul",525,155),
        N("v","matmul",525,430),
        N("kr","matmul",525,800),
        N("krrope","rope2d",780,760,{offset:0}),
        N("k","concat_cols",1030,340),
        N("qc","matmul",780,65),
        N("qr","matmul",780,550),
        N("qrrope","rope2d",1030,670,{offset:0}),
        N("q","concat_cols",1290,145),
        N("attention","attention",1570,230)
      ];
      edges=[
        ["x","kvlatent","a"],["downkv","kvlatent","b"],
        ["x","qlatent","a"],["downq","qlatent","b"],
        ["kvlatent","kc","a"],["uk","kc","b"],["kvlatent","v","a"],["uv","v","b"],
        ["x","kr","a"],["krweight","kr","b"],["kr","krrope","x"],
        ["kc","k","a"],["krrope","k","b"],
        ["qlatent","qc","a"],["qu","qc","b"],
        ["qlatent","qr","a"],["qrweight","qr","b"],["qr","qrrope","x"],
        ["qc","q","a"],["qrrope","q","b"],
        ["q","attention","q"],["k","attention","k"],["v","attention","v"]
      ];
    }else if(name==="moe_route"){
      title="任务 14 · MoE 分组路由 · 从 Bias 到专家索引";
      nodes=[
        N("x","input",25,160,{rows:3,width:4,seed:4}),
        N("score","linear",275,160,{out:4}),
        N("affinity","sigmoid",415,170),
        N("bias","input",275,520,{rows:1,width:4,seed:9}),
        N("biased","row_bias_add",540,185),
        N("mask","group_topk_mask",800,35,{groups:2,keepGroups:1}),
        N("select","masked_router_topk",1060,215,{k:2}),
        N("experts","input",795,490,{rows:4,width:4,seed:7}),
        N("gather","gather_routed_bank",1310,290),
        N("gates","router_gate_weights",1320,50,{mode:"identity",normalize:true}),
        N("mix","expert_bank_mix",1600,240)
      ];
      edges=[
        ["x","score","x"],["score","affinity","x"],["affinity","biased","scores"],["bias","biased","bias"],
        ["biased","mask","scores"],["biased","select","scores"],
        ["mask","select","mask"],["select","gather","indices"],["experts","gather","x"],
        ["affinity","gates","scores"],["select","gates","indices"],
        ["select","mix","indices"],["gates","mix","gates"],["experts","mix","experts"]
      ];
    }else if(name==="grpo_signals"){
      title="任务 15 · GRPO 前置信号 · Logprob 与组内优势";
      nodes=[
        N("logits","input",25,60,{rows:2,width:4,seed:2}),
        N("ids","token_ids",25,355,{rows:2,vocab:4,seed:3}),
        N("logp","selected_logprob",380,195),
        N("alllogp","log_softmax",380,20),
        N("rewards","input",25,580,{rows:2,width:4,seed:4}),
        N("advantages","group_advantage",390,530,{epsilon:0.00001})
      ];
      edges=[
        ["logits","logp","logits"],["ids","logp","labels"],
        ["logits","alllogp","x"],["rewards","advantages","rewards"]
      ];
    }else if(name==="residual"){
      title="任务 02 · 构建残差路径";
      nodes=[N("x","input",40,190,{rows:3,width:4,seed:2}),N("fc","linear",305,75,{out:4}),
             N("act","silu",550,75),N("sum","add",810,205)];
      edges=[["x","fc","x"],["fc","act","x"],["x","sum","a"],["act","sum","b"]];
    }else if(name==="attention"){
      title="任务 03 · 还原 Attention";
      nodes=[N("x","input",30,198,{rows:3,width:4,seed:3}),N("q","linear",285,42,{out:4}),
             N("k","linear",285,190,{out:4}),N("v","linear",285,340,{out:4}),
             N("attn","attention",565,187),N("proj","linear",855,187,{out:4})];
      edges=[["x","q","x"],["x","k","x"],["x","v","x"],["q","attn","q"],
             ["k","attn","k"],["v","attn","v"],["attn","proj","x"]];
    }else if(name==="gqa" || name==="decode"){
      const decode=name==="decode";
      title=decode?"任务 06 · GQA 的 Decode 与 KV 前缀":"任务 05 · 拆解 GQA / RoPE / Causal Attention";
      nodes=[
        N("xq","input",25,50,{rows:decode?1:3,width:4,seed:3}),
        N("xkv","input",25,370,{rows:3,width:4,seed:5}),
        N("q","linear",290,50,{out:8}),
        N("k","linear",290,260,{out:4}),
        N("v","linear",290,450,{out:4}),
        N("gqa","gqa",580,190,{heads:2,kvHeads:1,causal:true,rope:true}),
        N("out","linear",855,190,{out:4})
      ];
      edges=[["xq","q","x"],["xkv","k","x"],["xkv","v","x"],
             ["q","gqa","q"],["k","gqa","k"],["v","gqa","v"],
             ["gqa","out","x"]];
    }else if(name==="swiglu"){
      title="任务 07 · SwiGLU 门控 FFN";
      nodes=[N("x","input",25,190,{rows:3,width:4,seed:2}),
        N("gate","linear",280,75,{out:8}),N("up","linear",280,345,{out:8}),
        N("sw","swiglu",575,195),N("down","linear",860,195,{out:4})];
      edges=[["x","gate","x"],["x","up","x"],["gate","sw","gate"],
        ["up","sw","up"],["sw","down","x"]];
    }else if(name==="norm"){
      title="任务 08 · LayerNorm 与 RMSNorm";
      nodes=[N("x","input",25,190,{rows:3,width:4,seed:3}),
        N("ln","layernorm",310,75),N("rms","rmsnorm",310,330),
        N("ln_var","row_var",650,75),N("rms_var","row_var",650,330)];
      edges=[["x","ln","x"],["x","rms","x"],["ln","ln_var","x"],["rms","rms_var","x"]];
    }else if(name==="embedding"){
      title="任务 09 · Token ID → Embedding → 位置编码";
      nodes=[N("ids","token_ids",30,200,{rows:3,vocab:8,seed:1}),
        N("embed","embedding",310,200,{vocab:8,width:4}),
        N("pos","pos_encoding",620,200)];
      edges=[["ids","embed","ids"],["embed","pos","x"]];
    }else if(name==="classifier"){
      title="任务 10 · Classifier / Cross Entropy";
      nodes=[N("x","input",25,80,{rows:3,width:4,seed:2}),
        N("label","token_ids",30,375,{rows:3,vocab:8,seed:4}),
        N("logits","linear",320,80,{out:8}),
        N("soft","softmax",600,65),
        N("ce","cross_entropy",855,230)];
      edges=[["x","logits","x"],["logits","soft","x"],["logits","ce","logits"],
        ["label","ce","labels"]];
    }else if(name==="mask"){
      title="任务 11 · Causal Mask → Masked Softmax";
      nodes=[N("logits","input",25,150,{rows:3,width:3,seed:1}),
        N("mask","causal_mask",330,345),N("masked","masked_softmax",650,170)];
      edges=[["logits","mask","x"],["logits","masked","logits"],
        ["mask","masked","mask"]];
    }else if(name==="sampling"){
      title="任务 12 · Logits 温度 / Greedy / Top-k";
      nodes=[N("logits","input",25,180,{rows:3,width:8,seed:2}),
        N("temp","temperature",310,180,{temperature:0.7}),
        N("prob","softmax",605,40),N("greedy","argmax",605,220),
        N("top","topk",605,410,{k:3})];
      edges=[["logits","temp","x"],["temp","prob","x"],
        ["temp","greedy","x"],["temp","top","x"]];
    }else if(name==="loss"){
      title="任务 04 · 识别训练目标";
      nodes=[N("x","input",30,100,{rows:3,width:4,seed:1}),
             N("target","input",30,340,{rows:3,width:4,seed:9}),
             N("linear","linear",320,100,{out:4}),N("loss","mse",640,220)];
      edges=[["x","linear","x"],["linear","loss","prediction"],["target","loss","target"]];
    }else if(name==="blank"){
      title="自由创作 · 从零拼搭";nodes=[N("x","input",72,175,{rows:3,width:4,seed:1})];edges=[];
    }else{
      title="任务 01 · 从 FFN 开始";
      nodes=[N("x","input",38,170,{rows:3,width:4,seed:1}),N("up","linear",300,170,{out:8}),
             N("act","silu",560,170),N("down","linear",810,170,{out:4})];
      edges=[["x","up","x"],["up","act","x"],["act","down","x"]];
    }
    return validateGraph({schema:SCHEMA,title,nodes,edges:edges.map(([from,to,port])=>({from,to,port}))});
  }
  return Object.freeze({SCHEMA,defs,KINDS,MAX_NODES,MAX_EDGES,validateGraph,order,evaluate,connect,preset});
});
