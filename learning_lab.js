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
    if(n.kind.startsWith("nd_")){
      const out=BlockCatalog.execute(n,inputs);
      const ND=(typeof module==="object"&&module.exports)?require("./nd_tensor.js"):globalThis.SeetheTruthND;
      return {shape:ND.shape(out.values),values:out.values,flops:out.flops??null,internals:out.internals??null};
    }
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
    if(name==="activation_extension"){
      title="任务 34 · ELU / HardSwish · 双维度激活导数与拐点";
      nodes=[
        N("signal","input",30,225,{rows:2,width:4,seed:4}),
        N("elu_act","elu",350,60,{alpha:1}),
        N("hard_act","hardswish",350,330),
        N("tensor","nd_input",680,220,{b:1,h:2,t:2,d:4,seed:4}),
        N("elu4","nd_elu",1010,60,{alpha:1}),
        N("hard4","nd_hardswish",1010,330)
      ];
      edges=[["signal","elu_act","x"],["signal","hard_act","x"],
             ["tensor","elu4","x"],["tensor","hard4","x"]];
    }else if(name==="optimizer_decay_compare"){
      title="任务 35 · Adam vs AdamW · Adagrad vs RMSprop";
      nodes=[
        N("weights","parameter",30,160),
        N("gradient","input",30,520,{rows:2,width:2,seed:3}),
        N("adam","opt_adam",400,30,{lr:0.01,weightDecay:0.1}),
        N("adamw","opt_adamw",400,260,{lr:0.01,weightDecay:0.1}),
        N("adagrad","opt_adagrad",400,490,{lr:0.01,weightDecay:0.01,lrDecay:0}),
        N("rmsprop","opt_rmsprop",400,720,{lr:0.01,weightDecay:0.01,alpha:0.99})
      ];
      edges=nodes.slice(2).flatMap(n=>[["weights",n.id,"parameter"],["gradient",n.id,"gradient"]]);
    }else if(name==="activation_gallery"){
      title="任务 32 · 激活函数比较实验 · GELU/SiLU/Mish/ReLU²";
      nodes=[
        N("signal","input",30,300,{rows:2,width:4,seed:4}),
        N("gelu_cdf","gelu_erf",370,30),
        N("gelu_approx","gelu_tanh",370,230),
        N("mish_act","mish",370,430),
        N("relu_square","relu_squared",370,630),
        N("leaky","leaky_relu",700,230,{slope:0.1}),
        N("silu_act","silu",700,490)
      ];
      edges=nodes.slice(1).map(n=>["signal",n.id,"x"]);
    }else if(name==="optimizer_compare"){
      title="任务 33 · 同一梯度对照四种优化器 · 一步纯函数更新";
      nodes=[
        N("weights","parameter",30,190),
        N("gradient","input",30,520,{rows:2,width:2,seed:5}),
        N("adamw","opt_adamw",380,30,{lr:0.01,weightDecay:0.01}),
        N("sgd","opt_sgd_momentum",380,260,{lr:0.01,weightDecay:0.01,momentum:0.9}),
        N("rmsprop","opt_rmsprop",380,490,{lr:0.01,weightDecay:0.01,alpha:0.99}),
        N("lion","opt_lion",380,720,{lr:0.004,weightDecay:0.01})
      ];
      edges=nodes.slice(2).flatMap(n=>[["weights",n.id,"parameter"],["gradient",n.id,"gradient"]]);
    }else if(name==="nd_cached_gqa_decode"){
      title="任务 31 · 四轴 RoPE + GQA KV Cache · Prefill → Decode";
      nodes=[
        N("full_q","nd_input",25,35,{b:1,h:4,t:3,d:4,seed:2}),
        N("full_k","nd_input",25,320,{b:1,h:2,t:3,d:4,seed:4}),
        N("full_v","nd_input",25,770,{b:1,h:2,t:3,d:4,seed:6}),
        N("q_current","nd_cache_read",305,30,{start:2,length:1}),
        N("k_prefix","nd_cache_read",305,305,{start:0,length:2}),
        N("k_current","nd_cache_read",305,530,{start:2,length:1}),
        N("v_prefix","nd_cache_read",305,765,{start:0,length:2}),
        N("v_current","nd_cache_read",305,985,{start:2,length:1}),
        N("q_rope","nd_rope",590,30,{offset:2,rotaryDim:4,base:10000}),
        N("kp_rope","nd_rope",590,310,{offset:0,rotaryDim:4,base:10000}),
        N("kn_rope","nd_rope",590,535,{offset:2,rotaryDim:4,base:10000}),
        N("cached_k","nd_cache_append",880,420),
        N("cached_v","nd_cache_append",880,835),
        N("shared_k","nd_repeat_kv",1170,420,{queryHeads:4}),
        N("shared_v","nd_repeat_kv",1170,835,{queryHeads:4}),
        N("key_transpose","nd_permute",1455,410,{axis0:0,axis1:1,axis2:3,axis3:2}),
        N("scores","nd_batched_matmul",1730,180),
        N("scaled","nd_scale",2010,180,{factor:0.5}),
        N("causal","nd_causal_mask",2290,180,{queryStart:2,keyStart:0}),
        N("probabilities","nd_softmax_last",2570,180),
        N("context","nd_batched_matmul",2850,340)
      ];
      edges=[
       ["full_q","q_current","x"],["full_k","k_prefix","x"],
       ["full_k","k_current","x"],["full_v","v_prefix","x"],
       ["full_v","v_current","x"],["q_current","q_rope","x"],
       ["k_prefix","kp_rope","x"],["k_current","kn_rope","x"],
       ["kp_rope","cached_k","cache"],["kn_rope","cached_k","segment"],
       ["v_prefix","cached_v","cache"],["v_current","cached_v","segment"],
       ["cached_k","shared_k","x"],["cached_v","shared_v","x"],
       ["shared_k","key_transpose","x"],
       ["q_rope","scores","a"],["key_transpose","scores","b"],
       ["scores","scaled","x"],["scaled","causal","x"],
       ["causal","probabilities","x"],
       ["probabilities","context","a"],["shared_v","context","b"]
      ];
    }else if(name==="nd_trainable_mqa"||name==="nd_trainable_gqa"){
      const isMQA=name==="nd_trainable_mqa";
      const base=preset(isMQA?"nd_mqa_attention":"nd_gqa_attention");
      for(const id of ["query","key","value"]){
        const n=base.nodes.find(n=>n.id===id);
        const result=calculate(n,{});
        const ND=(typeof module==="object"&&module.exports)?require("./nd_tensor.js"):globalThis.SeetheTruthND;
        const x=result.values,sh=ND.shape(x),zero=ND.build(sh,()=>0);
        n.kind="nd_parameter";
        n.params={values:x,optimizer:{t:0,m:zero,v:ND.build(sh,()=>0)}};
      }
      return validateGraph({...base,title:isMQA?"任务 30 · MQA 四轴 Parameter + AdamW 完整训练":"任务 29 · GQA 四轴 Parameter + AdamW 完整训练"});
    }else if(name==="nd_mqa_attention"){
      const base=preset("nd_gqa_attention");
      base.title="任务 28 · MQA 一组 KV Head 与四组 Query Head 的共享梯度";
      for(const id of ["key","value"])base.nodes.find(n=>n.id===id).params.h=1;
      return validateGraph(base);
    }else if(name==="nd_gqa_attention"){
      title="任务 27 · GQA 四 Q Head / 两 KV Head · 多轴反向与 MSE";
      nodes=[
        N("query","nd_input",30,40,{b:1,h:4,t:3,d:2,seed:2}),
        N("key","nd_input",30,320,{b:1,h:2,t:3,d:2,seed:4}),
        N("value","nd_input",30,640,{b:1,h:2,t:3,d:2,seed:6}),
        N("key_groups","nd_repeat_kv",335,310,{queryHeads:4}),
        N("value_groups","nd_repeat_kv",335,610,{queryHeads:4}),
        N("key_transpose","nd_permute",640,295,{axis0:0,axis1:1,axis2:3,axis3:2}),
        N("scores","nd_batched_matmul",945,145),
        N("scaled","nd_scale",1245,140,{factor:0.7071067811865475}),
        N("causal","nd_causal_mask",1545,130,{queryStart:0,keyStart:0}),
        N("probabilities","nd_softmax_last",1850,130),
        N("context","nd_batched_matmul",2150,310),
        N("target","nd_input",1860,620,{b:1,h:4,t:3,d:2,seed:11}),
        N("loss","nd_mse",2470,345)
      ];
      edges=[
       ["key","key_groups","x"],["value","value_groups","x"],
       ["key_groups","key_transpose","x"],["query","scores","a"],["key_transpose","scores","b"],
       ["scores","scaled","x"],["scaled","causal","x"],
       ["causal","probabilities","x"],
       ["probabilities","context","a"],["value_groups","context","b"],
       ["context","loss","prediction"],["target","loss","target"]
      ];
    }else if(name==="nd_causal_attention"){
      title="任务 26 · 真正逐积木四轴因果多头 Attention";
      nodes=[
       N("query","nd_input",30,40,{b:1,h:2,t:3,d:4,seed:2}),
       N("key","nd_input",30,320,{b:1,h:2,t:3,d:4,seed:4}),
       N("value","nd_input",30,620,{b:1,h:2,t:3,d:4,seed:6}),
       N("key_transpose","nd_permute",325,310,{axis0:0,axis1:1,axis2:3,axis3:2}),
       N("scores","nd_batched_matmul",625,130),
       N("scale_scores","nd_scale",930,130,{factor:0.5}),
       N("causal","nd_causal_mask",1235,130,{queryStart:0,keyStart:0}),
       N("probabilities","nd_softmax_last",1540,130),
       N("context","nd_batched_matmul",1845,290)
      ];
      edges=[
       ["key","key_transpose","x"],
       ["query","scores","a"],["key_transpose","scores","b"],
       ["scores","scale_scores","x"],["scale_scores","causal","x"],
       ["causal","probabilities","x"],
       ["probabilities","context","a"],["value","context","b"]
      ];
    }else if(name==="nd_attention_lesson"){
      title="任务 25 · 四轴自由组合 · B/H/T/D、广播与多头矩阵乘";
      nodes=[
        N("queries","nd_input",40,80,{b:1,h:2,t:3,d:4,seed:1}),
        N("keys","nd_input",40,360,{b:1,h:2,t:4,d:3,seed:2}),
        N("batch_scores","nd_batched_matmul",355,200),
        N("bias","nd_input",40,680,{b:1,h:1,t:3,d:3,seed:3}),
        N("biased","nd_broadcast_add",660,260),
        N("summed","nd_reduce_sum",965,220,{axis:2}),
        N("flat","nd_flatten",1240,220),
        N("expand","nd_expand4d",1515,220),
        N("perm","nd_permute",1790,220,{axis0:0,axis1:2,axis2:1,axis3:3})
      ];
      // Q:[1,2,3,4], K:[1,2,4,3] => score:[1,2,3,3].
      // Bias [1,1,3,3] broadcasts over H. Sum over T -> [1,2,1,3],
      // Flatten [2,3] and expand to [1,1,2,3], then permute [1,2,1,3].
      edges=[
       ["queries","batch_scores","a"],["keys","batch_scores","b"],
       ["batch_scores","biased","a"],["bias","biased","b"],
       ["biased","summed","x"],["summed","flat","x"],
       ["flat","expand","x"],["expand","perm","x"]
      ];
    }else if(name==="trainable_attention"){
      title="任务 22 · 可训练 Q/K/V 参数 · Attention 梯度与 AdamW";
      const P=(id,x,y,m)=>N(id,"parameter",x,y,{values:m.map(row=>row.slice()),
        optimizer:{t:0,m:m.map(row=>row.map(()=>0)),v:m.map(row=>row.map(()=>0))}});
      nodes=[
        N("tokens","input",30,160,{rows:2,width:2,seed:2}),
        P("wq",35,420,[[.4,-.2],[.1,.6]]),
        P("wk",35,630,[[-.3,.5],[.7,.2]]),
        P("wv",35,830,[[.2,.7],[-.4,.3]]),
        N("q_proj","matmul",340,60),N("k_proj","matmul",340,290),
        N("v_proj","matmul",340,520),N("attend","attention",660,260),
        N("target","input",660,600,{rows:2,width:2,seed:6}),
        N("loss","mse",980,300)
      ];
      edges=[
        ["tokens","q_proj","a"],["wq","q_proj","b"],
        ["tokens","k_proj","a"],["wk","k_proj","b"],
        ["tokens","v_proj","a"],["wv","v_proj","b"],
        ["q_proj","attend","q"],["k_proj","attend","k"],["v_proj","attend","v"],
        ["attend","loss","prediction"],["target","loss","target"]
      ];
    }else if(name==="trainable_swiglu"){
      title="任务 23 · 可训练 SwiGLU FFN · W1/W3/W2 优化";
      const P=(id,x,y,m)=>N(id,"parameter",x,y,{values:m.map(row=>row.slice()),
        optimizer:{t:0,m:m.map(row=>row.map(()=>0)),v:m.map(row=>row.map(()=>0))}});
      nodes=[
        N("tokens","input",40,170,{rows:2,width:2,seed:4}),
        P("w1",40,390,[[.7,-.1],[.2,.5]]),
        P("w3",40,620,[[.3,.4],[-.5,.6]]),
        P("w2",360,690,[[.8,-.2],[.1,.4]]),
        N("gate","matmul",360,90),N("up","matmul",360,320),
        N("activated","swiglu",650,220),
        N("down","matmul",950,200),
        N("target","input",650,520,{rows:2,width:2,seed:7}),
        N("loss","mse",1230,290)
      ];
      edges=[
        ["tokens","gate","a"],["w1","gate","b"],
        ["tokens","up","a"],["w3","up","b"],
        ["gate","activated","gate"],["up","activated","up"],
        ["activated","down","a"],["w2","down","b"],
        ["down","loss","prediction"],["target","loss","target"]
      ];
    }else if(name==="trainable_moe"){
      const base=preset("moe_backward"),ids=["w1","w2","w3"];
      for(const n of base.nodes.filter(n=>ids.includes(n.id))){
        const numerical=calculate(n,{});
        const matrix=numerical.values.map(row=>row.slice());
        n.kind="parameter";n.params={values:matrix,
          optimizer:{t:0,m:matrix.map(row=>row.map(()=>0)),v:matrix.map(row=>row.map(()=>0))}};
      }
      return validateGraph({...base,title:"任务 24 · 可训练 MoE Expert W1/W3/W2 · 稀疏共享参数更新"});
    }else if(name==="trainable_core"){
      const source=preset("autodiff_core");
      const W=source.nodes.find(n=>n.id==="shared_weight");
      W.kind="parameter";W.params={
        values:[[0.35,-0.24],[0.08,0.57]],
        optimizer:{t:0,m:[[0,0],[0,0]],v:[[0,0],[0,0]]}
      };
      return validateGraph({...source,title:"任务 21 · 可训练参数共享 · Forward → Backward → AdamW → Forward"});
    }
    else if(name==="moe_backward"){
      const base=preset("moe_expert_ffn");
      return validateGraph({...base,title:"任务 20 · MoE 反向 · 稀疏专家参数共享与可微 Gate",
        nodes:[...base.nodes,N("teacher_target","input",2140,710,{rows:3,width:2,seed:6}),N("loss","mse",2410,360)],
        edges:[...base.edges,{from:"out",to:"loss",port:"prediction"},{from:"teacher_target",to:"loss",port:"target"}]});
    }
    else if(name==="autodiff_core"){
      title="任务 19 · 通用反向图 · 共享参数、RMSNorm、SwiGLU、Attention 与 MSE";
      nodes=[
        N("features","input",25,35,{rows:2,width:2,seed:1}),
        N("shared_weight","input",25,245,{rows:2,width:2,seed:3}),
        N("project","matmul",325,55),
        N("normalized","rmsnorm",600,65,{epsilon:0.00001}),
        N("gate","matmul",885,40),
        N("up","matmul",885,280),
        N("swiglu","swiglu",1180,145),
        N("attend","attention",1465,175),
        N("residual","add",1760,165),
        N("target","input",1465,475,{rows:2,width:2,seed:7}),
        N("loss","mse",2050,200)
      ];
      edges=[
        ["features","project","a"],["shared_weight","project","b"],
        ["project","normalized","x"],
        ["normalized","gate","a"],["shared_weight","gate","b"],
        ["normalized","up","a"],["shared_weight","up","b"],
        ["gate","swiglu","gate"],["up","swiglu","up"],
        ["swiglu","attend","q"],["swiglu","attend","k"],["swiglu","attend","v"],
        ["attend","residual","a"],["project","residual","b"],
        ["residual","loss","prediction"],["target","loss","target"]
      ];
    }else if(name==="grpo_objective"){
      title="任务 18 · GRPO 完整教学目标 · Token Logprob → Clip → Reference KL → Masked Loss";
      nodes=[
        N("new_logits","input",20,40,{rows:4,width:4,seed:2}),
        N("old_logits","input",20,240,{rows:4,width:4,seed:3}),
        N("ref_logits","input",20,440,{rows:4,width:4,seed:4}),
        N("actions","token_ids",20,640,{rows:4,vocab:4,seed:3}),
        N("new_logp","selected_logprob",280,40),
        N("old_logp","selected_logprob",280,240),
        N("ref_logp","selected_logprob",280,440),
        N("new_grid","response_token_reshape",540,40,{responses:2}),
        N("old_grid","response_token_reshape",540,240,{responses:2}),
        N("ref_grid","response_token_reshape",540,440,{responses:2}),
        N("rewards","input",20,845,{rows:1,width:2,seed:8}),
        N("group_adv","group_advantage",280,845,{epsilon:0.00001}),
        N("adv_per_response","transpose2d",540,845),
        N("ratio","policy_importance_ratio",820,160,{maxDelta:20}),
        N("adv_per_token","response_advantage_broadcast",820,680),
        N("surrogate","ppo_clipped_surrogate",1110,200,{epsilon:0.2}),
        N("reference_kl","sampled_reference_kl",1110,450,{maxDelta:20}),
        N("valid_mask","completion_length_mask",1110,700,{responses:2,tokens:2,firstLength:2,secondLength:1}),
        N("objective","masked_grpo_objective",1400,350,{beta:0.04}),
        N("loss","scale",1690,360,{factor:-1})
      ];
      edges=[
        ["new_logits","new_logp","logits"],["actions","new_logp","labels"],
        ["old_logits","old_logp","logits"],["actions","old_logp","labels"],
        ["ref_logits","ref_logp","logits"],["actions","ref_logp","labels"],
        ["new_logp","new_grid","x"],["old_logp","old_grid","x"],["ref_logp","ref_grid","x"],
        ["rewards","group_adv","rewards"],["group_adv","adv_per_response","x"],
        ["new_grid","ratio","current"],["old_grid","ratio","old"],
        ["adv_per_response","adv_per_token","advantage"],["ratio","adv_per_token","tokenGrid"],
        ["ratio","surrogate","ratio"],["adv_per_token","surrogate","advantage"],
        ["new_grid","reference_kl","current"],["ref_grid","reference_kl","reference"],
        ["surrogate","objective","surrogate"],["reference_kl","objective","kl"],["valid_mask","objective","mask"],
        ["objective","loss","x"]
      ];
    }else if(name==="mla_cache"){
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
