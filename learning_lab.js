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
  const StreamGraph=(typeof module==="object"&&module.exports)?require("./stream_graph_blocks.js"):globalThis.SeetheTruthStreamGraph;
  const Extensions=(typeof module==="object"&&module.exports)?require("./block_extension_registry.js"):globalThis.SeetheTruthBlockExtensions;
  const MAX_NODES=128,MAX_EDGES=256;
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
    ...BlockCatalog.defs,
    ...StreamGraph.defs,
    ...Extensions.defs
  });
  for(const kind of Extensions.kinds)if(Object.hasOwn(BlockCatalog.defs,kind)||Object.hasOwn(StreamGraph.defs,kind)||["input","linear","rmsnorm","silu","add","mul","matmul","softmax","attention","gqa","mse"].includes(kind))throw Error("extension overrides builtin kind "+kind);
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
      if(!finite(n.x)||!finite(n.y)||Math.abs(n.x)>50000||Math.abs(n.y)>50000)fail("积木坐标必须有限",n.id);
      const p=n.params||{}, type=n.kind;
      if(!p||typeof p!=="object"||Array.isArray(p))fail("无效参数",n.id);
      const params={...defs[type].params};
      for(const key of Object.keys(params)){
        if(Object.hasOwn(p,key))params[key]=p[key];
      }
      if(type==="parameter"||type==="nd_parameter"){
        if(p.frozen!==undefined&&typeof p.frozen!=="boolean")fail("Parameter frozen must be boolean",n.id);
        if(p.frozen===true)params.frozen=true;
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
        if(StreamGraph.isKind(type)){try{StreamGraph.validate(type,params)}catch(err){fail(err.message,n.id);}}
      if(Extensions.isKind(type)){try{Extensions.validate(type,params)}catch(err){fail(err.message,n.id);}}
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
    const normalized={schema:SCHEMA,title:typeof raw.title==="string"?raw.title.slice(0,80):"未命名实验",nodes,edges};
    if(raw.paperPatches!==undefined){
      if(!Array.isArray(raw.paperPatches)||raw.paperPatches.length>12)fail("invalid paper patch lineage");
      normalized.paperPatches=raw.paperPatches.filter(p=>p&&ids.has(p.outputId)&&ids.has(p.baseId)).map(p=>{
        if(p.paper!=="lora"||!/^[A-Za-z0-9_-]{1,36}$/.test(p.aId||"")||
           !/^[A-Za-z0-9_-]{1,36}$/.test(p.bId||"")||
           !ids.has(p.aId)||!ids.has(p.bId)||!Number.isSafeInteger(p.rank)||p.rank<1||p.rank>8||
           !finite(p.alpha)||p.alpha<=0||p.alpha>64)fail("invalid LoRA rewrite lineage");
        return {paper:"lora",outputId:p.outputId,baseId:p.baseId,
                aId:p.aId,bId:p.bId,rank:p.rank,alpha:p.alpha};
      });
    }
    return normalized;
  }
  // One dependency/port index for ordering, branch slicing and readiness.
  // Callers validateGraph first; this index does not impose model templates.
  function indexGraph(graph){
    const byId=new Map(graph.nodes.map(n=>[n.id,n]));
    const indegree=new Map(graph.nodes.map(n=>[n.id,0]));
    const children=new Map(graph.nodes.map(n=>[n.id,[]]));
    const parents=new Map(graph.nodes.map(n=>[n.id,[]]));
    const incomingByPort=new Map();
    for(const e of graph.edges){
      if(!byId.has(e.from)||!byId.has(e.to))fail("连线端点无效",e.to);
      const portKey=e.to+":"+e.port;
      if(incomingByPort.has(portKey))fail("同一输入端口不可连接两次",e.to);
      incomingByPort.set(portKey,e.from);
      indegree.set(e.to,indegree.get(e.to)+1);
      children.get(e.from).push(e.to);
      parents.get(e.to).push(e.from);
    }
    const queue=graph.nodes.filter(n=>indegree.get(n.id)===0).map(n=>n.id),ordered=[];
    for(let i=0;i<queue.length;i++){
      const id=queue[i];ordered.push(byId.get(id));
      for(const to of children.get(id)){
        indegree.set(to,indegree.get(to)-1);
        if(indegree.get(to)===0)queue.push(to);
      }
    }
    if(ordered.length!==graph.nodes.length)fail("检测到循环计算依赖：必须是有向无环图");
    return {ordered,parents,children,incomingByPort};
  }
  function order(graph){return indexGraph(graph).ordered;}
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
  function calculate(n,inputs,records){
    if(Extensions.isKind(n.kind))return Extensions.execute(n,inputs,records);
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
    const graph=validateGraph(raw), index=indexGraph(graph), sequence=index.ordered;
    const out=new Map(),steps=[];
    for(const n of sequence){
      const ins={},records={};
      for(const port of defs[n.kind].ports){
        const source=index.incomingByPort.get(n.id+":"+port);
        if(!source)fail(defs[n.kind].title+" 的 "+port+" 端口还没有连接",n.id);
        const upstream=out.get(source);
        if(!upstream?.values)fail("上游计算结果不存在",n.id);
        ins[port]=upstream.values;records[port]=upstream;
      }
      let inferred,value;
      try{
        inferred=Extensions.isKind(n.kind)?Extensions.infer(n,
          Object.fromEntries(Object.entries(records).map(([k,r])=>[k,r.shape])),
          Object.fromEntries(Object.entries(records).map(([k,r])=>[k,r.axes||null]))):StreamGraph.infer(n,
          Object.fromEntries(Object.entries(records).map(([k,r])=>[k,r.shape])),
          Object.fromEntries(Object.entries(records).map(([k,r])=>[k,r.logicalState||null])));
        value=StreamGraph.isKind(n.kind)?StreamGraph.run(n,records):calculate(n,ins,records);
      }catch(err){fail(err.message,n.id);}
      if(inferred&&JSON.stringify(value.shape)!==JSON.stringify(inferred.shape))fail("流状态静态/数值形状不一致",n.id);
      out.set(n.id,value);
      const {streamState,...publicValue}=value;
      const extensionSubsteps=Extensions.isKind(n.kind)?Extensions.substeps(n,ins,value):null;
      steps.push({id:n.id,kind:n.kind,...(extensionSubsteps?{extensionSubsteps}:{}),inputs:Object.fromEntries(Object.entries(ins).map(([k,v])=>[k,Extensions.tensorShape(v)])),
        ...publicValue,formula:defs[n.kind].formula,why:defs[n.kind].why,
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
    const recipe=Extensions.recipe(name);
    if(recipe)return validateGraph({schema:SCHEMA,...recipe});
    if(name==="nd_wan_i2v_full_block"){
      // Source-pinned Wan2.2 I2V A14B single Block:
      // self e0/e1/e2 -> norm3/text Cross -> norm2/e3/e4 ->
      // Linear1/GELU(tanh)/Linear2 -> e5 gate -> residual.
      // Compose existing typed operators; 2D GELU(tanh) is bridged with
      // nd_flatten + nd_expand4d, preserving true [1,1,8,H] coordinates.
      // Values remain seeded tiny Float64, never official checkpoint weights.
      const base=preset("nd_wan_i2v_self_cross");
      const ND=(typeof module==="object"&&module.exports)?require("./nd_tensor.js"):globalThis.SeetheTruthND;
      const N=(id,kind,x,y,params={})=>node(id,kind,x,y,params);
      const shape=[1,1,1,6],H=10;
      const parameter=(id,dimensions,seed,scale=0.16)=>{
        const values=ND.build(dimensions,idx=>ND.at(ND.seedTensor(dimensions,seed),idx)*scale);
        return N(id,"nd_parameter",200,3700+seed*32,{values,optimizer:{
          t:0,m:ND.build(dimensions,()=>0),v:ND.build(dimensions,()=>0)}});
      };
      const added=[
        N("norm2_ffn","nd_layernorm_last",9950,210,{epsilon:1e-6}),
        ...[3,4,5].map(i=>N("e"+i,"nd_input",50,3900+i*180,
          {b:1,h:1,t:8,d:6,seed:80+i})),
        ...[3,4,5].map(i=>parameter("mod"+i,shape,80+i,0.09)),
        ...[3,4,5].map(i=>N("e"+i+"_sum","nd_broadcast_add",10230,
          770+(i-3)*320)),
        N("ffn_norm_scale_product","nd_broadcast_mul",10620,175),
        N("ffn_norm_scaled","nd_broadcast_add",10960,175),
        N("ffn_conditioned","nd_broadcast_add",11300,175),
        parameter("ffn_w1",[1,1,6,H],90,0.21),
        N("ffn_b1","nd_input",200,4800,{b:1,h:1,t:1,d:H,seed:91}),
        N("ffn_up","nd_batched_matmul",11600,175),
        N("ffn_up_bias","nd_broadcast_add",11900,175),
        N("ffn_flat","nd_flatten",12210,175),
        N("ffn_gelu_tanh","gelu_tanh",12520,175),
        N("ffn_act4d","nd_expand4d",12830,175),
        parameter("ffn_w2",[1,1,H,6],92,0.19),
        N("ffn_b2","nd_input",200,5160,{b:1,h:1,t:1,d:6,seed:93}),
        N("ffn_down","nd_batched_matmul",13140,175),
        N("ffn_down_bias","nd_broadcast_add",13450,175),
        N("ffn_gated","nd_broadcast_mul",13760,175),
        N("after_ffn","nd_broadcast_add",14070,175)
      ];
      const edges=base.edges.map(e=>e.from==="after_cross"&&e.to==="loss"&&
        e.port==="prediction"?{...e,from:"after_ffn"}:e);
      const links=[
        ["after_cross","norm2_ffn","x"],
        ...[3,4,5].flatMap(i=>[["e"+i,"e"+i+"_sum","a"],
          ["mod"+i,"e"+i+"_sum","b"]]),
        ["norm2_ffn","ffn_norm_scale_product","a"],
        ["e4_sum","ffn_norm_scale_product","b"],
        ["norm2_ffn","ffn_norm_scaled","a"],
        ["ffn_norm_scale_product","ffn_norm_scaled","b"],
        ["ffn_norm_scaled","ffn_conditioned","a"],
        ["e3_sum","ffn_conditioned","b"],
        ["ffn_conditioned","ffn_up","a"],["ffn_w1","ffn_up","b"],
        ["ffn_up","ffn_up_bias","a"],["ffn_b1","ffn_up_bias","b"],
        ["ffn_up_bias","ffn_flat","x"],["ffn_flat","ffn_gelu_tanh","x"],
        ["ffn_gelu_tanh","ffn_act4d","x"],
        ["ffn_act4d","ffn_down","a"],["ffn_w2","ffn_down","b"],
        ["ffn_down","ffn_down_bias","a"],["ffn_b2","ffn_down_bias","b"],
        ["ffn_down_bias","ffn_gated","a"],["e5_sum","ffn_gated","b"],
        ["after_cross","after_ffn","a"],["ffn_gated","after_ffn","b"]
      ];
      return validateGraph({...base,
        title:"Wan2.2 I2V A14B · Self → Text Cross → e3/e4 FFN GELU(tanh) → e5 残差（完整单 Block）",
        nodes:[...base.nodes,...added],
        edges:[...edges,...links.map(([from,to,port])=>({from,to,port}))]});
    }

    if(name==="nd_wan_i2v_self_cross"){
      // Source: Wan-Video/Wan2.2@1ea34ff model.py 158-180, 203-259.
      // I2V A14B config sets cross_attn_norm=true. The norm3 affine
      // applies to x AFTER the e2 self-attention residual, not to context.
      const base=preset("nd_wan_adaln_self_gated");
      const ND=(typeof module==="object"&&module.exports)?require("./nd_tensor.js"):globalThis.SeetheTruthND;
      const N=(id,kind,x,y,params={})=>node(id,kind,x,y,params);
      const param=(id,shape,seed,scale=0.23,offset=0)=>{
        const baseValues=ND.seedTensor(shape,seed),
          values=ND.build(shape,idx=>offset+scale*ND.at(baseValues,idx)),
          zeros=ND.build(shape,()=>0);
        return N(id,"nd_parameter",200,2800+seed*55,
          {values,optimizer:{t:0,m:zeros,v:ND.build(shape,()=>0)}});
      };
      const weights=["q","k","v","o"].map((kind,i)=>
        param("cw_"+kind,[1,1,6,6],61+i));
      const bias=["q","k","v","o"].map((kind,i)=>
        N("cb_"+kind,"nd_input",30,3100+i*175,{b:1,h:1,t:1,d:6,seed:71+i}));
      const added=[
        N("text_context","nd_input",25,2700,{b:1,h:1,t:4,d:6,seed:60}),
        param("cross_norm3_gamma",[1,1,1,6],51,0.17,1),
        param("cross_norm3_beta",[1,1,1,6],52,0.08),
        N("cross_norm3_ln","nd_layernorm_last",5150,80,{epsilon:1e-6}),
        N("cross_norm3_mul","nd_broadcast_mul",5380,80),
        N("cross_norm3","nd_broadcast_add",5650,80),
        ...weights,...bias,
        N("cross_q_proj","nd_batched_matmul",5960,100),
        N("cross_k_proj","nd_batched_matmul",5960,460),
        N("cross_v_proj","nd_batched_matmul",5960,800),
        N("cross_q_bias","nd_broadcast_add",6280,100),
        N("cross_k_bias","nd_broadcast_add",6280,460),
        N("cross_v_bias","nd_broadcast_add",6280,800),
        param("cross_q_gamma",[1,1,1,6],53,0.12,1),
        param("cross_k_gamma",[1,1,1,6],54,0.12,1),
        N("cross_q_norm","nd_wan_qk_rmsnorm",6590,100,{epsilon:1e-6}),
        N("cross_k_norm","nd_wan_qk_rmsnorm",6590,460,{epsilon:1e-6}),
        N("cross_q_view","nd_reshape",6880,100,{b:1,h:8,t:1,d:6}),
        N("cross_k_view","nd_reshape",6880,460,{b:1,h:4,t:1,d:6}),
        N("cross_v_view","nd_reshape",6880,800,{b:1,h:4,t:1,d:6}),
        N("cross_q_heads","nd_permute",7200,100,{axis0:0,axis1:2,axis2:1,axis3:3}),
        N("cross_k_heads","nd_permute",7200,460,{axis0:0,axis1:2,axis2:1,axis3:3}),
        N("cross_v_heads","nd_permute",7200,800,{axis0:0,axis1:2,axis2:1,axis3:3}),
        N("cross_k_transpose","nd_permute",7510,460,{axis0:0,axis1:1,axis2:3,axis3:2}),
        N("cross_scores","nd_batched_matmul",7790,100),
        N("cross_scaled","nd_scale",8100,100,{factor:1/Math.sqrt(6)}),
        N("cross_probabilities","nd_softmax_last",8420,100),
        N("cross_context","nd_batched_matmul",8730,250),
        N("cross_out_proj","nd_batched_matmul",9050,220),
        N("cross_out_bias","nd_broadcast_add",9360,220),
        N("after_cross","nd_broadcast_add",9630,220)
      ];
      const edges=base.edges.map(e=>e.from==="after_self"&&e.to==="loss"&&
        e.port==="prediction"?{...e,from:"after_cross"}:{...e});
      const links=[
        ["after_self","cross_norm3_ln","x"],
        ["cross_norm3_ln","cross_norm3_mul","a"],
        ["cross_norm3_gamma","cross_norm3_mul","b"],
        ["cross_norm3_mul","cross_norm3","a"],["cross_norm3_beta","cross_norm3","b"],
        ["cross_norm3","cross_q_proj","a"],["cw_q","cross_q_proj","b"],
        ["text_context","cross_k_proj","a"],["cw_k","cross_k_proj","b"],
        ["text_context","cross_v_proj","a"],["cw_v","cross_v_proj","b"],
        ...["q","k","v"].flatMap(x=>[
          ["cross_"+x+"_proj","cross_"+x+"_bias","a"],["cb_"+x,"cross_"+x+"_bias","b"]]),
        ["cross_q_bias","cross_q_norm","x"],["cross_q_gamma","cross_q_norm","weight"],
        ["cross_k_bias","cross_k_norm","x"],["cross_k_gamma","cross_k_norm","weight"],
        ["cross_q_norm","cross_q_view","x"],["cross_k_norm","cross_k_view","x"],
        ["cross_v_bias","cross_v_view","x"],
        ...["q","k","v"].map(x=>["cross_"+x+"_view","cross_"+x+"_heads","x"]),
        ["cross_k_heads","cross_k_transpose","x"],
        ["cross_q_heads","cross_scores","a"],["cross_k_transpose","cross_scores","b"],
        ["cross_scores","cross_scaled","x"],["cross_scaled","cross_probabilities","x"],
        ["cross_probabilities","cross_context","a"],["cross_v_heads","cross_context","b"],
        ["cross_context","cross_out_proj","a"],["cw_o","cross_out_proj","b"],
        ["cross_out_proj","cross_out_bias","a"],["cb_o","cross_out_bias","b"],
        ["after_self","after_cross","a"],["cross_out_bias","after_cross","b"]
      ];
      return validateGraph({...base,
        title:"Wan2.2 I2V A14B pinned · self e0/e1/e2 → norm3 affine → text Cross-Attention residual",
        nodes:[...base.nodes,...added],
        edges:[...edges,...links.map(([from,to,port])=>({from,to,port}))]});
    }

    if(name==="nd_wan_adaln_self_gated"){
      // Extend the existing source-ordered trainable QKV lesson, not a
      // duplicate Attention implementation. e0/e1/e2 are explicit upstream
      // timestep/conditioning slices and M_i are shared trainable modulation.
      const base=preset("nd_wan_qkv_fullwidth_attention");
      const ND=(typeof module==="object"&&module.exports)?require("./nd_tensor.js"):globalThis.SeetheTruthND;
      const zero=sh=>ND.build(sh,()=>0);
      const param=(id,seed,scale)=>({
        id,kind:"nd_parameter",x:50,y:1930+Number(id.slice(-1))*140,
        params:{values:ND.build([1,1,1,6],idx=>
          ND.at(ND.seedTensor([1,1,1,6],seed),idx)*scale),
          optimizer:{t:0,m:zero([1,1,1,6]),v:zero([1,1,1,6])}}
      });
      const added=[
        {id:"norm1",kind:"nd_layernorm_last",x:270,y:-250,params:{epsilon:1e-6}},
        ...[0,1,2].map(i=>({id:"e"+i,kind:"nd_input",x:40,y:1900+i*150,
          params:{b:1,h:1,t:8,d:6,seed:31+i}})),
        param("mod0",40,.11),param("mod1",41,.08),param("mod2",42,.16),
        ...[0,1,2].map(i=>({id:"e"+i+"_sum",kind:"nd_broadcast_add",x:430,
          y:1930+i*160,params:{}})),
        {id:"norm_scale_product",kind:"nd_broadcast_mul",x:780,y:-250,params:{}},
        {id:"norm_scaled",kind:"nd_broadcast_add",x:1120,y:-260,params:{}},
        {id:"conditioned",kind:"nd_broadcast_add",x:1450,y:-260,params:{}},
        {id:"gated_attention",kind:"nd_broadcast_mul",x:4650,y:70,params:{}},
        {id:"after_self",kind:"nd_broadcast_add",x:4910,y:90,params:{}}
      ];
      const edges=base.edges.map(e=>({...e}));
      for(const edge of edges){
        if(["q_linear","k_linear","v_linear"].includes(edge.to)&&edge.from==="attention_input")
          edge.from="conditioned";
        if(edge.from==="out"&&edge.to==="loss"&&edge.port==="prediction")
          edge.from="after_self";
      }
      const more=[
        ["attention_input","norm1","x"],
        ...[0,1,2].flatMap(i=>[["e"+i,"e"+i+"_sum","a"],["mod"+i,"e"+i+"_sum","b"]]),
        ["norm1","norm_scale_product","a"],["e1_sum","norm_scale_product","b"],
        ["norm1","norm_scaled","a"],["norm_scale_product","norm_scaled","b"],
        ["norm_scaled","conditioned","a"],["e0_sum","conditioned","b"],
        ["out","gated_attention","a"],["e2_sum","gated_attention","b"],
        ["attention_input","after_self","a"],["gated_attention","after_self","b"]
      ];
      return validateGraph({...base,
        title:"Wan2.2 pinned · norm1 / e0-e1 AdaLN → QKV/3D Attention → e2 gated residual + Loss",
        nodes:[...base.nodes,...added],
        edges:[...edges,...more.map(([from,to,port])=>({from,to,port}))]});
    }

    if(name==="stream_state_branches"){
      const N=(id,kind,x,y,params={})=>node(id,kind,x,y,params);
      const nodes=[
        N("prefix","input",20,40,{rows:2,width:3,seed:1}),
        N("chunk_a","input",20,310,{rows:1,width:3,seed:2}),
        N("chunk_b","input",20,590,{rows:1,width:3,seed:3}),
        N("init","stream_state_init",300,40,{origin:10,capacity:5}),
        N("fork_a","stream_state_fork",530,50),
        N("fork_b","stream_state_fork",530,340),
        N("append_a","stream_state_append",770,65,{position:12}),
        N("append_b","stream_state_append",770,380,{position:12}),
        N("read_a","stream_state_read",1040,65,{start:11,length:2}),
        N("read_b","stream_state_read",1040,380,{start:11,length:2})
      ];
      const edges=[
        ["prefix","init","x"],["init","fork_a","state"],["init","fork_b","state"],
        ["fork_a","append_a","state"],["chunk_a","append_a","chunk"],
        ["fork_b","append_b","state"],["chunk_b","append_b","chunk"],
        ["append_a","read_a","state"],["append_b","read_b","state"]
      ].map(([from,to,port])=>({from,to,port}));
      return validateGraph({schema:SCHEMA,title:"状态积木示例 · 共享前缀的不可变双分叉",nodes,edges});
    }
    if(["deepinfer_mtp_budget","deepinfer_yoco_replay"].includes(name)){
      const N=(id,kind,x,y,params={})=>node(id,kind,x,y,params);
      let title,nodes,edges;
      if(name==="deepinfer_mtp_budget"){
        title="任务 58 · DeepInfer DSpark: Remap → 累积置信度 → 全局固定预算";
        nodes=[N("remap","parameter",35,115,{values:[[0],[1],[-1]],frozen:true,optimizer:{t:0,m:[[0],[0],[0]],v:[[0],[0],[0]]}}),
          N("prev_confidence","parameter",35,375,{values:[[0.9,0.8],[0.5,0.5]],frozen:true,optimizer:{t:0,m:[[0,0],[0,0]],v:[[0,0],[0,0]]}}),
          N("mtp","deepinfer_mtp_confidence_schedule",395,245,{totalQ:5})];
        edges=[["remap","mtp","remap"],["prev_confidence","mtp","prev_confidence"]];
      }else{
        title="任务 59 · DeepInfer YoCo: Mask vs Compact SWA 保留后缀";
        nodes=[N("q","input",35,110,{rows:4,width:3,seed:1}),
          N("k","input",35,340,{rows:4,width:3,seed:2}),
          N("v","input",35,570,{rows:4,width:2,seed:3}),
          N("mask","deepinfer_yoco_swa_mask",430,130,{window:2,keepTokens:3,backtracking:1}),
          N("compact","deepinfer_yoco_swa_compact",430,490,{window:2,keepTokens:3,backtracking:1})];
        edges=[["q","mask","q"],["k","mask","k"],["v","mask","v"],
          ["q","compact","q"],["k","compact","k"],["v","compact","v"]];
      }
      return validateGraph({schema:SCHEMA,title,nodes,
        edges:edges.map(([from,to,port])=>({from,to,port}))});
    }
    if(["inference_sparse_decode","inference_paged_kv","inference_speculative","inference_paged_sparse"].includes(name)){
      const N=(id,kind,x,y,params={})=>node(id,kind,x,y,params);
      let title,nodes=[],edges=[];
      if(name==="inference_sparse_decode"){
        title="任务 54 · DeepSelect Top-K → DSA Selected Attention";
        nodes=[N("scores","input",20,90,{rows:2,width:4,seed:1}),N("select","inference_topk",270,90,{topk:2}),
          N("q","input",20,440,{rows:2,width:4,seed:2}),N("k","input",20,650,{rows:4,width:4,seed:3}),
          N("v","input",20,870,{rows:4,width:3,seed:4}),
          N("sparse","inference_sparse_attention",555,155,{causal:false,queryOffset:0}),
          N("dense","attention",580,580)];
        edges=[["scores","select","scores"],["select","sparse","indices"],["q","sparse","q"],
          ["k","sparse","k"],["v","sparse","v"],["q","dense","q"],["k","dense","k"],["v","dense","v"]];
      }else if(name==="inference_paged_kv"){
        title="任务 55 · FlashMLA 相关概念：Paged KV Cache 地址映射";
        nodes=[N("cache","input",20,190,{rows:4,width:3,seed:1}),
          N("page_scores","input",20,480,{rows:2,width:2,seed:2}),
          N("page_ids","inference_topk",290,475,{topk:1}),
          N("position_scores","input",20,760,{rows:2,width:4,seed:3}),
          N("positions","inference_topk",290,760,{topk:1}),
          N("read","inference_paged_kv_read",600,200,{pageSize:2,logicalLength:4})];
        edges=[["page_scores","page_ids","scores"],["position_scores","positions","scores"],
          ["cache","read","cache"],["page_ids","read","blocks"],["positions","read","positions"]];
      }else if(name==="inference_speculative"){
        title="任务 56 · DeepSpec 相关：Draft/Target 贪心前缀验收";
        nodes=[N("draft_logits","input",20,150,{rows:3,width:4,seed:1}),
          N("target_logits","input",20,550,{rows:4,width:4,seed:2}),
          N("draft","inference_topk",300,150,{topk:1}),
          N("target","inference_topk",300,550,{topk:1}),
          N("accepted","inference_greedy_verify",640,240)];
        edges=[["draft_logits","draft","scores"],["target_logits","target","scores"],
          ["draft","accepted","draft"],["target","accepted","target"]];
      }else{
        title="任务 57 · Paged KV → Top-K Indexer → Sparse Attention Decode";
        nodes=[N("cache","input",20,180,{rows:4,width:4,seed:1}),
          N("page_scores","input",20,470,{rows:2,width:2,seed:2}),
          N("pages","inference_topk",275,470,{topk:1}),
          N("position_scores","input",20,760,{rows:2,width:4,seed:3}),
          N("positions","inference_topk",275,765,{topk:1}),
          N("read","inference_paged_kv_read",560,300,{pageSize:2,logicalLength:4}),
          N("q","input",560,35,{rows:2,width:4,seed:4}),
          N("indexer_scores","input",560,710,{rows:2,width:2,seed:5}),
          N("select","inference_topk",840,710,{topk:2}),
          N("sparse","inference_sparse_attention",1080,320,{causal:false,queryOffset:0})];
        edges=[["page_scores","pages","scores"],["position_scores","positions","scores"],
          ["cache","read","cache"],["pages","read","blocks"],["positions","read","positions"],
          ["q","sparse","q"],["read","sparse","k"],["read","sparse","v"],
          ["indexer_scores","select","scores"],["select","sparse","indices"]];
      }
      return validateGraph({schema:SCHEMA,title,nodes,edges:edges.map(([from,to,port])=>({from,to,port}))});
    }
    if(name==="video_causal_conv3d"){
      const N=(id,kind,x,y,params={})=>node(id,kind,x,y,params);
      const weights=[[0.25,-0.1,0.3,0.2],[-0.15,0.4,0.2,-0.25]];
      const graph={schema:SCHEMA,title:"任务 53 · 真五维 Conv3D：因果 VAE 卷积与历史帧缓存",nodes:[
        N("frames","input",30,180,{rows:8,width:2,seed:2}),
        N("kernel3d","parameter",30,440,{values:weights,optimizer:{t:0,
          m:weights.map(row=>row.map(()=>0)),v:weights.map(row=>row.map(()=>0))}}),
        N("causal_conv","video_causal_conv3d",330,180),
        N("activate","silu",620,180),
        N("temporal_attend","attention",880,180)
      ],edges:[
        {from:"frames",to:"causal_conv",port:"x"},
        {from:"kernel3d",to:"causal_conv",port:"kernel"},
        {from:"causal_conv",to:"activate",port:"x"},
        {from:"activate",to:"temporal_attend",port:"q"},
        {from:"activate",to:"temporal_attend",port:"k"},
        {from:"activate",to:"temporal_attend",port:"v"}]};
      return validateGraph(graph);
    }
    if(["vision_learned_conv","audio_mel_chain","diffusion_ddim_chain"].includes(name)){
      const N=(id,kind,x,y,params={})=>node(id,kind,x,y,params);
      let title,nodes=[],edges=[];
      if(name==="vision_learned_conv"){
        title="任务 50 · 真实卷积核 Parameter → Conv2D → Attention";
        nodes=[N("pixels","input",30,100,{rows:4,width:4,seed:1}),
          N("kernel","parameter",30,360,{values:[[0.2,-0.1,0.3,0.4],[-0.1,0.25,0.05,0.3]],
            optimizer:{t:0,m:[[0,0,0,0],[0,0,0,0]],v:[[0,0,0,0],[0,0,0,0]]}}),
          N("conv","vision_conv2d",330,170),
          N("features","silu",590,170),
          N("spatial_attn","attention",840,170)];
        edges=[["pixels","conv","x"],["kernel","conv","kernel"],["conv","features","x"],
          ["features","spatial_attn","q"],["features","spatial_attn","k"],["features","spatial_attn","v"]];
      }else if(name==="audio_mel_chain"){
        title="任务 51 · 真实 Hann STFT → HTK Mel → Log10 → Attention";
        nodes=[N("waveform","input",20,130,{rows:1,width:8,seed:2}),
          N("power","audio_stft_power",280,130),
          N("mel","audio_mel_filterbank",535,130),
          N("logmel","audio_log_power",780,130),
          N("projection","linear",1020,130,{out:4}),
          N("audio_attn","attention",1260,130)];
        edges=[["waveform","power","x"],["power","mel","x"],["mel","logmel","x"],
          ["logmel","projection","x"],["projection","audio_attn","q"],
          ["projection","audio_attn","k"],["projection","audio_attn","v"]];
      }else{
        title="任务 52 · 扩散 epsilon-DDIM 两步确定性去噪";
        nodes=[N("latent","input",20,125,{rows:2,width:4,seed:1}),
          N("pred_noise","input",20,390,{rows:2,width:4,seed:4}),
          N("step_t","diffusion_ddim_step",340,150,{alphaT:0.36,alphaPrevious:0.64}),
          N("step_previous","diffusion_ddim_step",620,150,{alphaT:0.64,alphaPrevious:1})];
        edges=[["latent","step_t","x"],["pred_noise","step_t","epsilon"],
          ["step_t","step_previous","x"],["pred_noise","step_previous","epsilon"]];
      }
      return validateGraph({schema:SCHEMA,title,nodes,edges:edges.map(([from,to,port])=>({from,to,port}))});
    }
    // Multimodal tokens stay explicit: image [H,W*C], video [T*H,W*C],
    // audio [1,N]. The numerical source is synthetic, not decoded media.
    if(["image_vit_patch","audio_spectrogram","video_vivit_tubelet","multimodal_cross_attention"].includes(name)){
      const n=(id,kind,x,y,params={})=>node(id,kind,x,y,params);
      let title,nodes=[],edges=[];
      if(name==="image_vit_patch"){
        title="任务 46 · 图像 ViT Patchify → Linear → Self-Attention";
        nodes=[n("pixels","input",20,120,{rows:4,width:4,seed:1}),
          n("patches","vision_patchify",280,120),
          n("image_embedding","linear",545,120,{out:4}),
          n("image_attend","attention",820,120)];
        edges=[["pixels","patches","x"],["patches","image_embedding","x"],
          ["image_embedding","image_attend","q"],["image_embedding","image_attend","k"],["image_embedding","image_attend","v"]];
      }else if(name==="audio_spectrogram"){
        title="任务 47 · 音频 Waveform → Hann STFT → Log Power → Attention";
        nodes=[n("wave","input",20,135,{rows:1,width:8,seed:2}),
          n("spectrum","audio_stft_power",280,135),
          n("log_power","audio_log_power",535,135),
          n("audio_embedding","linear",790,135,{out:4}),
          n("audio_attend","attention",1050,135)];
        edges=[["wave","spectrum","x"],["spectrum","log_power","x"],
          ["log_power","audio_embedding","x"],["audio_embedding","audio_attend","q"],
          ["audio_embedding","audio_attend","k"],["audio_embedding","audio_attend","v"]];
      }else if(name==="video_vivit_tubelet"){
        title="任务 48 · 视频 THWC Tubelet → Linear → Self-Attention";
        nodes=[n("frames","input",20,120,{rows:8,width:2,seed:3}),
          n("tubelets","video_tubelet",280,120),
          n("video_embedding","linear",545,120,{out:4}),
          n("video_attend","attention",820,120)];
        edges=[["frames","tubelets","x"],["tubelets","video_embedding","x"],
          ["video_embedding","video_attend","q"],["video_embedding","video_attend","k"],["video_embedding","video_attend","v"]];
      }else{
        title="任务 49 · 图像 / 音频 / 视频三路 Token 互注意力融合";
        nodes=[n("pixels","input",20,30,{rows:4,width:4,seed:1}),
          n("patches","vision_patchify",280,30),
          n("image_embedding","linear",540,30,{out:4}),
          n("wave","input",20,350,{rows:1,width:8,seed:2}),
          n("spectrum","audio_stft_power",280,350),
          n("log_power","audio_log_power",540,350),
          n("audio_embedding","linear",800,350,{out:4}),
          n("frames","input",20,690,{rows:8,width:2,seed:3}),
          n("tubelets","video_tubelet",280,690),
          n("video_embedding","linear",540,690,{out:4}),
          n("image_audio_cross","attention",1080,145),
          n("image_audio_video_cross","attention",1370,145)];
        edges=[["pixels","patches","x"],["patches","image_embedding","x"],
          ["wave","spectrum","x"],["spectrum","log_power","x"],["log_power","audio_embedding","x"],
          ["frames","tubelets","x"],["tubelets","video_embedding","x"],
          ["image_embedding","image_audio_cross","q"],["audio_embedding","image_audio_cross","k"],
          ["audio_embedding","image_audio_cross","v"],
          ["image_audio_cross","image_audio_video_cross","q"],["video_embedding","image_audio_video_cross","k"],
          ["video_embedding","image_audio_video_cross","v"]];
      }
      return validateGraph({schema:SCHEMA,title,nodes,edges:edges.map(([from,to,port])=>({from,to,port}))});
    }
    if(name==="loop_input_injection"){
      const Injection=typeof module==="object"&&module.exports?require("./loop_input_injection.js"):globalThis.SeetheTruthLoopInjection;
      if(!Injection)throw Error("loop input reinjection builder unavailable");
      return Injection.build();
    }
    if(name==="mor_token_depth_budget"){
      const MoR=typeof module==="object"&&module.exports?require("./adaptive_token_depth.js"):globalThis.SeetheTruthAdaptiveDepth;
      if(!MoR)throw Error("adaptive depth source builder unavailable");
      return MoR.build();
    }
    if(name==="loop_residual_scaling"){
      const Loop=typeof module==="object"&&module.exports?require("./loop_transformer.js"):globalThis.SeetheTruthLoopTransformer;
      if(!Loop)throw Error("loop residual scaling engine not loaded");
      return Loop.buildResidualScaling({loops:2,rule:"inverse_loops"});
    }
    if(name==="loopformer_consistency"){
      const Consistency=typeof module==="object"&&module.exports?require("./loop_consistency.js"):globalThis.SeetheTruthLoopConsistency;
      if(!Consistency)throw Error("LoopFormer-inspired source graph builder unavailable");
      return Consistency.build();
    }
    if(/^loop_transformer_[123]$/.test(name)){
      const Composer=typeof module==="object"&&module.exports?require("./loop_transformer.js"):globalThis.SeetheTruthLoopTransformer;
      if(!Composer)throw Error("Looped Transformer source graph builder is not loaded");
      return Composer.build({loops:Number(name.slice(-1))});
    }
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
    }else if(name==="nd_wan_qkv_fullwidth_attention"){
      title="Wan2.2 pinned · QKV Linear+bias → Q/K learned full-width RMSNorm → Head view → F/H/W RoPE → Attention → output Linear";
      const ND=(typeof module==="object"&&module.exports)?require("./nd_tensor.js"):globalThis.SeetheTruthND;
      const param=(sh,seed,scale=0.25,offset=0)=>{
        const v=ND.seedTensor(sh,seed),values=ND.build(sh,i=>offset+scale*ND.at(v,i)),
          zero=ND.build(sh,()=>0);
        return {values,optimizer:{t:0,m:zero,v:ND.build(sh,()=>0)}};
      };
      nodes=[
        N("attention_input","nd_input",20,25,{b:1,h:1,t:8,d:6,seed:4}),
        N("w_q","nd_parameter",40,350,param([1,1,6,6],5)),
        N("w_k","nd_parameter",40,650,param([1,1,6,6],7)),
        N("w_v","nd_parameter",40,950,param([1,1,6,6],9)),
        N("b_q","nd_input",40,1200,{b:1,h:1,t:1,d:6,seed:10}),
        N("b_k","nd_input",40,1430,{b:1,h:1,t:1,d:6,seed:11}),
        N("b_v","nd_input",40,1660,{b:1,h:1,t:1,d:6,seed:12}),
        N("gamma_q","nd_parameter",350,1200,param([1,1,1,6],13,0.18,1)),
        N("gamma_k","nd_parameter",350,1490,param([1,1,1,6],14,0.18,1)),
        N("q_linear","nd_batched_matmul",365,60),
        N("k_linear","nd_batched_matmul",365,350),
        N("v_linear","nd_batched_matmul",365,650),
        N("q_bias","nd_broadcast_add",695,60),
        N("k_bias","nd_broadcast_add",695,350),
        N("v_bias","nd_broadcast_add",695,650),
        N("q_norm","nd_wan_qk_rmsnorm",1015,60,{epsilon:0.000001}),
        N("k_norm","nd_wan_qk_rmsnorm",1015,350,{epsilon:0.000001}),
        N("q_view","nd_reshape",1350,60,{b:1,h:8,t:1,d:6}),
        N("k_view","nd_reshape",1350,350,{b:1,h:8,t:1,d:6}),
        N("v_view","nd_reshape",1350,650,{b:1,h:8,t:1,d:6}),
        N("q_heads","nd_permute",1670,60,{axis0:0,axis1:2,axis2:1,axis3:3}),
        N("k_heads","nd_permute",1670,350,{axis0:0,axis1:2,axis2:1,axis3:3}),
        N("v_heads","nd_permute",1670,650,{axis0:0,axis1:2,axis2:1,axis3:3}),
        N("q_rope","nd_wan_rope3d",1990,60,{frames:2,height:2,width:2,base:10000}),
        N("k_rope","nd_wan_rope3d",1990,350,{frames:2,height:2,width:2,base:10000}),
        N("kt","nd_permute",2300,350,{axis0:0,axis1:1,axis2:3,axis3:2}),
        N("scores","nd_batched_matmul",2610,160),
        N("scaled","nd_scale",2930,160,{factor:1/Math.sqrt(6)}),
        N("prob","nd_softmax_last",3250,160),
        N("context","nd_batched_matmul",3570,290),
        N("w_o","nd_parameter",3570,650,param([1,1,6,6],15)),
        N("b_o","nd_input",3930,650,{b:1,h:1,t:1,d:6,seed:16}),
        N("o_linear","nd_batched_matmul",3910,270),
        N("out","nd_broadcast_add",4240,270),
        N("target","nd_input",4240,640,{b:1,h:1,t:8,d:6,seed:17}),
        N("loss","nd_mse",4600,350)
      ];
      edges=[
        ["attention_input","q_linear","a"],["w_q","q_linear","b"],
        ["attention_input","k_linear","a"],["w_k","k_linear","b"],
        ["attention_input","v_linear","a"],["w_v","v_linear","b"],
        ["q_linear","q_bias","a"],["b_q","q_bias","b"],
        ["k_linear","k_bias","a"],["b_k","k_bias","b"],
        ["v_linear","v_bias","a"],["b_v","v_bias","b"],
        ["q_bias","q_norm","x"],["gamma_q","q_norm","weight"],
        ["k_bias","k_norm","x"],["gamma_k","k_norm","weight"],
        ["q_norm","q_view","x"],["k_norm","k_view","x"],["v_bias","v_view","x"],
        ["q_view","q_heads","x"],["k_view","k_heads","x"],["v_view","v_heads","x"],
        ["q_heads","q_rope","x"],["k_heads","k_rope","x"],
        ["k_rope","kt","x"],["q_rope","scores","a"],["kt","scores","b"],
        ["scores","scaled","x"],["scaled","prob","x"],
        ["prob","context","a"],["v_heads","context","b"],
        ["context","o_linear","a"],["w_o","o_linear","b"],
        ["o_linear","out","a"],["b_o","out","b"],
        ["out","loss","prediction"],["target","loss","target"]
      ];
    }else if(name==="nd_wan_video_rope_attention"){
      title="Wan2.2 固定源码 · 视频 F/H/W RoPE → 双头 Self-Attention 核心";
      nodes=[
        N("query","nd_input",30,35,{b:1,h:2,t:8,d:6,seed:3}),
        N("key","nd_input",30,330,{b:1,h:2,t:8,d:6,seed:5}),
        N("value","nd_input",30,640,{b:1,h:2,t:8,d:6,seed:7}),
        N("q_video_rope","nd_wan_rope3d",390,40,{frames:2,height:2,width:2,base:10000}),
        N("k_video_rope","nd_wan_rope3d",390,335,{frames:2,height:2,width:2,base:10000}),
        N("key_transpose","nd_permute",730,335,{axis0:0,axis1:1,axis2:3,axis3:2}),
        N("qk_scores","nd_batched_matmul",1070,130),
        N("scaled_scores","nd_scale",1410,130,{factor:1/Math.sqrt(6)}),
        N("probabilities","nd_softmax_last",1750,130),
        N("context","nd_batched_matmul",2070,320)
      ];
      edges=[
        ["query","q_video_rope","x"],["key","k_video_rope","x"],
        ["k_video_rope","key_transpose","x"],
        ["q_video_rope","qk_scores","a"],["key_transpose","qk_scores","b"],
        ["qk_scores","scaled_scores","x"],["scaled_scores","probabilities","x"],
        ["probabilities","context","a"],["value","context","b"]
      ];
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
    }else if(name==="loop_transformer_two"||name==="loop_transformer_three"){
      // Fixed-depth graph unrolling exposes exact repeated calls to one
      // causal Transformer block, while preserving a SINGLE Parameter identity
      // for each projection and FFN tensor across all recurrent steps.
      const nLoops=name==="loop_transformer_two"?2:3;
      title="任务 "+(nLoops===2?"39":"40")+" · Looped Transformer "+nLoops+" 次循环 · 共享 Attention/SwiGLU 权重";
      const mat=(rows,cols,seed)=>Array.from({length:rows},(_,i)=>Array.from({length:cols},(_,j)=>
        Number((Math.sin((i+1)*.47+(j+1)*.31+seed)*.24).toFixed(8))));
      const P=(id,x,y,v)=>N(id,"parameter",x,y,{values:v,
        optimizer:{t:0,m:v.map(row=>row.map(()=>0)),v:v.map(row=>row.map(()=>0))}});
      nodes=[
        N("ids","token_ids",20,115,{rows:4,vocab:8,seed:1}),
        N("prefix","slice_rows",265,65,{start:0,length:3}),
        N("labels","slice_rows",265,930,{start:1,length:3}),
        P("embed",20,370,mat(8,4,1)),
        N("lookup","gather_rows",485,120),
        N("pos","pos_encoding",685,120),
        N("mask","causal_mask",905,950),
        P("wq",75,1350,mat(4,4,2)),
        P("wk",75,1550,mat(4,4,3)),
        P("wv",75,1750,mat(4,4,4)),
        P("wo",75,1950,mat(4,4,5)),
        P("wgate",75,2150,mat(4,4,6)),
        P("wup",75,2350,mat(4,4,7)),
        P("wdown",75,2550,mat(4,4,8))
      ];
      edges=[
        ["ids","prefix","x"],["ids","labels","x"],
        ["prefix","lookup","indices"],["embed","lookup","x"],
        ["lookup","pos","x"],["pos","mask","x"]
      ];
      let state="pos";
      for(let loop=1;loop<=nLoops;loop++){
        const p="r"+loop+"_";
        const x=1010+(loop-1)*2700,y=100;
        const nd=(suffix,kind,dx,dy,params={})=>nodes.push(N(p+suffix,kind,x+dx,y+dy,params));
        const link=(source,port,to)=>edges.push([source,p+to,port]);
        nd("norm","rmsnorm",0,0,{epsilon:1e-5});
        nd("q","matmul",225,-15);nd("k","matmul",225,250);nd("v","matmul",225,510);
        nd("qr","rope2d",445,-15,{offset:0});
        nd("kr","rope2d",445,250,{offset:0});
        nd("kt","transpose2d",660,250);
        nd("score","matmul",865,110);
        nd("scale","scale",1065,105,{factor:0.5});
        nd("prob","masked_softmax",1260,105);
        nd("ctx","weighted_sum",1455,105);
        nd("attn","matmul",1640,105);
        nd("attn_res","add",1835,105);
        nd("ffnorm","rmsnorm",2010,105,{epsilon:1e-5});
        nd("gate","matmul",2200,-15);
        nd("up","matmul",2200,235);
        nd("swi","swiglu",2390,105);
        nd("down","matmul",2580,105);
        nd("state","add",2770,105);
        link(state,"x","norm");
        link(p+"norm","a","q");edges.push(["wq",p+"q","b"]);
        link(p+"norm","a","k");edges.push(["wk",p+"k","b"]);
        link(p+"norm","a","v");edges.push(["wv",p+"v","b"]);
        link(p+"q","x","qr");link(p+"k","x","kr");
        link(p+"kr","x","kt");
        link(p+"qr","a","score");link(p+"kt","b","score");
        link(p+"score","x","scale");
        link(p+"scale","logits","prob");edges.push(["mask",p+"prob","mask"]);
        link(p+"prob","weights","ctx");link(p+"v","values","ctx");
        link(p+"ctx","a","attn");edges.push(["wo",p+"attn","b"]);
        edges.push([state,p+"attn_res","a"]);link(p+"attn","b","attn_res");
        link(p+"attn_res","x","ffnorm");
        link(p+"ffnorm","a","gate");edges.push(["wgate",p+"gate","b"]);
        link(p+"ffnorm","a","up");edges.push(["wup",p+"up","b"]);
        link(p+"gate","gate","swi");link(p+"up","up","swi");
        link(p+"swi","a","down");edges.push(["wdown",p+"down","b"]);
        link(p+"attn_res","a","state");link(p+"down","b","state");
        state=p+"state";
      }
      const x=1010+nLoops*2700;
      nodes.push(N("final_norm","rmsnorm",x,125,{epsilon:1e-5}),
        N("tied_head","transpose2d",x,450),
        N("logits","matmul",x+245,125),
        N("loss","cross_entropy",x+490,125));
      edges.push([state,"final_norm","x"],["embed","tied_head","x"],
        ["final_norm","logits","a"],["tied_head","logits","b"],
        ["logits","loss","logits"],["labels","loss","labels"]);
    }else if(name==="industrial_gqa_rope_tied"){
      title="任务 38 · Qwen3 启发：QK Norm + RoPE + GQA + 权重绑定";
      // Heads=2, KV heads=1, head_dim=2, model_dim=4, V=8.
      // Qwen3 uses learnable Q/K Norm gains; these are shared across heads
      // as one Parameter per Q/K side. 2D slicing exposes every head operation.
      const mat=(rows,cols,seed)=>Array.from({length:rows},(_,i)=>Array.from({length:cols},(_,j)=>
        Number((Math.sin((i+1)*.47+(j+1)*.31+seed)*.24).toFixed(8))));
      const P=(id,x,y,v)=>N(id,"parameter",x,y,{values:v,
        optimizer:{t:0,m:v.map(row=>row.map(()=>0)),v:v.map(row=>row.map(()=>0))}});
      nodes=[
        N("ids","token_ids",20,160,{rows:4,vocab:8,seed:1}),
        N("prefix","slice_rows",260,60,{start:0,length:3}),
        N("labels","slice_rows",260,1120,{start:1,length:3}),
        P("embed",20,400,mat(8,4,1)),
        N("lookup","gather_rows",480,100),
        N("pre_norm","rmsnorm",695,100,{epsilon:1e-5}),
        P("wq",680,350,mat(4,4,2)),P("wk",680,590,mat(4,2,3)),
        P("wv",680,820,mat(4,2,4)),
        P("qgain",930,950,[[1,1]]),P("kgain",930,1140,[[1,1]]),
        N("qgainrows","repeat_rows",1190,950,{times:3}),
        N("kgainrows","repeat_rows",1190,1140,{times:3}),
        N("q","matmul",945,60),N("k","matmul",945,260),N("v","matmul",945,495),
        N("q0","slice_cols",1160,30,{start:0,width:2}),
        N("q1","slice_cols",1160,210,{start:2,width:2}),
        N("q0n","qk_norm",1360,30,{epsilon:1e-5}),
        N("q1n","qk_norm",1360,210,{epsilon:1e-5}),
        N("kn","qk_norm",1160,510,{epsilon:1e-5}),
        N("q0aff","mul",1430,30),N("q1aff","mul",1430,210),
        N("kaff","mul",1430,510),
        N("q0r","rope2d",1580,30,{offset:0}),
        N("q1r","rope2d",1580,210,{offset:0}),
        N("kr","rope2d",1360,510,{offset:0}),
        N("kt","transpose2d",1580,510),
        N("score0","matmul",1800,20),N("score1","matmul",1800,210),
        N("scale0","scale",2000,20,{factor:1/Math.sqrt(2)}),
        N("scale1","scale",2000,210,{factor:1/Math.sqrt(2)}),
        N("mask","causal_mask",2000,510),
        N("prob0","masked_softmax",2230,20),
        N("prob1","masked_softmax",2230,215),
        N("out0","weighted_sum",2460,20),
        N("out1","weighted_sum",2460,215),
        N("heads","concat_cols",2700,125),
        P("wo",2490,515,mat(4,4,5)),
        N("out_proj","matmul",2920,125),
        N("residual","add",3130,125),
        N("post_norm","rmsnorm",3350,125,{epsilon:1e-5}),
        N("tied_head","transpose2d",3130,520),
        N("logits","matmul",3570,125),
        N("loss","cross_entropy",3790,125)
      ];
      edges=[
        ["ids","prefix","x"],["ids","labels","x"],
        ["prefix","lookup","indices"],["embed","lookup","x"],["lookup","pre_norm","x"],
        ["pre_norm","q","a"],["wq","q","b"],
        ["pre_norm","k","a"],["wk","k","b"],
        ["pre_norm","v","a"],["wv","v","b"],
        ["q","q0","x"],["q","q1","x"],
        ["q0","q0n","x"],["q1","q1n","x"],["k","kn","x"],
        ["qgain","qgainrows","x"],["kgain","kgainrows","x"],
        ["q0n","q0aff","a"],["q1n","q1aff","a"],
        ["qgainrows","q0aff","b"],["qgainrows","q1aff","b"],
        ["kn","kaff","a"],["kgainrows","kaff","b"],
        ["q0aff","q0r","x"],["q1aff","q1r","x"],["kaff","kr","x"],
        ["kr","kt","x"],
        ["q0r","score0","a"],["kt","score0","b"],
        ["q1r","score1","a"],["kt","score1","b"],
        ["score0","scale0","x"],["score1","scale1","x"],
        ["scale0","mask","x"],
        ["scale0","prob0","logits"],["scale1","prob1","logits"],
        ["mask","prob0","mask"],["mask","prob1","mask"],
        ["prob0","out0","weights"],["v","out0","values"],
        ["prob1","out1","weights"],["v","out1","values"],
        ["out0","heads","a"],["out1","heads","b"],
        ["heads","out_proj","a"],["wo","out_proj","b"],
        ["lookup","residual","a"],["out_proj","residual","b"],
        ["residual","post_norm","x"],
        ["embed","tied_head","x"],
        ["post_norm","logits","a"],["tied_head","logits","b"],
        ["logits","loss","logits"],["labels","loss","labels"]
      ];
    }else if(name==="causal_transformer_lm"){
      title="任务 37 · 可训练因果 Transformer · QKᵀ→Mask→Softmax→PV→CE";
      // Pure composition of individual, reusable mathematics blocks.
      // Every projection is an independent trainable Parameter source.
      const mat=(rows,cols,seed)=>Array.from({length:rows},(_,i)=>Array.from({length:cols},(_,j)=>
        Number((Math.sin((i+1)*.47+(j+1)*.31+seed)*.24).toFixed(8))));
      const P=(id,x,y,v)=>N(id,"parameter",x,y,{values:v,
        optimizer:{t:0,m:v.map(row=>row.map(()=>0)),v:v.map(row=>row.map(()=>0))}});
      nodes=[
        N("token_ids","token_ids",20,160,{rows:4,vocab:8,seed:1}),
        N("prefix","slice_rows",260,65,{start:0,length:3}),
        N("labels","slice_rows",260,1220,{start:1,length:3}),
        P("embedding_table",20,420,mat(8,4,1)),
        N("lookup","gather_rows",510,180),
        N("pos","pos_encoding",735,180),
        P("wq",770,480,mat(4,4,2)),
        P("wk",770,700,mat(4,4,3)),
        P("wv",770,920,mat(4,4,4)),
        N("q","matmul",1030,50),N("k","matmul",1030,290),N("v","matmul",1030,535),
        N("kt","transpose2d",1265,300),
        N("scores","matmul",1495,180),
        N("scaled","scale",1710,175,{factor:0.5}),
        N("mask","causal_mask",1920,465),
        N("probs","masked_softmax",2140,180),
        N("context","weighted_sum",2345,175),
        N("residual","add",2545,175),
        N("norm","rmsnorm",2745,175,{epsilon:1e-5}),
        P("lm_head",2560,680,mat(4,8,5)),
        N("logits","matmul",2960,170),
        N("loss","cross_entropy",3185,170)
      ];
      edges=[
        ["token_ids","prefix","x"],["token_ids","labels","x"],
        ["prefix","lookup","indices"],["embedding_table","lookup","x"],
        ["lookup","pos","x"],
        ["pos","q","a"],["wq","q","b"],
        ["pos","k","a"],["wk","k","b"],
        ["pos","v","a"],["wv","v","b"],
        ["k","kt","x"],["q","scores","a"],["kt","scores","b"],
        ["scores","scaled","x"],
        ["scaled","mask","x"],
        ["scaled","probs","logits"],["mask","probs","mask"],
        ["probs","context","weights"],["v","context","values"],
        ["pos","residual","a"],["context","residual","b"],
        ["residual","norm","x"],
        ["norm","logits","a"],["lm_head","logits","b"],
        ["logits","loss","logits"],["labels","loss","labels"]
      ];
    }else if(name==="token_lm_trainable"){
      title="任务 36 · 真正的 Next-token 对齐：Token[t] → Label[t+1] → CE → AdamW";
      // Unlike the fixed Embedding teaching node, gather_rows is a real
      // differentiable lookup into one shared trainable Parameter table.
      const matrix=(rows,cols,seed)=>Array.from({length:rows},(_,r)=>Array.from({length:cols},(_,c)=>
        Number((Math.sin((r+1)*0.47+(c+1)*0.31+seed)*.24).toFixed(8))));
      const P=(id,x,y,values)=>N(id,"parameter",x,y,{
        values,optimizer:{t:0,m:values.map(row=>row.map(()=>0)),v:values.map(row=>row.map(()=>0))}
      });
      nodes=[
        N("token_ids","token_ids",25,145,{rows:4,vocab:8,seed:1}),
        N("input_tokens","slice_rows",300,80,{start:0,length:3}),
        N("labels","slice_rows",300,780,{start:1,length:3}),
        P("embedding_table",25,435,matrix(8,4,1)),
        N("lookup","gather_rows",585,220),
        P("lm_head",585,620,matrix(4,8,2)),
        N("logits","matmul",890,215),
        N("log_probs","log_softmax",1180,55),
        N("loss","cross_entropy",1190,300)
      ];
      edges=[
        ["token_ids","input_tokens","x"],["token_ids","labels","x"],
        ["input_tokens","lookup","indices"],["embedding_table","lookup","x"],
        ["lookup","logits","a"],["lm_head","logits","b"],
        ["logits","log_probs","x"],["logits","loss","logits"],["labels","loss","labels"]
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
  return Object.freeze({SCHEMA,defs,KINDS,MAX_NODES,MAX_EDGES,validateGraph,indexGraph,order,evaluate,connect,preset});
});
