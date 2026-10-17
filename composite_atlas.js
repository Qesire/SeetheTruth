/* Hierarchical model learning views. No new numeric op types and no GPU dependency.
 * Macro nodes are declarative teaching groups; expanded leaves are the same
 * validated 45 kinds from learning_lab.js and concept_flow.js.
 */
(function(root,create){"use strict";
 const api=create(typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning,
   typeof module==="object"&&module.exports?require("./concept_flow.js"):root.SeetheTruthConceptFlow);
 if(typeof module==="object"&&module.exports)module.exports=api;
 root.SeetheTruthComposites=api;
})(typeof window!=="undefined"?window:{},function(L,C){
 "use strict";
 function n(id,kind,x,y,params={}){return {id,kind,x,y,params};}
 function graph(title,nodes,edges){
   return L.validateGraph({schema:L.SCHEMA,title,nodes,
     edges:edges.map(([from,to,port])=>({from,to,port}))});
 }
 const notes={
  decoder:{
   title:"Decoder Layer · 以 Pre-Norm + GQA + SwiGLU 为例",kind:"architecture",
   intro:"整体先让 token 借 Attention 与其他位置交换信息，再通过 FFN 独立变换每个 token 的通道。两条残差支路携带主状态。",
   limits:"这是一种有代表性的教学选型，不是 GPT/LLaMA/Wan 的通用固定源码；真实 Norm、残差、门控和缓存位置须具体核对。",
   nodes:[
    n("x","input",20,285,{rows:3,width:4,seed:2}),
    n("norm1","rmsnorm",275,285),n("q","linear",550,45,{out:8}),
    n("k","linear",550,250,{out:4}),n("v","linear",550,450,{out:4}),
    n("attn","gqa",830,250,{heads:2,kvHeads:1,causal:true,rope:true}),
    n("out","linear",1090,250,{out:4}),n("res1","add",1360,280),
    n("norm2","rmsnorm",1620,280),n("gate","linear",1880,100,{out:8}),
    n("up","linear",1880,435,{out:8}),n("sw","swiglu",2140,270),
    n("down","linear",2400,270,{out:4}),n("res2","add",2680,310)
   ],
   edges:[
    ["x","norm1","x"],["norm1","q","x"],["norm1","k","x"],["norm1","v","x"],
    ["q","attn","q"],["k","attn","k"],["v","attn","v"],["attn","out","x"],
    ["x","res1","a"],["out","res1","b"],["res1","norm2","x"],
    ["norm2","gate","x"],["norm2","up","x"],["gate","sw","gate"],["up","sw","up"],
    ["sw","down","x"],["res1","res2","a"],["down","res2","b"]
   ],
   stages:[
    {title:"01 / Attention Pre-Norm",node:"norm1",detail:"在 Attention 分支入口按 RMS 调尺度；主残差 x 不经过该 Norm。",child:"norm"},
    {title:"02 / Q/K/V + GQA",node:"attn",detail:"经三条可学习投影，进入 Q 头分组、RoPE、因果屏蔽和 Value 汇聚。",child:"attention_gqa"},
    {title:"03 / Attention 残差",node:"res1",detail:"x1=x+Attention(RMSNorm(x))，从这里把更新流与原 x 汇合。",child:"residual"},
    {title:"04 / FFN Pre-Norm",node:"norm2",detail:"再对更新后的 x1 调尺度；FFN 不直接改写跨 token 的 Attention 权重。",child:"norm"},
    {title:"05 / Gated FFN",node:"down",detail:"gate/up 双投影，经 SiLU 门控逐元素相乘后映射回残差宽度。",child:"ffn"},
    {title:"06 / FFN 残差",node:"res2",detail:"x2=x1+FFN(RMSNorm(x1))，得到这一层的输出。",child:"residual"}
   ]
  },
  attention_gqa:{
   title:"GQA Attention · 投影、头共享与输出",kind:"subsystem",
   intro:"Q 决定从何处读取，K 匹配地址，V 提供内容；Hq 个查询头共用 Hkv 个 K/V 头。",
   limits:"GQA 内核的按头展开、cache 策略、Mask 非方阵对齐与 RoPE 维度必须看特定实现；该图只表示一批小型数值或符号语义。",
   preset:"gqa",
   stages:[
    {title:"01 / QKV 投影",node:"q",detail:"Q:[Tq,Hq·Dk]，K:[Tk,Hkv·Dk]，V:[Tk,Hkv·Dv]；线性投影是独立参数路径。",child:"qkv"},
    {title:"02 / 多头与 KV 共享",node:"gqa",detail:"每个 Q 头选取对应的 KV Head。GQA 与 MHA/MQA 的区别是 Hq:Hkv，不是 token 数。",child:null},
    {title:"03 / QKᵀ → Softmax → PV",node:"gqa",detail:"按头生成 [Tq,Tk] 分数，经 Mask/Softmax 后汇聚 V。下层示例明确是单头且无 Mask，不能冒充完整 GQA。",child:"attention_math"},
    {title:"04 / 输出投影",node:"out",detail:"拼接各头结果后由 Linear 映射回残差维；真实投影矩阵列次序需与 Head 顺序一致。",child:null}
   ]
  },
  qkv:{
   title:"Q/K/V · 含义相同但用途不同的投影",kind:"subsystem",
   intro:"三条路径形式上都是 Linear，但由下游被当作 Query、Key、Value 才获得不同功能。",
   limits:"数学上三个矩阵可融合为一次大 Linear 再按特征拆分；这是等价实现选择，非必然三个独立 kernel。",
   preset:"attention",
   stages:[
    {title:"01 / Query",node:"q",detail:"每个 Query 位置携带要寻找什么信息的匹配向量；输出参与分数行轴。"},
    {title:"02 / Key",node:"k",detail:"每个 Key 位置携带可被匹配的地址向量；经转置形成分数列轴。"},
    {title:"03 / Value",node:"v",detail:"与 Key 的位置数对齐，携带真正被聚合的内容；维度不必须等于 Dk。"},
    {title:"04 / Attention 汇聚",node:"attn",detail:"按 QK 的匹配权重混合 V；需要具体定义 Softmax 的归一化轴。",child:"attention_math"}
   ]
  },
  attention_math:{
   title:"单头 Attention · 展开每次矩阵运算",kind:"operator",
   intro:"把未屏蔽的缩放点积注意力拆为独立矩阵相乘、缩放、归一化和第二次相乘，供学习者逐步追踪收缩轴。",
   limits:"此处特意只演示一头且无因果 Mask、RoPE 和缓存，不等价于完整 GQA。1/√Dk 的教学例 Dk=4，因此 α=0.5。",
   nodes:[
    n("x","input",20,230,{rows:3,width:4,seed:3}),n("q","linear",290,25,{out:4}),
    n("k","linear",290,235,{out:4}),n("v","linear",290,465,{out:4}),
    n("kt","transpose2d",555,240),n("qk","matmul",825,95),
    n("scale","scale",1080,95,{factor:0.5}),n("soft","softmax",1335,95),
    n("pv","matmul",1590,250),n("proj","linear",1870,250,{out:4})
   ],
   edges:[
    ["x","q","x"],["x","k","x"],["x","v","x"],["k","kt","x"],
    ["q","qk","a"],["kt","qk","b"],["qk","scale","x"],
    ["scale","soft","x"],["soft","pv","a"],["v","pv","b"],["pv","proj","x"]
   ],
   stages:[
    {title:"01 / QKᵀ",node:"qk",detail:"Q[Tq,Dk] × Kᵀ[Dk,Tk] → scores[Tq,Tk]；Dk 是求和轴。"},
    {title:"02 / 缩放 1/√Dk",node:"scale",detail:"控制点积的典型数值尺度；本例 Dk=4，因此固定 factor=0.5。"},
    {title:"03 / 沿 Key 的 Softmax",node:"soft",detail:"在 Tk 轴计算归一化，而不是跨 Query 或 Feature 轴。"},
    {title:"04 / P × V",node:"pv",detail:"P[Tq,Tk] × V[Tk,Dv] → [Tq,Dv]；Tk 是求和轴。"},
    {title:"05 / Output Linear",node:"proj",detail:"将上下文映射至下游隐藏特征宽度，属于独立的可学习变换。"}
   ]
  },
  ffn:{
   title:"SwiGLU FFN · 低层通道计算的组合",kind:"subsystem",
   intro:"FFN 通常在各 token 的通道轴进行非线性变换；gate/up 双支路控制信息通过方式。",
   limits:"本文教学结构为 SiLU(Gate)⊙Up，再 Down；不能由连续 gate 推断硬稀疏或者自动算力节约。",
   preset:"swiglu",
   stages:[
    {title:"01 / Gate 投影",node:"gate",detail:"XWgᵀ 生成门控支路的连续中间特征；单独看 Linear 并不能知道它的用途。",child:"ffn_expanded"},
    {title:"02 / Up 投影",node:"up",detail:"XWuᵀ 生成候选内容支路，与 Gate shape 完全对齐。",child:"ffn_expanded"},
    {title:"03 / SwiGLU",node:"sw",detail:"SiLU(Gate) 逐元素乘以 Up；只有这一步发生双路信息交互。",child:"ffn_expanded"},
    {title:"04 / Down 投影",node:"down",detail:"将中间特征投影回主残差宽度，为后续 Add 准备。",child:"ffn_expanded"}
   ]
  },
  ffn_expanded:{
   title:"FFN 拆到最细 · Linear → SiLU → Multiply → Linear",kind:"operator",
   intro:"两路上投影的独立输出进入激活和逐元素乘法，最后经过下投影；本图数值上与父级 SwiGLU FFN 等价。",
   limits:"教学权重 ID 固定以构造可复验等价性；真实模型可能采用合并投影、不同 bias、不同门控顺序。",
   nodes:[
    n("x","input",25,220,{rows:3,width:4,seed:2}),
    n("gate","linear",295,55,{out:8}),n("up","linear",295,395,{out:8}),
    n("act","silu",560,55),n("mul","mul",820,190),n("down","linear",1110,190,{out:4})
   ],
   edges:[
    ["x","gate","x"],["x","up","x"],["gate","act","x"],
    ["act","mul","a"],["up","mul","b"],["mul","down","x"]
   ],
   stages:[
    {title:"01 / W_gate",node:"gate",detail:"每 token 对输入特征投影，产生门控的中间宽度。"},
    {title:"02 / SiLU",node:"act",detail:"逐坐标 SiLU(x)=xσ(x)，只改变数值不改变 shape。"},
    {title:"03 / W_up",node:"up",detail:"另一路独立线性投影，提供门控乘法的内容。"},
    {title:"04 / Hadamard",node:"mul",detail:"SiLU(G)⊙U 没有矩阵维收缩，仍输出 [T,Dff]。"},
    {title:"05 / W_down",node:"down",detail:"在 Dff 维收缩后回到残差 D，注意这一步才是矩阵乘法。"}
   ]
  },
  norm:{
   title:"Normalization · LayerNorm / RMSNorm 对照",kind:"subsystem",
   intro:"归一化操作保留 token 和通道宽度，但对整个特征向量的统计有不同处理。",
   limits:"本课程固定无仿射参数；真实模型的 γ/β、epsilon、计算 dtype、Pre/Post-Norm 位置要独立核实。",
   preset:"norm",
   stages:[
    {title:"01 / LayerNorm",node:"ln",detail:"按 token 减去每行 mean，再除以 sqrt(Var+ε)，可见中心化作用。"},
    {title:"02 / RMSNorm",node:"rms",detail:"不减均值，只除以 sqrt(mean(x²)+ε)，均值不一定为 0。"},
    {title:"03 / 输出方差对比",node:"ln_var",detail:"用同一行有偏总体方差观测两种归一化输出的差别。"}
   ]
  },
  residual:{
   title:"Residual · 不是所有 Add 都叫残差",kind:"subsystem",
   intro:"一条支路保留输入 x，另一条对 x 进行变换 F，最终相加形成 x+F(x)。",
   limits:"此处仅示意恒等跳跃路径，不是特定 Decoder 的 Norm、门控或初始化策略。",
   preset:"residual",
   stages:[
    {title:"01 / Identity",node:"x",detail:"原输入 x 直接参与输出加法，没有通过非线性分支。"},
    {title:"02 / Transformation",node:"act",detail:"F(x) 中的线性与激活可以改变数值，不改变残差的主输出 shape。"},
    {title:"03 / Addition",node:"sum",detail:"y=x+F(x)，反向传播时存在直接的恒等导数路径。"}
   ]
  }
 };
 function validate(){
   for(const [key,module] of Object.entries(notes)){
     if(!module.title||!module.intro||!module.limits||!Array.isArray(module.stages)||module.stages.length<3)
       throw Error("Invalid composite "+key);
     const g=module.preset?L.preset(module.preset):graph(module.title,module.nodes,module.edges);
     const run=C.evaluate(g);
     const ids=new Set(g.nodes.map(node=>node.id));
     for(const stage of module.stages){
       if(!ids.has(stage.node)||!run.steps.some(x=>x.id===stage.node))throw Error("Missing stage node "+key+"."+stage.node);
       if(stage.child&&!notes[stage.child])throw Error("Missing child "+key+"."+stage.child);
       if(stage.detail.length<20)throw Error("Shallow module explanation "+key);
     }
   }
   // Macro navigation must remain an acyclic pedagogical hierarchy.
   function visit(id,path){
     if(path.includes(id))throw Error("Recursive drilldown "+path.join("/")+"/"+id);
     for(const stage of notes[id].stages)if(stage.child)visit(stage.child,[...path,id]);
   }
   visit("decoder",[]);
   return {modules:Object.keys(notes).length,stageCount:Object.values(notes).reduce((s,m)=>s+m.stages.length,0)};
 }
 function get(id){return Object.hasOwn(notes,id)?notes[id]:null;}
 function build(id){
   const m=get(id);if(!m)throw Error("Unknown teaching module: "+id);
   return m.preset?L.preset(m.preset):graph(m.title,m.nodes,m.edges);
 }
 function stages(id){
   const m=get(id);if(!m)throw Error("Unknown teaching module: "+id);
   const trace=C.evaluate(build(id));
   return m.stages.map((item,i)=>({
     ...item,index:i,shape:trace.steps.find(x=>x.id===item.node).shape.slice(),
     formula:L.defs[build(id).nodes.find(n=>n.id===item.node).kind].formula,
     nodeKind:build(id).nodes.find(n=>n.id===item.node).kind
   }));
 }
 const checks=validate();
 return Object.freeze({get,build,stages,validate,checks,ids:Object.freeze(Object.keys(notes))});
});
