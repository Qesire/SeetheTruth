/* SeetheTruth tensor-level graph. Deterministic pedagogical decomposition;
   not a verified runtime graph or count of physical CUDA launches. */
(function(g){"use strict";
const ops={
x:["Residual input","X","[B,S,H]","[B,S,H]","input"],
norm:["Pre-norm","Z=Norm(X)","[B,S,H]","[B,S,H]","norm"],
qw:["Query weight","Wq","[H,hq*d]","[H,hq*d]","head"],
kw:["Key weight","Wk","[H,hkv*d]","[H,hkv*d]","head"],
vw:["Value weight","Wv","[H,hkv*d]","[H,hkv*d]","head"],
q:["Query GEMM","Q=ZWq","[B,S,H]","[B,S,hq*d]","attn"],
k:["Key GEMM","K=ZWk","[B,S,H]","[B,S,hkv*d]","attn"],
v:["Value GEMM","V=ZWv","[B,S,H]","[B,S,hkv*d]","attn"],
qview:["Q reshape","Q.view(B,S,hq,d)","[B,S,hq*d]","[B,S,hq,d]","norm"],
kview:["K reshape","K.view(B,S,hkv,d)","[B,S,hkv*d]","[B,S,hkv,d]","norm"],
vview:["V reshape","V.view(B,S,hkv,d)","[B,S,hkv*d]","[B,S,hkv,d]","norm"],
qperm:["Q transpose","Q.permute(0,2,1,3)","[B,S,hq,d]","[B,hq,S,d]","norm"],
kperm:["K transpose","K.permute(0,2,1,3)","[B,S,hkv,d]","[B,hkv,S,d]","norm"],
vperm:["V transpose","V.permute(0,2,1,3)","[B,S,hkv,d]","[B,hkv,S,d]","norm"],
qrope:["RoPE(Q)","Qp=R(p)Q","[B,hq,S,d]","[B,hq,S,d]","cond"],
krope:["RoPE(K)","Kp=R(p)K","[B,hkv,S,d]","[B,hkv,S,d]","cond"],
kcache:["Key cache","K≤t=append(Kold,Knew)","[B,hkv,S,d]","[B,hkv,T,d]","head"],
vcache:["Value cache","V≤t=append(Vold,Vnew)","[B,hkv,S,d]","[B,hkv,T,d]","head"],
kexpand:["GQA map K","KV(i)=floor(i/(hq/hkv))","[B,hkv,T,d]","[B,hq,T,d] logical","cond"],
vexpand:["GQA map V","KV(i)=floor(i/(hq/hkv))","[B,hkv,T,d]","[B,hq,T,d] logical","cond"],
kt:["Transpose K","Kt=transpose(K,-1,-2)","[B,hq,T,d]","[B,hq,d,T]","norm"],
qk:["Score GEMM","Scores=Q@Kt/sqrt(d)","[B,hq,S,d]×[B,hq,d,T]","[B,hq,S,T]","attn"],
mask:["Mask / broadcast","Scores'=Scores+Mask","[B,hq,S,T]","[B,hq,S,T]","gate"],
softmax:["Softmax","A=softmax(Scores',-1)","[B,hq,S,T]","[B,hq,S,T]","attn"],
av:["Attention GEMM","O=A@V","[B,hq,S,T]×[B,hq,T,d]","[B,hq,S,d]","ffn"],
layout:["Head axis reorder","O.permute(0,2,1,3)","[B,hq,S,d]","[B,S,hq,d]","norm"],
contiguous:["Contiguous boundary","O.contiguous() if needed","[B,S,hq,d]","[B,S,hq,d]","norm"],
merge:["Concat heads / reshape","O.reshape(B,S,hq*d)","[B,S,hq,d]","[B,S,hq*d]","norm"],
wo:["Output GEMM","Y=OWo","[B,S,hq*d]","[B,S,H]","ffn"],
residual:["Residual add","Y=X+Attention(X)","[B,S,H]","[B,S,H]","residual"],
ffnorm:["FFN norm","U=Norm(X)","[B,S,H]","[B,S,H]","norm"],
gatew:["Gate weight","Wgate","[H,F]","[H,F]","head"],
upw:["Up weight","Wup","[H,F]","[H,F]","head"],
gmm:["Gate GEMM","G=U@Wgate","[B,S,H]×[H,F]","[B,S,F]","gate"],
umm:["Up GEMM","V=U@Wup","[B,S,H]×[H,F]","[B,S,F]","ffn"],
silu:["SiLU","SiLU(G)=G*sigmoid(G)","[B,S,F]","[B,S,F]","gate"],
mul:["Hadamard gate","M=SiLU(G)⊙V","[B,S,F]","[B,S,F]","cond"],
down:["Down GEMM","ΔX=M@Wdown","[B,S,F]×[F,H]","[B,S,H]","ffn"],
ffnres:["FFN residual","Y=X+ΔX","[B,S,H]","[B,S,H]","output"],
w1:["GELU MLP GEMM 1","Z=X@W1+b1","[B,S,H]×[H,F]","[B,S,F]","ffn"],
gelu:["GELU","A=GELU(Z)","[B,S,F]","[B,S,F]","gate"],
w2:["GELU MLP GEMM 2","Y=A@W2+b2","[B,S,F]×[F,H]","[B,S,H]","ffn"],
c:["Condition","c=Embed(t,condition)","[B,C]","[B,C]","input"],

routerw:["Router weight","Wr","[H,E]","[H,E]","head"],
routermm:["Routing GEMM","Scores=X@Wr","[N,H]×[H,E]","[N,E]","router"],
topk:["Top-k selection","S_i=TopK(softmax(scores_i),k)","[N,E]","[N,k]","router"],
dispatch:["Dispatch tokens","X_e={X_i:e∈S_i}","[N,H]","[N_e,H]","router"],
e1g:["Expert A gate GEMM","G_e=X_e@W_e,gate","[N_e,H]×[H,F]","[N_e,F]","gate"],
e1u:["Expert A up GEMM","U_e=X_e@W_e,up","[N_e,H]×[H,F]","[N_e,F]","ffn"],
e1act:["Expert A SiLU","A_e=SiLU(G_e)","[N_e,F]","[N_e,F]","gate"],
e1mul:["Expert A gate product","M_e=A_e⊙U_e","[N_e,F]","[N_e,F]","cond"],
e1down:["Expert A down GEMM","Y_e=M_e@W_e,down","[N_e,F]×[F,H]","[N_e,H]","ffn"],
e2g:["Expert B gate GEMM","G_e=X_e@W_e,gate","[N_e,H]×[H,F]","[N_e,F]","gate"],
e2u:["Expert B up GEMM","U_e=X_e@W_e,up","[N_e,H]×[H,F]","[N_e,F]","ffn"],
e2act:["Expert B SiLU","A_e=SiLU(G_e)","[N_e,F]","[N_e,F]","gate"],
e2mul:["Expert B gate product","M_e=A_e⊙U_e","[N_e,F]","[N_e,F]","cond"],
e2down:["Expert B down GEMM","Y_e=M_e@W_e,down","[N_e,F]×[F,H]","[N_e,H]","ffn"],
combine:["Expert combine","Y_i=Σ_{e∈S_i}p_ieY_ie","[N,k,H]","[N,H]","ffn"],
moeout:["MoE output","X'=X+MoE(X)","[N,H]","[N,H]","output"],
csilu:["Condition activation","u=SiLU(c)","[B,C]","[B,C]","gate"],
cw:["AdaLN affine GEMM","a=u@W+b","[B,C]×[C,6*H]","[B,6*H]","cond"],
split:["Six parameter slots","(shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp)=chunk(a,6)","[B,6*H]","6 × [B,H]","norm"],
shift_msa:["MSA shift slot","shift_msa=chunk(a,6)[0]","[B,6*H]","[B,H]","cond"],
scale_msa:["MSA scale slot","scale_msa=chunk(a,6)[1]","[B,6*H]","[B,H]","cond"],
gate_msa_param:["MSA gate slot","gate_msa=chunk(a,6)[2]","[B,6*H]","[B,H]","cond"],
shift_mlp_param:["MLP shift slot","shift_mlp=chunk(a,6)[3]","[B,6*H]","[B,H]","cond"],
scale_mlp_param:["MLP scale slot","scale_mlp=chunk(a,6)[4]","[B,6*H]","[B,H]","cond"],
gate_mlp_param:["MLP gate slot","gate_mlp=chunk(a,6)[5]","[B,6*H]","[B,H]","cond"],
shift:["MSA modulation","u=(1+scale_msa)⊙Norm1(x)+shift_msa","[B,S,H], [B,H]","[B,S,H]","cond"],
attn_f:["MSA attention","v=Attention(u)","[B,S,H]","[B,S,H]","attn"],
gate_msa:["MSA residual gate","delta=gate_msa⊙v","[B,H] × [B,S,H]","[B,S,H]","gate"],
msa_res:["MSA residual add","h=x+delta","[B,S,H] × [B,S,H]","[B,S,H]","residual"],
norm2:["MLP pre-norm","z=Norm2(h)","[B,S,H]","[B,S,H]","norm"],
shift_mlp:["MLP modulation","w=(1+scale_mlp)⊙Norm2(h)+shift_mlp","[B,S,H], [B,H]","[B,S,H]","cond"],
mlp_f:["MLP transform","m=MLP(w)","[B,S,H]","[B,S,H]","ffn"],
gate_mlp:["MLP residual gate","epsilon=gate_mlp⊙m","[B,H] × [B,S,H]","[B,S,H]","gate"],
adaln_out:["MLP residual add","y=h+epsilon","[B,S,H] × [B,S,H]","[B,S,H]","output"]
};
const graphs={
attention:[
["x","norm"],["norm","q"],["norm","k"],["norm","v"],["qw","q"],["kw","k"],["vw","v"],
["q","qview"],["qview","qperm"],["qperm","qrope"],
["k","kview"],["kview","kperm"],["kperm","krope"],["krope","kcache"],["kcache","kexpand"],["kexpand","kt"],
["v","vview"],["vview","vperm"],["vperm","vcache"],["vcache","vexpand"],
["qrope","qk"],["kt","qk"],["qk","mask"],["mask","softmax"],["softmax","av"],["vexpand","av"],
["av","layout"],["layout","contiguous"],["contiguous","merge"],["merge","wo"],["wo","residual"],["x","residual"]],
ffn:[["x","ffnorm"],["ffnorm","gmm"],["ffnorm","umm"],["gatew","gmm"],["upw","umm"],["gmm","silu"],["silu","mul"],["umm","mul"],["mul","down"],["down","ffnres"],["x","ffnres"]],
gpt:[["x","ffnorm"],["ffnorm","w1"],["w1","gelu"],["gelu","w2"],["w2","ffnres"],["x","ffnres"]],
adaln:[
["x","norm"],["c","csilu"],["csilu","cw"],["cw","split"],
["split","shift_msa"],["split","scale_msa"],["split","gate_msa_param"],
["split","shift_mlp_param"],["split","scale_mlp_param"],["split","gate_mlp_param"],
["norm","shift"],["shift_msa","shift"],["scale_msa","shift"],
["shift","attn_f"],["attn_f","gate_msa"],["gate_msa_param","gate_msa"],
["x","msa_res"],["gate_msa","msa_res"],["msa_res","norm2"],
["norm2","shift_mlp"],["shift_mlp_param","shift_mlp"],["scale_mlp_param","shift_mlp"],
["shift_mlp","mlp_f"],["mlp_f","gate_mlp"],["gate_mlp_param","gate_mlp"],
["msa_res","adaln_out"],["gate_mlp","adaln_out"]
]
};

graphs.mha=graphs.attention.filter(([a,b])=>!["qrope","krope"].includes(a)&&!["qrope","krope"].includes(b)).concat([["qperm","qk"],["kperm","kcache"]]);
graphs.dit=[
["x","norm"],["norm","q"],["norm","k"],["norm","v"],["qw","q"],["kw","k"],["vw","v"],
["q","qview"],["qview","qperm"],["qperm","qk"],
["k","kview"],["kview","kperm"],["kperm","kt"],["kt","qk"],
["v","vview"],["vview","vperm"],["vperm","av"],
["qk","mask"],["mask","softmax"],["softmax","av"],["av","layout"],["layout","contiguous"],["contiguous","merge"],["merge","wo"],["wo","residual"],["x","residual"]
];
graphs.moe=[
["x","routermm"],["routerw","routermm"],["routermm","topk"],["topk","dispatch"],["x","dispatch"],
["dispatch","e1g"],["dispatch","e1u"],["e1g","e1act"],["e1act","e1mul"],["e1u","e1mul"],["e1mul","e1down"],["e1down","combine"],
["dispatch","e2g"],["dispatch","e2u"],["e2g","e2act"],["e2act","e2mul"],["e2u","e2mul"],["e2mul","e2down"],["e2down","combine"],
["topk","combine"],["combine","moeout"],["x","moeout"]
];
const notes={
qview:"reshape 只规定逻辑形状；不保证零拷贝，是否复制由 stride 与实现决定。",
kperm:"transpose/permute 通常更改 stride；下游 kernel 可能要求 contiguous。",
qrope:"RoPE 是以位置为条件的特征对子旋转；不等于交换 token 位置。",
krope:"Key 与 Query 使用一致的位置编码约定；Value 通常不施加 RoPE。",
kcache:"缓存管理与位置、滑窗和 paged KV 的索引有关；当前只是逻辑示意。",
vcache:"Decode 不重复投影历史 Value；训练态通常不用增量缓存。",
kexpand:"GQA 分组可以由 kernel 逻辑广播，不意味着真实复制 KV 张量。",
vexpand:"KV 与 Query heads 的组映射必须一致；并非另一轮真实矩阵乘。",
mask:"mask 的形状、广播规则与 −∞ 数值处理决定注意力可见区域。",
softmax:"FlashAttention 等融合实现可避免实体化完整 Scores/Softmax 矩阵。",
qk:"QKᵀ 是匹配路径；和 A·V 的内容聚合路径要分开分析。",
av:"A·V 是内容路径；物理实现可能和 QKᵀ、Softmax 同在一个 fused kernel。",
layout:"交换序列轴和头轴通常产生非连续视图，不应直接视为物理复制。",
contiguous:"这里是可选内存布局边界：若实际执行不需要复制，就不应计入额外 kernel。",
merge:"多头 concat 是数学上的轴合并；真实实现常用 reshape 而不是 torch.cat。",
gmm:"Gate 与 Up 两个 GEMM 在某些架构可合并为单次更宽的投影。",
mul:"这是逐元素乘法而不是 GEMM；broadcast 形状需要验证。",
down:"稀疏化后须检查 Down 投影的输出误差，而不仅是中间通道激活。",
dispatch:"实际每个 expert 的 N_e 依动态路由决定；这里展示 A/B 两条可能路径，不断言它们必被选中。",
routermm:"路由 GEMM 之后仍需要 top-k、dispatch 和容量控制。",
combine:"不同模型可能采用归一化权重、共享专家或不同 token dropping 策略。",
cw:"Pinned Meta DiT AdaLN-Zero uses SiLU followed by one affine projection producing six H-dimensional parameters; shapes are teaching contracts, not runtime observations.",
gate_msa:"AdaLN-Zero 的残差门控是连续缩放，不是二值稀疏选择。gate_msa=0 时该分支是恒等残差。",
gate_mlp:"gate_mlp=0 时该 MLP 分支是恒等残差；调制输入不能直接跳过 F(x) 加入残差。",
shift:"AdaLN 调制产生子层输入，不是可直接加到残差状态的子层输出。",
shift_mlp:"Attention 残差之后才对 MLP 执行独立归一化与调制。"
};
function type(st){if(['adaln','condition'].includes(st.focus))return 'adaln';if(st.focus==='ffn')return st.cfg.kind==='gpt'?'gpt':st.cfg.kind==='moe'?'moe':'ffn';return st.cfg.kind==='dit'?'dit':st.cfg.kind==='gpt'?'mha':'attention'}
function build({st,palette,node,edge}){
 const pairs=graphs[type(st)], ids=[...new Set(pairs.flat())],rank=new Map(ids.map(id=>[id,0]));
 for(let i=0;i<ids.length;i++){let changed=false;for(const [a,b] of pairs){let next=rank.get(a)+1;if(next>rank.get(b)){rank.set(b,next);changed=true}}if(!changed)break}
 const buckets=new Map();for(const id of ids){const r=rank.get(id);if(!buckets.has(r))buckets.set(r,[]);buckets.get(r).push(id)}
 const max=Math.max(1,...rank.values());
 const nodes=ids.map(id=>{const r=rank.get(id),siblings=buckets.get(r),i=siblings.indexOf(id),d=ops[id];return node('tx_'+id,d[0],(i-(siblings.length-1)/2)*2.2,-4.1+r/max*8.2,(i%3-1)*.3,Math.min(2.55,1.35+d[0].length*.048),.54,.72,palette[d[4]]||palette.attn,'TENSOR')});
 const edges=pairs.map(([a,b])=>edge('tx_'+a,'tx_'+b,type(st)==='attention'?'#67dfff':'#88edca'));
 return{nodes,edges};
}
function detail(id,st){
 const key=id.slice(3),d=ops[key];if(!d)return null;const c=st.cfg;
 let q=st.mode==='decode'&&c.kind!=='dit'?1:c.s,dhead=c.h/c.a;
 const subst=s=>s.replace(/hq/g,c.a).replace(/hkv/g,c.kv).replace(/\bH\b/g,c.h).replace(/\bF\b/g,c.f).replace(/\bd\b/g,dhead).replace(/\bB\b/g,c.b).replace(/\bS\b/g,q).replace(/\bT\b/g,c.s);
 return{name:d[0],what:"逻辑张量计算： "+d[1]+"。"+(notes[key]||"张量维度与实际内存行为需要源码或 Runtime Trace 验证。"),formula:d[1],lesson:notes[key]||"教学级结构，既不能证明执行到达，也不能据此推断内存复制或 CUDA kernel 数量。",shape:subst(d[2]),out:subst(d[3]),tags:["Tensor micrograph","LOGICAL",d[4]]};
}
g.SeetheTruthTensor={build,detail,ops,graphs};
})(window);