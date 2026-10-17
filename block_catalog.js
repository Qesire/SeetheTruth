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
