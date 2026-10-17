/* A Scratch-like scalar training *teaching* state machine, not an autograd
 * engine, large-matrix emulator, model training or GPU integration.
 *
 * Forward & reverse derivatives for L_i = (theta*x_i-y_i)^2/2 are exact.
 * Microbatch averaging, gradient-buffer accumulation, SGD and decoupled AdamW
 * state transitions use the same math conventions as reference implementations.
 * Symbolic mode returns annotated transitions but NO numerical values.
 */
(function(root,make){"use strict";const api=make();
if(typeof module==="object"&&module.exports)module.exports=api;
root.SeetheTruthTraining=api;
})(typeof window!=="undefined"?window:{},function(){"use strict";
const SOURCE={
 autograd:{title:"PyTorch: Autograd mechanics",url:"https://docs.pytorch.org/docs/stable/notes/autograd.html"},
 grad:{title:"PyTorch: zero_grad and None vs zero",url:"https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"},
 adamw:{title:"PyTorch: AdamW, bias correction and decoupled weight decay",url:"https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html"},
 sgd:{title:"PyTorch: SGD optimizer",url:"https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html"},
 mse:{title:"PyTorch: MSELoss reduction contracts",url:"https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html"}
};
const BLOCKS={
 parameter:{title:"参数 θ",category:"parameter",formula:"θ ∈ ℝ; requires_grad=True",
  how:"模型里可学习且持久的叶节点参数。前向计算读取当前 θ；反向梯度会累积到 θ.grad，直到显式清理。",
  practice:"参数值、梯度 buffer 和 optimizer 内部状态是三种不同的存储对象。真实 Transformer 参数可为巨大矩阵，本课程刻意只演示一个标量。",
  trap:"前向读取 θ 并不会更新 θ；调用 backward() 只产生梯度，optimizer.step() 才会尝试更新。",
  test:"不执行 Optimizer Step 时，第二个 microbatch 是否改变 θ？",refs:["autograd"]},
 zero_grad:{title:"清空梯度缓存",category:"state",formula:"θ.grad ← None",
  how:"开始一次有效 batch 的梯度累积窗口时，清理上次 step 留下的梯度。当前采用 set_to_none=True 的语义。",
  practice:"PyTorch 中 None 与数值零梯度不同：某些参数无梯度时，optimizer 可跳过其 Step；训练循环常在多个 backward 前仅清空一次。",
  trap:"如果在两个 microbatch 的 backward 之间 zero_grad，会抹去第一个 microbatch 的贡献。",
  test:"把 zero_grad 放进 microbatch 循环会破坏什么不变量？",refs:["grad","autograd"]},
 microbatch:{title:"取出一个 microbatch",category:"data",formula:"(xᵢ, yᵢ) · θ固定",
  how:"本玩具任务给出一个标量输入和监督目标；每个 microbatch 只读取同一份当前参数，并不在 microbatch 间更新 θ。",
  practice:"真正的 batch 可能含多个 token、padding mask 和复杂 reduction。这里每个 microbatch 恰好一个有效样本，因此平均权重相等。",
  trap:"Microbatch 数不等于序列长度；真实训练中 token 数不等的 microbatch 不能简单对 batch loss 做不加权均值。",
  test:"两个输入共用哪份 θ？改变 microbatch 顺序是否改变精确平均梯度？",refs:["mse","autograd"]},
 forward:{title:"Forward 预测",category:"forward",formula:"ŷᵢ = θ · xᵢ",
  how:"读取叶参数 θ 与当前数据 x，把可学习系数乘以输入，得到预测。前向机制不会写入 θ 或梯度缓存。",
  practice:"实际 Autograd 在 forward 期间记录所需算子的反向依赖；图中的相乘只是整个大模型计算流的最小局部样例。",
  trap:"此标量乘法可代表一处可微路径，不能外推真实 Transformer 的激活/参数矩阵或整模型语义。",
  test:"固定 θ 改 x，输出与反向对 θ 的敏感度分别怎么变？",refs:["autograd"]},
 loss:{title:"Loss 前向",category:"loss",formula:"ℓᵢ = ½(ŷᵢ−yᵢ)²",
  how:"构造预测误差并平方后乘 1/2；系数 1/2 是为了使导数简洁，仍是一种平方误差目标。",
  practice:"PyTorch MSELoss 的定义默认没有前面的 1/2，本课程明确使用 half-squared-error，不把二者的梯度系数混为一谈。",
  trap:"Loss 标量没有自动触发参数更新；还需反向、缓冲梯度与 optimizer step。",
  test:"如果去掉 1/2，损失及梯度都如何变化？",refs:["mse","autograd"]},
 seed_grad:{title:"反向起点",category:"backward",formula:"∂ℓᵢ/∂ŷᵢ = ŷᵢ−yᵢ",
  how:"从标量 Loss 向它的直接输入反向传播局部导数；误差可正可负，决定随后参数梯度的方向。",
  practice:"反向使用计算图和链式法则；真实多层图沿各依赖边累加 cotangent，不是简单把前向图箭头反过来。",
  trap:"这还不是参数 θ 的梯度，必须继续经过前向乘法对 θ 的局部导数。",
  test:"当预测刚好等于目标时，反向起点如何变化？",refs:["autograd"]},
 chain_rule:{title:"链式法则回到参数",category:"backward",formula:"∂ℓᵢ/∂θ=(ŷᵢ−yᵢ)·xᵢ",
  how:"由 ∂ŷᵢ/∂θ=xᵢ，沿 Forward 依赖链应用链式法则；得到该 microbatch 对叶参数的梯度。",
  practice:"复杂网络可能有多路贡献回到同一参数；梯度应按图中全部相关路径累加。这里只有一个路径。",
  trap:"这不是梯度的绝对值，负梯度也完全合法；优化器通常沿负梯度方向更新。",
  test:"xᵢ=0 时，即使预测误差非零，为什么这条路径的 θ 梯度为 0？",refs:["autograd"]},
 loss_scale:{title:"累积平均缩放",category:"backward",formula:"gᵢ = (∂ℓᵢ/∂θ)/N",
  how:"本课每个 microbatch 含同样数量的有效样本，因此把每次 loss/grad 除以 N，可以在 N 次 backward 后得到整体平均梯度。",
  practice:"梯度累积中 loss 除以 accumulation_steps 是常见写法，但不同有效 token 数、不同 reduction、DDP 等情形需要按真正的有效权重处理。",
  trap:"每份梯度不缩放直接相加是 sum-gradient，不是 mean-gradient；等效学习率因此不同。",
  test:"N=2 时不除以 2，会让标准 SGD 的单次更新增大多少倍？",refs:["autograd","mse"]},
 accumulate:{title:"累加参数梯度",category:"state",formula:"θ.grad ← (θ.grad ?? 0)+gᵢ",
  how:"把当前微批次缩放后的梯度加入叶参数的 grad buffer。这里区分开始时的 None 与已经有具体数值的状态。",
  practice:"PyTorch backward 对叶参数的 grad 默认累积，而不是每次覆盖；所以 zero_grad 时机直接影响优化语义。",
  trap:"梯度累积不是把多份参数更新做 N 次：整个窗口内 θ 必须保持不变，到窗口末尾才调用一次 optimizer。",
  test:"先 g1 后 g2 与先 g2 后 g1 在精确数学下是否相同？浮点有限精度呢？",refs:["autograd","grad"]},
 gradient_ready:{title:"完成有效 batch 梯度",category:"state",formula:"g=Σᵢ ∂ℓᵢ/∂θ ÷ N",
  how:"当 N 个微批次都完成 backward，缓存里的值就是这个教学批次平均 half-squared-error 的梯度。",
  practice:"真实分布式训练通常涉及跨设备梯度聚合与归一化；这个单参数例子不模拟通信，也不宣称性能。",
  trap:"累积 buffer 和 AdamW 的一阶动量 m 不同：前者属于本次 batch 的梯度，后者跨优化步骤持续存在。",
  test:"N=2，若每个 microbatch 单独更新一次 θ，与先平均梯度更新一次是否相同？",refs:["autograd","adamw"]},
 adam_moment:{title:"AdamW 一、二阶矩",category:"optimizer",formula:"mₜ=β₁mₜ₋₁+(1−β₁)g; vₜ=β₂vₜ₋₁+(1−β₂)g²",
  how:"AdamW 的 m、v 与参数 θ 一起跨 optimizer step 持久保存；m 近似追踪梯度平均，v 追踪平方梯度。",
  practice:"实际 AdamW 为每个受优化的参数维护相应状态，betas/eps 与 dtype 可能影响长期行为。",
  trap:"这里的“二阶矩”是梯度平方的指数移动平均，不是 Hessian、曲率矩阵或真正二阶优化。",
  test:"两次梯度符号相反时，m 与 v 的反应如何不同？",refs:["adamw"]},
 bias_correct:{title:"AdamW 偏差校正",category:"optimizer",formula:"m̂ₜ=mₜ/(1−β₁ᵗ); v̂ₜ=vₜ/(1−β₂ᵗ)",
  how:"m、v 初值均为零，初期指数移动平均偏小；t 从 1 开始，偏差校正消除这一初始化影响。",
  practice:"bias correction 的次数是 optimizer step 的次数，而非调用 backward 的次数、microbatch 数或 token 数。",
  trap:"忘记偏差校正通常会改变早期更新规模；t 不应在每个 microbatch 后增加。",
  test:"第一步 m₁=(1−β₁)g，校正后 m̂₁ 应等于多少？",refs:["adamw"]},
 decay:{title:"解耦权重衰减",category:"optimizer",formula:"θᵈ=θₜ₋₁(1−ηλ)",
  how:"AdamW 的 weight decay 直接作用在参数上，独立于梯度的动量估计，随后叠加自适应梯度更新。",
  practice:"AdamW decoupled decay 不同于将 λθ 放进 gradient 再求 m/v 的 L2 风格实现；有些参数组明确排除 bias/norm。",
  trap:"不能把 AdamW 的 weight_decay 称为所有架构的一般 L2 梯度正则项；SGD 对照课不含 decay。",
  test:"当 g=0 且 λ>0 时，AdamW 的 θ 还会改变吗？",refs:["adamw"]},
 update:{title:"Optimizer Step",category:"optimizer",formula:"SGD: θ←θ−ηg; AdamW: θ←θ(1−ηλ)−η m̂/(√v̂+ε)",
  how:"优化器读取当前参数、整次有效 batch 的梯度和其自身状态，进行本轮唯一的一次参数更新。",
  practice:"optimizer.step() 与 backward() 属于不同动作；update 改变的是参数，下一个 Forward 才使用新参数。",
  trap:"一次 Step 之后不一定每个 batch Loss 下降；AdamW 使用状态与衰减，不等价 SGD。",
  test:"改变 learning rate 会影响什么？在相同 g 下 SGD 和 AdamW 的方向/步幅必然相同吗？",refs:["adamw","sgd"]}
};
function validateConfig(c){
 const defaults={theta:.4,x1:2,y1:1,x2:-1,y2:2,optimizer:"adamw",
  lr:.1,weightDecay:.02,beta1:.9,beta2:.999,eps:1e-8,microbatches:2,steps:2};
 const v={...defaults,...c};
 for(const k of ["theta","x1","y1","x2","y2"])
  if(typeof v[k]!=="number"||!Number.isFinite(v[k])||Math.abs(v[k])>10)throw Error(k+" 必须是 -10 到 10 的有限数");
 for(const k of ["lr","weightDecay","beta1","beta2","eps"])
  if(typeof v[k]!=="number"||!Number.isFinite(v[k]))throw Error(k+" 需为有限数字");
 if(!(v.lr>0&&v.lr<=1)||v.weightDecay<0||v.weightDecay>1||
   !(v.beta1>=0&&v.beta1<1&&v.beta2>=0&&v.beta2<1)||
   !(v.eps>0&&v.eps<=.1))throw Error("学习率、beta、weight decay、eps 参数不符合数学定义");
 if(!["sgd","adamw"].includes(v.optimizer))throw Error("optimizer 必须是 sgd 或 adamw");
 if(!Number.isInteger(v.microbatches)||v.microbatches<1||v.microbatches>2)throw Error("microbatches 只能是 1 或 2");
 if(!Number.isInteger(v.steps)||v.steps<1||v.steps>2)throw Error("steps 只能是 1 或 2");
 return v;
}
const SAMPLE=[
 {x:"x₁",y:"y₁",pred:"θ·x₁",loss:"½(θ·x₁−y₁)²",deriv:"(θ·x₁−y₁)·x₁"},
 {x:"x₂",y:"y₂",pred:"θ·x₂",loss:"½(θ·x₂−y₂)²",deriv:"(θ·x₂−y₂)·x₂"}
];
function build(params={},mode="concept"){
 const cfg=validateConfig(params);
 if(mode!=="concept"&&mode!=="scalar")throw Error("mode 必须为 concept 或 scalar");
 // Dedicated no-arithmetic conceptual execution. Do not call numerical build,
 // nor calculate even toy gradients while this branch is selected.
 if(mode==="concept"){
   const events=[];
   const emit=(kind,phase,label,symbol,detail)=>{
     events.push({id:"e"+events.length,kind,phase,label,
       formula:BLOCKS[kind].formula,symbol,detail,value:null,state:null,meta:{}});
   };
   for(let epoch=0;epoch<cfg.steps;epoch++){
     const tag="第 "+(epoch+1)+" 次有效批次";
     emit("parameter",tag,"读取叶参数 θ","θ₍"+(epoch+1)+"₎","同一有效批次期间参数保持不变。");
     emit("zero_grad",tag,"清除上一轮梯度","θ.grad=None","只清除叶参数 grad，不清除 AdamW 的 m/v。");
     for(let i=0;i<cfg.microbatches;i++){
       const s=SAMPLE[i],sub=tag+" / microbatch "+(i+1);
       emit("microbatch",sub,"读取输入与标签","("+s.x+","+s.y+")","读取一份有效标量样本。");
       emit("forward",sub,"Forward 预测",s.pred,"读取参数的局部可微前向路径。");
       emit("loss",sub,"半平方误差 Loss",s.loss,"前向 Loss，不代表 backward 或 step。");
       emit("seed_grad",sub,"Backward 起点","∂ℓ/∂ŷ=("+s.pred+"−"+s.y+")",
          "用 half-squared-error 的局部导数。");
       emit("chain_rule",sub,"反传到 θ","∂ℓ/∂θ="+s.deriv,"乘以 ∂ŷ/∂θ="+s.x+"。");
       emit("loss_scale",sub,"按有效样本数缩放",
         "g"+(i+1)+"=("+s.deriv+")/"+cfg.microbatches,
         "等权 microbatch 的平均梯度贡献。");
       emit("accumulate",sub,"加入梯度 buffer","θ.grad=Σⱼ≤"+(i+1)+" gⱼ",
         "追加当前微批次的梯度，不改变 θ。");
     }
     emit("gradient_ready",tag,"有效批次平均梯度",
       "g=Σᵢ(∂ℓᵢ/∂θ)/"+cfg.microbatches,
       "所有 microbatch 的贡献已归一化并累积。");
     if(cfg.optimizer==="adamw"){
       emit("adam_moment",tag,"更新 m / v","mₜ=β₁mₜ₋₁+(1−β₁)g; vₜ=β₂vₜ₋₁+(1−β₂)g²",
        "跨 optimizer step 追踪一阶和平方梯度。");
       emit("bias_correct",tag,"校正初始零偏差","m̂ₜ=mₜ/(1−β₁ᵗ), v̂ₜ=vₜ/(1−β₂ᵗ)",
        "偏差校正计数只与 optimizer step 有关。");
       emit("decay",tag,"参数解耦衰减","θᵈ=θ(1−ηλ)",
        "在梯度自适应项之外执行解耦衰减。");
     }
     emit("update",tag,cfg.optimizer.toUpperCase()+" 参数更新",
       cfg.optimizer==="sgd"?"θₜ=θₜ₋₁−ηg":"θₜ=θₜ₋₁(1−ηλ)−η·m̂/(√v̂+ε)",
       "完成一次有效批次更新，下一轮前向读取新参数。");
   }
   return {config:cfg,mode,blocks:BLOCKS,sources:SOURCE,events,
     evidence:"symbolic_state_contract_no_numeric_training",final:null};
 }
 let theta=cfg.theta,grad=null,m=0,v=0,t=0;
 const events=[];
 const state=()=>mode==="scalar"?{theta,grad,m,v,step:t}:null;
 const add=(kind,phase,label,symbol,detail,value=null,meta={})=>{
   const block=BLOCKS[kind];
   events.push({id:"e"+events.length,kind,phase,label,formula:block.formula,
      symbol,detail,value:mode==="scalar"?value:null,
      state:state(),meta});
 };
 for(let epoch=0;epoch<cfg.steps;epoch++){
  const round=epoch+1,tag="第 "+round+" 次有效批次";
  add("parameter",tag,"读取叶参数 θ","θ₍"+round+"₎","参数进入本轮 forward 前保持不变。",theta);
  grad=null;
  add("zero_grad",tag,"清除上一轮梯度","θ.grad=None","清空后开始新的梯度累积窗口；这不会重置 AdamW 的 m/v。",null);
  const localGrads=[];
  for(let i=0;i<cfg.microbatches;i++){
   const s=SAMPLE[i],x=i?cfg.x2:cfg.x1,y=i?cfg.y2:cfg.y1;
   const pred=theta*x,err=pred-y,loss=.5*err*err,local=err*x,scaled=local/cfg.microbatches;
   const sub=tag+" / microbatch "+(i+1);
   add("microbatch",sub,"读取输入与标签","("+s.x+","+s.y+")","该轮每个 microbatch 都使用相同 θ；等权，每份只有一个有效标量样本。",null,{micro:i+1});
   add("forward",sub,"Forward 预测",s.pred,"参数只读；前向乘法把 θ 与输入映射为预测 ŷ。",pred);
   add("loss",sub,"半平方误差 Loss",s.loss,"注意本课程 ℓ=½e²，不是 PyTorch MSELoss 默认的 e²。",loss);
   add("seed_grad",sub,"Backward 起点","∂ℓ/∂ŷ=("+s.pred+"−"+s.y+")",
     "局部梯度沿计算图反向传播；还未到 θ。",err);
   add("chain_rule",sub,"反传到 θ","∂ℓ/∂θ="+s.deriv,
     "链式法则使用 ∂ŷ/∂θ="+s.x+"；反向与参数更新是两个独立动作。",local);
   add("loss_scale",sub,"按有效样本数缩放","g"+(i+1)+"=("+s.deriv+")/"+cfg.microbatches,
     "微批次样本量相同，因此本轮采用 1/N；不等量的 token batch 需另设计权重。",scaled);
   grad=(grad===null?0:grad)+scaled;
   localGrads.push(local);
   add("accumulate",sub,"加入梯度 buffer",
     "θ.grad=Σⱼ≤"+(i+1)+" gⱼ",
     "仅在这里写入梯度缓存；θ 此时仍不变。",grad,{completed:i+1,total:cfg.microbatches});
  }
  add("gradient_ready",tag,"有效批次平均梯度",
     "g=Σᵢ(∂ℓᵢ/∂θ)/"+cfg.microbatches,
     "所有 microbatch 累积完毕，才可执行一次 optimizer.step()。",grad);
  const old=theta;
  if(cfg.optimizer==="adamw"){
   t++;
   m=cfg.beta1*m+(1-cfg.beta1)*grad;
   v=cfg.beta2*v+(1-cfg.beta2)*grad*grad;
   add("adam_moment",tag,"更新 m / v","mₜ=β₁mₜ₋₁+(1−β₁)g; vₜ=β₂vₜ₋₁+(1−β₂)g²",
     "跨有效 batch 持续的 optimizer state。t 只在 optimizer step 增长。",m,{m,v});
   const mh=m/(1-cfg.beta1**t),vh=v/(1-cfg.beta2**t);
   add("bias_correct",tag,"校正初始零偏差","m̂ₜ=mₜ/(1−β₁ᵗ), v̂ₜ=vₜ/(1−β₂ᵗ)",
     "bias correction 用 optimizer step 计数，不用 microbatch 计数。",mh,{mhat:mh,vhat:vh});
   const decayed=old*(1-cfg.lr*cfg.weightDecay);
   add("decay",tag,"参数解耦衰减","θᵈ=θ(1−ηλ)","权重衰减不进入 m/v 的梯度动量通道。",decayed);
   theta=decayed-cfg.lr*mh/(Math.sqrt(vh)+cfg.eps);
  }else{
   t++;
  }
  add("update",tag,cfg.optimizer.toUpperCase()+" 参数更新",
    cfg.optimizer==="sgd"?"θₜ=θₜ₋₁−ηg":"θₜ=θₜ₋₁(1−ηλ)−η·m̂/(√v̂+ε)",
    "将本批次梯度用于一次且仅一次参数更新；下一轮才读取新 θ。",theta,
    {old,new:theta,gradient:grad,optimizer:cfg.optimizer,optimizerStep:t});
 }
 return {config:cfg,mode,blocks:BLOCKS,sources:SOURCE,events,
   evidence:mode==="concept"?"symbolic_state_contract_no_numeric_training":"deterministic_scalar_analytic_oracle_not_autograd",
   final:mode==="scalar"?{theta,grad,m,v,step:t}:null};
}
function validate(){
 const types=Object.keys(BLOCKS);
 if(types.length!==14)throw Error("unexpected training concept inventory "+types.length);
 for(const [id,b] of Object.entries(BLOCKS)){
   for(const [k,minLength] of Object.entries({title:3,formula:5,how:18,practice:18,trap:18,test:18}))
     if(!b[k]||b[k].length<minLength)throw Error("incomplete training block "+id+"."+k);
   for(const ref of b.refs)if(!SOURCE[ref])throw Error("missing reference "+id+"/"+ref);
 }
 const q=build({}, "scalar");
 if(!q.events.length||q.final.step!==2)throw Error("training invariant");
 return {types:types.length,stages:q.events.length,sourceCount:Object.keys(SOURCE).length};
}
const contract=validate();
return Object.freeze({BLOCKS,SOURCE,build,validateConfig,validate,contract});
});
