/* Pinned Wan2.2-I2V-A14B source topology + deterministic tiny Float64 arithmetic.
 * The numerical weights / upstream attention outputs are NOT official Wan tensors. */
(function(root,factory){'use strict';const a=factory();if(typeof module==='object'&&module.exports)module.exports=a;root.SeetheTruthWanFFNTrace=a;})(typeof window!=='undefined'?window:{},function(){'use strict';
const COMMIT='1ea34ff48f87168174e12956e200b1d908b1c5ff';
const SOURCE=Object.freeze({repo:'Wan-Video/Wan2.2',commit:COMMIT,path:'wan/modules/model.py',config:'wan/configs/wan_i2v_A14B.py',source:'https://github.com/Wan-Video/Wan2.2/blob/'+COMMIT+'/wan/modules/model.py',configUrl:'https://github.com/Wan-Video/Wan2.2/blob/'+COMMIT+'/wan/configs/wan_i2v_A14B.py',dim:5120,ffnDim:13824,layers:40,heads:40,eps:1e-6});
const check=(ok,msg)=>{if(!ok)throw Error(msg);};
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const mat=(T,C,values)=>{check(values.length===T*C&&values.every(finite),'invalid matrix');return{shape:[T,C],values:values.slice()};};
const add=(a,b)=>{check(a.length===b.length,'add shape mismatch');return a.map((v,i)=>v+b[i]);};
function linear(input,weights,bias,T,C,H){const y=[];for(let t=0;t<T;t++)for(let h=0;h<H;h++){let sum=bias[h];for(let k=0;k<C;k++)sum+=input[t*C+k]*weights[h*C+k];y.push(sum);}return y;}
const geluTanh=x=>0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*x*x*x)));
function norm2(x,T,C,eps=1e-6){const out=[];for(let t=0;t<T;t++){let mean=0;for(let c=0;c<C;c++)mean+=x[t*C+c];mean/=C;let variance=0;for(let c=0;c<C;c++)variance+=(x[t*C+c]-mean)**2;variance/=C;for(let c=0;c<C;c++)out.push((x[t*C+c]-mean)/Math.sqrt(variance+eps));}return out;}
function random(seed){let s=seed>>>0;return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return(s>>>0)/4294967296;};}
function makeConfig({tokens=3,dim=4,hidden=8,seed=17,gate=1,shift=1,scale=1}={}){
 check(Number.isInteger(tokens)&&tokens>=1&&tokens<=8,'tokens 1..8');
 check(Number.isInteger(dim)&&dim>=2&&dim<=12,'demo dim 2..12');
 check(Number.isInteger(hidden)&&hidden>=2&&hidden<=24,'demo hidden 2..24');
 check(Number.isInteger(seed)&&seed>0&&seed<=2147483647,'invalid seed');
 check([gate,shift,scale].every(x=>finite(x)&&x>=-2&&x<=2),'invalid modulation coefficient');
 const r=random(seed),sample=a=>Array.from({length:a},()=> (r()-.5)),T=tokens,C=dim,H=hidden;
 return {tokens:T,dim:C,hidden:H,seed,gate,shift,scale,
   afterSelf:sample(T*C).map(x=>x*1.8),crossDelta:sample(T*C).map(x=>x*.6),
   e0:sample(T*6*C).map(x=>x*.3),modulation:sample(6*C).map(x=>x*.2),
   w1:sample(H*C).map(x=>x*1.1),b1:sample(H).map(x=>x*.15),
   w2:sample(C*H).map(x=>x*.6),b2:sample(C).map(x=>x*.12)};
}
const STEPS=Object.freeze([
{id:'condition',name:'时间条件相加 / chunk(6)',lines:'237–240, 465–469',shape:'[T,6,C] → 6 × [T,C]',code:'e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)',why:'time_projection 在块外生成六组条件；块内 modulation 相加后切分。FFN 读取 e3/e4/e5：Shift/Scale/Gate。',error:'不是直接对原始 t 调 FFN；e3、e4、e5 的索引不能随意重排。'},
{id:'cross',name:'Cross-Attention 残差写回',lines:'249–251',shape:'[T,C] + [T,C] → [T,C]',code:'x = x + self.cross_attn(self.norm3(x), context, context_lens)',why:'FFN 的 x 必须是 Cross-Attention 残差写回后的张量。这里只由外部提供 cross_delta，不假装执行 cross-attention。',error:'把 FFN 放在 Cross-Attention 之前会更改函数。'},
{id:'norm',name:'norm2 / LayerNorm',lines:'211, 252–253',shape:'[T,C] → [T,C]',code:'self.norm2(x).float()',why:'对每个 token 沿 C 计算均值和方差；这里不是 RMSNorm，也没有 learned affine 参数。',error:'错误地按 token 轴归一化或者用 RMSNorm 代替。'},
{id:'affine',name:'FFN 输入的条件 Scale + Shift',lines:'252–253',shape:'[T,C] → [T,C]',code:'self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)',why:'先把归一化结果乘 1+e4，再加 e3。条件依赖 token 和 channel，影响第一层 Linear 的输入。',error:'如果先加 e3 再乘 scale，函数不再相同。'},
{id:'up',name:'Linear1：C → H',lines:'212–214',shape:'[T,C] × [H,C]ᵀ → [T,H]',code:'nn.Linear(dim, ffn_dim)',why:'每个 hidden channel 沿所有 C 输入通道积累点积；PyTorch 参数 W1 实际存储为 [H,C]，带 bias。',error:'不能把 weight 存储方向和数学乘法方向混同。'},
{id:'gelu',name:'逐坐标 GELU(tanh)',lines:'212–214',shape:'[T,H] → [T,H]',code:"nn.GELU(approximate='tanh')",why:'Wan2.2 官方 I2V 这一 FFN 只有单上投影 + GELU(tanh)，不具有 SwiGLU 的两条独立 gate/up 线性分支。',error:'GELU 不会天然使 GEMM 稀疏，也不等价于 exact-erf GELU。'},
{id:'down',name:'Linear2：H → C',lines:'212–214',shape:'[T,H] × [C,H]ᵀ → [T,C]',code:'nn.Linear(ffn_dim, dim)',why:'每个输出通道聚合全部 hidden 的结果；丢弃 hidden 坐标会改变多条输出通道。',error:'hidden-channel 筛选不等于 output-channel 筛选。'},
{id:'gate',name:'e5 输出门控',lines:'254–255',shape:'[T,C] ⊙ [T,C] → [T,C]',code:'y * e[5].squeeze(2)',why:'e5 在下投影之后，对最终 FFN delta 逐坐标缩放，而不是对 hidden activation 做硬二值 mask。',error:'把 e5 提前移到 hidden 会改变结果。'},
{id:'residual',name:'最终残差 x + delta',lines:'254–259',shape:'[T,C] + [T,C] → [T,C]',code:'x = x + y * e[5].squeeze(2)',why:'最终输出保留 Cross-Attention 后 x，叠加经门控的 FFN 改变量；当 e5 为 0，输出严格退化为入口 x。',error:'只展示 FFN(y) 而丢失 residual 会严重误导。'}
]);
function trace(options={}){
 const c=makeConfig(options),{tokens:T,dim:C,hidden:H}=c;
 const full=Array.from({length:T*6*C},(_,i)=>c.e0[i]+c.modulation[i%(6*C)]);
 const e=Array.from({length:6},(_,g)=>Array.from({length:T*C},(_,i)=>full[Math.floor(i/C)*6*C+g*C+i%C]));
 const x=add(c.afterSelf,c.crossDelta),n=norm2(x,T,C,SOURCE.eps);
 const a=n.map((v,i)=>v*(1+c.scale*e[4][i])+c.shift*e[3][i]);
 const up=linear(a,c.w1,c.b1,T,C,H),act=up.map(geluTanh);
 const down=linear(act,c.w2,c.b2,T,H,C),delta=down.map((v,i)=>v*c.gate*e[5][i]),out=add(x,delta);
 const states=[
  {main:mat(T,6*C,full),aux:{shift:mat(T,C,e[3]),scale:mat(T,C,e[4]),gate:mat(T,C,e[5])}},
  {main:mat(T,C,x),aux:{afterSelf:mat(T,C,c.afterSelf),crossDelta:mat(T,C,c.crossDelta)}},
  {main:mat(T,C,n),aux:{crossInput:mat(T,C,x)}},
  {main:mat(T,C,a),aux:{normalized:mat(T,C,n),scale:mat(T,C,e[4]),shift:mat(T,C,e[3])}},
  {main:mat(T,H,up),aux:{input:mat(T,C,a),weights:mat(H,C,c.w1),bias:mat(1,H,c.b1)}},
  {main:mat(T,H,act),aux:{preActivation:mat(T,H,up)}},
  {main:mat(T,C,down),aux:{input:mat(T,H,act),weights:mat(C,H,c.w2),bias:mat(1,C,c.b2)}},
  {main:mat(T,C,delta),aux:{ffn:mat(T,C,down),gate:mat(T,C,e[5])}},
  {main:mat(T,C,out),aux:{crossInput:mat(T,C,x),delta:mat(T,C,delta)}}
 ];
 return {source:SOURCE,config:c,e,steps:STEPS.map((info,i)=>({...info,...states[i]})),final:mat(T,C,out),
    flopsDemo:4*T*C*H,real:{dim:SOURCE.dim,hidden:SOURCE.ffnDim,layers:SOURCE.layers,params:2*SOURCE.dim*SOURCE.ffnDim+SOURCE.dim+SOURCE.ffnDim,flopsPerToken:4*SOURCE.dim*SOURCE.ffnDim},
    evidence:'source-pinned-structure_plus_synthetic_float64_not_original_wan_weights'};
}
function explainDot(data,step,token,channel){
 const info=data.steps[step];check(info&&Number.isInteger(token)&&token>=0&&token<info.main.shape[0],'invalid token');
 check(Number.isInteger(channel)&&channel>=0&&channel<info.main.shape[1],'invalid channel');
 const val=info.main.values[token*info.main.shape[1]+channel];
 if(step===4||step===6){
  const input=info.aux.input,weights=info.aux.weights,bias=info.aux.bias,N=input.shape[1];
  const terms=Array.from({length:N},(_,k)=>({k,x:input.values[token*N+k],w:weights.values[channel*N+k],product:input.values[token*N+k]*weights.values[channel*N+k]}));
  return {step,token,channel,value:val,bias:bias.values[channel],terms,reconstructed:bias.values[channel]+terms.reduce((sum,x)=>sum+x.product,0),formula:'bias[h] + Σ_k input[token,k] × W[h,k]'};
 }
 if(step===5){const u=info.aux.preActivation.values[token*info.main.shape[1]+channel];return{step,token,channel,value:val,input:u,formula:'0.5u × (1+tanh(√(2/π)(u+0.044715u³)))'};}
 if(step===8){const C=info.main.shape[1],base=info.aux.crossInput.values[token*C+channel],delta=info.aux.delta.values[token*C+channel];return{step,token,channel,value:val,base,delta,reconstructed:base+delta,formula:'x_after_cross[t,c]+delta[t,c]'};}
 return{step,token,channel,value:val,formula:info.code};
}
function hiddenChannelEffect(data,token,channel){
 check(data&&data.steps&&data.steps.length===9,'invalid FFN trace');
 const {tokens:T,dim:C,hidden:H,gate}=data.config;
 check(Number.isInteger(token)&&token>=0&&token<T,'invalid token');
 check(Number.isInteger(channel)&&channel>=0&&channel<H,'invalid hidden channel');
 const activation=data.steps[5].main.values[token*H+channel],weight=data.config.w2;
 const e5=data.e[5],dense=data.steps[8].main.values.slice(token*C,(token+1)*C);
 const effects=Array.from({length:C},(_,c)=>{
  const raw=activation*weight[c*H+channel],gated=raw*gate*e5[token*C+c];
  return {outputChannel:c,activation,weight:weight[c*H+channel],rawDelta:raw,gatedDelta:gated,
    dense:dense[c],ablated:dense[c]-gated};
 });
 return {token,hiddenChannel:channel,activation,effects,
  deltaL2:Math.hypot(...effects.map(e=>e.gatedDelta)),
  denseL2:Math.hypot(...dense),
  relativeDeltaL2:Math.hypot(...effects.map(e=>e.gatedDelta))/Math.max(Math.hypot(...dense),1e-12),
  exactScope:'zero_one_post_GELU_hidden_activation_before_W2_no_retraining'};
}
return Object.freeze({SOURCE,STEPS,makeConfig,linear,geluTanh,norm2,trace,explainDot,hiddenChannelEffect});
});