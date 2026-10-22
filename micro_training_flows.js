/* SeetheTruth bounded training flows: exact tiny numerical tensors, no GPU.
 * Educational reference interpreter; statistics never masquerade as measured traces.
 */
(function(root,factory){const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthMicroTrain=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
const LIMIT=256;
const finite=x=>typeof x==="number"&&Number.isFinite(x);
const clone=a=>a.map(r=>r.slice());
function tensor(a){if(!Array.isArray(a)||!a.length||a.length>16||!Array.isArray(a[0])||!a[0].length||a[0].length>16||a.some(r=>!Array.isArray(r)||r.length!==a[0].length||r.some(x=>!finite(x))))throw Error("expected finite 1..16 × 1..16 matrix");return clone(a);}
function zeros(a){return a.map(r=>r.map(()=>0));}
function add(a,b){if(a.length!==b.length||a[0].length!==b[0].length)throw Error("shape mismatch");return a.map((r,i)=>r.map((v,j)=>v+b[i][j]));}
function scale(a,s){return a.map(r=>r.map(x=>x*s));}
function transpose(a){return a[0].map((_,i)=>a.map(r=>r[i]));}
function mm(a,b){if(a[0].length!==b.length)throw Error("matmul contracted dimension mismatch");return a.map(r=>b[0].map((_,j)=>r.reduce((s,v,k)=>s+v*b[k][j],0)));}
function stats(a){const v=a.flat(),n=v.length,mean=v.reduce((s,x)=>s+x,0)/n;return {shape:[a.length,a[0].length],count:n,mean,rms:Math.sqrt(v.reduce((s,x)=>s+x*x,0)/n),min:Math.min(...v),max:Math.max(...v),zeros:v.filter(x=>x===0).length,l2:Math.hypot(...v)};}
class Tape{
 constructor(){this.nodes=new Map();this.parameters=new Map();this.sequence=0;this.events=[];}
 node(value,parents=[],name="operation",identity=null){const id="n"+(++this.sequence);const item={id,name,value:tensor(value),parents,identity,grad:null};this.nodes.set(id,item);this.events.push({phase:"forward",id,name,stats:stats(item.value),parameter:identity});return id;}
 input(v,name="input"){return this.node(v,[],name);}
 parameter(identity,value){if(typeof identity!=="string"||!identity||identity.length>80)throw Error("invalid parameter id");if(this.parameters.has(identity)){const n=this.nodes.get(this.parameters.get(identity));if(JSON.stringify(n.value)!==JSON.stringify(value))throw Error("shared parameter identity with different values");return n.id;}const id=this.node(value,[],"parameter:"+identity,identity);this.parameters.set(identity,id);return id;}
 value(id){const n=this.nodes.get(id);if(!n)throw Error("unknown node");return n.value;}
 op(name,value,parents){if(this.nodes.size>=LIMIT)throw Error("tape limit exceeded");return this.node(value,parents,name);}
 matmul(a,b){const A=this.value(a),B=this.value(b),Y=mm(A,B);return this.op("matmul",Y,[{id:a,vjp:g=>mm(g,transpose(B))},{id:b,vjp:g=>mm(transpose(A),g)}]);}
 add(a,b){const A=this.value(a),B=this.value(b);return this.op("add",add(A,B),[{id:a,vjp:g=>g},{id:b,vjp:g=>g}]);}
 multiply(a,b){const A=this.value(a),B=this.value(b);if(A.length!==B.length||A[0].length!==B[0].length)throw Error("multiply shape mismatch");return this.op("multiply",A.map((r,i)=>r.map((v,j)=>v*B[i][j])),[{id:a,vjp:g=>g.map((r,i)=>r.map((v,j)=>v*B[i][j]))},{id:b,vjp:g=>g.map((r,i)=>r.map((v,j)=>v*A[i][j]))}]);}
 relu(a){const A=this.value(a);return this.op("relu",A.map(r=>r.map(v=>Math.max(v,0))),[{id:a,vjp:g=>g.map((r,i)=>r.map((v,j)=>A[i][j]>0?v:0))}]);}
 meanSquare(a,target){const A=this.value(a),B=tensor(target);if(A.length!==B.length||A[0].length!==B[0].length)throw Error("loss shape mismatch");const n=A.length*A[0].length;const diff=A.map((r,i)=>r.map((v,j)=>v-B[i][j]));const loss=diff.flat().reduce((s,x)=>s+x*x,0)/n;return this.op("mse",[[loss]],[{id:a,vjp:g=>scale(diff,2*g[0][0]/n)}]);}
 backward(loss,seed=[[1]]){const sink=this.nodes.get(loss);if(!sink||JSON.stringify(stats(seed).shape)!==JSON.stringify(stats(sink.value).shape))throw Error("seed shape mismatch");const reachable=new Set();const visit=id=>{if(reachable.has(id))return;reachable.add(id);for(const p of this.nodes.get(id).parents)visit(p.id);};visit(loss);
 const grads=new Map([[loss,tensor(seed)]]);const ordered=[...this.nodes.values()].reverse();
 for(const n of ordered){if(!reachable.has(n.id)||!grads.has(n.id))continue;const g=grads.get(n.id);n.grad=n.grad?add(n.grad,g):clone(g);this.events.push({phase:"backward",id:n.id,name:n.name,gradient:stats(g),accumulated:stats(n.grad),incomingValues:clone(g),accumulatedValues:clone(n.grad)});
 for(const p of n.parents){const branch=p.vjp(g);grads.set(p.id,grads.has(p.id)?add(grads.get(p.id),branch):branch);this.events.push({phase:"vjp",from:n.id,to:p.id,gradient:stats(branch),incomingValues:clone(g),gradientValues:clone(branch)});}}
 return this.gradients();}
 gradients(){return Object.fromEntries([...this.parameters].map(([k,id])=>[k,this.nodes.get(id).grad?clone(this.nodes.get(id).grad):null]));}
 zeroGrad(){for(const n of this.nodes.values())n.grad=null;this.events.push({phase:"zero_grad"});}
}
function softmax(z){if(!Array.isArray(z)||z.length<2||z.length>16||z.some(x=>!finite(x)))throw Error("invalid logits");const mx=Math.max(...z),e=z.map(x=>Math.exp(x-mx)),d=e.reduce((s,x)=>s+x,0);return e.map(x=>x/d);}
function moe(tokens,expertWeights,routerLogits,{capacity=null}={}){const X=tensor(tokens),R=tensor(routerLogits);if(X.length!==R.length||!Array.isArray(expertWeights)||expertWeights.length<2||expertWeights.length>8||R[0].length!==expertWeights.length)throw Error("invalid MoE routing shapes");
 const W=expertWeights.map(tensor),outDim=W[0][0].length;if(W.some(w=>w.length!==X[0].length||w[0].length!==outDim))throw Error("expert dimension mismatch");
 if(capacity!==null&&(!Number.isSafeInteger(capacity)||capacity<1))throw Error("invalid capacity");
 const counts=W.map(()=>0),routed=[],output=[],gradW=W.map(zeros),probs=[];
 for(let i=0;i<X.length;i++){const p=softmax(R[i]);probs.push(p);const expert=p.indexOf(Math.max(...p));const accepted=capacity===null||counts[expert]<capacity;if(accepted)counts[expert]++;
 const y=accepted?mm([X[i]],W[expert])[0]:W[0][0].map(()=>0);output.push(y);routed.push({token:i,expert,accepted,probability:p[expert]});}
 // Top-1 dispatch is discrete here; router receives no task-loss gradient.
 return {output,routed,counts,probabilities:probs,stats:stats(output),contracts:{topK:1,routerTaskGradient:"not_implemented_discrete_dispatch",overflow:"zero_output",capacity:capacity===null?"unlimited":capacity,expertGradient:"computed_only_by_backward_moe"}};
}
function moeBackward(tokens,expertWeights,routerLogits,target,options={}){const f=moe(tokens,expertWeights,routerLogits,options),X=tensor(tokens),T=tensor(target);if(f.output.length!==T.length||f.output[0].length!==T[0].length)throw Error("target mismatch");const n=T.length*T[0].length,gradExpert=expertWeights.map(w=>zeros(tensor(w))),gradInput=zeros(X);
 let loss=0;f.routed.forEach(({token:i,expert,accepted})=>{for(let j=0;j<T[0].length;j++){const delta=f.output[i][j]-T[i][j];loss+=delta*delta/n;if(!accepted)continue;const dy=2*delta/n;for(let k=0;k<X[0].length;k++){gradExpert[expert][k][j]+=X[i][k]*dy;gradInput[i][k]+=expertWeights[expert][k][j]*dy;}}});
 return {...f,loss,gradExpert,gradInput,gradRouter:null,gradientStats:gradExpert.map(stats)};}
function ppo(logits,oldLogits,action,advantage,epsilon=.2){const p=softmax(logits),q=softmax(oldLogits);if(!Number.isInteger(action)||action<0||action>=p.length||!finite(advantage)||!finite(epsilon)||epsilon<0||epsilon>1)throw Error("invalid PPO inputs");
 const ratio=p[action]/q[action],clipped=Math.max(1-epsilon,Math.min(1+epsilon,ratio)),s1=ratio*advantage,s2=clipped*advantage,loss=-Math.min(s1,s2);
 const active=s1<=s2+1e-12 && !((ratio>1+epsilon&&advantage>0)||(ratio<1-epsilon&&advantage<0));
 const gradLogits=p.map((x,j)=>active?-advantage*ratio*((j===action?1:0)-x):0);
 const entropy=-p.reduce((s,x)=>s+x*Math.log(x),0),kl=q.reduce((s,x,j)=>s+x*Math.log(x/p[j]),0);
 return {probabilities:p,oldProbabilities:q,ratio,clippedRatio:clipped,loss,gradLogits,entropy,klOldNew:kl,active,contracts:{oldPolicy:"fixed_snapshot",advantage:"detached_input",objective:"single_action_clipped_surrogate",valueLoss:"excluded",sampling:"not_performed",referenceKL:"not_included"}};}
return Object.freeze({Tape,stats,moe,moeBackward,ppo,softmax});
});