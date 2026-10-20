/* Exact bounded reverse rules for a composed GRPO/PPO teaching DAG.
 * This is a deliberately scoped reverse-mode interpreter, not general autograd.
 * Graph edges/parameters are read from the live editor: rewiring affects gradients.
 */
(function(root,factory){const api=factory(root);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthGRPOBackward=api;})(typeof window!=="undefined"?window:{},function(root){"use strict";
const L=(typeof module==="object"&&module.exports)?require("./learning_lab.js"):root.SeetheTruthLearning;
const zeros=v=>v.map(row=>row.map(()=>0));
const shape=v=>[v.length,v[0].length];
const finite=v=>Array.isArray(v)&&v.length>0&&v.length<=8&&Array.isArray(v[0])&&v[0].length>0&&v[0].length<=16&&v.every(r=>Array.isArray(r)&&r.length===v[0].length&&r.every(Number.isFinite));
function clipDerivative(r,a,eps){
 if(!(r>0&&Number.isFinite(r)&&Number.isFinite(a)))throw Error("nonfinite policy ratio or advantage");
 const clipped=Math.max(1-eps,Math.min(1+eps,r));
 const raw=r*a,guarded=clipped*a;
 // At nondifferentiable ties choose the raw branch. Tests avoid these boundaries.
 const rawChosen=raw<=guarded;
 return {value:Math.min(raw,guarded),dr:rawChosen||r>=1-eps&&r<=1+eps?a:0,
   da:rawChosen?r:clipped,clipped:!rawChosen&&(r<1-eps||r>1+eps)};
}
function backward(raw,forward,opts={}){
 const graph=L.validateGraph(raw),nodes=L.order(graph);
 if(!forward||forward.mode==="concept"||!Array.isArray(forward.steps)||forward.steps.length!==nodes.length||forward.steps.some(s=>!finite(s.values)))throw Error("GRPO backward requires a completed numeric forward trace");
 const stepMap=new Map(forward.steps.map(s=>[s.id,s])),nodeMap=new Map(nodes.map(n=>[n.id,n]));
 if(stepMap.size!==nodes.length||nodes.some(n=>stepMap.get(n.id)?.kind!==n.kind))throw Error("trace and graph node identities differ; rerun forward");
 const outputs=Object.fromEntries(forward.steps.map(s=>[s.id,s.values]));
 const edges=new Map();
 graph.edges.forEach(e=>edges.set(e.to+":"+e.port,e.from));
 const source=(id,port)=>{const n=edges.get(id+":"+port);if(!n||!Object.hasOwn(outputs,n))throw Error("missing connected port "+id+"."+port);return n;};
 const value=(id,port)=>outputs[source(id,port)];
 const targetId=opts.targetId||nodes.find(n=>n.kind==="scale"&&n.params.factor===-1&&graph.edges.some(e=>e.to===n.id&&nodeMap.get(e.from)?.kind==="masked_grpo_objective"))?.id;
 if(!targetId||!nodeMap.has(targetId)||outputs[targetId].length!==1||outputs[targetId][0].length!==1)throw Error("select a scalar GRPO optimization loss output");
 const grads=new Map(),used=new Set(),cutEdges=[],local={},visited=[];
 const accumulate=(id,g)=>{
   if(!finite(g))throw Error("nonfinite gradient tensor");
   if(shape(g).some((v,i)=>v!==shape(outputs[id])[i]))throw Error("VJP shape mismatch for "+id);
   if(!grads.has(id))grads.set(id,zeros(outputs[id]));
   const stored=grads.get(id);used.add(id);
   stored.forEach((row,i)=>row.forEach((_,j)=>{stored[i][j]+=g[i][j];}));
 };
 const cut=(id,port,reason)=>cutEdges.push({nodeId:id,port,from:source(id,port),reason});
 accumulate(targetId,[[1]]);
 for(const n of [...nodes].reverse()){
   if(!grads.has(n.id))continue;
   const up=grads.get(n.id),id=n.id,p=n.params,inp=port=>value(id,port);
   visited.push(id);
   if(n.kind==="scale"){
     accumulate(source(id,"x"),up.map(row=>row.map(v=>v*p.factor)));continue;
   }
   if(n.kind==="masked_grpo_objective"){
     const a=inp("surrogate"),k=inp("kl"),m=inp("mask");
     const G=a.length,du=zeros(a),dk=zeros(k);
     for(let g=0;g<G;g++){
       const count=m[g].reduce((z,t)=>z+t,0);
       if(count<=0)throw Error("mask has empty completion");
       for(let t=0;t<a[g].length;t++){const w=up[0][0]*m[g][t]/(G*count);
         du[g][t]=w;dk[g][t]=-p.beta*w;}
     }
     accumulate(source(id,"surrogate"),du);accumulate(source(id,"kl"),dk);
     cut(id,"mask","valid-token selection is fixed rollout metadata");
     local[id]={vjp:"dJ/dSurrogate=mask/(G·validCount); dJ/dKL=−β·mask/(G·validCount)",validLengths:m.map(row=>row.reduce((a,b)=>a+b,0))};
     continue;
   }
   if(n.kind==="ppo_clipped_surrogate"){
     const ratio=inp("ratio"),A=inp("advantage"),dr=zeros(ratio),branch=ratio.map(row=>row.map(()=>0)),nearBoundary=ratio.map(row=>row.map(()=>0));
     ratio.forEach((row,i)=>row.forEach((r,j)=>{
       const d=clipDerivative(r,A[i][j],p.epsilon);dr[i][j]=up[i][j]*d.dr;
       branch[i][j]=d.clipped?1:0;
       nearBoundary[i][j]=Math.abs(r-(1-p.epsilon))<1e-10||Math.abs(r-(1+p.epsilon))<1e-10?1:0;
     }));
     accumulate(source(id,"ratio"),dr);
     cut(id,"advantage","response reward advantage is treated as stop-gradient policy-rollout signal");
     local[id]={vjp:"dSurrogate/dr = A on differentiable selected branch; zero when clipped branch saturates",clipSaturated:{shape:shape(branch),values:branch},nondifferentiableBoundary:{shape:shape(nearBoundary),values:nearBoundary}};
     continue;
   }
   if(n.kind==="policy_importance_ratio"){
     const r=outputs[id];accumulate(source(id,"current"),up.map((row,i)=>row.map((v,j)=>v*r[i][j])));
     cut(id,"old","rollout-old policy is a fixed denominator: no gradient through old logprob");
     local[id]={vjp:"dr/dlogπ_current=r; dlogπ_old is detached"};continue;
   }
   if(n.kind==="sampled_reference_kl"){
     const c=inp("current"),ref=inp("reference");
     accumulate(source(id,"current"),up.map((row,i)=>row.map((v,j)=>v*(1-Math.exp(ref[i][j]-c[i][j])))));
     cut(id,"reference","reference policy is frozen and provides constants only");
     local[id]={vjp:"dKL_sample/dlogπ_current = 1−exp(logπ_ref−logπ_current)"};continue;
   }
   if(n.kind==="response_token_reshape"){
     const input=inp("x"),G=outputs[id].length,T=outputs[id][0].length;
     if(input.length!==G*T||input[0].length!==1)throw Error("reshape adjoint incompatible");
     accumulate(source(id,"x"),Array.from({length:G*T},(_,n)=>[up[Math.floor(n/T)][n%T]]));
     continue;
   }
   if(n.kind==="selected_logprob"){
     const logits=inp("logits"),labels=inp("labels"),dlogits=zeros(logits);
     logits.forEach((row,i)=>{
       const mx=Math.max(...row),w=row.map(v=>Math.exp(v-mx)),sum=w.reduce((a,b)=>a+b,0);
       const label=labels[i][0];if(!Number.isSafeInteger(label)||label<0||label>=row.length)throw Error("invalid selected action label");
       row.forEach((_,j)=>{dlogits[i][j]=up[i][0]*((j===label?1:0)-w[j]/sum);});
     });
     accumulate(source(id,"logits"),dlogits);
     cut(id,"labels","sampled token IDs are discrete action identifiers");
     local[id]={vjp:"dlogπ(y)/dlogit_j = 1[j=y]−softmax_j(logits)"};continue;
   }
   if(n.kind==="input"||n.kind==="token_ids")continue;
   // A disconnected reward branch is not reached because surrogate.advantage is detached.
   throw Error("unsupported differentiable block "+n.kind+"; do not claim full-graph gradients");
 }
 const gradientById={};for(const n of nodes)gradientById[n.id]=grads.get(n.id)||zeros(outputs[n.id]);
 const norm=g=>Math.sqrt(g.flat().reduce((sum,v)=>sum+v*v,0));
 return {targetId,loss:outputs[targetId][0][0],gradients:gradientById,
   pathIds:visited,cutEdges,local,
   summary:{currentLogitsGradL2:Object.entries(gradientById).filter(([id])=>nodeMap.get(id).kind==="input"&&used.has(id)).map(([id,g])=>({id,l2:norm(g)})),activeNodes:used.size,zeroGradients:cutEdges.map(x=>x.from)},
   evidence:"exact bounded teaching VJPs for a GRPO loss graph; NOT parameter optimization or large-model backprop"};
}
return Object.freeze({backward,clipDerivative});
});
