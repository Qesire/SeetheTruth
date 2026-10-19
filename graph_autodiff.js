/* SeetheTruth bounded graph autodiff: numerical forward from LearningLab,
 * reverse VJP per supported primitive; no model training, no inferred gradients.
 */
(function(root,factory){const api=factory(root);if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthGraphAutodiff=api;})(typeof window!=="undefined"?window:{},function(root){"use strict";
const L=typeof module==="object"&&module.exports?require("./learning_lab.js"):root.SeetheTruthLearning;
const Expert=typeof module==="object"&&module.exports?require("./expert_ffn_vjp.js"):root.SeetheTruthExpertVJP;
const NDVJP=typeof module==="object"&&module.exports?require("./nd_autodiff.js"):root.SeetheTruthNDVJP;
const ND=typeof module==="object"&&module.exports?require("./nd_tensor.js"):root.SeetheTruthND;
const Activations=typeof module==="object"&&module.exports?require("./activation_math.js"):root.SeetheTruthActivations;
const finite=a=>{try{
 const walk=(v,depth)=>{if(depth>4)throw Error("rank");
  if(!Array.isArray(v))return typeof v==="number"&&Number.isFinite(v);
  if(!v.length||v.length>16)return false;
  const nested=Array.isArray(v[0]),length=v[0]?.length;
  return v.every(x=>Array.isArray(x)===nested&&(!nested||x.length===length)&&walk(x,depth+1));
 };
 return Array.isArray(a)&&Array.isArray(a[0])&&walk(a,0);
}catch(e){return false;}};
const z=a=>a.map(row=>Array.isArray(row)?z(row):0),tr=a=>a[0].map((_,j)=>a.map(row=>row[j]));
const shapeDeep=a=>{const sh=[];let v=a;while(Array.isArray(v)){sh.push(v.length);v=v[0];}return sh;};
const mm=(a,b)=>{if(a[0].length!==b.length)throw Error("VJP matrix dimensions mismatch");return a.map(row=>b[0].map((_,j)=>row.reduce((s,v,k)=>s+v*b[k][j],0)));};
const same=(a,b)=>{const x=shapeDeep(a),y=shapeDeep(b);return x.length===y.length&&x.every((v,i)=>v===y[i]);};
const had=(a,b)=>a.map((row,i)=>row.map((v,j)=>v*b[i][j]));
const scale=(a,k)=>a.map(row=>row.map(v=>v*k));
const add=(a,b)=>a.map((row,i)=>Array.isArray(row)?add(row,b[i]):row+b[i]);
const dsilu=x=>{const p=x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));return p+x*p*(1-p);};
const hashWeight=(id,r,c)=>{let h=2166136261;for(const ch of id+":"+r+":"+c)h=Math.imul(h^ch.charCodeAt(0),16777619);return ((h>>>0)%2001-1000)/2600;};
function backward(raw,forward,opts={}){
 const graph=L.validateGraph(raw),nodes=L.order(graph),trace=forward?.steps;
 if(!Array.isArray(trace)||trace.length!==nodes.length||forward.mode==="concept")throw Error("numeric forward trace required");
 const values=new Map(),byId=new Map(nodes.map(n=>[n.id,n])),edges=new Map();
 for(let i=0;i<nodes.length;i++){const step=trace[i],n=nodes[i];if(step.id!==n.id||step.kind!==n.kind||!finite(step.values))throw Error("stale or invalid forward trace");values.set(n.id,step.values);}
 for(const e of graph.edges)edges.set(e.to+":"+e.port,e.from);
 const chosen=opts.targetId||[...nodes].reverse().find(n=>values.get(n.id).length===1&&values.get(n.id)[0].length===1&&(n.kind==="mse"||n.kind==="nd_mse"||n.kind==="mae_loss"||n.kind==="cross_entropy"||n.kind==="masked_grpo_objective"))?.id;
 if(!chosen||!byId.has(chosen)||values.get(chosen).length!==1||values.get(chosen)[0].length!==1)throw Error("choose scalar loss node for graph reverse pass");
 const grads=new Map(),routes=[],cuts=[],local={};
 const src=(id,port)=>{const v=edges.get(id+":"+port);if(!v)throw Error("missing gradient input "+id+"."+port);return v;};
 const inp=(id,port)=>values.get(src(id,port));
 const addGrad=(id,g)=>{if(!finite(g)||!same(values.get(id),g))throw Error("VJP output shape or finite check failed for "+id);grads.set(id,grads.has(id)?add(grads.get(id),g):g.map(r=>r.slice()));};
 const send=(id,port,g)=>addGrad(src(id,port),g);
 const cut=(id,port,reason)=>cuts.push({nodeId:id,port,from:src(id,port),reason});
 addGrad(chosen,[[1]]);
 for(const n of [...nodes].reverse()){
  if(!grads.has(n.id))continue;
  const id=n.id,p=n.params,d=grads.get(id),kind=n.kind;
  routes.push(id);
  if(kind==="nd_parameter")continue;
  if(kind.startsWith("nd_")){
    const input={};
    for(const port of L.defs[kind].ports)input[port]=inp(id,port);
    const deriv=NDVJP.vjp(kind,input,p,d,values.get(id));
    for(const port of L.defs[kind].ports){
      if(deriv.cuts?.includes(port)){cut(id,port,"fixed N-D target or intentionally detached argument");continue;}
      if(!Object.hasOwn(deriv,port))throw Error("missing N-D VJP for "+id+"."+port);
      send(id,port,deriv[port]);
    }
    local[id]={vjp:"N-D exact reverse tensor VJP with explicit broadcasting, grouped-head accumulation and tensor axis semantics"};
    continue;
  }
  if(kind==="input"||kind==="parameter"||kind==="token_ids")continue;
  if(kind==="add"||kind==="subtract"){
    send(id,"a",d);send(id,"b",kind==="add"?d:scale(d,-1));continue;
  }
  if(kind==="mul"){send(id,"a",had(d,inp(id,"b")));send(id,"b",had(d,inp(id,"a")));continue;}
  if(kind==="matmul"){send(id,"a",mm(d,tr(inp(id,"b"))));send(id,"b",mm(tr(inp(id,"a")),d));continue;}
  if(kind==="linear"){
    const x=inp(id,"x"),dx=z(x);
    for(let i=0;i<x.length;i++)for(let j=0;j<x[0].length;j++)for(let o=0;o<d[0].length;o++)dx[i][j]+=d[i][o]*hashWeight(id,o,j);
    send(id,"x",dx);local[id]={vjp:"dL/dX=dL/dY·W (W generated deterministically by node id); no trainable weight tensor in this block"};continue;
  }
  if(kind==="scale"||kind==="bias"||kind==="residual_scale"){
    send(id,"x",scale(d,kind==="bias"?1:kind==="scale"?p.factor:p.factor));continue;
  }
  if(kind==="square"){send(id,"x",had(d,scale(inp(id,"x"),2)));continue;}
  if(kind==="transpose2d"){send(id,"x",tr(d));continue;}
  if(["gelu_erf","gelu_tanh","relu_squared","mish","leaky_relu","elu","hardswish"].includes(kind)){
    const x=inp(id,"x");
    send(id,"x",d.map((row,i)=>row.map((v,j)=>v*Activations.derivative(kind,x[i][j],p))));
    local[id]={vjp:"analytic scalar activation derivative, multiplied by incoming gradient"};continue;
  }
  if(kind==="silu"||kind==="sigmoid"||kind==="tanh"||kind==="relu"){
    const x=inp(id,"x"),y=values.get(id);
    send(id,"x",d.map((row,i)=>row.map((v,j)=>{
      const a=x[i][j],b=y[i][j];
      return v*(kind==="silu"?dsilu(a):kind==="sigmoid"?b*(1-b):kind==="tanh"?1-b*b:a>0?1:0);
    })));continue;
  }
  if(kind==="softmax"||kind==="log_softmax"){
    const y=values.get(id),pvals=kind==="softmax"?y:y.map(row=>row.map(Math.exp));
    send(id,"x",d.map((row,i)=>{
      const weighted=row.reduce((sum,v,j)=>sum+v*(kind==="softmax"?pvals[i][j]:1),0);
      return row.map((v,j)=>kind==="softmax"?pvals[i][j]*(v-weighted):v-pvals[i][j]*weighted);
    }));continue;
  }
  if(kind==="rmsnorm"||kind==="qk_norm"){
    const x=inp(id,"x"),eps=p.epsilon;
    send(id,"x",x.map((row,i)=>{
      const inv=1/Math.sqrt(row.reduce((sum,v)=>sum+v*v,0)/row.length+eps),
      q=row.reduce((sum,v,j)=>sum+v*d[i][j],0)/row.length;
      return row.map((v,j)=>d[i][j]*inv-v*inv*inv*inv*q);
    }));continue;
  }
  if(kind==="layernorm"){
    const x=inp(id,"x"),eps=p.epsilon;
    send(id,"x",x.map((row,i)=>{
      const mean=row.reduce((a,b)=>a+b,0)/row.length,dev=row.map(v=>v-mean),
        inv=1/Math.sqrt(dev.reduce((a,b)=>a+b*b,0)/row.length+eps),
        u=d[i].reduce((a,b)=>a+b,0)/row.length,
        w=d[i].reduce((a,v,j)=>a+v*dev[j],0)/row.length;
      return row.map((_,j)=>inv*(d[i][j]-u)-dev[j]*inv*inv*inv*w);
    }));continue;
  }
  if(kind==="softplus"){
    const x=inp(id,"x");
    send(id,"x",d.map((row,i)=>row.map((v,j)=>v*Activations.sigmoid(x[i][j]))));
    local[id]={vjp:"softplus derivative = sigmoid(x)"};continue;
  }
  if(kind==="geglu"){
    const g=inp(id,"gate"),u=inp(id,"up");
    send(id,"gate",g.map((row,i)=>row.map((v,j)=>d[i][j]*u[i][j]*Activations.derivative("gelu_tanh",v))));
    send(id,"up",g.map((row,i)=>row.map((v,j)=>d[i][j]*Activations.value("gelu_tanh",v))));
    local[id]={vjp:"GEGLU tanh: dGate=dY*Up*GELU'_tanh(Gate), dUp=dY*GELU_tanh(Gate)"};
    continue;
  }
  if(kind==="swiglu"){
    const g=inp(id,"gate"),u=inp(id,"up");
    send(id,"gate",g.map((row,i)=>row.map((v,j)=>d[i][j]*u[i][j]*dsilu(v))));
    send(id,"up",g.map((row,i)=>row.map((v,j)=>d[i][j]*v/(1+Math.exp(-v)))));
    continue;
  }
  if(kind==="expert_linear_bank"){
    const out=Expert.expertLinear(inp(id,"x"),inp(id,"indices"),inp(id,"weights"),d);
    send(id,"x",out.gradSelected);send(id,"weights",out.gradWeights);
    cut(id,"indices","integer route choice is fixed; expert ID is not differentiable");
    local[id]={vjp:"dW[e]=Σ_{routes using e} X_routeᵀ·dY_route; dX_route=dY_route·W[e]ᵀ",expertLoad:{shape:[out.load.length,1],values:out.load.map(v=>[v])}};continue;
  }
  if(kind==="dispatch_routed_tokens"){
    const out=Expert.dispatch(inp(id,"indices"),d);
    send(id,"x",out.gradInput);cut(id,"indices","discrete route ID selects copies, no index derivative");
    local[id]={vjp:"dX[t,:]=Σ_{k=0}^{K−1} dZ[tK+k,:]"};continue;
  }
  if(kind==="routed_weighted_reduce"){
    const out=Expert.weightedReduce(inp(id,"values"),inp(id,"gates"),d);
    send(id,"values",out.gradValues);send(id,"gates",out.gradGates);
    local[id]={vjp:"dV[tK+k,:]=gate[t,k]·dY[t,:]; dGate[t,k]=dot(dY[t,:],V[tK+k,:])"};continue;
  }
  if(kind==="router_gate_weights"){
    const scores=inp(id,"scores"),indices=inp(id,"indices"),ds=z(scores),eps=1e-15;
    const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
    for(let t=0;t<indices.length;t++){
      const sel=indices[t],v=sel.map(idx=>scores[t][idx]);
      const pval=p.mode==="identity"?v:p.mode==="sigmoid"?v.map(sigmoid):
        (()=>{const mx=Math.max(...v),ex=v.map(x=>Math.exp(x-mx)),sm=ex.reduce((a,b)=>a+b,0);return ex.map(x=>x/sm);})();
      const total=pval.reduce((a,b)=>a+b,0);
      if(p.normalize&&total<=eps)throw Error("router gate normalization denominator underflow");
      const dP=p.normalize?d[t].map((v,k)=>(v-d[t].reduce((sum,g,j)=>sum+g*(pval[j]/total),0))/total):d[t];
      const dSelected=p.mode==="identity"?dP:p.mode==="sigmoid"?dP.map((v,k)=>v*pval[k]*(1-pval[k])):
         dP.map((v,k)=>pval[k]*(v-dP.reduce((sum,g,j)=>sum+g*pval[j],0)));
      sel.forEach((expert,k)=>{ds[t][expert]+=dSelected[k];});
    }
    send(id,"scores",ds);cut(id,"indices","Top-k discrete selected IDs are fixed; gradient flows only through selected gate affinities");
    local[id]={vjp:"g=selectedAffinity/sum(selectedAffinity); dAffinity_i=(dG_i−Σ_j dG_j·g_j)/sum; duplicate selected IDs accumulate"};continue;
  }
  if(kind==="attention"){
    const q=inp(id,"q"),k=inp(id,"k"),v=inp(id,"v"),
      scores=mm(q,tr(k)),factor=1/Math.sqrt(q[0].length);
    const probabilities=scores.map(row=>{const shifted=row.map(x=>x*factor),mx=Math.max(...shifted),ex=shifted.map(x=>Math.exp(x-mx)),total=ex.reduce((a,b)=>a+b,0);return ex.map(x=>x/total);});
    const dp=mm(d,tr(v)),ds=dp.map((row,i)=>{
      const weighted=row.reduce((sum,val,j)=>sum+val*probabilities[i][j],0);
      return row.map((val,j)=>probabilities[i][j]*(val-weighted)*factor);
    });
    send(id,"v",mm(tr(probabilities),d));send(id,"q",mm(ds,k));send(id,"k",mm(tr(ds),q));
    local[id]={vjp:"dV=PᵀdY, dP=dYVᵀ, dS=softmax-VJP/√d, dQ=dS K, dK=dSᵀ Q"};continue;
  }
  if(kind==="mse"){
    const a=inp(id,"prediction"),b=inp(id,"target"),size=a.length*a[0].length;
    send(id,"prediction",a.map((row,i)=>row.map((v,j)=>2*(v-b[i][j])*d[0][0]/size)));
    cut(id,"target","target data is fixed during optimization");continue;
  }
  if(kind==="row_mean"||kind==="row_sum"){
    const a=inp(id,"x"),mult=kind==="row_mean"?1/a[0].length:1;
    send(id,"x",a.map((row,i)=>row.map(()=>d[i][0]*mult)));continue;
  }
  if(kind==="concat_cols"){
    const a=inp(id,"a"),cols=a[0].length;
    send(id,"a",d.map(row=>row.slice(0,cols)));send(id,"b",d.map(row=>row.slice(cols)));continue;
  }
  if(kind==="slice_cols"){
    const x=inp(id,"x"),dx=z(x);
    d.forEach((row,i)=>row.forEach((v,j)=>{dx[i][p.start+j]+=v;}));
    send(id,"x",dx);continue;
  }
  if(kind==="repeat_rows"){
    const x=inp(id,"x"),dx=z(x),times=p.times;
    for(let i=0;i<d.length;i++)for(let j=0;j<d[0].length;j++)dx[i%x.length][j]+=d[i][j];
    send(id,"x",dx);continue;
  }
  if(kind==="detach_demo"){cut(id,"x","explicit Detach blocks all adjoints");continue;}
  throw Error("unsupported ancestor "+kind+" at "+id+"; no fabricated gradient");
 }
 const out=Object.fromEntries(nodes.map(n=>[n.id,grads.get(n.id)||z(values.get(n.id))]));
 return {targetId:chosen,loss:values.get(chosen)[0][0],gradients:out,pathIds:routes,cutEdges:cuts,local,
   evidence:"bounded graph VJP with exact upstream values; not automatic GPU differentiation; no parameter update"};
}
return Object.freeze({backward});
});
