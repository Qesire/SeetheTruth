/* Closed-form local reverse rules for bounded routed Expert FFNs.
 * These are reusable micro-VJPs, not automatic differentiating the entire page graph.
 */
(function(root,factory){const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthExpertVJP=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
function tensor(x,name){if(!Array.isArray(x)||!x.length||x.length>8||!Array.isArray(x[0])||!x[0].length||x[0].length>16||x.some(row=>!Array.isArray(row)||row.length!==x[0].length||row.some(v=>!Number.isFinite(v))))throw Error(name+' must be finite bounded 2D');return x;}
const zeros=(rows,cols)=>Array.from({length:rows},()=>Array(cols).fill(0));
function shapeEqual(a,b){if(a.length!==b.length||a[0].length!==b[0].length)throw Error('gradient shape mismatch');}
function indicesOK(I,E){for(const row of I)for(const id of row)if(!Number.isSafeInteger(id)||id<0||id>=E)throw Error('expert index out of range');}
function expertLinear(selected,indices,weights,upstream){tensor(selected,'selected');tensor(indices,'indices');tensor(weights,'weights');tensor(upstream,'upstream');
const T=indices.length,K=indices[0].length,D=selected[0].length,H=weights[0].length;
if(T*K!==selected.length||weights.length%D||upstream.length!==selected.length||upstream[0].length!==H)throw Error('expert linear VJP shape mismatch');
const E=weights.length/D;indicesOK(indices,E);
const gradSelected=zeros(selected.length,D),gradWeights=zeros(weights.length,H),load=Array(E).fill(0);
for(let n=0;n<selected.length;n++){const e=indices[Math.floor(n/K)][n%K];load[e]++;
for(let d=0;d<D;d++)for(let h=0;h<H;h++){gradSelected[n][d]+=upstream[n][h]*weights[e*D+d][h];gradWeights[e*D+d][h]+=selected[n][d]*upstream[n][h];}}
return {gradSelected,gradWeights,load,contract:'expert contributions to identical parameter IDs are summed'};
}
function dispatch(indices,gradSelected){tensor(indices,'indices');tensor(gradSelected,'gradSelected');const K=indices[0].length,D=gradSelected[0].length;
if(indices.length*K!==gradSelected.length)throw Error('dispatch VJP shape mismatch');
const gradInput=zeros(indices.length,D);for(let n=0;n<gradSelected.length;n++)for(let d=0;d<D;d++)gradInput[Math.floor(n/K)][d]+=gradSelected[n][d];
return {gradInput,contract:'K selections share one source token, so adjoints sum'};}
function weightedReduce(values,gates,upstream){tensor(values,'values');tensor(gates,'gates');tensor(upstream,'upstream');const T=gates.length,K=gates[0].length,D=values[0].length;
if(values.length!==T*K||upstream.length!==T||upstream[0].length!==D)throw Error('routed reduce VJP shape mismatch');
const gradValues=zeros(T*K,D),gradGates=zeros(T,K);
for(let t=0;t<T;t++)for(let k=0;k<K;k++)for(let d=0;d<D;d++){gradValues[t*K+k][d]=gates[t][k]*upstream[t][d];gradGates[t][k]+=upstream[t][d]*values[t*K+k][d];}
return {gradValues,gradGates,contract:'weighted product rule; does not differentiate discrete selected expert indices'};}
function swiglu(gate,up,upstream){tensor(gate,'gate');tensor(up,'up');tensor(upstream,'upstream');shapeEqual(gate,up);shapeEqual(gate,upstream);
const gradGate=zeros(gate.length,gate[0].length),gradUp=zeros(gate.length,gate[0].length);
for(let i=0;i<gate.length;i++)for(let j=0;j<gate[0].length;j++){
const v=gate[i][j],sig=v>=0?1/(1+Math.exp(-v)):Math.exp(v)/(1+Math.exp(v)),silu=v*sig;
gradGate[i][j]=upstream[i][j]*up[i][j]*(sig+v*sig*(1-sig));gradUp[i][j]=upstream[i][j]*silu;}
return {gradGate,gradUp,contract:'SiLU gate derivative and elementwise product VJP'};}
return Object.freeze({expertLinear,dispatch,weightedReduce,swiglu});
});
