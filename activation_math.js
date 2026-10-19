/* Stable scalar activation formulas and analytic VJPs used by both 2D/4D blocks. */
(function(root,factory){const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthActivations=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
const points=[.0950125098376374,.2816035507792589,.4580167776572274,.6178762444026437,.755404408355003,.8656312023878317,.9445750230732326,.9894009349916499];
const weights=[.1894506104550685,.1826034150449236,.1691565193950025,.1495959888165767,.1246289712555339,.09515851168249278,.06225352393864789,.027152459411754095];
function erf(x){
 if(!Number.isFinite(x))throw Error("erf requires finite input");
 if(x===0)return 0;
 const sign=Math.sign(x),a=Math.abs(x);
 if(a>=6)return sign;
 const mid=a/2;let sum=0;
 for(let i=0;i<8;i++){const p=points[i];sum+=weights[i]*
   (Math.exp(-((mid*(1+p))**2))+Math.exp(-((mid*(1-p))**2)));}
 return sign*mid*(2/Math.sqrt(Math.PI))*sum;
}
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
const softplus=x=>Math.max(0,x)+Math.log1p(Math.exp(-Math.abs(x)));
const KINDS=Object.freeze(["gelu_erf","gelu_tanh","relu_squared","mish","leaky_relu","silu"]);
function value(kind,x,params={}){
 if(!KINDS.includes(kind)||!Number.isFinite(x))throw Error("activation requires supported function and finite scalar");
 switch(kind){
  case "gelu_erf":return .5*x*(1+erf(x/Math.SQRT2));
  case "gelu_tanh":{const u=Math.sqrt(2/Math.PI)*(x+.044715*x*x*x);return .5*x*(1+Math.tanh(u));}
  case "relu_squared":return Math.max(0,x)**2;
  case "mish":return x*Math.tanh(softplus(x));
  case "leaky_relu":{const alpha=params.slope??.01;
    if(!Number.isFinite(alpha)||alpha<0||alpha>1)throw Error("Leaky ReLU slope must lie in [0,1]");
    return x>=0?x:alpha*x;}
  case "silu":return x*sigmoid(x);
 }
}
function derivative(kind,x,params={}){
 if(!KINDS.includes(kind)||!Number.isFinite(x))throw Error("activation derivative requires supported finite scalar");
 switch(kind){
  case "gelu_erf":return .5*(1+erf(x/Math.SQRT2))+x*Math.exp(-x*x/2)/Math.sqrt(2*Math.PI);
  case "gelu_tanh":{const c=Math.sqrt(2/Math.PI),u=c*(x+.044715*x*x*x),t=Math.tanh(u);return .5*(1+t)+.5*x*(1-t*t)*c*(1+3*.044715*x*x);}
  case "relu_squared":return x>0?2*x:0;
  case "mish":{const t=Math.tanh(softplus(x));return t+x*sigmoid(x)*(1-t*t);}
  case "leaky_relu":{const alpha=params.slope??.01;
    if(!Number.isFinite(alpha)||alpha<0||alpha>1)throw Error("Leaky ReLU slope must lie in [0,1]");
    return x>0?1:x<0?alpha:1;}
  case "silu":{const s=sigmoid(x);return s+x*s*(1-s);}
 }
}
return Object.freeze({value,derivative,erf,softplus,sigmoid,KINDS});
});
