'use strict';
// Deterministic, different semantics from the 2x2 first certificate:
// rectangular matrix multiplication, mixed-sign ReLU with explicit zeros,
// and a true shared parameter traversed by two independent matmuls.
const L=require('../../learning_lab.js');
const n=(id,kind,params={})=>({id,kind,params,x:0,y:0});
const e=(from,to,port)=>({from,to,port});
const zeros=values=>values.map(row=>row.map(()=>0));
const p=(values,frozen=false)=>({values,frozen,optimizer:{
 kind:'adamw',t:0,m:zeros(values),v:zeros(values)}});
function rectangularRelu(){
 return L.validateGraph({schema:L.SCHEMA,title:'2x3 by 3x2 matmul, mixed-sign relu and fanout',
 nodes:[
  n('x','parameter',p([[.3,-.4,.6],[-.2,.8,.5]])),
  n('w','parameter',p([[.7,-.5],[.4,.9],[-.6,.2]])),
  n('project','matmul'),n('gate','relu'),
  n('scaled','scale',{factor:.35}),n('merge','add'),
  n('target','parameter',p([[.14,-.22],[.31,.4]],true)),
  n('loss','mse')
 ],edges:[
  e('x','project','a'),e('w','project','b'),
  e('project','gate','x'),e('project','scaled','x'),
  e('gate','merge','a'),e('scaled','merge','b'),
  e('merge','loss','prediction'),e('target','loss','target')
 ]});
}
function sharedWeight(){
 return L.validateGraph({schema:L.SCHEMA,title:'two independent branches reuse one real 3x2 parameter',
 nodes:[
  n('x','parameter',p([[.26,-.14,.63],[-.35,.44,.19]])),
  n('z','parameter',p([[-.21,.57,.18],[.48,-.38,.52]])),
  n('shared','parameter',p([[.17,-.6],[.73,.26],[-.33,.41]])),
  n('left','matmul'),n('right','matmul'),
  n('rightScaled','scale',{factor:-.6}),n('combined','add'),
  n('square','square'),n('target','parameter',p([[.09,-.12],[.14,.18]],true)),
  n('loss','mse')
 ],edges:[
  e('x','left','a'),e('shared','left','b'),
  e('z','right','a'),e('shared','right','b'),
  e('right','rightScaled','x'),
  e('left','combined','a'),e('rightScaled','combined','b'),
  e('combined','square','x'),e('square','loss','prediction'),
  e('target','loss','target')
 ]});
}
module.exports={rectangularRelu,sharedWeight};
