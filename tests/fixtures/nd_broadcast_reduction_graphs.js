'use strict';
/* Exact rank-4 broadcasting, shared-channel gain, and keepdim reduction.
 * This is a separately specified graph, not a new numerical operator.
 */
const L=require('../../learning_lab.js');
const ND=require('../../nd_tensor.js');
const n=(id,kind,params={})=>({id,kind,x:0,y:0,params});
const e=(from,to,port)=>({from,to,port});
function parameter(shape,seed,offset=0){
 const values=ND.build(shape,idx=>offset+ND.at(ND.seedTensor(shape,seed),idx)*0.5);
 return {values,optimizer:{t:0,m:ND.build(shape,()=>0),v:ND.build(shape,()=>0)}};
}
function make({id,shape,targetSeed,reduceAxis=2}){
 const channel=shape[3],head=shape[1],batch=shape[0];
 const graph={
  schema:L.SCHEMA,title:id,nodes:[
   n('x','nd_parameter',parameter(shape,5,0.08)),
   n('bias','nd_parameter',parameter([1,1,1,channel],9,-.11)),
   n('gain','nd_parameter',parameter([1,1,1,channel],13,.75)),
   n('biased','nd_broadcast_add'),
   n('gated','nd_broadcast_mul'),
   n('summed','nd_reduce_sum',{axis:reduceAxis}),
   n('target','nd_input',{b:batch,h:head,t:1,d:channel,seed:targetSeed}),
   n('loss','nd_mse')
  ],edges:[
   e('x','biased','a'),e('bias','biased','b'),
   e('biased','gated','a'),e('gain','gated','b'),
   e('gated','summed','x'),
   e('summed','loss','prediction'),e('target','loss','target')
  ]
 };
 return L.validateGraph(graph);
}
const cases=[
 {id:'nd_token_reduce_head_broadcast',shape:[1,2,3,2],targetSeed:22,reduceAxis:2,
  tags:['4d','broadcast_head_token','reduce_token','gain_gradient_aggregation']},
 {id:'nd_batch_reduce_channel_broadcast',shape:[2,1,2,3],targetSeed:29,reduceAxis:2,
  tags:['4d','batch_broadcast','reduce_token','channel_gradient_aggregation']}
];
module.exports={cases,create:id=>{
 const c=cases.find(x=>x.id===id);
 if(!c)throw Error('unknown ND fixture '+id);
 return make(c);
}};
