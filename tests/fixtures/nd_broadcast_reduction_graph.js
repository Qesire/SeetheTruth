'use strict';
// Source graph for N-D broadcasting and axis-reduction adjoints.
// Four-axis parameter [B,H,T,D] + singleton [1,1,1,D]; keepdim sum on T.
const L=require('../../learning_lab.js'),ND=require('../../nd_tensor.js');
const n=(id,kind,params={})=>({id,kind,params,x:0,y:0});
const e=(from,to,port)=>({from,to,port});
const createParam=(values,frozen=false)=>({
 values,frozen,optimizer:{t:0,m:ND.build(ND.shape(values),()=>0),v:ND.build(ND.shape(values),()=>0)}
});
function broadcastReduce(){
 const source=ND.build([1,2,3,2],([b,h,t,d])=>
  .11+(h+1)*.17+(t-1)*.13+(d?-.22:.06));
 const sharedBias=ND.build([1,1,1,2],([b,h,t,d])=>d?.23:-.14);
 const target=ND.build([1,2,1,2],([b,h,t,d])=>.09+(h+1)*.12+d*.03);
 return L.validateGraph({schema:L.SCHEMA,
  title:'4D singleton broadcast and token-axis keepdim reduction',
  nodes:[
   n('source','nd_parameter',createParam(source)),
   n('bias','nd_parameter',createParam(sharedBias)),
   n('offset','nd_broadcast_add'),
   n('summed','nd_reduce_sum',{axis:2}),
   n('target','nd_parameter',createParam(target,true)),
   n('loss','nd_mse')
  ],
  edges:[
   e('source','offset','a'),e('bias','offset','b'),
   e('offset','summed','x'),e('summed','loss','prediction'),
   e('target','loss','target')
  ]});
}
module.exports={broadcastReduce};
