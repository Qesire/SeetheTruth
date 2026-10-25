'use strict';
const L=require('../../learning_lab.js');
const n=(id,kind,params={})=>({id,kind,params,x:0,y:0});
const e=(from,to,port)=>({from,to,port});
const param=(values,frozen=false)=>({values,frozen,optimizer:{kind:'adamw',t:0,
 m:values.map(row=>row.map(()=>0)),v:values.map(row=>row.map(()=>0))}});
function create(){
return L.validateGraph({schema:L.SCHEMA,title:'multi-path exact training coordinate oracle',
 nodes:[
 n('x','parameter',param([[.22,-.43],[.61,.74]])),
 n('w','parameter',param([[.32,.19],[-.22,.44]])),
 n('project','matmul'),n('scaled','scale',{factor:.4}),
 n('sum','add'),n('out','square'),
 n('target','parameter',param([[.12,-.1],[.3,.09]],true)),n('loss','mse')
 ],edges:[
 e('x','project','a'),e('w','project','b'),
 e('project','scaled','x'),e('project','sum','a'),e('scaled','sum','b'),
 e('sum','out','x'),e('out','loss','prediction'),e('target','loss','target')
 ]});
}
module.exports={create};
