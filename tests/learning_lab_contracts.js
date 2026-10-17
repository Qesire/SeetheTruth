'use strict';
const assert=require('node:assert/strict');
const L=require('../learning_lab.js');
function approx(a,b,tol=1e-10){assert.ok(Math.abs(a-b)<=tol,String(a)+' != '+String(b))}
const clone=x=>JSON.parse(JSON.stringify(x));
for(const key of ['ffn','residual','attention','loss','blank']){
 const graph=L.preset(key),evaluated=L.evaluate(graph);
 assert.equal(evaluated.steps.length,graph.nodes.length,key);
 assert.ok(evaluated.steps.every(s=>s.values.every(row=>row.every(Number.isFinite))),key);
 assert.equal(evaluated.evidence,'deterministic_browser_teaching_simulation_not_model_execution');
 assert.deepEqual(L.evaluate(graph),evaluated);
}
{
 const r=L.evaluate(L.preset('ffn')).steps;
 assert.deepEqual(r.map(s=>s.kind),['input','linear','silu','linear']);
 assert.deepEqual(r.map(s=>s.shape),[[3,4],[3,8],[3,8],[3,4]]);
 assert.equal(r[1].flops,3*4*8*2);
 assert.equal(r[3].flops,3*8*4*2);
 const x=r[1].values[0][0];
 approx(r[2].values[0][0],x/(1+Math.exp(-x)));
}
{
 const r=L.evaluate(L.preset('residual')).steps;
 assert.deepEqual(r[r.length-1].shape,[3,4]);
 for(let i=0;i<3;i++)for(let j=0;j<4;j++)
   approx(r[3].values[i][j],r[0].values[i][j]+r[2].values[i][j]);
}
{
 const r=L.evaluate(L.preset('attention')).steps;
 const att=r.find(x=>x.kind==='attention'),P=att.internals.probabilities.values;
 assert.deepEqual(att.shape,[3,4]);
 assert.deepEqual(att.internals.scores.shape,[3,3]);
 for(const row of P)approx(row.reduce((a,b)=>a+b,0),1);
 const values=r.find(x=>x.id==='v').values;
 for(let i=0;i<3;i++)for(let j=0;j<4;j++)
   approx(att.values[i][j],P[i].reduce((s,w,k)=>s+w*values[k][j],0));
}
{
 const r=L.evaluate(L.preset('loss')).steps,mse=r[r.length-1];
 assert.deepEqual(mse.shape,[1,1]);assert.ok(mse.values[0][0]>=0);
 const pred=r.find(x=>x.id==='linear').values,target=r.find(x=>x.id==='target').values;
 const flatTarget=target.flat();
 approx(mse.values[0][0],pred.flat().reduce((s,v,i)=>s+(v-flatTarget[i])**2,0)/12);
}
{
 const n=(id,kind,x,y,params)=>({id,kind,x,y,params:params||{}});
 const g={schema:L.SCHEMA,title:'non-square',nodes:[
   n('a','input',0,0,{rows:2,width:3,seed:1}),
   n('b','input',0,130,{rows:3,width:5,seed:2}),
   n('mul','matmul',300,30,{})],edges:[
    {from:'a',to:'mul',port:'a'},{from:'b',to:'mul',port:'b'}]};
 assert.deepEqual(L.evaluate(g).steps[2].shape,[2,5]);
 assert.equal(L.evaluate(g).steps[2].flops,2*2*3*5);
 g.nodes[1].params.rows=2;
 assert.throws(()=>L.evaluate(g),/内维不匹配/);
}
{
 const g=L.preset('attention'),q=g.nodes.find(n=>n.id==='q');
 q.params.out=5;assert.throws(()=>L.evaluate(g),/Q\/K 的特征维度/);
 q.params.out=4;
 g.nodes.push({id:'alternate',kind:'input',x:20,y:470,params:{rows:2,width:4,seed:2}});
 g.edges=g.edges.filter(e=>!(e.from==='x'&&e.to==='v'));
 g.edges.push({from:'alternate',to:'v',port:'x'});
 assert.throws(()=>L.evaluate(g),/K\/V 的 token 数/);
}
{
 const g=L.preset('ffn');g.edges.pop();
 assert.throws(()=>L.evaluate(g),/还没有连接/);
}
{
 const g=L.preset('residual');
 assert.throws(()=>L.connect(g,'act','fc','x'),/循环计算依赖/);
}
{
 const g=L.preset('ffn');
 assert.throws(()=>L.connect(g,'x','up','invalid'),/端口无效/);
 assert.throws(()=>L.connect(g,'up','up','x'),/端点无效/);
 const duplicate=clone(g);duplicate.edges.push({...duplicate.edges[0]});
 assert.throws(()=>L.validateGraph(duplicate),/不可连接两次/);
}
{
 const g=L.preset('ffn');
 g.nodes[0].params.width=17;assert.throws(()=>L.evaluate(g),/1–16/);
 g.nodes[0].params.width=4;g.nodes[1].params.out=Infinity;
 assert.throws(()=>L.evaluate(g),/1–16/);
}
{
 const g=L.preset('ffn');
 g.nodes.push({id:'__proto__',kind:'input',x:0,y:0,params:{rows:1,width:2}});
 assert.doesNotThrow(()=>L.validateGraph(g));
 const evil=clone(g);evil.nodes[0].kind='<img src=x onerror=alert(1)>';
 assert.throws(()=>L.validateGraph(evil),/未知积木类型/);
}
{
 const p=L.preset('ffn'),hash=JSON.stringify(p);
 L.evaluate(p);assert.equal(JSON.stringify(p),hash);
}
console.log('Learning Lab shape, math, attention, residual, import & cycle contracts PASS');
