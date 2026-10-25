/* Deterministic left-to-right layered layout for user-composable DAGs.
 * Changes positions only; never rewrites nodes, ports, parameters, or edges.
 * Large graphs scroll horizontally instead of being shrunk to unreadability.
 */
(function(root,factory){const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthGraphLayout=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
function layout(raw,options={}){
 if(!raw||!Array.isArray(raw.nodes)||!Array.isArray(raw.edges)||!raw.nodes.length||raw.nodes.length>128)throw Error("invalid graph layout size");
 const rowGap=Number.isFinite(options.rowGap)?options.rowGap:218;
 const colGap=Number.isFinite(options.colGap)?options.colGap:286;
 const perCol=Number.isInteger(options.perCol)?options.perCol:5;
 if(rowGap<160||rowGap>350||colGap<230||colGap>450||perCol<1||perCol>8)throw Error("invalid layout bounds");
 const ids=new Set(),byId=new Map(),incoming=new Map(),adj=new Map(),parents=new Map();
 raw.nodes.forEach((n,i)=>{if(!n||typeof n.id!=="string"||ids.has(n.id))throw Error("invalid node ids");ids.add(n.id);byId.set(n.id,{...n,index:i});incoming.set(n.id,0);adj.set(n.id,[]);parents.set(n.id,[]);});
 raw.edges.forEach(e=>{if(!ids.has(e.from)||!ids.has(e.to)||e.from===e.to)throw Error("invalid layout edges");incoming.set(e.to,incoming.get(e.to)+1);adj.get(e.from).push(e.to);parents.get(e.to).push(e.from);});
 const queue=[...raw.nodes.filter(n=>incoming.get(n.id)===0).map(n=>n.id)],order=[],levels=new Map();
 for(let p=0;p<queue.length;p++){const id=queue[p];order.push(id);const lev=Math.max(0,...parents.get(id).map(pred=>(levels.get(pred)??-1)+1));levels.set(id,lev);for(const target of adj.get(id)){incoming.set(target,incoming.get(target)-1);if(incoming.get(target)===0)queue.push(target);}}
 if(order.length!==raw.nodes.length)throw Error("graph layout requires DAG");
 const groups=new Map();for(const id of order){const d=levels.get(id);if(!groups.has(d))groups.set(d,[]);groups.get(d).push(id);}
 const placements=new Map();let nextColumn=0;
 for(const depth of [...groups.keys()].sort((a,b)=>a-b)){
  const candidates=groups.get(depth);const rank=new Map(candidates.map((id,i)=>[id,i]));
  candidates.sort((a,b)=>{
   const score=id=>{const ps=parents.get(id).map(v=>placements.get(v)).filter(Boolean);return ps.length?ps.reduce((sum,x)=>sum+x.column*perCol+x.slot,0)/ps.length:Infinity;};
   const sa=score(a),sb=score(b);return (Number.isFinite(sa)&&Number.isFinite(sb)?sa-sb:(Number.isFinite(sa)?-1:Number.isFinite(sb)?1:0))||rank.get(a)-rank.get(b);
  });
  candidates.forEach((id,i)=>placements.set(id,{column:nextColumn+Math.floor(i/perCol),slot:i%perCol,depth}));
  nextColumn+=Math.ceil(candidates.length/perCol);
 }
 const nodes=raw.nodes.map(n=>{const p=placements.get(n.id);return {...n,x:Math.round(30+p.column*colGap),y:Math.round(30+p.slot*rowGap)};});
 const result={...raw,nodes,edges:raw.edges.map(e=>({...e}))};
 return result;
}
function metrics(graph){const nodes=graph.nodes,maxX=Math.max(...nodes.map(n=>n.x+220)),maxY=Math.max(...nodes.map(n=>n.y+175)),overlap=[];
for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++)if(Math.abs(nodes[i].x-nodes[j].x)<220&&Math.abs(nodes[i].y-nodes[j].y)<175)overlap.push([nodes[i].id,nodes[j].id]);
return {width:maxX,height:maxY,overlap,preservedEdges:graph.edges.length};}
return Object.freeze({layout,metrics});
});
