/* Mathematical shape/index rules for the free-composition teaching graph.
 * Pure/static, independent of CPU NumPy/PyTorch and model architecture.
 */
(function(root,factory){"use strict";
 const node=typeof module==="object"&&module.exports;
 const api=factory();
 if(node)module.exports=api;
 if(root)root.SeetheTruthShapeIndexRules=api;
})(typeof window!=="undefined"?window:null,function(){"use strict";
const KINDS=Object.freeze(["transpose2d","nd_permute","nd_reshape","nd_flatten",
 "nd_expand4d","nd_broadcast_add","nd_reduce_sum","nd_repeat_kv"].sort());
const supported=new Set(KINDS);
const prod=shape=>shape.reduce((a,b)=>a*b,1);
const valid=shape=>Array.isArray(shape)&&[2,4].includes(shape.length)&&shape.every(x=>Number.isSafeInteger(x)&&x>0);
const linear=(coords,shape)=>coords.reduce((offset,v,i)=>offset*shape[i]+v,0);
const coords=(flat,shape)=>{const result=Array(shape.length);for(let i=shape.length-1;i>=0;i--){result[i]=flat%shape[i];flat=Math.floor(flat/shape[i]);}return result;};
const sample=shape=>shape.map(n=>Math.min(1,n-1));
const shapeText=shape=>"["+shape.join(" × ")+"]";
const record=(port,shape,index)=>Object.freeze({port,shape:Object.freeze([...shape]),index:Object.freeze([...index])});
function analyze(node,inputs,options={}){
 if(!node||typeof node.kind!=="string"||!inputs||typeof inputs!=="object")throw Error("invalid shape rule request");
 const kind=node.kind,p=node.params||{};
 if(!supported.has(kind))return null;
 const get=port=>{const sh=inputs[port];if(!valid(sh))throw Error("missing/invalid shape input "+port+" for "+kind);return sh;};
 const x=kind==="nd_broadcast_add"?null:get("x");
 let outputShape,sourceIndices=[],gradientRule="",rule="";
 if(kind==="transpose2d"){
  if(x.length!==2)throw Error("transpose requires rank two");
  outputShape=[x[1],x[0]];rule="交换两个矩阵轴，元素线性存储顺序可能变化";gradientRule="反向执行同一次转置";
 }else if(kind==="nd_permute"){
  if(x.length!==4)throw Error("permute requires four axes");
  const order=[p.axis0,p.axis1,p.axis2,p.axis3];
  if(new Set(order).size!==4||order.some(v=>!Number.isInteger(v)||v<0||v>3))throw Error("invalid complete axis permutation");
  outputShape=order.map(axis=>x[axis]);rule="输出轴 i 对应输入轴 order[i]；不是 Reshape";gradientRule="逆置换梯度轴";
 }else if(kind==="nd_reshape"){
  outputShape=[p.b,p.h,p.t,p.d];
  if(!valid(outputShape)||prod(x)!==prod(outputShape))throw Error("reshape element count mismatch");
  rule="行主序线性位置保持不变，重新解释维度";gradientRule="将梯度按行主序 Reshape 回输入形状";
 }else if(kind==="nd_flatten"){
  if(x.length!==4)throw Error("flatten requires rank four");
  outputShape=[x[0]*x[1]*x[2],x[3]];rule="合并前三个轴，最后特征轴保持不变";gradientRule="将梯度按行主序恢复原四维形状";
 }else if(kind==="nd_expand4d"){
  if(x.length!==2)throw Error("expand4d requires rank two");
  outputShape=[1,1,...x];rule="添加两个长度为一的轴；没有复制有效元素";gradientRule="删除两个 singleton 梯度轴";
 }else if(kind==="nd_reduce_sum"){
  if(x.length!==4||!Number.isInteger(p.axis)||p.axis<0||p.axis>3)throw Error("reduce_sum requires a valid four-axis reduction");
  outputShape=x.map((n,i)=>i===p.axis?1:n);
  rule="沿轴 "+p.axis+" 求和并保留一个长度为一的轴（keepdim）";
  gradientRule="沿被约简轴复制上游梯度（不是取平均）";
 }else if(kind==="nd_repeat_kv"){
  if(x.length!==4||!Number.isInteger(p.queryHeads)||p.queryHeads<x[1]||p.queryHeads%x[1]!==0)throw Error("query heads must divide repeat ratio");
  outputShape=[x[0],p.queryHeads,x[2],x[3]];
  rule="每 "+(p.queryHeads/x[1])+" 个 Query Head 读取相同的 KV Head";
  gradientRule="将同一 KV Head 对应的全部 Query Head 梯度求和";
 }else if(kind==="nd_broadcast_add"){
  const a=get("a"),b=get("b");
  if(a.length!==4||b.length!==4)throw Error("broadcast add requires rank four");
  outputShape=a.map((v,i)=>{const w=b[i];if(v!==w&&v!==1&&w!==1)throw Error("incompatible broadcast axis "+i);return Math.max(v,w);});
  rule="从末轴对齐，仅长度为一的轴允许复用元素";
  gradientRule="对每个输入沿它被广播的轴汇总梯度";
 }
 const selected=options.outputIndex;
 if(selected!==undefined&&(!Array.isArray(selected)||selected.length!==outputShape.length||
   selected.some((v,i)=>!Number.isSafeInteger(v)||v<0||v>=outputShape[i])))
  throw Error("output index out of bounds for "+kind);
 const outIndex=selected?[...selected]:sample(outputShape);
 if(kind==="nd_broadcast_add"){
  for(const port of ["a","b"]){
   const sh=get(port);
   sourceIndices.push(record(port,sh,outIndex.map((v,i)=>sh[i]===1?0:v)));
  }
 }else if(kind==="nd_reduce_sum"){
  for(let index=0;index<x[p.axis];index++){
   const c=[...outIndex];c[p.axis]=index;sourceIndices.push(record("x",x,c));
  }
 }else{
  let index;
  if(kind==="transpose2d")index=[outIndex[1],outIndex[0]];
  else if(kind==="nd_permute"){
   index=[0,0,0,0];
   [p.axis0,p.axis1,p.axis2,p.axis3].forEach((srcAxis,i)=>index[srcAxis]=outIndex[i]);
  }else if(["nd_reshape","nd_flatten","nd_expand4d"].includes(kind))index=coords(linear(outIndex,outputShape),x);
  else if(kind==="nd_repeat_kv")index=[outIndex[0],Math.floor(outIndex[1]/(p.queryHeads/x[1])),outIndex[2],outIndex[3]];
  sourceIndices.push(record("x",x,index));
 }
 const reverseContributions=sourceIndices.map(source=>{
  const ranges=outputShape.map((size,axis)=>[outIndex[axis],outIndex[axis]]);
  if(kind==="nd_broadcast_add"){
   source.shape.forEach((size,axis)=>{if(size===1&&outputShape[axis]>1)ranges[axis]=[0,outputShape[axis]-1];});
  }else if(kind==="nd_repeat_kv"){
   const factor=p.queryHeads/x[1],head=source.index[1];
   ranges[1]=[head*factor,(head+1)*factor-1];
  }
  const outputCount=ranges.reduce((count,[start,end])=>count*(end-start+1),1);
  return Object.freeze({port:source.port,inputIndex:source.index,
   outputRanges:Object.freeze(ranges.map(r=>Object.freeze(r))),outputCount});
 });
 return Object.freeze({schema:"seethetruth-shape-index-v1",kind,
  inputShapes:Object.freeze(Object.fromEntries(Object.entries(inputs).map(([k,v])=>[k,Object.freeze([...v])]))),
  outputShape:Object.freeze(outputShape),outputIndex:Object.freeze(outIndex),
  sourceIndices:Object.freeze(sourceIndices),reverseContributions:Object.freeze(reverseContributions),rule,gradientRule,
  outputElements:prod(outputShape),indexExplanation:shapeText(outputShape)+" 输出索引 "+outIndex.join(",")+
   " 对应输入 "+sourceIndices.map(s=>s.port+"["+s.index.join(",")+"]").join(" + ")});
}
function emit(node,backend,port){
 if(!node||typeof node.kind!=="string"||typeof port!=="function"||!["numpy","torch"].includes(backend))throw Error("invalid shape lowering");
 const k=node.kind,p=node.params||{},x=()=>port("x"),pt=backend==="torch";
 if(!supported.has(k)||k==="nd_broadcast_add")return null;
 if(k==="transpose2d")return x()+".T";
 if(k==="nd_permute"){
  const axes=[p.axis0,p.axis1,p.axis2,p.axis3];
  return pt?x()+".permute("+axes.join(",")+")":"np.transpose("+x()+","+JSON.stringify(axes)+")";
 }
 if(k==="nd_reshape")return x()+".reshape("+ (pt?[p.b,p.h,p.t,p.d].join(","):JSON.stringify([p.b,p.h,p.t,p.d]))+")";
 if(k==="nd_flatten")return x()+".reshape(-1,"+x()+".shape[-1])";
 if(k==="nd_expand4d")return x()+"[None,None,:,:]";
 if(k==="nd_reduce_sum")return pt?x()+".sum(dim="+p.axis+",keepdim=True)":"np.sum("+x()+",axis="+p.axis+",keepdims=True)";
 if(k==="nd_repeat_kv")return pt?x()+".repeat_interleave("+p.queryHeads+"//"+x()+".shape[1],dim=1)":"np.repeat("+x()+","+p.queryHeads+"//"+x()+".shape[1],axis=1)";
 throw Error("unhandled shape lowering "+k);
}
return Object.freeze({kinds:KINDS,analyze,emit});
});
