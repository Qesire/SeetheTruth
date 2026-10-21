(function(){"use strict";
const I=window.SeetheTruthTensorIR,S=window.SeetheTruthSemanticState;
if(!I||!S)throw Error("semantic lab dependencies missing");
const F=(shape,axes,data)=>I.make(shape,axes,data);
const cases={
 stream(){
  const prefix=F([1,2,2],['B','T','D'],[1,2,3,4]);
  const current=F([1,1,2],['B','T','D'],[5,6]);
  const alternative=F([1,1,2],['B','T','D'],[50,60]);
  const initial=S.init(prefix,{axis:'T',origin:100,capacity:5});
  const advanced=S.append(initial,current,{position:102});
  const forked=S.append(S.fork(initial),alternative,{position:102});
  const window=S.read(advanced,101,2);
  const passed=initial.nextPosition===102&&advanced.nextPosition===103&&
   JSON.stringify(advanced.tensor.data)===JSON.stringify([1,2,3,4,5,6])&&
   JSON.stringify(window.tensor.data)===JSON.stringify([3,4,5,6])&&
   JSON.stringify(forked.tensor.data)===JSON.stringify([1,2,3,4,50,60]);
  return {title:"绝对位置、KV 追加与独立请求分叉",
   explanation:"来自 [B,T,D] 的前缀不会因另一条分支继续 Decode 而发生变化；追加位置 102 是显式输入。",
   prediction:"相同前缀可以得到不同的分支后缀，旧状态保持 next=102，且窗口 [101,103) 精确返回两条时间记录。",
   passed,values:{initial:{next:initial.nextPosition,prefix:initial.tensor.data},advanced:{next:advanced.nextPosition,values:advanced.tensor.data},
    fork:{next:forked.nextPosition,values:forked.tensor.data},readWindow:{start:window.logicalStart,end:window.logicalEnd,values:window.tensor.data},
    storagePolicy:advanced.storage}};
 },
 video(){
  const tensor=F([1,3,2,2,2],['B','T','H','W','C'],Array.from({length:24},(_,i)=>i+1));
  const chunks=[0,1,2].map(i=>S.slice(tensor,'T',i,1));
  const joined=S.concat(chunks,'T');
  const passed=joined.data.every((v,i)=>v===tensor.data[i])&&
   joined.axes.join(",")===tensor.axes.join(",")&&chunks[2].data[0]===17;
  return {title:"真正的五维视频时间分块与重组成",
   explanation:"把 THWC 视频张量按 T 轴拆成三个帧块，再按原顺序重组；数据位置逐项复现。",
   prediction:"最终 24 个元素、[B,T,H,W,C] 轴与原张量完全一致。最后一帧从 17 开始。",
   passed,values:{axes:tensor.axes,shape:tensor.shape,firstFrame:chunks[0].data,lastFrame:chunks[2].data,
    joined:joined.data,exactEqual:passed}};
 },
 axis(){
  const x=F([2,3],['T','D'],[1,2,3,4,5,6]),bias=F([3],['D'],[10,20,30]);
  const y=S.binary(x,bias,'add',['T','D']);
  let rejected=false;try{S.binary(x,F([2],['Tk'],[8,9]),'add',['T','D']);}catch(e){rejected=/named-axis/.test(e.message);}
  const passed=rejected&&JSON.stringify(y.data)===JSON.stringify([11,22,33,14,25,36]);
  return {title:"轴名严格对应的广播",
   explanation:"偏置位于特征轴 D，不能悄悄匹配为时间轴 T 或 Key 时间轴 Tk。",
   prediction:"D 偏置广播得到 [11,22,33;14,25,36]，带 Tk 的不相容广播被拒绝。",
   passed,values:{shape:y.shape,axes:y.axes,values:y.data,wrongAxisRejected:rejected}};
 },
 reject(){
  const st=S.init(F([1,2,2],['B','T','D'],[1,2,3,4]),{axis:'T',origin:4,capacity:5});
  let message="not rejected";
  try{S.append(st,F([1,1,2],['B','T','D'],[5,6]),{position:5});}catch(e){message=e.message;}
  const passed=/out-of-order\/stale append position/.test(message);
  return {title:"拒绝过期时间位置写入",
   explanation:"具名形状匹配仍不足以证明状态转换合法；时间索引必须与下一个可用位置完全一致。",
   prediction:"当前位置是 6，试图以位置 5 再写入必须失败，而且原始流状态保持不变。",
   passed,values:{origin:st.origin,nextPosition:st.nextPosition,attemptedPosition:5,error:message,original:st.tensor.data}};
 }
};
const $=x=>document.getElementById(x);
function run(name){
 if(!Object.hasOwn(cases,name))throw Error("unknown example "+name);
 const result=cases[name]();if(!result.passed)throw Error(name+" independent mathematical condition failed");
 $("caseTitle").textContent=result.title;$("explanation").textContent=result.explanation;
 $("prediction").textContent=result.prediction;$("status").textContent="数值合同通过 · "+name;
 $("trace").textContent=JSON.stringify(result.values,null,2);
 for(const b of document.querySelectorAll("[data-scenario]"))
  b.setAttribute("aria-pressed",String(b.dataset.scenario===name));
 return result;
}
for(const btn of document.querySelectorAll("[data-scenario]"))
 btn.addEventListener("click",()=>run(btn.dataset.scenario));
run("stream");window.SeetheTruthStateLab=Object.freeze({run,cases:Object.freeze(Object.keys(cases))});
})();
