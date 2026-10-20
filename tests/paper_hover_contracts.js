const assert=require("node:assert/strict");
global.SeetheTruthBlockKnowledge={get(id){return id==="linear"?{title:"Linear",formula:"Y=XW",flow:"沿输入通道缩并",practice:"多投影可融合",trap:"线性并不说明用途",exercise:"对比形状",epistemic:"文献说明",refs:["transformer"]}:null;},sources:{transformer:["Attention Is All You Need","https://arxiv.org/abs/1706.03762"]}};
const H=require("../paper_hover.js");
const result=H.resolve("linear");
assert.equal(result.papers[0].url,"https://arxiv.org/abs/1706.03762");
assert.equal(result.flow,"沿输入通道缩并");
assert.equal(H.resolve("unknown"),null);
console.log("paper hover knowledge wiring PASS");