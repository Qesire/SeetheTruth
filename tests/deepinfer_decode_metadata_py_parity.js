'use strict';
const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const D=require('../deepinfer_decode_metadata.js');
const PY=[
'import json,sys,torch',
'case=json.loads(sys.argv[1]); cfg=case["input"]; ss=cfg["sequences"]',
'b=cfg["blockSize"]; ratio=cfg["blockTableCompressRatio"]; per=b//ratio',
'lengths=[]; offset=[0]; blocks=[]; accum=[]; prefix=0',
'for s in ss:',
'  max_k=s["seqlenK"]-s["seqlenQ"]+cfg["maxSeqlenQ"]',
'  count=1 if s["isPlaceholder"] else ((max_k//ratio+per-1)//per)',
'  lengths.append(count)',
'  blocks.extend(s["blockTable"][:count]); offset.append(len(blocks))',
'  prefix+=s["seqlenQ"]; accum.append(prefix)',
'result={"n":lengths,"off":offset,"blocks":blocks,"cu":torch.tensor(accum,dtype=torch.int32).tolist()}',
'if "states" in case:',
'  sampled=[]; updated=[]',
'  for s,st,q,r in zip(ss,case["states"],case["corrected"]["seqlensQ"],case["corrected"]["remap"]):',
'    d={"seqId":s["seqId"],"q":q,"k":q+st["cachedTokens"],"refreshed":r>=0}',
'    if r>=0:',
'      m=st["mtp"]; slot=m["infoSlot"]',
'      d["read"]=slot if m["hasDraftInfo"] else 0',
'      d["write"]=slot if slot!=0 else s.get("mtpInfoWriteSlot",0)',
'      d["last"]=m["lastOutputToken"]',
'      g=m.get("goldenTokens"); at=st.get("outputLen",0)',
'      d["golden"]=(None if not g or at>=len(g) else [g[j] if j<len(g) else -1 for j in range(at,at+2*case["nextn"]+1)])',
'    else:',
'      d["read"]=s.get("mtpInfoReadSlot",0);d["write"]=s.get("mtpInfoWriteSlot",0);d["golden"]=s.get("goldenToken")',
'    sampled.append(d)',
'  result["sampled"]=sampled',
'print(json.dumps(result))'
].join('\n');
function ref(caseData){
 return JSON.parse(execFileSync('python3',['-c',PY,JSON.stringify(caseData)],{
  encoding:'utf8',timeout:120000,maxBuffer:2*1024*1024}).trim().split('\n').at(-1));
}
const seq=(seqId,seqlenQ,seqlenK,blockTable,stateId=1,isPlaceholder=false)=>
 ({seqId,seqlenQ,seqlenK,blockTable,stateId,isPlaceholder});
const cases=[
 {blockSize:4,maxNumBlocks:8,maxSeqlenQ:3,blockTableCompressRatio:2,needDefaultSeqlens:true,
  sequences:[seq('A',1,13,[5,6,7,8]),seq('B',1,5,[2,3])]},
 {blockSize:8,maxNumBlocks:8,maxSeqlenQ:3,blockTableCompressRatio:4,needDefaultSeqlens:false,
  sequences:[seq('X',2,12,[1,2,3,4]),seq('PAD',1,1,[7],2,true)]},
 {blockSize:2,maxNumBlocks:4,maxSeqlenQ:1,blockTableCompressRatio:1,needDefaultSeqlens:true,
  sequences:[seq('U',1,1,[])]},
 {blockSize:4,maxNumBlocks:8,maxSeqlenQ:2,blockTableCompressRatio:2,needDefaultSeqlens:true,
  sequences:[]}
];
for(const input of cases){
 const expected=ref({input}),actual=D.prepare(input);
 assert.deepEqual(actual.numBlocks,expected.n);
 assert.deepEqual(actual.blockTableOffsets,expected.off);
 assert.deepEqual(actual.blockTables,expected.blocks);
 assert.deepEqual(actual.cuSeqlensQ,expected.cu);
}
const input=cases[0],states=[
 {seqId:'A',cachedTokens:12,outputLen:1,soStatus:2,soMatcher:'demo',
  mtp:{infoSlot:5,hasDraftInfo:true,lastOutputToken:88,goldenTokens:[10,11,12,13,14,15]}},
 {seqId:'B',cachedTokens:4,outputLen:0,mtp:{infoSlot:0,hasDraftInfo:false,lastOutputToken:-1,goldenTokens:null}}
], corrected={remap:[1,-1],seqlensQ:[2,1]};
const py=ref({input,states,corrected,nextn:2});
const js=D.updateForSampling(D.prepare(input),corrected,states,{nextn:2});
const mapped=js.metadata.map(m=>({seqId:m.seqId,q:m.seqlenQ,k:m.seqlenK,
 refreshed:m.refreshed,read:m.mtpInfoReadSlot,write:m.mtpInfoWriteSlot,
 golden:m.goldenToken,last:m.lastOutputToken}));
py.sampled.forEach((r,i)=>{
 assert.equal(mapped[i].seqId,r.seqId);
 assert.equal(mapped[i].q,r.q);assert.equal(mapped[i].k,r.k);
 assert.equal(mapped[i].refreshed,r.refreshed);
 assert.equal(mapped[i].read,r.read);assert.equal(mapped[i].write,r.write);
 assert.deepEqual(mapped[i].golden,r.golden);
 if(r.last!==undefined)assert.equal(mapped[i].last,r.last);
});
console.log('DEEPINFER_DECODE_TORCH_PARITY_PASS 4 independent packing cases and source MTP state update');
