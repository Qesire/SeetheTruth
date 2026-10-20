'use strict';
const assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const Z=require('../deepinfer_zos_metadata.js');
const PY=String.raw\`
import sys,json,torch
c=json.loads(sys.argv[1])
prev=c["prev"]; current=c["curr"]; nextn=c["nextn"]; pad=c["pad"]
same=len(current)==len(prev) and all(
    not p.get("isPlaceholder",False) and v["seqId"]==p["seqId"]
    for v,p in zip(current,prev))
mapping={p["seqId"]:i for i,p in enumerate(prev) if not p.get("isPlaceholder",False)}
r=[i if same else mapping.get(v["seqId"],-1) for i,v in enumerate(current)]
if all(x<0 for x in r):
    print(json.dumps(None));sys.exit(0)
q=[v["seqlenQ"] for v in current]+[1]*(pad-len(current))
k=[v["seqlenK"]-(v["seqlenQ"] if nextn>0 else 0) for v in current]+[1]*(pad-len(current))
if c.get("varlen",False):
    mr=torch.tensor(r,dtype=torch.long)
    rates=torch.tensor(c["rates"],dtype=torch.float64)
    gathered=torch.where(mr[:,None]>=0,rates[mr.clamp(min=0)],torch.zeros((len(mr),nextn),dtype=torch.float64))
    ranks=torch.argsort(torch.cumprod(gathered,dim=1).reshape(-1),descending=True,stable=True)
    counts=torch.bincount(ranks[:c["totalQ"]-len(current)]//nextn,minlength=len(current))
    q[:len(current)]=(1+counts).tolist()
kref=torch.tensor(k[:len(current)],dtype=torch.int64)
kref+=torch.tensor(q[:len(current)],dtype=torch.int64)
if nextn>0:
    m=torch.tensor(r,dtype=torch.int64)
    accepted=torch.tensor(c["accepted"],dtype=torch.int64)
    kref-=torch.where(m>=0,nextn-accepted[m.clamp(min=0)],torch.zeros_like(m))
k[:len(current)]=kref.tolist()
print(json.dumps({"r":r+[-1]*(pad-len(current)),"q":q,"k":k}))
\`.replace(/\\\`/g,'\`');
const S=(seqId,seqlenQ,seqlenK,isPlaceholder=false)=>({seqId,seqlenQ,seqlenK,isPlaceholder});
const tests=[
 {prev:[S('A',1,12),S('B',1,16),S('C',1,9)],
  curr:[S('B',1,18),S('D',1,5),S('A',1,13)],
  nextn:2,pad:5,accepted:[1,2,0]},
 {prev:[S('A',1,12),S('B',1,16)],curr:[S('A',1,17),S('B',1,19)],
  nextn:2,pad:2,accepted:[0,2]},
 {prev:[S('A',1,12),S('B',1,16)],curr:[S('B',1,18),S('C',1,5),S('A',1,13)],
  nextn:2,pad:3,accepted:[1,2],varlen:true,totalQ:6,rates:[[.9,.8],[.7,.6]]},
 {prev:[S('A',1,12)],curr:[S('D',1,3)],
  nextn:2,pad:2,accepted:[1]},
 {prev:[S('A',1,12,true),S('B',1,9)],curr:[S('A',1,4),S('B',1,10)],
  nextn:3,pad:3,accepted:[2,0],varlen:true,totalQ:4,rates:[[.6,.5,.4],[.9,.8,.7]]},
 {prev:[S('A',1,12)],curr:[S('A',1,13)],nextn:0,pad:2,accepted:[0]}
];
for(const t of tests){
 const prepared=Z.prepare(t.curr,t.prev,{nextn:t.nextn,paddedLength:t.pad});
 const actual=prepared?Z.correct(prepared,{accepted:t.accepted,
   variableMtp:!!t.varlen,confidenceRates:t.rates,totalQ:t.totalQ}):null;
 const output=JSON.parse(execFileSync('python3',['-c',PY,JSON.stringify(t)],{
   encoding:'utf8',timeout:120000,maxBuffer:2*1024*1024}).trim().split('\n').at(-1));
 if(!output){assert.equal(actual,null);continue;}
 assert.deepEqual(actual.remap,output.r,'C++ remap parity');
 assert.deepEqual(actual.seqlensQ,output.q,'reference MTP Q parity');
 assert.deepEqual(actual.seqlensK,output.k,'reference accepted-token K parity');
}
console.log('DEEPINFER_ZOS_TORCH_REFERENCE_PARITY_PASS 6 C++ remap and torch reference scenarios');
