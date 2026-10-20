/* DeepInfer C++ BlockManager & Scheduler bounded lifecycle reference, not GPU allocator.
 * Source: csrc/engine/block_manager.hpp, scheduler.hpp, sequence.hpp.
 * Pure transitions allow exact FIFO free-list/ref-count and fork/preemption audit.
 */
(function(root,factory){"use strict";const api=factory();if(typeof module==="object"&&module.exports)module.exports=api;root.SeetheTruthKVLifecycle=api;})(typeof window!=="undefined"?window:{},function(){"use strict";
 const fail=s=>{throw Error(s)};
 const uint=(x,label,min=0,max=10000)=>{if(!Number.isSafeInteger(x)||x<min||x>max)fail(label+" outside "+min+".."+max);return x};
 const checkId=id=>{if(typeof id!=="string"||!/^[A-Za-z0-9_-]{1,24}$/.test(id))fail("invalid request ID");return id};
 const copy=s=>JSON.parse(JSON.stringify(s));
 function initial({numBlocks=8,blockSize=4}={}){
  uint(numBlocks,"numBlocks",2,64);uint(blockSize,"blockSize",1,64);
  return {numBlocks,blockSize,free:Array.from({length:numBlocks-1},(_,i)=>i+1),
   refs:Array(numBlocks).fill(0),requests:{},events:[],copies:[],revision:0};
 }
 function blocksNeeded(s,blockSize){
  return Math.ceil((s.tokenLen+s.inflight+s.draft)/blockSize);
 }
 function canAllocate(s,n){
  uint(n,"requested blocks",0,64);
  return n>=s.numBlocks?"NEVER":s.free.length<n?"LATER":"OK";
 }
 function verify(s){
  if(!s||!Number.isSafeInteger(s.numBlocks)||!Array.isArray(s.free)||!Array.isArray(s.refs)||s.refs.length!==s.numBlocks||s.refs[0]!==0)fail("invalid block pool state");
  const freeSet=new Set(s.free);
  if(freeSet.size!==s.free.length||s.free.some(id=>!Number.isInteger(id)||id<=0||id>=s.numBlocks))fail("invalid free FIFO");
  const owners=Array(s.numBlocks).fill(0);
  for(const [id,req] of Object.entries(s.requests)){
   checkId(id);
   if(!["WAITING","RUNNING","FINISHED","FINISHED_PREEMPTED"].includes(req.status))fail("unknown request status");
   if(!Array.isArray(req.blocks)||new Set(req.blocks).size!==req.blocks.length)fail("duplicate block in single request");
   for(const b of req.blocks){if(!Number.isInteger(b)||b<=0||b>=s.numBlocks)fail("invalid owned block");owners[b]++}
   if(req.status.startsWith("FINISHED")&&req.blocks.length)fail("finished request still holds blocks");
  }
  for(let i=1;i<s.numBlocks;i++){
   if(s.refs[i]!==owners[i]||s.refs[i]<0)fail("reference count mismatch on block "+i);
   if((s.refs[i]===0)!==freeSet.has(i))fail("block free/allocated conflict "+i);
  }
  if(s.free.length+Array.from({length:s.numBlocks-1},(_,i)=>s.refs[i+1]>0?1:0).reduce((a,b)=>a+b,0)!==s.numBlocks-1)fail("pool capacity conservation failed");
  return true;
 }
 function take(s,n){
  const status=canAllocate(s,n);if(status!=="OK")fail("allocation "+status+" requested="+n);
  const out=[];
  for(let i=0;i<n;i++){const id=s.free.shift();s.refs[id]=1;out.push(id)}
  return out;
 }
 function unpin(s,ids){
  // BlockManager.free_blocks_nolock reverses the supplied list before freeing.
  for(const id of ids.slice().reverse()){
   if(s.refs[id]<=0)fail("double-free block "+id);
   s.refs[id]--;
   if(s.refs[id]===0)s.free.push(id);
  }
 }
 function pin(s,ids){
  for(const id of ids){if(s.refs[id]<=0)fail("cannot pin free block "+id);s.refs[id]++}
 }
 function advance(prev,operation){
  verify(prev);
  if(!operation||typeof operation!=="object")fail("operation required");
  const s=copy(prev),op=operation.op,id=operation.id;
  const item=()=>{checkId(id);const r=s.requests[id];if(!r)fail("request not found: "+id);return r};
  const ensureActive=r=>{if(r.status.startsWith("FINISHED"))fail("request already finished")};
  const log=(kind,data={})=>s.events.push({revision:prev.revision+1,op:kind,id,...data});
  if(op==="arrive"){
   checkId(id);if(s.requests[id])fail("duplicate request");
   s.requests[id]={status:"WAITING",tokenLen:uint(operation.tokenLen,"tokenLen",0),
    cachedTokens:0,inflight:0,draft:0,structuredStatus:0,blocks:[],branch:false,recomputes:0};
   log(op);
  }else if(op==="allocate"){
   const r=item();ensureActive(r);
   if(r.status!=="WAITING"||r.blocks.length)fail("allocate requires unallocated WAITING request");
   const need=blocksNeeded(r,s.blockSize);
   r.blocks=take(s,need);r.status="RUNNING";
   log(op,{status:"OK",allocated:r.blocks.slice(),need});
  }else if(op==="update_workload"){
   const r=item();ensureActive(r);if(r.status!=="RUNNING"||r.branch)fail("workload requires active normal request");
   r.tokenLen=uint(operation.tokenLen,"tokenLen",0);
   r.inflight=uint(operation.inflight??r.inflight,"inflight",0);
   r.draft=uint(operation.draft??r.draft,"draft",0);
   r.cachedTokens=uint(operation.cachedTokens??r.cachedTokens,"cachedTokens",0);
   log(op,{required:blocksNeeded(r,s.blockSize)});
  }else if(op==="append_slot"){
   const r=item();ensureActive(r);if(r.status!=="RUNNING"||r.branch)fail("append requires running non-branch");
   const n=blocksNeeded(r,s.blockSize),d=n-r.blocks.length;
   if(d>1)fail("Scheduler._append_slot only appends one block at a time");
   if(d===1)r.blocks.push(...take(s,1));
   if(d<0){unpin(s,r.blocks.slice(n));r.blocks=r.blocks.slice(0,n)}
   log(op,{required:n,delta:d,blocks:r.blocks.slice()});
  }else if(op==="fork_pin"){
   checkId(operation.from);const parent=s.requests[operation.from];
   if(!parent||parent.status!=="RUNNING"||parent.branch)fail("fork requires running parent");
   checkId(id);if(s.requests[id])fail("duplicate child");
   const cached=uint(operation.cachedTokens,"cachedTokens",0);
   if(cached>parent.tokenLen||cached>parent.blocks.length*s.blockSize)fail("fork cached prefix exceeds parent blocks");
   const len=uint(operation.tokenLen??parent.tokenLen,"tokenLen",cached);
   if(len<cached)fail("child shorter than cached prefix");
   pin(s,parent.blocks);
   s.requests[id]={status:"WAITING",tokenLen:len,cachedTokens:cached,inflight:0,draft:0,structuredStatus:0,
    blocks:parent.blocks.slice(),branch:true,recomputes:0,parent:operation.from};
   log(op,{parent:operation.from,pinned:parent.blocks.slice(),cached});
  }else if(op==="fork_activate"){
   const r=item();if(!r.branch||r.status!=="WAITING")fail("fork_activate requires pinned child");
   const full=Math.floor(r.cachedTokens/s.blockSize),total=blocksNeeded(r,s.blockSize),n=total-full;
   if(n<0)fail("child block count smaller than cached prefix");
   const status=canAllocate(s,n);if(status!=="OK")fail("fork allocation "+status);
   const newly=take(s,n),old=r.blocks.slice(),partial=r.cachedTokens%s.blockSize!==0;
   if(partial){if(old.length<=full||!newly.length)fail("partial clone missing source/destination");
    s.copies.push({step:prev.revision+1,from:old[full],to:newly[0],type:"clone_partial_tail"});}
   // Pinned old blocks beyond the complete prefix are released in source-order
   // via free_blocks_nolock, which reverses its vector.
   unpin(s,old.slice(full));
   r.blocks=[...old.slice(0,full),...newly];r.branch=false;r.status="RUNNING";
   log(op,{sharedCompletePrefix:full,copiedPartialTail:partial,blocks:r.blocks.slice(),
     requiredNew:n});
  }else if(op==="preempt"){
   const r=item();if(r.status!=="RUNNING")fail("preempt requires RUNNING");
   const terminate=operation.stop===true||r.structuredStatus!==0;
   unpin(s,r.blocks);r.blocks=[];r.cachedTokens=0;r.inflight=0;r.draft=0;
   r.status=terminate?"FINISHED_PREEMPTED":"WAITING";
   if(!terminate)r.recomputes++;
   log(op,{outcome:terminate?"stop":"recompute",reason:r.structuredStatus!==0?"structured_output":
    operation.stop===true?"stop_preempted_request":"memory_pressure"});
  }else if(op==="set_structured"){
   const r=item();ensureActive(r);r.structuredStatus=uint(operation.status,"structuredStatus",0,2);log(op,{status:r.structuredStatus});
  }else if(op==="finish"){
   const r=item();ensureActive(r);unpin(s,r.blocks);r.blocks=[];r.status="FINISHED";
   log(op,{reason:operation.reason||"normal"});
  }else fail("unknown operation "+op);
  s.revision++;
  verify(s);
  return s;
 }
 function replay(config,ops){
  if(!Array.isArray(ops)||ops.length>80)fail("operation list must have <=80 items");
  let s=initial(config);verify(s);
  const timeline=[copy(s)];
  for(const op of ops){s=advance(s,op);timeline.push(s)}
  return {state:s,timeline};
 }
 return Object.freeze({initial,verify,canAllocate,blocksNeeded,advance,replay});
});
