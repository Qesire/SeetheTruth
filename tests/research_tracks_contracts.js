'use strict';
/* Fast F tier: precise small-number oracles and adversarial state controls.
 * These tests certify only the bounded research teaching fixtures.
 */
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const Wan=require('../wan_ffn_trace.js');
const R=require('../research_tracks.js');
const C=require('../teaching_curriculum.js');
const close=(x,y,eps=2e-12)=>assert.ok(Math.abs(x-y)<=eps,
 'mismatch '+x+' vs '+y);
for(const seed of [17,31,47])for(const selection of ['gated','activation'])
 for(const scope of ['static','dynamic'])for(const keep of [1,2,4,6,8]){
  const result=R.videoSparsity({seed,selection,scope,keep});
  const reference=Wan.trace({seed,tokens:3,dim:4,hidden:8});
  const {tokens:T,dim:D,hidden:H}=reference.config;
  const activations=reference.steps[5].main.values,gate=reference.config.gate;
  for(let t=0;t<T;t++)for(let c=0;c<D;c++){
   const dropped=Array.from({length:H},(_,h)=>h).filter(h=>!result.kept[t].includes(h));
   const expected=reference.final.values[t*D+c]-dropped.reduce((sum,h)=>
    sum+activations[t*H+h]*reference.config.w2[c*H+h]*gate*reference.e[5][t*D+c],0);
   close(result.sparse[t*D+c],expected);
  }
  close(result.errorL2,Math.hypot(...result.dropped));
  assert.equal(result.kept.length,T);
  for(const ids of result.kept){assert.equal(ids.length,keep);assert.equal(new Set(ids).size,keep);}
  if(scope==='static')assert.deepEqual(result.kept[0],result.kept[1]);
  close(result.idealMacFraction,scope==='static'?keep/H:(1+keep/H)/2);
  if(keep===H){close(result.errorL2,0);assert.deepEqual(result.sparse,result.dense);}
  assert.match(result.costNote,/MAC|裁剪/);
 }
const base={chosen:-0.6,rejected:-1.4,refChosen:-1,refRejected:-1.4,beta:0.6};
const d=R.dpo(base),eps=1e-6;
const loss=(key,v)=>R.dpo({...base,[key]:v}).loss;
close((loss('chosen',base.chosen+eps)-loss('chosen',base.chosen-eps))/(2*eps),d.gradChosen,1e-9);
close((loss('rejected',base.rejected+eps)-loss('rejected',base.rejected-eps))/(2*eps),d.gradRejected,1e-9);
close(d.gradChosen+d.gradRejected,0);
close(R.dpo({...base,beta:0}).loss,Math.log(2));
close(R.dpo({...base,beta:0}).gradChosen,0);
assert.throws(()=>R.dpo({...base,chosen:Infinity}),/finite/);
const g=R.grpo({rewardGap:2,ratioShift:0.4,epsilon:0.2});
close(g.terms.reduce((s,t)=>s+t.advantage,0),0);
close(g.terms.reduce((s,t)=>s+t.advantage*t.advantage,0)/3,1);
assert.equal(g.terms[0].clipActive,true);
assert.equal(g.terms[2].clipActive,true);
assert.deepEqual(R.grpo({rewardGap:0}).terms.map(t=>t.advantage),[0,0,0]);
close(R.grpo({rewardGap:0}).loss,0);
assert.throws(()=>R.grpo({ratioShift:1}),/ratioShift/);
for(const architecture of ['react','plan_execute','supervisor']){
 const normal=R.agent({architecture,fault:false,approved:true});
 assert.equal(normal.status,'completed');
 assert.equal(normal.events.at(-1).action,'finalize');
 assert.equal(normal.events.at(-2).action,'verify');
 assert.deepEqual(normal,R.agent({architecture,fault:false,approved:true}),
  'scripted traces must be replayable');
 const denied=R.agent({architecture,fault:false,approved:false});
 assert.equal(denied.status,'blocked');
 assert.ok(denied.events.every(e=>!['write_tool','verify','finalize'].includes(e.action)));
 const fault=R.agent({architecture,fault:true,approved:true});
 assert.equal(fault.status,architecture==='plan_execute'?'failed':'completed');
 assert.equal(fault.events.some(e=>e.action==='retry_read'),architecture!=='plan_execute');
 for(let i=0;i<fault.events.length;i++){
  const step=fault.events[i];
  assert.equal(step.id,'e'+i);
  assert.deepEqual(step.dependsOn,i?['e'+(i-1)]:[]);
 }
}
assert.throws(()=>R.agent({architecture:'unknown'}),/architecture/);
assert.throws(()=>R.agent({approved:'yes'}),/boolean/);
for(const id of ['research_video_sparse','research_llm_posttrain','research_agent_arch']){
 const l=C.lookup(id);assert.ok(l);
 assert.ok(fs.existsSync(path.join(root,l.href)));
}
const html=fs.readFileSync(path.join(root,'teaching.html'),'utf8');
const hub=fs.readFileSync(path.join(root,'teaching_hub.js'),'utf8');
const policy=JSON.parse(fs.readFileSync(path.join(root,'static_asset_policy.json'),'utf8'));
const assets=fs.readFileSync(path.join(root,'android/teaching-assets.txt'),'utf8');
const server=fs.readFileSync(path.join(root,'atlas_server.py'),'utf8');
assert.ok(html.includes('id="track-video"')&&html.includes('id="track-post"')&&html.includes('id="track-agent"'));
assert.ok(html.indexOf('src="wan_ffn_trace.js"')<html.indexOf('src="research_tracks.js"'));
assert.ok(html.indexOf('src="research_tracks.js"')<html.indexOf('src="teaching_hub.js"'));
assert.ok(hub.includes('Research.videoSparsity(')&&hub.includes('Research.dpo(')&&
 hub.includes('Research.grpo(')&&hub.includes('Research.agent('));
assert.ok(policy.androidAssets.includes('research_tracks.js'));
assert.ok(assets.split(/\r?\n/).includes('research_tracks.js'));
assert.ok(server.includes('"/research_tracks.js":"research_tracks.js"'));
console.log('RESEARCH_TRACKS_PASS synthetic Wan channel ablation / DPO finite difference / GRPO group invariants / Agent fault and approval / shared delivery');
