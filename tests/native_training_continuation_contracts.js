'use strict';
const assert=require('node:assert/strict');
const Core=require('../learning_experience_core.js');
const Snap=require('../observation_snapshots.js');
const P=require('../engineering_training_pipeline.js');
const actions=['micro','micro','micro','micro','optimizer','micro','micro','micro','micro','optimizer'];
const config={seed:7,samplerSeed:19,lr:.08,microBatchSize:2,accumulationSteps:4};
const input={config,actions};
const run=Core.executeLab('engineering_pipeline','timeline',input);
assert.equal(run.value.snapshots.length,actions.length+1);
assert.equal(run.stateTrace.steps.length,actions.length+1);
assert.equal(run.stateTrace.evidenceTier,'EXECUTABLE_BIGRAM_OPTIMIZER_STATE_SIMULATION');
const independent=P.create(config),original=[independent.checkpoint()];
for(const a of actions){if(a==='micro')independent.microStep();else independent.optimizerStep();original.push(independent.checkpoint());}
assert.deepEqual(run.value.snapshots,original);
const originalSnap=Snap.create(Core,run,'exact microbatch reference');
assert.equal(Snap.restore(Core,originalSnap).record.operation,'timeline');
for(const idx of [0,2,4,5,6,10]){
 const cursor=Snap.at(Core,originalSnap,'train:'+idx);
 assert.equal(cursor.cursor.index,idx);
 assert.equal(Snap.restore(Core,cursor).checkpoint.stepId,'train:'+idx);
 const continued=Snap.resumeTraining(Core,cursor);
 assert.equal(continued.kind,'local_training_continuation');
 assert.equal(continued.continuedFromCheckpoint,true);
 assert.equal(continued.finalReplayed,true);
 assert.deepEqual(continued.restoredCheckpoint,original[idx]);
 assert.deepEqual(continued.resumedTimeline,original.slice(idx+1));
 assert.deepEqual(continued.remainingActions,actions.slice(idx));
 assert.equal(continued.restoredCheckpoint.pending.microSteps,
              original[idx].pending.microSteps);
 assert.deepEqual(continued.restoredCheckpoint.model.firstMoment,
                  original[idx].model.firstMoment);
 assert.deepEqual(continued.restoredCheckpoint.model.secondMoment,
                  original[idx].model.secondMoment);
 assert.deepEqual(continued.restoredCheckpoint.sampler,original[idx].sampler);
}
const micro=Snap.resumeTraining(Core,Snap.at(Core,originalSnap,'train:2'));
assert.equal(micro.restoredCheckpoint.model.step,0,'microbatch must not update optimizer');
assert.equal(micro.restoredCheckpoint.pending.microSteps,2);
assert.equal(micro.resumedTimeline[2].model.step,1,'optimizer updates after four microbatches');
const optimizerBoundary=Snap.resumeTraining(Core,Snap.at(Core,originalSnap,'train:5'));
assert.equal(optimizerBoundary.restoredCheckpoint.pending.microSteps,0);
assert.equal(optimizerBoundary.restoredCheckpoint.model.step,1);
const wrong=structuredClone(originalSnap);
wrong.input.actions[1]='optimizer';
assert.throws(()=>Snap.restore(Core,wrong),/must finish every microbatch/);
const altered=structuredClone(Snap.at(Core,originalSnap,'train:3'));
altered.cursor.stateDigest='fnv1a32:deadbeef:5';
assert.throws(()=>Snap.resumeTraining(Core,altered),/state cursor replay differs/);
const shifted=structuredClone(Snap.at(Core,originalSnap,'train:3'));
shifted.cursor.index++;
assert.throws(()=>Snap.resumeTraining(Core,shifted),/state cursor no longer matches/);
assert.throws(()=>Snap.resumeTraining(Core,originalSnap),/requires a verified native timeline cursor/);
const legacy=Core.executeLab('engineering_pipeline','train',{config,steps:2});
assert.throws(()=>Snap.resumeTraining(Core,Snap.at(Core,Snap.create(Core,legacy),'train:1')),
 /requires a verified native timeline cursor/);
assert.throws(()=>Snap.resumeTraining(Core,Snap.at(Core,Snap.create(Core,Core.executeLab('training','scalar',{config:{steps:1}})),'scalar:1')),
 /requires a verified native timeline cursor/);
assert.throws(()=>Core.executeLab('engineering_pipeline','timeline',{config,actions:['optimizer']}),/must finish every microbatch/);
assert.throws(()=>Core.executeLab('engineering_pipeline','timeline',{config,actions:['unverified']}),/explicit micro\/optimizer actions/);
assert.throws(()=>Core.executeLab('engineering_pipeline','timeline',{config,actions:Array(25).fill('micro')}),/1..24/);
assert.deepEqual(P.create(config).checkpoint(),original[0],'timeline must not contaminate engine singleton state');
console.log('TRAINING_NATIVE_CONTINUATION_PASS: exact micro and optimizer CPU engine source, pending gradient and sampler replay, adversarial cursors');
