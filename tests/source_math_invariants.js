'use strict';
/* Fast source-math obligations: real LearningLab/GraphAutodiff against
 * independently written scalar equations. No NumPy, PyTorch, network or GPU.
 * This does not confer cross-backend or model-wide verification.
 */
const assert = require('node:assert/strict');
const Learning = require('../learning_lab.js');
const Autodiff = require('../graph_autodiff.js');
const {cases, create} = require('./fixtures/nd_broadcast_reduction_graphs.js');

function compare(expected, actual, at, tolerance=1e-10) {
  if (Array.isArray(expected)) {
    assert.ok(Array.isArray(actual), at + ' expected tensor');
    assert.equal(actual.length, expected.length, at + ' shape mismatch');
    expected.forEach((value, i) => compare(value, actual[i], at+'['+i+']', tolerance));
    return;
  }
  assert.ok(typeof actual === 'number' && Number.isFinite(actual), at+' nonfinite');
  assert.ok(Math.abs(expected-actual) <= tolerance + 1e-10*Math.abs(expected),
    at+' mismatch: expected '+expected+' got '+actual);
}
const map4 = (tensor, fn) => tensor.map((batch,b) =>
  batch.map((head,h) => head.map((token,t) =>
    token.map((value,c) => fn(value,b,h,t,c)))));
function manualOracle(trace, shape) {
  const byId = Object.fromEntries(trace.steps.map(x => [x.id,x.values]));
  const {x,bias,gain,target} = byId;
  const [B,H,T,C] = shape;
  const biased = map4(x,(value,b,h,t,c) => value+bias[0][0][0][c]);
  const gated = map4(biased,(value,b,h,t,c) => value*gain[0][0][0][c]);
  const summed = Array.from({length:B},(_,b) =>
    Array.from({length:H},(_,h) => [
      Array.from({length:C},(_,c) => {
        let result=0;
        for(let t=0;t<T;t++)result+=gated[b][h][t][c];
        return result;
      })
    ]));
  let loss=0;
  const count=B*H*C;
  for(let b=0;b<B;b++)for(let h=0;h<H;h++)for(let c=0;c<C;c++)
    loss+=(summed[b][h][0][c]-target[b][h][0][c])**2/count;
  const gradSummed = summed.map((batch,b) => batch.map((head,h) =>
    [head[0].map((value,c) => 2*(value-target[b][h][0][c])/count)]));
  const gradGated = map4(gated,(_,b,h,t,c) => gradSummed[b][h][0][c]);
  const gradBiased = map4(biased,(_,b,h,t,c) =>
    gradGated[b][h][t][c]*gain[0][0][0][c]);
  const gradBias = [ [ [ Array.from({length:C},(_,c) => {
    let sum=0;
    for(let b=0;b<B;b++)for(let h=0;h<H;h++)for(let t=0;t<T;t++)
      sum+=gradBiased[b][h][t][c];
    return sum;
  }) ] ] ];
  const gradGain = [ [ [ Array.from({length:C},(_,c) => {
    let sum=0;
    for(let b=0;b<B;b++)for(let h=0;h<H;h++)for(let t=0;t<T;t++)
      sum+=gradGated[b][h][t][c]*biased[b][h][t][c];
    return sum;
  }) ] ] ];
  return {
    forward:{biased,gated,summed,loss:[[loss]]},
    gradient:{x:gradBiased,bias:gradBias,gain:gradGain,
      biased:gradBiased,gated:gradGated,summed:gradSummed}
  };
}
const expectedKinds={x:'nd_parameter',bias:'nd_parameter',
  gain:'nd_parameter',biased:'nd_broadcast_add',gated:'nd_broadcast_mul',
  summed:'nd_reduce_sum',target:'nd_input',loss:'nd_mse'};
const expectedEdges=[
  ['x','biased','a'],['bias','biased','b'],
  ['biased','gated','a'],['gain','gated','b'],
  ['gated','summed','x'],['summed','loss','prediction'],['target','loss','target']
];
let forwardCoordinates=0,gradientCoordinates=0;
for(const spec of cases) {
  const graph=create(spec.id), untouched=JSON.stringify(graph);
  assert.deepEqual(Object.fromEntries(graph.nodes.map(n=>[n.id,n.kind])),
    expectedKinds, spec.id+' expected exact semantic source graph');
  assert.deepEqual(graph.edges.map(e=>[e.from,e.to,e.port]),expectedEdges,
    spec.id+' mathematical input lineage');
  assert.equal(graph.nodes.find(n=>n.id==='summed').params.axis,2);
  const trace=Learning.evaluate(graph);
  const gradient=Autodiff.backward(graph,trace,{targetId:'loss'});
  const oracle=manualOracle(trace,spec.shape);
  const steps=Object.fromEntries(trace.steps.map(x=>[x.id,x.values]));
  for(const [node, expected] of Object.entries(oracle.forward)){
    compare(expected,steps[node],spec.id+'/forward/'+node);
    forwardCoordinates+=expected.flat(Infinity).length;
  }
  for(const [node, expected] of Object.entries(oracle.gradient)){
    assert.ok(gradient.pathIds.includes(node),spec.id+'/'+node+' missing whole-loss path');
    compare(expected,gradient.gradients[node],spec.id+'/gradient/'+node);
    gradientCoordinates+=expected.flat(Infinity).length;
  }
  // A broken reduction that retains only the first B/H/T contributor must be caught.
  const truncated=structuredClone(oracle.gradient.gain);
  truncated[0][0][0][0]=oracle.gradient.gated[0][0][0][0]*
    oracle.forward.biased[0][0][0][0];
  assert.throws(()=>compare(oracle.gradient.gain,truncated,
    spec.id+'/deliberately_truncated_gradient'),/mismatch/);
  const mismatched=structuredClone(graph);
  mismatched.nodes.find(n=>n.id==='bias').params.values[0][0][0].push(123);
  assert.throws(()=>Learning.evaluate(mismatched),
    spec.id+' mismatched source shape must fail');
  assert.equal(JSON.stringify(graph),untouched,spec.id+' fixture mutation');
}
assert.equal(cases.length,2);
assert.ok(forwardCoordinates>20&&gradientCoordinates>30);
console.log('SOURCE_MATH_FAST_PASS '+JSON.stringify({
  fixtures:cases.map(c=>c.id),forwardCoordinates,gradientCoordinates,
  oracle:'independent scalar equations',runtime:'node-only',
  claim:'source graph fixture only'
}));
