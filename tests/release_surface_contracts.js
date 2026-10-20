'use strict';
// A tiny distribution/landing-page gate. No external dependencies or GPU.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const root=path.resolve(__dirname,'..');
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const L=require('../learning_lab.js'),B=require('../block_catalog.js');
const readme=read('README.md'),html=read('learn.html'),server=read('atlas_server.py');
const release=read('docs/release-readiness.md');
assert.equal(L.KINDS.length,11+B.KINDS.length,'all 11 original and extension definitions registered');
assert.equal(new Set(L.KINDS).size,L.KINDS.length,'all block kinds are unique');
const guided=[...html.matchAll(/data-preset="([^"]+)"/g)].map(m=>m[1]);
assert.equal(guided.filter(x=>x==='blank').length,1,'one blank canvas entry');
assert.equal(new Set(guided).size,guided.length,'each course is represented once');
const lessons=guided.filter(x=>x!=='blank');
assert.equal(lessons.length,59,'all 59 guided lessons must appear exactly once');
assert.match(readme,new RegExp('\\*\\*'+L.KINDS.length+' 种积木\\*\\*'),'landing block count must match executable registry');
assert.match(readme,new RegExp('\\*\\*'+lessons.length+' 门引导课程\\*\\*'),'landing lesson count must match selectable presets');
assert.match(readme,/NumPy Python/,'landing must name the real export backend');
assert.match(readme,/PyTorch CPU/,'landing must document trainable PyTorch backend');
assert.match(readme,/不支持的节点会明确拒绝/,'do not imply arbitrary block compilation');
assert.match(readme,/无需 GPU/,'GPU must not be mandatory for learning');
assert.match(readme,/TinyStories-1M CPU 采集及独立二次复跑已通过/,'pinned real-model replay evidence must be disclosed');
assert.match(readme,/用户本地导入但未独立复跑的 JSON 仍仅属于来源声明/,'unattested imported JSON must not be promoted to real-model evidence');
for(const route of ['learn.html','tradeoffs.html','training.html','cache.html','paper_lab.html','micro_training.html','real_model_lab.html','index.html','telemetry.html']){
 assert.ok(fs.existsSync(path.join(root,route)),route+' actually exists');
 assert.ok(server.includes('"/'+route+'"'),route+' registered by server');
 assert.ok(readme.includes(']('+route+')'),route+' linked from README');
}
assert.ok(html.includes('src="multimodal_ops.js"'),'multimodal numerical core must be loaded');
assert.ok(server.includes('"/multimodal_ops.js"'),'multimodal numerical core must be served');
assert.ok(html.includes('src="multimodal_advanced.js"'),'advanced multimodal module loads');
assert.ok(server.includes('"/multimodal_advanced.js"'),'advanced multimodal operator route exists');
assert.ok(html.includes('src="video_causal_conv3d.js"'),'native video Conv3D kernel must be loaded');
assert.ok(server.includes('"/video_causal_conv3d.js"'),'native video operator registered for local HTTP');
assert.ok(guided.includes('video_causal_conv3d'),'lesson 53 registered');
assert.ok(html.includes('src="inference_ops.js"'),'inference CPU kernels must be loaded');
assert.ok(server.includes('"/inference_ops.js"'),'inference CPU module must have HTTP route');
assert.ok(html.includes('src="deepinfer_runtime_ops.js"'),'DeepInfer runtime module must be loaded');
assert.ok(server.includes('"/deepinfer_runtime_ops.js"'),'DeepInfer runtime HTTP route must exist');
for(const preset of ['deepinfer_mtp_budget','deepinfer_yoco_replay'])assert.ok(guided.includes(preset),'missing DeepInfer source-ground lesson: '+preset);
for(const preset of ['inference_sparse_decode','inference_paged_kv','inference_speculative','inference_paged_sparse'])assert.ok(guided.includes(preset),'missing inference lesson: '+preset);
for(const preset of ['vision_learned_conv','audio_mel_chain','diffusion_ddim_chain'])assert.ok(guided.includes(preset),'missing advanced multimodal lesson: '+preset);
for(const preset of ['image_vit_patch','audio_spectrogram','video_vivit_tubelet','multimodal_cross_attention'])assert.ok(guided.includes(preset),'missing multimodal lesson: '+preset);
assert.ok(html.includes('src="graph_cost_report.js"'),'live workbench cost analyzer must be loaded');
assert.ok(server.includes('"/graph_cost_report.js"'),'live cost analyzer served by local HTTP');
assert.ok(html.includes('id="graphCostPanel"'),'live graph cost panel must exist');
assert.ok(html.includes('id="freezeGraphBaseline"')&&html.includes('id="graphCostComparison"'),'cost counterfactual controls must exist');
assert.ok(html.includes('id="graphCostNodeList"'),'graph cost evidence must expose per-node inspector links');
assert.ok(html.includes('href="tradeoffs.html"'),'Learning Studio must link the architecture cost laboratory');
assert.ok(read('tradeoffs.html').includes('src="architecture_tradeoffs.js"'),'tradeoff page must load its cost model');
assert.ok(server.includes('"/architecture_tradeoffs.js"'),'cost model must be served by the local web server');
for(const source of ['python_codegen.js','torch_codegen.js','loop_transformer.js','loop_input_injection.js','adaptive_token_depth.js','loop_consistency.js','loop_design_tradeoffs.js','architecture_why.js','training_objective.js','paper_graph_rewrites.js','industrial_source_inspector.js','model_trace.js','model_trace_bridge.js','paper_hover.js','block_catalog.js','graph_readiness.js']){
 assert.ok(html.includes('src="'+source+'"'),source+' loaded in the public workbench');
}
for(const doc of ['docs/executable-python-codegen.md','docs/real-model-sampling.md','docs/release-readiness.md']){
 assert.ok(fs.existsSync(path.join(root,doc)),'required release docs exist: '+doc);
}
assert.match(release,/LICENSE/,'license decision must be tracked');
console.log('RELEASE_SURFACE_PASS blocks='+L.KINDS.length+' lessons='+lessons.length+' blank=1');
