'use strict';
const fs=require('node:fs'),path=require('node:path');
const L=require('../learning_lab.js'),Codegen=require('../python_codegen.js');
const root=path.resolve('artifacts/executable-python');
fs.mkdirSync(root,{recursive:true});
const names=[
 'nd_cached_gqa_decode','nd_trainable_gqa','nd_trainable_mqa',
 'trainable_swiglu','optimizer_compare','activation_gallery'
];
for(const name of names){
 const output=Codegen.compile(L.preset(name));
 fs.writeFileSync(path.join(root,name+'.py'),output.source,'utf8');
 console.log('Bundled executable '+name+'.py ('+output.nodes.length+' operations)');
}
fs.writeFileSync(path.join(root,'README.md'),[
 '# SeetheTruth standalone Python programs',
 '',
 'These .py files were generated from editable Scratch-like computation graphs,',
 'not copied from the browser runtime. Each uses genuine NumPy tensor operations.',
 '',
 '## Run',
 '',
 'Install Python >=3.10 and NumPy:',
 '',
 '    python -m pip install "numpy>=1.26,<3"',
 '    python nd_cached_gqa_decode.py',
 '    python nd_trainable_gqa.py',
 '    python optimizer_compare.py',
 '',
 'Each program prints JSON containing every intermediate tensor keyed by the',
 'original block ID. Import it and call run_graph(overrides={...}) to change',
 'a source tensor (Input or Parameter) and recalculate the entire DAG.',
 '',
 'The NumPy exporter implements forward tensor operations and functional',
 'single-step optimizer preview blocks. It does not automatically create',
 'an autograd training loop from a Parameter node; the workbench separately',
 'supports bounded VJP and stateful training.',
 '',
 'The GitHub numerical-contract workflow executes generated Python scripts',
 'and checks intermediate arrays against the JS graph interpreter.',
 'Unsupported blocks fail export instead of generating placeholder math.',
 ''
].join('\n'),'utf8');
