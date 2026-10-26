'use strict';
/* Run the real small 5D semantic execution through a third scalar reference
 * and two actually executed, independent Python CPU tensor backends.
 */
const {spawnSync}=require('node:child_process');
const path=require('node:path');
const {payload}=require('./semantic_5d_source_oracle.js');
const result=spawnSync('python3',
 [path.join(__dirname,'python_semantic_5d_reference.py')],{
  input:JSON.stringify(payload),encoding:'utf8',timeout:120000,maxBuffer:4*1024*1024
 });
if(result.error||result.status!==0)
 throw Error('independent 5D CPU oracle failed: '+(result.error?.message||result.stderr||result.stdout).slice(-7000));
if(!result.stdout.includes('SEMANTIC_5D_BACKENDS_EXECUTED_PASS'))
 throw Error('independent 5D oracle did not report executed proof');
process.stdout.write(result.stdout);
