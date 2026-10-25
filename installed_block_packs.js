/* Trusted, build-time installed block pack manifest.
 * One ordered source of truth for installed pack identities and entry modules.
 * Not a remote loader or authority to run third-party JavaScript.
 */
(function(root,factory){'use strict';
 const api=factory(typeof module==='object'&&module.exports?require:null,root);
 if(typeof module==='object'&&module.exports)module.exports=api;
 if(root)root.SeetheTruthInstalledBlockPacks=api;
})(typeof window!=='undefined'?window:null,function(nodeRequire,root){'use strict';
 const SCHEMA='seethetruth-installed-block-packs-v1';
 const entries=Object.freeze([
  Object.freeze({id:'semantic_video',module:'semantic_block_pack.js',global:'SeetheTruthSemanticBlockPack'}),
  Object.freeze({id:'named_axis',module:'axis_block_pack.js',global:'SeetheTruthAxisBlockPack'})
 ]);
 function fail(message){throw Error('installed block packs: '+message);}
 function validate(items){
  if(!Array.isArray(items)||!items.length||items.length>32)fail('expected 1..32 build-time pack entries');
  const ids=new Set(),paths=new Set(),globals=new Set();
  for(const e of items){
   if(!e||typeof e!=='object'||Array.isArray(e)||Object.keys(e).sort().join(',')!=='global,id,module'||
      typeof e.id!=='string'||!/^[a-z][a-z0-9_]{1,35}$/.test(e.id)||
      typeof e.module!=='string'||!/^[a-z][a-z0-9_]*_block_pack\.js$/.test(e.module)||
      typeof e.global!=='string'||!/^SeetheTruth[A-Za-z0-9]+$/.test(e.global))
    fail('invalid trusted pack entry');
   if(ids.has(e.id)||paths.has(e.module)||globals.has(e.global))
    fail('duplicate pack identity/module/global');
   ids.add(e.id);paths.add(e.module);globals.add(e.global);
  }
  return Object.freeze(items.map(e=>Object.freeze({...e})));
 }
 function resolve(items,loader){
  const approved=validate(items);
  if(typeof loader!=='function')fail('missing source module resolver');
  const packs=approved.map(e=>{
   const pack=loader(e);
   if(!pack||pack.name!==e.id||!pack.defs||
      typeof pack.validate!=='function'||typeof pack.infer!=='function'||
      typeof pack.execute!=='function')
    fail('installed pack source/identity mismatch: '+e.id);
   return pack;
  });
  return Object.freeze(packs);
 }
 const installed=validate(entries);
 function load(){return resolve(installed,e=>{
  if(nodeRequire)return nodeRequire('./'+e.module);
  if(!root||!Object.prototype.hasOwnProperty.call(root,e.global))
   fail('browser source script missing before extension registry: '+e.module);
  return root[e.global];
 });}
 return Object.freeze({SCHEMA,entries:installed,validate,resolve,load});
});
