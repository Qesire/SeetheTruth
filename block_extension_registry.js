/* Core block-extension contract. One installed pack contributes definitions, parameter
 * contracts, symbolic shapes, actual numerical execution and optional source-derived
 * operator substeps. Extension dispatch is kind-independent; no per-kind core switch. */
(function(root,factory){'use strict';
 const deps=typeof module==='object'&&module.exports
  ?[require('./semantic_block_pack.js'),require('./paper_block_pack.js')]
  :[root.SeetheTruthSemanticBlockPack,root.SeetheTruthPaperBlockPack];
 const api=factory(deps);
 if(typeof module==='object'&&module.exports)module.exports=api;
 root.SeetheTruthBlockExtensions=api;
})(typeof window!=='undefined'?window:{},function(installed){'use strict';
 const numeric=x=>typeof x==='number'&&Number.isFinite(x);
 function must(v,msg){if(!v)throw Error('block extension: '+msg);}
 function tensorShape(value){
  const dims=[];let t=value;
  while(Array.isArray(t)){must(dims.length<5&&t.length>0&&t.length<=64,'invalid rank/dimensions');
   dims.push(t.length);const first=Array.isArray(t[0]);for(const x of t)must(Array.isArray(x)===first,'ragged numeric tensor');
   if(first){const len=t[0].length;must(t.every(x=>x.length===len),'ragged numerical tensor');}
   t=t[0];
  }
  must(numeric(t)&&dims.length>=1,'nonfinite or empty tensor');
  const walk=(v,depth)=>{
   if(depth===dims.length){must(numeric(v),'nonfinite tensor value');return;}
   must(Array.isArray(v)&&v.length===dims[depth],'ragged tensor');
   for(const x of v)walk(x,depth+1);
  };
  must(dims.reduce((p,x)=>p*x,1)<=4096,'tensor exceeds numerical teaching budget');walk(value,0);
  return dims;
 }
 function createRegistry(packs){
  must(Array.isArray(packs)&&packs.length>0,'at least one installed extension pack required');
  const definitions={}, owners={},packNames=new Set();
  for(const pack of packs){
   must(pack&&typeof pack.name==='string'&&/^[a-z][a-z0-9_]{1,35}$/.test(pack.name)&&!packNames.has(pack.name),'invalid or duplicate pack id');
   packNames.add(pack.name);
   must(pack.defs&&typeof pack.defs==='object'&&typeof pack.validate==='function'&&typeof pack.infer==='function'&&typeof pack.execute==='function','pack requires defs+validate+infer+execute');
   for(const [kind,def]of Object.entries(pack.defs)){
    must(/^[a-z][a-z0-9_]{2,45}$/.test(kind)&&!Object.hasOwn(definitions,kind),'invalid/duplicate registered block kind '+kind);
    must(def&&typeof def.title==='string'&&typeof def.en==='string'&&typeof def.category==='string'&&
     Array.isArray(def.ports)&&new Set(def.ports).size===def.ports.length&&def.ports.every(s=>/^[a-z][a-z0-9_]{0,24}$/.test(s))&&
     def.params&&typeof def.params==='object'&&!Array.isArray(def.params)&&typeof def.formula==='string'&&
     typeof def.why==='string'&&typeof def.lesson==='string'&&typeof def.caution==='string','invalid block metadata '+kind);
    must(typeof def.source==='string'&&/^https:\/\//.test(def.source),'extension requires real academic or source reference '+kind);
    owners[kind]=pack;definitions[kind]=Object.freeze({...def,ports:Object.freeze([...def.ports]),params:Object.freeze({...def.params}),pack:pack.name});
   }
  }
  function get(kind){must(typeof kind==='string'&&Object.hasOwn(owners,kind),'unknown extension kind '+kind);return owners[kind];}
  function validate(kind,params){const p=get(kind),raw={...definitions[kind].params,...params};p.validate(kind,raw);return raw;}
  function infer(node,inputs){
   const p=get(node.kind),params=validate(node.kind,node.params||{});
   const out=p.infer(node.kind,params,inputs||{});
   must(Array.isArray(out)&&out.length>0&&out.length<=5&&out.every(n=>Number.isSafeInteger(n)&&n>=1),'invalid symbolic output shape');
   must(out.reduce((a,b)=>a*b,1)<=4096,'output teaching budget exceeded');
   return Object.freeze({shape:out.slice(),rule:definitions[node.kind].formula,conditions:[],
    meaning:definitions[node.kind].why});
  }
  function execute(node,inputs){
   const p=get(node.kind),params=validate(node.kind,node.params||{}),operandShapes=Object.fromEntries(Object.entries(inputs||{}).map(([k,v])=>[k,tensorShape(v)]));
   const symbolic=infer(node,operandShapes).shape;
   const run=p.execute(node.kind,params,inputs||{});
   must(run&&run.values,'missing actual extension numeric result');
   const actual=tensorShape(run.values);
   must(actual.join(',')===symbolic.join(','),'numeric output disagrees with symbolic inferred shape: '+node.kind);
   must(run.internals==null||typeof run.internals==='object','invalid internal execution trace');
   return {values:run.values,shape:actual,flops:run.flops??null,internals:run.internals??null};
  }
  function substeps(node,inputs,run){
   const p=get(node.kind);if(typeof p.trace!=='function')return null;
   const data=p.trace(node.kind,validate(node.kind,node.params||{}),inputs,run);
   must(data==null||Array.isArray(data)&&data.length<=12,'invalid module microscope substeps');
   return data;
  }
  return Object.freeze({defs:Object.freeze(definitions),kinds:Object.freeze(Object.keys(definitions)),
   packs:Object.freeze([...packNames]),isKind:kind=>Object.hasOwn(owners,kind),
   validate,infer,execute,substeps,tensorShape});
 }
 const built=createRegistry(installed);
 return Object.freeze({...built,createRegistry});
});