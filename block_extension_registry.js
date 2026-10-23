/* Core block-extension contract. One installed pack contributes definitions, parameter
 * contracts, symbolic shapes, actual numerical execution and optional source-derived
 * operator substeps. Extension dispatch is kind-independent; no per-kind core switch. */
(function(root,factory){'use strict';
 const deps=typeof module==='object'&&module.exports
  ?[require('./semantic_block_pack.js'),require('./axis_block_pack.js')]
  :[root.SeetheTruthSemanticBlockPack,root.SeetheTruthAxisBlockPack];
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
  const definitions={}, owners={},packNames=new Set(),recipes={},recipeNames=new Set();
  const axisSignature=a=>Array.isArray(a)&&a.length>=1&&a.length<=5&&
   new Set(a).size===a.length&&a.every(x=>typeof x==='string'&&/^[A-Za-z][A-Za-z0-9_]{0,15}$/.test(x));
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
    must(def.outputAxes===undefined||axisSignature(def.outputAxes),'invalid output axis signature '+kind);
    must(def.portAxes===undefined||def.portAxes&&typeof def.portAxes==='object'&&
      Object.entries(def.portAxes).every(([port,axes])=>def.ports.includes(port)&&axisSignature(axes)),
      'invalid named input axes '+kind);
    owners[kind]=pack;definitions[kind]=Object.freeze({...def,ports:Object.freeze([...def.ports]),params:Object.freeze({...def.params}),
      ...(def.outputAxes?{outputAxes:Object.freeze([...def.outputAxes])}:{}),
      ...(def.portAxes?{portAxes:Object.freeze(Object.fromEntries(Object.entries(def.portAxes).map(([k,v])=>[k,Object.freeze([...v])])))}:{}),
      pack:pack.name});
   }
    if(pack.recipes!==undefined){
     must(Array.isArray(pack.recipes)&&pack.recipes.length<=24,'invalid pack recipes');
     for(const recipe of pack.recipes){
      must(recipe&&typeof recipe.id==='string'&&/^[a-z][a-z0-9_]{3,60}$/.test(recipe.id)&&!recipeNames.has(recipe.id),'duplicate or invalid recipe id');
      must(typeof recipe.title==='string'&&recipe.title.length>3&&typeof recipe.summary==='string'&&recipe.summary.length>12&&recipe.graph&&
       Array.isArray(recipe.graph.nodes)&&recipe.graph.nodes.length>0&&recipe.graph.nodes.length<=24&&Array.isArray(recipe.graph.edges)&&recipe.graph.edges.length<=48,
       'invalid registry recipe graph');
      recipeNames.add(recipe.id);
      recipes[recipe.id]={id:recipe.id,title:recipe.title,summary:recipe.summary,pack:pack.name,graph:recipe.graph};
     }
    }
  }
  function get(kind){must(typeof kind==='string'&&Object.hasOwn(owners,kind),'unknown extension kind '+kind);return owners[kind];}
  function validate(kind,params){const p=get(kind),raw={...definitions[kind].params,...params};p.validate(kind,raw);return raw;}
  function infer(node,inputs,inputAxes=null){
   const p=get(node.kind),params=validate(node.kind,node.params||{});
   const spec=definitions[node.kind];
   if(inputAxes!==null&&spec.portAxes){
    for(const [port,want]of Object.entries(spec.portAxes)){
     must(Array.isArray(inputAxes[port])&&JSON.stringify(inputAxes[port])===JSON.stringify(want),
      'named-axis port mismatch '+node.kind+'.'+port+': expected '+want.join(',')+' got '+(inputAxes[port]||[]).join(','));
    }
   }
   const out=p.infer(node.kind,params,inputs||{});
   must(Array.isArray(out)&&out.length>0&&out.length<=5&&out.every(n=>Number.isSafeInteger(n)&&n>=1),'invalid symbolic output shape');
   must(out.reduce((a,b)=>a*b,1)<=4096,'output teaching budget exceeded');
   must(!spec.outputAxes||spec.outputAxes.length===out.length,'output named-axis rank differs from inferred shape '+node.kind);
   return Object.freeze({shape:out.slice(),axes:spec.outputAxes?spec.outputAxes.slice():null,
    rule:definitions[node.kind].formula,conditions:[],
    meaning:definitions[node.kind].why});
  }
  function execute(node,inputs,records=null){
   const p=get(node.kind),params=validate(node.kind,node.params||{}),operandShapes=Object.fromEntries(Object.entries(inputs||{}).map(([k,v])=>[k,tensorShape(v)]));
   const required=definitions[node.kind].portAxes||{};
   if(Object.keys(required).length)must(records&&typeof records==='object',
    'named-axis execution requires typed upstream edge records: '+node.kind);
   const actualAxes=records?Object.fromEntries(Object.entries(records).map(([k,r])=>[k,r.axes||null])):null;
   const symbolic=infer(node,operandShapes,actualAxes).shape;
   const run=p.execute(node.kind,params,inputs||{});
   must(run&&run.values,'missing actual extension numeric result');
   const actual=tensorShape(run.values);
   must(actual.join(',')===symbolic.join(','),'numeric output disagrees with symbolic inferred shape: '+node.kind);
   must(run.internals==null||typeof run.internals==='object','invalid internal execution trace');
   return {values:run.values,shape:actual,axes:definitions[node.kind].outputAxes?definitions[node.kind].outputAxes.slice():null,
    flops:run.flops??null,internals:run.internals??null};
  }
  function substeps(node,inputs,run){
   const p=get(node.kind);if(typeof p.trace!=='function')return null;
   const data=p.trace(node.kind,validate(node.kind,node.params||{}),inputs,run,node);
   must(data==null||Array.isArray(data)&&data.length<=12,'invalid module microscope substeps');
   return data;
  }
  return Object.freeze({defs:Object.freeze(definitions),kinds:Object.freeze(Object.keys(definitions)),
   packs:Object.freeze([...packNames]),recipes:Object.freeze(Object.values(recipes).map(r=>Object.freeze({id:r.id,title:r.title,summary:r.summary,pack:r.pack}))),
   recipe:id=>Object.hasOwn(recipes,id)?JSON.parse(JSON.stringify(recipes[id].graph)):null,
   isKind:kind=>Object.hasOwn(owners,kind),
   validate,infer,execute,substeps,tensorShape});
 }
 const built=createRegistry(installed);
 return Object.freeze({...built,createRegistry});
});