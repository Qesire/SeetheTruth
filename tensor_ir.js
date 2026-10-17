/* Semantic Tensor IR for SeetheTruth Learning Studio.
 * Exact tiny Float64 numerical math, not model weights, GPU kernels or a compiler.
 * Elements and ranks are bounded; axis meaning travels alongside every tensor.
 */
(function(root,factory){
  'use strict';
  const lib=factory();
  if(typeof module==='object'&&module.exports)module.exports=lib;
  root.SeetheTruthTensorIR=lib;
})(typeof window!=='undefined'?window:{},function(){
  'use strict';
  const MAX_RANK=5,MAX_ELEMENTS=4096;
  const fail=s=>{throw new Error(s);};
  function integer(v,low,high,label){
    if(!Number.isSafeInteger(v)||v<low||v>high)fail(label+' must be integer in ['+low+','+high+']');
    return v;
  }
  function size(shape){return shape.reduce((a,b)=>a*b,1);}
  function make(shape,axes,data){
    if(!Array.isArray(shape)||shape.length<1||shape.length>MAX_RANK)fail('tensor rank exceeds semantic IR bounds');
    shape.forEach((v,i)=>integer(v,1,32,'dimension '+i));
    if(size(shape)>MAX_ELEMENTS)fail('tensor element limit exceeded');
    if(!Array.isArray(axes)||axes.length!==shape.length||new Set(axes).size!==axes.length||
        axes.some(x=>typeof x!=='string'||!/^[A-Za-z][A-Za-z0-9_]{0,15}$/.test(x)))
      fail('tensor axes must be distinct named semantic dimensions');
    if(!Array.isArray(data)||data.length!==size(shape)||data.some(x=>typeof x!=='number'||!Number.isFinite(x)))
      fail('tensor numerical payload must be finite Float64 and match shape');
    return {shape:shape.slice(),axes:axes.slice(),data:data.slice(),
      dtype:'float64',evidence:'deterministic_teaching_micro_tensor'};
  }
  function strides(shape){let st=Array(shape.length).fill(1);for(let i=shape.length-2;i>=0;i--)st[i]=st[i+1]*shape[i+1];return st;}
  function get(t,indices){const stride=strides(t.shape);if(indices.length!==stride.length)fail('index rank mismatch');
    let at=0;for(let i=0;i<indices.length;i++){if(!Number.isInteger(indices[i])||indices[i]<0||indices[i]>=t.shape[i])fail('index out of bounds');at+=indices[i]*stride[i];}return t.data[at];}
  function fromMatrix(m,axes=['T','D']){
    if(!Array.isArray(m)||!m.length||!Array.isArray(m[0])||!m[0].length||
       m.some(r=>!Array.isArray(r)||r.length!==m[0].length))fail('expected rectangular 2D matrix');
    return make([m.length,m[0].length],axes,m.flat());
  }
  function reshape(t,newShape,newAxes){if(size(newShape)!==size(t.shape))fail('reshape changes element count');
    return make(newShape,newAxes,t.data);}
  function transpose(t,order){
    if(!Array.isArray(order)||order.length!==t.shape.length||
       new Set(order).size!==order.length||order.some(i=>!Number.isInteger(i)||i<0||i>=t.shape.length))
      fail('transpose must be a bijection of axes');
    const shape=order.map(i=>t.shape[i]),axes=order.map(i=>t.axes[i]),result=Array(size(t.shape));
    const oldSt=strides(t.shape),newSt=strides(shape);
    for(let dest=0;dest<result.length;dest++){
      let rest=dest,old=0;
      for(let i=0;i<shape.length;i++){const idx=Math.floor(rest/newSt[i]);rest%=newSt[i];old+=idx*oldSt[order[i]];}
      result[dest]=t.data[old];
    }
    return make(shape,axes,result);
  }
  function headPack(matrix,heads,tokenAxis,headAxis,featureAxis){
    const m=fromMatrix(matrix,[tokenAxis,'features']);
    integer(heads,1,8,'head count');
    if(m.shape[1]%heads!==0)fail('projection width must divide number of heads');
    const packed=reshape(m,[m.shape[0],heads,m.shape[1]/heads],[tokenAxis,headAxis,featureAxis]);
    return reshape(transpose(packed,[1,0,2]),[1,heads,m.shape[0],m.shape[1]/heads],['B',headAxis,tokenAxis,featureAxis]);
  }
  function plane(t,fixed={}){
    if(t.shape.length<2)fail('matrix plane requires rank >= 2');
    const shape=t.shape,axes=t.axes,indices=Array(shape.length).fill(0);
    for(let i=0;i<shape.length-2;i++){
      const v=fixed[axes[i]]??0;integer(v,0,shape[i]-1,axes[i]+' selected index');indices[i]=v;
    }
    const m=Array.from({length:shape.at(-2)},(_,r)=>Array.from({length:shape.at(-1)},(_,c)=>{
      indices[shape.length-2]=r;indices[shape.length-1]=c;return get(t,indices);
    }));
    return {view:m,shape:shape.slice(),axes:axes.slice(),
      selected:Object.fromEntries(axes.slice(0,-2).map((a,i)=>[a,indices[i]]))};
  }
  function fromHeads(t){
    if(t.shape.length!==4||t.shape[0]!==1)fail('expected [B=1,H,T,D] tensor');
    const [_,h,tokens,dim]=t.shape;
    const rows=Array.from({length:tokens},(_,token)=>Array.from({length:h*dim},(_,col)=>
      get(t,[0,Math.floor(col/dim),token,col%dim])));
    return rows;
  }
  function rotary(t,positionOffset=0,theta=10000){
    if(t.shape.length!==4||t.shape.at(-1)%2!==0)fail('RoPE requires even Dk in [B,H,T,Dk]');
    integer(positionOffset,0,64,'RoPE position offset');
    const [B,H,T,D]=t.shape,data=t.data.slice();
    // Pair adjacent channels; base is dimension-indexed, same for Q and K.
    for(let b=0;b<B;b++)for(let h=0;h<H;h++)for(let token=0;token<T;token++)
      for(let i=0;i<D;i+=2){
        const angle=(token+positionOffset)/Math.pow(theta,i/D);
        const ca=Math.cos(angle),sa=Math.sin(angle);
        const at=((b*H+h)*T+token)*D+i;
        const x=t.data[at],y=t.data[at+1];
        data[at]=x*ca-y*sa;data[at+1]=x*sa+y*ca;
      }
    return make(t.shape,t.axes,data);
  }
  function validateMatrix(m,name){return fromMatrix(m).data.length;}
  function gqa(q,k,v,opts={}){
    validateMatrix(q,'Q');validateMatrix(k,'K');validateMatrix(v,'V');
    const heads=integer(opts.heads??2,1,8,'query heads');
    const kvHeads=integer(opts.kvHeads??1,1,8,'KV heads');
    if(heads%kvHeads!==0)fail('GQA query heads must be divisible by KV heads');
    if(k.length!==v.length)fail('GQA K/V token counts differ');
    if(q[0].length%heads||k[0].length%kvHeads||v[0].length%kvHeads)
      fail('Q/K/V projection widths not divisible by their own head counts');
    const Dk=q[0].length/heads,Dk2=k[0].length/kvHeads,Dv=v[0].length/kvHeads;
    if(Dk!==Dk2)fail('Q and K per-head Dk must match');
    const causal=opts.causal===true,rope=opts.rope===true,Tq=q.length,Tk=k.length;
    // A short query against longer K/V is a decode-like cached prefix. This
    // does NOT create or update a KV cache; K/V are already provided in full.
    if(causal&&Tk<Tq)fail('causal query sequence cannot exceed available key prefix');
    const offset=causal?Tk-Tq:0;
    const qHeads=headPack(q,heads,'Tq','Hq','Dk');
    const kHeads=headPack(k,kvHeads,'Tk','Hkv','Dk');
    const vHeads=headPack(v,kvHeads,'Tk','Hkv','Dv');
    const qr=rope?rotary(qHeads,offset):qHeads;
    const kr=rope?rotary(kHeads,0):kHeads;
    const score=[],masked=[],weights=[],context=[];
    const group=heads/kvHeads;
    for(let h=0;h<heads;h++){
      const kh=Math.floor(h/group);
      for(let tq=0;tq<Tq;tq++){
        const raw=[],allowed=[];
        for(let tk=0;tk<Tk;tk++){
          let dot=0;for(let d=0;d<Dk;d++)dot+=get(qr,[0,h,tq,d])*get(kr,[0,kh,tk,d]);
          raw.push(dot/Math.sqrt(Dk));
          allowed.push(!causal||tk<=offset+tq);
        }
        const max=Math.max(...raw.filter((_,j)=>allowed[j]));
        if(!Number.isFinite(max))fail('all causal attention positions masked');
        const exp=raw.map((s,j)=>allowed[j]?Math.exp(s-max):0);
        const denominator=exp.reduce((a,b)=>a+b,0);
        const probs=exp.map(x=>x/denominator);
        score.push(...raw);masked.push(...raw.map((x,j)=>allowed[j]?x:0));weights.push(...probs);
        for(let d=0;d<Dv;d++){
          let val=0;for(let tk=0;tk<Tk;tk++)val+=probs[tk]*get(vHeads,[0,kh,tk,d]);
          context.push(val);
        }
      }
    }
    const scores=make([1,heads,Tq,Tk],['B','Hq','Tq','Tk'],score);
    const maskedScores=make([1,heads,Tq,Tk],['B','Hq','Tq','Tk'],masked);
    const probabilities=make([1,heads,Tq,Tk],['B','Hq','Tq','Tk'],weights);
    const ctx=make([1,heads,Tq,Dv],['B','Hq','Tq','Dv'],context);
    const output=fromHeads(ctx);
    return {output,shape:[Tq,heads*Dv],knownMatmulFlops:2*heads*Tq*Tk*(Dk+Dv),
      teachingConfig:{heads,kvHeads,groupSize:group,causal,rope,queryPositionOffset:offset,
        scoreScaling:1/Math.sqrt(Dk),cachedPrefixProvidedAsInput:causal&&Tk>Tq,
        createsOrUpdatesKVCache:false},
      stages:{
        q_heads:qHeads,k_heads:kHeads,v_heads:vHeads,
        q_positioned:qr,k_positioned:kr,
        scores,masked_scores:maskedScores,probabilities,context_heads:ctx,
        merged_output:fromMatrix(output,['Tq','HqDv'])
      },
      masks:causal?{shape:[Tq,Tk],allowed:Array.from({length:Tq},(_,i)=>
        Array.from({length:Tk},(_,j)=>j<=offset+i))}:null};
  }
  return Object.freeze({MAX_ELEMENTS,make,size,get,fromMatrix,reshape,transpose,headPack,plane,fromHeads,rotary,gqa});
});
