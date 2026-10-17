#!/usr/bin/env python3
"""SeetheTruth static Python AST graph compiler. No import/exec of analyzed code."""
from __future__ import annotations
import ast, argparse, json, re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

MAX_NODES=800
MAX_EDGES=3200
MAX_FILES=800

@dataclass
class Def:
    path:Path
    cls:str|None
    name:str
    tree:ast.AST

def expr(n):
    try:return ast.unparse(n)
    except Exception:return type(n).__name__ if n else ""

SEMANTICS={
"linear":("MatMul","Affine/linear projection","Y=XW^T+b","Weight, bias and output dimension must be read from constructor."),
"matmul":("MatMul","Contract tensor axes","C=A@B","The contracted axes are not known without real shapes."),
"bmm":("MatMul","Batched matrix multiplication","C_i=A_i@B_i","Batch and contracted axes must match."),
"mm":("MatMul","Matrix multiplication","C=A@B","May differ from batched matmul."),
"einsum":("MatMul","Index-labelled tensor contraction","Y=einsum(spec,tensors)","Read the actual subscript pattern."),
"addmm":("MatMul","Add a matrix product","Y=beta*X+alpha*(A@B)","May fuse add and GEMM."),
"baddbmm":("MatMul","Batched affine matmul","Y=beta*X+alpha*bmm(A,B)","Read alpha/beta and batch dimensions; not an independent forward GEMM count if a fused kernel."),
"reshape":("Shape","Reshape logical axes","reshape(X,new_shape)","Element count conserved; physical copy may occur."),
"view":("Shape","View tensor with requested strides/layout","view(X,new_shape)","Requires stride-compatible layout."),
"flatten":("Shape","Coalesce selected axes","flatten(X,start,end)","Check which axes are collapsed."),
"unflatten":("Shape","Expand an axis into multiple axes","unflatten(X,dim,sizes)","Product of new sizes must match."),
"transpose":("Shape","Swap two axes","transpose(X,i,j)","Usually a stride-only view."),
"permute":("Shape","Reorder axes","permute(X,axes)","May become non-contiguous."),
"movedim":("Shape","Move tensor axes","movedim(X,source,dest)","Axis order changes."),
"contiguous":("Layout","Request a contiguous memory layout","Y=contiguous(X)","May copy the tensor."),
"unsqueeze":("Shape","Insert singleton broadcast axis","unsqueeze(X,dim)","Does not change element count."),
"squeeze":("Shape","Remove size-one axis","squeeze(X,dim)","Must verify actual axis size."),
"expand":("Broadcast","Logical broadcast without copying values","expand(X,new_shape)","Expanded strides may be zero."),
"repeat":("Broadcast","Materialize repeated tensor content","repeat(X,repeats)","Can allocate new memory."),
"repeat_interleave":("Broadcast","Repeat slices along an axis","repeat_interleave(X,repeat)","Potential GQA materialization."),
"broadcast_to":("Broadcast","Broadcast shape-compatible axes","broadcast_to(X,shape)","Logical broadcast may be a view."),
"cat":("Concat","Concatenate tensors along a chosen axis","cat(xs,dim)","Output size sums along concat axis."),
"concat":("Concat","Concatenate tensors along a chosen axis","concat(xs,dim)","Output size sums along concat axis."),
"stack":("Concat","Insert axis and stack tensors","stack(xs,dim)","Requires equal input shapes."),
"split":("Split","Partition a tensor along an axis","split(X,sections,dim)","Multiple output tensors, possibly views."),
"chunk":("Split","Split a tensor into chunks","chunk(X,chunks,dim)","Last chunk may have a different length."),
"gather":("Index","Read values at tensor indices","gather(X,dim,index)","Index shape determines output."),
"scatter":("Index","Write values into selected coordinates","scatter(X,dim,index,source)","Could be in-place or not."),
"masked_fill":("Mask","Overwrite masked tensor values","masked_fill(X,mask,value)","Broadcast mask shape and numeric dtype matter."),
"where":("Mask","Conditionally select tensor elements","where(mask,a,b)","All operands use broadcasting semantics."),
"triu":("Mask","Keep upper-triangular portion","triu(X,k)","Causal meaning depends on whether mask or scores are transformed."),
"tril":("Mask","Keep lower-triangular portion","tril(X,k)","Observe masking convention."),
"softmax":("Softmax","Normalize logits along one axis","A=exp(S)/sum(exp(S))","Softmax axis and mask handling must be checked."),
"scaled_dot_product_attention":("Fused","Fused attention logical equivalent","O=softmax(QK^T/sqrt(d)+M)V","A backend may not materialize Scores, A, or separate GEMMs."),
"flash_attn_func":("Fused","Fused attention implementation","O=FlashAttention(Q,K,V)","Physical kernels differ from logical operator decomposition."),
"apply_rotary_pos_emb":("RoPE","Apply position-conditioned feature rotations","Q'=R(pos)Q; K'=R(pos)K","Must inspect positions, partial dims and order."),
"rotary_emb":("RoPE","Rotary position encoding module/call","x'=R(pos)x","Name-based interpretation; inspect implementation."),
"embedding":("Index","Embedding table lookup","X=Table[token]","Lookup is not a dense GEMM."),

"layernorm":("Norm","LayerNorm via module constructor","Y=(X-mean)/sqrt(var+eps)*gamma+beta","Check normalized_shape and epsilon."),
"rmsnorm":("Norm","RMSNorm via module constructor","Y=X/sqrt(mean(X²)+eps)*gamma","Check epsilon and axis."),
"log_softmax":("Softmax","Stable log probability normalization","logp_i=z_i-logsumexp(z)","Token and sequence reduction must be separated."),
"log_prob":("Loss","Evaluate sample log probability","log p_theta(action|state)","Check stochastic policy distribution and reduction dimensions."),
"mean":("Pointwise","Average over specified axis","mean(X,dim)","Check dim and reduction semantics."),
"std":("Pointwise","Standard deviation along specified axis","std(X,dim)","Zero variance and correction affect GRPO group advantage."),
"exp":("Pointwise","Elementwise exponent","exp(X)","In policy ratios, clamp/overflow need attention."),
"clamp":("Pointwise","Clip into interval","min(max(X,lo),hi)","PPO/GRPO clipped objectives must check actual bounds."),
"minimum":("Pointwise","Elementwise minimum","min(A,B)","May implement clipped-policy conservative surrogate."),
"detach":("Grad","Stop gradient along an edge","stop_gradient(X)","Value can still flow while Jacobian is zero."),
"no_grad":("Grad","Disable tracking within context","torch.no_grad()","The actual context manager execution is runtime-dependent."),
"layer_norm":("Norm","Center and scale hidden features","(X-mean)/sqrt(var+eps)*gamma+beta","Check normalized axes and dtype."),
"rms_norm":("Norm","RMS normalization","X/sqrt(mean(X^2)+eps)*gamma","Unlike LayerNorm, no mean subtraction."),
"silu":("Pointwise","Smooth nonlinear gate","SiLU(x)=x*sigmoid(x)","Used in SwiGLU but does not alone imply SwiGLU."),
"gelu":("Pointwise","Gaussian error linear unit","GELU(x)=x*Phi(x)","Check approximate variant."),
"topk":("Route","Select largest scored candidates","topk(X,k)","Top-k does not alone prove expert routing."),
"cross_entropy":("Loss","Categorical NLL objective","L=-sum(y*log_softmax(logits))","Target shift, reduction and mask must be verified."),
"mse_loss":("Loss","Squared error objective","L=mean((pred-target)^2)","Does not establish diffusion parameterization."),
"backward":("Grad","Trigger autodiff graph traversal","g=∂L/∂theta","Does not prove all parameters are trainable."),
"zero_grad":("Grad","Clear accumulated parameter gradients","grad=0 or None","Receiver and optimizer semantics matter."),
"step":("Update","Advance optimizer/scheduler/environment","state'=step(state)","Need receiver type to know which state changes.")
}
TYPE={
"MatMul":"attn","Shape":"norm","Layout":"norm","Broadcast":"cond","Concat":"cond","Split":"cond",
"Mask":"gate","Softmax":"attn","Fused":"attn","RoPE":"cond","Index":"input","Norm":"norm",
"Pointwise":"gate","Route":"router","Loss":"head","Grad":"cond","Update":"head","Call":"ffn"
}
def rule(name):
    key=name.lower().split(".")[-1]
    key=re.sub(r"_?forward$","",key) if key.endswith("_forward") else key
    return SEMANTICS.get(key,("Call","Python function/module call","y=call(inputs)","Cannot infer numerical mapping from call syntax alone."))

class Index:
    def __init__(self,root):
        self.root=root;self.defs={};self.modules=defaultdict(dict);self.errors=[];self.count=0
    def load(self,path):
        files=sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            if self.count>=MAX_FILES:self.errors.append("file cap reached");break
            if any(v in file.parts for v in (".git","__pycache__",".venv")):continue
            if file.is_symlink():
                self.errors.append(str(file)+": symlink skipped")
                continue
            try:
                if file.stat().st_size>2_000_000:self.errors.append(str(file)+": skipped >2MB");continue
                tree=ast.parse(file.read_text(encoding="utf8"),filename=str(file))
            except (SyntaxError,UnicodeError,OSError) as e:self.errors.append(str(file)+": "+str(e));continue
            self.count+=1
            for item in tree.body:
                if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    self.defs.setdefault(item.name,Def(file,None,item.name,item))
                elif isinstance(item,ast.ClassDef):
                    for f in item.body:
                        if not isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
                        key=item.name+"."+f.name
                        if key in self.defs:
                            self.errors.append("duplicate class method "+key);continue
                        self.defs[key]=Def(file,item.name,f.name,f)
                        if f.name=="__init__":
                            for n in ast.walk(f):
                                if not isinstance(n,(ast.Assign,ast.AnnAssign)):continue
                                targets=n.targets if isinstance(n,ast.Assign) else [n.target]
                                for target in targets:
                                    if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=="self" and isinstance(n.value,ast.Call):
                                        self.modules[item.name][target.attr]=expr(n.value.func).split(".")[-1]
        if not self.defs:raise ValueError("No Python functions found")
    def choose(self,entry):
        if entry:
            matches=[k for k in self.defs if k==entry or k.endswith("."+entry)]
            if entry in self.defs:return entry
            if len(matches)==1:return matches[0]
            raise ValueError("Entry is ambiguous/missing: "+str(entry))
        forwards=[k for k in self.defs if k.endswith(".forward")]
        return max(forwards or list(self.defs),key=lambda k:len(list(ast.walk(self.defs[k].tree))))

class Graph:
    def __init__(self,index,entry,depth):
        self.i=index;self.entry=entry;self.depth=depth;self.nodes=[];self.edges=[];self.last_effect=None
        self.unresolved=set();self.scopes=[];self.branches=0;self.loops=0;self.truncated=False
    def add(self,label,kind,purpose,formula,why,fn,at,parent,certainty,**extra):
        if len(self.nodes)>=MAX_NODES:self.truncated=True;return None
        ident="n%03d"%(len(self.nodes)+1)
        n={"id":ident,"label":label[:85],"type":TYPE.get(kind,"ffn"),
           "purpose":purpose,"formula":formula,
           "input":"未由静态 AST 确定","output":"未由静态 AST 确定","why":why,
           "train":"Training path requires actual gradient/runtime evidence.",
           "infer":"Runtime branch and kernel selection are not statically verified.",
           "probe":"Inspect source parameters, then validate shapes/strides with an opt-in trace.",
           "source":{"file":str(fn.path.relative_to(self.i.root)) if fn.path.is_relative_to(self.i.root) else fn.path.name,
                     "line":getattr(at,"lineno",getattr(fn.tree,"lineno",1))},
           "certainty":certainty,"parent":parent,"expression":expr(at)[:320],**extra}
        self.nodes.append(n);return ident
    def edge(self,a,b,kind="data"):
        if not a or not b or a==b:return
        e={"from":a,"to":b,"kind":kind}
        if e not in self.edges and len(self.edges)<MAX_EDGES:self.edges.append(e)
    def read(self,ex,fn,parent,env,guard,depth,seen):
        if ex is None:return []
        if isinstance(ex,ast.Name):return [env[ex.id]] if ex.id in env else []
        if isinstance(ex,ast.Call):
            incoming=[]
            if isinstance(ex.func,ast.Attribute):incoming+=self.read(ex.func.value,fn,parent,env,guard,depth,seen)
            for a in ex.args:incoming+=self.read(a,fn,parent,env,guard,depth,seen)
            for a in ex.keywords:incoming+=self.read(a.value,fn,parent,env,guard,depth,seen)
            callee=expr(ex.func);kind,purpose,formula,why=rule(callee)
            constructor=None
            if callee.startswith("self.") and fn.cls:
                key=callee.split(".")[1].split("[")[0]
                constructor=self.i.modules[fn.cls].get(key)
                if constructor and constructor.lower() in SEMANTICS:
                    kind,purpose,formula,why=rule(constructor)
            certainty="constructor_and_semantic_rule" if constructor and constructor.lower() in SEMANTICS else "semantic_rule" if kind!="Call" else "source_call"
            label=callee.split(".")[-1]+"()"
            if constructor:label=callee[5:]+" : "+constructor
            n=self.add(label,kind,purpose,formula,why,fn,ex,parent,certainty,callee=callee,module_constructor=constructor,condition=guard)
            for a in incoming:self.edge(a,n)
            if callee.endswith((".backward",".step",".zero_grad")):
                self.edge(self.last_effect,n);self.last_effect=n
            nested=None
            if callee.startswith("self.") and constructor and constructor+".forward" in self.i.defs:
                nested=constructor+".forward"
            elif callee.startswith("self.") and fn.cls:
                method=fn.cls+"."+callee.split(".")[1]
                if method in self.i.defs:nested=method
            elif callee in self.i.defs:nested=callee
            if nested and nested not in seen and depth<self.depth and n:
                self.function(nested,n,depth+1,seen|{nested})
            elif nested in seen if nested else False:self.unresolved.add("recursive: "+str(nested))
            elif callee.startswith("self.") and fn.cls and not constructor and not nested:
                self.unresolved.add("unresolved self-call: "+fn.cls+"."+callee.split(".")[1])
            return [n] if n else []
        if isinstance(ex,ast.IfExp):
            condition=self.read(ex.test,fn,parent,env,guard,depth,seen)
            truth=self.read(ex.body,fn,parent,env,expr(ex.test),depth,seen)
            false=self.read(ex.orelse,fn,parent,env,"not ("+expr(ex.test)+")",depth,seen)
            ident=self.add("φ conditional","Call","Conditional expression result","a if condition else b",
                           "Analysis-only branch union; neither expression is proven executed.",
                           fn,ex,parent,"analysis_construct",condition=expr(ex.test))
            for dep in [*condition,*truth,*false]:self.edge(dep,ident)
            return [ident] if ident else []
        if isinstance(ex,ast.BinOp):
            inputs=self.read(ex.left,fn,parent,env,guard,depth,seen)+self.read(ex.right,fn,parent,env,guard,depth,seen)
            name=type(ex.op).__name__;kind="MatMul" if name=="MatMult" else "Pointwise"
            formula="Z=A@B" if name=="MatMult" else expr(ex)[:150]
            n=self.add(name,kind,"AST binary "+name,formula,"Broadcast and dtype require runtime validation.",fn,ex,parent,"source_operator")
            for p in inputs:self.edge(p,n)
            return [n] if n else []
        if isinstance(ex,ast.Attribute):return self.read(ex.value,fn,parent,env,guard,depth,seen)
        if isinstance(ex,ast.Subscript):return self.read(ex.value,fn,parent,env,guard,depth,seen)+self.read(ex.slice,fn,parent,env,guard,depth,seen)
        if isinstance(ex,(ast.List,ast.Tuple,ast.Set)):return list(dict.fromkeys(sum((self.read(x,fn,parent,env,guard,depth,seen) for x in ex.elts),[])))
        if isinstance(ex,ast.Dict):return list(dict.fromkeys(sum((self.read(x,fn,parent,env,guard,depth,seen) for x in [*ex.keys,*ex.values] if x),[])))
        out=[]
        for ch in ast.iter_child_nodes(ex):out+=self.read(ch,fn,parent,env,guard,depth,seen)
        return list(dict.fromkeys(out))
    def statements(self,seq,fn,parent,env,guard,depth,seen):
        for st in seq:
            if self.truncated:break
            if isinstance(st,(ast.Assign,ast.AnnAssign)):
                values=self.read(st.value,fn,parent,env,guard,depth,seen)
                targets=st.targets if isinstance(st,ast.Assign) else [st.target]
                for target in targets:
                    if isinstance(target,ast.Name) and values:env[target.id]=values[-1]
                    elif isinstance(target,(ast.Tuple,ast.List)):
                        for t in target.elts:
                            if isinstance(t,ast.Name) and values:env[t.id]=values[-1]
            elif isinstance(st,ast.AugAssign):
                lhs=self.read(st.target,fn,parent,env,guard,depth,seen)
                rhs=self.read(st.value,fn,parent,env,guard,depth,seen)
                opname=type(st.op).__name__
                op=self.add(opname+"=","MatMul" if isinstance(st.op,ast.MatMult) else "Pointwise",
                            "Augmented assignment operation",expr(st),
                            "Source-level augmented assignment; potential in-place mutation or reassignment depends on operand implementation.",
                            fn,st,parent,"source_operator",condition=guard)
                for dep in [*lhs,*rhs]:self.edge(dep,op)
                if isinstance(st.target,ast.Name) and op:env[st.target.id]=op
                elif isinstance(st.target,(ast.Subscript,ast.Attribute)):
                    self.unresolved.add("in-place alias effects not resolved at "+str(st.lineno))
            elif isinstance(st,ast.Return):
                sources=self.read(st.value,fn,parent,env,guard,depth,seen)
                n=self.add("Return","Call","Function return","return value","Actual return branch is runtime-dependent.",fn,st,parent,"source_return")
                for p in sources:self.edge(p,n)
            elif isinstance(st,ast.Expr):self.read(st.value,fn,parent,env,guard,depth,seen)
            elif isinstance(st,ast.If):
                self.branches+=1;condition=expr(st.test)
                n=self.add("if "+condition[:40],"Call","Control-flow predicate","if "+condition,"Both possible paths are shown, neither is claimed executed.",fn,st,parent,"source_control")
                for d in self.read(st.test,fn,parent,env,guard,depth,seen):self.edge(d,n)
                left=dict(env);right=dict(env)
                self.statements(st.body,fn,parent,left,condition,depth,seen)
                self.statements(st.orelse,fn,parent,right,"not ("+condition+")",depth,seen)
                for variable in sorted(set(left)|set(right)):
                    a,b=left.get(variable),right.get(variable)
                    if a and b and a!=b:
                        p=self.add("φ "+variable,"Call","Static branch merge","φ(then,else)","Analysis-only join, not a real operator.",fn,st,parent,"analysis_construct")
                        self.edge(a,p);self.edge(b,p)
                        if p:env[variable]=p
                    elif a or b:
                        p=self.add("φ? "+variable,"Call","Conditionally defined variable","φ(defined,undefined)","Analysis-only uncertain join. Not defined on both branches; runtime may raise UnboundLocalError.",fn,st,parent,"analysis_construct",conditional_defined=True)
                        self.edge(a or b,p)
                        if p:env[variable]=p
            elif isinstance(st,(ast.For,ast.While,ast.AsyncFor)):
                self.loops+=1
                n=self.add("Loop "+expr(st.iter if isinstance(st,(ast.For,ast.AsyncFor)) else st.test)[:40],"Call","Loop scope","iteration","Exact iteration counts not determined.",fn,st,parent,"source_control")
                first=len(self.nodes)
                self.statements(st.body,fn,parent,env,"loop",depth,seen)
                body=[x for x in self.nodes[first:] if x["parent"]==parent]
                if body:self.edge(n,body[0]["id"]);self.edge(body[-1]["id"],n,"feedback")
            elif isinstance(st,(ast.With,ast.AsyncWith)):
                for context in st.items:self.read(context.context_expr,fn,parent,env,guard,depth,seen)
                self.statements(st.body,fn,parent,env,guard,depth,seen)
            elif isinstance(st,ast.Try):
                self.statements(st.body,fn,parent,env,guard,depth,seen)
                for h in st.handlers:self.statements(h.body,fn,parent,dict(env),"exception",depth,seen)
                self.statements(st.finalbody,fn,parent,env,guard,depth,seen)
            else:self.unresolved.add(type(st).__name__+" at "+str(getattr(st,"lineno","?")))
    def function(self,key,parent,depth,seen):
        fn=self.i.defs[key];self.scopes.append(key);env={}
        params=[*fn.tree.args.posonlyargs,*fn.tree.args.args,*fn.tree.args.kwonlyargs]
        for p in params:
            if p.arg in ("self","cls"):continue
            n=self.add(p.arg,"Index","Function parameter",p.arg,"Input shape is not verified.",fn,p,parent,"source_parameter")
            if n:env[p.arg]=n
        self.statements(fn.tree.body,fn,parent,env,None,depth,seen)
    def build(self):
        self.function(self.entry,None,0,{self.entry})
        return {"schema":"model-atlas-graph-v1","name":"SOURCE · "+self.entry,
          "meta":{"analysis":"static-python-ast","entry":self.entry,"entry_file":str(self.i.defs[self.entry].path.name),
            "source_files":self.i.count,"nested_scopes":self.scopes,"conditional_branches":self.branches,
            "loops":self.loops,"unresolved_calls_and_limits":sorted(self.unresolved),
            "parse_errors":self.i.errors,"truncated":self.truncated,"runtime_validated":False,"shape_verified":False},
          "nodes":self.nodes,"edges":self.edges}
def parse_paths(paths:list[str],entry:str|None=None,max_depth:int=3)->dict[str,Any]:
    resolved=[Path(s).resolve() for s in paths]
    for p in resolved:
        if not p.exists():raise FileNotFoundError(p)
    base=resolved[0] if resolved[0].is_dir() else resolved[0].parent
    idx=Index(base)
    for p in resolved:idx.load(p)
    chosen=idx.choose(entry)
    graph=Graph(idx,chosen,max(0,min(5,max_depth))).build()
    graph["meta"]["entry_selection"]="explicit" if entry else "automatic_heuristic"
    return graph
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("sources",nargs="+")
    ap.add_argument("--entry")
    ap.add_argument("--out","-o",required=True)
    ap.add_argument("--max-depth",type=int,default=3)
    a=ap.parse_args()
    graph=parse_paths(a.sources,a.entry,a.max_depth)
    Path(a.out).write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding="utf8")
    print(graph["name"],len(graph["nodes"]),"nodes",len(graph["edges"]),"edges")
