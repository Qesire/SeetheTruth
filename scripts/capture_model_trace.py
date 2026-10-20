#!/usr/bin/env python3
"""Capture bounded *real model* activations, gradients and parameter statistics.
Usage:
 python scripts/capture_model_trace.py --model roneneldan/TinyStories-1M --revision <COMMIT_SHA> --out samples/tinystories.json
Requires locally available or network-accessible torch, transformers. CPU only.
No model-specific architecture assumptions; modules are discovered at runtime.
"""
import argparse, hashlib, json, math, pathlib, platform
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--revision",required=True,help="immutable Hugging Face git commit SHA")
    parser.add_argument("--out",required=True)
    parser.add_argument("--text",default="Once upon a time, a little robot")
    parser.add_argument("--max-tokens",type=int,default=24)
    parser.add_argument("--max-elements",type=int,default=64)
    a=parser.parse_args()
    if not (len(a.revision)==40 and all(c in "0123456789abcdef" for c in a.revision.lower())):
        parser.error("--revision must be a full immutable 40-char commit SHA")
    if not (1<=a.max_tokens<=64 and 1<=a.max_elements<=256):
        parser.error("sample bounds exceeded")
    import torch, transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.manual_seed(0)
    # Do not execute remotely supplied modeling code.
    tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,trust_remote_code=False).cpu().float().eval()
    ids=tok(a.text,return_tensors="pt",truncation=True,max_length=a.max_tokens)["input_ids"]
    if ids.shape[1]<2: raise ValueError("prompt must yield >=2 tokens")
    chosen=None
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Linear):
            chosen=(name,module)
            break
    if chosen is None: raise ValueError("model has no torch.nn.Linear layer; add an architecture-specific adapter")
    name,layer=chosen
    captures={}
    def fw(_m,inp,out):
        if inp and isinstance(inp[0],torch.Tensor) and isinstance(out,torch.Tensor):
            captures["input"]=inp[0].detach().float().flatten()[:a.max_elements].tolist()
            captures["output"]=out.detach().float().flatten()[:a.max_elements].tolist()
    handle=layer.register_forward_hook(fw)
    model.zero_grad(set_to_none=True)
    try:
        result=model(input_ids=ids,labels=ids)
        loss=result.loss
        if not torch.isfinite(loss):raise ValueError("nonfinite model loss")
        loss.backward()
    finally: handle.remove()
    grad=layer.weight.grad
    if grad is None:raise ValueError("selected linear weight did not receive a gradient")
    def summarize(t):
        v=t.detach().float().reshape(-1)
        if not torch.isfinite(v).all():raise ValueError("non-finite values")
        return {"shape":list(t.shape),"count":v.numel(),"mean":v.mean().item(),"rms":v.square().mean().sqrt().item(),
          "min":v.min().item(),"max":v.max().item(),"l2":v.norm().item(),"zeroFraction":(v==0).float().mean().item(),
          "sample":v[:a.max_elements].tolist()}
    weights=layer.weight.detach()
    trace={
      "schema":"seethetruth-real-model-trace-v1",
      "provenance":{"kind":"real_model_execution","modelId":a.model,"revision":a.revision,
        "framework":"pytorch","torchVersion":torch.__version__,"transformersVersion":transformers.__version__,
        "device":"cpu","dtype":"float32","seed":0,"modulePath":name,
        "moduleType":"torch.nn.Linear","objective":"causal_language_modeling_shifted_cross_entropy",
        "promptSha256":hashlib.sha256(a.text.encode()).hexdigest(),
        "tokenIds":ids[0].tolist(),"python":platform.python_version(),
        "capture":"single CPU forward+backward; no optimizer step"},
      "loss":loss.detach().item(),
      "parameter":{"identity":name+".weight","weight":summarize(weights),"gradient":summarize(grad)},
      "activation":{"input":captures.get("input"),"output":captures.get("output")},
      "limitations":["Flattened samples preserve value order but not full tensor topology",
        "Truncated parameter samples do not represent complete GEMM operands",
        "Activation captured at model module runtime, not synthetic or kernel benchmark",
        "Numerical trace cannot be replayed as an equivalent full forward using truncated tensors"]
    }
    if trace["activation"]["input"] is None:raise ValueError("module was not executed")
    p=pathlib.Path(a.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(trace,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(f"captured {name}, tokens={ids.numel()}, loss={trace['loss']:.6f}, out={p}")
if __name__=="__main__":main()
