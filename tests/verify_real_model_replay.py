#!/usr/bin/env python3
"""Compare TWO fresh, independently executed CPU traces from one pinned model.
This is reproducibility evidence, not a full independent audit of upstream code.
"""
import argparse, json, math, pathlib
def compare(a,b,atol=3e-5,rtol=2e-5):
    if isinstance(a,(int,float)) and not isinstance(a,bool):
        if not isinstance(b,(int,float)) or not math.isfinite(a) or not math.isfinite(b):
            raise AssertionError("non-finite replay numeric value")
        if not math.isclose(a,b,abs_tol=atol,rel_tol=rtol):
            raise AssertionError("replay discrepancy: %s vs %s"%(a,b))
        return abs(a-b)
    if isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b):
            raise AssertionError("replay list length differs")
        return max((compare(x,y,atol,rtol) for x,y in zip(a,b)),default=0.0)
    if a!=b:
        raise AssertionError("replay metadata mismatch: %r != %r"%(a,b))
    return 0.0
def verify(one,two):
    if one.get("schema")!="seethetruth-real-model-trace-v1":
        raise AssertionError("unexpected capture schema")
    a=one["provenance"];b=two["provenance"]
    if a["kind"]!="real_model_execution" or a["modelId"]!="roneneldan/TinyStories-1M" or \
        len(a["revision"])!=40 or a["revision"]!=b["revision"]:
        raise AssertionError("expected pinned real TinyStories CPU capture")
    fields=["modelId","revision","modulePath","moduleType","device","dtype","objective","promptSha256","tokenIds","torchVersion","transformersVersion"]
    for k in fields:
        if a.get(k)!=b.get(k):raise AssertionError("changed replay provenance field "+k)
    if one["parameter"]["identity"]!=a["modulePath"]+".weight":
        raise AssertionError("parameter identity mismatch")
    checked={}
    for name,path in [
       ("loss",["loss"]),
       ("weight",["parameter","weight"]),
       ("gradient",["parameter","gradient"]),
       ("input",["activation","input"]),
       ("output",["activation","output"])]:
        aa,bb=one,two
        for key in path:aa=aa[key];bb=bb[key]
        if isinstance(aa,dict):
            for field in ["shape","count","mean","rms","min","max","l2","zeroFraction","sample"]:
                if field not in aa or field not in bb:raise AssertionError("missing "+name+" "+field)
                checked[name+"."+field]=compare(aa[field],bb[field])
        else:checked[name]=compare(aa,bb)
    return {"schema":"seethetruth-real-model-repeat-v1","status":"TWO_INDEPENDENT_CPU_RUNS_NUMERICALLY_AGREE",
       "modelId":a["modelId"],"revision":a["revision"],"modulePath":a["modulePath"],
       "objective":a["objective"],"tokens":len(a["tokenIds"]),"loss":one["loss"],
       "maxAbsoluteReplayDifference":max(checked.values()),"perFieldMaxAbsoluteDifference":checked,
       "limitations":["CPU reproducibility at a fixed upstream SHA and software environment only",
         "not a full-model numerical equivalence proof","not GPU profiling","flattened sample is not a full tensor"]}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("first");p.add_argument("second");p.add_argument("--manifest",required=True)
    a=p.parse_args()
    one=json.loads(pathlib.Path(a.first).read_text()),json.loads(pathlib.Path(a.second).read_text())
    result=verify(*one)
    path=pathlib.Path(a.manifest);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print("REPEAT_REAL_MODEL_CPU_PASS",result["modelId"],result["revision"],"maxDiff",result["maxAbsoluteReplayDifference"])
if __name__=="__main__":main()
