#!/usr/bin/env python3
"""Local-only SeetheTruth source analyzer: no model imports, weight loading or execution."""
from __future__ import annotations
import argparse, json, os, re, subprocess, tempfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
from atlas_parser import parse_paths

ROOT=Path(__file__).resolve().parent
MAX_BODY=65536
MAX_RESPONSE=5_000_000
HOST="127.0.0.1"

def normalize_repo_url(url:str)->str:
    try:u=urlparse(url.strip())
    except ValueError:raise ValueError("Invalid URL") from None
    if u.scheme!="https" or u.port not in (None,443) or u.username or u.password or u.query or u.fragment:
        raise ValueError("Only plain HTTPS GitHub/Hugging Face repository URLs are supported")
    if u.hostname not in ("github.com","huggingface.co"):
        raise ValueError("Repository host must be github.com or huggingface.co")
    parts=[p for p in u.path.split("/") if p]
    if len(parts)<2 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+",x) for x in parts[:2]):
        raise ValueError("Expected owner/repository")
    repo=parts[1].removesuffix(".git")
    if repo in ("","..") or parts[0] in (".",".."):raise ValueError("Invalid repository")
    return "https://"+u.hostname+"/"+parts[0]+"/"+repo

def sparse_clone(url:str,directory:Path):
    # No arbitrary remotes, shell interpolation, submodules, LFS smudge, hooks or source execution.
    clean=normalize_repo_url(url)
    env={**os.environ,"GIT_TERMINAL_PROMPT":"0","GIT_LFS_SKIP_SMUDGE":"1",
         "GIT_CONFIG_NOSYSTEM":"1","GIT_CONFIG_GLOBAL":os.devnull}
    base=["git","-c","protocol.file.allow=never","-c","core.hooksPath="+os.devnull]
    def run(args,timeout=45):
        r=subprocess.run(base+args,cwd=directory if directory.exists() else None,
                         env=env,capture_output=True,text=True,timeout=timeout,check=False)
        if r.returncode!=0:raise ValueError("git failed: "+r.stderr[-500:])
    work=directory/"checkout"
    run(["clone","--no-checkout","--depth","1","--filter=blob:none","--sparse",clean,str(work)],60)
    r=subprocess.run(base+["sparse-checkout","set","--no-cone","*.py","**/*.py"],
        cwd=work,env=env,capture_output=True,text=True,timeout=20,check=False)
    if r.returncode:raise ValueError("Could not configure Python-only checkout: "+r.stderr[-300:])
    r=subprocess.run(base+["checkout"],cwd=work,env=env,capture_output=True,text=True,timeout=60,check=False)
    if r.returncode:raise ValueError("Checkout failed: "+r.stderr[-300:])
    return work

class Handler(BaseHTTPRequestHandler):
    server_version="SeetheTruth/0.2"
    def emit(self,code,obj):
        data=json.dumps(obj,ensure_ascii=False).encode("utf8")
        if len(data)>MAX_RESPONSE:
            code,data=413,b'{"error":"Analysis result exceeds size limit"}'
        self.send_response(code)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(data)))
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.send_header("Access-Control-Allow-Origin","null" if False else "")
        self.end_headers();self.wfile.write(data)
    def do_GET(self):
        path=self.path.split("?")[0]
        if path=="/healthz":return self.emit(200,{"ok":True,"mode":"static-source"})
        mapping={"/":"index.html","/teaching.html":"teaching.html","/teaching_curriculum.js":"teaching_curriculum.js","/teaching_hub.js":"teaching_hub.js","/mobile_teaching.css":"mobile_teaching.css","/teaching_edition.js":"teaching_edition.js","/index.html":"index.html","/tensor_subgraph.js":"tensor_subgraph.js","/source_evidence_levels.js":"source_evidence_levels.js","/consensus_knowledge.js":"consensus_knowledge.js","/telemetry.html":"telemetry.html","/semantic_state.html":"semantic_state.html","/semantic_state.js":"semantic_state.js","/stream_graph_blocks.js":"stream_graph_blocks.js","/semantic_state_lab.js":"semantic_state_lab.js","/capabilities.html":"capabilities.html","/block_capabilities.js":"block_capabilities.js","/operator_contracts.js":"operator_contracts.js","/execution_ir.js":"execution_ir.js","/graph_session.js":"graph_session.js","/capabilities_page.js":"capabilities_page.js","/learn.html":"learn.html","/tradeoffs.html":"tradeoffs.html","/architecture_tradeoffs.js":"architecture_tradeoffs.js","/learning_lab.js":"learning_lab.js","/loop_transformer.js":"loop_transformer.js","/loop_input_injection.js":"loop_input_injection.js","/adaptive_token_depth.js":"adaptive_token_depth.js","/loop_consistency.js":"loop_consistency.js","/loop_design_tradeoffs.js":"loop_design_tradeoffs.js","/architecture_why.js":"architecture_why.js","/training_objective.js":"training_objective.js","/tensor_ir.js":"tensor_ir.js","/cache.html":"cache.html","/kv_cache_lab.js":"kv_cache_lab.js","/multimodal_ops.js":"multimodal_ops.js","/multimodal_advanced.js":"multimodal_advanced.js","/video_causal_conv3d.js":"video_causal_conv3d.js","/inference_ops.js":"inference_ops.js","/deepinfer_runtime_ops.js":"deepinfer_runtime_ops.js","/deepinfer_worker_lab.html":"deepinfer_worker_lab.html","/deepinfer_worker_events.js":"deepinfer_worker_events.js","/deepinfer_zos_metadata.js":"deepinfer_zos_metadata.js","/deepinfer_decode_metadata.js":"deepinfer_decode_metadata.js","/deepinfer_kv_lifecycle.js":"deepinfer_kv_lifecycle.js","/deepinfer_kv_lifecycle_lab.js":"deepinfer_kv_lifecycle_lab.js","/deepinfer_kv_lifecycle_lab.html":"deepinfer_kv_lifecycle_lab.html","/deepinfer_scheduler_workflow.js":"deepinfer_scheduler_workflow.js","/deepinfer_scheduler_lab.js":"deepinfer_scheduler_lab.js","/deepinfer_scheduler_lab.html":"deepinfer_scheduler_lab.html","/deepinfer_prefill_lab.html":"deepinfer_prefill_lab.html","/deepinfer_prefill_lab.js":"deepinfer_prefill_lab.js","/deepinfer_prefill_policy.js":"deepinfer_prefill_policy.js","/android_teaching.html":"android_teaching.html","/android_teaching_mode.js":"android_teaching_mode.js","/deepinfer_worker_lab.js":"deepinfer_worker_lab.js","/block_catalog.js":"block_catalog.js","/activation_math.js":"activation_math.js","/nd_tensor.js":"nd_tensor.js","/block_knowledge.js":"block_knowledge.js","/composite_atlas.js":"composite_atlas.js","/training.html":"training.html","/training_flow.js":"training_flow.js","/concept_flow.js":"concept_flow.js","/paper_hover.js":"paper_hover.js","/industrial_source_inspector.js":"industrial_source_inspector.js","/paper_graph_rewrites.js":"paper_graph_rewrites.js","/graph_layout.js":"graph_layout.js","/grpo_backward.js":"grpo_backward.js","/graph_autodiff.js":"graph_autodiff.js","/nd_autodiff.js":"nd_autodiff.js","/graph_readiness.js":"graph_readiness.js","/graph_cost_report.js":"graph_cost_report.js","/numpy_stream_lowering.js":"numpy_stream_lowering.js","/python_codegen.js":"python_codegen.js","/torch_stream_lowering.js":"torch_stream_lowering.js","/torch_codegen.js":"torch_codegen.js","/branch_execution.js":"branch_execution.js","/expert_ffn_vjp.js":"expert_ffn_vjp.js","/adamw_micro.js":"adamw_micro.js","/nd_adamw_micro.js":"nd_adamw_micro.js","/train_state.js":"train_state.js","/optimizer_rules.js":"optimizer_rules.js","/paper_lab.html":"paper_lab.html","/paper_patches.js":"paper_patches.js","/micro_training.html":"micro_training.html","/micro_training_flows.js":"micro_training_flows.js","/model_trace.js":"model_trace.js","/model_trace_bridge.js":"model_trace_bridge.js","/real_model_lab.html":"real_model_lab.html"}
        if path not in mapping:
            self.send_error(404,"Not found");return
        target=ROOT/mapping[path]
        if not target.is_file():
            self.send_error(404,"Missing file");return
        body=target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type","application/javascript; charset=utf-8" if path.endswith(".js") else "text/css; charset=utf-8" if path.endswith(".css") else "text/html; charset=utf-8")
        self.send_header("Content-Length",str(len(body)))
        self.send_header("X-Content-Type-Options","nosniff")
        self.end_headers();self.wfile.write(body)
    def do_POST(self):
        if self.path!="/api/analyze":return self.emit(404,{"error":"Unknown endpoint"})
        origin=self.headers.get("Origin")
        valid=("http://127.0.0.1:"+str(self.server.server_port),
               "http://localhost:"+str(self.server.server_port))
        if origin and origin not in valid:return self.emit(403,{"error":"Unsupported origin"})
        try:
            n=int(self.headers.get("Content-Length","0"))
            if n<2 or n>MAX_BODY:raise ValueError("Request body must be a small JSON object")
            doc=json.loads(self.rfile.read(n))
            if not isinstance(doc,dict):raise ValueError("Request must be a JSON object")
            source=str(doc.get("source") or doc.get("path") or doc.get("url") or "").strip()
            if not source or len(source)>2048:raise ValueError("Provide a source directory, .py path or public repo URL")
            entry=doc.get("entry")
            if entry is not None and (not isinstance(entry,str) or len(entry)>200):raise ValueError("Invalid entry")
            if source.startswith("https://"):
                with tempfile.TemporaryDirectory(prefix="seethetruth-") as tmp:
                    root=sparse_clone(source,Path(tmp))
                    result=parse_paths([str(root)],entry=entry or None,max_depth=3)
            else:
                # This is an opt-in local workstation service, never exposed to the network.
                path=Path(source).expanduser().resolve()
                if not path.exists() or not (path.is_dir() or path.suffix==".py"):
                    raise ValueError("Local path must exist and be a .py file or directory")
                result=parse_paths([str(path)],entry=entry or None,max_depth=3)
            return self.emit(200,{"graph":result})
        except (ValueError,KeyError,OSError,subprocess.TimeoutExpired) as e:
            return self.emit(400,{"error":str(e)[:1200]})
        except Exception as e:
            # Do not send stack traces or local environment paths unnecessarily.
            return self.emit(500,{"error":type(e).__name__+": analyzer failed"})
    def log_message(self,fmt,*args):
        print("%s - %s"%(self.address_string(),fmt%args))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--port",type=int,default=8765)
    args=ap.parse_args()
    if not 1024<=args.port<=65535:ap.error("port must be between 1024 and 65535")
    server=ThreadingHTTPServer((HOST,args.port),Handler)
    print("SeetheTruth local-only: http://127.0.0.1:"+str(args.port))
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=="__main__":main()
