"""Real Chromium evidence of the extensible executable-block Scratch-like learning UI."""
import json
import os
import struct
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

DEST=Path(os.environ.get("SEETHTRUTH_CATALOG_SHOTS","catalog_screenshots"))


class BlockLibraryVisualAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DEST.mkdir(exist_ok=True,parents=True)
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.playwright=sync_playwright().start()
        cls.browser=cls.playwright.chromium.launch(headless=True,args=["--no-sandbox"])
        cls.base="http://127.0.0.1:"+str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.page=self.browser.new_page(viewport={"width":1600,"height":1000},device_scale_factor=1)
        self.errors=[]
        self.page.on("pageerror",lambda e:self.errors.append(str(e)))
        self.page.goto(self.base+"/learn.html",wait_until="domcontentloaded")
        self.page.wait_for_function("window.SeetheTruthLearning && document.querySelectorAll('#palette [data-add]').length===window.SeetheTruthLearning.KINDS.length")
        self.page.locator("#studyMode").select_option("numeric")

    def tearDown(self):
        self.assertEqual(self.errors,[],"Browser JavaScript errors")
        self.page.close()

    def assert_all_graph_blocks_visible(self):
        p=self.page
        data=p.evaluate("""() => {
          const viewport=document.querySelector('#viewport').getBoundingClientRect();
          const nodes=[...document.querySelectorAll('#blocks .block')].map(n=>{
            const r=n.getBoundingClientRect();
            return {id:n.dataset.node,left:r.left,right:r.right,top:r.top,bottom:r.bottom};
          });
          return {viewport:{left:viewport.left,right:viewport.right,top:viewport.top,bottom:viewport.bottom},
                  zoom:Number(document.querySelector('#board').style.zoom),nodes};
        }""")
        self.assertGreater(data["zoom"],0.45)
        for n in data["nodes"]:
            self.assertGreaterEqual(n["left"],data["viewport"]["left"]-3,n["id"])
            self.assertLessEqual(n["right"],data["viewport"]["right"]+3,n["id"])
            self.assertLessEqual(n["bottom"],data["viewport"]["bottom"]+3,n["id"])

    def shot(self,name):
        p=self.page
        p.evaluate("document.fonts.ready")
        image=p.screenshot(path=str(DEST/name),full_page=True,animations="disabled")
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        w,h=struct.unpack(">II",image[16:24])
        self.assertGreaterEqual(w,1600)
        self.assertGreaterEqual(h,980)
        self.assertFalse(p.evaluate("document.documentElement.scrollWidth>innerWidth+2"),
                         "document has horizontal overflow")
        self.assertIsNotNone(p.locator("#palette").bounding_box())
        return {"file":name,"width":w,"height":h,"bytes":len(image),
                "search":p.locator("#blockSearch").input_value(),
                "visible_catalog_items":p.locator("#palette [data-add]").count(),
                "mission":p.locator("#workspaceTitle").inner_text()}

    def test_free_wiring_shape_repair_without_model_template(self):
        p=self.page
        p.locator('[data-preset="blank"]').click()
        p.locator('[data-add="transpose2d"]').click()
        t=p.locator('#blocks .block').last.get_attribute("data-node")
        self.assertEqual(p.locator('[data-node="'+t+'"]').get_attribute("data-readiness"),"missing")
        p.locator('[data-out="x"]').click()
        p.locator('[data-in="'+t+'"][data-port="x"]').click()
        self.assertEqual(p.locator('[data-node="'+t+'"]').get_attribute("data-readiness"),"ready")
        p.locator('[data-add="matmul"]').click()
        mm=p.locator('#blocks .block').last.get_attribute("data-node")
        for port in ("a","b"):
            p.locator('[data-out="x"]').click()
            p.locator('[data-in="'+mm+'"][data-port="'+port+'"]').click()
        self.assertEqual(p.locator('[data-node="'+mm+'"]').get_attribute("data-readiness"),"shape-error")
        self.assertIn("形状错误 1",p.locator("#graphReadiness").inner_text())
        self.shot("35-free-wiring-shape-error.png")
        p.locator("#checkGraph").click()
        self.assertIn("尚不可运行",p.locator("#status").inner_text())
        p.locator('[data-out="'+t+'"]').click()
        p.locator('[data-in="'+mm+'"][data-port="b"]').click()
        p.locator('[data-add="softmax"]').click()
        soft=p.locator('#blocks .block').last.get_attribute("data-node")
        p.locator('[data-out="'+mm+'"]').click()
        p.locator('[data-in="'+soft+'"][data-port="x"]').click()
        self.assertEqual(p.locator('[data-node="'+soft+'"]').get_attribute("data-readiness"),"ready")
        self.assertIn("3×3",p.locator('[data-node="'+soft+'"]').inner_text())
        self.assertIn("输入完整",p.locator("#graphReadiness").inner_text())
        p.locator("#checkGraph").click()
        self.assertIn("CPU 小张量执行",p.locator("#status").inner_text())
        p.locator("#run").click()
        self.assertIn("1 / 4",p.locator("#stepInfo").inner_text())
        for _ in range(3):p.locator("#next").click()
        self.assertIn("4 / 4",p.locator("#stepInfo").inner_text())
        self.assertIn("✓ 已运行",p.locator("#status").inner_text())
        p.locator("#autoLayout").click()
        self.shot("36-free-wiring-arbitrary-valid.png")

    def test_search_filter_circuits_and_evidence_screens(self):
        p=self.page
        size=p.evaluate("window.SeetheTruthLearning.KINDS.length")
        self.assertGreaterEqual(size,64)
        self.assertEqual(p.locator("#palette [data-add]").count(),size)
        self.assertIn(f"{size} / {size}",p.locator("#blockCount").inner_text())
        self.assertGreaterEqual(p.locator("#catalogCategories button").count(),10)
        screens=[self.shot("01-library-all-blocks.png")]
        p.locator("#blockSearch").fill("SwiGLU")
        self.assertEqual(p.locator("#palette [data-add]").count(),1)
        self.assertIn("SwiGLU",p.locator("#palette").inner_text())
        screens.append(self.shot("02-library-search-swiglu.png"))
        p.locator("#blockSearch").fill("")
        p.locator('[data-catalog="loss"]').click()
        self.assertEqual(p.locator("#palette [data-add]").count(),4)
        self.assertEqual(p.locator("#blockCount").inner_text(),f"4 / {size}")
        screens.append(self.shot("03-library-loss-category.png"))
        p.locator('[data-catalog="all"]').click()
        p.locator('[data-preset="swiglu"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),5)
        p.locator("#run").click()
        self.assertIn("1 / 5",p.locator("#stepInfo").inner_text())
        for _ in range(3):p.locator("#next").click()
        self.assertIn("4 / 5",p.locator("#stepInfo").inner_text())
        self.assertIn("SwiGLU",p.locator("#inspectorTitle").inner_text())
        self.assertEqual(p.locator("#matrix .cell").count(),24)
        self.assert_all_graph_blocks_visible()
        screens.append(self.shot("04-swiglu-numeric-execution.png"))
        p.locator('[data-preset="classifier"]').click()
        p.locator("#run").click()
        for _ in range(4):p.locator("#next").click()
        self.assertIn("5 / 5",p.locator("#stepInfo").inner_text())
        self.assertIn("交叉熵",p.locator("#inspectorTitle").inner_text())
        self.assertEqual(p.locator("#matrix .cell").count(),1)
        self.assert_all_graph_blocks_visible()
        screens.append(self.shot("05-classifier-crossentropy.png"))
        p.locator('[data-preset="mask"]').click()
        p.locator("#run").click()
        p.locator("#next").click()
        p.locator("#next").click()
        self.assertIn("3 / 3",p.locator("#stepInfo").inner_text())
        self.assertIn("Mask Softmax",p.locator("#inspectorTitle").inner_text())
        self.assertEqual(p.locator("#matrix .cell").count(),9)
        self.assert_all_graph_blocks_visible()
        screens.append(self.shot("06-causal-mask-output.png"))
        (DEST/"manifest.json").write_text(json.dumps({
          "schema":"seethetruth-catalog-visual-evidence-v1","browser":"Playwright Chromium",
          "viewport":[1600,1000],"catalogue_count":size,
          "captures":screens+[{"file":"08-mla_lite-numeric.png"},{"file":"08-moe_route-numeric.png"},{"file":"08-grpo_signals-numeric.png"},{"file":"09-paper-hover-low-rank.png"},{"file":"10-mla-auto-layout-readable.png"},{"file":"14-mla-cache-input-and-writes.png"},{"file":"15-mla-absorbed-vs-rebuild-numeric.png"},{"file":"16-mla-absorb-paper-hover.png"},{"file":"17-mla-final-zero-difference.png"},{"file":"07-qualitative-gqa-no-matrix.png",
            "description":"No numeric matrices or hardware measurements; GQA dimensions only"}],
          "tests":"qualitative GQA without numeric cells, opt-in numeric SwiGLU/CE/Masked Softmax, search and category",
          "limit":"browser teaching simulation, not checkpoint or GPU execution"
        },ensure_ascii=False,indent=2),encoding="utf-8")

    def test_default_qualitative_library_capture(self):
        p=self.page
        p.locator("#studyMode").select_option("concept")
        p.locator('[data-preset="gqa"]').click()
        p.locator("#run").click()
        for _ in range(5):
            p.locator("#next").click()
        self.assertIn("6 / 7",p.locator("#stepInfo").inner_text())
        self.assertIn("推断 shape",p.locator("#shape").inner_text())
        self.assertIn("内部阶段的形状推断",p.locator("#intermediates").inner_text())
        self.assertEqual(p.locator("#matrix .cell").count(),0)
        self.assertNotIn("FLOPs",p.locator("#metrics").inner_text())
        self.shot("07-qualitative-gqa-no-matrix.png")

    def test_layout_is_readable_undoable_and_does_not_cover_blocks(self):
        p=self.page
        p.locator('[data-preset="mla_lite"]').click()
        self.assertGreaterEqual(float(p.locator("#board").evaluate("(e)=>e.style.zoom")),0.72)
        coords=p.evaluate("""() => [...document.querySelectorAll('#blocks .block')].map(n=>({
          id:n.dataset.node,x:parseFloat(n.style.left),y:parseFloat(n.style.top)
        }))""")
        for i,a in enumerate(coords):
            for b in coords[i+1:]:
                self.assertTrue(abs(a["x"]-b["x"])>=220 or abs(a["y"]-b["y"])>=175,
                                "layout collision "+a["id"]+" / "+b["id"])
        p.locator("#run").click()
        self.assertIn("1 / 20",p.locator("#stepInfo").inner_text())
        self.shot("10-mla-auto-layout-readable.png")
        before_fit=float(p.locator("#board").evaluate("(e)=>e.style.zoom"))
        p.locator("#zoomFit").click()
        after_fit=float(p.locator("#board").evaluate("(e)=>e.style.zoom"))
        self.assertLess(after_fit,before_fit)
        p.locator('[data-preset="ffn"]').click()
        before=p.evaluate("() => [...document.querySelectorAll('#blocks .block')].map(n=>[n.style.left,n.style.top])")
        p.locator("#autoLayout").click()
        after=p.evaluate("() => [...document.querySelectorAll('#blocks .block')].map(n=>[n.style.left,n.style.top])")
        self.assertNotEqual(before,after)
        p.locator("#undo").click()
        undone=p.evaluate("() => [...document.querySelectorAll('#blocks .block')].map(n=>[n.style.left,n.style.top])")
        self.assertEqual(before,undone)

    def test_configurable_qkv_ffn_and_moe_parameter_training(self):
        import json
        p=self.page
        cases=[
            ("trainable_attention",10,["wq","wk","wv"],"32-qkv-three-parameter-train.png"),
            ("trainable_swiglu",10,["w1","w3","w2"],"33-swiglu-three-parameter-train.png"),
            ("trainable_moe",24,["w1","w3","w2"],"34-moe-three-parameter-train.png")
        ]
        for lesson,nodes,weight_ids,screenshot in cases:
            p.locator('[data-preset="'+lesson+'"]').click()
            self.assertEqual(p.locator("#blocks .block").count(),nodes)
            p.locator("#studyMode").select_option("numeric")
            p.locator("#trainLearningRate").select_option("0.004")
            p.locator("#trainWeightDecay").select_option("0")
            p.locator("#trainCount").select_option("2")
            p.locator("#trainStep").click()
            self.assertIn("AdamW step 2",p.locator("#trainSummary").inner_text())
            self.assertIn("Learning rate 0.004",p.locator("#trainSummary").inner_text())
            self.assertIn("weight decay 0",p.locator("#trainSummary").inner_text())
            self.assertIn("Step 2:",p.locator("#trainSummary").inner_text())
            self.assertEqual(p.locator("#trainSummary svg").count(),1)
            self.assertIn("‖∇θ‖₂",p.locator("#trainSummary").inner_text())
            self.assertIn("‖Δθ‖₂",p.locator("#trainSummary").inner_text())
            with p.expect_download() as d:
                p.locator("#save").click()
            state=json.loads(open(d.value.path(),encoding="utf-8").read())
            param={n["id"]:n for n in state["nodes"] if n["kind"]=="parameter"}
            self.assertEqual(set(param),set(weight_ids))
            self.assertTrue(all(param[id]["params"]["optimizer"]["t"]==2 for id in weight_ids))
            p.locator("#trainSummary").scroll_into_view_if_needed()
            self.shot(screenshot)
            p.locator("#undo").click()
            with p.expect_download() as previous:
                p.locator("#save").click()
            old=json.loads(open(previous.value.path(),encoding="utf-8").read())
            self.assertTrue(all(n["params"]["optimizer"]["t"]==0 for n in old["nodes"] if n["kind"]=="parameter"),
                            "all parameter moments must atomically roll back after batched training")

    def test_parameter_train_two_steps_undo_and_json_resume(self):
        import json
        p=self.page
        p.locator('[data-preset="trainable_core"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),11)
        p.locator("#studyMode").select_option("concept")
        self.assertTrue(p.locator("#trainStep").is_disabled())
        p.locator("#studyMode").select_option("numeric")
        self.assertFalse(p.locator("#trainStep").is_disabled())
        p.locator("#trainStep").click()
        self.assertIn("AdamW step 1",p.locator("#trainSummary").inner_text())
        self.assertIn("Loss",p.locator("#trainSummary").inner_text())
        p.locator("#trainSummary").scroll_into_view_if_needed()
        self.shot("29-parameter-persisted-train-step-one.png")
        p.locator("#trainStep").click()
        self.assertIn("AdamW step 2",p.locator("#trainSummary").inner_text())
        self.shot("30-parameter-persisted-train-step-two.png")
        with p.expect_download() as downloaded:
            p.locator("#save").click()
        raw=json.loads(open(downloaded.value.path(),encoding="utf-8").read())
        W=next(n for n in raw["nodes"] if n["id"]=="shared_weight")
        self.assertEqual(W["kind"],"parameter")
        self.assertEqual(W["params"]["optimizer"]["t"],2)
        self.assertEqual(len(W["params"]["values"]),2)
        self.assertEqual(len(W["params"]["optimizer"]["m"]),2)
        self.assertEqual(len(W["params"]["optimizer"]["v"]),2)
        p.locator("#undo").click()
        self.assertTrue(p.locator("#trainSummary").is_hidden())
        with p.expect_download() as d1:
            p.locator("#save").click()
        restored_previous=json.loads(open(d1.value.path(),encoding="utf-8").read())
        self.assertEqual(next(n for n in restored_previous["nodes"] if n["id"]=="shared_weight")["params"]["optimizer"]["t"],1)
        p.locator("#load").set_input_files(downloaded.value.path())
        self.assertTrue(p.locator("#trainSummary").is_hidden())
        with p.expect_download() as d2:
            p.locator("#save").click()
        resumed=json.loads(open(d2.value.path(),encoding="utf-8").read())
        self.assertEqual(next(n for n in resumed["nodes"] if n["id"]=="shared_weight")["params"],W["params"])
        p.locator("#trainStep").click()
        self.assertIn("AdamW step 3",p.locator("#trainSummary").inner_text())
        p.locator("#trainSummary").scroll_into_view_if_needed()
        self.shot("31-parameter-resumed-third-step.png")

    def test_moe_unified_weight_and_router_gate_backward(self):
        p=self.page
        p.locator('[data-preset="moe_backward"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),24)
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        p.locator("#backward").click()
        self.assertIn("24 / 24",p.locator("#stepInfo").inner_text())
        i=p.evaluate("""() => window.SeetheTruthLearning.order(
           window.SeetheTruthLearning.preset('moe_backward')).findIndex(n=>n.id==='w1')""")
        p.locator('#trace [data-step="'+str(i)+'"]').click()
        self.assertEqual(p.locator("#gradientInspector .cell").count(),16)
        self.assertEqual(p.locator("#adamwPreview").count(),1)
        p.locator("#gradientInspector").scroll_into_view_if_needed()
        self.shot("27-moe-expert-weight-bank-gradients.png")
        p.locator("#adamwPreview").click()
        self.assertEqual(p.locator("#adamwResult .cell").count(),32)
        p.locator("#adamwResult").scroll_into_view_if_needed()
        self.shot("28-moe-expert-bank-adamw-preview.png")
        trace=p.evaluate("""() => {
           const g=window.SeetheTruthLearning.preset('moe_backward'),
                 f=window.SeetheTruthLearning.evaluate(g),
                 b=window.SeetheTruthGraphAutodiff.backward(g,f);
           return {gate:b.gradients.affinity,
                   discrete:b.gradients.select,stop:b.cutEdges};}""")
        self.assertTrue(any(abs(v)>1e-8 for row in trace["gate"] for v in row))
        self.assertTrue(all(v==0 for row in trace["discrete"] for v in row))
        self.assertIn("indices",[e["port"] for e in trace["stop"]])

    def test_unified_graph_autodiff_shared_weight_attention_gradient(self):
        p=self.page
        p.locator('[data-preset="autodiff_core"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),11)
        self.assertTrue(p.locator("#backward").is_disabled())
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        self.assertFalse(p.locator("#backward").is_disabled())
        p.locator("#backward").click()
        self.assertIn("11 / 11",p.locator("#stepInfo").inner_text())
        p.locator("#gradientInspector").scroll_into_view_if_needed()
        self.assertEqual(p.locator("#gradientInspector .cell").count(),1)
        self.shot("24-unified-autodiff-loss-backward.png")
        trace=p.evaluate("""() => {
          const g=window.SeetheTruthLearning.preset('autodiff_core'),
                f=window.SeetheTruthLearning.evaluate(g),
                b=window.SeetheTruthGraphAutodiff.backward(g,f);
          return {shared:b.gradients.shared_weight,x:b.gradients.features,
                  target:b.gradients.target,
                  paths:b.pathIds,stop:b.cutEdges};
        }""")
        self.assertEqual(len(trace["shared"]),2)
        self.assertTrue(any(abs(v)>1e-9 for row in trace["shared"] for v in row))
        self.assertTrue(all(v==0 for row in trace["target"] for v in row))
        self.assertIn("attention",trace["paths"] and ["attention" if x=="attend" else x for x in trace["paths"]])
        i=p.evaluate("""() => window.SeetheTruthLearning.order(
            window.SeetheTruthLearning.preset('autodiff_core')).findIndex(n=>n.id==='shared_weight')""")
        p.locator('#trace [data-step="'+str(i)+'"]').click()
        self.assertEqual(p.locator("#gradientInspector .cell").count(),4)
        p.locator("#gradientInspector").scroll_into_view_if_needed()
        self.assertIn("∂LOSS",p.locator("#gradientInspector").inner_text())
        self.shot("25-shared-weight-accumulated-matmul-gradients.png")
        self.assertEqual(p.locator("#adamwPreview").count(),1)
        initial=p.locator("#matrix .cell").all_inner_texts()
        p.locator("#adamwPreview").click()
        self.assertIn("AdamW",p.locator("#adamwResult").inner_text())
        self.assertEqual(p.locator("#adamwResult .cell").count(),8)
        self.assertEqual(initial,p.locator("#matrix .cell").all_inner_texts(),
                         "optimizer preview must not silently overwrite original graph weights")
        p.locator("#adamwResult").scroll_into_view_if_needed()
        self.shot("26-adamw-shared-parameter-update-preview.png")
        p.locator('[data-preset="grpo_objective"]').click()
        self.assertFalse(p.locator("#backward").is_enabled())
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        p.locator("#backward").click()
        self.assertIn("20 / 20",p.locator("#stepInfo").inner_text())
        self.assertNotIn("不可用",p.locator("#status").inner_text())

    def test_grpo_gradient_trace_clip_and_stop_boundaries(self):
        p=self.page
        self.assertTrue(p.locator("#backward").is_disabled())
        p.locator('[data-preset="grpo_objective"]').click()
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        self.assertFalse(p.locator("#backward").is_disabled())
        p.locator("#backward").click()
        self.assertIn("20 / 20",p.locator("#stepInfo").inner_text())
        self.assertIn("∂LOSS",p.locator("#gradientInspector").inner_text())
        p.locator("#gradientInspector").scroll_into_view_if_needed()
        self.assertLess(p.locator("#gradientInspector .cell").first.bounding_box()["y"],1000)
        self.shot("21-grpo-reverse-final-loss-gradient.png")
        p.locator('#trace [data-step="0"]').click()
        self.assertEqual(p.locator("#gradientInspector .cell").count(),16)
        self.assertIn("参与可求导",p.locator("#gradientInspector").inner_text())
        trace=p.evaluate("""() => {
            const g=window.SeetheTruthLearning.preset('grpo_objective'),
                f=window.SeetheTruthLearning.evaluate(g),
                b=window.SeetheTruthGRPOBackward.backward(g,f);
            return {policyGrad:b.gradients.new_logits,
                    oldGrad:b.gradients.old_logits,
                    refGrad:b.gradients.ref_logits,
                    cutEdges:b.cutEdges,clip:b.local.surrogate.clipSaturated.values};
        }""")
        self.assertEqual(trace["policyGrad"][3],[0,0,0,0])
        self.assertTrue(any(abs(z)>1e-8 for row in trace["policyGrad"][:3] for z in row))
        self.assertTrue(all(z==0 for row in trace["oldGrad"] for z in row))
        self.assertTrue(all(z==0 for row in trace["refGrad"] for z in row))
        self.assertIn("old",[e["port"] for e in trace["cutEdges"]])
        self.assertIn("reference",[e["port"] for e in trace["cutEdges"]])
        self.assertIn("advantage",[e["port"] for e in trace["cutEdges"]])
        p.locator("#gradientInspector").scroll_into_view_if_needed()
        self.assertLess(p.locator("#gradientInspector .cell").first.bounding_box()["y"],1000)
        self.shot("22-grpo-backward-current-policy-gradients.png")
        i=p.evaluate("""() => window.SeetheTruthLearning.order(
             window.SeetheTruthLearning.preset('grpo_objective')).findIndex(n=>n.id==='surrogate')""")
        p.locator('#trace [data-step="'+str(i)+'"]').click()
        self.assertIn("clip",p.locator("#gradientInspector").inner_text().lower())
        self.assertIn("停止梯度",p.locator("#gradientInspector").inner_text())
        p.locator("#gradientInspector").scroll_into_view_if_needed()
        self.assertLess(p.locator("#gradientInspector .cell").first.bounding_box()["y"],1000)
        self.shot("23-grpo-clip-vjp-detach-explanation.png")
        p.locator('[data-preset="ffn"]').click()
        self.assertTrue(p.locator("#backward").is_disabled())
        self.assertNotIn("∂LOSS",p.locator("#gradientInspector").inner_text())

    def test_grpo_full_loss_as_composable_workbench_graph(self):
        p=self.page
        p.locator('[data-preset="grpo_objective"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),20)
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        self.assertIn("1 / 20",p.locator("#stepInfo").inner_text())
        self.shot("18-grpo-new-old-reference-and-rewards.png")
        for _ in range(19):p.locator("#next").click()
        self.assertIn("20 / 20",p.locator("#stepInfo").inner_text())
        self.assertEqual(p.locator("#matrix .cell").count(),1)
        self.assertIn("1 × 1",p.locator("#metrics").inner_text())
        self.assertNotIn("无法运行",p.locator("#status").inner_text())
        p.locator("#viewport").evaluate("(e)=>{e.scrollLeft=e.scrollWidth;e.scrollTop=0;}")
        p.locator("#matrix").scroll_into_view_if_needed()
        self.assertEqual(p.locator("#matrix .cell").count(),1)
        self.shot("19-grpo-final-clipped-masked-loss.png")
        p.locator("#blockSearch").fill("Clipped Policy Surrogate")
        card=p.locator('#palette [data-add="ppo_clipped_surrogate"]')
        self.assertEqual(card.count(),1)
        card.hover()
        p.locator("#paperHoverInspector").wait_for(state="visible",timeout=4000)
        self.assertIn("PPO",p.locator("#paperHoverInspector").inner_text())
        self.shot("20-grpo-clipped-paper-hover.png")

    def test_mla_compressed_cache_and_absorbed_scores(self):
        p=self.page
        p.locator('[data-preset="mla_cache"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),35)
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        self.assertIn("1 / 35",p.locator("#stepInfo").inner_text())
        self.shot("14-mla-cache-input-and-writes.png")
        for _ in range(34):p.locator("#next").click()
        self.assertIn("35 / 35",p.locator("#stepInfo").inner_text())
        self.assertNotIn("无法运行",p.locator("#status").inner_text())
        p.locator("#viewport").evaluate("(e)=>{e.scrollLeft=e.scrollWidth;e.scrollTop=0;}")
        self.shot("15-mla-absorbed-vs-rebuild-numeric.png")
        self.assertEqual(p.locator("#matrix .cell").count(),2)
        vals=[float(v) for v in p.locator("#matrix .cell").all_inner_texts()]
        self.assertTrue(all(abs(v)<0.005 for v in vals),"MLA explicit/absorbed output difference must display zero")
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("17-mla-final-zero-difference.png")
        p.locator("#blockSearch").fill("Absorb K Projection")
        card=p.locator('#palette [data-add="query_weight_absorb"]')
        self.assertEqual(card.count(),1)
        card.hover()
        p.locator("#paperHoverInspector").wait_for(state="visible",timeout=4000)
        self.assertIn("DeepSeek-V3",p.locator("#paperHoverInspector").inner_text())
        self.assertIn("W",p.locator("#paperHoverInspector").inner_text())
        self.shot("16-mla-absorb-paper-hover.png")

    def test_routed_expert_swiglu_shared_expert_and_research_hover(self):
        p=self.page
        p.locator('[data-preset="moe_expert_ffn"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),22)
        self.assertGreaterEqual(float(p.locator("#board").evaluate("(e)=>e.style.zoom")),0.72)
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        self.assertIn("1 / 22",p.locator("#stepInfo").inner_text())
        for _ in range(21):p.locator("#next").click()
        self.assertIn("22 / 22",p.locator("#stepInfo").inner_text())
        self.assertNotIn("无法运行",p.locator("#status").inner_text())
        self.assertIn("2",p.locator("#shape").inner_text())
        self.shot("11-expert-ffn-beginning.png")
        p.locator("#viewport").evaluate("(e)=>{e.scrollLeft=e.scrollWidth;e.scrollTop=0;}")
        self.shot("12-expert-ffn-output.png")
        p.locator("#blockSearch").fill("Expert-specific Linear")
        card=p.locator('#palette [data-add="expert_linear_bank"]')
        self.assertEqual(card.count(),1)
        card.hover()
        p.locator("#paperHoverInspector").wait_for(state="visible",timeout=4000)
        self.assertIn("DeepSeek-V3",p.locator("#paperHoverInspector").inner_text())
        self.assertIn("W",p.locator("#paperHoverInspector").inner_text())
        self.shot("13-expert-ffn-paper-hover.png")

    def test_new_model_family_presets_and_paper_hover(self):
        p=self.page
        for mission,count in [("mla_lite",20),("moe_route",11),("grpo_signals",6)]:
            p.locator('[data-preset="'+mission+'"]').click()
            self.assertEqual(p.locator("#blocks .block").count(),count)
            p.locator("#run").click()
            self.assertIn("1 / "+str(count),p.locator("#stepInfo").inner_text())
            for _ in range(count-1):
                p.locator("#next").click()
            self.assertIn(str(count)+" / "+str(count),p.locator("#stepInfo").inner_text())
            self.assertNotIn("无法运行",p.locator("#status").inner_text())
            self.shot("08-"+mission+"-numeric.png")
        p.locator("#blockSearch").fill("Low-rank")
        card=p.locator('#palette [data-add="low_rank_project"]')
        self.assertEqual(card.count(),1)
        card.hover()
        p.locator("#paperHoverInspector").wait_for(state="visible",timeout=4000)
        self.assertIn("DeepSeek-V3",p.locator("#paperHoverInspector").inner_text())
        self.assertTrue(p.locator('#paperHoverInspector a[href="https://arxiv.org/abs/2412.19437"]').count())
        self.shot("09-paper-hover-low-rank.png")

    def test_legacy_and_new_preset_numerical_runs(self):
        p=self.page
        for mission,count in [("ffn",4),("residual",4),("gqa",7),
                              ("embedding",3),("norm",5),("sampling",5),("mask",3)]:
            p.locator('[data-preset="'+mission+'"]').click()
            self.assertEqual(p.locator("#blocks .block").count(),count)
            p.locator("#run").click()
            self.assertIn("1 / "+str(count),p.locator("#stepInfo").inner_text(),mission)
            for _ in range(count-1):p.locator("#next").click()
            self.assertIn(str(count)+" / "+str(count),p.locator("#stepInfo").inner_text())
            self.assertNotIn("无法运行",p.locator("#status").inner_text())


if __name__=="__main__":
    unittest.main()
