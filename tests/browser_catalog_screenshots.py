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

    def test_trainable_gqa_mqa_nd_adamw_roundtrip_and_loss_curve(self):
        import json
        p=self.page
        for preset,heads,name in [
            ("nd_trainable_gqa",2,"48-trainable-gqa-4d-adamw.png"),
            ("nd_trainable_mqa",1,"49-trainable-mqa-4d-adamw.png")
        ]:
            p.locator('[data-preset="'+preset+'"]').click()
            self.assertEqual(p.locator("#blocks .block").count(),13)
            self.assertIn("13/13",p.locator("#graphReadiness").inner_text())
            p.locator("#studyMode").select_option("numeric")
            p.locator("#trainLearningRate").select_option("0.004")
            p.locator("#trainCount").select_option("2")
            p.locator("#trainStep").click()
            self.assertIn("AdamW step 2",p.locator("#trainSummary").inner_text())
            self.assertIn("Loss",p.locator("#trainSummary").inner_text())
            self.assertEqual(p.locator("#trainSummary svg").count(),1)
            self.assertEqual(p.locator("#trainSummary strong").count(),3)
            self.assertIn("分支 loss",p.locator("#stepInfo").inner_text())
            with p.expect_download() as saved:
                p.locator("#save").click()
            graph=json.loads(open(saved.value.path(),encoding="utf-8").read())
            params={n["id"]:n["params"] for n in graph["nodes"] if n["kind"]=="nd_parameter"}
            self.assertEqual(set(params),{"query","key","value"})
            self.assertEqual(len(params["query"]["values"][0]),4)
            self.assertEqual(len(params["key"]["values"][0]),heads)
            self.assertTrue(all(v["optimizer"]["t"]==2 for v in params.values()))
            p.locator("#trainSummary").scroll_into_view_if_needed()
            self.shot(name)
            p.locator("#undo").click()
            with p.expect_download() as previous:
                p.locator("#save").click()
            earlier=json.loads(open(previous.value.path(),encoding="utf-8").read())
            self.assertTrue(all(n["params"]["optimizer"]["t"]==0 for n in earlier["nodes"] if n["kind"]=="nd_parameter"))
            p.locator("#load").set_input_files(saved.value.path())
            p.locator("#trainStep").click()
            self.assertIn("AdamW step 4",p.locator("#trainSummary").inner_text())

    def test_gqa_mqa_four_axis_gradient_and_head_sharing(self):
        p=self.page
        for preset,expected_kv,name in [
            ("nd_gqa_attention",2,"46-gqa-shared-kv-backward.png"),
            ("nd_mqa_attention",1,"47-mqa-one-kv-head-backward.png")
        ]:
            p.locator('[data-preset="'+preset+'"]').click()
            self.assertEqual(p.locator("#blocks .block").count(),13)
            p.locator("#studyMode").select_option("numeric")
            p.locator("#run").click()
            self.assertFalse(p.locator("#backward").is_disabled())
            p.locator("#backward").click()
            self.assertIn("13 / 13",p.locator("#stepInfo").inner_text())
            i=p.evaluate("""(preset)=>window.SeetheTruthLearning.order(
               window.SeetheTruthLearning.preset(preset)).findIndex(n=>n.id==='key')""",preset)
            p.locator('#trace [data-step="'+str(i)+'"]').click()
            self.assertEqual(p.locator("#gradientInspector [data-nd-slice]").count(),2)
            self.assertEqual(p.locator("#gradientInspector .cell").count(),6)
            self.assertIn("∂LOSS",p.locator("#gradientInspector").inner_text())
            self.assertIn("1×"+str(expected_kv)+"×3×2",p.locator("#metrics").inner_text().replace(" ",""))
            test=p.evaluate("""(preset)=>{
              const L=window.SeetheTruthLearning,ND=window.SeetheTruthND,A=window.SeetheTruthGraphAutodiff,
              graph=L.preset(preset),forward=L.evaluate(graph),back=A.backward(graph,forward);
              const grad=back.gradients.key;
              return {shape:ND.shape(grad),nonzero:ND.flatten(grad).some(v=>Math.abs(v)>1e-9),
                targetZero:ND.flatten(back.gradients.target).every(v=>v===0),
                detached:back.cutEdges.some(e=>e.nodeId==='loss'&&e.port==='target')};
            }""",preset)
            self.assertEqual(test["shape"],[1,expected_kv,3,2])
            self.assertTrue(test["nonzero"])
            self.assertTrue(test["targetZero"])
            self.assertTrue(test["detached"])
            if expected_kv>1:p.locator('#gradientInspector [data-nd-slice="1"]').select_option("1")
            p.locator("#gradientInspector").scroll_into_view_if_needed()
            self.shot(name)

    def test_adam_adagrad_and_elu_hardswish_extensions(self):
        import json
        p=self.page
        p.locator('[data-preset="activation_extension"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),6)
        self.assertIn("6/6",p.locator("#graphReadiness").inner_text())
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        result=p.evaluate("""() => {
          const L=window.SeetheTruthLearning,run=L.evaluate(L.preset('activation_extension'));
          return Object.fromEntries(run.steps.map(s=>[s.id,{shape:s.shape,values:s.values}]));
        }""")
        self.assertEqual(result["elu_act"]["shape"],[2,4])
        self.assertEqual(result["elu4"]["shape"],[1,2,2,4])
        self.assertNotEqual(result["elu_act"]["values"],result["hard_act"]["values"])
        self.assertNotEqual(result["elu4"]["values"],result["hard4"]["values"])
        names=p.evaluate("""() => window.SeetheTruthLearning.order(
                 window.SeetheTruthLearning.preset('activation_extension')).map(n=>n.id)""")
        p.locator('#trace [data-step="'+str(names.index("hard4"))+'"]').click()
        p.locator('#matrix [data-nd-slice="1"]').select_option("1")
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("59-elu-hardswish-four-axis-activation.png")
        p.locator('[data-preset="optimizer_decay_compare"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),6)
        p.locator("#run").click()
        vals=p.evaluate("""() => {
          const L=window.SeetheTruthLearning,out=L.evaluate(L.preset('optimizer_decay_compare'));
          return Object.fromEntries(out.steps.map(s=>[s.id,s.values]));
        }""")
        self.assertNotEqual(vals["adam"],vals["adamw"])
        self.assertNotEqual(vals["adagrad"],vals["rmsprop"])
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("60-adam-adamw-adagrad-rmsprop-canvas.png")
        p.locator('[data-preset="trainable_core"]').click()
        p.locator("#trainCount").select_option("2")
        for optimizer,screenshot in [
            ("adam","61-coupled-adam-persistent-train.png"),
            ("adagrad","62-adagrad-accumulator-persistent-train.png")
        ]:
            p.locator("#trainOptimizer").select_option(optimizer)
            if optimizer=="adagrad":
                p.locator("#optimizerAdvanced summary").click()
                p.locator("#trainLRDecay").select_option("1")
                self.assertEqual(p.locator("#trainLRDecay").input_value(),"1")
                p.locator("#optimizerAdvanced summary").click()
            p.locator("#trainStep").click()
            self.assertIn("Step 2",p.locator("#trainSummary").inner_text())
            with p.expect_download() as exported:
                p.locator("#save").click()
            graph=json.loads(open(exported.value.path(),encoding="utf-8").read())
            state=next(n for n in graph["nodes"] if n["id"]=="shared_weight")["params"]["optimizer"]
            self.assertEqual(state["kind"],optimizer)
            self.assertEqual(state["t"],2)
            if optimizer=="adagrad":self.assertEqual(state["config"]["lrDecay"],1)
            self.assertEqual(p.locator("#trainOptimizer").input_value(),optimizer)
            p.locator("#trainSummary").scroll_into_view_if_needed()
            self.shot(screenshot)
        p.locator("#undo").click()
        with p.expect_download() as before:
            p.locator("#save").click()
        prior=json.loads(open(before.value.path(),encoding="utf-8").read())
        prev_state=next(n for n in prior["nodes"] if n["id"]=="shared_weight")["params"]["optimizer"]
        self.assertEqual(prev_state["kind"],"adagrad")
        self.assertEqual(prev_state["t"],0)
        p.locator("#optimizerAdvanced summary").click()
        p.locator("#trainLRDecay").select_option("0")
        p.locator("#optimizerAdvanced summary").click()
        p.locator("#load").set_input_files(exported.value.path())
        self.assertEqual(p.locator("#trainOptimizer").input_value(),"adagrad")
        p.wait_for_function("document.querySelector('#trainLRDecay').value === '1'")
        self.assertEqual(p.locator("#trainLRDecay").input_value(),"1")
        p.locator("#trainStep").click()
        self.assertIn("adagrad step 4",p.locator("#trainSummary").inner_text())

    def test_activation_and_optimizer_free_composition_lessons(self):
        p=self.page
        p.locator('[data-preset="activation_gallery"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),7)
        self.assertIn("7/7",p.locator("#graphReadiness").inner_text())
        p.locator("#run").click()
        self.assertIn("1 / 7",p.locator("#stepInfo").inner_text())
        results=p.evaluate("""() => {
          const L=window.SeetheTruthLearning;
          const run=L.evaluate(L.preset('activation_gallery'));
          return {ids:run.steps.map(x=>x.id),
            outputs:Object.fromEntries(run.steps.map(s=>[s.id,s.values]))};
        }""")
        for k in ("gelu_cdf","gelu_approx","mish_act","relu_square","leaky","silu_act"):
            self.assertIn(k,results["ids"])
        self.assertNotEqual(results["outputs"]["gelu_cdf"],results["outputs"]["mish_act"])
        p.locator('#trace [data-step="'+str(results["ids"].index("gelu_cdf"))+'"]').click()
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("57-activation-gallery-gelu-cdf.png")
        p.locator('[data-preset="optimizer_compare"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),6)
        p.locator("#run").click()
        self.assertIn("1 / 6",p.locator("#stepInfo").inner_text())
        stats=p.evaluate("""() => {
          const L=window.SeetheTruthLearning,B=window.SeetheTruthBlockCatalog;
          const steps=L.evaluate(L.preset('optimizer_compare')).steps;
          return {ids:steps.map(x=>x.id),
            updates:Object.fromEntries(steps.map(s=>[s.id,s.values]))};
        }""")
        for k in ("adamw","sgd","rmsprop","lion"):
            self.assertIn(k,stats["ids"])
        self.assertNotEqual(stats["updates"]["adamw"],stats["updates"]["lion"])
        p.locator('#trace [data-step="'+str(stats["ids"].index("lion"))+'"]').click()
        self.assertEqual(p.locator("#matrix .cell").count(),4)
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("58-free-optimizer-lion-update.png")

    def test_activation_palette_and_four_optimizer_live_training(self):
        import json
        p=self.page
        # Every new activation is present and executes actual bounded values.
        all_kinds=p.evaluate("window.SeetheTruthLearning.KINDS")
        for kind in ("gelu_erf","relu_squared","mish","leaky_relu",
                     "nd_gelu_erf","nd_silu","nd_relu_squared","nd_mish",
                     "opt_adamw","opt_sgd_momentum","opt_rmsprop","opt_lion"):
            self.assertIn(kind,all_kinds)
        outcomes=p.evaluate("""() => {
            const B=window.SeetheTruthBlockCatalog,ND=window.SeetheTruthND;
            const x=[[-2,-.5,.5,2]];
            return {
              erf:B.execute({kind:'gelu_erf',params:{}},{x}).values[0],
              squared:B.execute({kind:'relu_squared',params:{}},{x}).values[0],
              mish:B.execute({kind:'mish',params:{}},{x}).values[0],
              ndShape:B.execute({kind:'nd_mish',params:{}},
                  {x:ND.seedTensor([1,2,2,4],3)}).shape
            };
        }""")
        self.assertEqual(outcomes["squared"],[0,0,.25,4])
        self.assertEqual(outcomes["ndShape"],[1,2,2,4])
        self.assertNotEqual(outcomes["erf"],outcomes["mish"])
        # Changing optimizers resets incompatible moments but not parameter values.
        p.locator('[data-preset="trainable_core"]').click()
        p.locator("#trainCount").select_option("2")
        for kind,expected_file in [
            ("sgd_momentum","53-sgd-momentum-live-optimizer.png"),
            ("rmsprop","54-rmsprop-live-optimizer.png"),
            ("lion","55-lion-live-optimizer.png"),
            ("adamw","56-adamw-live-optimizer.png")
        ]:
            p.locator("#trainOptimizer").select_option(kind)
            self.assertEqual(p.locator("#trainOptimizer").input_value(),kind)
            p.locator("#trainStep").click()
            summary=p.locator("#trainSummary").inner_text()
            self.assertIn(kind,summary.lower())
            self.assertIn("Step 2",summary)
            with p.expect_download() as saved:
                p.locator("#save").click()
            data=json.loads(open(saved.value.path(),encoding="utf-8").read())
            param=next(n for n in data["nodes"] if n["id"]=="shared_weight")
            self.assertEqual(param["params"]["optimizer"]["kind"],kind)
            self.assertEqual(param["params"]["optimizer"]["t"],2)
            p.locator("#trainSummary").scroll_into_view_if_needed()
            self.shot(expected_file)
        # Restore moments by project import and continue using the saved optimizer.
        with p.expect_download() as exported:
            p.locator("#save").click()
        p.locator("#load").set_input_files(exported.value.path())
        self.assertEqual(p.locator("#trainOptimizer").input_value(),"adamw")
        p.locator("#trainStep").click()
        self.assertIn("AdamW",p.locator("#trainSummary").inner_text().replace("adamw","AdamW"))

    def test_lora_paper_patch_rewrites_real_graph_and_trains(self):
        p=self.page
        p.locator('[data-preset="autodiff_core"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),11)
        p.locator('[data-node="project"] .block-head').click()
        self.assertFalse(p.locator("#applyLoRA").is_disabled())
        p.locator("#applyLoRA").click()
        self.assertEqual(p.locator("#blocks .block").count(),17)
        self.assertIn("LoRA",p.locator("#workspaceTitle").inner_text())
        self.assertIn("LoRA",p.locator("#blockKnowledge").inner_text())
        self.assertEqual(p.locator('[data-node="project"]').get_attribute("data-paper-patch"),"lora")
        self.assertIn("前向初始化等价",p.locator("#status").inner_text())
        self.assertFalse(p.locator("#trainStep").is_disabled())
        p.locator("#trainStep").click()
        self.assertIn("AdamW step 1",p.locator("#trainSummary").inner_text())
        with p.expect_download() as d:
            p.locator("#save").click()
        obj=json.load(open(d.value.path(),encoding="utf-8"))
        self.assertEqual(len(obj["paperPatches"]),1)
        self.assertEqual(obj["paperPatches"][0]["paper"],"lora")
        self.shot("54-lora-paper-rewrite-trainable-graph.png")

    def test_loopformer_consistency_mechanism_and_industry_why_tradeoff(self):
        p=self.page
        p.locator('[data-preset="loopformer_consistency"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),61)
        self.assertEqual(p.locator('[data-node="loss"]').get_attribute("data-readiness"),"ready")
        p.locator('[data-node="short_consistency"] .block-head').click()
        self.assertIn("为什么这么设计",p.locator("#blockKnowledge").inner_text())
        self.assertIn("LoopFormer",p.locator("#blockKnowledge").inner_text())
        self.assertIn("stop_gradient",p.locator("#blockKnowledge").inner_text())
        self.assertIn("尚未实现",p.locator("#blockKnowledge").inner_text())
        p.locator('[data-node="Wq"] .block-head').click()
        self.assertIn("不共享同深度对照",p.locator("#blockKnowledge").inner_text())
        self.assertIn("168",p.locator("#blockKnowledge").inner_text())
        self.assertIn("440",p.locator("#blockKnowledge").inner_text())
        self.assertIn("MatMul MAC",p.locator("#blockKnowledge").inner_text())
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        self.assertIn("✓ 已运行",p.locator("#status").inner_text())
        p.locator('[data-node="loss"] .block-head').click()
        self.assertFalse(p.locator("#backward").is_disabled())
        p.locator("#backward").click()
        self.assertIn("反向传播",p.locator("#status").inner_text())
        p.locator("#trainCount").select_option("2")
        p.locator("#trainStep").click()
        self.assertIn("已提交 2 步",p.locator("#status").inner_text())
        with p.expect_download() as d:
            p.locator("#save").click()
        exported=json.loads(Path(d.value.path()).read_text(encoding="utf8"))
        self.assertEqual(len(exported["nodes"]),61)
        params=[n for n in exported["nodes"] if n["kind"]=="parameter"]
        self.assertEqual(len(params),8)
        self.assertTrue(all(n["params"]["optimizer"]["t"]==2 for n in params))
        p.locator("#pythonBackend").select_option("numpy")
        with p.expect_download() as d:
            p.locator("#exportPython").click()
        numpy=Path(d.value.path()).read_text(encoding="utf8")
        self.assertIn('v["short_consistency"]',numpy)
        p.locator("#pythonBackend").select_option("torch")
        with p.expect_download() as d:
            p.locator("#exportPython").click()
        torch=Path(d.value.path()).read_text(encoding="utf8")
        self.assertIn('.detach().clone()',torch)
        self.assertIn('self.loss_id = "loss"',torch)
        self.assertEqual(torch.count('torch.nn.Parameter('),8)
        p.locator('[data-node="Wq"] .block-head').click()
        self.shot("62-loopformer-inspired-real-consistency-why-and-cost.png")

    def test_loop_depth_weighted_supervision_keeps_trained_parameter_identity(self):
        p=self.page
        p.locator('[data-preset="loop_transformer_2"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),57)
        self.assertFalse(p.locator("#applyLoopDepth").is_disabled())
        self.assertEqual(p.locator("#loopDepth").input_value(),"2")
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        self.assertIn("✓ 已运行",p.locator("#status").inner_text())
        p.locator('[data-node="loss"] .block-head').click()
        p.locator("#backward").click()
        self.assertIn("反向传播",p.locator("#status").inner_text())
        p.locator("#trainCount").select_option("2")
        p.locator("#trainStep").click()
        self.assertIn("已提交 2 步",p.locator("#status").inner_text())
        with p.expect_download() as d:
            p.locator("#save").click()
        before=json.loads(Path(d.value.path()).read_text(encoding="utf-8"))
        params_before={n["id"]:n["params"] for n in before["nodes"] if n["kind"]=="parameter"}
        self.assertEqual(len(params_before),8)
        self.assertTrue(all(n["optimizer"]["t"]==2 for n in params_before.values()))
        self.assertAlmostEqual(sum(n["params"]["factor"] for n in before["nodes"] if n["id"] in ("weight0","weight1")),1)
        p.locator("#loopDepth").select_option("3")
        p.locator("#applyLoopDepth").click()
        self.assertIn("循环深度 2 → 3",p.locator("#status").inner_text())
        self.assertEqual(p.locator("#blocks .block").count(),79)
        with p.expect_download() as d:
            p.locator("#save").click()
        after=json.loads(Path(d.value.path()).read_text(encoding="utf-8"))
        self.assertEqual(params_before,{n["id"]:n["params"] for n in after["nodes"] if n["kind"]=="parameter"})
        self.assertAlmostEqual(sum(n["params"]["factor"] for n in after["nodes"] if n["id"] in ("weight0","weight1","weight2")),1)
        p.locator('[data-node="loss"] .block-head').click()
        p.locator("#run").click()
        p.locator("#pythonBackend").select_option("torch")
        with p.expect_download() as d:
            p.locator("#exportPython").click()
        source=Path(d.value.path()).read_text(encoding="utf-8")
        self.assertIn('v["r2_ce"]',source)
        self.assertIn('self.loss_id = "loss"',source)
        self.assertEqual(source.count('torch.nn.Parameter('),8)
        self.shot("61-loop-weighted-depth-3-real-torch.png")
        p.locator("#loopDepth").select_option("1")
        p.locator("#applyLoopDepth").click()
        self.assertEqual(p.locator("#blocks .block").count(),35)
        with p.expect_download() as d:
            p.locator("#save").click()
        one=json.loads(Path(d.value.path()).read_text(encoding="utf-8"))
        self.assertEqual(params_before,{n["id"]:n["params"] for n in one["nodes"] if n["kind"]=="parameter"})
        self.assertFalse(p.locator("#applyLoopDepth").is_disabled())

    def test_looped_transformer_shared_parameters_recurrent_depth_and_real_exports(self):
        p=self.page
        for recipe,depth,count in (("loop_transformer_two",2,56),("loop_transformer_three",3,75)):
            p.locator('[data-preset="'+recipe+'"]').click()
            self.assertEqual(p.locator("#blocks .block").count(),count)
            self.assertEqual(p.locator('[data-node="loss"]').get_attribute("data-readiness"),"ready")
            p.locator("#run").click()
            self.assertIn("✓ 已运行",p.locator("#status").inner_text())
            p.locator('[data-node="loss"] .block-head').click()
            p.locator("#backward").click()
            self.assertIn("反向传播",p.locator("#status").inner_text())
            self.assertEqual(p.locator('[data-node="r1_gate"]').get_attribute("data-industrial-ref"),"loop_ffn")
            self.assertEqual(p.locator('[data-node="r2_prob"]').get_attribute("data-industrial-ref"),"loop_causal")
            p.locator('[data-node="wq"] .block-head').click()
            self.assertIn("跨深度共享",p.locator("#blockKnowledge").inner_text())
            p.locator('[data-node="r1_qr"] .block-head').hover()
            p.locator("#paperHoverInspector").wait_for(state="visible",timeout=5000)
            self.assertIn("Looped Causal Attention",p.locator("#paperHoverInspector").inner_text())
            p.locator('[data-node="loss"] .block-head').click()
            p.locator("#trainCount").select_option("2")
            p.locator("#trainStep").click()
            self.assertIn("已提交 2 步",p.locator("#status").inner_text())
            with p.expect_download() as d:
                p.locator("#save").click()
            saved=json.loads(Path(d.value.path()).read_text(encoding="utf8"))
            weights=[n for n in saved["nodes"] if n["kind"]=="parameter"]
            self.assertEqual(len(weights),8)
            self.assertEqual({n["id"] for n in weights},
              {"embed","wq","wk","wv","wo","wgate","wup","wdown"})
            self.assertTrue(all(n["params"]["optimizer"]["t"]==2 for n in weights))
            for backend in ("numpy","torch"):
                p.locator("#pythonBackend").select_option(backend)
                with p.expect_download() as d:
                    p.locator("#exportPython").click()
                source=Path(d.value.path()).read_text(encoding="utf8")
                self.assertIn('r1_norm',source)
                self.assertIn('r'+str(depth)+'_state',source)
                self.assertIn('wq',source)
                self.assertIn('wdown',source)
                if backend=="torch":
                    self.assertIn("torch.nn.Parameter",source)
                    self.assertIn("self._optimizer.load_state_dict",source)
                else:
                    self.assertIn("np",source)
            self.shot("60-loop-transformer-"+str(depth)+"-recurrences-real-code.png")

    def test_industry_qwen3_inspired_gqa_rope_tied_head_executable(self):
        p=self.page
        p.locator('[data-preset="industrial_gqa_rope_tied"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),45)
        self.assertEqual(p.locator('[data-node="loss"]').get_attribute("data-readiness"),"ready")
        p.locator("#run").click()
        self.assertIn("✓ 已运行",p.locator("#status").inner_text())
        p.locator('[data-node="loss"] .block-head').click()
        p.locator("#backward").click()
        self.assertIn("反向传播",p.locator("#status").inner_text())
        p.locator("#trainCount").select_option("2")
        p.locator("#trainStep").click()
        self.assertIn("已提交 2 步",p.locator("#status").inner_text())
        with p.expect_download() as item:
            p.locator("#save").click()
        saved=json.loads(Path(item.value.path()).read_text(encoding="utf-8"))
        weights=[n for n in saved["nodes"] if n["kind"]=="parameter"]
        self.assertEqual({n["id"] for n in weights},{"embed","qgain","kgain","wq","wk","wv","wo"})
        self.assertTrue(all(n["params"]["optimizer"]["t"]==2 for n in weights))
        self.assertEqual(next(n for n in saved["nodes"] if n["id"]=="tied_head")["kind"],"transpose2d")
        p.locator("#pythonBackend").select_option("numpy")
        with p.expect_download() as item:
            p.locator("#exportPython").click()
        numpy=Path(item.value.path()).read_text(encoding="utf-8")
        self.assertIn("_rope2d",numpy)
        self.assertIn("np.concatenate",numpy)
        p.locator("#pythonBackend").select_option("torch")
        with p.expect_download() as item:
            p.locator("#exportPython").click()
        torch=Path(item.value.path()).read_text(encoding="utf-8")
        self.assertIn("_rope2d",torch)
        self.assertIn("torch.nn.Parameter",torch)
        self.assertIn("torch.cat",torch)
        self.assertIn("self._optimizer.load_state_dict",torch)
        self.assertNotIn('self.parameters["lm_head"]',torch)
        p.locator('[data-node="qgain"] .block-head').click()
        self.assertIn("Qwen3",p.locator("#blockKnowledge").inner_text())
        self.assertIn("q_norm",p.locator("#blockKnowledge").inner_text())
        p.locator('[data-node="tied_head"] .block-head').click()
        self.assertIn("tie_word_embeddings",p.locator("#blockKnowledge").inner_text())
        self.assertEqual(p.locator('[data-node="q0r"]').get_attribute("data-industrial-ref"),"rope")
        p.locator('[data-node="q0r"] .block-head').hover()
        p.locator("#paperHoverInspector").wait_for(state="visible",timeout=5000)
        self.assertIn("RoPE",p.locator("#paperHoverInspector").inner_text())
        self.assertIn("Qwen3",p.locator("#paperHoverInspector").inner_text())
        self.shot("59-real-qwen-inspired-gqa-rope-tied-weights.png")

    def test_trainable_causal_transformer_full_graph_and_export(self):
        p=self.page
        p.locator('[data-preset="causal_transformer_lm"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),23)
        self.assertEqual(p.locator('[data-node="loss"]').get_attribute("data-readiness"),"ready")
        p.locator("#run").click()
        self.assertIn("✓ 已运行",p.locator("#status").inner_text())
        p.locator('[data-node="loss"] .block-head').click()
        self.assertFalse(p.locator("#backward").is_disabled())
        p.locator("#backward").click()
        self.assertIn("反向传播",p.locator("#status").inner_text())
        p.locator("#trainCount").select_option("2")
        p.locator("#trainStep").click()
        self.assertIn("已提交 2 步",p.locator("#status").inner_text())
        self.assertIn("AdamW step 2",p.locator("#trainSummary").inner_text())
        with p.expect_download() as d:
            p.locator("#save").click()
        graph=json.loads(Path(d.value.path()).read_text(encoding="utf-8"))
        params=[n for n in graph["nodes"] if n["kind"]=="parameter"]
        self.assertEqual({n["id"] for n in params},{"embedding_table","wq","wk","wv","lm_head"})
        self.assertTrue(all(n["params"]["optimizer"]["t"]==2 for n in params))
        p.locator("#pythonBackend").select_option("numpy")
        with p.expect_download() as d:
            p.locator("#exportPython").click()
        numpy=Path(d.value.path()).read_text(encoding="utf-8")
        self.assertIn("_masked_softmax",numpy)
        self.assertIn("_cross_entropy",numpy)
        p.locator("#pythonBackend").select_option("torch")
        with p.expect_download() as d:
            p.locator("#exportPython").click()
        torch=Path(d.value.path()).read_text(encoding="utf-8")
        self.assertIn("torch.nn.Parameter",torch)
        self.assertIn("_masked_softmax",torch)
        self.assertIn("F.cross_entropy",torch)
        self.assertIn("self._optimizer.load_state_dict",torch)
        self.assertIn("PyTorch CPU",p.locator("#status").inner_text())
        p.locator('[data-node="probs"] .block-head').click()
        self.shot("58-real-executable-causal-transformer.png")

    def test_real_token_embedding_crossentropy_training_and_code_export(self):
        p=self.page
        p.locator('[data-preset="token_lm_trainable"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),9)
        self.assertEqual(p.locator('[data-node="loss"]').get_attribute("data-readiness"),"ready")
        p.locator("#run").click()
        self.assertIn("✓ 已运行",p.locator("#status").inner_text())
        p.locator('[data-node="loss"] .block-head').click()
        self.assertFalse(p.locator("#backward").is_disabled())
        p.locator("#backward").click()
        self.assertIn("反向传播",p.locator("#status").inner_text())
        p.locator("#trainCount").select_option("2")
        p.locator("#trainStep").click()
        self.assertIn("已提交 2 步",p.locator("#status").inner_text())
        self.assertIn("AdamW step 2",p.locator("#trainSummary").inner_text())
        with p.expect_download() as item:
            p.locator("#save").click()
        saved=json.loads(Path(item.value.path()).read_text(encoding="utf-8"))
        table=next(n for n in saved["nodes"] if n["id"]=="embedding_table")
        self.assertEqual(table["params"]["optimizer"]["t"],2)
        p.locator("#pythonBackend").select_option("numpy")
        with p.expect_download() as item:
            p.locator("#exportPython").click()
        np_source=Path(item.value.path()).read_text(encoding="utf-8")
        self.assertIn("_cross_entropy",np_source)
        self.assertIn("_gather_rows",np_source)
        p.locator("#pythonBackend").select_option("torch")
        with p.expect_download() as item:
            p.locator("#exportPython").click()
        torch_source=Path(item.value.path()).read_text(encoding="utf-8")
        self.assertIn("F.cross_entropy",torch_source)
        self.assertIn("torch.nn.Parameter",torch_source)
        self.assertIn("self._optimizer.load_state_dict",torch_source)
        self.assertIn("PyTorch CPU",p.locator("#status").inner_text())
        self.shot("57-trainable-token-embedding-ce-export.png")

    def test_free_mixed_loss_mae_optimizer_and_export(self):
        p=self.page
        p.locator('[data-preset="trainable_swiglu"]').click()
        p.locator('[data-add="mae_loss"]').click()
        mae=p.locator("#blocks .block").last.get_attribute("data-node")
        self.assertEqual(p.locator('[data-node="'+mae+'"]').get_attribute("data-readiness"),"missing")
        for upstream,port in (("down","prediction"),("target","target")):
            p.locator('[data-out="'+upstream+'"]').click()
            p.locator('[data-in="'+mae+'"][data-port="'+port+'"]').click()
        self.assertEqual(p.locator('[data-node="'+mae+'"]').get_attribute("data-readiness"),"ready")
        p.locator("#studyMode").select_option("numeric")
        p.locator('[data-node="'+mae+'"] .block-head').click()
        self.assertFalse(p.locator("#trainStep").is_disabled())
        p.locator("#trainStep").click()
        self.assertIn("已提交 1 步",p.locator("#status").inner_text())
        self.assertIn("AdamW step 1",p.locator("#trainSummary").inner_text())
        p.locator("#pythonBackend").select_option("torch")
        with p.expect_download() as downloaded:
            p.locator("#exportPython").click()
        source=Path(downloaded.value.path()).read_text(encoding="utf-8")
        self.assertIn(".abs().mean()",source)
        self.assertIn("loss.sum().backward()",source)
        self.assertIn("torch.optim.AdamW",source)
        self.assertIn("PyTorch CPU",p.locator("#status").inner_text())
        self.shot("55-real-mae-graph-training-export.png")

    def test_torch_export_is_real_autograd_and_not_fake_fallback(self):
        p=self.page
        p.locator('[data-preset="trainable_attention"]').click()
        p.locator("#pythonBackend").select_option("torch")
        with p.expect_download() as download:
            p.locator("#exportPython").click()
        code=open(download.value.path(),encoding="utf-8").read()
        self.assertIn("import torch",code)
        self.assertIn("torch.nn.Parameter",code)
        self.assertIn("loss.sum().backward()",code)
        self.assertIn("self._optimizer.load_state_dict",code)
        self.assertIn("GraphRuntime",code)
        self.assertIn("PyTorch CPU",p.locator("#status").inner_text())
        self.assertIn("PyTorch CPU",p.locator("#pythonExportCaption").inner_text())
        p.locator('[data-preset="trainable_moe"]').click()
        with p.expect_download() as download:
            p.locator("#exportPython").click()
        code=Path(download.value.path()).read_text(encoding="utf-8")
        self.assertIn("_expert_linear_bank",code)
        self.assertIn("torch.einsum",code)
        self.assertIn("self.parameters",code)
        self.assertIn("PyTorch CPU",p.locator("#status").inner_text())
        p.locator("#pythonExportPanel summary").click()
        self.assertIn("_expert_linear_bank",p.locator("#pythonExportSource").inner_text())
        self.shot("56-moe-executable-pytorch-code.png")
        p.locator('[data-preset="grpo_objective"]').click()
        p.locator("#exportPython").click()
        self.assertIn("不支持",p.locator("#status").inner_text())

    def test_export_real_python_code_whole_and_partial_graph(self):
        p=self.page
        p.locator('[data-preset="nd_cached_gqa_decode"]').click()
        with p.expect_download() as downloaded:
            p.locator("#exportPython").click()
        artifact=downloaded.value
        self.assertTrue(artifact.suggested_filename.endswith(".py"))
        script=Path(artifact.path()).read_text(encoding="utf-8")
        self.assertIn("import numpy as np",script)
        self.assertIn("def run_graph(overrides=None):",script)
        self.assertIn("_rope(",script)
        self.assertIn("_causal(",script)
        self.assertIn("np.concatenate(",script)
        self.assertIn("return v",script)
        self.assertFalse(p.locator("#pythonExportPanel").is_hidden())
        p.locator("#pythonExportPanel summary").click()
        self.assertIn("nd_cache_append",p.locator("#pythonExportSource").inner_text())
        p.locator("#pythonExportPanel").scroll_into_view_if_needed()
        self.shot("53-export-executable-numpy-rope-kv-code.png")
        p.locator('[data-preset="blank"]').click()
        p.locator('[data-add="silu"]').click()   # Incomplete neighbor.
        p.locator("#exportPython").click()
        self.assertIn("Python 导出失败",p.locator("#status").inner_text())
        p.locator("#exportScope").select_option("selected")
        p.locator('[data-node="x"] .block-foot').click()
        with p.expect_download() as sliced:
            p.locator("#exportPython").click()
        code=Path(sliced.value.path()).read_text(encoding="utf-8")
        self.assertIn('v["x"]',code)
        self.assertNotIn('v["silu',code)
        self.assertIn("所选分支",p.locator("#status").inner_text())

    def test_four_axis_cached_gqa_rope_incremental_decode(self):
        p=self.page
        p.locator('[data-preset="nd_cached_gqa_decode"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),21)
        self.assertIn("21/21",p.locator("#graphReadiness").inner_text())
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        names=p.evaluate("""() => window.SeetheTruthLearning.order(
          window.SeetheTruthLearning.preset('nd_cached_gqa_decode')).map(n=>n.id)""")
        self.assertEqual(len(names),21)
        self.assertEqual(names[-1],"context")
        p.locator('#trace [data-step="'+str(names.index("cached_k"))+'"]').click()
        self.assertIn("1×2×3×4",p.locator("#metrics").inner_text().replace(" ",""))
        self.assertEqual(p.locator("#matrix .cell").count(),12)
        p.locator('#matrix [data-nd-slice="1"]').select_option("1")
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("50-rope-gqa-kv-cache-prefill-append.png")
        p.locator('#trace [data-step="'+str(names.index("probabilities"))+'"]').click()
        self.assertIn("1×4×1×3",p.locator("#metrics").inner_text().replace(" ",""))
        self.assertEqual(p.locator("#matrix .cell").count(),3)
        p.locator('#matrix [data-nd-slice="1"]').select_option("2")
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("51-rope-cached-query-attention-probability.png")
        p.locator('#trace [data-step="'+str(names.index("context"))+'"]').click()
        self.assertIn("1×4×1×4",p.locator("#metrics").inner_text().replace(" ",""))
        self.assertEqual(p.locator("#matrix .cell").count(),4)
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("52-cached-gqa-decoding-context.png")
        verification=p.evaluate("""() => {
          const L=window.SeetheTruthLearning,ND=window.SeetheTruthND,
                out=Object.fromEntries(L.evaluate(L.preset('nd_cached_gqa_decode')).steps.map(s=>[s.id,s.values]));
          const K=ND.repeatKVHeads(ND.rope4(out.full_k,0,4),4),
                Q=ND.rope4(out.full_q,0,4),V=ND.repeatKVHeads(out.full_v,4);
          const full=ND.matmul4(ND.softmaxLast(
            ND.causalMask4(ND.scale(ND.matmul4(Q,ND.permute(K,[0,1,3,2])),.5))),V);
          const expected=ND.flatten(ND.cacheRead4(full,2,1)),
                actual=ND.flatten(out.context);
          return {maxError:Math.max(...actual.map((v,i)=>Math.abs(v-expected[i]))),
                  shape:ND.shape(out.context),
                  kvCacheShape:ND.shape(out.cached_k),
                  knownFlops:L.evaluate(L.preset('nd_cached_gqa_decode')).totalKnownFlops};
        }""")
        self.assertEqual(verification["shape"],[1,4,1,4])
        self.assertEqual(verification["kvCacheShape"],[1,2,3,4])
        self.assertEqual(verification["knownFlops"],192)
        self.assertLess(verification["maxError"],1e-12)

    def test_composed_causal_multihead_attention_probabilities_and_context(self):
        p=self.page
        p.locator('[data-preset="nd_causal_attention"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),9)
        self.assertIn("9/9",p.locator("#graphReadiness").inner_text())
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        steps=p.evaluate("""() => window.SeetheTruthLearning.order(
            window.SeetheTruthLearning.preset('nd_causal_attention')).map(n=>n.id)""")
        self.assertEqual(len(steps),9)
        self.assertEqual(steps[-1],"context")
        prob=steps.index("probabilities")
        p.locator('#trace [data-step="'+str(prob)+'"]').click()
        self.assertIn("1×2×3×3",p.locator("#metrics").inner_text().replace(" ",""))
        self.assertEqual(p.locator("#matrix .cell").count(),9)
        self.assertEqual(p.locator("#matrix .cell").all_inner_texts()[1:3],["0.00","0.00"])
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("43-causal-attention-head0-probability-mask.png")
        before=p.locator("#matrix .cell").all_inner_texts()
        p.locator('#matrix [data-nd-slice="1"]').select_option("1")
        after=p.locator("#matrix .cell").all_inner_texts()
        self.assertNotEqual(before,after)
        self.assertEqual(after[1:3],["0.00","0.00"])
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("44-causal-attention-head1-probabilities.png")
        p.locator('#trace [data-step="'+str(steps.index("context"))+'"]').click()
        self.assertIn("1×2×3×4",p.locator("#metrics").inner_text().replace(" ",""))
        self.assertEqual(p.locator("#matrix .cell").count(),12)
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("45-causal-attention-context-output.png")
        result=p.evaluate("""() => {
          const ND=window.SeetheTruthND,L=window.SeetheTruthLearning,
                run=L.evaluate(L.preset('nd_causal_attention')),
                data=Object.fromEntries(run.steps.map(s=>[s.id,s.values]));
          return {prob:data.probabilities,context:data.context,knownFlops:run.totalKnownFlops,
                  contextShape:ND.shape(data.context)};
        }""")
        self.assertEqual(result["contextShape"],[1,2,3,4])
        self.assertEqual(result["knownFlops"],288)
        for h in range(2):
            for t in range(3):
                probs=result["prob"][0][h][t]
                self.assertAlmostEqual(sum(probs),1,places=12)
                self.assertTrue(all(probs[j]==0 for j in range(t+1,3)))

    def test_four_axis_multidimensional_value_slices_and_real_batched_gemm(self):
        p=self.page
        p.locator('[data-preset="nd_attention_lesson"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),9)
        self.assertIn("四轴",p.locator("#workspaceTitle").inner_text())
        self.assertIn("9/9",p.locator("#graphReadiness").inner_text())
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        self.assertIn("1 / 9",p.locator("#stepInfo").inner_text())
        self.assertEqual(p.locator("#matrix [data-nd-slice]").count(),2)
        self.assertEqual(p.locator("#matrix .cell").count(),12)
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("39-four-axis-tensor-source.png")
        head=p.locator('#matrix [data-nd-slice="1"]')
        first=p.locator("#matrix .cell").all_inner_texts()
        head.select_option("1")
        second=p.locator("#matrix .cell").all_inner_texts()
        self.assertNotEqual(first,second,"Head=0 and Head=1 must show different real tensor values")
        self.assertEqual(p.locator("#matrix .cell").count(),12)
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("40-four-axis-head-slice-one.png")
        # Inspect batched GEMM; mathematical output must contain two Heads.
        i=p.evaluate("""() => window.SeetheTruthLearning.order(
           window.SeetheTruthLearning.preset('nd_attention_lesson')).findIndex(n=>n.id==='batch_scores')""")
        for _ in range(i):p.locator("#next").click()
        self.assertIn("1×2×3×3",p.locator("#metrics").inner_text().replace(" ",""))
        self.assertEqual(p.locator("#matrix .cell").count(),9)
        p.locator('#matrix [data-nd-slice="1"]').select_option("1")
        p.locator("#matrix").scroll_into_view_if_needed()
        self.assertIn("144 FLOPs",p.locator("#metrics").inner_text())
        self.shot("41-batched-head-matmul-output.png")
        for _ in range(8-i):p.locator("#next").click()
        self.assertIn("9 / 9",p.locator("#stepInfo").inner_text())
        self.assertIn("1×2×1×3",p.locator("#metrics").inner_text().replace(" ",""))
        self.assertEqual(p.locator("#matrix .cell").count(),3)
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("42-four-axis-permute-numeric-result.png")

    def test_training_selected_loss_with_unfinished_side_branch(self):
        p=self.page
        p.locator('[data-preset="trainable_core"]').click()
        p.locator('[data-add="relu"]').click()
        stray=p.locator('#blocks .block').last.get_attribute("data-node")
        self.assertEqual(p.locator('[data-node="'+stray+'"]').get_attribute("data-readiness"),"missing")
        p.locator("#trainCount").select_option("2")
        p.locator('[data-node="loss"] .block-foot').click()
        p.locator("#trainStep").click()
        self.assertIn("AdamW step 2",p.locator("#trainSummary").inner_text())
        self.assertIn("Loss",p.locator("#trainSummary").inner_text())
        self.assertEqual(p.locator('#blocks .block').count(),12)
        self.assertEqual(p.locator('[data-node="'+stray+'"]').get_attribute("data-readiness"),"missing")
        self.assertIn("分支 loss",p.locator("#stepInfo").inner_text())
        with p.expect_download() as saved:
            p.locator("#save").click()
        data=json.loads(open(saved.value.path(),encoding="utf-8").read())
        self.assertEqual(len(data["nodes"]),12)
        self.assertEqual(next(n for n in data["nodes"] if n["id"]=="shared_weight")["params"]["optimizer"]["t"],2)
        self.assertEqual(next(n for n in data["nodes"] if n["id"]==stray)["kind"],"relu")
        p.locator("#trainSummary").scroll_into_view_if_needed()
        self.shot("38-train-partial-graph-with-side-branch.png")
        p.locator("#undo").click()
        with p.expect_download() as prior:
            p.locator("#save").click()
        previous=json.loads(open(prior.value.path(),encoding="utf-8").read())
        self.assertEqual(next(n for n in previous["nodes"] if n["id"]=="shared_weight")["params"]["optimizer"]["t"],0)
        self.assertEqual(len(previous["nodes"]),12)

    def test_partial_canvas_branch_execution_and_advisory_port_previews(self):
        p=self.page
        p.locator('[data-preset="blank"]').click()
        p.locator('[data-add="transpose2d"]').click()
        transpose=p.locator('#blocks .block').last.get_attribute("data-node")
        p.locator('[data-out="x"]').click()
        self.assertEqual(p.locator('[data-in="'+transpose+'"][data-port="x"]').get_attribute("data-preview"),"compatible")
        p.locator('[data-in="'+transpose+'"][data-port="x"]').click()
        p.locator('[data-add="matmul"]').click()
        matmul=p.locator('#blocks .block').last.get_attribute("data-node")
        p.locator('[data-out="x"]').click()
        p.locator('[data-in="'+matmul+'"][data-port="a"]').click()
        p.locator('[data-out="x"]').click()
        self.assertEqual(p.locator('[data-in="'+matmul+'"][data-port="b"]').get_attribute("data-preview"),"incompatible")
        p.locator('[data-out="'+transpose+'"]').click()
        self.assertEqual(p.locator('[data-in="'+matmul+'"][data-port="b"]').get_attribute("data-preview"),"compatible")
        p.locator('[data-in="'+matmul+'"][data-port="b"]').click()
        p.locator('[data-add="softmax"]').click()
        soft=p.locator('#blocks .block').last.get_attribute("data-node")
        p.locator('[data-out="'+matmul+'"]').click()
        p.locator('[data-in="'+soft+'"][data-port="x"]').click()
        p.locator('[data-add="relu"]').click()
        stray=p.locator('#blocks .block').last.get_attribute("data-node")
        self.assertEqual(p.locator('[data-node="'+stray+'"]').get_attribute("data-readiness"),"missing")
        p.locator("#run").click()
        self.assertIn("无法运行",p.locator("#status").inner_text())
        self.assertIn("缺少",p.locator("#graphReadiness").inner_text())
        p.locator('[data-node="'+soft+'"] .block-foot').click()
        p.locator("#runBranch").click()
        self.assertIn("4 / 4",p.locator("#stepInfo").inner_text())
        self.assertIn("分支",p.locator("#stepInfo").inner_text())
        self.assertEqual(p.locator("#matrix .cell").count(),9)
        self.assertIn("保留 1 个",p.locator("#status").inner_text())
        self.assertEqual(p.locator("#blocks .block").count(),5)
        p.locator("#matrix").scroll_into_view_if_needed()
        self.shot("37-free-canvas-partial-branch-execution.png")
        p.locator('[data-out="x"]').click()
        p.locator('[data-in="'+stray+'"][data-port="x"]').click()
        self.assertIn("输入完整",p.locator("#graphReadiness").inner_text())
        p.locator("#run").click()
        self.assertIn("1 / 5",p.locator("#stepInfo").inner_text())
        self.assertEqual(p.locator("#blocks .block").count(),5)

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
        p.locator("#run").click()
        for _ in range(3):p.locator("#next").click()
        self.assertIn("4 / 4",p.locator("#stepInfo").inner_text())
        self.assertEqual(p.locator("#matrix .cell").count(),9)
        p.locator("#matrix").scroll_into_view_if_needed()
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
        loss_count=p.evaluate("window.SeetheTruthLearning.KINDS.filter(k=>window.SeetheTruthLearning.defs[k].category==='loss').length")
        self.assertEqual(p.locator("#palette [data-add]").count(),loss_count)
        self.assertEqual(p.locator("#blockCount").inner_text(),f"{loss_count} / {size}")
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
