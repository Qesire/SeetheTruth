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
        self.shot("22-grpo-backward-current-policy-gradients.png")
        i=p.evaluate("""() => window.SeetheTruthLearning.order(
             window.SeetheTruthLearning.preset('grpo_objective')).findIndex(n=>n.id==='surrogate')""")
        p.locator('#trace [data-step="'+str(i)+'"]').click()
        self.assertIn("clip",p.locator("#gradientInspector").inner_text().lower())
        self.assertIn("停止梯度",p.locator("#gradientInspector").inner_text())
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
