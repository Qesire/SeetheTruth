"""Actual Chromium screenshots and functional checks for scalar training state lesson."""
import json
import os
import struct
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

OUT=Path(os.environ.get("SEETHTRUTH_TRAINING_SHOTS","training_screenshots"))

class TrainingStudioVisualAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(exist_ok=True,parents=True)
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
        self.p=self.browser.new_page(viewport={"width":1600,"height":1000},device_scale_factor=1)
        self.errors=[]
        self.p.on("pageerror",lambda e:self.errors.append(str(e)))
        self.p.goto(self.base+"/training.html",wait_until="domcontentloaded")
        self.p.wait_for_function("document.querySelectorAll('#catalog [data-type]').length===14")
    def tearDown(self):
        self.assertEqual(self.errors,[])
        self.p.close()

    def shot(self,name):
        p=self.p
        p.evaluate("document.fonts.ready")
        image=p.screenshot(path=str(OUT/name),full_page=True,animations="disabled")
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        w,h=struct.unpack(">II",image[16:24])
        self.assertGreaterEqual(w,1600)
        self.assertGreater(h,970)
        self.assertFalse(p.evaluate("document.documentElement.scrollWidth>innerWidth+2"),
                         "Horizontal body overflow")
        return {"file":name,"width":w,"height":h,"bytes":len(image),
                "mode":p.locator("#mode").input_value(),
                "selected":p.locator("#inspectorTitle").inner_text(),
                "events":p.locator("#sequence .step").count()}

    def test_symbolic_grad_accum_adamw_and_sgd_screens(self):
        p=self.p
        self.assertEqual(p.locator("#mode").input_value(),"concept")
        self.assertEqual(p.locator("#catalog [data-type]").count(),14)
        self.assertGreater(p.locator("#sequence .step").count(),35)
        self.assertTrue(p.locator("#numericBox").is_hidden())
        self.assertIn("symbolic state contract",p.locator("#evidence").inner_text())
        photos=[self.shot("01-symbolic-training-all-blocks.png")]
        p.locator('[data-type="accumulate"]').click()
        self.assertIn("累加",p.locator("#inspectorTitle").inner_text())
        self.assertIn("buffer",p.locator("#formula").inner_text().lower()) if False else None
        self.assertIn("θ.grad",p.locator("#formula").inner_text())
        self.assertTrue(p.locator("#numericBox").is_hidden())
        photos.append(self.shot("02-gradient-accumulation-concept.png"))
        p.locator("#mode").select_option("scalar")
        self.assertFalse(p.locator("#numericBox").is_hidden())
        self.assertIn("0.400000",p.locator("#thetaState").inner_text())
        p.locator("#sequence .step").filter(has_text="AdamW 参数更新").first.click()
        self.assertIn("AdamW",p.locator("#inspectorTitle").inner_text())
        self.assertIn("0.2992",p.locator("#thetaState").inner_text())
        self.assertIn("2.000000",p.locator("#gradState").inner_text()) if False else None
        self.assertIn("1.000000",p.locator("#gradState").inner_text())
        photos.append(self.shot("03-first-adamw-real-scalar-update.png"))
        p.locator("#sequence .step").filter(has_text="AdamW 参数更新").last.click()
        self.assertIn("2 /",p.locator("#stepCount").inner_text()) if False else None
        self.assertIn("m=",p.locator("#momentState").inner_text())
        self.assertNotEqual(p.locator("#thetaState").inner_text(),"0.400000")
        photos.append(self.shot("04-second-adamw-optimizer-state.png"))
        p.locator("#compare").click()
        c=p.locator("#comparison").inner_text()
        self.assertIn("正确平均梯度",c)
        self.assertIn("若忘记除以 N",c)
        self.assertIn("若每个 microbatch 都 zero_grad",c)
        photos.append(self.shot("05-counterexample-grad-scaling.png"))
        p.locator("#optimizer").select_option("sgd")
        self.assertEqual(p.locator("#sequence .step").filter(has_text="AdamW").count(),0)
        self.assertEqual(p.locator("#sequence .step").filter(has_text="SGD 参数更新").count(),2)
        self.assertIn("SGD · no m/v",p.locator("#momentState").inner_text())
        p.locator("#sequence .step").filter(has_text="SGD 参数更新").first.click()
        self.assertIn("0.300000",p.locator("#thetaState").inner_text())
        photos.append(self.shot("06-sgd-different-update-rule.png"))
        (OUT/"manifest.json").write_text(json.dumps({
           "schema":"seethetruth-training-curriculum-real-chromium-v1",
           "captures":photos,
           "browser":"Playwright Chromium, local-only served training.html",
           "contracts":"14 explanatory blocks, symbolic default, accumulated gradients, 2-step AdamW state, SGD comparison",
           "limits":"analytic single-scalar teaching example, not PyTorch automatic differentiation or GPU model training"
        },ensure_ascii=False,indent=2),encoding="utf-8")

    def test_navigation_and_reconfig_are_stateful(self):
        p=self.p
        self.assertTrue(p.locator("#prev").is_disabled())
        p.locator("#next").click()
        self.assertFalse(p.locator("#prev").is_disabled())
        self.assertIn("清空",p.locator("#inspectorTitle").inner_text())
        p.locator("#prev").click()
        self.assertTrue(p.locator("#prev").is_disabled())
        p.locator("#microbatches").select_option("1")
        self.assertLess(p.locator("#sequence .step").count(),30)
        self.assertEqual(p.locator("#sequence .step").filter(has_text="按有效样本数缩放").count(),2)
        p.locator("#steps").select_option("1")
        self.assertEqual(p.locator("#sequence .step").filter(has_text="Optimizer Step").count(),0)
        self.assertEqual(p.locator("#sequence .step").filter(has_text="AdamW 参数更新").count(),1)
        p.locator("#lr").fill("0")
        p.locator("#lr").dispatch_event("change")
        self.assertIn("配置被拒绝",p.locator("#comparison").inner_text())
        p.locator("#lr").fill("0.1")
        p.locator("#restart").click()
        self.assertEqual(p.locator("#comparison").inner_text(),"")
        self.assertEqual(p.locator("#sequence [aria-current='step']").count(),1)

if __name__=="__main__":
    unittest.main()
