"""Browser screenshot proof for hierarchical and fully symbolic model drill-down.

All PNGs are captured by real Playwright Chromium after clicks, and tests
assert stage shape labels and actual leaf inspector contents, not placeholders.
"""
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

OUT=Path(os.environ.get("SEETHTRUTH_COMPOSITE_SHOTS","composite_screenshots"))

class CompositeHierarchyChromium(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True,exist_ok=True)
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
        cls.base="http://127.0.0.1:"+str(cls.server.server_port)
    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.server.shutdown()
        cls.server.server_close()
    def setUp(self):
        self.p=self.browser.new_page(viewport={"width":1600,"height":1000},device_scale_factor=1)
        self.errors=[]
        self.p.on("pageerror",lambda e:self.errors.append(str(e)))
        self.p.goto(self.base+"/learn.html",wait_until="domcontentloaded")
        self.p.wait_for_function("document.querySelectorAll('#palette [data-add]').length===45")
    def tearDown(self):
        self.assertEqual(self.errors,[],"JavaScript errors")
        self.p.close()
    def screenshot(self,name):
        p=self.p
        p.evaluate("document.fonts.ready")
        png=p.screenshot(path=str(OUT/name),full_page=True,animations="disabled")
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        width,height=struct.unpack(">II",png[16:24])
        self.assertGreaterEqual(width,1600)
        self.assertGreaterEqual(height,950)
        self.assertFalse(p.evaluate("document.documentElement.scrollWidth>innerWidth+2"),
                         "Horizontal page overflow after expanding hierarchy")
        return {"file":name,"width":width,"height":height,"bytes":len(png)}
    def test_decoder_gqa_attention_math_and_ffn_nested_drilldown(self):
        p=self.p
        self.assertEqual(p.locator("#studyMode").input_value(),"concept")
        p.locator("#compositeLaunch").click()
        self.assertTrue(p.locator("#viewport").is_hidden())
        self.assertIn("Decoder Layer",p.locator("#compositeTitle").inner_text())
        self.assertEqual(p.locator("#compositeStages .composite-card").count(),6)
        self.assertIn("输出 3 × 4",p.locator("#compositeStages").inner_text())
        self.assertIn("通用固定源码",p.locator("#compositeCaveat").inner_text())
        shots=[self.screenshot("01-decoder-overview-six-stages.png")]
        p.locator('#compositeStages [data-comp-stage="1"]').click()
        self.assertFalse(p.locator("#viewport").is_hidden())
        self.assertIn("GQA Attention",p.locator("#compositeTitle").inner_text())
        self.assertEqual(p.locator("#compositeStages .composite-card").count(),4)
        self.assertEqual(p.locator("#blocks .block").count(),7)
        self.assertEqual(p.locator("#compositeTrail button").count(),2)
        shots.append(self.screenshot("02-gqa-hierarchy-and-source-graph.png"))
        p.locator('#compositeStages [data-comp-stage="2"]').click()
        self.assertIn("单头 Attention",p.locator("#compositeTitle").inner_text())
        self.assertIn("不等价于完整 GQA",p.locator("#compositeCaveat").inner_text())
        self.assertEqual(p.locator("#compositeStages .composite-card").count(),5)
        self.assertEqual(p.locator("#blocks .block").count(),10)
        p.locator("#run").click()
        self.assertIn("1 / 10",p.locator("#stepInfo").inner_text())
        p.locator('#compositeStages [data-comp-stage="0"]').click()
        self.assertIn("矩阵乘法",p.locator("#inspectorTitle").inner_text())
        self.assertIn("推断 shape = [3, 3]",p.locator("#shape").inner_text())
        self.assertEqual(p.locator("#matrix .cell").count(),0)
        self.assertIn("Q[Tq,Dk]",p.locator("#compositeStages").inner_text())
        shots.append(self.screenshot("03-single-head-expanded-qk-matmul.png"))
        p.locator('#compositeTrail button[data-comp-crumb="0"]').click()
        self.assertEqual(p.locator("#compositeStages .composite-card").count(),6)
        p.locator('#compositeStages [data-comp-stage="4"]').click()
        self.assertIn("SwiGLU FFN",p.locator("#compositeTitle").inner_text())
        p.locator('#compositeStages [data-comp-stage="2"]').click()
        self.assertIn("FFN 拆到最细",p.locator("#compositeTitle").inner_text())
        self.assertEqual(p.locator("#blocks .block").count(),6)
        p.locator("#run").click()
        p.locator('#compositeStages [data-comp-stage="3"]').click()
        self.assertIn("逐元素",p.locator("#inspectorTitle").inner_text())
        self.assertIn("推断 shape = [3, 8]",p.locator("#shape").inner_text())
        self.assertIn("SiLU",p.locator("#compositeStages").inner_text())
        shots.append(self.screenshot("04-ffn-silu-hadamard-downprojection.png"))
        (OUT/"manifest.json").write_text(json.dumps({
            "schema":"seethetruth-composite-chromium-v1",
            "browser":"Playwright Chromium",
            "viewport":[1600,1000],
            "captures":shots,
            "proof":"root macro 6 stages -> GQA 4 -> single-head 5 exact matmul stages -> FFN 6 primitive blocks",
            "mode":"qualitative only; no GPU, no pretrained model and no real weights",
            "boundary":"single-head mathematical subview is not equivalent to complete GQA with RoPE/causal mask"
        },ensure_ascii=False,indent=2),encoding="utf-8")
    def test_back_navigate_restore_lessons_and_optional_numeric(self):
        p=self.p
        p.locator("#compositeLaunch").click()
        p.locator('#compositeStages [data-comp-stage="0"]').click()
        self.assertIn("Normalization",p.locator("#compositeTitle").inner_text())
        self.assertEqual(p.locator("#compositeStages .composite-card").count(),3)
        p.locator("#compositeBack").click()
        self.assertIn("Decoder Layer",p.locator("#compositeTitle").inner_text())
        p.locator("#compositeBack").click()
        self.assertTrue(p.locator("#compositeNavigator").is_hidden())
        self.assertFalse(p.locator("#viewport").is_hidden())
        p.locator('[data-preset="ffn"]').click()
        p.locator("#studyMode").select_option("numeric")
        p.locator("#run").click()
        self.assertIn("1 / 4",p.locator("#stepInfo").inner_text())
        p.locator("#next").click()
        self.assertEqual(p.locator("#matrix .cell").count(),24)
        p.locator("#compositeLaunch").click()
        self.assertEqual(p.locator("#studyMode").input_value(),"numeric")
        p.locator("#studyMode").select_option("concept")
        p.locator("#run").click()
        self.assertEqual(p.locator("#matrix .cell").count(),0)

if __name__=="__main__":
    unittest.main()
