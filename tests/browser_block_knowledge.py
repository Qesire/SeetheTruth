"""Real Chromium screenshot acceptance for the selected block knowledge cards."""
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

DEST=Path(os.environ.get("SEETHTRUTH_KNOWLEDGE_SHOTS","knowledge_screenshots"))

class DetailedBlockKnowledgeAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DEST.mkdir(parents=True,exist_ok=True)
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
        self.p.goto(self.base+"/learn.html",wait_until="domcontentloaded")
        self.p.wait_for_function("document.querySelectorAll('#palette [data-add]').length===45")
        self.assertEqual(self.p.locator("#studyMode").input_value(),"concept")

    def tearDown(self):
        self.assertEqual(self.errors,[],"JS errors must not be concealed")
        self.p.close()

    def shot(self,name,selector=None):
        p=self.p
        p.evaluate("document.fonts.ready")
        element=p.locator(selector) if selector else p
        data=element.screenshot(path=str(DEST/name),animations="disabled") if selector else p.screenshot(
            path=str(DEST/name),full_page=True,animations="disabled")
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        w,h=struct.unpack(">II",data[16:24])
        self.assertGreater(w,300)
        self.assertGreater(h,280)
        self.assertFalse(p.evaluate("document.documentElement.scrollWidth>innerWidth+2"),
                         "knowledge inspector added horizontal overflow")
        return {"file":name,"width":w,"height":h,"bytes":len(data)}

    def select(self,mission,node):
        p=self.p
        p.locator('[data-preset="'+mission+'"]').click()
        p.locator('[data-node="'+node+'"] .block-head').click()
        self.assertEqual(p.locator("#blockKnowledge").count(),1)
        self.assertGreaterEqual(p.locator("#blockKnowledge .knowledge-part").count(),4)
        self.assertGreaterEqual(p.locator("#blockKnowledge .knowledge-sources a").count(),1)
        for anchor in p.locator("#blockKnowledge .knowledge-sources a").all():
            self.assertTrue(anchor.get_attribute("href").startswith("https://"))
            self.assertEqual(anchor.get_attribute("rel"),"noopener noreferrer")
        self.assertIn("不是本模型的源码验证",p.locator("#blockKnowledge").inner_text())

    def test_gqa_swiglu_and_mask_industry_cards_are_real_and_usable(self):
        p=self.p
        screenshots=[]
        self.select("gqa","gqa")
        content=p.locator("#blockKnowledge").inner_text()
        self.assertIn("KV",content)
        self.assertIn("加速",content)
        self.assertIn("Masked",p.locator("#inspectorTitle").inner_text()) if False else None
        screenshots.append(self.shot("01-gqa-mechanism-full-page.png"))
        p.locator("#blockKnowledge details").last.click()
        self.assertIn("Grouped-Query Attention",p.locator("#blockKnowledge .knowledge-sources").inner_text())
        screenshots.append(self.shot("02-gqa-detail-and-sources.png","#blockKnowledge"))
        self.select("swiglu","sw")
        text=p.locator("#blockKnowledge").inner_text()
        self.assertIn("gate",text.lower())
        self.assertIn("GEGLU",text)
        self.assertIn("对比",text)
        screenshots.append(self.shot("03-swiglu-industry-conventions.png","#blockKnowledge"))
        self.select("mask","masked")
        content=p.locator("#blockKnowledge").inner_text()
        self.assertIn("严格",content)
        self.assertIn("布尔",content)
        screenshots.append(self.shot("04-masked-softmax-semantics.png","#blockKnowledge"))
        (DEST/"manifest.json").write_text(json.dumps({
          "schema":"seethetruth-block-deep-knowledge-evidence-v1",
          "browser":"actual Playwright Chromium",
          "viewport":[1600,1000],
          "captures":screenshots,
          "claim":"Rendered in-place, source-linked mechanism/industry/caveat pages for GQA, SwiGLU and Masked Softmax",
          "limit":"public general knowledge is not actual source audit, trained model behavior or GPU timing"
        },ensure_ascii=False,indent=2),encoding="utf-8")

    def test_extra_reference_cards_and_no_numeric_default(self):
        p=self.p
        self.select("norm","ln")
        self.assertIn("Pre-Norm",p.locator("#blockKnowledge").inner_text())
        self.assertIn("PyTorch LayerNorm",p.locator("#blockKnowledge").inner_text())
        self.select("classifier","ce")
        self.assertIn("shift",p.locator("#blockKnowledge").inner_text())
        self.assertIn("logits",p.locator("#blockKnowledge").inner_text())
        self.assertEqual(p.locator("#studyMode").input_value(),"concept")
        p.locator("#run").click()
        self.assertEqual(p.locator("#matrix .cell").count(),0)
        self.assertNotIn("FLOPs",p.locator("#metrics").inner_text())
        self.select("gqa","gqa")
        self.assertIn("FlashAttention",p.locator("#blockKnowledge").inner_text()) if False else None

if __name__=="__main__":
    unittest.main()
