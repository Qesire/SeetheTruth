"""Real Chromium acceptance for cross-node scalar provenance and VJP."""
import sys
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

class ElementLineageBrowserAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.playwright=sync_playwright().start()
        cls.browser=cls.playwright.chromium.launch(headless=True,args=["--no-sandbox"])
        cls.url="http://127.0.0.1:"+str(cls.server.server_port)+"/learn.html"

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def test_cross_graph_dependency_and_gradient_trace(self):
        page=self.browser.new_page(viewport={"width":1440,"height":960})
        errors=[]
        page.on("pageerror",lambda err:errors.append(str(err)))
        try:
            page.goto(self.url,wait_until="domcontentloaded")
            page.wait_for_function("document.querySelectorAll('#blocks .block').length>0")
            page.locator("#studyMode").select_option("concept")
            page.locator('[data-preset="nd_attention_lesson"]').click()
            page.locator('#blocks [data-node="perm"] .block-formula').click()
            section=page.locator("#graphLineageSection")
            self.assertTrue(section.is_visible())
            self.assertEqual(page.locator("[data-lineage-axis]").count(),4)
            page.locator('[data-lineage-axis="1"]').fill("1")
            page.locator('[data-lineage-axis="1"]').press("Tab")
            page.locator("#traceGraphElement").click()
            text=page.locator("#graphLineageResult").inner_text()
            self.assertIn("静态依赖推导",text)
            self.assertIn("已覆盖全部受支持路径",text)
            self.assertIn("逐层元素来源",text)
            for upstream in ["queries","keys","bias","batch_scores","summed","expand","flat"]:
                self.assertIn(upstream,text)
            page.locator('#blocks [data-node="biased"] .block-formula').click()
            self.assertTrue(section.is_visible())
            self.assertNotIn("queries",page.locator("#graphLineageResult").inner_text())
            page.locator("#studyMode").select_option("numeric")
            page.locator("#run").click()
            page.locator('#blocks [data-node="perm"] .block-formula').click()
            page.locator("#traceGraphElement").click()
            out=page.locator("#graphLineageResult").inner_text()
            self.assertIn("真实微型数值轨迹",out)
            page.locator("#graphLineageResult summary").first.click()
            page.locator("#graphLineageResult .lineage-node summary").first.click()
            self.assertIn("∂输出/∂该元素",page.locator("#graphLineageResult").inner_text())
            page.locator('[data-preset="ffn"]').click()
            page.locator('#blocks [data-node="act"] .block-formula').click()
            page.locator("#traceGraphElement").click()
            out=page.locator("#graphLineageResult").inner_text()
            self.assertIn("逐元素规则边界",out)
            self.assertNotIn("已覆盖全部受支持路径",out)
            self.assertEqual(errors,[],"Page errors: "+str(errors))
        finally:
            page.close()

if __name__=="__main__":
    unittest.main()
