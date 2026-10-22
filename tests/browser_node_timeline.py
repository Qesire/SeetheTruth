"""Chromium regression of the actual Learning Studio node-level training inspector."""
import sys
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

class NodeTimelineBrowserAcceptance(unittest.TestCase):
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

    def test_training_timeline_is_node_and_step_aware(self):
        page=self.browser.new_page(viewport={"width":1440,"height":960})
        errors=[]
        page.on("pageerror",lambda e:errors.append(str(e)))
        try:
            page.goto(self.url,wait_until="domcontentloaded")
            page.wait_for_function("document.querySelectorAll('#blocks .block').length>0")
            page.locator('[data-preset="trainable_core"]').click()
            page.locator("#studyMode").select_option("numeric")
            page.locator("#trainCount").select_option("2")
            page.locator("#trainStep").click()
            page.locator("#nodeTimelineDetail").wait_for(timeout=15000)
            self.assertNotIn("训练未提交",page.locator("#status").inner_text())
            self.assertIn("shared_weight",page.locator("#nodeTimelineDetail").inner_text())
            self.assertIn("反向梯度",page.locator("#nodeTimelineDetail").inner_text())
            self.assertEqual(page.locator("#timelineStep option").count(),2)
            page.locator('[data-timeline-phase="optimizer"]').click()
            self.assertIn("step 2",page.locator("#nodeTimelineDetail").inner_text())
            page.locator("#timelineStep").select_option("0")
            self.assertIn("step 1",page.locator("#nodeTimelineDetail").inner_text())
            page.locator('[data-timeline-phase="forward"]').click()
            self.assertIn("前向输出",page.locator("#nodeTimelineDetail").inner_text())
            page.locator('[data-timeline-phase="updated"]').click()
            self.assertIn("更新后输出",page.locator("#nodeTimelineDetail").inner_text())
            page.locator('#blocks [data-node="loss"]').click()
            self.assertIn("loss",page.locator("#nodeTimelineDetail").inner_text())
            page.locator('[data-timeline-phase="optimizer"]').click()
            self.assertIn("未执行参数更新",page.locator("#nodeTimelineDetail").inner_text())
            self.assertEqual(errors,[],"JS errors: "+str(errors))
        finally:
            page.close()

if __name__=="__main__":
    unittest.main()
