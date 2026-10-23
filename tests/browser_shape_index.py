"""Real Chromium inspection of static tensor index mapping without numerical mode."""
import sys
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

class ShapeIndexBrowserAcceptance(unittest.TestCase):
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
    def test_shape_coordinates_and_gradients_in_static_mode(self):
        page=self.browser.new_page(viewport={"width":1440,"height":960})
        errors=[]
        page.on("pageerror",lambda err:errors.append(str(err)))
        try:
            page.goto(self.url,wait_until="domcontentloaded")
            page.wait_for_function("document.querySelectorAll('#blocks .block').length>0")
            page.locator("#studyMode").select_option("concept")
            page.locator('[data-preset="nd_attention_lesson"]').click()
            for ident,expected in [
                ("biased","被广播的轴"),
                ("summed","被约简轴"),
                ("flat","行主序"),
                ("perm","逆置换")
            ]:
                page.locator('#blocks [data-node="'+ident+'"] .block-formula').click()
                element=page.locator("#shapeIndexSection")
                self.assertTrue(element.is_visible(),ident)
                detail=page.locator("#shapeIndexInspector").inner_text()
                self.assertIn("输出",detail,ident)
                self.assertIn("索引",detail,ident)
                self.assertIn(expected,detail,ident)
                self.assertIn("反向梯度",page.locator("#shapeIndexSection").inner_text())
                self.assertGreater(page.locator("#shapeIndexInspector [data-shape-axis]").count(),0)
                self.assertNotIn("该输出元素实际计算值",page.locator("#shapeIndexInspector").inner_text())

            page.locator('#blocks [data-node="biased"] .block-formula').click()
            controls=page.locator("#shapeIndexInspector [data-shape-axis]")
            self.assertEqual(controls.count(),4)
            controls.nth(1).select_option("0")
            self.assertEqual(page.locator('#shapeIndexInspector [data-shape-axis="1"]').input_value(),"0")
            value0=page.locator("#shapeIndexInspector").inner_text()
            self.assertIn("影响 2 个输出坐标",value0)
            controls=page.locator("#shapeIndexInspector [data-shape-axis]")
            controls.nth(1).select_option("1")
            value1=page.locator("#shapeIndexInspector").inner_text()
            self.assertNotEqual(value0,value1)
            page.locator('#blocks [data-node="summed"] .block-formula').click()
            self.assertIn("本输出汇聚 3 个输入元素",page.locator("#shapeIndexInspector").inner_text())
            page.locator('#blocks [data-node="biased"] .block-formula').click()
            self.assertEqual(page.locator('#shapeIndexInspector [data-shape-axis="1"]').input_value(),"1")
            page.locator("#studyMode").select_option("numeric")
            page.locator("#run").click()
            page.locator('#blocks [data-node="biased"] .block-formula').click()
            self.assertIn("该输出元素实际计算值",page.locator("#shapeIndexInspector").inner_text())
            page.locator('[data-preset="nd_cached_gqa_decode"]').click()
            page.locator('#blocks [data-node="shared_k"] .block-formula').click()
            self.assertIn("KV Head",page.locator("#shapeIndexInspector").inner_text())
            self.assertIn("Query Head",page.locator("#shapeIndexInspector").inner_text())
            page.locator('#shapeIndexInspector [data-shape-axis="1"]').select_option("3")
            self.assertIn("影响 2 个输出坐标",page.locator("#shapeIndexInspector").inner_text())
            self.assertEqual(page.locator('#shapeIndexInspector [data-shape-axis="1"]').input_value(),"3")
            page.locator('#blocks [data-node="full_k"] .block-formula').click()
            self.assertTrue(page.locator("#shapeIndexSection").is_hidden())
            self.assertEqual(errors,[],"Browser JavaScript errors: "+str(errors))
        finally:
            page.close()
if __name__=="__main__":
    unittest.main()
