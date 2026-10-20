"""Real Chromium narrow-screen acceptance for the Android teaching edition."""
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
from playwright.sync_api import sync_playwright

OUT=Path("android_teaching_screenshots")
class AndroidLearningBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True,exist_ok=True)
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
        cls.base=f"http://127.0.0.1:{cls.server.server_port}"
    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
    def setUp(self):
        self.p=self.browser.new_page(viewport={"width":390,"height":844},
                                      device_scale_factor=2,accept_downloads=True,
                                      is_mobile=True,has_touch=True)
        self.errors=[]
        self.p.on("pageerror",lambda e:self.errors.append(str(e)))
    def tearDown(self):
        self.assertEqual(self.errors,[])
        self.p.close()
    def test_learning_portal_and_lesson_navigation(self):
        p=self.p
        r=p.goto(self.base+"/android_teaching.html")
        self.assertEqual(r.status,200)
        self.assertIn("模型课堂",p.title())
        self.assertEqual(p.locator(".cards .card").count(),8)
        self.assertTrue(p.locator("text=纯教学版").is_visible())
        p.screenshot(path=str(OUT/"android-home-390px.png"),full_page=True,animations="disabled")
        p.locator('a[href="learn.html?mission=image_vit_patch"]').click()
        p.wait_for_url("**/learn.html?mission=image_vit_patch")
        p.add_script_tag(url=self.base+"/android_teaching_mode.js")
        p.wait_for_function("document.documentElement.dataset.sttTeaching==='1'")
        self.assertFalse(p.locator("#exportPython").is_visible())
        self.assertFalse(p.locator("#save").is_visible())
        self.assertFalse(p.locator("#pythonExportPanel").is_visible())
        self.assertTrue(p.locator("#studyMode").is_visible())
        self.assertGreater(p.locator("[data-preset]").count(),50)
        p.screenshot(path=str(OUT/"android-multimodal-learning-390px.png"),
                     full_page=True,animations="disabled")
    def test_mobile_inference_teaching_routes(self):
        p=self.p
        for dest,needle in [
            ("tradeoffs.html","GQA"),
            ("deepinfer_kv_lifecycle_lab.html","KV Cache"),
            ("deepinfer_scheduler_lab.html","Scheduler")
        ]:
            res=p.goto(self.base+"/"+dest)
            self.assertEqual(res.status,200)
            p.add_script_tag(url=self.base+"/android_teaching_mode.js")
            self.assertFalse(p.locator("#download").is_visible() if p.locator("#download").count() else False)
            self.assertIn(needle,p.locator("body").inner_text())
        p.screenshot(path=str(OUT/"android-inference-teaching-390px.png"),
                     full_page=True,animations="disabled")

if __name__=="__main__":
    unittest.main()
