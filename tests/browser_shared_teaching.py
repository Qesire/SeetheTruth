"""Real Chromium acceptance of identical PC/Android study data and artifact-free phone UX."""
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
from playwright.sync_api import sync_playwright

OUT=Path("shared_teaching_screenshots")
class SharedTeachingBrowser(unittest.TestCase):
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

    def test_same_pc_and_mobile_curriculum_explanations(self):
        normal=self.browser.new_page(viewport={"width":1440,"height":920})
        phone=self.browser.new_page(viewport={"width":390,"height":844},is_mobile=True,has_touch=True,
                                    device_scale_factor=2)
        errors=[]
        normal.on("pageerror",lambda e:errors.append("pc:"+str(e)))
        phone.on("pageerror",lambda e:errors.append("phone:"+str(e)))
        self.assertEqual(normal.goto(self.base+"/teaching.html").status,200)
        self.assertEqual(phone.goto(self.base+"/teaching.html?edition=teaching").status,200)
        for p in (normal,phone):
            p.wait_for_function("window.SeetheTruthTeachingHub && window.SeetheTruthTeachingHub.lessons===10")
            self.assertEqual(p.locator("#cards button.card").count(),10)
            p.locator("#voteCase").select_option("any")
            self.assertEqual(p.evaluate("window.SeetheTruthLastSharedVote.point"),"ANY")
            self.assertEqual(p.evaluate("window.SeetheTruthLastSharedVote.perRank"),["prompt","empty-prompt"])
            p.locator("#voteCase").select_option("fallback")
            self.assertTrue(p.evaluate("window.SeetheTruthLastSharedVote.secondVoteFallback"))
            self.assertEqual(p.evaluate("window.SeetheTruthLastSharedVote.step"),"decode")
            p.locator('[data-lesson="prefill"]').click()
            self.assertIn("Prefill / Decode",p.locator("#detailTitle").inner_text())
            self.assertIn("ALL",p.locator("#why").inner_text())
            p.locator("#reveal").click()
            self.assertIn("第二轮",p.locator("#answer").inner_text())
            self.assertTrue(p.locator("#openLesson").get_attribute("href").endswith("deepinfer_prefill_lab.html"))
        self.assertEqual(normal.locator("#theory").inner_text(),phone.locator("#theory").inner_text())
        self.assertEqual(normal.locator("#answer").inner_text(),phone.locator("#answer").inner_text())
        self.assertTrue(phone.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2'),
                        "mobile teaching page must not overflow width")
        normal.screenshot(path=str(OUT/"curriculum-desktop.png"),full_page=True,animations="disabled")
        phone.screenshot(path=str(OUT/"curriculum-android-390.png"),full_page=True,animations="disabled")
        normal.close();phone.close()
        self.assertEqual(errors,[])

    def test_mobile_teaching_hides_generated_artifacts_but_keeps_math(self):
        p=self.browser.new_page(viewport={"width":390,"height":844},is_mobile=True,has_touch=True,
                                device_scale_factor=2)
        errors=[]
        p.on("pageerror",lambda e:errors.append(str(e)))
        self.assertEqual(p.goto(self.base+"/teaching.html?edition=teaching").status,200)
        p.locator('[data-lesson="building"]').click()
        p.locator("#openLesson").click()
        p.wait_for_url("**/learn.html?edition=teaching")
        p.wait_for_function("document.body.classList.contains('teaching-edition')")
        self.assertTrue(p.locator("#run").is_visible())
        self.assertTrue(p.locator("#blockSearch").is_visible())
        self.assertFalse(p.locator("#save").is_visible())
        self.assertFalse(p.locator("#exportPython").is_visible())
        self.assertFalse(p.locator('#pythonExportPanel').is_visible())
        self.assertTrue(p.locator("[data-preset]").count()>=59)
        p.screenshot(path=str(OUT/"shared-graph-learning-android.png"),
                     full_page=True,animations="disabled")
        p.close()
        self.assertEqual(errors,[])

if __name__=="__main__":
    unittest.main()
