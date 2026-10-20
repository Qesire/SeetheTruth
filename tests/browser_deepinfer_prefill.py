"""Chromium browser contract for DeepInfer vote and chunk teaching page."""
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
from playwright.sync_api import sync_playwright

OUT=Path("deepinfer_prefill_screenshots")
class DeepInferPrefillBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True,exist_ok=True)
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
    def setUp(self):
        self.page=self.browser.new_page(viewport={"width":1190,"height":950})
        self.errors=[]
        self.page.on("pageerror",lambda e:self.errors.append(str(e)))
        response=self.page.goto(f"http://127.0.0.1:{self.server.server_port}/deepinfer_prefill_lab.html")
        self.assertEqual(response.status,200)
        self.page.wait_for_function("!!window.SeetheTruthPrefillLast?.chunk")
    def tearDown(self):
        self.assertEqual(self.errors,[])
        self.page.close()
    def test_all_any_secondary_fallback_and_chunk(self):
        p=self.page
        self.assertEqual(p.locator("#votePoint").inner_text(),"ANY")
        self.assertEqual(p.locator("#voteStep").inner_text(),"Prompt")
        self.assertEqual(p.locator("#voteCount").inner_text(),"1")
        self.assertEqual(p.evaluate("window.SeetheTruthPrefillLast.vote.perRank"),
                         ["prompt","empty-prompt"])
        self.assertEqual(p.locator("#chunkQ").inner_text(),"1")
        self.assertEqual(p.locator("#chunkRemaining").inner_text(),"6")
        self.assertEqual(p.locator("#tokens .token.selected").count(),1)
        p.screenshot(path=str(OUT/"prefill-any-image-safe-shrink.png"),full_page=True,animations="disabled")
        p.locator("#rank0").select_option("later")
        self.assertEqual(p.locator("#voteStep").inner_text(),"Decode")
        self.assertEqual(p.evaluate("window.SeetheTruthPrefillLast.vote.secondVoteFallback"),True)
        p.locator("#decodeStep").fill("2")
        p.locator("#rank0").select_option("yes")
        self.assertEqual(p.locator("#votePoint").inner_text(),"ALL")
        self.assertEqual(p.locator("#voteStep").inner_text(),"Decode")
        p.locator("#rank1").select_option("yes")
        self.assertEqual(p.locator("#voteStep").inner_text(),"Prompt")
        self.assertEqual(p.locator("#voteCount").inner_text(),"2")
        p.locator("#scenario").select_option("grow")
        self.assertEqual(p.locator("#chunkQ").inner_text(),"5")
        p.locator("#scenario").select_option("busy")
        self.assertEqual(p.locator("#chunkStatus").inner_text(),"DEFER")
        p.locator("#scenario").select_option("aligned")
        self.assertEqual(p.locator("#chunkQ").inner_text(),"7")
        self.assertEqual(p.evaluate("window.SeetheTruthPrefillLast.chunk.charged"),8)
        p.screenshot(path=str(OUT/"prefill-all-cp-alignment.png"),full_page=True,animations="disabled")
    def test_mobile_layout_and_source_link(self):
        p=self.page
        p.set_viewport_size({"width":390,"height":844})
        p.reload()
        p.wait_for_function("!!window.SeetheTruthPrefillLast?.chunk")
        self.assertEqual(p.locator("#chunkQ").inner_text(),"1")
        self.assertEqual(p.locator(".two > section").count(),2)
        self.assertGreater(p.locator('a[href*="deepseek-ai/DeepInfer"]').count(),0)
        p.screenshot(path=str(OUT/"prefill-mobile-course.png"),full_page=True,animations="disabled")

if __name__=="__main__":
    unittest.main()
