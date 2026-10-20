"""Real Chromium acceptance for physical KV refs/fork/preempt educational state."""
import json
import threading
import unittest
from pathlib import Path
import sys
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
from playwright.sync_api import sync_playwright

DEST=Path("deepinfer_kv_screenshots")
class DeepInferKVBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DEST.mkdir(parents=True,exist_ok=True)
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
    def setUp(self):
        self.p=self.browser.new_page(viewport={"width":1480,"height":1100},accept_downloads=True)
        self.errors=[]
        self.p.on("pageerror",lambda e:self.errors.append(str(e)))
        res=self.p.goto(f"http://127.0.0.1:{self.server.server_port}/deepinfer_kv_lifecycle_lab.html")
        self.assertEqual(res.status,200)
        self.p.wait_for_function("!!window.SeetheTruthLastKVState")
    def tearDown(self):
        self.assertEqual(self.errors,[])
        self.p.close()
    def test_fork_pin_clone_refcount_and_fifo_download(self):
        p=self.p
        self.assertEqual(p.locator("#blocks .cell").count(),7)
        self.assertEqual(p.locator("#freeCount").inner_text(),"6")
        self.assertEqual(p.locator("#blocks .cell.reserved").count(),1)
        self.assertEqual(p.locator("#timeline .entry").count(),10)
        p.locator("#timeline .entry").nth(4).click()
        state=p.evaluate("window.SeetheTruthLastKVState")
        self.assertEqual(state["requests"]["A"]["blocks"],[1,2])
        self.assertEqual(state["requests"]["B"]["blocks"],[3])
        p.locator("#next").click()
        pin=p.evaluate("window.SeetheTruthLastKVState")
        self.assertEqual(pin["refs"][1:3],[2,2])
        self.assertEqual(p.locator("#blocks .cell.shared").count(),2)
        p.locator("#next").click()
        child=p.evaluate("window.SeetheTruthLastKVState")
        self.assertEqual(child["requests"]["C"]["blocks"],[1,4])
        self.assertEqual(child["copies"][0]["from"],2)
        self.assertEqual(child["copies"][0]["to"],4)
        p.screenshot(path=str(DEST/"shared-prefix-tail-clone.png"),full_page=True,animations="disabled")
        p.locator("#timeline .entry").last.click()
        final=p.evaluate("window.SeetheTruthLastKVState")
        self.assertEqual(final["free"],[5,6,2,4,1,3])
        self.assertEqual(final["refs"],[0,0,0,0,0,0,0])
        self.assertEqual(p.locator("#freeCount").inner_text(),"6")
        with p.expect_download() as done:
            p.locator("#download").click()
        exported=json.loads(Path(done.value.path()).read_text(encoding="utf8"))
        self.assertEqual(exported["timeline"][-1]["free"],final["free"])
        self.assertEqual(exported["timeline"][6]["copies"][0]["type"],"clone_partial_tail")
    def test_preempt_recompute_so_stop_and_invalid_editor_atomicity(self):
        p=self.p
        p.locator("#scenario").select_option("preempt")
        p.locator("#timeline .entry").nth(5).click()
        mid=p.evaluate("window.SeetheTruthLastKVState")
        self.assertEqual(mid["requests"]["X"]["status"],"WAITING")
        self.assertEqual(mid["requests"]["X"]["recomputes"],1)
        self.assertEqual(mid["requests"]["X"]["blocks"],[])
        p.screenshot(path=str(DEST/"preempt-recompute.png"),full_page=True,animations="disabled")
        p.locator("#timeline .entry").last.click()
        last=p.evaluate("window.SeetheTruthLastKVState")
        self.assertEqual(last["requests"]["X"]["status"],"FINISHED_PREEMPTED")
        self.assertEqual(len(last["free"]),4)
        p.locator("#operations").evaluate('(e)=>{e.closest("details").open=true;}')
        p.locator("#operations").fill('{"config":{"numBlocks":4,"blockSize":4},"ops":[{"op":"arrive","id":"A","tokenLen":16},{"op":"allocate","id":"A"}]}')
        p.locator("#apply").click()
        self.assertIn("输入被拒绝",p.locator("#status").inner_text())
        self.assertEqual(p.evaluate("window.SeetheTruthLastKVState.requests.X.status"),"FINISHED_PREEMPTED")
        p.locator("#scenario").select_option("pressure")
        self.assertEqual(p.locator("#timeline .entry").count(),8)
        p.screenshot(path=str(DEST/"resource-pressure.png"),full_page=True,animations="disabled")

if __name__=="__main__":
    unittest.main()
