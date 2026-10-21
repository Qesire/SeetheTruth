"""Chromium proof that the same small tensor/state arithmetic runs on PC and phone."""
import sys,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

DEST=Path("semantic_state_screenshots")
class SemanticStateBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DEST.mkdir(parents=True,exist_ok=True)
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
        cls.base=f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()

    def test_pc_and_mobile_mathematical_cases(self):
        snapshots=[]
        for width in (1440,390):
            page=self.browser.new_page(viewport={"width":width,"height":880},
                                       is_mobile=width==390,has_touch=width==390)
            errors=[]
            page.on("pageerror",lambda err:errors.append(str(err)))
            response=page.goto(self.base+"/semantic_state.html")
            self.assertEqual(response.status,200)
            page.wait_for_function("window.SeetheTruthStateLab && window.SeetheTruthStateLab.cases.length===4")
            self.assertEqual(page.locator("[data-scenario]").count(),4)
            evidence=[]
            for kind in ["stream","video","axis","reject"]:
                page.locator('[data-scenario="'+kind+'"]').click()
                self.assertIn("数值合同通过",page.locator("#status").inner_text())
                result=page.evaluate("(k)=>window.SeetheTruthStateLab.run(k)",kind)
                self.assertTrue(result["passed"],kind)
                evidence.append({"kind":kind,"values":result["values"]})
            self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"),width+2)
            page.screenshot(path=str(DEST/("semantic-desktop.png" if width==1440 else "semantic-phone.png")),
                            full_page=True,animations="disabled")
            snapshots.append(evidence)
            self.assertEqual(errors,[],"JS console must be error-free")
            page.close()
        self.assertEqual(snapshots[0],snapshots[1],"PC and phone must compute identical data")
if __name__=="__main__":unittest.main()
