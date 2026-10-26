"""Real Chromium acceptance: complete connected Wan I2V Self/Cross/FFN graph."""
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler


class WanFullBlockBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(
            target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
        cls.url=f"http://127.0.0.1:{cls.server.server_port}/learn.html"

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def test_complete_source_graph_desktop_and_phone(self):
        for width in (1440,390):
            page=self.browser.new_page(viewport={"width":width,"height":850},
                is_mobile=width==390,has_touch=width==390)
            errors=[]
            page.on("pageerror",lambda error:errors.append(str(error)))
            try:
                self.assertEqual(page.goto(self.url,wait_until="domcontentloaded").status,200)
                page.locator('[data-preset="nd_wan_i2v_full_block"]').click()
                page.wait_for_function(
                    "document.querySelectorAll('#blocks [data-node]').length===115",
                    timeout=30000)
                self.assertIn("115/115",page.locator("#graphReadiness").inner_text())
                for node in ("after_self","after_cross","norm2_ffn",
                             "ffn_gelu_tanh","ffn_down_bias","ffn_gated","after_ffn","loss"):
                    self.assertEqual(page.locator('#blocks [data-node="'+node+'"]').count(),1)
                page.locator("#studyMode").select_option("numeric")
                page.locator("#run").click()
                page.wait_for_function(
                    "document.querySelector('#status')?.textContent.includes('已运行')",
                    timeout=30000)
                self.assertIn("✓ 已运行",page.locator("#status").inner_text())
                # The actual installed graph engine supplies the observability
                # envelope; the browser never reconstructs the source values.
                proof=page.evaluate("""()=>{
                    const L=window.SeetheTruthLearning;
                    const g=L.preset('nd_wan_i2v_full_block');
                    const evaluated=L.evaluate(g);
                    const nodes=new Map(evaluated.steps.map(s=>[s.id,s]));
                    return {count:evaluated.steps.length,loss:nodes.get('loss').values[0][0],
                       afterCross:nodes.get('after_cross').shape,
                       afterFFN:nodes.get('after_ffn').shape,
                       gelu:nodes.get('ffn_gelu_tanh').shape};
                }""")
                self.assertEqual(proof["count"],115)
                self.assertEqual(proof["afterCross"],[1,1,8,6])
                self.assertEqual(proof["afterFFN"],[1,1,8,6])
                self.assertEqual(proof["gelu"],[8,8])
                self.assertGreaterEqual(proof["loss"],0)
                self.assertEqual(errors,[])
            finally:
                page.close()


if __name__=="__main__":
    unittest.main()
