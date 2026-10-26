"""Chromium PC/phone acceptance for the source-derived block capability matrix."""
import sys
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

DEST=Path("capability_screenshots")

class CapabilityPageAcceptance(unittest.TestCase):
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
        cls.browser.close();cls.pw.stop()
        cls.server.shutdown();cls.server.server_close()

    def test_matrix_is_computed_on_both_screens(self):
        errors=[]
        pages=[]
        for width in [1440,390]:
            page=self.browser.new_page(viewport={"width":width,"height":840},
                is_mobile=(width==390),has_touch=(width==390))
            pages.append(page)
            page.on("pageerror",lambda e:errors.append(str(e)))
            response=page.goto(self.base+"/capabilities.html")
            self.assertEqual(response.status,200)
            page.wait_for_function("window.SeetheTruthCapabilities && document.querySelectorAll('#rows tr').length===window.SeetheTruthCapabilities.audit().counts.registered")
            counts=page.evaluate("window.SeetheTruthCapabilities.audit().counts")
            extension_count=page.evaluate("window.SeetheTruthBlockExtensions.kinds.length")
            self.assertEqual(extension_count,8)
            core_native=page.evaluate("window.SeetheTruthBlockCatalog.KINDS.length")
            stream_native=page.evaluate("window.SeetheTruthStreamGraph.kinds.length")
            self.assertEqual(counts["registered"],11+core_native+stream_native+extension_count)
            self.assertEqual(counts["knowledgeCards"],counts["registered"])
            backend=page.evaluate("""()=>{
                const kinds=new Set(window.SeetheTruthLearning.KINDS);
                const numpy=window.SeetheTruthPythonExport.supportedKinds;
                const torch=window.SeetheTruthTorchExport.supportedKinds;
                return {numpy:numpy.filter(k=>kinds.has(k)).length,
                        torch:torch.filter(k=>kinds.has(k)).length,
                        ndMulNumPy:numpy.includes('nd_broadcast_mul'),
                        ndMulTorch:torch.includes('nd_broadcast_mul')};
            }""")
            self.assertEqual(counts["numpyEligible"],backend["numpy"])
            self.assertEqual(counts["torchEligible"],backend["torch"])
            self.assertTrue(backend["ndMulNumPy"] and backend["ndMulTorch"])
            self.assertEqual(page.locator("#registered").inner_text(),str(counts["registered"]))
            self.assertEqual(page.locator("#rows tr").count(),counts["registered"])
            page.locator("#filter").select_option("no-numpy")
            self.assertEqual(page.locator("#rows tr").count(),counts["numpyMissing"])
            page.locator("#search").fill("gqa")
            self.assertEqual(page.locator("#rows tr").count(),1)
            self.assertIn("gqa",page.locator("#rows tr").inner_text())
            page.locator("#search").fill("")
            page.locator("#filter").select_option("no-torch")
            self.assertEqual(page.locator("#rows tr").count(),counts["torchMissing"])
            self.assertFalse(page.evaluate("document.documentElement.scrollWidth > window.innerWidth+2"))
            page.screenshot(path=str(DEST/("capability-desktop.png" if width>390 else "capability-phone.png")),
                full_page=True,animations="disabled")
        for page in pages:page.close()
        self.assertEqual(errors,[])

if __name__=="__main__":
    unittest.main()
