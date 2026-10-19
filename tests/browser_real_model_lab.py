"""Chromium acceptance for the evidence-only import surface.
All local fixtures are SYNTHETIC STRUCTURAL TESTS, NEVER REAL_CAPTURE evidence.
"""
import json, math, pathlib, tempfile, threading, unittest
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from atlas_server import Handler
DEST=pathlib.Path("real_model_screenshots")
def stats(a):
    return dict(shape=[2,2],count=4,mean=sum(a)/4,
      rms=math.sqrt(sum(x*x for x in a)/4),l2=math.sqrt(sum(x*x for x in a)),
      min=min(a),max=max(a),zeroFraction=sum(x==0 for x in a)/4,sample=a)
def synthetic_test_only():
    return {
      "schema":"seethetruth-real-model-trace-v1",
      "provenance":{"kind":"real_model_execution","modelId":"TEST_FIXTURE_NOT_REAL/CHECK_ONLY",
        "revision":"a"*40,"modulePath":"test_linear","moduleType":"torch.nn.Linear",
        "device":"cpu","dtype":"float32","framework":"pytorch","objective":"test_only",
        "promptSha256":"b"*64,"tokenIds":[1,2,3]},
      "loss":.5,"parameter":{"identity":"test_linear.weight",
        "weight":stats([1,2,3,4]),"gradient":stats([-1,0,1,0])},
      "activation":{"input":[1,2,3],"output":[4,5,6]}
    }
class RealTraceBrowserAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DEST.mkdir(parents=True,exist_ok=True)
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.playwright=sync_playwright().start()
        cls.browser=cls.playwright.chromium.launch(headless=True,args=["--no-sandbox"])
    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.playwright.stop();cls.server.shutdown();cls.server.server_close()
    def setUp(self):
        self.p=self.browser.new_page(viewport={"width":1440,"height":950})
        self.errors=[]
        self.p.on("pageerror",lambda e:self.errors.append(str(e)))
        self.p.goto("http://127.0.0.1:%d/real_model_lab.html"%self.server.server_port)
    def tearDown(self):
        self.assertFalse(self.errors,"JS errors: "+str(self.errors))
        self.p.close()
    def put(self,payload):
        with tempfile.NamedTemporaryFile("w",suffix=".json",encoding="utf8") as f:
            json.dump(payload,f,allow_nan=False);f.flush()
            self.p.locator("#traceUpload").set_input_files(f.name)
    def test_no_false_proof_and_structural_view(self):
        p=self.p
        self.assertIn("尚未导入",p.locator("#status").inner_text())
        self.assertTrue(p.locator("#capture").is_hidden())
        p.screenshot(path=str(DEST/"01-no-real-samples.png"),full_page=True)
        self.put({"schema":"invalid","code":"alert(1)"})
        self.assertIn("未通过验证",p.locator("#status").inner_text())
        self.assertTrue(p.locator("#capture").is_hidden())
        self.put(synthetic_test_only())
        self.assertIn("仍需按 revision 独立复跑",p.locator("#status").inner_text())
        self.assertIn("TEST_FIXTURE_NOT_REAL",p.locator("#modelName").inner_text())
        self.assertEqual(p.locator("#paramStats .sample").count(),2)
        self.assertEqual(p.locator("#activationStats .sample").count(),2)
        self.assertFalse(p.locator("#capture").is_hidden())
        self.assertFalse(p.evaluate("document.documentElement.scrollWidth>innerWidth+2"))
        p.screenshot(path=str(DEST/"02-synthetic-structural-test-not-real-capture.png"),full_page=True)
        bad=synthetic_test_only();bad["parameter"]["weight"]["l2"]=0
        self.put(bad)
        self.assertIn("未通过验证",p.locator("#status").inner_text())
        self.assertTrue(p.locator("#capture").is_hidden())
        (DEST/"manifest.json").write_text(json.dumps({"evidence":"synthetic UI contract, NOT a captured model",
        "files":["01-no-real-samples.png","02-synthetic-structural-test-not-real-capture.png"]}),encoding="utf8")
if __name__=="__main__":unittest.main(verbosity=2)
