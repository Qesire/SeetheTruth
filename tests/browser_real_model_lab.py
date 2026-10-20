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
    def test_real_capture_if_explicitly_supplied(self):
        import os
        capture=os.getenv("SEETHTRUTH_REAL_MODEL_CAPTURE")
        if not capture:
            return
        p=self.p
        p.locator("#traceUpload").set_input_files(capture)
        p.wait_for_function("!document.querySelector('#capture').hidden")
        self.assertIn("roneneldan/TinyStories-1M",p.locator("#modelName").inner_text())
        self.assertIn("仍需按 revision 独立复跑",p.locator("#status").inner_text())
        self.assertEqual(p.locator("#paramStats .sample").count(),2)
        self.assertEqual(p.locator("#activationStats .sample").count(),2)
        self.assertIn("64 × 64",p.locator("#paramStats").inner_text())
        self.assertFalse(p.evaluate("document.documentElement.scrollWidth>innerWidth+2"))
        p.screenshot(path=str(DEST/"03-genuine-tinystories-pinned-cpu-capture.png"),full_page=True)
        (DEST/"manifest.json").write_text(json.dumps({"evidence":"real CPU model trace imported from pinned CI capture, UI does not independently verify origin",
          "real_file":"03-genuine-tinystories-pinned-cpu-capture.png",
          "other_files":"01-no-real-samples.png,02-synthetic-structural-test-not-real-capture.png"}),encoding="utf8")
    def test_learning_canvas_model_reference_remains_read_only(self):
        p=self.p
        p.goto("http://127.0.0.1:%d/learn.html"%self.server.server_port)
        p.locator('[data-preset="trainable_attention"]').click()
        p.locator('[data-node="q_proj"] .block-head').click()
        with p.expect_download() as d:
            p.locator("#save").click()
        before=json.loads(pathlib.Path(d.value.path()).read_text(encoding="utf8"))
        self.assertTrue(p.locator("#realTracePanel").inner_text()=="")
        with tempfile.NamedTemporaryFile("w",suffix=".json",encoding="utf8") as tmp:
            json.dump(synthetic_test_only(),tmp);tmp.flush()
            p.locator("#traceReferenceFile").set_input_files(tmp.name)
        self.assertIn("只读关联积木 q_proj",p.locator("#status").inner_text())
        self.assertIn("TEST_FIXTURE_NOT_REAL",p.locator("#realTracePanel").inner_text())
        self.assertIn("不修改执行图、参数、优化器、导出代码",p.locator("#realTracePanel").inner_text())
        self.assertEqual(p.locator("#realTracePanel .real-sample-values").count(),4)
        p.locator('[data-node="attend"] .block-head').click()
        self.assertEqual(p.locator("#realTracePanel").inner_text(),"")
        self.assertTrue(p.locator("#bindRealTrace").is_disabled())
        p.locator('[data-node="wq"] .block-head').click()
        self.assertFalse(p.locator("#bindRealTrace").is_disabled())
        p.locator("#bindRealTrace").click()
        self.assertIn("关联到 wq",p.locator("#traceReferenceStatus").inner_text())
        p.screenshot(path=str(DEST/"04-synthetic-reference-on-real-learning-canvas.png"),full_page=True)
        with p.expect_download() as d:
            p.locator("#save").click()
        after=json.loads(pathlib.Path(d.value.path()).read_text(encoding="utf8"))
        self.assertEqual(before,after,"read-only evidence changed the serialized graph")
        p.locator("#clearRealTrace").click()
        self.assertIn("尚未导入",p.locator("#traceReferenceStatus").inner_text())
        self.assertEqual(p.locator("#realTracePanel").inner_text(),"")

    def test_real_model_capture_on_learning_canvas_when_available(self):
        import os
        capture=os.getenv("SEETHTRUTH_REAL_MODEL_CAPTURE")
        if not capture:return
        p=self.p
        p.goto("http://127.0.0.1:%d/learn.html"%self.server.server_port)
        p.locator('[data-preset="autodiff_core"]').click()
        p.locator('[data-node="project"] .block-head').click()
        p.locator("#traceReferenceFile").set_input_files(capture)
        p.wait_for_function("document.querySelector('#realTracePanel').textContent.includes('roneneldan/TinyStories-1M') || document.querySelector('#status').classList.contains('error')",timeout=10000)
        self.assertIn("roneneldan/TinyStories-1M",p.locator("#realTracePanel").inner_text())
        self.assertIn("64×64",p.locator("#realTracePanel").inner_text())
        self.assertEqual(p.locator("#realTracePanel .real-sample-values").count(),4)
        self.assertFalse(p.evaluate("document.documentElement.scrollWidth>innerWidth+2"))
        p.screenshot(path=str(DEST/"05-real-tinystories-observation-linked-to-matmul.png"),full_page=True)

if __name__=="__main__":unittest.main(verbosity=2)
