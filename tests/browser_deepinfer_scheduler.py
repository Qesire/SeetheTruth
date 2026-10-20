"""Real Chromium verification for the Scheduler → KV → Worker execution chain."""
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
from playwright.sync_api import sync_playwright

DEST=Path("deepinfer_scheduler_screenshots")
class DeepInferIntegratedBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DEST.mkdir(parents=True,exist_ok=True)
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
    def setUp(self):
        self.page=self.browser.new_page(viewport={"width":1500,"height":1080},accept_downloads=True)
        self.errors=[]
        self.page.on("pageerror",lambda e:self.errors.append(str(e)))
        r=self.page.goto(f"http://127.0.0.1:{self.server.server_port}/deepinfer_scheduler_lab.html")
        self.assertEqual(r.status,200)
        self.page.wait_for_function("!!window.SeetheTruthLastSchedulerRun")
    def tearDown(self):
        self.assertEqual(self.errors,[])
        self.page.close()
    def test_memory_pressure_preempt_then_explicit_readmission(self):
        p=self.page
        self.assertEqual(p.locator("#blocks .block").count(),7)
        self.assertEqual(p.locator("#decisionRows tr").count(),4)
        initial=p.evaluate("window.SeetheTruthLastSchedulerState")
        self.assertEqual(initial["free"],[])
        p.locator("#next").click()
        state=p.evaluate("window.SeetheTruthLastSchedulerState")
        run=p.evaluate("window.SeetheTruthLastSchedulerRun")
        self.assertEqual(run["decisions"][0]["preempted"][0]["id"],"A")
        self.assertEqual(run["decisions"][0]["preempted"][0]["outcome"],"recompute")
        self.assertEqual(state["requests"]["A"]["status"],"WAITING")
        self.assertEqual(run["decisions"][0]["scheduled"],["B","C"])
        self.assertEqual(run["worker"]["steps"][0]["sequenceIds"],["B","C"])
        self.assertEqual(p.locator("#workerTimeline svg rect").count()>0,True)
        p.screenshot(path=str(DEST/"memory-pressure-preempt.png"),full_page=True,animations="disabled")
        p.locator("#next").click()
        self.assertEqual(p.evaluate("window.SeetheTruthLastSchedulerRun.decisions[1].preempted[0].id"),"B")
        p.locator("#next").click()
        self.assertEqual(p.evaluate("window.SeetheTruthLastSchedulerState.requests.A.status"),"RUNNING")
        self.assertEqual(p.evaluate("window.SeetheTruthLastSchedulerRun.decisions[2].scheduled.includes('A')"),True)
        p.screenshot(path=str(DEST/"readmitted-recompute-request.png"),full_page=True,animations="disabled")
        with p.expect_download() as d:
            p.locator("#download").click()
        exported=json.loads(Path(d.value.path()).read_text(encoding="utf8"))
        self.assertEqual(exported["summary"]["preemptions"],2)
        self.assertEqual(len(exported["worker"]["steps"]),4)
        self.assertEqual(exported["snapshots"][-1]["free"].__len__(),6)
    def test_structured_output_termination_and_ample_capacity(self):
        p=self.page
        p.locator("#scenario").select_option("structured")
        p.locator("#next").click()
        outcome=p.evaluate("window.SeetheTruthLastSchedulerRun")
        self.assertEqual(outcome["decisions"][0]["preempted"][0]["outcome"],"stop")
        self.assertEqual(p.evaluate("window.SeetheTruthLastSchedulerState.requests.S.status"),"FINISHED_PREEMPTED")
        p.screenshot(path=str(DEST/"structured-output-stop.png"),full_page=True,animations="disabled")
        p.locator("#scenario").select_option("ample")
        out=p.evaluate("window.SeetheTruthLastSchedulerRun")
        self.assertEqual(out["summary"]["preemptions"],0)
        self.assertEqual(out["summary"]["finalFreeBlocks"],9)
        p.locator("#stopPreempted").check()
        p.locator("#scenario").select_option("pressure")
        self.assertEqual(p.evaluate("window.SeetheTruthLastSchedulerRun.decisions[0].preempted[0].outcome"),"stop")
    def test_editor_rejects_invalid_workflow_atomically(self):
        p=self.page
        p.locator("#editor").evaluate('(e)=>{e.closest("details").open=true;}')
        before=p.evaluate("window.SeetheTruthLastSchedulerRun.summary")
        p.locator("#editor").fill('{"config":{"numBlocks":4,"blockSize":4},"setup":[{"op":"arrive","id":"A","tokenLen":16},{"op":"allocate","id":"A"}],"cycles":[{"outputLens":{"A":1}}]}')
        p.locator("#apply").click()
        self.assertIn("输入被拒绝",p.locator("#status").inner_text())
        self.assertEqual(p.evaluate("window.SeetheTruthLastSchedulerRun.summary"),before)
        p.locator("#editor").fill(json.dumps({
            "config":{"numBlocks":6,"blockSize":4},
            "setup":[{"op":"arrive","id":"Z","tokenLen":4},{"op":"allocate","id":"Z"}],
            "cycles":[{"outputLens":{"Z":0}},{"outputLens":{"Z":1},"after":[{"op":"finish","id":"Z"}]}]
        }))
        p.locator("#apply").click()
        self.assertIn("验证通过",p.locator("#status").inner_text())
        self.assertEqual(p.evaluate("window.SeetheTruthLastSchedulerRun.summary.finalFreeBlocks"),5)
        self.assertEqual(p.locator("#decisionRows tr").count(),2)
        p.screenshot(path=str(DEST/"custom-scheduler-workflow.png"),full_page=True,animations="disabled")
if __name__=="__main__":
    unittest.main()
