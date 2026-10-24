"""Unified learning kernel: Chromium evidence from real PC and Android-width pages."""
import sys, threading, unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

class UnifiedLearningBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
  threading.Thread(target=cls.server.serve_forever,daemon=True).start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
  cls.base=f"http://127.0.0.1:{cls.server.server_port}"
 @classmethod
 def tearDownClass(cls):
  cls.browser.close()
  cls.pw.stop()
  cls.server.shutdown()
  cls.server.server_close()

 def test_studio_and_shared_hub_same_contract(self):
  page=self.browser.new_page(viewport={"width":1366,"height":900})
  errors=[]
  page.on("pageerror",lambda e:errors.append(str(e)))
  try:
   self.assertEqual(page.goto(self.base+"/learn.html").status,200)
   page.wait_for_function("!!window.SeetheTruthUnifiedWorkbench")
   counts=page.evaluate("window.SeetheTruthUnifiedWorkbench.counts")
   self.assertGreaterEqual(counts["operators"],146)
   self.assertEqual(counts["graphLessons"],65)  # 60 guided including source-pinned Wan video RoPE + blank + 2 pack recipes + state shortcut
   self.assertEqual(counts["courses"],10)
   self.assertEqual(counts["labs"],17)
   evidence=page.evaluate("window.SeetheTruthUnifiedWorkbench.inspect().evidence")
   self.assertEqual(evidence,"STATIC_ONLY")
   page.locator('[data-preset="nd_attention_lesson"]').click()
   self.assertEqual(page.evaluate("window.SeetheTruthUnifiedWorkbench.inspect().kind"),"graph")
   self.assertEqual(errors,[])
  finally:
   page.close()
  phone=self.browser.new_page(viewport={"width":390,"height":844},is_mobile=True,has_touch=True)
  errors=[]
  phone.on("pageerror",lambda e:errors.append(str(e)))
  try:
   self.assertEqual(phone.goto(self.base+"/teaching.html?edition=teaching").status,200)
   phone.wait_for_function("!!window.SeetheTruthTeachingHub")
   self.assertEqual(phone.evaluate("window.SeetheTruthTeachingHub.coreSchema"),
    "seethetruth-learning-experience-v1")
   phone.locator("#voteCase").select_option("fallback")
   self.assertTrue(phone.evaluate("window.SeetheTruthLastSharedVote.secondVoteFallback"))
   self.assertEqual(errors,[])
  finally:
   phone.close()

 def test_restored_engineering_journeys_share_actual_core(self):
  for route,id,global_name,count in [
   ("engineering_journey.html","engineering_journey","SeetheTruthEngineeringJourney",12),
   ("engineering_pipeline.html","engineering_pipeline","SeetheTruthEngineeringPipeline",6)]:
   with self.subTest(route=route):
    page=self.browser.new_page(viewport={"width":1280,"height":900})
    errors=[]
    page.on("pageerror",lambda e:errors.append(str(e)))
    try:
     self.assertEqual(page.goto(self.base+"/"+route).status,200)
     page.wait_for_function("(name) => !!window[name] && !!window.SeetheTruthSpecializedLesson",arg=global_name)
     self.assertEqual(page.evaluate("window.SeetheTruthSpecializedLesson.id"),id)
     self.assertEqual(page.evaluate("(name) => window[name].lessons",arg=global_name),count)
     self.assertEqual(errors,[])
    finally:
     page.close()

 def test_shared_numeric_specialist_adapters(self):
  page=self.browser.new_page(viewport={"width":1280,"height":850})
  errors=[]
  page.on("pageerror",lambda e:errors.append(str(e)))
  try:
   self.assertEqual(page.goto(self.base+"/wan_ffn_lab.html").status,200)
   page.wait_for_function("!!window.SeetheTruthSpecializedLesson")
   result=page.evaluate("""() => {
    const x=window.SeetheTruthSpecializedLesson.execute("trace",{options:{
      seed:17,tokens:3,dim:4,hidden:8}});
    return {steps:x.computationTrace.steps.length,
     tier:x.computationTrace.evidenceTier,
     parity:JSON.stringify(x.value)===JSON.stringify(
      window.SeetheTruthWanFFNTrace.trace({seed:17,tokens:3,dim:4,hidden:8}))};
   }""")
   self.assertEqual(result["steps"],9)
   self.assertEqual(page.evaluate("window.SeetheTruthWanFFNLearning.stages"),9)
   self.assertEqual(result["tier"],"PINNED_WAN_SOURCE_SYNTHETIC_FLOAT64")
   self.assertTrue(result["parity"])
   self.assertEqual(errors,[])
  finally:
   page.close()
  page=self.browser.new_page(viewport={"width":390,"height":844},is_mobile=True,has_touch=True)
  errors=[]
  page.on("pageerror",lambda e:errors.append(str(e)))
  try:
   self.assertEqual(page.goto(self.base+"/engineering_pipeline.html").status,200)
   page.wait_for_function("!!window.SeetheTruthSpecializedLesson")
   x=page.evaluate("""() => {
    const r=window.SeetheTruthSpecializedLesson.execute("train",{config:{seed:7},steps:2});
    return {steps:r.stateTrace.steps.length,tier:r.stateTrace.evidenceTier};
   }""")
   self.assertEqual(x["steps"],3)
   self.assertEqual(x["tier"],"EXECUTABLE_BIGRAM_OPTIMIZER_STATE_SIMULATION")
   self.assertEqual(errors,[])
  finally:
   page.close()

 def test_independent_lab_adapter_preserves_domain_truth(self):
  for route,id,domain in [
   ("deepinfer_worker_lab.html","deepinfer_worker","event_dag"),
   ("deepinfer_kv_lifecycle_lab.html","deepinfer_kv","kv_state_machine"),
   ("wan_ffn_lab.html","wan_ffn","source_operator"),
   ("deepinfer_prefill_lab.html","deepinfer_prefill","scheduler_decision")]:
   with self.subTest(route=route):
    page=self.browser.new_page(viewport={"width":1280,"height":800})
    errors=[]
    page.on("pageerror",lambda e:errors.append(str(e)))
    try:
     self.assertEqual(page.goto(self.base+"/"+route).status,200)
     page.wait_for_function("!!window.SeetheTruthSpecializedLesson")
     contract=page.evaluate("""() => {
      const l=window.SeetheTruthSpecializedLesson;
      return {id:l.id,runtime:l.runtime,evidence:l.evidence,engine:l.engine};
     }""")
     self.assertEqual(contract["id"],id)
     self.assertEqual(contract["runtime"],domain)
     self.assertEqual(contract["evidence"],"NOT_EXECUTED")
     self.assertTrue(contract["engine"].endswith(".js"))
     if id in ("deepinfer_worker","deepinfer_kv","deepinfer_scheduler","deepinfer_prefill"):
      # Change the real lab controller's scenario; the inspector must replay
      # that accepted input rather than displaying an unrelated fixture.
      if id=="deepinfer_worker":
       page.locator("#forwardTime").fill("11")
      elif id=="deepinfer_kv":
       page.locator("#scenario").select_option("preempt")
      elif id=="deepinfer_scheduler":
       page.locator("#scenario").select_option("ample")
      else:
       page.locator("#scenario").select_option("grow")
      page.locator("#runUnifiedState").click()
      self.assertGreater(page.locator("#unifiedStateSteps .sm-step").count(),1)
      self.assertIn("当前实验结果一致",page.locator("#unifiedStateStatus").inner_text())
      self.assertTrue(page.evaluate("""() => {
       const x=window.SeetheTruthCurrentDomainRun;
       const y=window.SeetheTruthUnifiedStateLab.accepted;
       return x && y && x.id===y.id &&
          JSON.stringify(x.input)===JSON.stringify(y.input);
      }"""))
      if id=="deepinfer_kv":
       self.assertEqual(page.evaluate("window.SeetheTruthUnifiedStateLab.frame.steps.length"),
        page.evaluate("window.SeetheTruthCurrentDomainRun.input.operations.length+1"))
      if id=="deepinfer_scheduler":
       self.assertEqual(page.evaluate("window.SeetheTruthUnifiedStateLab.frame.steps.length"),
        page.evaluate("window.SeetheTruthCurrentDomainRun.observed.decisions.length+1"))
      if id=="deepinfer_prefill":
       self.assertEqual(page.evaluate("window.SeetheTruthUnifiedStateLab.frame.steps[1].dependencies.length"),0)
     self.assertEqual(errors,[])
    finally:
     page.close()

if __name__=="__main__":
 unittest.main()
