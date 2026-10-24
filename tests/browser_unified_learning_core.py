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
   self.assertEqual(counts["graphLessons"],62)  # 59 guided lessons + blank canvas + 2 extension recipes
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

 def test_independent_lab_adapter_preserves_domain_truth(self):
  for route,id,domain in [
   ("deepinfer_worker_lab.html","deepinfer_worker","event_dag"),
   ("deepinfer_kv_lifecycle_lab.html","deepinfer_kv","kv_state_machine"),
   ("wan_ffn_lab.html","wan_ffn","source_operator")]:
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
     self.assertEqual(errors,[])
    finally:
     page.close()

if __name__=="__main__":
 unittest.main()
