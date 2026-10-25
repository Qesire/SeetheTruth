"""Shared observation UI consumes accepted graph/Wan/Worker records in real Chromium."""
import sys,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

class UnifiedObservationBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
  threading.Thread(target=cls.server.serve_forever,daemon=True).start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
  cls.base=f"http://127.0.0.1:{cls.server.server_port}"
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
 def visit(self,route,width):
  page=self.browser.new_page(viewport={"width":width,"height":880},is_mobile=width==390,has_touch=width==390)
  errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
  self.assertEqual(page.goto(self.base+"/"+route).status,200)
  page.wait_for_function("!!window.SeetheTruthObservationPanel")
  return page,errors
 def test_graph_numeric_static_and_diff(self):
  for width in (1366,390):
   with self.subTest(width=width):
    page,errors=self.visit("learn.html",width)
    try:
     page.locator('#unifiedObservationRefresh').click()
     self.assertIn('STATIC_ONLY',page.locator('#unifiedObservationStatus').inner_text())
     self.assertIn('无已执行原生轨迹',page.locator('#unifiedObservationDetail').inner_text())
     page.locator('[data-preset="residual"]').click()
     page.locator('#studyMode').select_option('numeric')
     page.locator('#run').click()
     page.locator('#unifiedObservationRefresh').click()
     self.assertGreater(page.locator('#unifiedObservationStep option').count(),2)
     page.locator('#unifiedObservationStep').select_option(label='4. sum')
     page.locator('#unifiedObservationInspect').click()
     detail=page.locator('#unifiedObservationDetail').inner_text()
     self.assertIn('"replayVerified": true',detail)
     self.assertIn('"coordinate"',detail)
     page.locator('#unifiedObservationPin').click()
     page.locator('[data-preset="attention"]').click()
     page.locator('#run').click()
     page.locator('#unifiedObservationCompare').click()
     self.assertIn('"inputEqual": false',page.locator('#unifiedObservationDiff').inner_text())
     self.assertEqual(errors,[])
    finally:page.close()
 def test_wan_numeric_and_gate_baseline(self):
  for width in (1366,390):
   with self.subTest(width=width):
    page,errors=self.visit("wan_ffn_lab.html",width)
    try:
     page.locator('#unifiedObservationRefresh').click()
     self.assertIn('PINNED_SOURCE_SYNTHETIC',page.locator('#unifiedObservationStatus').inner_text())
     page.locator('#unifiedObservationStep').select_option(label='5. up')
     page.locator('#unifiedObservationCoordinates input').first.fill('1')
     page.locator('#unifiedObservationInspect').click()
     self.assertIn('"reconstructable": true',page.locator('#unifiedObservationDetail').inner_text())
     page.locator('#unifiedObservationPin').click()
     page.locator('#zeroGate').click()
     page.locator('#unifiedObservationCompare').click()
     self.assertIn('"valueEqual": false',page.locator('#unifiedObservationDiff').inner_text())
     self.assertEqual(errors,[])
    finally:page.close()
 def test_shared_state_adapter_mounts_across_deepinfer_domains(self):
  cases=[('deepinfer_prefill_lab.html','deepinfer_prefill'),
         ('deepinfer_kv_lifecycle_lab.html','deepinfer_kv'),
         ('deepinfer_scheduler_lab.html','deepinfer_scheduler')]
  for width in (1366,390):
   for route,lab in cases:
    with self.subTest(width=width,lab=lab):
     page,errors=self.visit(route,width)
     try:
      page.wait_for_function("!!window.SeetheTruthCurrentDomainRun")
      self.assertEqual(page.evaluate('window.SeetheTruthSpecializedLesson.id'),lab)
      page.locator('#unifiedObservationRefresh').click()
      self.assertIn('已核验',page.locator('#unifiedObservationStatus').inner_text())
      self.assertGreater(page.locator('#unifiedObservationStep option').count(),0)
      self.assertEqual(page.locator('#unifiedObservationCoordinates input').count(),0)
      page.locator('#unifiedObservationInspect').click()
      self.assertIn('"replayVerified": true',page.locator('#unifiedObservationDetail').inner_text())
      page.locator('#unifiedObservationPin').click()
      page.locator('#unifiedObservationCompare').click()
      self.assertIn('"valueEqual": true',page.locator('#unifiedObservationDiff').inner_text())
      self.assertEqual(errors,[])
     finally:page.close()
 def test_worker_state_no_fabricated_coordinates(self):
  for width in (1366,390):
   with self.subTest(width=width):
    page,errors=self.visit("deepinfer_worker_lab.html",width)
    try:
     page.locator('#unifiedObservationRefresh').click()
     self.assertIn('ABSTRACT_TIMELINE',page.locator('#unifiedObservationStatus').inner_text())
     self.assertEqual(page.locator('#unifiedObservationCoordinates input').count(),0)
     page.locator('#unifiedObservationInspect').click()
     self.assertIn('"before"',page.locator('#unifiedObservationDetail').inner_text())
     page.locator('#unifiedObservationPin').click()
     page.locator('#unifiedObservationCompare').click()
     self.assertIn('"valueEqual": true',page.locator('#unifiedObservationDiff').inner_text())
     self.assertEqual(errors,[])
    finally:page.close()
if __name__=="__main__":
 unittest.main()
