"""Shared observation UI consumes accepted graph/Wan/Worker records in real Chromium."""
import sys,threading,unittest,json
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
     page.locator('#run').click()  # conceptual traversal is not numeric evidence
     page.locator('#unifiedObservationRefresh').click()
     self.assertIn('STATIC_ONLY',page.locator('#unifiedObservationStatus').inner_text())
     self.assertEqual(page.locator('#unifiedObservationCoordinates input').count(),0)
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
 def test_portable_snapshot_cross_page_handoff_and_corruption(self):
  page,errors=self.visit("wan_ffn_lab.html",1366)
  try:
   page.locator('#unifiedObservationRefresh').click()
   baseline=page.evaluate("""() => {
    const core=window.SeetheTruthUnifiedCore;
    const ui=window.SeetheTruthObservationPanel;
    return window.SeetheTruthObservationSnapshots.create(core,ui.current,"Wan portable");
   }""")
   self.assertEqual(baseline['kind'],'specialized_execution')
   self.assertEqual(baseline['label'],'Wan portable')
   page.goto(self.base+'/deepinfer_worker_lab.html')
   page.wait_for_function('!!window.SeetheTruthObservationPanel')
   page.locator('#unifiedObservationImport').set_input_files({
    'name':'wan-observation.json','mimeType':'application/json',
    'buffer':json.dumps(baseline).encode('utf8')
   })
   page.wait_for_function("document.querySelector('#unifiedObservationOpenSource')?.hidden===false")
   self.assertTrue(page.locator('#unifiedObservationOpenSource').is_visible())
   self.assertIn('尚未重放核验',page.locator('#unifiedObservationStatus').inner_text())
   page.locator('#unifiedObservationOpenSource').click()
   page.wait_for_url('**/wan_ffn_lab.html')
   page.wait_for_function('!!window.SeetheTruthObservationPanel')
   self.assertIn('已恢复并重放验证',page.locator('#unifiedObservationStatus').inner_text())
   page.locator('#unifiedObservationCompare').click()
   self.assertIn('"valueEqual": true',page.locator('#unifiedObservationDiff').inner_text())
   corrupt=dict(baseline)
   corrupt['checksum']='fnv1a32:deadbeef:123'
   outcome=page.evaluate("""(snap)=>{
    try{window.SeetheTruthObservationPanel.importSnapshot(snap);return 'accepted'}
    catch(e){return e.message;}
   }""",corrupt)
   self.assertIn('replayed source result differs',outcome)
   self.assertEqual(errors,[])
  finally:page.close()
 def test_graph_snapshot_roundtrip_static_and_numeric(self):
  page,errors=self.visit("learn.html",390)
  try:
   page.locator('[data-preset="residual"]').click()
   page.locator('#studyMode').select_option('numeric')
   page.locator('#run').click()
   page.locator('#unifiedObservationRefresh').click()
   snapshot=page.evaluate("""() => {
    const ui=window.SeetheTruthObservationPanel;
    return window.SeetheTruthObservationSnapshots.create(window.SeetheTruthUnifiedCore,ui.current);
   }""")
   self.assertTrue(snapshot['numeric'])
   restored=page.evaluate('(snap)=>window.SeetheTruthObservationPanel.importSnapshot(snap)',snapshot)
   self.assertTrue(restored['restored'])
   page.locator('#unifiedObservationCompare').click()
   self.assertIn('"valueEqual": true',page.locator('#unifiedObservationDiff').inner_text())
   with page.expect_download() as info:
    page.locator('#unifiedObservationExport').click()
   self.assertIn('seethetruth-observation-',info.value.suggested_filename)
   page.locator('[data-preset="attention"]').click()
   page.locator('#run').click()
   with page.expect_download() as branch_download:
    page.locator('#unifiedObservationFork').click()
   self.assertIn('-branch.json',branch_download.value.suggested_filename)
   self.assertIn('父快照',page.locator('#unifiedObservationDiff').inner_text())
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
