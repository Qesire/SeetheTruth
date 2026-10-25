"""Native Bigram training actions and portable optimizer rewind in Chromium."""
import sys,threading,unittest,json
from http.server import ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

class NativeTrainingTimelineBrowser(unittest.TestCase):
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
 def test_source_native_pending_gradient_resume(self):
  for width in (1366,390):
   with self.subTest(width=width):
    page=self.browser.new_page(viewport={"width":width,"height":850},
      is_mobile=width==390,has_touch=width==390,accept_downloads=True)
    errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
    try:
     self.assertEqual(page.goto(self.base+'/engineering_pipeline.html').status,200)
     page.wait_for_function("!!window.SeetheTruthNativeTrainingTimeline")
     self.assertTrue(page.locator('#trainingTimelineMicro').is_disabled())
     page.locator('#trainingTimelineStart').click()
     self.assertTrue(page.locator('#trainingTimelineOptimizer').is_disabled())
     for i in range(4):
      page.locator('#trainingTimelineMicro').click()
      if i<3:self.assertTrue(page.locator('#trainingTimelineOptimizer').is_disabled())
     self.assertTrue(page.locator('#trainingTimelineOptimizer').is_enabled())
     page.locator('#trainingTimelineOptimizer').click()
     self.assertEqual(page.evaluate("window.SeetheTruthNativeTrainingTimeline.accepted.value.snapshots.at(-1).model.step"),1)
     self.assertEqual(page.locator('#unifiedObservationStep option').count(),6)
     page.locator('#unifiedObservationStep').select_option(label='3. train:2')
     with page.expect_download() as capture:
      page.locator('#unifiedObservationCheckpoint').click()
     self.assertIn('-checkpoint.json',capture.value.suggested_filename)
     self.assertTrue(page.locator('#unifiedObservationResumeTraining').is_enabled())
     page.locator('#unifiedObservationResumeTraining').click()
     info=json.loads(page.locator('#unifiedObservationDetail').inner_text())
     self.assertEqual(info['stepId'],'train:2')
     self.assertEqual(info['optimizerStep'],0)
     self.assertEqual(info['pendingMicroSteps'],2)
     self.assertEqual(info['verifiedSuffixStates'],3)
     self.assertTrue(info['finalReplayed'])
     self.assertTrue(info['continuedFromCheckpoint'])
     snapshot=page.evaluate("""()=> {
      const ui=window.SeetheTruthNativeTrainingTimeline.panel;
      const core=window.SeetheTruthUnifiedCore;
      return window.SeetheTruthObservationSnapshots.at(
       core,window.SeetheTruthObservationSnapshots.create(core,ui.current),'train:2');
     }""")
     page.locator('#trainingTimelineStart').click()
     page.locator('#unifiedObservationImport').set_input_files({
      'name':'training-state.json','mimeType':'application/json',
      'buffer':json.dumps(snapshot).encode('utf8')
     })
     page.wait_for_function("document.querySelector('#unifiedObservationStatus')?.textContent.includes('train:2')")
     self.assertEqual(page.locator('#unifiedObservationStep').input_value(),'2')
     self.assertTrue(page.locator('#unifiedObservationResumeTraining').is_enabled())
     page.locator('#unifiedObservationResumeTraining').click()
     self.assertIn('"finalReplayed": true',page.locator('#unifiedObservationDetail').inner_text())
     self.assertTrue(page.locator('#trainingTimelineMicro').is_enabled(),
      'the independent teaching controller is not modified by readonly checkpoint import')
     self.assertEqual(errors,[])
    finally:page.close()
if __name__=='__main__':unittest.main()
