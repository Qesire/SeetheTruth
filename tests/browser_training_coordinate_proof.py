"""Desktop/mobile real parameter-element proof from a committed graph training run."""
import sys,threading,unittest,json
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

class TrainingCoordinateProofBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
  threading.Thread(target=cls.server.serve_forever,daemon=True).start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
  cls.base=f"http://127.0.0.1:{cls.server.server_port}/learn.html"
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
 def test_executed_training_parameter_coordinate_and_path_proof(self):
  for width in [1366,390]:
   with self.subTest(viewport=width):
    page=self.browser.new_page(viewport={"width":width,"height":900},
       is_mobile=width==390,has_touch=width==390)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    try:
     self.assertEqual(page.goto(self.base).status,200)
     page.locator('[data-preset="trainable_core"]').click()
     page.locator('#studyMode').select_option('numeric')
     page.locator('#trainStep').click()
     self.assertNotIn('训练未提交',page.locator('#status').inner_text())
     self.assertEqual(page.locator('#trainingProofExecute').count(),1)
     page.locator('#trainingProofParameter').fill('shared_weight')
     page.locator('#trainingProofIndex').fill('0,1')
     page.locator('#trainingProofVia').fill('project')
     page.locator('#trainingProofViaIndex').fill('0,0')
     page.locator('#trainingProofExecute').click()
     status=page.locator('#trainingProofStatus').inner_text()
     self.assertIn('已核验参数 shared_weight',status)
     body=json.loads(page.locator('#trainingProofDetail').inner_text())
     self.assertEqual(body['parameterId'],'shared_weight')
     self.assertEqual(body['index'],[0,1])
     self.assertTrue(isinstance(body['wholeLossGradient'],(int,float)))
     self.assertEqual(body['optimizer']['stepAfter'],1)
     self.assertNotEqual(body['parameter']['before'],body['parameter']['after'])
     self.assertEqual(body['selectedVia']['nodeId'],'project')
     self.assertEqual(page.evaluate('window.SeetheTruthUnifiedWorkbench.selectedNode'),'shared_weight')
     focus=page.evaluate('window.SeetheTruthCoordinateFocusSession.current')
     self.assertEqual(focus['domain'],'training_parameter')
     self.assertEqual(focus['evidence'],'EXECUTED_BOUNDED_CPU_TRAINING_COORDINATE')
     self.assertEqual(focus['coordinate'],[0,1])
     self.assertEqual(focus['value'],body['parameter']['after'])
     self.assertEqual(focus['transition']['wholeLossGradient'],body['wholeLossGradient'])
     self.assertIn('全 Loss 梯度',page.locator('#sharedCoordinateFocus').inner_text())
     page.locator('#sharedCoordinateDrilldown').click()
     self.assertIn('原生参数更新',page.locator('#sharedCoordinateDrilldownResult').inner_text())
     self.assertEqual(page.locator("#coordinateJourneyCount").inner_text(),"1 / 6")
     self.assertIn('更新前',page.locator('#coordinateJourneyTitle').inner_text())
     page.locator("#coordinateJourneyNext").click()
     self.assertIn("全 Loss 梯度",page.locator("#coordinateJourneyTitle").inner_text())
     self.assertIn("原生证据值",page.locator("#sharedCoordinateJourney").inner_text())
     page.locator("#coordinateJourneyNext").click()
     self.assertIn("优化器状态",page.locator("#coordinateJourneyTitle").inner_text())
     page.locator("#coordinateJourneyNext").click()
     self.assertIn("更新后",page.locator("#coordinateJourneyTitle").inner_text())
     self.assertIn("参数位移",page.locator("#sharedCoordinateJourney").inner_text())
     Path("training_coordinate_screenshots").mkdir(exist_ok=True)
     page.screenshot(path=f"training_coordinate_screenshots/optimizer-timeline-{width}.png",
                     animations="disabled",full_page=True)
     # Reject unsafe/invalid coordinates without displaying stale accepted proof.
     invalid=f"{focus['shape'][0]},0"
     page.locator('#trainingProofIndex').fill(invalid)
     self.assertEqual(page.locator('#trainingProofIndex').input_value(),invalid)
     self.assertEqual(page.locator('#trainingProofIndex').count(),1)
     self.assertIn('坐标验证未完成',page.locator('#trainingProofStatus').inner_text())
     self.assertEqual(page.locator('#trainingProofDetail').inner_text(),'')
     self.assertEqual(page.evaluate('window.SeetheTruthCoordinateFocusSession.current'),None)
     page.locator('#trainingProofExecute').click()
     self.assertIn('坐标验证未完成',page.locator('#trainingProofStatus').inner_text())
     self.assertEqual(page.locator('#trainingProofDetail').inner_text(),'')
     self.assertEqual(page.evaluate('window.SeetheTruthCoordinateFocusSession.current'),None)
     self.assertTrue(page.locator('#sharedCoordinateJourney').is_hidden())
     # A second valid request must actually execute, not merely leave a stale status.
     page.locator('#trainingProofIndex').fill('0,1')
     self.assertIn('坐标验证未完成',page.locator('#trainingProofStatus').inner_text())
     page.locator('#trainingProofExecute').click()
     self.assertIn('已核验参数 shared_weight',page.locator('#trainingProofStatus').inner_text())
     self.assertEqual(page.evaluate('window.SeetheTruthCoordinateFocusSession.current.nodeId'),'shared_weight')
     self.assertEqual(errors,[])
    finally:page.close()
if __name__=='__main__':unittest.main()
