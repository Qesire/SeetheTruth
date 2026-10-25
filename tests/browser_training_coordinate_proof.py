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
     # Reject unsafe/invalid coordinates without displaying stale accepted proof.
     page.locator('#trainingProofIndex').fill('10,10')
     page.locator('#trainingProofExecute').click()
     self.assertIn('坐标验证未完成',page.locator('#trainingProofStatus').inner_text())
     self.assertEqual(page.locator('#trainingProofDetail').inner_text(),'')
     self.assertEqual(errors,[])
    finally:page.close()
if __name__=='__main__':unittest.main()
