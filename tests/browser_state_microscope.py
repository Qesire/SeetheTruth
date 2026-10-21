"""Browser proof: shared state microscope in two independent real CPU simulators
and the explicitly non-universal capabilities coverage matrix."""
import sys,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
from playwright.sync_api import sync_playwright

OUT=Path('state_microscope_screenshots')
class StateMicroscopeBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  OUT.mkdir(parents=True,exist_ok=True)
  cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
  cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=['--no-sandbox'])
  cls.url='http://127.0.0.1:'+str(cls.server.server_port)
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()

 def test_kv_genuine_refcount_and_state_delta(self):
  p=self.browser.new_page(viewport={'width':1420,'height':930})
  errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
  self.assertEqual(p.goto(self.url+'/deepinfer_kv_lifecycle_lab.html').status,200)
  p.wait_for_function("document.querySelectorAll('#stateMicroscope .sm-step').length > 8")
  self.assertIn('SOURCE_GROUNDED_KV_CPU_STATE_SIMULATION',p.locator('#stateMicroscope').inner_text())
  p.locator('#next').click()
  self.assertIn('refs',p.locator('#stateMicroscope').inner_text())
  self.assertIn('freeQueue',p.locator('#stateMicroscope').inner_text())
  p.locator('#stateMicroscope .sm-step').nth(5).click()
  self.assertIn('fork',p.locator('#stateMicroscope').inner_text().lower())
  p.screenshot(path=str(OUT/'kv-shared-state-delta.png'),full_page=True,animations='disabled')
  self.assertEqual(errors,[]);p.close()

 def test_worker_causal_timing_without_fake_gpu_measurement(self):
  p=self.browser.new_page(viewport={'width':1420,'height':930})
  errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
  self.assertEqual(p.goto(self.url+'/deepinfer_worker_lab.html').status,200)
  p.wait_for_function("document.querySelectorAll('#stateMicroscope .sm-step').length > 8")
  self.assertIn('SOURCE_GROUNDED_WORKER_ASSUMED_DURATIONS',p.locator('#stateMicroscope').inner_text())
  self.assertIn('非实测',p.locator('#stateMicroscope').inner_text())
  p.locator('#stateMicroscope .sm-step').nth(3).click()
  self.assertIn('实际前置依赖',p.locator('#stateMicroscope').inner_text())
  self.assertIn('completedEvents',p.locator('#stateMicroscope .sm-changes').inner_text())
  p.screenshot(path=str(OUT/'worker-shared-causal-event.png'),full_page=True,animations='disabled')
  self.assertEqual(errors,[]);p.close()

 def test_capability_gap_inventory_on_phone(self):
  p=self.browser.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True,device_scale_factor=2)
  errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
  self.assertEqual(p.goto(self.url+'/capabilities.html?edition=teaching').status,200)
  p.wait_for_function("document.querySelectorAll('#surfaceAuditRows tr').length === 20")
  self.assertEqual(p.locator('#coreTensorCount').inner_text(),'2')
  self.assertEqual(p.locator('#coreStateCount').inner_text(),'4')
  self.assertIn('尚待复用',p.locator('#surfaceAuditRows').inner_text())
  p.screenshot(path=str(OUT/'mobile-universal-audit-matrix.png'),full_page=True,animations='disabled')
  self.assertEqual(errors,[]);p.close()

if __name__=='__main__':unittest.main()
