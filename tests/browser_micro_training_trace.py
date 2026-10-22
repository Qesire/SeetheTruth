"""Browser acceptance for shared Tape/MoE/PPO operator microscope."""
import threading,unittest,sys
from pathlib import Path
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
from playwright.sync_api import sync_playwright
OUT=Path('micro_training_trace_screenshots')
class MicroTrainingTraceBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  OUT.mkdir(parents=True,exist_ok=True)
  cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
  cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=['--no-sandbox'])
  cls.root='http://127.0.0.1:'+str(cls.server.server_port)
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
 def test_actual_shared_weight_forward_backward(self):
  p=self.browser.new_page(viewport={'width':1440,'height':900})
  errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
  self.assertEqual(p.goto(self.root+'/micro_training.html').status,200)
  p.wait_for_function("document.querySelectorAll('#computeMicroscope .cmp-step').length >= 10")
  self.assertIn('真实 Tape',p.locator('#computeMicroscope').inner_text())
  p.locator('#computeMicroscope .cmp-step').nth(2).click()
  self.assertIn('matmul',p.locator('#computeMicroscope .cmp-meta').inner_text())
  self.assertGreater(p.locator('#computeMicroscope .cmp-terms tr').count(),2)
  self.assertIn('误差',p.locator('#computeMicroscope .cmp-breakdown').inner_text())
  p.locator('#computeMicroscope .cmp-step').last.click()
  self.assertIn('Backward',p.locator('#computeMicroscope .cmp-meta').inner_text())
  self.assertIn('VJP',p.locator('#computeMicroscope').inner_text())
  p.screenshot(path=str(OUT/'tape-shared-parameter-gradient.png'),full_page=True,animations='disabled')
  self.assertEqual(errors,[]);p.close()
 def test_moe_capacity_and_ppo_clip_on_mobile(self):
  p=self.browser.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True,device_scale_factor=2)
  errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
  self.assertEqual(p.goto(self.root+'/micro_training.html?edition=teaching').status,200)
  p.locator('[data-case="moe"]').click()
  self.assertEqual(p.locator('#computeMicroscope .cmp-step').count(),9)
  p.locator('#computeMicroscope .cmp-step').nth(3).click()
  self.assertIn('Expert ID',p.locator('#computeMicroscope .cmp-meta').inner_text())
  p.locator('#computeMicroscope .cmp-step').nth(4).click()
  self.assertEqual(p.locator('#computeMicroscope .cmp-substep').count(),3)
  p.locator('#computeMicroscope .cmp-substep').nth(1).click()
  self.assertGreater(p.locator('#computeMicroscope .cmp-terms tr').count(),2)
  p.locator('#val_r').fill('-5')  # both tokens now select Expert 1
  p.locator('#val_capacity').fill('1')
  p.locator('#run').click()
  self.assertIn('overflow',p.locator('#metrics').inner_text())
  p.locator('#computeMicroscope .cmp-step').nth(4).click()
  p.locator('#computeMicroscope .cmp-substep').nth(2).click()
  self.assertIn('capacity blocked',p.locator('#computeMicroscope .cmp-meta').inner_text())
  self.assertEqual(p.locator('#computeMicroscope .cmp-terms tr').count(),0)
  p.screenshot(path=str(OUT/'moe-routing-capacity-mobile.png'),full_page=True,animations='disabled')
  p.locator('[data-case="ppo"]').click()
  self.assertEqual(p.locator('#computeMicroscope .cmp-step').count(),8)
  p.locator('#val_logit').fill('8')
  p.locator('#run').click()
  p.locator('#computeMicroscope .cmp-step').last.click()
  self.assertIn('gradient',p.locator('#computeMicroscope .cmp-meta').inner_text().lower())
  self.assertIn('0',p.locator('#computeMicroscope .cmp-breakdown').inner_text())
  p.screenshot(path=str(OUT/'ppo-clipped-vjp-mobile.png'),full_page=True,animations='disabled')
  p.locator('#val_logit').fill('0')
  p.locator('#run').click()
  p.locator('#computeMicroscope .cmp-step').last.click()
  self.assertNotEqual(p.locator('#computeMicroscope .cmp-breakdown').inner_text().strip(), '')
  self.assertEqual(errors,[]);p.close()
if __name__=='__main__':unittest.main()
