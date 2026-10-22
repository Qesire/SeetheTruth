"""Real browser interactions for the shared paper computation microscope."""
import threading,unittest,sys
from pathlib import Path
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
from playwright.sync_api import sync_playwright
OUT=Path('paper_compute_trace_screenshots')
class PaperComputeTraceBrowser(unittest.TestCase):
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
 def test_lora_and_dora_gemms_norms_and_weight_updates(self):
  p=self.browser.new_page(viewport={'width':1450,'height':940})
  errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
  self.assertEqual(p.goto(self.root+'/paper_lab.html').status,200)
  p.wait_for_function("document.querySelectorAll('#computeMicroscope .cmp-step').length===9")
  p.locator('#computeMicroscope .cmp-step').nth(6).click()
  self.assertIn('Rank-up',p.locator('#computeMicroscope .cmp-meta').inner_text())
  self.assertGreater(p.locator('#computeMicroscope .cmp-terms tr').count(),2)
  p.screenshot(path=str(OUT/'lora-lowrank-matmul-dot.png'),full_page=True,animations='disabled')
  p.locator('#param_alpha').fill('0');p.locator('#calculate').click()
  self.assertIn('[',p.locator('#numeric').inner_text())
  p.locator('[data-patch="dora_linear"]').click()
  self.assertEqual(p.locator('#computeMicroscope .cmp-step').count(),11)
  p.locator('#computeMicroscope .cmp-step').nth(8).click()
  self.assertIn('column norms',p.locator('#computeMicroscope .cmp-meta').inner_text().lower())
  p.locator('#computeMicroscope .cmp-step').nth(10).click()
  self.assertIn('X W_eff',p.locator('#computeMicroscope .cmp-meta').inner_text())
  self.assertGreater(p.locator('#computeMicroscope .cmp-terms tr').count(),2)
  p.screenshot(path=str(OUT/'dora-column-norm-effective-gemm.png'),full_page=True,animations='disabled')
  self.assertEqual(errors,[]);p.close()
 def test_online_softmax_tile_recurrence_and_symbolic_boundary_on_phone(self):
  p=self.browser.new_page(viewport={'width':390,'height':844},device_scale_factor=2,is_mobile=True,has_touch=True)
  errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
  self.assertEqual(p.goto(self.root+'/paper_lab.html?edition=teaching').status,200)
  p.locator('[data-patch="online_attention"]').click()
  self.assertEqual(p.locator('#computeMicroscope .cmp-step').count(),8)
  p.locator('#computeMicroscope .cmp-step').nth(3).click()
  self.assertEqual(p.locator('#computeMicroscope .cmp-substep').count(),7)
  p.locator('#computeMicroscope .cmp-substep').nth(1).click()
  self.assertIn('Scores',p.locator('#computeMicroscope .cmp-meta').inner_text())
  self.assertGreater(p.locator('#computeMicroscope .cmp-terms tr').count(),2)
  p.locator('#computeMicroscope .cmp-substep').nth(4).click()
  self.assertIn('Denominator',p.locator('#computeMicroscope .cmp-meta').inner_text())
  p.screenshot(path=str(OUT/'online-softmax-tile-mobile.png'),full_page=True,animations='disabled')
  p.locator('#param_chunk').fill('2');p.locator('#calculate').click()
  self.assertEqual(p.locator('#computeMicroscope .cmp-step').count(),6)
  self.assertNotIn('错误',p.locator('#numeric').inner_text())
  p.locator('[data-patch="switch_top1"]').click()
  self.assertEqual(p.locator('#computeMicroscope .cmp-step').count(),0)
  self.assertIn('结构合同',p.locator('#numeric').inner_text())
  self.assertEqual(errors,[]);p.close()
if __name__=='__main__':unittest.main()
