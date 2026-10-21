"""Acceptance for universal microscope in freely assembled graph + Wan source page."""
import threading,unittest,sys
from pathlib import Path
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
from playwright.sync_api import sync_playwright

OUT=Path('compute_microscope_screenshots')
class UniversalMicroscopeBrowser(unittest.TestCase):
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

 def test_editor_numeric_and_static_with_two_architectures(self):
  p=self.browser.new_page(viewport={'width':1550,'height':970})
  failures=[];p.on('pageerror',lambda e:failures.append(str(e)))
  self.assertEqual(p.goto(self.root+'/learn.html').status,200)
  p.wait_for_function('window.SeetheTruthMicroscopeWorkbench !== undefined')
  p.locator('#studyMode').select_option('numeric');p.locator('#run').click()
  p.wait_for_function('window.SeetheTruthMicroscopeWorkbench.stageCount >= 4')
  self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.evidence'),'EXECUTABLE_DETERMINISTIC_TEACHING_TENSORS')
  p.locator('#computeMicroscope .cmp-step').nth(1).click()
  self.assertIn('Linear',p.locator('#computeMicroscope .cmp-meta').inner_text())
  self.assertGreater(p.locator('#computeMicroscope .cmp-terms tr').count(),2)
  self.assertIn('误差',p.locator('#computeMicroscope .cmp-breakdown').inner_text())
  p.locator('#computeMicroscope .cmp-cell').nth(5).click()
  self.assertIn('[1,1]',p.locator('#computeMicroscope .cmp-breakdown').inner_text())
  p.screenshot(path=str(OUT/'editor-linear-dot-numeric.png'),full_page=False,animations='disabled')
  p.locator('[data-preset="swiglu"]').click()
  p.locator('#run').click()
  p.wait_for_function('window.SeetheTruthMicroscopeWorkbench.stageCount >= 6')
  self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.evidence'),'EXECUTABLE_DETERMINISTIC_TEACHING_TENSORS')
  self.assertGreater(p.locator('#computeMicroscope .cmp-step').count(),5)
  p.screenshot(path=str(OUT/'editor-swiglu-graph.png'),full_page=False,animations='disabled')
  p.locator('#studyMode').select_option('concept');p.locator('#run').click()
  self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.evidence'),'STATIC_SHAPES_NO_EXECUTION')
  self.assertEqual(p.locator('#computeMicroscope .cmp-cell').count(),0)
  self.assertIn('静态形状推断',p.locator('#computeMicroscope').inner_text())
  self.assertEqual(failures,[]);p.close()

 def test_pinned_wan_adapter_reuses_component_and_mobile(self):
  p=self.browser.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True,device_scale_factor=2)
  failures=[];p.on('pageerror',lambda e:failures.append(str(e)))
  self.assertEqual(p.goto(self.root+'/wan_ffn_lab.html?edition=teaching').status,200)
  p.wait_for_function("window.SeetheTruthWanFFNLearning?.stages===9")
  self.assertEqual(p.locator('#wanSharedMicroscope .cmp-step').count(),9)
  self.assertIn('PINNED_WAN_SOURCE_SYNTHETIC_FLOAT64',p.locator('#wanSharedMicroscope').inner_text())
  p.locator('#wanSharedMicroscope .cmp-step').nth(4).click()
  self.assertEqual(p.evaluate('window.SeetheTruthWanFFNLearning.current'),4)
  self.assertGreater(p.locator('#wanSharedMicroscope .cmp-terms tr').count(),2)
  self.assertIn('model.py',p.locator('#wanSharedMicroscope .cmp-link').inner_text())
  self.assertTrue(p.evaluate('document.documentElement.scrollWidth<=innerWidth+2'))
  self.assertEqual(p.locator('a[download],input[type=file],#exportPython').count(),0)
  p.screenshot(path=str(OUT/'wan-shared-inspector-phone.png'),full_page=True,animations='disabled')
  self.assertEqual(failures,[]);p.close()

if __name__=='__main__':unittest.main()
