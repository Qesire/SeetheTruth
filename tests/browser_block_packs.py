"""Real Chromium acceptance for pack-owned free-form recipe."""
import sys,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
class PackBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
  threading.Thread(target=cls.server.serve_forever,daemon=True).start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=['--no-sandbox'])
  cls.url=f'http://127.0.0.1:{cls.server.server_port}/learn.html'
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
 def test_pack_free_graph(self):
  page=self.browser.new_page(viewport={'width':1440,'height':960})
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  try:
   page.goto(self.url,wait_until='domcontentloaded')
   page.locator('[data-preset="extension_named_axis_composition"]').click()
   self.assertEqual(page.locator('#blocks [data-node]').count(),7)
   self.assertIn('7/7',page.locator('#graphReadiness').inner_text())
   page.locator('#studyMode').select_option('numeric')
   page.locator('#run').click()
   self.assertIn('✓ 已运行',page.locator('#status').inner_text())
   page.locator('#learningPathPanel summary').first.click()
   page.locator('#learningPathNumeric').click()
   self.assertIn('微型数值已验证',page.locator('#learningPathSteps').inner_text())
   page.locator('[data-preset="extension_video_time_composition"]').click()
   self.assertEqual(page.locator('#blocks [data-node]').count(),4)
   page.locator('#run').click()
   page.locator('#blocks [data-node="join"] .block-formula').click()
   self.assertIn('1×3×2×2×2',page.locator('#matrix').inner_text())
   self.assertEqual(page.locator('#matrix [data-nd-slice]').count(),3)
   page.locator('#matrix [data-nd-slice="1"]').select_option('2')
   self.assertEqual(page.locator('#matrix .nd-plane .cell').count(),4)
   self.assertEqual(errors,[])
  finally:page.close()
if __name__=='__main__':unittest.main()
