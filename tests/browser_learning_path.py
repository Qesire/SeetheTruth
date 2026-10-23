"""Chromium test for an automatically derived, falsifiable graph learning route."""
import sys,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
class TestRoute(unittest.TestCase):
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
 def test_learning_route(self):
  page=self.browser.new_page(viewport={'width':1440,'height':900})
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  try:
   page.goto(self.url,wait_until='domcontentloaded')
   page.locator('#learningPathPanel summary').first.click()
   page.wait_for_function("document.querySelectorAll('[data-route-node]').length>=2")
   route=page.locator('#learningPathSteps')
   self.assertIn('静态形状',route.inner_text())
   page.locator('#learningPathNumeric').click()
   self.assertIn('微型数值已验证',route.inner_text())
   page.locator('[data-route-node]').last.click()
   self.assertTrue(page.locator('#inspectorTitle').inner_text())
   page.locator('#challenge').click()
   self.assertIn('未完成',route.inner_text())
   self.assertNotIn('微型数值已验证',route.inner_text())
   self.assertEqual(errors,[])
  finally:page.close()
if __name__=='__main__':unittest.main()
