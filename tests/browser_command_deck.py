"""Command Deck Chromium acceptance and screenshot evidence."""
import sys,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
OUT=Path("command_deck_screenshots")
ROUTES=["index.html","learn.html","teaching.html","android_teaching.html",
 "deepinfer_kv_lifecycle_lab.html","deepinfer_scheduler_lab.html",
 "deepinfer_worker_lab.html"]
class CommandDeckBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  OUT.mkdir(exist_ok=True)
  cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
  cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
 def test_theme_coverage_and_mobile_readability(self):
  for route in ROUTES:
   for width in ([1440,390] if route in ["learn.html","teaching.html",
     "android_teaching.html","deepinfer_kv_lifecycle_lab.html"] else [1440]):
    with self.subTest(route=route,width=width):
     page=self.browser.new_page(viewport={"width":width,"height":900},
      is_mobile=width==390,has_touch=width==390)
     errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
     try:
      r=page.goto(f"http://127.0.0.1:{self.server.server_port}/{route}",
       wait_until="domcontentloaded")
      self.assertEqual(r.status,200)
      page.wait_for_function("""() => [...document.styleSheets].some(
       x=>x.href&&x.href.endsWith('/command_deck.css'))""")
      self.assertEqual(page.locator("body").get_attribute("data-theme"),"command-deck")
      values=page.evaluate("""() => {const s=getComputedStyle(document.body);return {
       bg:s.backgroundImage,cyan:s.getPropertyValue('--cyan').trim(),color:s.color};}""")
      self.assertIn("gradient",values["bg"])
      self.assertEqual(values["cyan"].lower(),"#67e0f3")
      self.assertNotEqual(values["color"],"rgb(0, 0, 0)")
      if route.startswith("deepinfer"):
       self.assertTrue(page.locator("#stateContrast").is_visible())
       self.assertTrue(page.locator("#stateLearningGuide").is_visible())
      if route in ["teaching.html","android_teaching.html"] and width==390:
       self.assertFalse(page.evaluate("document.documentElement.scrollWidth>innerWidth+2"))
      if width==1440 and route in ["learn.html","teaching.html","android_teaching.html"]:
       page.screenshot(path=str(OUT/(route.replace(".html","")+"-command-deck.png")),
        full_page=True,animations="disabled")
      self.assertEqual(errors,[])
     finally:page.close()
if __name__=="__main__":unittest.main()
