"""Guided state-event learning with native replay and cross-scenario invalidation."""
import threading,unittest,sys
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

class GuidedStates(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
  cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
 def test_state_learning(self):
  screenshots=Path("state_learning_guide_screenshots");screenshots.mkdir(exist_ok=True)
  for domain,route,change in [
    ("deepinfer_kv","deepinfer_kv_lifecycle_lab.html","preempt"),
    ("deepinfer_scheduler","deepinfer_scheduler_lab.html","ample"),
    ("deepinfer_worker","deepinfer_worker_lab.html","decode")]:
   for width in (1480,390):
    with self.subTest(domain=domain,width=width):
     page=self.browser.new_page(viewport={"width":width,"height":940},
       is_mobile=width==390,has_touch=width==390)
     errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
     try:
      r=page.goto(f"http://127.0.0.1:{self.server.server_port}/{route}")
      self.assertEqual(r.status,200)
      page.wait_for_function("!!window.SeetheTruthStateLearningGuide")
      self.assertEqual(page.evaluate("SeetheTruthStateLearningGuide.id"),domain)
      self.assertTrue(page.locator("#stateGuideNext").is_disabled())
      page.locator("#stateGuideAction").click()
      page.wait_for_function("SeetheTruthStateLearningGuide.current")
      self.assertTrue(page.evaluate("SeetheTruthStateLearningGuide.verified"))
      frame=page.evaluate("SeetheTruthUnifiedStateLab.frame")
      self.assertGreater(len(frame["steps"]),2)
      self.assertTrue(frame["evidenceTier"].startswith("SOURCE_GROUNDED_"))
      self.assertGreaterEqual(page.evaluate("SeetheTruthStateLearningGuide.count"),2)
      page.locator("#stateGuideNext").click()
      self.assertTrue(page.locator("#stateGuideNext").is_disabled())
      step=page.evaluate("SeetheTruthStateLearningGuide.selectedEvent")
      self.assertIn(step,[s["id"] for s in frame["steps"]])
      page.locator("#stateGuideAction").click()
      self.assertTrue(page.evaluate("SeetheTruthStateLearningGuide.verified"))
      self.assertEqual(page.locator("#unifiedStateSteps .sm-step[aria-current=step]").count(),1)
      if width==1480:page.screenshot(path=str(screenshots/f"{domain}-learning-guide.png"),full_page=True,animations="disabled")
      page.locator("#scenario").select_option(change)
      page.wait_for_function("!SeetheTruthStateLearningGuide.current")
      self.assertTrue(page.locator("#stateGuideNext").is_disabled())
      self.assertEqual(page.evaluate("SeetheTruthStateLearningGuide.step"),0)
      page.locator("#stateGuideAction").click()
      self.assertTrue(page.evaluate("SeetheTruthStateLearningGuide.current"))
      self.assertEqual(errors,[])
      self.assertLessEqual(page.locator("#stateLearningGuide").evaluate(
       "(e)=>e.getBoundingClientRect().width"),width+2)
     finally:page.close()


 def test_native_ab_mechanism_comparison(self):
  screens=Path("state_learning_guide_screenshots");screens.mkdir(exist_ok=True)
  for domain,route,alt in [
    ("deepinfer_kv","deepinfer_kv_lifecycle_lab.html","preempt"),
    ("deepinfer_scheduler","deepinfer_scheduler_lab.html","ample"),
    ("deepinfer_worker","deepinfer_worker_lab.html","nooverlap")]:
   for width in (1480,390):
    with self.subTest(domain=domain,width=width):
     page=self.browser.new_page(viewport={"width":width,"height":900},
       is_mobile=width==390,has_touch=width==390)
     errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
     try:
      r=page.goto(f"http://127.0.0.1:{self.server.server_port}/{route}")
      self.assertEqual(r.status,200)
      page.wait_for_function("!!window.SeetheTruthStateContrast")
      self.assertTrue(page.locator("#stateContrastCompare").is_disabled())
      page.locator("#stateContrastCapture").click()
      self.assertEqual(page.evaluate("SeetheTruthStateContrast.baseline"),domain)
      page.locator("#stateContrastCompare").click()
      self.assertIn("相同",page.locator("#stateContrastStatus").inner_text())
      self.assertFalse(page.evaluate("SeetheTruthStateContrast.compared"))
      page.locator("#scenario").select_option(alt)
      page.locator("#stateContrastCompare").click()
      self.assertTrue(page.evaluate("SeetheTruthStateContrast.compared"))
      self.assertTrue(page.evaluate("SeetheTruthStateContrast.replayVerified"))
      self.assertTrue(page.evaluate("SeetheTruthStateContrast.current"))
      self.assertGreater(page.evaluate("SeetheTruthStateContrast.changedSteps.length"),0)
      self.assertGreater(page.locator("#stateContrastResults .command-contrast-event").count(),0)
      self.assertGreater(page.locator('#stateContrastResults details.command-contrast-diff').count(),0)
      valid=page.locator("#stateContrastResults .command-contrast-event:not([disabled])")
      if valid.count():
       valid.first.click()
       self.assertTrue(page.locator("#unifiedStateSteps .sm-step[aria-current=step]").count())
      self.assertEqual(page.locator("body").get_attribute("data-theme"),"command-deck")
      self.assertGreater(page.evaluate("""() => Number.parseFloat(
       getComputedStyle(document.querySelector('#stateContrast')).borderTopWidth)"""),0)
      if width==1480:page.screenshot(
       path=str(screens/f"{domain}-ab-comparison.png"),full_page=True,animations="disabled")
      # Another native scenario selection invalidates B, but keeps A for a new contrast.
      page.locator("#scenario").select_option("fork" if domain=="deepinfer_kv"
        else "pressure" if domain=="deepinfer_scheduler" else "mixed")
      page.wait_for_function("!SeetheTruthStateContrast.compared")
      self.assertEqual(page.evaluate("SeetheTruthStateContrast.baseline"),domain)
      page.locator("#stateContrastReset").click()
      self.assertTrue(page.locator("#stateContrastCompare").is_disabled())
      self.assertEqual(errors,[])
     finally:page.close()


 def test_controlled_native_intervention(self):
  for domain,route in [
    ("deepinfer_kv","deepinfer_kv_lifecycle_lab.html"),
    ("deepinfer_scheduler","deepinfer_scheduler_lab.html"),
    ("deepinfer_worker","deepinfer_worker_lab.html")]:
   with self.subTest(domain=domain):
    page=self.browser.new_page(viewport={"width":1440,"height":900})
    errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
    try:
     response=page.goto(f"http://127.0.0.1:{self.server.server_port}/{route}")
     self.assertEqual(response.status,200)
     page.wait_for_function("!!window.SeetheTruthStateContrast")
     page.locator("#stateContrastCapture").click()
     original=page.evaluate("SeetheTruthStateContrast.baselineInput")
     self.assertIsNotNone(original)
     page.locator("#stateContrastIntervene").click()
     after=page.evaluate("SeetheTruthCurrentDomainRun.input")
     self.assertNotEqual(original,after)
     self.assertNotIn("干预未生效",page.locator("#stateContrastStatus").inner_text())
     if domain=="deepinfer_kv":
      self.assertEqual(after["config"]["numBlocks"],original["config"]["numBlocks"]+1)
      self.assertEqual(after["operations"],original["operations"])
     elif domain=="deepinfer_scheduler":
      self.assertFalse(original["options"]["stopPreempted"])
      self.assertTrue(after["options"]["stopPreempted"])
     else:
      a=original["options"]["durations"]["forward"]
      self.assertEqual(after["options"]["durations"]["forward"],a+2)
      self.assertEqual(after["steps"],original["steps"])
     page.locator("#stateContrastCompare").click()
     self.assertTrue(page.evaluate("SeetheTruthStateContrast.replayVerified"))
     self.assertGreater(page.evaluate("SeetheTruthStateContrast.changedSteps.length"),0)
     self.assertEqual(errors,[])
    finally:page.close()

if __name__=="__main__":unittest.main()
