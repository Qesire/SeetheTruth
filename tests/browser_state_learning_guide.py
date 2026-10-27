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
      page.locator("#stateContrastHypothesis").fill("更改输入后，哪些事件依赖和资源状态会变化？")
      page.locator("#stateContrastCapture").click()
      self.assertTrue(page.locator("#stateContrastHypothesis").is_disabled())
      self.assertTrue(page.locator("#stateContrastCapture").is_disabled())
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
      self.assertEqual(page.locator('#stateContrastResults .command-contrast-metric').count(),3)
      self.assertTrue(page.locator("#stateContrastExplanation").is_visible())
      self.assertIn("实验前预测",page.locator("#stateContrastResults").inner_text())
      self.assertGreater(page.locator("#stateContrastResults .command-contrast-event").count(),0)
      self.assertEqual(page.locator('#stateContrastResults .command-contrast-metric').count(),3)
      self.assertTrue(page.locator("#stateContrastExplanation").is_visible())
      page.locator("#stateContrastExplanation").fill("两个输入的资源状态不同；还需要调整请求负载排除替代解释。")
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
      self.assertTrue(page.locator("#stateContrastHypothesis").is_enabled())
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
     page.locator("#stateContrastHypothesis").fill("增加一个条件后，哪些状态与事件会改变？")
     page.locator("#stateContrastCapture").click()
     original=page.evaluate("SeetheTruthStateContrast.baselineInput")
     self.assertIsNotNone(original)
     self.assertTrue(page.locator("#stateContrastHypothesis").is_disabled())
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
     self.assertIn("实验前预测",page.locator("#stateContrastResults").inner_text())
     self.assertGreater(page.evaluate("SeetheTruthStateContrast.changedSteps.length"),0)
     self.assertEqual(errors,[])
    finally:page.close()


 def test_native_graph_bidirectional_handoff(self):
  """Native A/B -> real graph mission -> replay-checked native selected cursor."""
  for domain,route,course in [
    ("deepinfer_worker","deepinfer_worker_lab.html","attention"),
    ("deepinfer_kv","deepinfer_kv_lifecycle_lab.html","decode"),
    ("deepinfer_scheduler","deepinfer_scheduler_lab.html","decode")]:
   for width in (1440,390):
    with self.subTest(domain=domain,width=width):
     page=self.browser.new_page(viewport={"width":width,"height":900},
       is_mobile=width==390,has_touch=width==390)
     errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
     try:
      root=f"http://127.0.0.1:{self.server.server_port}/"
      self.assertEqual(page.goto(root+route).status,200)
      page.wait_for_function("!!window.SeetheTruthStateContrast && !!window.SeetheTruthLearningHandoff")
      page.locator("#stateContrastHypothesis").fill("控制一个输入，哪些事件变化仍是源引擎能证实的？")
      page.locator("#stateContrastCapture").click()
      page.locator("#stateContrastIntervene").click()
      page.locator("#stateContrastCompare").click()
      self.assertTrue(page.evaluate("SeetheTruthStateContrast.replayVerified"))
      page.locator("#stateGraphBridge").click()
      page.wait_for_url("**/learn.html?nativeGraphBridge=1")
      page.wait_for_function("!!window.SeetheTruthNativeGraphBridge")
      self.assertTrue(page.evaluate("SeetheTruthNativeGraphBridge.replayVerified"))
      self.assertEqual(page.evaluate("SeetheTruthNativeGraphBridge.course"),course)
      self.assertTrue(page.evaluate("SeetheTruthNativeGraphBridge.compared"))
      self.assertTrue(page.evaluate("SeetheTruthNativeGraphBridge.mechanismVerified"))
      self.assertFalse(page.evaluate("SeetheTruthNativeGraphBridge.numericVerified"))
      self.assertEqual(page.locator("#nativeGraphMechanismPorts button").count(),3)
      first_source=page.locator("#nativeGraphMechanismPorts button").first.get_attribute("data-source-node")
      page.locator("#nativeGraphMechanismPorts button").first.click()
      self.assertEqual(page.evaluate("SeetheTruthUnifiedWorkbench.selectedNode"),first_source)
      self.assertTrue(all("端口 "+p in page.locator("#nativeGraphMechanismPorts").inner_text()
        for p in ("Q","K","V")))
      self.assertEqual(len(page.evaluate("SeetheTruthNativeGraphBridge.portSources")),3)
      page.locator("#nativeGraphBridgeProbe").click()
      page.wait_for_function("SeetheTruthNativeGraphBridge.numericVerified")
      self.assertEqual(page.locator("#nativeGraphMechanismNumeric").get_attribute("data-verified"),"true")
      self.assertIn("已执行并重放核验",page.locator("#nativeGraphMechanismNumeric").inner_text())
      self.assertTrue(page.evaluate("SeetheTruthNativeGraphBridge.coordinateVerified"))
      coordinate=page.evaluate("SeetheTruthNativeGraphBridge.coordinate")
      self.assertLess(coordinate["error"],1e-10)
      self.assertAlmostEqual(coordinate["probabilitySum"],1.0,places=9)
      self.assertEqual(page.locator("#nativeGraphCoordinateTerms tbody tr").count(),coordinate["keyCount"])
      self.assertEqual(page.locator("#nativeGraphCoordinateSummary").get_attribute("data-verified"),"true")
      if course=="decode":
       page.locator("#nativeGraphCoordhead").select_option("1")
       self.assertEqual(page.evaluate("SeetheTruthNativeGraphBridge.coordinate.head"),1)
       self.assertEqual(page.evaluate("SeetheTruthNativeGraphBridge.coordinate.maskedCount"),0)
      else:
       page.locator("#nativeGraphCoordqueryToken").select_option("2")
       self.assertEqual(page.evaluate("SeetheTruthNativeGraphBridge.coordinate.queryToken"),2)
      options=page.locator("#nativeGraphCoordchannel option").count()
      page.locator("#nativeGraphCoordchannel").select_option(str(options-1))
      coord2=page.evaluate("SeetheTruthNativeGraphBridge.coordinate")
      self.assertEqual(coord2["channel"],options-1)
      self.assertLess(coord2["error"],1e-10)
      self.assertTrue(page.evaluate("SeetheTruthNativeGraphBridge.scoreVerified"))
      self.assertEqual(page.locator("#nativeGraphScoreFeatures tbody tr").count(),
       page.evaluate("SeetheTruthNativeGraphBridge.score.headWidth"))
      score=page.evaluate("SeetheTruthNativeGraphBridge.score")
      self.assertLess(score["scoreError"],1e-10)
      self.assertLess(score["softmaxError"],1e-10)
      self.assertEqual(score["rotary"],course=="decode")
      page.locator('#nativeGraphCoordinateTerms tr[data-key-token="1"]').click()
      self.assertEqual(page.locator("#nativeGraphScoreKey").input_value(),"1")
      self.assertEqual(page.evaluate("SeetheTruthNativeGraphBridge.score.keyToken"),1)
      page.locator("#nativeGraphScoreKey").select_option("2")
      score=page.evaluate("SeetheTruthNativeGraphBridge.score")
      self.assertEqual(score["keyToken"],2)
      self.assertLess(score["scoreError"],1e-10)
      self.assertIn("Score",page.locator("#nativeGraphScoreSummary").inner_text())

      page.locator("#studyMode").select_option("concept")
      page.wait_for_function("!SeetheTruthNativeGraphBridge.coordinateVerified")
      self.assertFalse(page.evaluate("SeetheTruthNativeGraphBridge.scoreVerified"))
      self.assertTrue(page.locator("#nativeGraphCoordinateLab").is_hidden())
      page.locator("#studyMode").select_option("numeric")
      page.locator("#nativeGraphBridgeProbe").click()
      page.wait_for_function("SeetheTruthNativeGraphBridge.coordinateVerified")

      # The original A/B remains available when a new graph removes the relevant operation.
      page.locator('.mission[data-preset="ffn"]').click()
      page.wait_for_function("!SeetheTruthNativeGraphBridge.mechanismVerified")
      self.assertFalse(page.evaluate("SeetheTruthNativeGraphBridge.numericVerified"))
      self.assertFalse(page.evaluate("SeetheTruthNativeGraphBridge.coordinateVerified"))
      self.assertFalse(page.evaluate("SeetheTruthNativeGraphBridge.scoreVerified"))
      self.assertTrue(page.locator("#nativeGraphCoordinateLab").is_hidden())
      page.locator('.mission[data-preset="'+course+'"]').click()
      page.wait_for_function("SeetheTruthNativeGraphBridge.mechanismVerified")
      self.assertFalse(page.evaluate("SeetheTruthNativeGraphBridge.numericVerified"))
      selected=page.evaluate("SeetheTruthNativeGraphBridge.selectedStepId")
      self.assertEqual(page.locator(".mission.active").get_attribute("data-preset"),course)
      self.assertIn(selected,page.locator("#nativeGraphBridgeEvidence").inner_text())
      self.assertIn("不同的数学语义",page.locator("#nativeGraphBridgePanel").inner_text())
      self.assertIn("nativeGraphReturn=1",page.locator("#nativeGraphBridgeBack").get_attribute("href"))
      page.locator("#nativeGraphBridgeBack").click()
      page.wait_for_url("**/"+route+"?nativeGraphReturn=1")
      page.wait_for_function("!!window.SeetheTruthNativeGraphReturn")
      self.assertEqual(page.evaluate("SeetheTruthNativeGraphReturn.selectedStepId"),selected)
      self.assertTrue(page.evaluate("SeetheTruthNativeGraphReturn.hasBaseline"))
      self.assertEqual(page.locator("#unifiedStateSteps .sm-step[aria-current=step]").count(),1)
      title=page.evaluate("""() => SeetheTruthUnifiedStateLab.frame.steps.find(
       step=>step.id===SeetheTruthNativeGraphReturn.selectedStepId).title""")
      self.assertIn(title,page.locator("#unifiedStateSteps .sm-step[aria-current=step]").inner_text())
      page.locator("#stateGraphReturnToggle").click()
      self.assertIn("A",page.locator("#stateGraphReturnStatus").inner_text())
      page.locator("#stateGraphReturnToggle").click()
      self.assertIn(selected,page.locator("#stateGraphReturnStatus").inner_text())
      self.assertEqual(errors,[])
     finally:page.close()

if __name__=="__main__":unittest.main()
