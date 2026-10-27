"""Focused real-Chromium state experiment A/B → graph → same coordinate acceptance."""
from pathlib import Path
import threading
import unittest
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

SHOTS=Path("state_ab_screenshots")
CASES=(("stream","b",5,"append_b"),("video","b",22,"join"),
       ("axis","a",4,"add"),("reject","a",3,"init"))

class StateABBridgeChromium(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  SHOTS.mkdir(exist_ok=True)
  cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
  cls.worker=threading.Thread(target=cls.server.serve_forever,daemon=True)
  cls.worker.start()
  cls.base="http://127.0.0.1:"+str(cls.server.server_port)
  cls.playwright=sync_playwright().start()
  cls.browser=cls.playwright.chromium.launch(headless=True,args=["--no-sandbox"])
 @classmethod
 def tearDownClass(cls):
  cls.browser.close()
  cls.playwright.stop()
  cls.server.shutdown()
  cls.server.server_close()

 def test_coordinate_navigation_and_replay_desktop_and_mobile(self):
  for viewport,label in [({"width":1600,"height":1000},"desktop"),({"width":390,"height":844},"phone")]:
   page=self.browser.new_page(viewport=viewport,device_scale_factor=1)
   errors=[]
   page.on("pageerror",lambda e:errors.append(str(e)))
   try:
    for name,branch,index,node_id in CASES:
     with self.subTest(device=label,scenario=name):
      page.goto(f"{self.base}/semantic_state.html?state_case={name}",wait_until="load")
      button=page.locator(f"#{branch}CoordinateGrid [data-coordinate='{branch}:{index}']")
      button.wait_for(timeout=15000)
      button.click()
      source_focus=page.evaluate("window.SeetheTruthCoordinateFocusSession.current")
      self.assertEqual(source_focus["schema"],"seethetruth-coordinate-focus-v1")
      self.assertEqual(source_focus["domain"],"semantic_state")
      self.assertEqual(source_focus["branch"],branch)
      self.assertEqual(source_focus["flatIndex"],index)
      self.assertIn("实际来源",page.locator("#coordinateDetail").inner_text())
      page.screenshot(path=str(SHOTS/f"{name}-{label}-lab.png"),animations="disabled",full_page=True)
      destination=page.locator("#coordinateGraphLink").get_attribute("href")
      self.assertIn(f"state_node={node_id}&state_branch={branch}&state_index={index}",destination)
      page.locator("#coordinateGraphLink").click()
      page.wait_for_url("**/learn.html?*",timeout=20000)
      try:
       page.wait_for_function("""id => {
         const p=document.getElementById('stateHandoff');
         return p && !p.innerText.includes('课程关联待检查') &&
           p.querySelector('[aria-pressed=true]') &&
           document.querySelector('#blocks [data-node="'+id+'"].selected');
       }""",arg=node_id,timeout=20000)
      except Exception:
       panel=page.locator("#stateHandoff").inner_text() if page.locator("#stateHandoff").count() else "(missing)"
       selected=page.locator("#blocks .selected").evaluate_all("(xs)=>xs.map(e=>e.dataset.node)")
       raise AssertionError(f"{label} {name} expected {node_id}, panel={panel!r}, selected={selected!r}, js={errors!r}")
      self.assertEqual(page.locator(".inspector #stateHandoff").count(),1)
      unobstructed=page.evaluate("""() => {
        const a=document.querySelector('#stateHandoff').getBoundingClientRect();
        const b=document.querySelector('.stage').getBoundingClientRect();
        return a.right<=b.left+1 || a.left>=b.right-1 || a.bottom<=b.top+1 || a.top>=b.bottom-1;
      }""")
      self.assertTrue(unobstructed,"state evidence card must not cover the graph canvas")
      if label=="phone":
       self.assertTrue(page.locator("#stateHandoffJump").is_visible())
       chips=page.locator("#stateHandoffTrail .state-handoff-path-chip")
       self.assertGreaterEqual(chips.count(),2)
       upstream=chips.first.inner_text().replace("→ ","").strip()
       chips.first.click()
       self.assertEqual(page.locator(f"#blocks [data-node='{upstream}'].selected").count(),1)
       page.screenshot(path=str(SHOTS/f"{name}-phone-upstream.png"),animations="disabled",full_page=True)
       chips.last.click()
       self.assertEqual(page.locator(f"#blocks [data-node='{node_id}'].selected").count(),1)
      panel=page.locator("#stateHandoff")
      self.assertIn("实验坐标",panel.inner_text())
      self.assertIn("独立示例",panel.inner_text())
      self.assertGreater(page.locator("#blocks .state-evidence-path").count(),0)
      handoff_focus=page.evaluate("window.SeetheTruthCoordinateFocusSession.current")
      self.assertEqual(handoff_focus["domain"],"semantic_state")
      page.locator('#sharedCoordinateDrilldown').click()
      self.assertIn('独立数值执行',page.locator('#sharedCoordinateDrilldownResult').inner_text())
      self.assertEqual(page.locator('#coordinateJourneyCount').inner_text(),'1 / 1')
      self.assertIn('机制与执行实例',page.locator('#coordinateJourneyTitle').inner_text())
      self.assertIn("实验原生数值",page.locator("#sharedCoordinateFocus").inner_text())
      self.assertEqual(handoff_focus["executionId"],source_focus["executionId"])
      self.assertEqual(handoff_focus["flatIndex"],source_focus["flatIndex"])
      page.screenshot(path=str(SHOTS/f"{name}-{label}.png"),animations="disabled",full_page=True)
      panel.locator("a").click()
      page.wait_for_url("**/semantic_state.html?*",timeout=20000)
      selected_coordinate=page.locator(f"#{branch}CoordinateGrid [data-coordinate='{branch}:{index}']")
      selected_coordinate.wait_for(timeout=15000)
      self.assertEqual(selected_coordinate.get_attribute("aria-pressed"),"true")
      self.assertEqual(errors,[])
   finally:
    page.close()

if __name__=="__main__":
 unittest.main()
