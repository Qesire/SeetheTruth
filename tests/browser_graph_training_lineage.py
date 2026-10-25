"""Bidirectional graph ↔ training state links in the real Learning Studio."""
import sys,threading,unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

class GraphTrainingLineageBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
  threading.Thread(target=cls.server.serve_forever,daemon=True).start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
  cls.base=f"http://127.0.0.1:{cls.server.server_port}/learn.html"
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
 def test_real_graph_training_state_and_bidirectional_canvas_selection(self):
  for width in [1366,390]:
   with self.subTest(viewport=width):
    page=self.browser.new_page(viewport={"width":width,"height":900},
      is_mobile=width==390,has_touch=width==390)
    errors=[];page.on("pageerror",lambda e:errors.append(str(e)))
    try:
     self.assertEqual(page.goto(self.base).status,200)
     page.wait_for_function("!!window.SeetheTruthUnifiedWorkbench && !!window.SeetheTruthObservationPanel")
     page.locator('[data-preset="trainable_core"]').click()
     page.locator('#studyMode').select_option('numeric')
     page.locator('#trainCount').select_option('2')
     page.locator('#trainStep').click()
     self.assertNotIn('训练未提交',page.locator('#status').inner_text())
     page.locator('#graphTrainingObservationMode').select_option('training')
     page.locator('#unifiedObservationRefresh').click()
     self.assertIn('已核验 training',page.locator('#unifiedObservationStatus').inner_text())
     self.assertEqual(page.evaluate('window.SeetheTruthObservationPanel.current.stateTrace.evidenceTier'),
                      'EXECUTABLE_BOUNDED_GRAPH_VJP_OPTIMIZER_STATE')
     result=page.evaluate("""()=>{
      const record=window.SeetheTruthObservationPanel.current;
      return {kind:record.kind,id:record.id,op:record.operation,
       links:record.stateTrace.graphLinks,
       nodes:record.value.timeline.nodes.map(n=>n.id),
       replay:window.SeetheTruthUnifiedCore.replay(record).reproduced};
     }""")
     self.assertEqual(result['op'],'graph_step')
     self.assertEqual(result['id'],'training')
     self.assertTrue(result['replay'])
     self.assertTrue(result['links']['shared_weight']['optimizer'])
     self.assertEqual(set(result['nodes']),set(result['links']))
     page.locator('#unifiedObservationStep').select_option(
      label=page.evaluate("""()=>{
        const select=document.querySelector('#unifiedObservationStep');
        return [...select.options].find(o=>o.textContent.includes('graph:optimizer:'))?.textContent;
      }"""))
     page.locator('#unifiedObservationInspect').click()
     self.assertIn('"phase": "optimizer"',page.locator('#unifiedObservationDetail').inner_text())
     self.assertEqual(page.evaluate('window.SeetheTruthUnifiedWorkbench.selectedNode'),'shared_weight')
     page.locator('#graphTrainingLocateNode').click()
     self.assertIn('graph:optimizer',page.locator('#graphTrainingLocateStatus').inner_text())
     # Original graph timeline and the unified state panel can share the selected node.
     page.locator('#timelineStep').select_option('0')
     page.locator('#unifiedObservationRefresh').click()
     self.assertEqual(page.evaluate('window.SeetheTruthObservationPanel.current.value.session.transaction.records[0].optimizerStep'),1)
     self.assertEqual(errors,[])
    finally:page.close()
if __name__=="__main__":unittest.main()
