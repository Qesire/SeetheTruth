"""Cross-platform Chromium: specialist pages execute common source-driven core, not copied UI arithmetic."""
import sys,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
class UnifiedSpecialists(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
  threading.Thread(target=cls.server.serve_forever,daemon=True).start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
  cls.base=f"http://127.0.0.1:{cls.server.server_port}"
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
 def test_core_is_actual_executive_on_desktop_and_mobile(self):
  for width in (1366,390):
   for route,lab in [('paper_lab.html','paper'),('micro_training.html','micro_training'),
                     ('semantic_state.html','semantic_state'),('capabilities.html','capabilities')]:
    with self.subTest(width=width,lab=lab):
     page=self.browser.new_page(viewport={'width':width,'height':840},
                                is_mobile=(width==390),has_touch=(width==390))
     errors=[]
     page.on('pageerror',lambda e:errors.append(str(e)))
     try:
      self.assertEqual(page.goto(self.base+'/'+route).status,200)
      page.wait_for_function('!!window.SeetheTruthSpecializedLesson')
      self.assertEqual(page.evaluate('window.SeetheTruthSpecializedLesson.id'),lab)
      if lab=='paper':
       page.locator('[data-patch="lora_linear"]').click()
       self.assertIn('output',page.locator('#numeric').inner_text())
       direct=page.evaluate("""() => window.SeetheTruthSpecializedLesson.execute('lora',{
        x:[[1,2]],w:[[1,0],[0,1]],a:[[1],[2]],b:[[3,4]],alpha:2}).value.output""")
       self.assertEqual(direct,[[31,42]])
       page.locator('[data-patch="online_attention"]').click()
       self.assertIn('output',page.locator('#numeric').inner_text())
      elif lab=='micro_training':
       for scenario in ('shared','moe','ppo'):
        page.locator('[data-case="'+scenario+'"]').click()
        actual=page.evaluate('window.SeetheTruthLastMicroRun')
        self.assertEqual(actual['scenario'],scenario)
        self.assertEqual(actual['record']['value']['scenario'],scenario)
        self.assertNotIn('计算失败',page.locator('#details').inner_text())
      elif lab=='semantic_state':
       for scenario in ('stream','video','axis','reject'):
        page.locator('[data-scenario="'+scenario+'"]').click()
        self.assertIn('数值合同通过',page.locator('#status').inner_text())
        self.assertTrue(page.evaluate('(s)=>window.SeetheTruthSpecializedLesson.execute("scenario",{name:s}).value.passed',scenario))
      else:
       page.wait_for_function("document.querySelectorAll('#rows tr').length===146")
       self.assertEqual(page.evaluate("window.SeetheTruthSpecializedLesson.execute('audit',{}).value.rows.length"),146)
      self.assertEqual(errors,[])
     finally:
      page.close()
 def test_graph_paper_rewrite_mutates_only_selected_graph(self):
  page=self.browser.new_page(viewport={'width':1366,'height':850})
  errors=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  try:
   self.assertEqual(page.goto(self.base+'/learn.html').status,200)
   page.locator('[data-preset="autodiff_core"]').click()
   page.locator('[data-node="project"] .block-head').click()
   page.locator('#applyLoRA').click()
   value=page.evaluate("""() => {
    const graph=window.SeetheTruthUnifiedWorkbench.inspect().graph;
    return {patches:graph.paperPatches?.length||0,
      output:graph.nodes.find(x=>x.id==='project')?.kind};
   }""")
   self.assertEqual(value,{'patches':1,'output':'add'})
   self.assertEqual(errors,[])
  finally:
   page.close()
if __name__=='__main__':
 unittest.main()
