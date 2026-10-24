"""Browser acceptance for source, model and telemetry evidence-only execution boundaries."""
import sys,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
class EvidenceOnlyBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
  threading.Thread(target=cls.server.serve_forever,daemon=True).start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=['--no-sandbox'])
  cls.base=f'http://127.0.0.1:{cls.server.server_port}'
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
 def test_explicit_read_only_evidence_routes(self):
  for route,domain in [('index.html','source_atlas'),('real_model_lab.html','real_model'),('telemetry.html','telemetry')]:
   with self.subTest(route=route):
    page=self.browser.new_page(viewport={'width':1280,'height':800})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    try:
     self.assertEqual(page.goto(self.base+'/'+route).status,200)
     page.wait_for_function('!!window.SeetheTruthUnifiedCore && !!window.SeetheTruthSpecializedLesson')
     self.assertEqual(page.evaluate('window.SeetheTruthSpecializedLesson.id'),domain)
     self.assertEqual(page.evaluate('window.SeetheTruthSpecializedLesson.evidence'),'NOT_EXECUTED')
     if domain=='source_atlas':
      out=page.evaluate("""() => window.SeetheTruthUnifiedCore.inspectEvidence('source_atlas','project',{
       graph:{meta:{analysis:'static-python-ast'},nodes:[
        {id:'a',parent:null,label:'decoder'},{id:'b',parent:'a',callee:'linear',label:'linear'}],
        edges:[{from:'a',to:'b',kind:'hierarchy'}]},scope:'a',level:'matrix'})""")
      self.assertEqual(out['data']['projection']['matches'],['b'])
      self.assertEqual(out['evidenceTier'],'STATIC_SOURCE_ONLY')
      self.assertFalse(out['executed'])
     elif domain=='telemetry':
      page.locator('#demo').click()
      result=page.evaluate('window.SeetheTruthLastTelemetryEvidence')
      self.assertEqual(result['evidenceTier'],'SYNTHETIC_TELEMETRY_DEMO')
      self.assertFalse(result['independentlyReplayed'])
      self.assertEqual(result['data']['counts']['snapshots'],1)
      self.assertFalse(page.locator('#body').is_hidden() is False and page.locator('#empty').is_hidden() is False)
     else:
      result=page.evaluate("""() => {
       try{window.SeetheTruthUnifiedCore.inspectEvidence('real_model','validate_import',{
        trace:{schema:'seethetruth-real-model-trace-v1'}});return false;}
       catch(e){return /provenance/.test(e.message);}
      }""")
      self.assertTrue(result)
     self.assertEqual(errors,[])
    finally:
     page.close()
if __name__=='__main__':unittest.main()
