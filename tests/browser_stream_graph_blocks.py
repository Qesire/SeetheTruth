"""Desktop Chromium proof: the real Scratch canvas can import, validate and execute typed state bricks."""
import json,sys,threading,unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
OUT=Path("stream_graph_screenshots")

class StreamGraphBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  OUT.mkdir(parents=True,exist_ok=True)
  cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
  cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
  cls.thread.start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
  cls.url="http://127.0.0.1:"+str(cls.server.server_port)+"/learn.html"
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
 def setUp(self):
  self.page=self.browser.new_page(viewport={"width":1600,"height":920},device_scale_factor=1,accept_downloads=True)
  self.errors=[]
  self.page.on("pageerror",lambda e:self.errors.append(str(e)))
  self.assertEqual(self.page.goto(self.url).status,200)
  self.page.wait_for_function("window.SeetheTruthLearning && window.SeetheTruthStreamGraph")
 def tearDown(self):
  self.assertEqual(self.errors,[])
  self.page.close()
 def load(self,graph):
  data=json.dumps(graph,ensure_ascii=False).encode("utf-8")
  self.page.locator("#load").set_input_files({"name":"stream-graph.json","mimeType":"application/json","buffer":data})
  self.assertIn("已导入学习工程",self.page.locator("#status").inner_text())
 def test_real_canvas_typed_execution_and_serialization(self):
  p=self.page
  kinds=["stream_state_init","stream_state_append","stream_state_read","stream_state_fork"]
  for kind in kinds:self.assertEqual(p.locator('[data-add="'+kind+'"]').count(),1)
  graph=json.loads((Path(__file__).resolve().parents[1]/"tests/stream_graph_demo.json").read_text(encoding="utf-8"))
  p.locator("#stateGraphExample").click()
  self.assertEqual(p.locator("#blocks .block").count(),10)
  self.assertIn("状态积木示例",p.locator("#workspaceTitle").inner_text())
  self.load(graph)
  self.assertEqual(p.locator("#blocks .block").count(),10)
  for kind in kinds:self.assertTrue(p.locator('[data-node][data-readiness="ready"]').count()>=1)
  p.locator("#studyMode").select_option("concept")
  p.locator("#run").click()
  self.assertIn("已推演",p.locator("#status").inner_text())
  p.locator("#studyMode").select_option("numeric")
  p.locator("#run").click()
  self.assertIn("已运行 10",p.locator("#status").inner_text())
  p.locator('[data-node="read_a"] .block-head').click()
  self.assertEqual(p.locator("[data-node='read_a']").get_attribute("data-readiness"),"ready")
  r=p.evaluate("""(serialized) => {
   const graph=window.SeetheTruthLearning.validateGraph(JSON.parse(serialized));
   return window.SeetheTruthLearning.evaluate(graph).steps.filter(x=>['init','append_a','append_b','read_a','read_b'].includes(x.id)).map(x=>({id:x.id,shape:x.shape,values:x.values,logicalState:x.logicalState}));
  }""",json.dumps(graph,ensure_ascii=False))
  results={x["id"]:x for x in r}
  self.assertEqual(results["init"]["logicalState"]["nextPosition"],12)
  self.assertEqual(results["append_a"]["logicalState"]["nextPosition"],13)
  self.assertEqual(results["append_b"]["logicalState"]["nextPosition"],13)
  self.assertEqual(results["read_a"]["shape"],[2,3])
  self.assertEqual(results["read_a"]["values"][0],results["read_b"]["values"][0])
  self.assertNotEqual(results["read_a"]["values"][1],results["read_b"]["values"][1])
  p.screenshot(path=str(OUT/"graph-state-fork-full-canvas.png"),full_page=True,animations="disabled")
 def test_state_port_type_error_is_visible(self):
  p=self.page
  graph=json.loads((Path(__file__).resolve().parents[1]/"tests/stream_graph_demo.json").read_text(encoding="utf-8"))
  for edge in graph["edges"]:
   if edge["to"]=="append_a" and edge["port"]=="state":edge["from"]="chunk_a"
  self.load(graph)
  self.assertEqual(p.locator('[data-node="append_a"]').get_attribute("data-readiness"),"shape-error")
  p.locator("#studyMode").select_option("numeric")
  p.locator("#run").click()
  self.assertIn("expected stream-state",p.locator("#status").inner_text())
  self.assertIn("无法运行",p.locator("#status").inner_text())
  p.screenshot(path=str(OUT/"graph-state-wrong-port.png"),full_page=True,animations="disabled")

if __name__=="__main__":unittest.main()
