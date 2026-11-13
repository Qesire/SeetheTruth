"""Unified learning kernel: Chromium evidence from real PC and Android-width pages."""
import sys, threading, unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

class UnifiedLearningBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
  threading.Thread(target=cls.server.serve_forever,daemon=True).start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
  cls.base=f"http://127.0.0.1:{cls.server.server_port}"
 @classmethod
 def tearDownClass(cls):
  cls.browser.close()
  cls.pw.stop()
  cls.server.shutdown()
  cls.server.server_close()

 def test_studio_and_shared_hub_same_contract(self):
  page=self.browser.new_page(viewport={"width":1366,"height":900})
  errors=[]
  page.on("pageerror",lambda e:errors.append(str(e)))
  try:
   self.assertEqual(page.goto(self.base+"/learn.html").status,200)
   page.wait_for_function("!!window.SeetheTruthUnifiedWorkbench")
   counts=page.evaluate("window.SeetheTruthUnifiedWorkbench.counts")
   self.assertGreaterEqual(counts["operators"],146)
   # Match the real source-declared Studio presets, installed pack recipes
   # and the intentionally hidden standalone stream-state shortcut.
   registered=page.evaluate("""() => {
     const visible=[...document.querySelectorAll('[data-preset]')].map(e=>e.dataset.preset);
     const packs=window.SeetheTruthBlockExtensions.recipes.map(r=>r.id);
     const all=[...visible,...packs,'stream_state_branches'];
     // Pack recipe buttons may already have been materialized in the
     // live DOM; the union is the canonical identity set, not DOM count.
     return new Set(all).size;
   }""")
   self.assertEqual(counts["graphLessons"],registered)
   self.assertEqual(counts["courses"],11)
   self.assertEqual(counts["labs"],17)
   evidence=page.evaluate("window.SeetheTruthUnifiedWorkbench.inspect().evidence")
   self.assertEqual(evidence,"STATIC_ONLY")
   page.locator('[data-preset="nd_attention_lesson"]').click()
   self.assertEqual(page.evaluate("window.SeetheTruthUnifiedWorkbench.inspect().kind"),"graph")
   self.assertEqual(errors,[])
  finally:
   page.close()
  phone=self.browser.new_page(viewport={"width":390,"height":844},is_mobile=True,has_touch=True)
  errors=[]
  phone.on("pageerror",lambda e:errors.append(str(e)))
  try:
   self.assertEqual(phone.goto(self.base+"/teaching.html?edition=teaching").status,200)
   phone.wait_for_function("!!window.SeetheTruthTeachingHub")
   self.assertEqual(phone.evaluate("window.SeetheTruthTeachingHub.coreSchema"),
    "seethetruth-learning-experience-v1")
   phone.locator("#voteCase").select_option("fallback")
   self.assertTrue(phone.evaluate("window.SeetheTruthLastSharedVote.secondVoteFallback"))
   self.assertEqual(errors,[])
  finally:
   phone.close()

 def test_vlm_course_opens_the_actual_guided_computation_graph(self):
  page=self.browser.new_page(viewport={"width":390,"height":844},is_mobile=True,has_touch=True)
  errors=[]
  page.on("pageerror",lambda e:errors.append(str(e)))
  try:
   self.assertEqual(page.goto(self.base+"/teaching.html?edition=teaching").status,200)
   page.wait_for_function("!!window.SeetheTruthTeachingHub")
   page.evaluate("window.SeetheTruthTeachingHub.select('vlm')")
   href=page.locator("#openLesson").get_attribute("href")
   self.assertIn("mission=extension_vlm_early_fusion",href)
   self.assertIn("edition=teaching",href)
   self.assertEqual(page.goto(self.base+"/"+href).status,200)
   page.wait_for_function("!!window.SeetheTruthUnifiedWorkbench")
   page.wait_for_function("""() =>
     document.querySelector(".stage").getBoundingClientRect().top < window.innerHeight
     && document.querySelector(".stage").getBoundingClientRect().top >= -60""")
   opened=page.evaluate("""() => {
     const s=window.SeetheTruthUnifiedWorkbench.inspect();
     return {title:s.graph.title, kinds:s.graph.nodes.map(n=>n.kind),
       active:document.querySelector('[data-preset="extension_vlm_early_fusion"]').classList.contains('active')};
   }""")
   self.assertIn("VLM",opened["title"])
   self.assertIn("vlm_token_concat",opened["kinds"])
   self.assertIn("vlm_text_tail",opened["kinds"])
   self.assertTrue(opened["active"])
   self.assertEqual(errors,[])
  finally:
   page.close()

 def test_vlm_query_patch_selection_renders_and_runs_in_chromium(self):
  page=self.browser.new_page(viewport={"width":1366,"height":900})
  errors=[]
  page.on("pageerror",lambda e:errors.append(str(e)))
  try:
   self.assertEqual(page.goto(self.base+"/learn.html?mission=extension_vlm_query_patch_selection").status,200)
   page.wait_for_function("!!window.SeetheTruthUnifiedWorkbench")
   self.assertTrue(page.locator('[data-preset="extension_vlm_query_patch_selection"]').get_attribute("class").find("active")>=0)
   source=page.evaluate("""() => {
     const graph=window.SeetheTruthUnifiedWorkbench.inspect().graph;
     return {kinds:graph.nodes.map(n=>n.kind),nodes:graph.nodes.length};
   }""")
   self.assertIn("vlm_patch_grid_lookup",source["kinds"])
   self.assertIn("inference_sparse_attention",source["kinds"])
   self.assertIn("attention",source["kinds"])
   page.locator("#studyMode").select_option("numeric")
   page.locator("#run").click()
   self.assertIn("已运行",page.locator("#status").inner_text())
   values=page.evaluate("""() => {
     const graph=window.SeetheTruthUnifiedWorkbench.inspect().graph;
     const run=window.SeetheTruthLearning.evaluate(graph);
     const entries=Object.fromEntries(run.steps.map(s=>[s.id,s.values]));
     return {indices:entries.top_patches,coordinates:entries.patch_coordinates,
       delta:entries.readout_delta,loss:entries.readout_error[0][0]};
   }""")
   self.assertEqual(len(values["indices"]),2)
   self.assertEqual(len(values["coordinates"]),4)
   self.assertGreaterEqual(values["loss"],0)
   for index,coordinate in zip(sum(values["indices"],[]),values["coordinates"]):
    self.assertEqual(coordinate,[index//2,index%2])
   self.assertEqual(errors,[])
  finally:
   page.close()

 def test_vlm_first_visit_course_deep_link_and_filters(self):
  page=self.browser.new_page(viewport={"width":390,"height":844},is_mobile=True,has_touch=True)
  errors=[]
  page.on("pageerror",lambda e:errors.append(str(e)))
  try:
   self.assertEqual(page.goto(self.base+"/teaching.html?course=vlm&edition=teaching").status,200)
   page.wait_for_function("!!window.SeetheTruthTeachingHub")
   self.assertIn("VLM",page.locator("#detailTitle").inner_text())
   self.assertIn("mission=extension_vlm_early_fusion",page.locator("#openLesson").get_attribute("href"))
   page.locator("#fundamentals").click()
   self.assertEqual(page.locator('[data-lesson="vlm"]').count(),1)
   self.assertEqual(page.locator('[data-lesson="cache"]').count(),0)
   page.locator("#inference").click()
   self.assertEqual(page.locator('[data-lesson="vlm"]').count(),0)
   self.assertEqual(page.locator('[data-lesson="cache"]').count(),1)
   self.assertFalse(page.evaluate('document.documentElement.scrollWidth > window.innerWidth+2'))
   self.assertEqual(errors,[])
  finally:
   page.close()
  page=self.browser.new_page(viewport={"width":1366,"height":900})
  try:
   self.assertEqual(page.goto(self.base+"/teaching.html?course=not-a-lesson").status,200)
   page.wait_for_function("!!window.SeetheTruthTeachingHub")
   self.assertEqual(
    page.locator("#detailTitle").inner_text(),
    page.evaluate("window.SeetheTruthCurriculum.lessons[0].title"),
    "unknown course link must resolve to the registered first lesson, regardless of title copy")
  finally:
   page.close()

 def test_vlm_guided_bricks_have_no_visible_overlap(self):
  page=self.browser.new_page(viewport={"width":1366,"height":900})
  errors=[]
  page.on("pageerror",lambda e:errors.append(str(e)))
  try:
   for mission in ["extension_vlm_early_fusion","extension_vlm_compressed_prefix",
                   "extension_vlm_cross_attention","extension_vlm_query_patch_selection"]:
    self.assertEqual(page.goto(self.base+"/learn.html?mission="+mission).status,200)
    page.wait_for_function("!!window.SeetheTruthUnifiedWorkbench")
    bounds=page.evaluate("""() => [...document.querySelectorAll('#blocks [data-node]')].map(el=>{
      const r=el.getBoundingClientRect();
      return {id:el.dataset.node,x:r.left,y:r.top,w:r.width,h:r.height};
    })""")
    self.assertGreater(len(bounds),5)
    for i,a in enumerate(bounds):
     self.assertGreater(a["w"],0)
     self.assertGreater(a["h"],0)
     for b in bounds[i+1:]:
      overlap=min(a["x"]+a["w"],b["x"]+b["w"])-max(a["x"],b["x"])
      height=min(a["y"]+a["h"],b["y"]+b["h"])-max(a["y"],b["y"])
      self.assertFalse(overlap>1 and height>1,
       f"{mission}: visible bricks overlap: {a['id']}/{b['id']} by {overlap:.1f}x{height:.1f}")
   self.assertEqual(errors,[])
  finally:
   page.close()

 def test_restored_engineering_journeys_share_actual_core(self):
  for route,id,global_name,count in [
   ("engineering_journey.html","engineering_journey","SeetheTruthEngineeringJourney",12),
   ("engineering_pipeline.html","engineering_pipeline","SeetheTruthEngineeringPipeline",6)]:
   with self.subTest(route=route):
    page=self.browser.new_page(viewport={"width":1280,"height":900})
    errors=[]
    page.on("pageerror",lambda e:errors.append(str(e)))
    try:
     self.assertEqual(page.goto(self.base+"/"+route).status,200)
     page.wait_for_function("(name) => !!window[name] && !!window.SeetheTruthSpecializedLesson",arg=global_name)
     self.assertEqual(page.evaluate("window.SeetheTruthSpecializedLesson.id"),id)
     self.assertEqual(page.evaluate("(name) => window[name].lessons",arg=global_name),count)
     self.assertEqual(errors,[])
    finally:
     page.close()

 def test_shared_numeric_specialist_adapters(self):
  page=self.browser.new_page(viewport={"width":1280,"height":850})
  errors=[]
  page.on("pageerror",lambda e:errors.append(str(e)))
  try:
   self.assertEqual(page.goto(self.base+"/wan_ffn_lab.html").status,200)
   page.wait_for_function("!!window.SeetheTruthSpecializedLesson")
   result=page.evaluate("""() => {
    const x=window.SeetheTruthSpecializedLesson.execute("trace",{options:{
      seed:17,tokens:3,dim:4,hidden:8}});
    return {steps:x.computationTrace.steps.length,
     tier:x.computationTrace.evidenceTier,
     parity:JSON.stringify(x.value)===JSON.stringify(
      window.SeetheTruthWanFFNTrace.trace({seed:17,tokens:3,dim:4,hidden:8}))};
   }""")
   self.assertEqual(result["steps"],9)
   self.assertEqual(page.evaluate("window.SeetheTruthWanFFNLearning.stages"),9)
   self.assertEqual(result["tier"],"PINNED_WAN_SOURCE_SYNTHETIC_FLOAT64")
   self.assertTrue(result["parity"])
   self.assertEqual(errors,[])
  finally:
   page.close()
  page=self.browser.new_page(viewport={"width":390,"height":844},is_mobile=True,has_touch=True)
  errors=[]
  page.on("pageerror",lambda e:errors.append(str(e)))
  try:
   self.assertEqual(page.goto(self.base+"/engineering_pipeline.html").status,200)
   page.wait_for_function("!!window.SeetheTruthSpecializedLesson")
   x=page.evaluate("""() => {
    const r=window.SeetheTruthSpecializedLesson.execute("train",{config:{seed:7},steps:2});
    return {steps:r.stateTrace.steps.length,tier:r.stateTrace.evidenceTier};
   }""")
   self.assertEqual(x["steps"],3)
   self.assertEqual(x["tier"],"EXECUTABLE_BIGRAM_OPTIMIZER_STATE_SIMULATION")
   self.assertEqual(errors,[])
  finally:
   page.close()

 def test_independent_lab_adapter_preserves_domain_truth(self):
  for route,id,domain in [
   ("deepinfer_worker_lab.html","deepinfer_worker","event_dag"),
   ("deepinfer_kv_lifecycle_lab.html","deepinfer_kv","kv_state_machine"),
   ("wan_ffn_lab.html","wan_ffn","source_operator"),
   ("deepinfer_prefill_lab.html","deepinfer_prefill","scheduler_decision")]:
   with self.subTest(route=route):
    page=self.browser.new_page(viewport={"width":1280,"height":800})
    errors=[]
    page.on("pageerror",lambda e:errors.append(str(e)))
    try:
     self.assertEqual(page.goto(self.base+"/"+route).status,200)
     page.wait_for_function("!!window.SeetheTruthSpecializedLesson")
     contract=page.evaluate("""() => {
      const l=window.SeetheTruthSpecializedLesson;
      return {id:l.id,runtime:l.runtime,evidence:l.evidence,engine:l.engine};
     }""")
     self.assertEqual(contract["id"],id)
     self.assertEqual(contract["runtime"],domain)
     self.assertEqual(contract["evidence"],"NOT_EXECUTED")
     self.assertTrue(contract["engine"].endswith(".js"))
     if id in ("deepinfer_worker","deepinfer_kv","deepinfer_scheduler","deepinfer_prefill"):
      # Change the real lab controller's scenario; the inspector must replay
      # that accepted input rather than displaying an unrelated fixture.
      if id=="deepinfer_worker":
       page.locator("#forwardTime").fill("11")
      elif id=="deepinfer_kv":
       page.locator("#scenario").select_option("preempt")
      elif id=="deepinfer_scheduler":
       page.locator("#scenario").select_option("ample")
      else:
       page.locator("#scenario").select_option("grow")
      page.locator("#runUnifiedState").click()
      self.assertGreater(page.locator("#unifiedStateSteps .sm-step").count(),1)
      self.assertIn("当前实验结果一致",page.locator("#unifiedStateStatus").inner_text())
      self.assertTrue(page.evaluate("""() => {
       const x=window.SeetheTruthCurrentDomainRun;
       const y=window.SeetheTruthUnifiedStateLab.accepted;
       return x && y && x.id===y.id &&
          JSON.stringify(x.input)===JSON.stringify(y.input);
      }"""))
      if id=="deepinfer_kv":
       self.assertEqual(page.evaluate("window.SeetheTruthUnifiedStateLab.frame.steps.length"),
        page.evaluate("window.SeetheTruthCurrentDomainRun.input.operations.length+1"))
      if id=="deepinfer_scheduler":
       self.assertEqual(page.evaluate("window.SeetheTruthUnifiedStateLab.frame.steps.length"),
        page.evaluate("window.SeetheTruthCurrentDomainRun.observed.decisions.length+1"))
      if id=="deepinfer_prefill":
       self.assertEqual(page.evaluate("window.SeetheTruthUnifiedStateLab.frame.steps[1].dependencies.length"),0)
     self.assertEqual(errors,[])
    finally:
     page.close()

if __name__=="__main__":
 unittest.main()
