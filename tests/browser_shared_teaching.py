"""Real Chromium acceptance of identical PC/Android study data and artifact-free phone UX."""
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
from playwright.sync_api import sync_playwright

OUT=Path("shared_teaching_screenshots")
class SharedTeachingBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True,exist_ok=True)
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
        cls.base=f"http://127.0.0.1:{cls.server.server_port}"
    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()

    def test_same_pc_and_mobile_curriculum_explanations(self):
        normal=self.browser.new_page(viewport={"width":1440,"height":920})
        phone=self.browser.new_page(viewport={"width":390,"height":844},is_mobile=True,has_touch=True,
                                    device_scale_factor=2)
        errors=[]
        normal.on("pageerror",lambda e:errors.append("pc:"+str(e)))
        phone.on("pageerror",lambda e:errors.append("phone:"+str(e)))
        self.assertEqual(normal.goto(self.base+"/teaching.html").status,200)
        self.assertEqual(phone.goto(self.base+"/teaching.html?edition=teaching").status,200)
        for p in (normal,phone):
            p.wait_for_function("window.SeetheTruthTeachingHub && window.SeetheTruthTeachingHub.lessons===13")
            self.assertEqual(p.locator("#cards button.card").count(),13)
            p.locator("#courseGrain").select_option("system")
            self.assertEqual(p.locator("#cards button.card").count(),4)
            p.locator("#courseGrain").select_option("operation")
            self.assertEqual(p.locator("#cards button.card").count(),1)
            self.assertIn("G3",p.locator("#cards button.card .course-grain-meta").inner_text())
            p.locator("#courseGrain").select_option("all")
            self.assertEqual(p.locator("#cards button.card").count(),13)
            self.assertEqual(p.locator("#architectureSteps button").count(),5)
            self.assertEqual(p.locator("#granularitySteps button").count(),5)
            self.assertEqual(p.evaluate("window.SeetheTruthGranularityExplorer.current().sourceId"),"course:research_video_sparse")
            p.locator('#granularitySteps [data-grain="atomic"]').click()
            self.assertEqual(p.evaluate("window.SeetheTruthGranularityExplorer.current().sourceId"),"lab:wan_ffn")
            self.assertIn("NOT_CLAIMED",p.locator("#granularityProof").inner_text())
            p.locator('#architectureTracks [data-track="agent"]').click()
            self.assertEqual(p.evaluate("window.SeetheTruthGranularityExplorer.current().sourceId"),"research:agent")
            self.assertIn("SCRIPTED",p.locator("#granularityLabel").inner_text())
            p.locator('#architectureTracks [data-track="posttrain"]').click()
            p.locator('#granularitySteps [data-grain="operation"]').click()
            self.assertIn("block=opt_adamw",p.locator("#granularityLaunch").get_attribute("href"))
            self.assertEqual(p.evaluate("window.SeetheTruthGranularityExplorer.current().targetKind"),"operator")
            p.locator('#granularitySteps [data-grain="system"]').click()
            p.locator('#architectureTracks [data-track="video"]').click()
            self.assertEqual(p.evaluate("window.SeetheTruthArchitectureAtlas.current().track"),"video")
            p.locator('#architectureSteps [data-layer="runtime"]').click()
            self.assertIn("Wan FFN",p.locator("#architectureArtifact").inner_text())
            p.locator('#architectureTracks [data-track="agent"]').click()
            p.locator('#architectureSteps [data-layer="evidence"]').click()
            self.assertEqual(p.evaluate("window.SeetheTruthArchitectureAtlas.current().level"),"evidence")
            self.assertIn("真实工具",p.locator("#architectureEvidence").inner_text())
            self.assertTrue(p.locator("#architectureLaunch").get_attribute("href").endswith("semantic_state.html"))
            p.locator('#architectureTracks [data-track="posttrain"]').click()
            self.assertIn("训练",p.locator("#architectureGoal").inner_text())
            p.locator("#vKeep").select_option("8")
            self.assertEqual(p.evaluate("window.SeetheTruthLastResearch.video.errorL2"),0)
            p.locator("#vKeep").select_option("4")
            self.assertGreater(p.evaluate("window.SeetheTruthLastResearch.video.errorL2"),0)
            p.locator("#pObjective").select_option("grpo")
            p.locator("#pParam").select_option("0")
            self.assertEqual(p.evaluate("window.SeetheTruthLastResearch.post.loss"),0)
            p.locator("#pObjective").select_option("dpo")
            p.locator("#pParam").select_option("0.5")
            self.assertLess(p.evaluate("window.SeetheTruthLastResearch.post.gradChosen"),0)
            p.locator("#aApproved").select_option("false")
            self.assertEqual(p.evaluate("window.SeetheTruthLastResearch.agent.status"),"blocked")
            self.assertFalse(p.evaluate("window.SeetheTruthLastResearch.agent.events.some(e => e.action === 'write_tool')"))
            p.locator("#aApproved").select_option("true")
            p.locator("#aArchitecture").select_option("plan_execute")
            self.assertEqual(p.evaluate("window.SeetheTruthLastResearch.agent.status"),"failed")
            p.locator("#aArchitecture").select_option("supervisor")
            self.assertEqual(p.evaluate("window.SeetheTruthLastResearch.agent.status"),"completed")
            p.locator("#voteCase").select_option("any")
            self.assertEqual(p.evaluate("window.SeetheTruthLastSharedVote.point"),"ANY")
            self.assertEqual(p.evaluate("window.SeetheTruthLastSharedVote.perRank"),["prompt","empty-prompt"])
            p.locator("#voteCase").select_option("fallback")
            self.assertTrue(p.evaluate("window.SeetheTruthLastSharedVote.secondVoteFallback"))
            self.assertEqual(p.evaluate("window.SeetheTruthLastSharedVote.step"),"decode")
            p.locator('[data-lesson="prefill"]').click()
            self.assertIn("Prefill / Decode",p.locator("#detailTitle").inner_text())
            self.assertIn("ALL",p.locator("#why").inner_text())
            p.locator("#reveal").click()
            self.assertIn("第二轮",p.locator("#answer").inner_text())
            self.assertTrue(p.locator("#openLesson").get_attribute("href").endswith("deepinfer_prefill_lab.html"))
        self.assertEqual(normal.locator("#theory").inner_text(),phone.locator("#theory").inner_text())
        self.assertEqual(normal.locator("#answer").inner_text(),phone.locator("#answer").inner_text())
        self.assertTrue(phone.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2'),
                        "mobile teaching page must not overflow width")
        normal.screenshot(path=str(OUT/"curriculum-desktop.png"),full_page=True,animations="disabled")
        phone.screenshot(path=str(OUT/"curriculum-android-390.png"),full_page=True,animations="disabled")
        normal.close();phone.close()
        self.assertEqual(errors,[])

    def test_mobile_teaching_hides_generated_artifacts_but_keeps_math(self):
        p=self.browser.new_page(viewport={"width":390,"height":844},is_mobile=True,has_touch=True,
                                device_scale_factor=2)
        errors=[]
        p.on("pageerror",lambda e:errors.append(str(e)))
        self.assertEqual(p.goto(self.base+"/teaching.html?edition=teaching").status,200)
        p.locator('[data-lesson="building"]').click()
        p.locator("#openLesson").click()
        p.wait_for_url("**/learn.html?edition=teaching")
        p.wait_for_function("document.body.classList.contains('teaching-edition')")
        self.assertTrue(p.locator("#run").is_visible())
        self.assertTrue(p.locator("#blockSearch").is_visible())
        self.assertFalse(p.locator("#save").is_visible())
        self.assertFalse(p.locator("#exportPython").is_visible())
        self.assertFalse(p.locator('#pythonExportPanel').is_visible())
        self.assertTrue(p.locator("[data-preset]").count()>=59)
        self.assertTrue(p.locator("#architectureBlockRoles").is_visible())
        p.locator('[data-role-filter="runtime"]').click()
        self.assertEqual(p.locator('#architectureBlockRoles [data-role-filter="runtime"]').get_attribute("aria-pressed"),"true")
        self.assertGreater(p.locator('#palette [data-role="runtime"]').count(),0)
        self.assertEqual(p.locator('#palette button[data-role="operator"]').count(),0)
        p.locator('[data-role-filter="all"]').click()
        self.assertGreater(p.locator("#palette button[data-add]").count(),100)

        self.assertEqual(p.goto(self.base+"/learn.html?edition=teaching&block=opt_adamw").status,200)
        p.wait_for_function('document.querySelector("[data-grain-focus=true]") !== null')
        self.assertEqual(p.locator('[data-grain-focus="true"]').get_attribute("data-add"),"opt_adamw")
        self.assertEqual(p.goto(self.base+"/learn.html?edition=teaching&composite=ffn").status,200)
        p.wait_for_function('!document.querySelector("#compositeNavigator").hidden')
        self.assertIn("FFN",p.locator("#compositeTitle").inner_text())

        p.screenshot(path=str(OUT/"shared-graph-learning-android.png"),
                     full_page=True,animations="disabled")
        p.close()
        self.assertEqual(errors,[])

if __name__=="__main__":
    unittest.main()
