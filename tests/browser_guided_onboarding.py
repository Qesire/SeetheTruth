"""Browser verification of real-user action → source-derived learning evidence."""
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler


class GuidedFirstUse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(headless=True, args=["--no-sandbox"])
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/learn.html"

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def test_desktop_and_phone_guidance(self):
        for width in (1366, 390):
            with self.subTest(width=width):
                page = self.browser.new_page(viewport={"width": width, "height": 850},
                    is_mobile=width == 390, has_touch=width == 390)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                try:
                    self.assertEqual(page.goto(self.url, wait_until="domcontentloaded").status, 200)
                    page.wait_for_function("!!window.SeetheTruthLearnerGuide && !!window.SeetheTruthUnifiedWorkbench")
                    self.assertTrue(page.locator("#learnerGuide").is_visible())
                    self.assertTrue(page.locator("#guideNext").is_disabled())
                    page.locator("#guideAction").click()
                    self.assertEqual(page.evaluate("window.SeetheTruthUnifiedWorkbench.selectedNode"), "up")
                    self.assertTrue(page.evaluate("window.SeetheTruthLearnerGuide.verified"))
                    page.locator("#guideNext").click()
                    page.locator("#guideAction").click()
                    self.assertEqual(page.locator("#studyMode").input_value(), "numeric")
                    self.assertTrue(page.locator("#guideNext").is_disabled(), "switching modes is not execution")
                    page.locator("#run").click()
                    page.wait_for_function("window.SeetheTruthLearnerGuide.verified")
                    result = page.evaluate("""() => {
                        const r=window.SeetheTruthUnifiedWorkbench.inspectAccepted();
                        return {numeric:r.numericOk,ids:r.session.trace.steps.map(x=>x.id),
                            shape:Object.fromEntries(r.session.trace.steps.filter(x=>['x','up','down'].includes(x.id)).map(x=>[x.id,x.shape]))};
                    }""")
                    self.assertTrue(result["numeric"])
                    self.assertEqual(result["ids"], ["x", "up", "act", "down"])
                    self.assertEqual(result["shape"], {"x": [3,4], "up": [3,8], "down": [3,4]})
                    page.locator("#guideNext").click()
                    page.locator("#guideAction").click()
                    self.assertEqual(page.evaluate("window.SeetheTruthUnifiedWorkbench.selectedNode"), "act")
                    page.locator("#guideNext").click()
                    page.locator("#guideAction").click()
                    self.assertIn("3×4", page.locator("#guideEvidence").inner_text())
                    screenshots = Path(__file__).resolve().parents[1] / "guided_onboarding_screenshots"
                    screenshots.mkdir(exist_ok=True)
                    page.screenshot(path=str(screenshots / f"guided-first-use-{width}.png"))
                    page.locator("#guideNext").click()
                    self.assertTrue(page.locator("#learnerGuide").is_visible())
                    self.assertEqual(page.evaluate("window.SeetheTruthLearnerGuide.course"), "attention")
                    self.assertEqual(page.evaluate("window.SeetheTruthLearnerGuide.step"), 1)
                    self.assertFalse(page.evaluate("window.SeetheTruthLearnerGuide.verified"),
                                     "prior FFN numerical evidence cannot complete Attention")
                    page.locator("#guideAction").click()
                    self.assertEqual(page.evaluate("window.SeetheTruthUnifiedWorkbench.selectedNode"), "q")
                    self.assertTrue(page.evaluate("window.SeetheTruthLearnerGuide.verified"))
                    page.locator("#guideNext").click()
                    self.assertTrue(page.locator("#guideNext").is_disabled(),
                                    "a new lesson must obtain its own accepted numerical trace")
                    self.assertLessEqual(page.locator("#learnerGuide").evaluate(
                        "(e)=>e.getBoundingClientRect().width"), width + 2)
                    self.assertEqual(errors, [])
                finally:
                    page.close()


    def test_curriculum_adapts_to_real_source_nodes(self):
        page = self.browser.new_page(viewport={"width": 1366, "height": 850})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            self.assertEqual(page.goto(self.url,wait_until="domcontentloaded").status,200)
            focus = {
                "attention": "q",
                "trainable_core": "shared_weight",
                "nd_wan_i2v_full_block": "after_self",
                "image_vit_patch": "patches",
                "audio_spectrogram": "spectrum",
                "video_causal_conv3d": "causal_conv",
                "nd_gqa_attention": "key_groups"
            }
            for lesson, expected in focus.items():
                with self.subTest(lesson=lesson):
                    page.locator('[data-preset="' + lesson + '"]').click()
                    page.wait_for_function("(id) => window.SeetheTruthLearnerGuide.course === id", arg=lesson)
                    self.assertEqual(page.evaluate("window.SeetheTruthLearnerGuide.step"), 1)
                    page.locator("#guideAction").click()
                    self.assertEqual(page.evaluate("window.SeetheTruthUnifiedWorkbench.selectedNode"), expected)
                    self.assertTrue(page.evaluate("window.SeetheTruthLearnerGuide.verified"))
                    page.locator("#guideNext").click()
                    self.assertTrue(page.locator("#guideNext").is_disabled(),
                      "lesson requires current graph numerical evidence before progressing")
            self.assertEqual(errors, [])
        finally:
            page.close()

    def test_graph_edit_revokes_numeric_progress(self):
        page = self.browser.new_page(viewport={"width": 1366, "height": 850})
        try:
            self.assertEqual(page.goto(self.url,wait_until="domcontentloaded").status,200)
            page.wait_for_function("!!window.SeetheTruthLearnerGuide")
            page.locator("#guideAction").click()
            page.locator("#guideNext").click()
            page.locator("#guideAction").click()
            page.locator("#run").click()
            page.wait_for_function("window.SeetheTruthLearnerGuide.verified")
            page.locator("#challenge").click()
            page.wait_for_function("!window.SeetheTruthLearnerGuide.verified")
            self.assertTrue(page.locator("#guideNext").is_disabled())
            self.assertNotIn("已完成 4 个节点的微型计算",page.locator("#guideEvidence").inner_text())
        finally:
            page.close()


if __name__ == "__main__":
    unittest.main()
