"""Screenshot-backed visible-browser acceptance of the stateful KV learning page.

Tests compare rendered DOM counters/heatmap to live JS state via label and
cell assertions; PNGs are captured from the same real Chromium page.
The screenshots are evidence of a rendered browser, not generative art.
"""
import json
import os
import struct
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler


OUT = Path(os.environ.get("SEETHTRUTH_SCREENSHOT_DIR", "screenshots"))


class KVBrowserScreenshots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True, exist_ok=True)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:" + str(cls.server.server_port)
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True, args=["--no-sandbox"])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 1600, "height": 1000},
                                          device_scale_factor=1)
        self.errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.goto(self.base + "/cache.html", wait_until="domcontentloaded")
        self.page.wait_for_function("document.querySelectorAll('#cacheSlots .slot').length === 12")

    def tearDown(self):
        self.assertEqual(self.errors, [], "browser JS errors")
        self.page.close()

    def screenshot(self, name):
        page = self.page
        page.evaluate("document.fonts.ready")
        png = page.screenshot(path=str(OUT / name), full_page=True, animations="disabled")
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height = struct.unpack(">II", png[16:24])
        self.assertGreaterEqual(width, 1400)
        self.assertGreaterEqual(height, 900)
        return {"file": name, "width": width, "height": height,
                "bytes": len(png)}

    def assert_screen(self, phase, length, queries, heat_cells):
        page = self.page
        self.assertEqual(page.locator("#phaseMetric").inner_text(), phase)
        self.assertEqual(page.locator("#lengthMetric").inner_text(), length)
        self.assertEqual(page.locator("#queryMetric").inner_text(), queries)
        self.assertEqual(page.locator("#cacheSlots .filled").count(), int(length.split()[0]))
        self.assertEqual(page.locator("#matrix .heat").count(), heat_cells)
        filled = int(length.split()[0])
        self.assertEqual(page.locator("#keyValues .kv-mini-cell").count(), filled*4)
        self.assertEqual(page.locator("#valueValues .kv-mini-cell").count(), filled*4)
        if filled:
            self.assertEqual(page.locator("#cacheSlots .latest").count(), queries == "3" and 3 or 1)
            current = page.locator("#timeline li.active")
            self.assertEqual(current.count(), 1)
            self.assertTrue(current.evaluate(
                "(e) => { const box=e.parentElement.getBoundingClientRect();"
                " const r=e.getBoundingClientRect();"
                " return r.top>=box.top-2 && r.bottom<=box.bottom+3; }"),
                "Active timeline step must be visible inside scroll region")

        self.assertFalse(page.evaluate("document.documentElement.scrollWidth > innerWidth + 2"),
                         "unexpected horizontal document overflow")
        for element in ["nav", "#phaseMetric", "#cacheSlots", "#matrix", "#observed"]:
            rect = page.locator(element).bounding_box()
            self.assertIsNotNone(rect, element)
            self.assertGreater(rect["width"], 10)
            self.assertGreaterEqual(rect["x"], 0)

    def test_visible_initial_prefill_and_decode(self):
        p = self.page
        self.assertIn("STATEFUL LAB", p.locator("nav").inner_text())
        self.assertIn("KV Cache", p.locator("h1").inner_text())
        self.assert_screen("EMPTY", "0 / 7", "—", 0)
        first = self.screenshot("01-kv-empty.png")
        p.locator("#advance").click()
        self.assert_screen("PREFILL", "3 / 7", "3", 9)
        self.assertEqual(p.locator("#cacheSlots .latest").count(), 3)
        self.assertIn("PREFILL", p.locator("#timeline").inner_text())
        second = self.screenshot("02-kv-prefill.png")
        p.locator("#advance").click()
        self.assert_screen("DECODE", "4 / 7", "1", 4)
        self.assertEqual(p.locator("#cacheSlots .latest").count(), 1)
        self.assertIn("Q 1", p.locator("#timeline").inner_text())
        p.locator("#advance").click()
        p.locator("#advance").click()
        self.assert_screen("DECODE", "6 / 7", "1", 6)
        self.assertEqual(p.locator("#timeline li").count(), 4)
        p.locator("#headSlice").select_option("1")
        self.assertEqual(p.locator("#matrix .heat").count(), 6)
        third = self.screenshot("03-kv-decode-head1.png")
        before = p.locator("#cacheSlots .filled").count()
        p.locator('[data-tick="0"]').click()
        self.assert_screen("PREFILL", "3 / 7", "3", 9)
        self.assertEqual(p.locator("#timeline li").count(), 4)
        p.locator("#advance").click()
        self.assert_screen("DECODE", "7 / 7", "1", 7)
        self.assertTrue(p.locator("#advance").is_disabled())
        self.assertEqual(before+1,p.locator("#cacheSlots .filled").count())
        fourth = self.screenshot("04-kv-finished.png")
        p.locator("#reset").click()
        self.assert_screen("EMPTY", "0 / 7", "—", 0)
        self.assertEqual(p.locator("#timeline li").count(), 1)
        (OUT / "manifest.json").write_text(
            json.dumps({"schema": "seethetruth-browser-screenshot-evidence-v1",
                        "headless_browser": "Chromium (Playwright)",
                        "viewport": [1600, 1000],
                        "states": [first, second, third, fourth],
                        "tested_transitions": "empty/prefill/decode/review/finish/reset",
                        "evidence_limits": "rendered deterministic teaching micro-model, not production LLM"},
                       indent=2, ensure_ascii=False), encoding="utf-8")

    def test_invalid_config_does_not_erase_live_session(self):
        p = self.page
        p.locator("#advance").click()
        p.locator("#prefix").fill("10")
        p.locator("#prefix").dispatch_event("change")
        self.assertIn("配置不合法", p.locator("#status").inner_text())
        # Failed config must NOT claim it created a fresh, valid cache.
        self.assertEqual(p.locator("#lengthMetric").inner_text(), "3 / 7")
        self.assertEqual(p.locator("#cacheSlots .filled").count(), 3)
        p.locator("#prefix").fill("3")
        p.locator("#prefix").dispatch_event("change")
        self.assert_screen("EMPTY", "0 / 7", "—", 0)

if __name__ == "__main__":
    unittest.main()
