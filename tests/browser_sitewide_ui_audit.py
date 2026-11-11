"""Site-wide human-interface acceptance.

The source-bound application has twenty HTML surfaces, not twenty independent
UX standards. Test keyboard access, route discovery, desktop/phone first
viewport, and script health. Screenshots are observed UI artifacts.
"""
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

PAGES=(
    "index.html", "learn.html", "teaching.html", "android_teaching.html",
    "cache.html", "capabilities.html", "tradeoffs.html", "training.html",
    "micro_training.html", "paper_lab.html", "real_model_lab.html",
    "semantic_state.html", "telemetry.html", "wan_ffn_lab.html",
    "deepinfer_prefill_lab.html", "deepinfer_scheduler_lab.html",
    "deepinfer_kv_lifecycle_lab.html", "deepinfer_worker_lab.html",
    "engineering_journey.html", "engineering_pipeline.html",
)
OUT=Path("sitewide_ui_screenshots")

class SitewideHumanInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True,exist_ok=True)
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(headless=True,args=["--no-sandbox"])
        cls.base=f"http://127.0.0.1:{cls.server.server_port}"
        cls.audit=[]

    @classmethod
    def tearDownClass(cls):
        (OUT/"manifest.json").write_text(json.dumps({
            "purpose":"actual first-viewport evidence, not mobile real-device certificate",
            "pages":cls.audit
        },ensure_ascii=False,indent=2),encoding="utf-8")
        cls.browser.close()
        cls.pw.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def test_twenty_surfaces_offer_real_navigation_and_keyboard_entry(self):
        self.assertEqual(len(PAGES),20)
        for name in PAGES:
            for width,height in ((1440,900),(390,844)):
                with self.subTest(page=name,width=width):
                    page=self.browser.new_page(
                        viewport={"width":width,"height":height},
                        is_mobile=width==390,has_touch=width==390)
                    errors=[]
                    page.on("pageerror",lambda e:errors.append(str(e)))
                    try:
                        response=page.goto(self.base+"/"+name,wait_until="domcontentloaded",
                                           timeout=20000)
                        self.assertEqual(response.status,200,name)
                        self.assertEqual(page.locator("main#stt-main").count(),1,name)
                        self.assertEqual(page.locator(".stt-skip").count(),1,name)
                        self.assertGreater(page.locator("h1").count(),0,name)
                        self.assertGreater(page.locator('a[href^="teaching.html"]').count(),0,
                            name+" needs a return path to the shared curriculum")
                        page.locator(".stt-skip").focus()
                        self.assertTrue(page.evaluate("""() =>
                            document.activeElement?.classList.contains('stt-skip')"""))
                        self.assertTrue(page.evaluate("""() =>
                            getComputedStyle(document.activeElement).outlineStyle!=='none'"""),
                            name+" keyboard focus must be visible")
                        data=page.evaluate("""() => {
                            const screen=document.documentElement;
                            const all=[...document.querySelectorAll('button')];
                            const unlabeled=all.filter(b=>
                                !b.textContent.trim()&&!b.getAttribute('aria-label')
                                &&!b.getAttribute('title')).length;
                            return {
                                viewport:window.innerWidth,
                                pageWidth:screen.scrollWidth,
                                bodyWidth:document.body.scrollWidth,
                                buttons:all.length,
                                unlabeledButtons:unlabeled,
                                links:document.querySelectorAll('a').length,
                                horizontalOverflow:Math.max(0,screen.scrollWidth-window.innerWidth),
                                title:document.title
                            };
                        }""")
                        data.update({"route":name,"viewportWidth":width,"errors":errors[:]})
                        self.audit.append(data)
                        page.locator(".stt-skip").blur()
                        page.screenshot(path=str(OUT/(name.replace(".html","")+
                            ("-desktop.png" if width==1440 else "-phone.png"))),
                            full_page=False,animations="disabled")
                        self.assertLessEqual(data["horizontalOverflow"],2,
                            name+" phone/desktop screen has unintended whole-document overflow")
                        self.assertEqual(errors,[],name+" page scripts failed")
                    finally:
                        page.close()

    def test_primary_headers_reveal_optional_labs_without_crowding(self):
        page=self.browser.new_page(viewport={"width":390,"height":844},
                                   is_mobile=True,has_touch=True)
        errors=[]
        page.on("pageerror",lambda e:errors.append(str(e)))
        try:
            self.assertEqual(page.goto(self.base+"/learn.html").status,200)
            first=page.evaluate("""() => ({
                stage:document.querySelector('.stage').getBoundingClientRect().top,
                catalog:document.querySelector('#stt-catalog').getBoundingClientRect().top
            })""")
            self.assertLess(first["stage"],first["catalog"],
                "mobile runnable graph must appear before the long catalogue")
            self.assertTrue(page.locator("#jumpToCatalog").is_visible())
            page.locator("#jumpToCatalog").click()
            page.wait_for_function("""() => document.querySelector('#stt-catalog')
                .getBoundingClientRect().top < window.innerHeight""")
            self.assertTrue(page.locator("#jumpToGraph").is_visible())
            page.locator("#jumpToGraph").click()
            page.wait_for_function("""() => document.querySelector('#stt-main')
                .getBoundingClientRect().top < window.innerHeight""")
            menu=page.locator(".head-actions .stt-lab-menu")
            self.assertEqual(menu.count(),1)
            self.assertFalse(menu.locator('a[href="cache.html"]').is_visible())
            menu.locator("summary").click()
            self.assertTrue(menu.locator('a[href="cache.html"]').is_visible())
            self.assertIn("课程中心",page.locator(".head-actions").inner_text())
            self.assertEqual(page.locator("#blocks .block").count()>0,True)
            self.assertEqual(page.goto(self.base+"/index.html").status,200)
            indexMenu=page.locator(".header-right .stt-lab-menu")
            self.assertEqual(indexMenu.count(),1)
            indexMenu.locator("summary").click()
            self.assertTrue(page.locator("#analyzeOpen").is_visible())
            self.assertTrue(page.locator("#importOpen").is_visible())
            self.assertEqual(errors,[])
        finally:
            page.close()

if __name__=="__main__":
    unittest.main()
