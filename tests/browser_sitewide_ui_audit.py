"""Site-wide human-interface acceptance.

The source-bound application has twenty HTML surfaces, not twenty independent
UX standards. Test keyboard access, route discovery, desktop/phone first
viewport, and script health. Screenshots are observed UI artifacts.
"""
import json
import os
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
        launch={"headless":True,"args":["--no-sandbox"]}
        if os.getenv("SEETHETRUTH_CHROMIUM_PATH"):
            launch["executable_path"]=os.environ["SEETHETRUTH_CHROMIUM_PATH"]
        cls.browser=cls.pw.chromium.launch(**launch)
        cls.base=f"http://127.0.0.1:{cls.server.server_port}"
        cls.audit=[]

    @classmethod
    def tearDownClass(cls):
        (OUT/"manifest.json").write_text(json.dumps({
            "purpose":"font-backed first-viewport evidence; not mobile real-device certification",
            "cjkFontProbe":getattr(cls,"font_probe",None),
            "pages":cls.audit
        },ensure_ascii=False,indent=2),encoding="utf-8")
        cls.browser.close()
        cls.pw.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def test_cjk_glyphs_are_rendered_not_missing_glyph_boxes(self):
        """Catch tofu screenshots even when every UI action and route passes."""
        page=self.browser.new_page(viewport={"width":390,"height":844})
        try:
            self.assertEqual(page.goto(self.base+"/teaching.html").status,200)
            cls=type(self)
            cls.font_probe=page.evaluate("""() => {
                const canvas=document.createElement('canvas');
                canvas.width=180; canvas.height=64;
                const ctx=canvas.getContext('2d');
                ctx.font='32px "Noto Sans CJK SC", sans-serif';
                ctx.fillStyle='#000';ctx.textBaseline='top';
                const pixels=text=>{
                    ctx.clearRect(0,0,180,64);
                    ctx.fillText(text,4,4);
                    return new Uint8Array(ctx.getImageData(0,0,180,64).data);
                };
                const a=pixels('中文'), b=pixels('科技');
                let changed=0,painted=0;
                for(let i=3;i<a.length;i+=4){
                    if(a[i]!==b[i])changed++;
                    if(a[i]>0)painted++;
                }
                return {differentGlyphPixels:changed,drawnPixels:painted,
                    bodyFont:getComputedStyle(document.body).fontFamily};
            }""")
            self.assertGreater(cls.font_probe["drawnPixels"],200,
                "Chinese text did not draw even a visible glyph")
            self.assertGreater(cls.font_probe["differentGlyphPixels"],200,
                "Chinese characters rendered as identical tofu boxes; screenshot QA invalid")
        finally:
            page.close()

    def test_narrow_320px_first_use_routes(self):
        """Catch text-dependent overflow at a second, smaller mobile viewport."""
        for name in ("teaching.html","learn.html","index.html","wan_ffn_lab.html"):
            with self.subTest(route=name):
                page=self.browser.new_page(viewport={"width":320,"height":740},
                    is_mobile=True,has_touch=True)
                try:
                    self.assertEqual(page.goto(self.base+"/"+name).status,200)
                    page.screenshot(path=str(OUT/(name[:-5]+"-320px.png")),
                        full_page=False,animations="disabled")
                    overflow=page.evaluate("""() =>
                        document.documentElement.scrollWidth-window.innerWidth""")
                    self.assertLessEqual(overflow,2,name+" overflows at 320px")
                finally:
                    page.close()

    def test_course_action_and_real_execution_state(self):
        page=self.browser.new_page(viewport={"width":390,"height":844},
            is_mobile=True,has_touch=True)
        try:
            self.assertEqual(page.goto(self.base+"/teaching.html?course=vlm").status,200)
            self.assertIn("图像信息怎样改变文本预测",page.locator("#detailTitle").inner_text())
            self.assertIn("视觉前缀",page.locator("#openLesson").inner_text())
            self.assertIn("mission=extension_vlm_early_fusion",
                page.locator("#openLesson").get_attribute("href"))
            # The user's first numeric experiment is a small, fast FFN fixture.
            self.assertEqual(page.goto(self.base+"/learn.html").status,200)
            self.assertIn("尚未运行",page.locator("#executionMode").inner_text())
            page.locator("#run").click()
            self.assertIn("无数值输出",page.locator("#executionMode").inner_text())
            page.locator("#studyMode").select_option("numeric")
            self.assertIn("尚未运行",page.locator("#executionMode").inner_text(),
                "changing mode must revoke stale execution evidence")
            page.locator("#run").click()
            self.assertIn("合成数值",page.locator("#executionMode").inner_text())
            self.assertIn("第 1 /",page.locator("#executionStep").inner_text())
        finally:
            page.close()

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
            self.assertEqual(page.locator('.head-actions a[href="teaching.html"]').count(),1,
                "primary teaching entry must link to the canonical shared course route")
            self.assertEqual(page.locator("#blocks .block").count()>0,True)
            self.assertEqual(page.goto(self.base+"/index.html").status,200)
            indexMenu=page.locator(".header-right .stt-lab-menu")
            self.assertEqual(indexMenu.count(),1)
            indexMenu.locator("summary").click()
            self.assertTrue(page.locator("#analyzeOpen").is_visible())
            self.assertTrue(page.locator("#importOpen").is_visible())
            page.locator("#importOpen").click()
            self.assertTrue(page.locator("#modal").evaluate("(el)=>el.classList.contains('open')"),
                "opening tools must allow an actual import-dialog click, not just visibility")
            self.assertEqual(errors,[])
        finally:
            page.close()

if __name__=="__main__":
    unittest.main()
