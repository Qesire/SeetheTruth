"""Real Chromium evidence of the 45 executable-block Scratch-like learning UI."""
import json
import os
import struct
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
from playwright.sync_api import sync_playwright
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler

DEST=Path(os.environ.get("SEETHTRUTH_CATALOG_SHOTS","catalog_screenshots"))


class BlockLibraryVisualAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DEST.mkdir(exist_ok=True,parents=True)
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.playwright=sync_playwright().start()
        cls.browser=cls.playwright.chromium.launch(headless=True,args=["--no-sandbox"])
        cls.base="http://127.0.0.1:"+str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.page=self.browser.new_page(viewport={"width":1600,"height":1000},device_scale_factor=1)
        self.errors=[]
        self.page.on("pageerror",lambda e:self.errors.append(str(e)))
        self.page.goto(self.base+"/learn.html",wait_until="domcontentloaded")
        self.page.wait_for_function("document.querySelectorAll('#palette [data-add]').length===45")
        self.page.locator("#studyMode").select_option("numeric")

    def tearDown(self):
        self.assertEqual(self.errors,[],"Browser JavaScript errors")
        self.page.close()

    def assert_all_graph_blocks_visible(self):
        p=self.page
        data=p.evaluate("""() => {
          const viewport=document.querySelector('#viewport').getBoundingClientRect();
          const nodes=[...document.querySelectorAll('#blocks .block')].map(n=>{
            const r=n.getBoundingClientRect();
            return {id:n.dataset.node,left:r.left,right:r.right,top:r.top,bottom:r.bottom};
          });
          return {viewport:{left:viewport.left,right:viewport.right,top:viewport.top,bottom:viewport.bottom},
                  zoom:Number(document.querySelector('#board').style.zoom),nodes};
        }""")
        self.assertGreater(data["zoom"],0.45)
        for n in data["nodes"]:
            self.assertGreaterEqual(n["left"],data["viewport"]["left"]-3,n["id"])
            self.assertLessEqual(n["right"],data["viewport"]["right"]+3,n["id"])
            self.assertLessEqual(n["bottom"],data["viewport"]["bottom"]+3,n["id"])

    def shot(self,name):
        p=self.page
        p.evaluate("document.fonts.ready")
        image=p.screenshot(path=str(DEST/name),full_page=True,animations="disabled")
        self.assertTrue(image.startswith(b"\x89PNG\r\n\x1a\n"))
        w,h=struct.unpack(">II",image[16:24])
        self.assertGreaterEqual(w,1600)
        self.assertGreaterEqual(h,980)
        self.assertFalse(p.evaluate("document.documentElement.scrollWidth>innerWidth+2"),
                         "document has horizontal overflow")
        self.assertIsNotNone(p.locator("#palette").bounding_box())
        return {"file":name,"width":w,"height":h,"bytes":len(image),
                "search":p.locator("#blockSearch").input_value(),
                "visible_catalog_items":p.locator("#palette [data-add]").count(),
                "mission":p.locator("#workspaceTitle").inner_text()}

    def test_search_filter_circuits_and_evidence_screens(self):
        p=self.page
        self.assertEqual(p.locator("#palette [data-add]").count(),45)
        self.assertIn("45 / 45",p.locator("#blockCount").inner_text())
        self.assertGreaterEqual(p.locator("#catalogCategories button").count(),10)
        screens=[self.shot("01-library-all-blocks.png")]
        p.locator("#blockSearch").fill("SwiGLU")
        self.assertEqual(p.locator("#palette [data-add]").count(),1)
        self.assertIn("SwiGLU",p.locator("#palette").inner_text())
        screens.append(self.shot("02-library-search-swiglu.png"))
        p.locator("#blockSearch").fill("")
        p.locator('[data-catalog="loss"]').click()
        self.assertEqual(p.locator("#palette [data-add]").count(),4)
        self.assertEqual(p.locator("#blockCount").inner_text(),"4 / 45")
        screens.append(self.shot("03-library-loss-category.png"))
        p.locator('[data-catalog="all"]').click()
        p.locator('[data-preset="swiglu"]').click()
        self.assertEqual(p.locator("#blocks .block").count(),5)
        p.locator("#run").click()
        self.assertIn("1 / 5",p.locator("#stepInfo").inner_text())
        for _ in range(3):p.locator("#next").click()
        self.assertIn("4 / 5",p.locator("#stepInfo").inner_text())
        self.assertIn("SwiGLU",p.locator("#inspectorTitle").inner_text())
        self.assertEqual(p.locator("#matrix .cell").count(),24)
        self.assert_all_graph_blocks_visible()
        screens.append(self.shot("04-swiglu-numeric-execution.png"))
        p.locator('[data-preset="classifier"]').click()
        p.locator("#run").click()
        for _ in range(4):p.locator("#next").click()
        self.assertIn("5 / 5",p.locator("#stepInfo").inner_text())
        self.assertIn("交叉熵",p.locator("#inspectorTitle").inner_text())
        self.assertEqual(p.locator("#matrix .cell").count(),1)
        self.assert_all_graph_blocks_visible()
        screens.append(self.shot("05-classifier-crossentropy.png"))
        p.locator('[data-preset="mask"]').click()
        p.locator("#run").click()
        p.locator("#next").click()
        p.locator("#next").click()
        self.assertIn("3 / 3",p.locator("#stepInfo").inner_text())
        self.assertIn("Mask Softmax",p.locator("#inspectorTitle").inner_text())
        self.assertEqual(p.locator("#matrix .cell").count(),9)
        self.assert_all_graph_blocks_visible()
        screens.append(self.shot("06-causal-mask-output.png"))
        (DEST/"manifest.json").write_text(json.dumps({
          "schema":"seethetruth-catalog-visual-evidence-v1","browser":"Playwright Chromium",
          "viewport":[1600,1000],"catalogue_count":45,
          "captures":screens+[{"file":"07-qualitative-gqa-no-matrix.png",
            "description":"No numeric matrices or hardware measurements; GQA dimensions only"}],
          "tests":"qualitative GQA without numeric cells, opt-in numeric SwiGLU/CE/Masked Softmax, search and category",
          "limit":"browser teaching simulation, not checkpoint or GPU execution"
        },ensure_ascii=False,indent=2),encoding="utf-8")

    def test_default_qualitative_library_capture(self):
        p=self.page
        p.locator("#studyMode").select_option("concept")
        p.locator('[data-preset="gqa"]').click()
        p.locator("#run").click()
        for _ in range(5):
            p.locator("#next").click()
        self.assertIn("6 / 7",p.locator("#stepInfo").inner_text())
        self.assertIn("推断 shape",p.locator("#shape").inner_text())
        self.assertIn("内部阶段的形状推断",p.locator("#intermediates").inner_text())
        self.assertEqual(p.locator("#matrix .cell").count(),0)
        self.assertNotIn("FLOPs",p.locator("#metrics").inner_text())
        self.shot("07-qualitative-gqa-no-matrix.png")

    def test_legacy_and_new_preset_numerical_runs(self):
        p=self.page
        for mission,count in [("ffn",4),("residual",4),("gqa",7),
                              ("embedding",3),("norm",5),("sampling",5),("mask",3)]:
            p.locator('[data-preset="'+mission+'"]').click()
            self.assertEqual(p.locator("#blocks .block").count(),count)
            p.locator("#run").click()
            self.assertIn("1 / "+str(count),p.locator("#stepInfo").inner_text(),mission)
            for _ in range(count-1):p.locator("#next").click()
            self.assertIn(str(count)+" / "+str(count),p.locator("#stepInfo").inner_text())
            self.assertNotIn("无法运行",p.locator("#status").inner_text())


if __name__=="__main__":
    unittest.main()
