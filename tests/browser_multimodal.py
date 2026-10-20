"""Real Chromium multimodal pipeline acceptance and screenshot evidence."""
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright
from atlas_server import Handler

DEST=Path("multimodal_screenshots")
class MultimodalBrowserAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DEST.mkdir(parents=True,exist_ok=True)
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
        self.page=self.browser.new_page(viewport={"width":1500,"height":1000})
        self.errors=[]
        self.page.on("pageerror",lambda e:self.errors.append(str(e)))
        response=self.page.goto(self.base+"/learn.html",wait_until="domcontentloaded")
        self.assertEqual(response.status,200)
        self.page.wait_for_function("window.SeetheTruthMultimodal&&window.SeetheTruthBlockKnowledge")
        self.page.locator("#studyMode").select_option("numeric")
    def tearDown(self):
        self.assertEqual(self.errors,[])
        self.page.close()
    def test_all_four_real_multimodal_lessons_and_code_exports(self):
        p=self.page
        cases=[
          ("image_vit_patch","vision_patchify","patches","vision_patchify"),
          ("audio_spectrogram","audio_stft_power","spectrum","audio_stft_power"),
          ("video_vivit_tubelet","video_tubelet","tubelets","video_tubelet"),
          ("multimodal_cross_attention","audio_log_power","log_power","audio_log_power"),
        ]
        for i,(preset,kind,focus,marker) in enumerate(cases):
            p.locator('[data-preset="'+preset+'"]').click()
            self.assertEqual(p.locator('[data-node="'+focus+'"]').count(),1)
            p.locator("#run").click()
            self.assertNotIn("尚未运行",p.locator("#stepInfo").inner_text())
            p.locator('[data-node="'+focus+'"] .block-head').click()
            self.assertGreaterEqual(p.locator("#blockKnowledge .knowledge-sources a").count(),1)
            p.locator("#pythonBackend").select_option("numpy")
            with p.expect_download() as d:
                p.locator("#exportPython").click()
            source=Path(d.value.path()).read_text(encoding="utf8")
            self.assertIn(marker,source)
            p.locator("#pythonBackend").select_option("torch")
            with p.expect_download() as d:
                p.locator("#exportPython").click()
            source=Path(d.value.path()).read_text(encoding="utf8")
            self.assertIn("import torch",source)
            self.assertIn("class GraphRuntime",source)
            p.screenshot(path=str(DEST/("%02d-"%i+preset+".png")),full_page=True,animations="disabled")
        self.assertEqual(len(list(DEST.glob("*.png"))),4)
        self.assertEqual(p.locator('#palette [data-add="vision_patchify"]').count(),1)
        self.assertEqual(p.locator('#palette [data-add="audio_stft_power"]').count(),1)
        self.assertEqual(p.locator('#palette [data-add="video_tubelet"]').count(),1)


    def test_trainable_conv_mel_and_ddim_real_browser(self):
        p=self.page
        cases=[
          ("vision_learned_conv","conv","vision_conv2d","F.conv2d"),
          ("audio_mel_chain","mel","audio_mel_filterbank","torch.tensor"),
          ("diffusion_ddim_chain","step_t","diffusion_ddim_step","torch")
        ]
        for i,(preset,node,kind,torch_anchor) in enumerate(cases):
            p.locator('[data-preset="'+preset+'"]').click()
            self.assertEqual(p.locator('[data-node="'+node+'"]').count(),1)
            p.locator("#run").click()
            self.assertNotIn("尚未运行",p.locator("#stepInfo").inner_text())
            p.locator('[data-node="'+node+'"] .block-head').click()
            self.assertGreaterEqual(p.locator("#blockKnowledge .knowledge-sources a").count(),1)
            p.locator("#pythonBackend").select_option("numpy")
            with p.expect_download() as d:
                p.locator("#exportPython").click()
            script=Path(d.value.path()).read_text(encoding="utf8")
            self.assertIn("# "+kind,script)
            p.locator("#pythonBackend").select_option("torch")
            with p.expect_download() as d:
                p.locator("#exportPython").click()
            script=Path(d.value.path()).read_text(encoding="utf8")
            self.assertIn(torch_anchor,script)
            p.screenshot(path=str(DEST/("advanced-%02d-"%i+preset+".png")),
                         full_page=True,animations="disabled")
        self.assertEqual(p.locator('#palette [data-add="vision_conv2d"]').count(),1)
        self.assertEqual(p.locator('#palette [data-add="audio_mel_filterbank"]').count(),1)
        self.assertEqual(p.locator('#palette [data-add="diffusion_ddim_step"]').count(),1)

    def test_native_causal_conv3d_real_browser_and_code_export(self):
        p=self.page
        p.locator('[data-preset="video_causal_conv3d"]').click()
        self.assertEqual(p.locator('[data-node="causal_conv"]').count(),1)
        p.locator('#run').click()
        self.assertNotIn("尚未运行",p.locator("#stepInfo").inner_text())
        p.locator('[data-node="causal_conv"] .block-head').click()
        card=p.locator('#blockKnowledge')
        self.assertIn("因果",card.inner_text())
        self.assertGreaterEqual(card.locator('.knowledge-sources a').count(),1)
        p.locator('#pythonBackend').select_option('numpy')
        with p.expect_download() as d:
            p.locator('#exportPython').click()
        numpy_source=Path(d.value.path()).read_text(encoding='utf8')
        self.assertIn('for dt in range(2)',numpy_source)
        p.locator('#pythonBackend').select_option('torch')
        with p.expect_download() as d:
            p.locator('#exportPython').click()
        torch_source=Path(d.value.path()).read_text(encoding='utf8')
        self.assertIn('F.conv3d(',torch_source)
        self.assertIn('F.pad(',torch_source)
        self.assertEqual(p.locator('#palette [data-add="video_causal_conv3d"]').count(),1)
        p.screenshot(path=str(DEST/'native-5d-causal-conv3d.png'),
                     full_page=True,animations='disabled')

if __name__=="__main__":
    unittest.main()
