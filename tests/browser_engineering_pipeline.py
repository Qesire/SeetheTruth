"""Chromium UI checks for the six-stage real engineering training pipeline."""
import threading, unittest, sys
from pathlib import Path
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
from playwright.sync_api import sync_playwright

OUT=Path('engineering_pipeline_screenshots')
class PipelineBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True,exist_ok=True)
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(headless=True,args=['--no-sandbox'])
        cls.url=f'http://127.0.0.1:{cls.server.server_port}/engineering_pipeline.html'
    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()

    def test_desktop_full_accumulation_resume_and_inference(self):
        p=self.browser.new_page(viewport={'width':1420,'height':920})
        errors=[];p.on('pageerror',lambda error:errors.append(str(error)))
        self.assertEqual(p.goto(self.url).status,200)
        p.wait_for_function("window.SeetheTruthEngineeringPipeline?.lessons === 6")
        self.assertTrue(p.locator('.stage').nth(1).is_disabled())
        p.locator('#answer').select_option('2')
        p.locator('#run').click()
        self.assertIn('机制判断未通过',p.locator('#log').inner_text())
        self.assertEqual(p.evaluate('window.SeetheTruthEngineeringPipeline.completed'),0)
        answers=[0,1,1,0,1,0]
        for i,correct in enumerate(answers):
            self.assertEqual(p.evaluate('window.SeetheTruthEngineeringPipeline.stage'),i)
            p.locator('#answer').select_option(str(correct));p.locator('#run').click()
            message=p.locator('#log').inner_text()
            self.assertIn('PASS',message,f'phase {i} failed: {message}')
            if i==1:
                self.assertEqual(p.locator('#stateMicroscope .sm-step').count(),2)
                self.assertEqual(p.evaluate('window.SeetheTruthEngineeringPipeline.optimizerStep'),0)
                self.assertEqual(p.evaluate('window.SeetheTruthEngineeringPipeline.pendingMicroSteps'),1)
            if i==2:
                self.assertIn('max |Wsplit-Wfull|',message)
                self.assertEqual(p.evaluate('window.SeetheTruthEngineeringPipeline.optimizerStep'),1)
                p.screenshot(path=str(OUT/'02-accumulation-desktop.png'),full_page=True,animations='disabled')
            if i==3:self.assertEqual(p.evaluate('window.SeetheTruthEngineeringPipeline.optimizerStep'),11)
            if i==4:
                self.assertIn('bit-exact',message)
                self.assertEqual(p.evaluate('window.SeetheTruthEngineeringPipeline.optimizerStep'),14)
                self.assertEqual(p.evaluate('window.SeetheTruthEngineeringPipeline.provenance'),14)
                p.screenshot(path=str(OUT/'04-resumed-training-desktop.png'),full_page=True,animations='disabled')
            if i==5:
                self.assertEqual(p.locator('#stateMicroscope .sm-step').count(),6)
                self.assertIn('repeat same seed stable',message)
                p.screenshot(path=str(OUT/'05-decode-handoff-desktop.png'),full_page=True,animations='disabled')
            if i<5:p.locator('#next').click()
        self.assertEqual(p.evaluate('window.SeetheTruthEngineeringPipeline.completed'),6)
        self.assertEqual(p.locator('#schedule span').count(),24)
        self.assertEqual(p.locator('#weights span').count(),64)
        p.locator('#reset').click()
        self.assertEqual(p.evaluate('window.SeetheTruthEngineeringPipeline.completed'),0)
        self.assertEqual(p.evaluate('window.SeetheTruthEngineeringPipeline.optimizerStep'),0)
        self.assertEqual(errors,[])
        p.close()

    def test_mobile_shared_numerical_engine_and_layout(self):
        p=self.browser.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True,device_scale_factor=2)
        errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
        self.assertEqual(p.goto(self.url+'?edition=teaching').status,200)
        p.wait_for_function("window.SeetheTruthEngineeringPipeline?.lessons === 6")
        self.assertTrue(p.evaluate('document.documentElement.scrollWidth <= innerWidth+2'),'horizontal overflow on phone')
        self.assertEqual(p.locator('a[download],input[type=file],#exportPython').count(),0)
        p.locator('#answer').select_option('0');p.locator('#run').click()
        self.assertIn('PASS',p.locator('#log').inner_text())
        self.assertEqual(p.locator('#schedule span').count(),24)
        self.assertEqual(p.locator('#weights span').count(),64)
        p.screenshot(path=str(OUT/'00-sampler-android-390.png'),full_page=True,animations='disabled')
        self.assertEqual(errors,[])
        p.close()

if __name__=='__main__':unittest.main()
