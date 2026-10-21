"""Chromium acceptance: real end-to-end training/inference and Android-size gates."""
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
from playwright.sync_api import sync_playwright

class EngineeringJourneyBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
        cls.pw=sync_playwright().start()
        cls.browser=cls.pw.chromium.launch(headless=True,args=['--no-sandbox'])
        cls.url=f'http://127.0.0.1:{cls.server.server_port}/engineering_journey.html'
    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()

    def test_desktop_full_numeric_journey(self):
        page=self.browser.new_page(viewport={'width':1440,'height':940})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        self.assertEqual(page.goto(self.url).status,200)
        page.wait_for_function('window.SeetheTruthEngineeringJourney?.lessons === 12')
        self.assertTrue(page.locator('.stage').nth(1).is_disabled())
        page.locator('#answerSelect').select_option('2')
        page.locator('#run').click()
        self.assertIn('机制判断不成立',page.locator('#result').inner_text())
        self.assertTrue(page.locator('#next').is_disabled())
        answers=[0,1,1,1,1,0,0,1,1,1,1,0]
        for i,answer in enumerate(answers):
            self.assertEqual(page.evaluate('window.SeetheTruthEngineeringJourney.stage'),i)
            page.locator('#answerSelect').select_option(str(answer))
            page.locator('#run').click()
            result=page.locator('#result').inner_text()
            self.assertIn('PASS',result, f'stage {i}: {result}')
            if i==7:self.assertEqual(page.evaluate('window.SeetheTruthEngineeringJourney.modelStep'),120)
            if i==9:self.assertTrue(page.evaluate('window.SeetheTruthEngineeringJourney.snapshotReady'))
            if i==10:self.assertIn('decode steps',result)
            if i<11:page.locator('#next').click()
        self.assertEqual(page.evaluate('window.SeetheTruthEngineeringJourney.completed'),12)
        self.assertEqual(page.evaluate('window.SeetheTruthEngineeringJourney.modelStep'),120)
        page.locator('#reset').click()
        self.assertEqual(page.evaluate('window.SeetheTruthEngineeringJourney.completed'),0)
        self.assertEqual(page.evaluate('window.SeetheTruthEngineeringJourney.modelStep'),0)
        self.assertTrue(page.locator('.stage').nth(1).is_disabled())
        self.assertEqual(errors,[])
        page.close()

    def test_android_size_no_exports_and_no_overflow(self):
        page=self.browser.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True,device_scale_factor=2)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        self.assertEqual(page.goto(self.url+'?edition=teaching').status,200)
        page.wait_for_function('window.SeetheTruthEngineeringJourney?.lessons === 12')
        self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 2'))
        self.assertEqual(page.locator('a[download],input[type=file],#exportPython').count(),0)
        page.locator('#answerSelect').select_option('0');page.locator('#run').click()
        self.assertIn('PASS',page.locator('#result').inner_text())
        self.assertEqual(errors,[])
        page.close()

if __name__=='__main__':unittest.main()
