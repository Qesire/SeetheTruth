"""Real Chromium acceptance for the DeepInfer logical Worker event laboratory."""
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright
from atlas_server import Handler

OUT=Path('deepinfer_worker_screenshots')
class DeepInferWorkerBrowser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True,exist_ok=True)
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.playwright=sync_playwright().start()
        cls.browser=cls.playwright.chromium.launch(headless=True,args=['--no-sandbox'])
        cls.url='http://127.0.0.1:'+str(cls.server.server_port)
    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.server.shutdown()
        cls.server.server_close()
    def setUp(self):
        self.page=self.browser.new_page(viewport={'width':1460,'height':1060},accept_downloads=True)
        self.errors=[]
        self.page.on('pageerror',lambda e:self.errors.append(str(e)))
        response=self.page.goto(self.url+'/deepinfer_worker_lab.html',wait_until='domcontentloaded')
        self.assertEqual(response.status,200)
        self.page.wait_for_function('!!window.SeetheTruthLastWorkerComparison')
    def tearDown(self):
        self.assertEqual(self.errors,[])
        self.page.close()
    def test_visual_event_dependencies_and_reproducible_export(self):
        p=self.page
        self.assertGreater(p.locator('#zosChart svg rect.event').count(),15)
        self.assertGreater(p.locator('#blockingChart svg rect.event').count(),12)
        cmp=p.evaluate('window.SeetheTruthLastWorkerComparison')
        self.assertEqual(len(cmp['zos']['released']),4)
        self.assertEqual(len(cmp['non']['released']),4)
        self.assertEqual(cmp['zos']['released'][-1]['via'],'flush')
        self.assertGreater(cmp['non']['makespan'],cmp['zos']['makespan'])
        p.locator('#zosChart svg rect.event').first.click()
        self.assertIn('Dependencies:',p.locator('#details').inner_text())
        p.screenshot(path=str(OUT/'zos-vs-nonzos-default.png'),full_page=True,animations='disabled')
        p.locator('#postTime').evaluate("(el) => { el.value='9'; el.dispatchEvent(new Event('input', {bubbles:true})); }")
        self.assertEqual(p.locator('#postValue').inner_text(),'9')
        self.assertNotEqual(p.locator('#deltaTime').inner_text(),'—')
        with p.expect_download() as d:
            p.locator('#download').click()
        export=json.loads(Path(d.value.path()).read_text(encoding='utf8'))
        self.assertEqual(export['zos']['released'][-1]['via'],'flush')
        self.assertTrue(export['zos']['events'])
        self.assertIn('not measured',export['zos']['note'])
        p.locator('#scenario').select_option('nooverlap')
        updated=p.evaluate('window.SeetheTruthLastWorkerComparison')
        self.assertEqual([e['stage'] for e in updated['zos']['events']].count('correct_zos_metadata'),0)
        p.locator('#flush').uncheck()
        self.assertIn('待 Flush',p.locator('#results').inner_text())
        self.assertIsNotNone(p.evaluate('window.SeetheTruthLastWorkerComparison.zos.pendingStep'))
        p.screenshot(path=str(OUT/'decode-nooverlap-pending.png'),full_page=True,animations='disabled')
    def test_custom_steps_validation_and_placeholder(self):
        p=self.page
        p.locator('#customSteps').evaluate('(e)=>{ e.closest("details").open=true; }')
        p.locator('#customSteps').fill('[{"mode":"decode","sequenceIds":["A"]},{"mode":"placeholder","sequenceIds":[]},{"mode":"decode","sequenceIds":["A"]}]')
        p.locator('#applySteps').click()
        self.assertIn('已执行合法',p.locator('#editorStatus').inner_text())
        result=p.evaluate('window.SeetheTruthLastWorkerComparison')
        self.assertEqual(len(result['zos']['released']),2)
        self.assertEqual(len([e for e in result['zos']['events'] if e['stage']=='forward']),3)
        self.assertEqual(len([e for e in result['zos']['events'] if e['stage']=='correct_zos_metadata']),0)
        p.locator('#customSteps').fill('[{"mode":"decode","sequenceIds":["A","A"]}]')
        p.locator('#applySteps').click()
        self.assertIn('输入被拒绝',p.locator('#editorStatus').inner_text())
        self.assertEqual(p.evaluate('window.SeetheTruthLastWorkerComparison.zos.steps.length'),3)
        p.screenshot(path=str(OUT/'placeholder-and-validation.png'),full_page=True,animations='disabled')


    def test_source_exact_zos_qk_and_invalid_fixture(self):
        p=self.page
        p.wait_for_function('window.SeetheTruthLastWorkerComparison.zos.metadataTrace.length === 2')
        traces=p.evaluate('window.SeetheTruthLastWorkerComparison.zos.metadataTrace')
        self.assertEqual(traces[0]['remap'],[1,-1,-1])
        self.assertEqual(traces[0]['seqlensQ'],[2,1,1])
        self.assertEqual(traces[0]['seqlensK'],[14,5,1])
        self.assertEqual(traces[0]['previousAccepted'],[1,2])
        self.assertEqual(p.locator('#metadataRows tr').count(),2)
        p.screenshot(path=str(OUT/'zos-exact-qk-metadata.png'),full_page=True,animations='disabled')
        p.locator('#metadataJson').evaluate('(e)=>{ e.closest("details").open=true; }')
        original=p.locator('#metadataJson').input_value()
        p.locator('#metadataJson').fill('{"nextn":2,"variableMtp":true,"records":[]}')
        p.locator('#applyMetadata').click()
        self.assertIn('Fixture 被拒绝',p.locator('#metadataStatus').inner_text())
        self.assertEqual(p.evaluate('window.SeetheTruthLastWorkerComparison.zos.metadataTrace'),traces)
        p.locator('#metadataJson').fill(original)
        p.locator('#applyMetadata').click()
        self.assertIn('已执行',p.locator('#metadataStatus').inner_text())
        p.locator('#numericMetadata').uncheck()
        self.assertEqual(p.evaluate('window.SeetheTruthLastWorkerComparison.zos.metadataTrace.length'),0)
        self.assertEqual(p.locator('#metadataRows tr').count(),0)


    def test_decode_compressed_block_table_and_mtp_slot_real_ui(self):
        p=self.page
        p.wait_for_function('window.SeetheTruthLastWorkerComparison.zos.metadataTrace[0].decode !== null')
        trace=p.evaluate('window.SeetheTruthLastWorkerComparison.zos.metadataTrace[0]')
        decoded=trace['decode']
        self.assertEqual(decoded['packed']['blockTables'],[7,4,8,9,2,3])
        self.assertEqual(decoded['packed']['blockTableOffsets'],[0,4,6])
        self.assertEqual(decoded['packed']['numBlocks'],[4,2])
        self.assertEqual(decoded['packed']['cuSeqlensQ'],[1,2])
        self.assertEqual(decoded['sampling']['metadata'][0]['seqlenK'],14)
        self.assertEqual(decoded['sampling']['metadata'][0]['mtpInfoReadSlot'],5)
        self.assertEqual(decoded['sampling']['metadata'][0]['mtpInfoWriteSlot'],5)
        self.assertEqual(decoded['sampling']['metadata'][0]['goldenToken'],[11,12,13,14,15])
        self.assertEqual(decoded['sampling']['states'][0]['mtp']['hasDraftInfo'],False)
        self.assertEqual(p.locator('#decodeRows tr').count(),1)
        p.screenshot(path=str(OUT/'decode-compressed-kv-mtp-slots.png'),full_page=True,animations='disabled')
        p.locator('#decodeJson').evaluate('(el)=>{el.closest("details").open=true;}')
        original=p.locator('#decodeJson').input_value()
        p.locator('#decodeJson').fill('{"step":2,"input":{},"states":[]}')
        p.locator('#applyDecode').click()
        self.assertIn('Fixture 被拒绝',p.locator('#decodeStatus').inner_text())
        self.assertEqual(p.evaluate('window.SeetheTruthLastWorkerComparison.zos.metadataTrace[0].decode.packed.blockTableOffsets'),[0,4,6])
        p.locator('#decodeJson').fill(original)
        p.locator('#applyDecode').click()
        self.assertIn('已完成',p.locator('#decodeStatus').inner_text())
        p.locator('#decodeEnabled').uncheck()
        self.assertEqual(p.locator('#decodeRows tr').count(),0)
        self.assertEqual(p.evaluate('window.SeetheTruthLastWorkerComparison.zos.metadataTrace[0].decode'),None)

if __name__=='__main__':
    unittest.main()
