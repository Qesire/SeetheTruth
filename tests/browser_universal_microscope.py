"""A real Chromium proof that the free-form workbench shares one 2D–5D numerical microscope."""
import sys,threading,unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from playwright.sync_api import sync_playwright
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler
OUT=Path('microscope_screenshots')
class MicroscopeBrowser(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  OUT.mkdir(parents=True,exist_ok=True)
  cls.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
  threading.Thread(target=cls.server.serve_forever,daemon=True).start()
  cls.pw=sync_playwright().start()
  cls.browser=cls.pw.chromium.launch(headless=True,args=['--no-sandbox'])
  cls.base=f'http://127.0.0.1:{cls.server.server_port}/learn.html'
 @classmethod
 def tearDownClass(cls):
  cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close()
 def test_ffn_attention_and_video(self):
  p=self.browser.new_page(viewport={'width':1490,'height':960})
  errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
  try:
   self.assertEqual(p.goto(self.base).status,200)
   p.locator('#studyMode').select_option('numeric')
   p.locator('#run').click()
   p.wait_for_function('window.SeetheTruthMicroscopeWorkbench?.stageCount===4')
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.evidence'),'EXECUTABLE_DETERMINISTIC_TEACHING_TENSORS')
   p.locator('#computeMicroscope .cmp-step').nth(1).click()
   self.assertGreater(p.locator('#computeMicroscope .cmp-terms tr').count(),2)
   self.assertIn('误差',p.locator('#computeMicroscope .cmp-breakdown').inner_text())
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   self.assertIn('已覆盖全部可验证路径',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   self.assertIn('累计灵敏度',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   self.assertIn('整图导出',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   p.screenshot(path=str(OUT/'ffn-forward-and-backward-coordinate.png'),animations='disabled')
   p.screenshot(path=str(OUT/'ffn-coordinate-proof.png'),animations='disabled')
   p.locator('#computeMicroscope .cmp-step').nth(2).click()
   self.assertIn('silu',p.locator('#computeMicroscope .cmp-meta').inner_text().lower())
   self.assertIn('逐行/逐坐标运算 silu',p.locator('#computeMicroscope .cmp-breakdown').inner_text())
   self.assertIn('误差',p.locator('#computeMicroscope .cmp-breakdown').inner_text())
   p.screenshot(path=str(OUT/'ffn-silu-nonlinearity-proof.png'),animations='disabled')
   p.locator('[data-preset="residual"]').click()
   p.locator('#run').click()
   p.locator('#computeMicroscope .cmp-step').last.click()
   self.assertIn('逐坐标原生运算',p.locator('#computeMicroscope .cmp-breakdown').inner_text())
   self.assertIn('误差',p.locator('#computeMicroscope .cmp-breakdown').inner_text())
   p.locator('[data-preset="attention"]').click()
   p.locator('#run').click()
   p.wait_for_function('window.SeetheTruthMicroscopeWorkbench?.stageCount===6')
   p.locator('#computeMicroscope .cmp-step').nth(4).click()
   self.assertEqual(p.locator('#computeMicroscope .cmp-substep').count(),4)
   p.locator('#computeMicroscope .cmp-substep').nth(1).click()
   self.assertGreater(p.locator('#computeMicroscope .cmp-terms tr').count(),2)
   self.assertIn('QK',p.locator('#computeMicroscope .cmp-meta').inner_text())
   self.assertGreater(p.locator('#computeMicroscope .cmp-terms tr').count(),2)
   p.locator('#computeMicroscope .cmp-substep').nth(2).click()
   self.assertIn('逐行/逐坐标运算 softmax',p.locator('#computeMicroscope .cmp-breakdown').inner_text())
   self.assertGreater(p.locator('#computeMicroscope .cmp-terms tr').count(),2)
   # Unified Attention composite: use the actual graph output, not a
   # standalone second simulated Attention explainer.
   p.locator('#computeMicroscope .cmp-substep').first.click()
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   evidence=p.locator('#computeMicroscope .cmp-vjp-result').inner_text()
   self.assertIn('key_softmax_value',evidence)
   self.assertIn('scaledScore=',evidence)
   self.assertIn('probability=',evidence)
   self.assertIn('累计灵敏度',evidence)
   p.screenshot(path=str(OUT/'attention-keywise-coordinate-jacobian.png'),animations='disabled')
   p.locator('[data-preset="extension_video_time_composition"]').click()
   p.locator('#run').click()
   p.wait_for_function('window.SeetheTruthMicroscopeWorkbench?.stageCount===4')
   p.locator('#computeMicroscope .cmp-step').last.click()
   p.locator('#computeMicroscope .cmp-substep').last.click()
   self.assertEqual(p.locator('#computeMicroscope .cmp-out .cmp-axes select').count(),3)
   self.assertIn('1 × 3 × 2 × 2 × 2',p.locator('#computeMicroscope .cmp-meta').inner_text())
   p.locator('#computeMicroscope .cmp-out .cmp-axes select').nth(1).select_option('2')
   self.assertIn('精确来源',p.locator('#computeMicroscope .cmp-breakdown').inner_text())
   self.assertIn('B=0,T=2,H=0',p.locator('#computeMicroscope .cmp-breakdown').inner_text())
   self.assertIn('B / T / H / W / C',p.locator('#computeMicroscope .cmp-meta').inner_text())
   p.locator('#computeMicroscope .cmp-substep').first.click()
   self.assertEqual(p.locator('#computeMicroscope .cmp-vjp-trigger').count(),1)
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   self.assertNotIn('反向路径拒绝',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   self.assertIn('整图导出',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   p.screenshot(path=str(OUT/'video-five-axes-coordinate.png'),animations='disabled')
   # Real MSE training objective: the same selected output coordinate now
   # distinguishes a single-seeded VJP from the FULL loss adjoint.
   p.locator('[data-preset="trainable_swiglu"]').click()
   p.locator('#studyMode').select_option('numeric');p.locator('#run').click()
   p.wait_for_function('window.SeetheTruthMicroscopeWorkbench?.stageCount===10')
   # SwiGLU shared gate/up path: inspect the same real training graph.
   for button in p.locator('#computeMicroscope .cmp-step').all():
    button.click()
    if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')=='activated':
     break
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'activated')
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   swi=p.locator('#computeMicroscope .cmp-vjp-result').inner_text()
   self.assertIn('gate_activation',swi)
   self.assertIn('elementwise_product',swi)
   self.assertNotIn('反向路径拒绝',swi)
   p.screenshot(path=str(OUT/'swiglu-two-path-coordinate-jacobian.png'),animations='disabled')
   p.locator('#computeMicroscope .cmp-step').last.click()
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'loss')
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   loss_report=p.locator('#computeMicroscope .cmp-vjp-result').inner_text()
   self.assertIn('TRAINING LOSS',loss_report)
   self.assertIn('∂L/∂yᵢ=1.00000000',loss_report)
   self.assertIn('停止梯度',loss_report)
   self.assertNotIn('反向路径拒绝',loss_report)
   p.screenshot(path=str(OUT/'trainable-swiglu-loss-vjp-stop-edges.png'),animations='disabled')
   for button in p.locator('#computeMicroscope .cmp-step').all():
    button.click()
    if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')=='w2':
     break
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'w2')
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   param_report=p.locator('#computeMicroscope .cmp-vjp-result').inner_text()
   self.assertIn('完整 dL/dSource',param_report)
   self.assertIn('经此输出',param_report)
   self.assertNotIn('反向路径拒绝',param_report)
   p.screenshot(path=str(OUT/'trainable-swiglu-parameter-loss-vjp.png'),animations='disabled')
   p.locator('[data-preset="trainable_moe"]').click()
   p.locator('#studyMode').select_option('numeric');p.locator('#run').click()
   for button in p.locator('#computeMicroscope .cmp-step').all():
    button.click()
    if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')=='expert_gate':
     break
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'expert_gate')
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   expert=p.locator('#computeMicroscope .cmp-vjp-result').inner_text()
   self.assertIn('expert_channel_product',expert)
   self.assertIn('DISCRETE ROUTING',expert)
   self.assertIn('固定离散选择',expert)
   self.assertNotIn('反向路径拒绝',expert)
   p.screenshot(path=str(OUT/'moe-expert-coordinate-vjp-and-cuts.png'),animations='disabled')
   p.locator('#studyMode').select_option('concept');p.locator('#run').click()
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.evidence'),'STATIC_SHAPES_NO_EXECUTION')
   self.assertEqual(p.locator('#computeMicroscope .cmp-cell').count(),0)
   self.assertEqual(p.locator('#computeMicroscope .cmp-vjp-trigger').count(),0)
   self.assertEqual(errors,[])
  finally:p.close()
 def test_nd_cached_gqa_coordinate_lineage(self):
  p=self.browser.new_page(viewport={'width':1450,'height':960})
  errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
  try:
   p.goto(self.base)
   p.locator('[data-preset="nd_cached_gqa_decode"]').click()
   p.locator('#studyMode').select_option('numeric');p.locator('#run').click()
   p.wait_for_function('window.SeetheTruthMicroscopeWorkbench?.stageCount>17')
   for button in p.locator('#computeMicroscope .cmp-step').all():
    button.click()
    if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')=='q_rope':
     break
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'q_rope')
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   rope=p.locator('#computeMicroscope .cmp-vjp-result').inner_text()
   self.assertIn('adjacent_rope_pair',rope)
   self.assertIn('absolutePosition=2',rope)
   self.assertIn('已覆盖全部可验证路径',rope)
   self.assertNotIn('反向路径拒绝',rope)
   p.screenshot(path=str(OUT/'cached-gqa-absolute-rope-jacobian.png'),animations='disabled')
   for button in p.locator('#computeMicroscope .cmp-step').all():
    button.click()
    if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')=='cached_k':
     break
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'cached_k')
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   self.assertIn('immutable_cache_append',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   p.screenshot(path=str(OUT/'cached-gqa-prefix-and-token-provenance.png'),animations='disabled')
   for button in p.locator('#computeMicroscope .cmp-step').all():
    button.click()
    if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')=='context':
     break
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'context')
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   context=p.locator('#computeMicroscope .cmp-vjp-result').inner_text()
   self.assertIn('已覆盖全部可验证路径',context)
   self.assertIn('累计灵敏度',context)
   self.assertNotIn('反向路径拒绝',context)
   p.screenshot(path=str(OUT/'cached-gqa-end-to-end-vjp.png'),animations='disabled')
   self.assertEqual(errors,[])
  finally:p.close()
 def test_wan_pinned_video_rope_3d(self):
  p=self.browser.new_page(viewport={'width':1500,'height':960})
  errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
  try:
   p.goto(self.base)
   p.locator('[data-preset="nd_wan_video_rope_attention"]').click()
   p.locator('#studyMode').select_option('numeric');p.locator('#run').click()
   p.wait_for_function('window.SeetheTruthMicroscopeWorkbench?.stageCount===10')
   for button in p.locator('#computeMicroscope .cmp-step').all():
    button.click()
    if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')=='q_video_rope':
     break
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'q_video_rope')
   p.locator('#computeMicroscope .cmp-out .cmp-cell').nth(4).click()
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   text=p.locator('#computeMicroscope .cmp-vjp-result').inner_text()
   self.assertIn('wan_video_3d_rope',text)
   self.assertIn('axis=W',text)
   self.assertIn('frequencyPair=2',text)
   self.assertIn('已覆盖全部可验证路径',text)
   self.assertNotIn('反向路径拒绝',text)
   p.screenshot(path=str(OUT/'wan-video-fhw-rope-coordinate.png'),animations='disabled')
   for button in p.locator('#computeMicroscope .cmp-step').all():
    button.click()
    if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')=='context':
     break
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'context')
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   evidence=p.locator('#computeMicroscope .cmp-vjp-result').inner_text()
   self.assertIn('已覆盖全部可验证路径',evidence)
   self.assertIn('累计灵敏度',evidence)
   self.assertNotIn('反向路径拒绝',evidence)
   p.screenshot(path=str(OUT/'wan-video-composable-attention-context.png'),animations='disabled')
   self.assertEqual(errors,[])
  finally:p.close()
 def test_wan_projected_qkv_learned_fullwidth_norm(self):
  for width in [1440,390]:
   p=self.browser.new_page(viewport={'width':width,'height':880},
    is_mobile=width==390,has_touch=width==390)
   errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
   try:
    p.goto(self.base)
    p.locator('[data-preset="nd_wan_qkv_fullwidth_attention"]').click()
    p.locator('#studyMode').select_option('numeric');p.locator('#run').click()
    p.wait_for_function('window.SeetheTruthMicroscopeWorkbench?.stageCount===36')
    for btn in p.locator('#computeMicroscope .cmp-step').all():
     btn.click()
     if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')=='q_norm':
      break
    self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'q_norm')
    p.locator('#computeMicroscope .cmp-vjp-trigger').click()
    evidence=p.locator('#computeMicroscope .cmp-vjp-result').inner_text()
    self.assertIn('wan_qk_fullwidth_rmsnorm',evidence)
    self.assertIn('weight → gamma_q',evidence)
    self.assertIn('invRms=',evidence)
    self.assertIn('已覆盖全部可验证路径',evidence)
    self.assertNotIn('反向路径拒绝',evidence)
    p.screenshot(path=str(OUT/('mobile-wan-qkv-gamma-fullwidth.png' if width==390
     else 'desktop-wan-qkv-gamma-fullwidth.png')),full_page=width==390,animations='disabled')
    self.assertEqual(errors,[])
   finally:p.close()
 def test_wan_adaln_norm1_conditioned_self_gate(self):
  for width in [1450,390]:
   p=self.browser.new_page(viewport={'width':width,'height':900},
    is_mobile=width==390,has_touch=width==390)
   errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
   try:
    p.goto(self.base)
    p.locator('[data-preset="nd_wan_adaln_self_gated"]').click()
    p.locator('#studyMode').select_option('numeric');p.locator('#run').click()
    p.wait_for_function('window.SeetheTruthMicroscopeWorkbench?.stageCount===51')
    for target,keyword in [('norm1','wan_norm1_lastchannel_layernorm'),
                           ('norm_scale_product','broadcast_gate')]:
     for btn in p.locator('#computeMicroscope .cmp-step').all():
      btn.click()
      if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')==target:
       break
     self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),target)
     p.locator('#computeMicroscope .cmp-vjp-trigger').click()
     content=p.locator('#computeMicroscope .cmp-vjp-result').inner_text()
     self.assertIn(keyword,content)
     self.assertIn('已覆盖全部可验证路径',content)
     self.assertNotIn('反向路径拒绝',content)
     p.screenshot(path=str(OUT/(('mobile-' if width==390 else 'desktop-')+
        'wan-'+target+'-scalar-vjp.png')),full_page=width==390,animations='disabled')
    self.assertEqual(errors,[])
   finally:p.close()
 def test_mobile_named_axis(self):
  p=self.browser.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True)
  errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
  try:
   p.goto(self.base)
   p.locator('[data-preset="extension_named_axis_composition"]').click()
   p.locator('#studyMode').select_option('numeric');p.locator('#run').click()
   p.wait_for_function('window.SeetheTruthMicroscopeWorkbench?.stageCount===7')
   p.locator('#computeMicroscope .cmp-step').nth(3).click()
   self.assertGreater(p.locator('#computeMicroscope .cmp-substep').count(),1)
   p.locator('#computeMicroscope .cmp-substep').last.click()
   self.assertIn('输出张量',p.locator('#computeMicroscope').inner_text())
   self.assertGreaterEqual(p.locator('#computeMicroscope .cmp-vjp-trigger').count(),0)
   self.assertIn('D / T',p.locator('#computeMicroscope .cmp-meta').inner_text())
   p.screenshot(path=str(OUT/'mobile-named-axis-microscope.png'),full_page=True,animations='disabled')
   p.locator('[data-preset="trainable_swiglu"]').click()
   p.locator('#studyMode').select_option('numeric');p.locator('#run').click()
   p.wait_for_function('window.SeetheTruthMicroscopeWorkbench?.stageCount===10')
   p.locator('#computeMicroscope .cmp-step').last.click()
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   self.assertIn('TRAINING LOSS',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   self.assertIn('停止梯度',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   p.screenshot(path=str(OUT/'mobile-training-loss-adjoint.png'),full_page=True,animations='disabled')
   p.locator('[data-preset="trainable_moe"]').click()
   p.locator('#studyMode').select_option('numeric');p.locator('#run').click()
   for button in p.locator('#computeMicroscope .cmp-step').all():
    button.click()
    if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')=='expert_gate':
     break
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'expert_gate')
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   self.assertIn('expert_channel_product',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   self.assertIn('DISCRETE ROUTING',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   self.assertNotIn('反向路径拒绝',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   p.screenshot(path=str(OUT/'mobile-moe-expert-coordinate-jacobian.png'),full_page=True,animations='disabled')
   p.locator('[data-preset="nd_cached_gqa_decode"]').click()
   p.locator('#studyMode').select_option('numeric');p.locator('#run').click()
   p.wait_for_function('window.SeetheTruthMicroscopeWorkbench?.stageCount>17')
   for button in p.locator('#computeMicroscope .cmp-step').all():
    button.click()
    if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')=='q_rope':
     break
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'q_rope')
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   self.assertIn('adjacent_rope_pair',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   self.assertNotIn('反向路径拒绝',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   p.screenshot(path=str(OUT/'mobile-cached-gqa-rope-coordinate.png'),full_page=True,animations='disabled')
   p.locator('[data-preset="nd_wan_video_rope_attention"]').click()
   p.locator('#studyMode').select_option('numeric');p.locator('#run').click()
   p.wait_for_function('window.SeetheTruthMicroscopeWorkbench?.stageCount===10')
   for button in p.locator('#computeMicroscope .cmp-step').all():
    button.click()
    if p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode')=='k_video_rope':
     break
   self.assertEqual(p.evaluate('window.SeetheTruthMicroscopeWorkbench.visibleNode'),'k_video_rope')
   p.locator('#computeMicroscope .cmp-out .cmp-cell').nth(2).click()
   p.locator('#computeMicroscope .cmp-vjp-trigger').click()
   self.assertIn('wan_video_3d_rope',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   self.assertIn('axis=H',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   self.assertNotIn('反向路径拒绝',p.locator('#computeMicroscope .cmp-vjp-result').inner_text())
   p.screenshot(path=str(OUT/'mobile-wan-video-rope-fhw.png'),full_page=True,animations='disabled')
   self.assertEqual(errors,[])
  finally:p.close()
if __name__=='__main__':unittest.main()
