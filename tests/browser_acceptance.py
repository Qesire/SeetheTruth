"""Browser acceptance with simulated graph and deterministic tensor telemetry.

No model weights, no remote sites, no untrusted Python imports. Run with
python tests/browser_acceptance.py after installing Playwright Chromium.
"""
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas_server import Handler


def synthetic_telemetry():
    matrix = [[-2.0, 0.0, 2.0], [1.0, None, -1.0]]
    return {
        "schema": "seethetruth-telemetry-v1",
        "name": "Static acceptance dataset",
        "meta": {"measurement": "DETERMINISTIC_TEST_FIXTURE_NOT_MEASURED",
                 "gpu_activity": False, "tensor_values_opt_in": True,
                 "notes": ["Known matrix and histogram bins, not a model profile"]},
        "modules": [{"name": "block.attn", "type": "SyntheticAttention",
                     "calls": 2, "host_wall_ms": 4.5,
                     "last_outputs": [{"shape": [1, 2, 3], "stride": [6, 3, 1],
                                       "dtype": "float32"}]}],
        "operators": [
            {"name": "aten::mm", "count": 3, "self_cpu_ms": 2.1, "self_device_ms": None},
            {"name": "aten::mul", "count": 1, "self_cpu_ms": 0.5, "self_device_ms": None},
        ],
        "snapshots": [{
            "module": "block.attn",
            "tensor": {"shape": [1, 2, 3], "stride": [6, 3, 1], "dtype": "float32",
                       "device": "cpu", "numel": 6, "logical_bytes": 24,
                       "contiguous": True},
            "summary": {
                "population": 6, "sampled": 6, "finite_count": 5,
                "nonfinite_count": 1, "sample_zero_fraction": 1/6,
                "statistics": {"min": -2.0, "max": 2.0, "mean": 0.0,
                               "std": 1.41421356237},
                "histogram": {"bins": [1] + [0]*5 + [1] + [0]*5 + [1] + [0]*5 + [1] + [0]*4 + [1],
                              "domain": [-2.0, 2.0]},
                "matrix": matrix,
                "matrix_original_shape": [2, 3],
                "matrix_stride": [1, 1],
                "leading_indices": [0],
            },
        }],
        "gpu_memory": None,
    }


def synthetic_graph():
    return {
        "schema": "model-atlas-graph-v1",
        "name": "Known structure / fixture",
        "meta": {"analysis": "runtime-torch-dispatch", "runtime_validated": True},
        "nodes": [
            {"id": "root", "label": "Root", "type": "head",
             "purpose": "Known root", "module": "block.attn", "parent": None,
             "formula": "f(x)"},
            {"id": "inner", "label": "MatMul", "type": "attn",
             "parent": "root", "purpose": "Known matrix node",
             "formula": "Y=AW", "input": "[1,2,3]", "output": "[1,2,3]"},
        ],
        "edges": [{"from": "root", "to": "inner", "kind": "hierarchy"}],
    }


class BrowserAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:" + str(cls.server.server_port)
        cls.engine = sync_playwright().start()
        cls.browser = cls.engine.chromium.launch(headless=True, args=["--no-sandbox"])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.engine.stop()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 1570, "height": 950})
        self.errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.dialogs = []
        self.page.on("dialog", lambda d: (self.dialogs.append(d.message), d.accept()))

    def tearDown(self):
        self.assertFalse(self.errors, f"JavaScript runtime errors: {self.errors}")
        self.page.close()

    def attach(self, selector, name, data):
        self.page.locator(selector).set_input_files({
            "name": name, "mimeType": "application/json",
            "buffer": json.dumps(data, ensure_ascii=False).encode("utf8"),
        })

    def load_atlas(self):
        self.page.goto(self.base + "/", wait_until="domcontentloaded")
        self.page.wait_for_function("document.querySelector('#scene')?.width>0")

    def load_lab(self):
        self.page.goto(self.base + "/learn.html", wait_until="domcontentloaded")
        self.page.wait_for_function("document.querySelectorAll('#blocks .block').length > 0")

    def test_learning_lab_ffn_and_attention_stepwise_numeric_execution(self):
        self.load_lab()
        self.assertIn("TEACHING SIMULATION", self.page.locator("body").inner_text())
        self.assertEqual(self.page.locator("#blocks .block").count(), 4)
        self.page.locator("#run").click()
        self.assertIn("1 / 4", self.page.locator("#stepInfo").inner_text())
        self.assertIn("shape = [3, 4]", self.page.locator("#shape").inner_text())
        self.page.locator("#next").click()
        self.assertIn("2 / 4", self.page.locator("#stepInfo").inner_text())
        self.assertIn("shape = [3, 8]", self.page.locator("#shape").inner_text())
        self.assertGreater(self.page.locator("#matrix .cell").count(), 0)
        self.page.locator('[data-preset="attention"]').click()
        self.assertEqual(self.page.locator("#blocks .block").count(), 6)
        self.page.locator("#run").click()
        for _ in range(4):
            self.page.locator("#next").click()
        self.assertIn("5 / 6", self.page.locator("#stepInfo").inner_text())
        self.assertIn("缩放点积注意力", self.page.locator("#inspectorTitle").inner_text())
        self.assertIn("3×3", self.page.locator("#intermediates").inner_text())
        self.page.locator("#intermediates details").first.click()
        self.assertEqual(self.page.locator("#intermediates details").first.locator(".cell").count(), 9)

    def test_learning_lab_palette_connection_and_parameter_edit(self):
        self.load_lab()
        self.page.locator('[data-preset="blank"]').click()
        self.assertEqual(self.page.locator("#blocks .block").count(), 1)
        self.page.locator('[data-add="linear"]').click()
        self.assertEqual(self.page.locator("#blocks .block").count(), 2)
        self.page.locator("#run").click()
        self.assertIn("还没有连接", self.page.locator("#status").inner_text())
        self.page.locator('[data-out="x"]').click()
        self.page.locator('[data-node^="lab"] [data-in][data-port="x"]').click()
        self.page.locator("#run").click()
        self.assertIn("1 / 2", self.page.locator("#stepInfo").inner_text())
        self.page.locator("#next").click()
        self.assertIn("shape = [3, 4]", self.page.locator("#shape").inner_text())
        self.page.locator('[data-node^="lab"] .block-head').click()
        self.page.locator('[data-param="out"]').fill("6")
        self.page.locator('[data-param="out"]').dispatch_event("change")
        self.page.locator("#run").click()
        self.page.locator("#next").click()
        self.assertIn("shape = [3, 6]", self.page.locator("#shape").inner_text())
        self.page.locator("#undo").click()
        self.page.locator("#run").click()
        self.page.locator("#next").click()
        self.assertIn("shape = [3, 4]", self.page.locator("#shape").inner_text())

    def test_learning_lab_break_and_repair_challenge(self):
        self.load_lab()
        self.page.locator("#challenge").click()
        self.assertIn("已拆除", self.page.locator("#status").inner_text())
        self.page.locator("#run").click()
        self.assertIn("还没有连接", self.page.locator("#status").inner_text())
        self.page.locator('[data-out="act"]').click()
        self.page.locator('[data-in="down"][data-port="x"]').click()
        self.page.locator("#run").click()
        self.assertIn("1 / 4", self.page.locator("#stepInfo").inner_text())
        for _ in range(3):
            self.page.locator("#next").click()
        self.assertIn("shape = [3, 4]", self.page.locator("#shape").inner_text())

    def test_learning_lab_import_rejects_untrusted_shapes_transactionally(self):
        self.load_lab()
        good = self.page.locator("#blocks .block").count()
        self.attach("#load", "unsafe.json", {
            "schema": "seethetruth-learning-lab-v1", "title": "<script>alert(1)</script>",
            "nodes": [{"id": "evil", "kind": "<img src=x onerror=alert(1)>",
                       "x": 3, "y": 5, "params": {}}],
            "edges": []})
        self.assertIn("导入被拒绝", self.page.locator("#status").inner_text())
        self.assertEqual(self.page.locator("#blocks .block").count(), good)
        self.assertIn("任务 01", self.page.locator("#workspaceTitle").inner_text())

    def test_five_level_presets_render_and_inspector(self):
        self.load_atlas()
        for architecture in ("gqa", "gpt", "moe", "dit"):
            self.page.locator(f'[data-preset="{architecture}"]').click()
            self.assertTrue(self.page.locator("#modelChip").inner_text())
            for level in ("macro", "block", "operator", "matrix", "tensor"):
                button = self.page.locator(f'[data-level="{level}"]')
                self.assertFalse(button.is_disabled())
                button.click()
                self.assertIn("active", button.get_attribute("class"))
                self.assertTrue(self.page.locator("#inspectContent").inner_text().strip())
                self.assertGreater(self.page.locator("#scene").evaluate("(e)=>e.width"), 0)

    def test_mechanism_panel_shows_math_goal_limits_and_literature(self):
        self.load_atlas()
        self.page.locator('[data-preset="gqa"]').click()
        self.page.locator('[data-level="block"]').click()
        panel = self.page.locator("#academicConsensus")
        self.assertTrue(panel.is_visible())
        text = panel.inner_text()
        self.assertIn("Grouped-Query Attention", text)
        self.assertIn("预期终端效果", text)
        self.assertIn("适用前提", text)
        self.assertIn("如何对照实验验证", text)
        links = panel.locator('a[href^="https://"]')
        self.assertGreaterEqual(links.count(), 1)
        self.assertIn("noopener", links.first.get_attribute("rel"))
        self.page.locator('[data-preset="dit"]').click()
        self.page.locator('[data-level="operator"]').click()
        self.assertIn("AdaLN", self.page.locator("#academicConsensus").inner_text())

    def test_mechanism_does_not_guess_unknown_imported_module(self):
        self.load_atlas()
        unknown = synthetic_graph()
        unknown["meta"]["analysis"] = "static-python-ast"
        unknown["meta"]["entry"] = "GenericBlock.forward"
        unknown["nodes"][0]["callee"] = "self.mysterious_transform"
        self.page.locator("#importOpen").click()
        self.page.locator("#jsonText").fill(json.dumps(unknown))
        self.page.locator("#importApply").click()
        self.assertEqual(self.page.locator("#academicConsensus").count(), 0)
        known = synthetic_graph()
        known["meta"]["analysis"] = "static-python-ast"
        known["meta"]["entry"] = "GenericBlock.forward"
        known["nodes"][0]["callee"] = "self.proj"
        known["nodes"][0]["module_constructor"] = "Linear"
        self.page.locator("#importOpen").click()
        self.page.locator("#jsonText").fill(json.dumps(known))
        self.page.locator("#importApply").click()
        self.assertIn("线性投影", self.page.locator("#academicConsensus").inner_text())
        self.page.locator('[data-nav="node"][data-scope="root"]').click()
        # Inside the child scope the selected node is a different computation;
        # its parent literature role must not leak into the child.
        self.assertEqual(self.page.locator("#academicConsensus").count(), 0)
        self.page.locator('[data-nav="home"]').click()
        self.assertIn("线性投影", self.page.locator("#academicConsensus").inner_text())
        self.assertIn("尚未验证", self.page.locator("#academicConsensus").inner_text())

    def test_training_inference_and_rl_mode_switch(self):
        self.load_atlas()
        for mode in ("train", "infer", "rl", "model"):
            self.page.locator(f'[data-workflow="{mode}"]').click()
            self.assertIn("active", self.page.locator(f'[data-workflow="{mode}"]').get_attribute("class"))
            self.assertGreater(self.page.locator("#scene").evaluate("(e)=>e.width"), 0)
        self.page.locator('[data-workflow="rl"]').click()
        self.page.locator("#rlSelector").select_option("ppo")
        self.assertIn("PPO", self.page.locator("#stageTitle").inner_text().upper())
        self.page.locator("#rlSelector").select_option("grpo")
        self.assertIn("GRPO", self.page.locator("#stageTitle").inner_text().upper())

    def test_config_rejects_invalid_head_count_and_accepts_valid(self):
        self.load_atlas()
        self.page.locator("#importOpen").click()
        self.page.locator("#jsonText").fill(json.dumps({
            "model_type": "llama", "hidden_size": 96, "num_hidden_layers": 2,
            "num_attention_heads": 7, "num_key_value_heads": 1,
        }))
        self.page.locator("#importApply").click()
        self.assertIn("整除", self.page.locator("#importError").inner_text())
        self.page.locator("#jsonText").fill(json.dumps({
            "model_type": "llama", "hidden_size": 96, "num_hidden_layers": 2,
            "num_attention_heads": 8, "num_key_value_heads": 2,
            "intermediate_size": 256,
        }))
        self.page.locator("#importApply").click()
        self.assertFalse(self.page.locator("#modal").evaluate("(e)=>e.classList.contains('open')"))
        self.assertIn("96", self.page.locator("#shapeChip").inner_text())

    def test_source_matrix_tensor_tabs_show_only_ast_evidence(self):
        # Real non-executing parser output, not a handwritten simulated scene.
        import tempfile
        from atlas_parser import parse_paths
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "mini_source.py"
            path.write_text(
                "class ToyAttention:\n"
                "    def forward(self, x):\n"
                "        y = x.transpose(-1, -2)\n"
                "        return y @ x\n"
                "class Block:\n"
                "    def __init__(self):\n"
                "        self.attn = ToyAttention()\n"
                "    def forward(self, x):\n"
                "        return self.attn(x)\n",
                encoding="utf8")
            graph = parse_paths([str(path)], entry="Block.forward", max_depth=2)
        self.assertFalse(graph["meta"]["shape_verified"])
        self.assertTrue(any(n["label"] == "MatMult" for n in graph["nodes"]))
        self.load_atlas()
        self.page.locator("#importOpen").click()
        self.page.locator("#jsonText").fill(json.dumps(graph, ensure_ascii=False))
        self.page.locator("#importApply").click()
        self.assertFalse(self.page.locator('[data-level="matrix"]').is_disabled())
        self.assertFalse(self.page.locator('[data-level="tensor"]').is_disabled())
        self.page.locator('[data-level="matrix"]').click()
        self.assertIn("矩阵证据", self.page.locator("#stageTitle").inner_text())
        self.assertIn("1 对应运算", self.page.locator("#nodeCount").inner_text())
        self.assertIn("NO VERIFIED SHAPES", self.page.locator("#chainLabel").inner_text())
        self.page.locator('[data-level="tensor"]').click()
        self.assertIn("张量操作", self.page.locator("#stageTitle").inner_text())
        self.assertIn("2 对应运算", self.page.locator("#nodeCount").inner_text())
        attn = next(n for n in graph["nodes"] if n.get("callee") == "self.attn")
        self.page.locator('[data-nav="node"][data-scope="' + attn["id"] + '"]').click()
        self.assertIn("attn", self.page.locator("#stageTitle").inner_text())
        self.page.locator('[data-level="matrix"]').click()
        self.assertIn("1 对应运算", self.page.locator("#nodeCount").inner_text())
        self.page.locator('[data-nav="home"]').click()
        self.page.locator('[data-level="macro"]').click()
        self.assertIn("整体", self.page.locator("#stageTitle").inner_text())

    def test_hierarchy_and_cycle_rejection(self):
        self.load_atlas()
        self.page.locator("#importOpen").click()
        fixture = synthetic_graph()
        self.page.locator("#jsonText").fill(json.dumps(fixture))
        self.page.locator("#importApply").click()
        self.assertIn("Known structure", self.page.locator("#stageTitle").inner_text())
        self.page.locator('[data-nav="node"][data-scope="root"]').click()
        self.assertIn("Root", self.page.locator("#stageTitle").inner_text())
        self.page.locator('[data-nav="back"]').click()
        self.assertIn("Known structure", self.page.locator("#stageTitle").inner_text())
        self.page.locator("#importOpen").click()
        corrupt = synthetic_graph()
        corrupt["nodes"][0]["parent"] = "inner"
        self.page.locator("#jsonText").fill(json.dumps(corrupt))
        self.page.locator("#importApply").click()
        self.assertTrue(self.page.locator("#importError").inner_text().strip())

    def test_failed_import_is_transactional(self):
        self.load_atlas()
        self.page.locator("#importOpen").click()
        self.page.locator("#jsonText").fill(json.dumps(synthetic_graph()))
        self.page.locator("#importApply").click()
        self.assertIn("Known structure", self.page.locator("#stageTitle").inner_text())
        self.page.locator("#importOpen").click()
        self.page.locator("#jsonText").fill(json.dumps({
            "model_type": "llama", "hidden_size": 96,
            "num_hidden_layers": 2, "num_attention_heads": 7,
        }))
        self.page.locator("#importApply").click()
        self.assertIn("整除", self.page.locator("#importError").inner_text())
        self.page.locator("#modalClose").click()
        # A failed replacement must not destroy the previous import's
        # internal graph data, so normal navigation must remain possible.
        self.page.locator('[data-nav="node"][data-scope="root"]').click()
        self.assertIn("Root", self.page.locator("#stageTitle").inner_text())

    def test_source_runtime_attribution_evidence_and_unmatched_ui(self):
        self.load_atlas()
        graph = synthetic_graph()
        graph["nodes"][0]["source_attribution"] = {
            "status": "matched", "static_node_id": "n001",
            "static_source": {"file": "toy.py", "line": 8, "sha256": "a" * 64},
            "evidence": "exact_file_sha256_line_and_module_type",
        }
        self.page.locator("#importOpen").click()
        self.page.locator("#jsonText").fill(json.dumps(graph, ensure_ascii=False))
        self.page.locator("#importApply").click()
        self.assertIn("源码映射已核验", self.page.locator("#inspectContent").inner_text())
        self.assertIn("toy.py:8", self.page.locator("#inspectContent").inner_text())
        graph["nodes"][0]["source_attribution"] = {"status": "source_mismatch"}
        self.page.locator("#importOpen").click()
        self.page.locator("#jsonText").fill(json.dumps(graph, ensure_ascii=False))
        self.page.locator("#importApply").click()
        self.assertIn("源码尚未核验", self.page.locator("#inspectContent").inner_text())
        self.assertNotIn("源码映射已核验", self.page.locator("#inspectContent").inner_text())

    def test_telemetry_per_invocation_cost_ui(self):
        self.page.goto(self.base + "/telemetry.html", wait_until="domcontentloaded")
        fixture = synthetic_telemetry()
        fixture["modules"][0]["exclusive_self_cpu_ms"] = 0.12
        fixture["modules"][0]["inclusive_attributed_cpu_ms"] = 0.78
        fixture["module_calls"] = [{
            "name": "block.attn", "type": "SyntheticAttention",
            "invocation_index": 1, "invocation_key": "block.attn#1",
            "exclusive_self_cpu_ms": 0.12,
            "inclusive_attributed_cpu_ms": 0.78,
        }]
        self.attach("#file", "same-pass.json", fixture)
        self.page.wait_for_function(
            "document.querySelector('#modules')?.textContent.includes('独占 ATen')")
        self.assertIn("0.12", self.page.locator("#modules").inner_text())
        self.assertIn("0.78", self.page.locator("#modules").inner_text())
        self.page.locator("#modules .module").first.click()
        self.assertIn("block.attn#1", self.page.locator("#moduleCalls").inner_text())

    def test_same_pass_module_profiler_cost_and_unmatched_state(self):
        self.load_atlas()
        fixture = synthetic_graph()
        fixture["nodes"][0]["phase"] = "forward"
        fixture["nodes"][0]["same_pass_profile"] = {
            "status": "matched", "invocation_key": "block.attn#1",
            "run_id": "simulated-in-browser-test",
            "exclusive_self_cpu_ms": 0.125,
            "inclusive_attributed_cpu_ms": 0.750,
            "exclusive_operator_events": 2, "inclusive_operator_events": 8,
            "host_wall_ms": 1.2, "exclusive_self_device_ms": None,
        }
        self.page.locator("#importOpen").click()
        self.page.locator("#jsonText").fill(json.dumps(fixture))
        self.page.locator("#importApply").click()
        displayed = self.page.locator("#inspectContent").inner_text()
        self.assertIn("SAME-PASS PROFILER", displayed)
        self.assertIn("0.125", displayed)
        self.assertIn("0.75", displayed)
        self.assertIn("不", displayed)
        fixture["nodes"][0]["same_pass_profile"] = {
            "status": "tensor_signature_mismatch"}
        self.page.locator("#importOpen").click()
        self.page.locator("#jsonText").fill(json.dumps(fixture))
        self.page.locator("#importApply").click()
        displayed = self.page.locator("#inspectContent").inner_text()
        self.assertIn("tensor_signature_mismatch", displayed)
        self.assertNotIn("0.125", displayed)

    def test_runtime_module_exact_join(self):
        self.load_atlas()
        self.page.locator("#importOpen").click()
        self.page.locator("#jsonText").fill(json.dumps(synthetic_graph()))
        self.page.locator("#importApply").click()
        self.attach("#telemetryFile", "known.json", synthetic_telemetry())
        self.page.wait_for_function("document.querySelector('#inspectContent')?.textContent.includes('4.5')")
        self.assertIn("4.5", self.page.locator("#inspectContent").inner_text())
        self.assertIn("PROFILE JOIN", self.page.locator("#inspectContent").inner_text())

    def test_telemetry_visual_modes_and_known_metadata(self):
        self.page.goto(self.base + "/telemetry.html", wait_until="domcontentloaded")
        self.attach("#file", "known.json", synthetic_telemetry())
        self.assertFalse(self.page.locator("#body").is_hidden())
        self.assertIn("aten::mm", self.page.locator("#operators").inner_text())
        self.assertIn("block.attn", self.page.locator("#selTitle").inner_text())
        self.assertIn("1, 2, 3", self.page.locator("#details").inner_text())
        self.assertIn("16.67%", self.page.locator("#details").inner_text())
        for mode in ("signed", "magnitude", "zero"):
            self.page.locator("#heatmode").select_option(mode)
            self.assertGreater(self.page.locator("#heatmap").evaluate("(e)=>e.width"), 0)
        for mode in ("hist", "rowenergy", "colabs"):
            self.page.locator("#histmode").select_option(mode)
            self.assertGreater(self.page.locator("#hist").evaluate("(e)=>e.width"), 0)
        self.assertIn("DETERMINISTIC_TEST_FIXTURE", self.page.locator("#evidence").inner_text())

    def test_malformed_tensor_data_rejected_before_canvas_render(self):
        self.page.goto(self.base + "/telemetry.html", wait_until="domcontentloaded")
        cases = []
        bad = synthetic_telemetry()
        bad["snapshots"][0]["tensor"]["shape"] = "1x2x3"
        cases.append(("shape", bad))
        bad = synthetic_telemetry()
        bad["snapshots"][0]["summary"]["matrix"][0][0] = "<img src=x onerror=alert(1)>"
        cases.append(("cell", bad))
        bad = synthetic_telemetry()
        bad["snapshots"][0]["summary"]["histogram"]["bins"][0] = 999
        cases.append(("histogram", bad))
        bad = synthetic_telemetry()
        bad["operators"][0]["self_cpu_ms"] = -5
        cases.append(("negative time", bad))
        bad = synthetic_telemetry()
        bad["meta"]["notes"] = "not an array"
        cases.append(("malformed meta", bad))
        for label, payload in cases:
            with self.subTest(label=label):
                with self.page.expect_event("dialog") as event:
                    self.attach("#file", label + ".json", payload)
                self.assertIn("加载失败", event.value.message)
                self.assertFalse(self.page.locator("#empty").is_hidden())
        self.attach("#file", "recover.json", synthetic_telemetry())
        self.page.wait_for_function("!document.querySelector('#body').hidden")
        self.assertIn("block.attn", self.page.locator("#selTitle").inner_text())

    def test_matrix_visual_modes_produce_distinct_pixels(self):
        self.page.goto(self.base + "/telemetry.html", wait_until="domcontentloaded")
        self.attach("#file", "known.json", synthetic_telemetry())
        self.page.wait_for_function("!document.querySelector('#body').hidden")
        draw = lambda selector: self.page.locator(selector).evaluate("(e)=>e.toDataURL('image/png')")
        signed = draw("#heatmap")
        self.page.locator("#heatmode").select_option("zero")
        zero = draw("#heatmap")
        self.assertNotEqual(signed, zero)
        hist = draw("#hist")
        self.page.locator("#histmode").select_option("rowenergy")
        rowenergy = draw("#hist")
        self.page.locator("#histmode").select_option("colabs")
        colabs = draw("#hist")
        self.assertNotEqual(hist, rowenergy)
        self.assertNotEqual(rowenergy, colabs)

    def test_telemetry_invalid_schema_shows_error_without_breakage(self):
        self.page.goto(self.base + "/telemetry.html", wait_until="domcontentloaded")
        with self.page.expect_event("dialog") as event:
            self.attach("#file", "invalid.json", {"schema": "random", "operators": []})
        self.assertIn("加载失败", event.value.message)
        self.assertFalse(self.page.locator("#empty").is_hidden())
        self.page.locator("#demo").click()
        self.assertIn("SYNTHETIC", self.page.locator("#evidence").inner_text())
        self.assertGreater(self.page.locator("#heatmap").evaluate("(e)=>e.width"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
