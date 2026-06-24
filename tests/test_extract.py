import json
import unittest
from pathlib import Path

from llm_analyzer.extract import extract_architecture_from_config
from llm_analyzer.kernel_graph import build_kernel_graph
from llm_analyzer.render import render_mermaid_attention, render_mermaid_mlp, render_mermaid_moe


class ExtractArchitectureTest(unittest.TestCase):
    def test_llama4_multimodal_moe_config(self):
        fixture = Path(__file__).parent / "fixtures" / "llama4_config.json"
        config = json.loads(fixture.read_text(encoding="utf-8"))

        arch = extract_architecture_from_config(config, "meta-llama/Llama-4-Maverick-17B-128E")

        self.assertEqual(arch.model_type, "llama4")
        self.assertEqual(arch.summary["layers"], 48)
        self.assertEqual(arch.summary["hidden_size"], 5120)
        self.assertEqual(arch.summary["attention_heads"], 40)
        self.assertEqual(arch.summary["kv_heads"], 8)
        self.assertEqual(arch.summary["moe"]["experts"], 128)
        self.assertEqual(arch.summary["moe"]["activated_experts"], 1)
        self.assertEqual(arch.summary["moe"]["moe_layers"], 24)
        self.assertEqual(arch.summary["moe"]["dense_mlp_layers"], 24)
        self.assertEqual(arch.text_decoder["layer_type"], "Llama4TextDecoderLayer")
        self.assertEqual(len(arch.layers), 48)
        self.assertEqual(arch.layers[0].layer_type, "Llama4TextDecoderLayer[DenseMLP]")
        self.assertEqual(arch.layers[1].layer_type, "Llama4TextDecoderLayer[MoE]")
        self.assertIsNotNone(arch.vision_encoder)

    def test_detailed_mermaid_renderers(self):
        fixture = Path(__file__).parent / "fixtures" / "llama4_config.json"
        config = json.loads(fixture.read_text(encoding="utf-8"))
        arch = extract_architecture_from_config(config, "meta-llama/Llama-4-Maverick-17B-128E")

        attention = render_mermaid_attention(arch, 0)
        dense_mlp = render_mermaid_mlp(arch, 0)
        moe = render_mermaid_moe(arch, 1)

        self.assertIn("scores = Q K^T * scale", attention)
        self.assertIn("gate_proj", dense_mlp)
        self.assertIn("router linear", moe)
        self.assertIn("shared expert MLP", moe)

    def test_kernel_flow_graphs_are_flat_and_named(self):
        fixture = Path(__file__).parent / "fixtures" / "llama4_config.json"
        config = json.loads(fixture.read_text(encoding="utf-8"))
        arch = extract_architecture_from_config(config, "meta-llama/Llama-4-Maverick-17B-128E")

        dense_layer = build_kernel_graph(arch, "layer", 0)
        moe_layer = build_kernel_graph(arch, "layer", 1)
        moe_detail = build_kernel_graph(arch, "moe", 1)

        dense_ops = [node.op_type for node in dense_layer.nodes]
        moe_ops = [node.op_type for node in moe_layer.nodes]
        detail_names = [node.name for node in moe_detail.nodes]

        self.assertIn("RoPE", dense_ops)
        self.assertIn("MatMulQK", dense_ops)
        self.assertIn("Linear", dense_ops)
        self.assertIn("RouterLinear", moe_ops)
        self.assertIn("GroupedGEMM", moe_ops)
        self.assertIn("router", detail_names)
        self.assertIn("add_routed_shared", detail_names)
        self.assertIn("formula", dense_layer.nodes[0].attrs)
        self.assertIn("input_links", dense_layer.nodes[0].attrs)
        self.assertIn("output_links", dense_layer.nodes[0].attrs)
        self.assertEqual(dense_layer.tensors["attention_scores"].shape, ["batch", 40, "sequence", "kv_sequence"])

    def test_deepseek_v4_mla_hc_and_moe_graphs(self):
        fixture = Path(__file__).parent / "fixtures" / "deepseek_v4_config.json"
        config = json.loads(fixture.read_text(encoding="utf-8"))
        arch = extract_architecture_from_config(config, "deepseek-ai/DeepSeek-V4-Pro")

        self.assertEqual(arch.model_type, "deepseek_v4")
        self.assertEqual(arch.text_decoder["layer_type"], "DeepseekV4Block")
        self.assertEqual(arch.summary["hyper_connections"]["hc_mult"], 4)
        self.assertEqual(arch.text_decoder["attention"]["type"], "MLA sparse attention")
        self.assertEqual(arch.text_decoder["attention"]["qk_nope_head_dim"], 448)
        self.assertEqual(arch.layers[0].layer_type, "DeepseekV4Block[MoE]")
        self.assertTrue(arch.layers[0].components[7].details["hash_routing"])
        self.assertEqual(arch.layers[2].components[3].details["compress_ratio"], 4)

        layer0 = build_kernel_graph(arch, "layer", 0)
        layer2_attention = build_kernel_graph(arch, "attention", 2)
        layer2_moe = build_kernel_graph(arch, "moe", 2)

        layer0_ops = [node.op_type for node in layer0.nodes]
        attention_ops = [node.op_type for node in layer2_attention.nodes]
        moe_ops = [node.op_type for node in layer2_moe.nodes]

        self.assertIn("HyperConnectionPre", layer0_ops)
        self.assertIn("HashRouteLookup", layer0_ops)
        self.assertIn("SparseAttention", attention_ops)
        self.assertIn("GatedKVCompression", attention_ops)
        self.assertIn("SparseIndexScore", attention_ops)
        self.assertIn("FP4GroupedGEMM", moe_ops)
        self.assertIn("ApplyRouterWeights", moe_ops)
        self.assertEqual(layer2_attention.tensors["q_sparse"].shape, ["batch", "sequence", 128, 512])

    def test_glm_moe_dsa_indexshare_graphs(self):
        fixture = Path(__file__).parent / "fixtures" / "glm_moe_dsa_config.json"
        config = json.loads(fixture.read_text(encoding="utf-8"))
        arch = extract_architecture_from_config(config, "zai-org/GLM-5.2")

        self.assertEqual(arch.model_type, "glm_moe_dsa")
        self.assertEqual(arch.text_decoder["layer_type"], "GlmMoeDsaDecoderLayer")
        self.assertEqual(arch.summary["moe"]["moe_layers"], 5)
        self.assertEqual(arch.layers[0].layer_type, "GlmMoeDsaDecoderLayer[DenseMLP]")
        self.assertEqual(arch.layers[3].layer_type, "GlmMoeDsaDecoderLayer[MoE]")
        self.assertEqual(arch.layers[3].components[2].details["indexer_type"], "shared")
        self.assertEqual(arch.layers[6].components[2].details["indexer_type"], "full")
        self.assertEqual(arch.layers[3].components[6].details["intermediate_size"], 2048)
        self.assertEqual(arch.text_decoder["attention"]["qk_nope_head_dim"], 192)

        dense_layer = build_kernel_graph(arch, "layer", 0)
        shared_attention = build_kernel_graph(arch, "attention", 3)
        full_attention = build_kernel_graph(arch, "attention", 6)
        moe = build_kernel_graph(arch, "moe", 3)

        dense_ops = [node.op_type for node in dense_layer.nodes]
        shared_ops = [node.op_type for node in shared_attention.nodes]
        full_ops = [node.op_type for node in full_attention.nodes]
        moe_ops = [node.op_type for node in moe.nodes]

        self.assertIn("DynamicSparseAttention", dense_ops)
        self.assertIn("IndexShareReuse", shared_ops)
        self.assertIn("SparseIndexScore", full_ops)
        self.assertIn("RouterBiasAdd", moe_ops)
        self.assertIn("WeightedExpertSum", moe_ops)
        self.assertEqual(full_attention.tensors["q_sparse"].shape, ["batch", "sequence", 64, 256])

    def test_kimi_k25_multimodal_mla_and_int4_moe_graphs(self):
        fixture = Path(__file__).parent / "fixtures" / "kimi_k25_config.json"
        config = json.loads(fixture.read_text(encoding="utf-8"))
        arch = extract_architecture_from_config(config, "moonshotai/Kimi-K2.5")

        self.assertEqual(arch.model_type, "kimi_k25")
        self.assertEqual(arch.text_decoder["layer_type"], "KimiK25DeepseekDecoderLayer")
        self.assertEqual(arch.summary["moe"]["moe_layers"], 3)
        self.assertEqual(arch.summary["moe"]["dense_mlp_layers"], 1)
        self.assertEqual(arch.summary["moe"]["expert_dtype"], "int4-packed")
        self.assertEqual(arch.vision_encoder["layers"], 27)
        self.assertEqual(arch.vision_encoder["projector_type"], "patchmerger")
        self.assertEqual(arch.layers[0].layer_type, "KimiK25DeepseekDecoderLayer[DenseMLP]")
        self.assertEqual(arch.layers[1].layer_type, "KimiK25DeepseekDecoderLayer[MoE]")
        self.assertEqual(arch.layers[1].components[2].details["qk_head_dim"], 192)
        self.assertEqual(arch.layers[1].components[2].details["kv_lora_rank"], 512)
        self.assertEqual(arch.layers[1].components[6].details["expert_dtype"], "int4-packed")

        attention = build_kernel_graph(arch, "attention", 1)
        moe = build_kernel_graph(arch, "moe", 1)
        dense = build_kernel_graph(arch, "layer", 0)

        attention_ops = [node.op_type for node in attention.nodes]
        moe_ops = [node.op_type for node in moe.nodes]
        dense_ops = [node.op_type for node in dense.nodes]
        attention_diagram = render_mermaid_attention(arch, 1)

        self.assertIn("SplitKVLatent", attention_ops)
        self.assertIn("MatMulQK", attention_ops)
        self.assertIn("MatMulPV", attention_ops)
        self.assertNotIn("DynamicSparseAttention", attention_ops)
        self.assertIn("Int4GroupedGEMM", moe_ops)
        self.assertIn("RouterBiasAdd", moe_ops)
        self.assertIn("WeightedExpertSum", moe_ops)
        self.assertIn("scores = Q K^T", attention_diagram)
        self.assertIn("Linear", dense_ops)
        self.assertEqual(attention.tensors["q_dense"].shape, ["batch", "sequence", 64, 192])
        self.assertEqual(attention.tensors["v_heads"].shape, ["batch", "sequence", 64, 128])
        self.assertEqual(moe.tensors["routed_gate"].shape, ["tokens", 8, 2048])


if __name__ == "__main__":
    unittest.main()
