import json
from pathlib import Path
from typing import Any, Dict, List

from .kernel_graph import KernelGraph, TensorSpec, build_kernel_graph
from .schema import Architecture


class OnnxExportError(RuntimeError):
    pass


def export_onnx_kernel_graph(
    architecture: Architecture,
    output_path: Path,
    level: str,
    layer_index: int,
) -> None:
    graph = build_kernel_graph(architecture, level=level, layer_index=layer_index)
    _write_onnx(graph, output_path)


def _write_onnx(graph: KernelGraph, output_path: Path) -> None:
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError as exc:
        raise OnnxExportError(
            "ONNX export requires the optional 'onnx' dependency. "
            "Install it with: python -m pip install -e '.[onnx]'"
        ) from exc

    visible_names = _visible_tensor_names(graph)
    visible_specs = {
        name: _visible_tensor_spec(spec, visible_names[name])
        for name, spec in graph.tensors.items()
    }
    boundary_names = {visible_names[spec.name] for spec in graph.inputs + graph.outputs}

    graph_inputs = [_value_info(helper, TensorProto, visible_specs[spec.name]) for spec in graph.inputs]
    graph_outputs = [_value_info(helper, TensorProto, visible_specs[spec.name]) for spec in graph.outputs]
    value_infos = [
        _value_info(helper, TensorProto, spec)
        for spec in sorted(visible_specs.values(), key=lambda item: item.name)
        if spec.name not in boundary_names
    ]

    nodes = [
        helper.make_node(
            node.op_type,
            [visible_names[name] for name in node.inputs],
            [visible_names[name] for name in node.outputs],
            name=node.name,
            domain="llm_analyzer",
            **_onnx_attrs(_node_attrs_with_visible_links(node.attrs, visible_names)),
        )
        for node in graph.nodes
    ]

    onnx_graph = helper.make_graph(
        nodes,
        graph.name,
        graph_inputs,
        graph_outputs,
        initializer=[],
        value_info=value_infos,
    )
    model = helper.make_model(
        onnx_graph,
        producer_name="llm-analyzer",
        doc_string="Metadata-only kernel-flow graph. Contains no model weights.",
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("llm_analyzer", 1),
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))


def _visible_tensor_names(graph: KernelGraph) -> Dict[str, str]:
    used: Dict[str, int] = {}
    visible: Dict[str, str] = {}
    for name, spec in graph.tensors.items():
        base = "%s%s" % (_compact_tensor_alias(name), _compact_shape(spec.shape))
        count = used.get(base, 0) + 1
        used[base] = count
        visible[name] = base if count == 1 else "%s_%d" % (base, count)
    return visible


def _visible_tensor_spec(spec: TensorSpec, visible_name: str) -> TensorSpec:
    return TensorSpec(
        name=visible_name,
        shape=spec.shape,
        dtype=spec.dtype,
        description="Visible link: %s. Original tensor: %s. %s" % (
            visible_name,
            spec.name,
            spec.description,
        ),
    )


def _node_attrs_with_visible_links(attrs: Dict[str, Any], visible_names: Dict[str, str]) -> Dict[str, Any]:
    remapped = dict(attrs)
    remapped["visible_input_links"] = _visible_links(attrs.get("input_links"), visible_names)
    remapped["visible_output_links"] = _visible_links(attrs.get("output_links"), visible_names)
    return remapped


def _visible_links(links: Any, visible_names: Dict[str, str]) -> List[Dict[str, Any]]:
    if not isinstance(links, list):
        return []
    visible = []
    for link in links:
        if not isinstance(link, dict):
            continue
        original_name = link.get("name")
        visible.append(
            {
                "name": visible_names.get(original_name, original_name),
                "original_name": original_name,
                "shape": link.get("shape"),
                "dtype": link.get("dtype"),
                "description": link.get("description"),
            }
        )
    return visible


def _compact_shape(shape: List[Any]) -> str:
    return "[%s]" % ",".join(_compact_dim(dim) for dim in shape)


def _compact_dim(dim: Any) -> str:
    aliases = {
        "batch": "B",
        "sequence": "S",
        "kv_sequence": "KV",
        "sparse_kv_sequence": "SKV",
        "compressed_sequence": "CS",
        "tokens": "T",
        "heads": "H",
        "idx_heads": "IH",
        "kv_heads": "KH",
        "head_dim": "D",
        "qk_dim": "QKD",
        "qk_nope_dim": "QN",
        "qk_rope_dim": "QRD",
        "kv_rank": "KVR",
        "v_dim": "VD",
        "idx_dim": "ID",
        "q_rank": "QR",
        "o_rank": "OR",
        "groups": "G",
        "group_width": "GW",
        "hc": "HC",
        "hidden": "h",
        "intermediate": "I",
        "top_k": "K",
        "index_topk": "IK",
        "heads*head_dim": "HD",
        "heads*qk_dim": "HQK",
        "heads*v_dim": "HVD",
        "heads*(qk_nope+v)": "HQV",
        "kv_rank+qk_rope": "KVRR",
        "qk_nope+v": "QNV",
        "idx_heads*idx_dim": "IHD",
        "kv_heads*head_dim": "KHD",
        "2*intermediate": "2I",
        "2*head_dim": "2D",
        "3*hidden": "3H",
        "groups*o_rank": "GOR",
    }
    return aliases.get(str(dim), str(dim))


def _compact_tensor_alias(name: str) -> str:
    aliases = {
        "layer_input": "x",
        "input_ids": "ids",
        "attn_sink": "sink",
        "attention_input": "x",
        "mlp_input": "x",
        "moe_input": "x",
        "input_layernorm_out": "xnorm",
        "attn_norm_out": "an",
        "ffn_norm_out": "fn",
        "hc_attn_pre_out": "hap",
        "hc_attn_post_out": "hao",
        "hc_ffn_pre_out": "hfp",
        "q_proj_out": "q",
        "qkv_packed": "qkv",
        "q_latent": "ql",
        "q_latent_norm": "qln",
        "q_packed": "qp",
        "q_rope_in": "qRi",
        "k_proj_out": "k",
        "v_proj_out": "v",
        "q_heads": "qh",
        "q_scaled": "qs",
        "q_sparse": "qsp",
        "q_dense": "qden",
        "kv_a_packed": "kva",
        "kv_latent": "kvl",
        "kv_latent_norm": "kvln",
        "kv_b_packed": "kvb",
        "kv_heads": "kvh",
        "k_nope": "kN",
        "k_rope_in": "kRi",
        "k_rope_expanded": "kRe",
        "k_sparse": "ksp",
        "k_dense": "kden",
        "k_heads": "kh",
        "v_heads": "vh",
        "q_rope": "qR",
        "k_rope": "kR",
        "q_nope": "qN",
        "k_nope": "kN",
        "q_norm": "qnorm",
        "k_norm": "knorm",
        "k_cache": "kC",
        "v_cache": "vC",
        "sparse_index_scores": "iscore",
        "sparse_topk_indices": "sidx",
        "sparse_topk_scores": "sscore",
        "k_repeated": "krep",
        "v_repeated": "vrep",
        "attention_scores": "score",
        "attention_scores_alibi": "scoreA",
        "masked_scores": "scoreM",
        "attention_probs": "prob",
        "attention_probs_dropout": "probD",
        "attention_context": "ctx",
        "attention_context_invrope": "ctxR",
        "attention_groups": "grp",
        "attention_o_lora": "olr",
        "attention_o_lora_flat": "olf",
        "attention_merged": "ctxM",
        "attention_output": "attn",
        "kv_compact": "kv",
        "kv_norm": "kvn",
        "kv_pos": "kvp",
        "kv_quant": "kvq",
        "kv_window_cache": "kvw",
        "window_topk_indices": "widx",
        "compress_kv_raw": "ckvr",
        "compress_gate_raw": "cgr",
        "compressed_kv_pool": "ckvp",
        "compressed_kv_norm": "ckvn",
        "compressed_kv_pos": "ckvR",
        "compressed_kv_quant": "ckvq",
        "compressed_kv_cache": "ckv",
        "compressed_topk_indices": "cidx",
        "compressed_topk_scores": "cscore",
        "sparse_kv": "skv",
        "sparse_topk_indices": "sidx",
        "index_q_packed": "iqp",
        "index_q_heads": "iqh",
        "index_q_rope": "iqR",
        "index_k": "ik",
        "index_k_norm": "ikn",
        "index_q_rotated": "iqrot",
        "index_q_quant": "iqq",
        "index_weights": "iw",
        "compressed_index_scores": "iscore",
        "post_attention_residual": "xattn",
        "post_attention_layernorm_out": "xffn",
        "mlp_gate": "gate",
        "mlp_up": "up",
        "mlp_gate_activated": "act",
        "mlp_product": "mid",
        "mlp_output": "ffn",
        "layer_output": "y",
        "tokens_flat": "tok",
        "router_logits": "rlogit",
        "router_scores": "rs",
        "router_scores_biased": "rsb",
        "router_topk_indices": "topi",
        "router_topk_scores": "tops",
        "router_selected_scores": "rsel",
        "router_weights": "rw",
        "routed_tokens": "rtok",
        "routed_gate_up": "rgu",
        "routed_gate": "rg",
        "routed_gate_clamped": "rgc",
        "routed_up": "ru",
        "routed_up_clamped": "ruc",
        "routed_gate_activated": "ract",
        "routed_product": "rmid",
        "routed_weighted_product": "rwmid",
        "routed_expert_output": "rexpert",
        "routed_output": "rout",
        "shared_gate": "sg",
        "shared_gate_clamped": "sgc",
        "shared_up": "su",
        "shared_up_clamped": "suc",
        "shared_gate_activated": "sact",
        "shared_product": "smid",
        "shared_output": "sout",
        "moe_flat_output": "moeF",
        "moe_output": "moe",
    }
    if name in aliases:
        return aliases[name]

    compact = name
    for suffix in ("_out", "_output", "_scores", "_states", "_hidden"):
        if compact.endswith(suffix):
            compact = compact[: -len(suffix)]
    compact = compact.replace("attention", "attn")
    compact = compact.replace("projection", "proj")
    compact = compact.replace("router", "r")
    compact = compact.replace("expert", "exp")
    compact = compact.replace("layernorm", "ln")
    compact = compact.replace("_", "")
    return compact[:12] or "t"


def _value_info(helper: Any, tensor_proto: Any, spec: TensorSpec) -> Any:
    value_info = helper.make_tensor_value_info(
        spec.name,
        _tensor_dtype(tensor_proto, spec.dtype),
        spec.shape,
    )
    if spec.description:
        value_info.doc_string = spec.description
    return value_info


def _tensor_dtype(tensor_proto: Any, dtype: str) -> int:
    normalized = dtype.lower()
    if normalized in ("float16", "fp16"):
        return tensor_proto.FLOAT16
    if normalized in ("bfloat16", "bf16"):
        return tensor_proto.BFLOAT16
    if normalized in ("int64", "long"):
        return tensor_proto.INT64
    return tensor_proto.FLOAT


def _onnx_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, bool):
            clean[key] = int(value)
        elif isinstance(value, (str, int, float)):
            clean[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value):
            clean[key] = list(value)
        elif isinstance(value, (list, tuple)) and all(isinstance(item, float) for item in value):
            clean[key] = list(value)
        elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            clean[key] = list(value)
        else:
            clean[key] = json.dumps(value, sort_keys=True)
    return clean
