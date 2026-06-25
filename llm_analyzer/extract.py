import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schema import Architecture, Component, Layer


def load_config(snapshot_dir: Path) -> Dict[str, Any]:
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("No config.json found in %s" % snapshot_dir)
    return json.loads(config_path.read_text(encoding="utf-8"))


def extract_from_snapshot(
    snapshot_dir: Path,
    model_id: str,
    revision: str,
    files: Optional[List[str]] = None,
    skipped_weight_files: Optional[List[str]] = None,
) -> Architecture:
    config = load_config(snapshot_dir)
    return extract_architecture_from_config(
        config=config,
        model_id=model_id,
        revision=revision,
        files=files or [],
        skipped_weight_files=skipped_weight_files or [],
    )


def extract_architecture_from_config(
    config: Dict[str, Any],
    model_id: str,
    revision: str = "main",
    files: Optional[List[str]] = None,
    skipped_weight_files: Optional[List[str]] = None,
) -> Architecture:
    text_config = _text_config(config)
    vision_config = _vision_config(config)

    model_type = str(config.get("model_type") or text_config.get("model_type") or "unknown")
    architectures = list(config.get("architectures") or [])

    layer_count = _pick_int(text_config, "num_hidden_layers", "n_layer", "num_layers", "n_layers")
    layer_count = layer_count if layer_count is not None else 0
    base_layer_type = _layer_type(model_type, text_config)
    moe_layers = _moe_layers(text_config, layer_count)
    layers = [
        Layer(
            index=index,
            name="layers.%d" % index,
            layer_type=_layer_type_for_index(base_layer_type, text_config, index, moe_layers),
            components=_layer_components(model_type, text_config, index, index in moe_layers),
        )
        for index in range(layer_count)
    ]

    text_decoder = _text_decoder_summary(text_config, model_type, layer_count, base_layer_type, moe_layers)
    vision_encoder = _vision_summary(vision_config)
    summary = _model_summary(config, text_config, vision_config, layer_count)
    notes = _notes(config, text_config, vision_config)

    return Architecture(
        model_id=model_id,
        revision=revision,
        model_type=model_type,
        architectures=architectures,
        summary=summary,
        text_decoder=text_decoder,
        vision_encoder=vision_encoder,
        layers=layers,
        files=files or [],
        skipped_weight_files=skipped_weight_files or [],
        notes=notes,
    )


def _text_config(config: Dict[str, Any]) -> Dict[str, Any]:
    nested = config.get("text_config")
    if isinstance(nested, dict):
        merged = dict(nested)
        for key in (
            "vocab_size",
            "tie_word_embeddings",
            "torch_dtype",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
        ):
            if key in config and key not in merged:
                merged[key] = config[key]
        return merged
    return config


def _vision_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    nested = config.get("vision_config")
    if isinstance(nested, dict):
        merged = dict(nested)
        if "use_unified_vision_chunk" in config and "use_unified_vision_chunk" not in merged:
            merged["use_unified_vision_chunk"] = config["use_unified_vision_chunk"]
        return merged
    return None


def _model_summary(
    config: Dict[str, Any],
    text_config: Dict[str, Any],
    vision_config: Optional[Dict[str, Any]],
    layer_count: int,
) -> Dict[str, Any]:
    experts = _expert_count(text_config)
    activated_experts = _activated_experts(text_config)
    moe_layers = _moe_layers(text_config, layer_count)
    summary = {
        "model_type": config.get("model_type") or text_config.get("model_type"),
        "architectures": config.get("architectures") or [],
        "modalities": _modalities(config, vision_config),
        "layers": layer_count,
        "hidden_size": _pick_int(text_config, "hidden_size", "n_embed", "n_embd", "d_model", "model_dim"),
        "intermediate_size": _pick_int(text_config, "intermediate_size", "ffn_dim", "n_inner") or _bloom_mlp_intermediate(text_config),
        "attention_heads": _pick_int(text_config, "num_attention_heads", "n_head", "num_heads"),
        "kv_heads": _kv_heads(text_config),
        "vocab_size": _pick_int(text_config, "vocab_size"),
        "max_position_embeddings": _pick_int(
            text_config,
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
            "seq_length",
        ),
        "rope": _rope_summary(text_config),
        "moe": {
            "enabled": experts is not None and experts > 1,
            "experts": experts,
            "activated_experts": activated_experts,
            "shared_experts": _pick_int(text_config, "num_shared_experts", "n_shared_experts"),
            "moe_layers": len(moe_layers),
            "dense_mlp_layers": layer_count - len(moe_layers),
            "interleave_moe_layer_step": _pick_int(text_config, "interleave_moe_layer_step"),
            "scoring_func": text_config.get("scoring_func") or text_config.get("score_func"),
            "topk_method": text_config.get("topk_method"),
            "expert_dtype": _expert_dtype(text_config),
        },
        "hyper_connections": _hyper_connection_summary(text_config),
        "mtp": _mtp_summary(text_config),
    }
    return _drop_none(summary)


def _text_decoder_summary(
    text_config: Dict[str, Any],
    model_type: str,
    layer_count: int,
    layer_type: str,
    moe_layers: List[int],
) -> Dict[str, Any]:
    return _drop_none(
        {
            "model_type": text_config.get("model_type") or model_type,
            "layer_type": layer_type,
            "layers": layer_count,
            "moe_layers": moe_layers,
            "dense_mlp_layers": [index for index in range(layer_count) if index not in set(moe_layers)],
            "norm": _norm_type(text_config),
            "activation": text_config.get("hidden_act") or text_config.get("activation_function"),
            "attention": _drop_none(
                {
                    "type": _attention_type(text_config),
                    "heads": _pick_int(text_config, "num_attention_heads", "n_head", "num_heads"),
                    "kv_heads": _kv_heads(text_config),
                    "head_dim": _head_dim(text_config),
                    "q_lora_rank": _pick_int(text_config, "q_lora_rank"),
                    "kv_lora_rank": _pick_int(text_config, "kv_lora_rank"),
                    "qk_head_dim": _qk_head_dim(text_config),
                    "qk_rope_head_dim": _pick_int(text_config, "qk_rope_head_dim", "rope_head_dim"),
                    "qk_nope_head_dim": _qk_nope_head_dim(text_config),
                    "v_head_dim": _pick_int(text_config, "v_head_dim"),
                    "o_lora_rank": _pick_int(text_config, "o_lora_rank"),
                    "o_groups": _pick_int(text_config, "o_groups"),
                    "attention_bias": text_config.get("attention_bias"),
                    "projection_layout": _mimo_v2_projection_layout(text_config),
                    "partial_rotary_factor": text_config.get("partial_rotary_factor"),
                    "attention_value_scale": text_config.get("attention_value_scale"),
                    "sliding_window": _pick_int(text_config, "sliding_window"),
                    "sliding_window_size": _pick_int(text_config, "sliding_window_size"),
                    "hybrid_layer_pattern": text_config.get("hybrid_layer_pattern"),
                    "attention_chunk_size": _pick_int(text_config, "attention_chunk_size"),
                    "attention_dropout": text_config.get("attention_dropout"),
                    "swa_head_dim": _pick_int(text_config, "swa_head_dim"),
                    "swa_v_head_dim": _pick_int(text_config, "swa_v_head_dim"),
                    "swa_num_attention_heads": _pick_int(text_config, "swa_num_attention_heads"),
                    "swa_num_key_value_heads": _pick_int(text_config, "swa_num_key_value_heads"),
                    "swa_rope_theta": text_config.get("swa_rope_theta"),
                    "add_full_attention_sink_bias": text_config.get("add_full_attention_sink_bias"),
                    "add_swa_attention_sink_bias": text_config.get("add_swa_attention_sink_bias"),
                    "use_qk_norm": text_config.get("use_qk_norm") if text_config.get("use_qk_norm") is not None else text_config.get("qk_norm"),
                    "compress_ratios": text_config.get("compress_ratios"),
                    "compress_rope_theta": text_config.get("compress_rope_theta"),
                    "index_n_heads": _pick_int(text_config, "index_n_heads"),
                    "index_head_dim": _pick_int(text_config, "index_head_dim"),
                    "index_topk": _pick_int(text_config, "index_topk"),
                    "index_topk_freq": _pick_int(text_config, "index_topk_freq"),
                    "index_skip_topk_offset": _pick_int(text_config, "index_skip_topk_offset"),
                    "indexer_types": text_config.get("indexer_types"),
                    "index_share_for_mtp_iteration": text_config.get("index_share_for_mtp_iteration"),
                    "rope_interleave": text_config.get("rope_interleave"),
                    "indexer_rope_interleave": text_config.get("indexer_rope_interleave"),
                }
            ),
            "mlp": _mlp_summary(text_config, is_moe_layer=bool(moe_layers)),
            "hyper_connections": _hyper_connection_summary(text_config),
            "mtp": _mtp_summary(text_config),
        }
    )


def _vision_summary(vision_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not vision_config:
        return None
    return _drop_none(
        {
            "model_type": vision_config.get("model_type"),
            "layers": _pick_int(vision_config, "num_hidden_layers", "vt_num_hidden_layers", "num_layers", "n_layers", "depth"),
            "hidden_size": _pick_int(vision_config, "hidden_size", "vt_hidden_size", "d_model", "model_dim"),
            "intermediate_size": _pick_int(vision_config, "intermediate_size", "vt_intermediate_size", "ffn_dim"),
            "attention_heads": _pick_int(vision_config, "num_attention_heads", "vt_num_attention_heads", "num_heads"),
            "kv_heads": _pick_int(vision_config, "num_key_value_heads", "num_kv_heads"),
            "out_hidden_size": _pick_int(vision_config, "out_hidden_size"),
            "image_size": _pick_int(vision_config, "image_size"),
            "patch_size": _pick_int(vision_config, "patch_size"),
            "temporal_patch_size": _pick_int(vision_config, "temporal_patch_size"),
            "spatial_merge_size": _pick_int(vision_config, "spatial_merge_size"),
            "window_size": _pick_int(vision_config, "window_size", "visual_token_window_size"),
            "full_attention_blocks": vision_config.get("fullatt_block_indexes"),
            "attention_impl": vision_config.get("_attn_implementation"),
            "projector_type": vision_config.get("mm_projector_type"),
            "projector_hidden_size": _pick_int(vision_config, "mm_hidden_size"),
            "projector_output_hidden_size": _pick_int(vision_config, "text_hidden_size"),
            "merge_kernel_size": vision_config.get("merge_kernel_size"),
            "merge_type": vision_config.get("merge_type"),
            "video_attention_type": vision_config.get("video_attn_type"),
            "unified_vision_chunk": vision_config.get("use_unified_vision_chunk"),
        }
    )


def _layer_components(
    model_type: str,
    text_config: Dict[str, Any],
    layer_index: int,
    is_moe_layer: bool,
) -> List[Component]:
    if _is_deepseek_v4(model_type, text_config):
        return _deepseek_v4_layer_components(text_config, layer_index, is_moe_layer)
    if _is_bloom(model_type, text_config):
        return _bloom_layer_components(text_config, layer_index, is_moe_layer)
    if _is_kimi_k25(model_type, text_config):
        return _kimi_k25_layer_components(text_config, layer_index, is_moe_layer)
    if _is_glm_moe_dsa(model_type, text_config):
        return _glm_moe_dsa_layer_components(text_config, layer_index, is_moe_layer)
    if _is_hy_v3(model_type, text_config):
        return _hy_v3_layer_components(text_config, layer_index, is_moe_layer)
    if _is_mimo_v2(model_type, text_config):
        return _mimo_v2_layer_components(text_config, layer_index, is_moe_layer)

    norm = _norm_type(text_config)
    rope = _rope_summary(text_config)
    layer_uses_rope = _layer_uses_rope(text_config, layer_index)
    if layer_uses_rope is not None:
        rope["layer_uses_rope"] = layer_uses_rope
    components = [
        Component("input", "activation"),
        Component("input_layernorm", norm),
        Component(
            "q_proj/k_proj/v_proj",
            "projection",
            _drop_none(
                {
                    "attention_heads": _pick_int(text_config, "num_attention_heads", "n_head", "num_heads"),
                    "kv_heads": _kv_heads(text_config),
                    "head_dim": _head_dim(text_config),
                }
            ),
        ),
        Component("rotary_position_embedding", "RoPE" if layer_uses_rope is not False else "NoPE", rope),
        Component(
            "attention",
            "attention_kernel",
            _drop_none(
                {
                    "sliding_window": _pick_int(text_config, "sliding_window"),
                    "attention_chunk_size": _pick_int(text_config, "attention_chunk_size"),
                    "attention_dropout": text_config.get("attention_dropout"),
                }
            ),
        ),
        Component("o_proj", "projection"),
        Component("residual_add_0", "residual"),
        Component("post_attention_layernorm", norm),
    ]

    experts = _expert_count(text_config)
    if is_moe_layer and experts is not None and experts > 1:
        components.extend(
            [
                Component(
                    "router",
                    "MoE router",
                    _drop_none(
                        {
                            "experts": experts,
                            "activated_experts": _activated_experts(text_config),
                            "shared_experts": _shared_expert_count(text_config),
                            "router_aux_loss_coef": text_config.get("router_aux_loss_coef"),
                            "scoring_func": text_config.get("scoring_func") or text_config.get("score_func"),
                            "topk_method": text_config.get("topk_method"),
                        }
                    ),
                ),
                Component("expert_gate/up/down_proj", "routed expert MLP", _mlp_summary(text_config, is_moe_layer=True)),
            ]
        )
        if _shared_expert_count(text_config):
            components.append(Component("shared_expert_gate/up/down_proj", "shared expert MLP", _shared_expert_summary(text_config)))
        components.append(Component("combine_experts", "MoE combine"))
    else:
        components.extend(
            [
                Component(
                    "gate_proj/up_proj",
                    "projection",
                    _drop_none(
                        {
                            "intermediate_size": _pick_int(
                                text_config,
                                "intermediate_size_mlp",
                                "intermediate_size",
                                "ffn_dim",
                                "n_inner",
                            )
                        }
                    ),
                ),
                Component(
                    "activation",
                    str(text_config.get("hidden_act") or text_config.get("activation_function") or "activation"),
                ),
                Component("down_proj", "projection"),
            ]
        )

    components.append(Component("residual_add_1", "residual"))
    return components


def _glm_moe_dsa_layer_components(
    text_config: Dict[str, Any],
    layer_index: int,
    is_moe_layer: bool,
) -> List[Component]:
    norm = _norm_type(text_config)
    components = [
        Component("input", "activation"),
        Component("input_layernorm", norm),
        Component("glm_dsa_attention", "GLM DSA/MLA sparse attention", _glm_dsa_attention_details(text_config, layer_index)),
        Component("residual_add_0", "residual"),
        Component("post_attention_layernorm", norm),
    ]

    experts = _expert_count(text_config)
    if is_moe_layer and experts is not None and experts > 1:
        router = _drop_none(
            {
                "experts": experts,
                "activated_experts": _activated_experts(text_config),
                "shared_experts": _pick_int(text_config, "num_shared_experts", "n_shared_experts"),
                "scoring_func": text_config.get("scoring_func") or text_config.get("score_func"),
                "topk_method": text_config.get("topk_method"),
                "topk_group": _pick_int(text_config, "topk_group"),
                "route_scale": text_config.get("routed_scaling_factor") or text_config.get("route_scale"),
                "norm_topk_prob": text_config.get("norm_topk_prob"),
                "score_correction_bias": True,
            }
        )
        components.extend(
            [
                Component("router", "MoE router", router),
                Component("expert_gate/up/down_proj", "routed expert SwiGLU", _mlp_summary(text_config, is_moe_layer=True)),
                Component("shared_expert_gate/up/down_proj", "shared expert SwiGLU", _shared_expert_summary(text_config)),
                Component("combine_experts", "MoE combine"),
            ]
        )
    else:
        components.extend(
            [
                Component(
                    "gate_proj/up_proj",
                    "projection",
                    _drop_none({"intermediate_size": _pick_int(text_config, "intermediate_size", "ffn_dim", "n_inner")}),
                ),
                Component(
                    "activation",
                    str(text_config.get("hidden_act") or text_config.get("activation_function") or "activation"),
                ),
                Component("down_proj", "projection"),
            ]
        )

    components.append(Component("residual_add_1", "residual"))
    return components


def _mimo_v2_layer_components(
    text_config: Dict[str, Any],
    layer_index: int,
    is_moe_layer: bool,
) -> List[Component]:
    norm = _norm_type(text_config)
    projection_layout = _mimo_v2_projection_layout(text_config)
    attention_kind = "fused-QKV" if projection_layout == "fused_qkv" else "split-QKV"
    components = [
        Component("input", "activation"),
        Component("input_layernorm", norm),
        Component("mimo_v2_attention", "MiMo V2 %s hybrid GQA" % attention_kind, _mimo_v2_attention_details(text_config, layer_index)),
        Component("residual_add_0", "residual"),
        Component("post_attention_layernorm", norm),
    ]

    experts = _expert_count(text_config)
    if is_moe_layer and experts is not None and experts > 1:
        components.extend(
            [
                Component("router", "MiMo noaux_tc MoE router", _mimo_router_details(text_config)),
                Component("expert_gate/up/down_proj", "routed expert SwiGLU", _mlp_summary(text_config, is_moe_layer=True)),
                Component("combine_experts", "MoE weighted index_add combine"),
            ]
        )
    else:
        components.extend(
            [
                Component(
                    "gate_proj/up_proj",
                    "projection",
                    _drop_none({"intermediate_size": _pick_int(text_config, "intermediate_size", "ffn_dim", "n_inner")}),
                ),
                Component(
                    "activation",
                    str(text_config.get("hidden_act") or text_config.get("activation_function") or "activation"),
                ),
                Component("down_proj", "projection"),
            ]
        )

    components.append(Component("residual_add_1", "residual"))
    return components


def _hy_v3_layer_components(
    text_config: Dict[str, Any],
    layer_index: int,
    is_moe_layer: bool,
) -> List[Component]:
    norm = _norm_type(text_config)
    components = [
        Component("input", "activation"),
        Component("input_layernorm", norm),
        Component("hy_v3_attention", "Hy3 GQA with Q/K RMSNorm", _hy_v3_attention_details(text_config, layer_index)),
        Component("residual_add_0", "residual"),
        Component("post_attention_layernorm", norm),
    ]

    experts = _expert_count(text_config)
    if is_moe_layer and experts is not None and experts > 1:
        components.extend(
            [
                Component("router", "Hy3 sigmoid bias top-k router", _hy_v3_router_details(text_config)),
                Component("expert_gate/up/down_proj", "routed expert SwiGLU", _mlp_summary(text_config, is_moe_layer=True)),
                Component("shared_expert_gate/up/down_proj", "shared expert SwiGLU", _shared_expert_summary(text_config)),
                Component("combine_experts", "routed plus shared MoE combine"),
            ]
        )
    else:
        components.extend(
            [
                Component(
                    "gate_proj/up_proj",
                    "projection",
                    _drop_none({"intermediate_size": _pick_int(text_config, "intermediate_size", "ffn_dim", "n_inner")}),
                ),
                Component(
                    "activation",
                    str(text_config.get("hidden_act") or text_config.get("activation_function") or "activation"),
                ),
                Component("down_proj", "projection"),
            ]
        )

    components.append(Component("residual_add_1", "residual"))
    return components


def _hy_v3_attention_details(text_config: Dict[str, Any], layer_index: int) -> Dict[str, Any]:
    heads = _pick_int(text_config, "num_attention_heads", "n_head", "num_heads")
    kv_heads = _kv_heads(text_config)
    head_dim = _head_dim(text_config)
    q_size = _product_int(heads, head_dim)
    kv_size = _product_int(kv_heads, head_dim)
    return _drop_none(
        {
            "type": "gqa",
            "attention_heads": heads,
            "kv_heads": kv_heads,
            "kv_groups": _mimo_kv_groups(heads, kv_heads),
            "head_dim": head_dim,
            "q_size": q_size,
            "k_size": kv_size,
            "v_size": kv_size,
            "o_hidden_size": q_size,
            "qk_norm": text_config.get("qk_norm") if text_config.get("qk_norm") is not None else text_config.get("use_qk_norm"),
            "q_norm": "HYV3RMSNorm",
            "k_norm": "HYV3RMSNorm",
            "rope_theta": (text_config.get("rope_parameters") or {}).get("rope_theta") if isinstance(text_config.get("rope_parameters"), dict) else text_config.get("rope_theta"),
            "rope_type": (text_config.get("rope_parameters") or {}).get("rope_type") if isinstance(text_config.get("rope_parameters"), dict) else text_config.get("rope_type"),
            "attention_bias": text_config.get("attention_bias"),
            "attention_dropout": text_config.get("attention_dropout"),
            "fp32_softmax": True,
            "enable_attention_fp32_softmax": text_config.get("enable_attention_fp32_softmax"),
            "layer_index": layer_index,
        }
    )


def _hy_v3_router_details(text_config: Dict[str, Any]) -> Dict[str, Any]:
    use_sigmoid = text_config.get("moe_router_use_sigmoid")
    return _drop_none(
        {
            "experts": _expert_count(text_config),
            "activated_experts": _activated_experts(text_config),
            "shared_experts": _shared_expert_count(text_config),
            "scoring_func": "sigmoid" if use_sigmoid is not False else "softmax",
            "topk_method": "sigmoid_bias_topk" if text_config.get("moe_router_enable_expert_bias") else "topk",
            "route_scale": text_config.get("router_scaling_factor"),
            "norm_topk_prob": text_config.get("route_norm"),
            "score_correction_bias": text_config.get("moe_router_enable_expert_bias"),
            "output_router_logits": text_config.get("output_router_logits"),
            "enable_moe_fp32_combine": text_config.get("enable_moe_fp32_combine"),
            "use_grouped_mm": text_config.get("use_grouped_mm"),
        }
    )


def _mimo_v2_attention_details(text_config: Dict[str, Any], layer_index: int) -> Dict[str, Any]:
    is_swa = _mimo_layer_is_swa(text_config, layer_index)
    heads = _mimo_attention_heads(text_config, is_swa)
    kv_heads = _mimo_kv_heads(text_config, is_swa)
    head_dim = _mimo_head_dim(text_config, is_swa)
    v_head_dim = _mimo_v_head_dim(text_config, is_swa, head_dim)
    q_size = _product_int(heads, head_dim)
    k_size = _product_int(kv_heads, head_dim)
    v_size = _product_int(kv_heads, v_head_dim)
    partial_rotary_factor = _as_float(text_config.get("partial_rotary_factor"))
    rope_dim = int(head_dim * partial_rotary_factor) if head_dim is not None and partial_rotary_factor is not None else None
    return _drop_none(
        {
            "type": "sliding_window_attention" if is_swa else "full_attention",
            "attention_heads": heads,
            "kv_heads": kv_heads,
            "kv_groups": _mimo_kv_groups(heads, kv_heads),
            "head_dim": head_dim,
            "v_head_dim": v_head_dim,
            "q_size": q_size,
            "k_size": k_size,
            "v_size": v_size,
            "qkv_size": _sum_ints(q_size, k_size, v_size),
            "o_hidden_size": _product_int(heads, v_head_dim),
            "projection_layout": _mimo_v2_projection_layout(text_config),
            "partial_rotary_factor": partial_rotary_factor,
            "rope_dim": rope_dim,
            "rope_theta": text_config.get("swa_rope_theta") if is_swa else text_config.get("rope_theta"),
            "attention_value_scale": text_config.get("attention_value_scale"),
            "sliding_window": _pick_int(text_config, "sliding_window", "sliding_window_size") if is_swa else None,
            "attention_chunk_size": _pick_int(text_config, "attention_chunk_size"),
            "attention_sink_bias": (
                text_config.get("add_swa_attention_sink_bias") if is_swa else text_config.get("add_full_attention_sink_bias")
            ),
            "attention_dropout": text_config.get("attention_dropout"),
            "attention_bias": text_config.get("attention_bias"),
            "layer_index": layer_index,
        }
    )


def _mimo_router_details(text_config: Dict[str, Any]) -> Dict[str, Any]:
    return _drop_none(
        {
            "experts": _expert_count(text_config),
            "activated_experts": _activated_experts(text_config),
            "shared_experts": _pick_int(text_config, "num_shared_experts", "n_shared_experts"),
            "scoring_func": text_config.get("scoring_func") or text_config.get("score_func"),
            "topk_method": text_config.get("topk_method"),
            "n_group": _pick_int(text_config, "n_group"),
            "topk_group": _pick_int(text_config, "topk_group"),
            "route_scale": text_config.get("routed_scaling_factor") if text_config.get("routed_scaling_factor") is not None else 1.0,
            "norm_topk_prob": text_config.get("norm_topk_prob"),
            "score_correction_bias": text_config.get("topk_method") == "noaux_tc",
        }
    )


def _glm_dsa_attention_details(text_config: Dict[str, Any], layer_index: int) -> Dict[str, Any]:
    return _drop_none(
        {
            "type": _attention_type(text_config),
            "attention_heads": _pick_int(text_config, "num_attention_heads", "n_head", "num_heads"),
            "kv_heads": _pick_int(text_config, "num_key_value_heads", "num_kv_heads", "n_kv_heads"),
            "q_lora_rank": _pick_int(text_config, "q_lora_rank"),
            "kv_lora_rank": _pick_int(text_config, "kv_lora_rank"),
            "qk_head_dim": _qk_head_dim(text_config),
            "qk_nope_head_dim": _pick_int(text_config, "qk_nope_head_dim"),
            "qk_rope_head_dim": _pick_int(text_config, "qk_rope_head_dim"),
            "v_head_dim": _pick_int(text_config, "v_head_dim"),
            "head_dim": _pick_int(text_config, "head_dim"),
            "rope": _rope_summary(text_config),
            "rope_interleave": text_config.get("rope_interleave"),
            "indexer_type": _glm_indexer_type(text_config, layer_index),
            "indexer_rope_interleave": text_config.get("indexer_rope_interleave"),
            "index_n_heads": _pick_int(text_config, "index_n_heads"),
            "index_head_dim": _pick_int(text_config, "index_head_dim"),
            "index_topk": _pick_int(text_config, "index_topk"),
            "index_topk_freq": _pick_int(text_config, "index_topk_freq"),
            "index_skip_topk_offset": _pick_int(text_config, "index_skip_topk_offset"),
            "index_share_for_mtp_iteration": text_config.get("index_share_for_mtp_iteration"),
        }
    )


def _bloom_layer_components(
    text_config: Dict[str, Any],
    layer_index: int,
    is_moe_layer: bool,
) -> List[Component]:
    norm = _norm_type(text_config)
    components = [
        Component("input", "activation"),
        Component("input_layernorm", norm),
        Component("bloom_attention", "BLOOM fused-QKV attention with ALiBi", _bloom_attention_details(text_config, layer_index)),
        Component("residual_add_0", "residual"),
        Component("post_attention_layernorm", norm),
        Component(
            "dense_h_to_4h",
            "projection",
            _drop_none({"intermediate_size": _bloom_mlp_intermediate(text_config)}),
        ),
        Component("activation", str(text_config.get("hidden_act") or text_config.get("activation_function") or "gelu")),
        Component("dense_4h_to_h", "projection"),
        Component("residual_add_1", "residual"),
    ]
    return components


def _bloom_attention_details(text_config: Dict[str, Any], layer_index: int) -> Dict[str, Any]:
    return _drop_none(
        {
            "type": _attention_type(text_config),
            "attention_heads": _pick_int(text_config, "num_attention_heads", "n_head", "num_heads"),
            "kv_heads": _pick_int(text_config, "num_attention_heads", "n_head", "num_heads"),
            "head_dim": _head_dim(text_config),
            "alibi": True,
            "attention_dropout": text_config.get("attention_dropout") or text_config.get("attn_pdrop"),
            "attention_softmax_in_fp32": text_config.get("attention_softmax_in_fp32"),
            "pretraining_tp": _pick_int(text_config, "pretraining_tp"),
            "layer_index": layer_index,
        }
    )


def _kimi_k25_layer_components(
    text_config: Dict[str, Any],
    layer_index: int,
    is_moe_layer: bool,
) -> List[Component]:
    norm = _norm_type(text_config)
    components = [
        Component("input", "activation"),
        Component("input_layernorm", norm),
        Component("kimi_mla_attention", "Kimi/DeepSeek-V3 MLA attention", _kimi_mla_attention_details(text_config, layer_index)),
        Component("residual_add_0", "residual"),
        Component("post_attention_layernorm", norm),
    ]

    experts = _expert_count(text_config)
    if is_moe_layer and experts is not None and experts > 1:
        components.extend(
            [
                Component("router", "MoE router", _kimi_router_details(text_config)),
                Component("expert_gate/up/down_proj", "routed int4-packed expert SwiGLU", _mlp_summary(text_config, is_moe_layer=True)),
                Component("shared_expert_gate/up/down_proj", "shared expert SwiGLU", _shared_expert_summary(text_config)),
                Component("combine_experts", "MoE combine"),
            ]
        )
    else:
        components.extend(
            [
                Component(
                    "gate_proj/up_proj",
                    "projection",
                    _drop_none({"intermediate_size": _pick_int(text_config, "intermediate_size", "ffn_dim", "n_inner")}),
                ),
                Component(
                    "activation",
                    str(text_config.get("hidden_act") or text_config.get("activation_function") or "activation"),
                ),
                Component("down_proj", "projection"),
            ]
        )

    components.append(Component("residual_add_1", "residual"))
    return components


def _kimi_mla_attention_details(text_config: Dict[str, Any], layer_index: int) -> Dict[str, Any]:
    return _drop_none(
        {
            "type": _attention_type(text_config),
            "attention_heads": _pick_int(text_config, "num_attention_heads", "n_head", "num_heads"),
            "kv_heads": _pick_int(text_config, "num_key_value_heads", "num_kv_heads", "n_kv_heads"),
            "q_lora_rank": _pick_int(text_config, "q_lora_rank"),
            "kv_lora_rank": _pick_int(text_config, "kv_lora_rank"),
            "qk_head_dim": _qk_head_dim(text_config),
            "qk_nope_head_dim": _qk_nope_head_dim(text_config),
            "qk_rope_head_dim": _pick_int(text_config, "qk_rope_head_dim", "rope_head_dim"),
            "v_head_dim": _pick_int(text_config, "v_head_dim"),
            "rope": _rope_summary(text_config),
            "attention_bias": text_config.get("attention_bias"),
            "attention_dropout": text_config.get("attention_dropout"),
            "flash_attention": True if text_config.get("_attn_implementation") == "flash_attention_2" else None,
            "softmax_scale_dim": _qk_head_dim(text_config),
            "layer_index": layer_index,
        }
    )


def _kimi_router_details(text_config: Dict[str, Any]) -> Dict[str, Any]:
    return _drop_none(
        {
            "experts": _expert_count(text_config),
            "activated_experts": _activated_experts(text_config),
            "shared_experts": _pick_int(text_config, "num_shared_experts", "n_shared_experts"),
            "scoring_func": text_config.get("scoring_func") or text_config.get("score_func"),
            "topk_method": text_config.get("topk_method"),
            "n_group": _pick_int(text_config, "n_group"),
            "topk_group": _pick_int(text_config, "topk_group"),
            "route_scale": text_config.get("routed_scaling_factor") or text_config.get("route_scale"),
            "norm_topk_prob": text_config.get("norm_topk_prob"),
            "score_correction_bias": text_config.get("topk_method") == "noaux_tc",
            "seq_aux": text_config.get("seq_aux"),
            "aux_loss_alpha": text_config.get("aux_loss_alpha"),
        }
    )


def _deepseek_v4_layer_components(
    text_config: Dict[str, Any],
    layer_index: int,
    is_moe_layer: bool,
) -> List[Component]:
    norm = _norm_type(text_config)
    experts = _expert_count(text_config)
    router = _drop_none(
        {
            "experts": experts,
            "activated_experts": _activated_experts(text_config),
            "shared_experts": _pick_int(text_config, "num_shared_experts", "n_shared_experts"),
            "scoring_func": text_config.get("scoring_func") or text_config.get("score_func"),
            "topk_method": text_config.get("topk_method"),
            "route_scale": text_config.get("routed_scaling_factor") or text_config.get("route_scale"),
            "norm_topk_prob": text_config.get("norm_topk_prob"),
            "num_hash_layers": _pick_int(text_config, "num_hash_layers"),
            "hash_routing": _deepseek_hash_routing(text_config, layer_index),
        }
    )
    attention = _deepseek_attention_details(text_config, layer_index)
    hyper_connections = _hyper_connection_summary(text_config)
    components = [
        Component("input", "activation", _drop_none({"shape": "[batch, sequence, hc_mult, hidden]"})),
        Component("hc_attn_pre", "HyperConnectionPre", hyper_connections),
        Component("attn_norm", norm),
        Component("mla_sparse_attention", "DeepSeek MLA sparse attention", attention),
        Component("hc_attn_post", "HyperConnectionPost", hyper_connections),
        Component("hc_ffn_pre", "HyperConnectionPre", hyper_connections),
        Component("ffn_norm", norm),
    ]

    if is_moe_layer and experts is not None and experts > 1:
        components.extend(
            [
                Component("router", "MoE router", router),
                Component("expert_gate/up/down_proj", "routed FP4 expert SwiGLU", _mlp_summary(text_config, is_moe_layer=True)),
                Component("shared_expert_gate/up/down_proj", "shared expert SwiGLU", _shared_expert_summary(text_config)),
                Component("combine_experts", "MoE combine"),
            ]
        )
    else:
        components.extend(
            [
                Component(
                    "gate_proj/up_proj",
                    "projection",
                    _drop_none({"intermediate_size": _pick_int(text_config, "intermediate_size", "ffn_dim", "n_inner")}),
                ),
                Component(
                    "activation",
                    str(text_config.get("hidden_act") or text_config.get("activation_function") or "activation"),
                ),
                Component("down_proj", "projection"),
            ]
        )

    components.append(Component("hc_ffn_post", "HyperConnectionPost", hyper_connections))
    return components


def _deepseek_attention_details(text_config: Dict[str, Any], layer_index: int) -> Dict[str, Any]:
    head_dim = _pick_int(text_config, "head_dim")
    rope_head_dim = _pick_int(text_config, "qk_rope_head_dim", "rope_head_dim")
    compress_ratio = _deepseek_compress_ratio(text_config, layer_index)
    return _drop_none(
        {
            "type": _attention_type(text_config),
            "attention_heads": _pick_int(text_config, "num_attention_heads", "n_head", "num_heads"),
            "kv_heads": _pick_int(text_config, "num_key_value_heads", "num_kv_heads", "n_kv_heads"),
            "head_dim": head_dim,
            "q_lora_rank": _pick_int(text_config, "q_lora_rank"),
            "qk_rope_head_dim": rope_head_dim,
            "qk_nope_head_dim": _qk_nope_head_dim(text_config),
            "o_groups": _pick_int(text_config, "o_groups"),
            "o_lora_rank": _pick_int(text_config, "o_lora_rank"),
            "sliding_window": _pick_int(text_config, "sliding_window"),
            "compress_ratio": compress_ratio,
            "compress_mode": _deepseek_compress_mode(compress_ratio),
            "compress_rope_theta": text_config.get("compress_rope_theta"),
            "rope": _rope_summary(text_config),
            "index_n_heads": _pick_int(text_config, "index_n_heads"),
            "index_head_dim": _pick_int(text_config, "index_head_dim"),
            "index_topk": _pick_int(text_config, "index_topk"),
        }
    )


def _attention_type(text_config: Dict[str, Any]) -> str:
    model_type = str(text_config.get("model_type") or "").lower()
    if model_type.startswith("bloom"):
        return "MHA + ALiBi"
    if model_type.startswith("gpt_oss"):
        return "GQA + RoPE + sliding window"
    if model_type.startswith("deepseek_v4"):
        return "MLA sparse attention"
    if model_type.startswith("kimi_k2") or model_type.startswith("kimi_k25"):
        return "Kimi/DeepSeek-V3 MLA"
    if model_type.startswith("glm_moe_dsa"):
        return "GLM DSA/MLA"
    if model_type.startswith("hy_v3"):
        return "Hy3 GQA + Q/K RMSNorm + RoPE"
    if model_type.startswith("mimo_v2"):
        return "MiMo V2 hybrid GQA + partial RoPE"
    if _pick_int(text_config, "q_lora_rank") is not None:
        return "MLA"
    return "MHA/GQA"


def _mimo_v2_projection_layout(text_config: Dict[str, Any]) -> Optional[str]:
    layout = text_config.get("attention_projection_layout")
    if layout:
        return str(layout)
    if str(text_config.get("model_type") or "").lower().startswith("mimo_v2"):
        return "split"
    return None


def _head_dim(text_config: Dict[str, Any]) -> Optional[int]:
    explicit = _pick_int(text_config, "head_dim")
    if explicit is not None:
        return explicit
    hidden_size = _pick_int(text_config, "hidden_size", "n_embed", "n_embd", "d_model", "model_dim")
    heads = _pick_int(text_config, "num_attention_heads", "n_head", "num_heads")
    if hidden_size is not None and heads:
        return hidden_size // heads
    return None


def _kv_heads(text_config: Dict[str, Any]) -> Optional[int]:
    explicit = _pick_int(text_config, "num_key_value_heads", "num_kv_heads", "n_kv_heads")
    if explicit is not None:
        return explicit
    if str(text_config.get("model_type") or "").lower().startswith("bloom"):
        return _pick_int(text_config, "num_attention_heads", "n_head", "num_heads")
    return None


def _qk_head_dim(text_config: Dict[str, Any]) -> Optional[int]:
    explicit = _pick_int(text_config, "qk_head_dim")
    if explicit is not None:
        return explicit
    nope_dim = _pick_int(text_config, "qk_nope_head_dim")
    rope_dim = _pick_int(text_config, "qk_rope_head_dim", "rope_head_dim")
    if nope_dim is not None and rope_dim is not None:
        return nope_dim + rope_dim
    return _pick_int(text_config, "head_dim")


def _qk_nope_head_dim(text_config: Dict[str, Any]) -> Optional[int]:
    explicit = _pick_int(text_config, "qk_nope_head_dim")
    if explicit is not None:
        return explicit
    qk_head_dim = _pick_int(text_config, "qk_head_dim")
    rope_head_dim = _pick_int(text_config, "qk_rope_head_dim", "rope_head_dim")
    if qk_head_dim is not None and rope_head_dim is not None:
        return qk_head_dim - rope_head_dim
    head_dim = _pick_int(text_config, "head_dim")
    if head_dim is None or rope_head_dim is None:
        return None
    return head_dim - rope_head_dim


def _deepseek_compress_ratio(text_config: Dict[str, Any], layer_index: int) -> Optional[int]:
    ratios = text_config.get("compress_ratios")
    if isinstance(ratios, list) and layer_index < len(ratios):
        return _as_int(ratios[layer_index])
    return _pick_int(text_config, "compress_ratio")


def _deepseek_compress_mode(compress_ratio: Optional[int]) -> Optional[str]:
    if compress_ratio is None:
        return None
    if compress_ratio == 0:
        return "sliding_window_only"
    if compress_ratio == 4:
        return "overlap_compression_with_learned_indexer"
    return "pooled_compression"


def _deepseek_hash_routing(text_config: Dict[str, Any], layer_index: int) -> Optional[bool]:
    hash_layers = _pick_int(text_config, "num_hash_layers")
    if hash_layers is None:
        return None
    return layer_index < hash_layers


def _glm_indexer_type(text_config: Dict[str, Any], layer_index: int) -> Optional[str]:
    indexer_types = text_config.get("indexer_types")
    if isinstance(indexer_types, list) and layer_index < len(indexer_types):
        return str(indexer_types[layer_index])
    return None


def _hyper_connection_summary(text_config: Dict[str, Any]) -> Dict[str, Any]:
    return _drop_none(
        {
            "hc_mult": _pick_int(text_config, "hc_mult"),
            "hc_sinkhorn_iters": _pick_int(text_config, "hc_sinkhorn_iters"),
            "hc_eps": text_config.get("hc_eps"),
        }
    )


def _mtp_summary(text_config: Dict[str, Any]) -> Dict[str, Any]:
    nextn_layers = _pick_int(text_config, "num_nextn_predict_layers", "n_mtp_layers")
    return _drop_none(
        {
            "nextn_predict_layers": nextn_layers if nextn_layers and nextn_layers > 0 else None,
        }
    )


def _is_deepseek_v4(model_type: str, text_config: Dict[str, Any]) -> bool:
    text_model_type = str(text_config.get("model_type") or model_type or "").lower()
    return text_model_type.startswith("deepseek_v4")


def _is_bloom(model_type: str, text_config: Dict[str, Any]) -> bool:
    text_model_type = str(text_config.get("model_type") or model_type or "").lower()
    return text_model_type.startswith("bloom")


def _is_kimi_k25(model_type: str, text_config: Dict[str, Any]) -> bool:
    top_model_type = str(model_type or "").lower()
    text_model_type = str(text_config.get("model_type") or model_type or "").lower()
    return top_model_type.startswith("kimi_k25") or text_model_type.startswith("kimi_k2")


def _is_glm_moe_dsa(model_type: str, text_config: Dict[str, Any]) -> bool:
    text_model_type = str(text_config.get("model_type") or model_type or "").lower()
    return text_model_type.startswith("glm_moe_dsa")


def _is_hy_v3(model_type: str, text_config: Dict[str, Any]) -> bool:
    text_model_type = str(text_config.get("model_type") or model_type or "").lower()
    return text_model_type.startswith("hy_v3")


def _is_mimo_v2(model_type: str, text_config: Dict[str, Any]) -> bool:
    text_model_type = str(text_config.get("model_type") or model_type or "").lower()
    return text_model_type.startswith("mimo_v2")


def _mlp_summary(text_config: Dict[str, Any], is_moe_layer: bool = False) -> Dict[str, Any]:
    experts = _expert_count(text_config)
    intermediate_keys = (
        ("moe_intermediate_size", "expert_intermediate_size", "intermediate_size")
        if is_moe_layer
        else ("intermediate_size_mlp", "intermediate_size", "ffn_dim", "n_inner")
    )
    return _drop_none(
        {
            "type": "MoE" if is_moe_layer and experts is not None and experts > 1 else "dense",
            "intermediate_size": _pick_int(text_config, *intermediate_keys),
            "moe_intermediate_size": _pick_int(text_config, "moe_intermediate_size", "expert_intermediate_size"),
            "activation": text_config.get("hidden_act") or text_config.get("activation_function"),
            "experts": experts if is_moe_layer else None,
            "activated_experts": _activated_experts(text_config) if is_moe_layer else None,
            "scoring_func": _router_scoring_func(text_config) if is_moe_layer else None,
            "topk_method": _router_topk_method(text_config) if is_moe_layer else None,
            "route_scale": _router_route_scale(text_config) if is_moe_layer else None,
            "norm_topk_prob": _router_norm_topk(text_config) if is_moe_layer else None,
            "expert_dtype": _expert_dtype(text_config) if is_moe_layer else None,
            "swiglu_limit": text_config.get("swiglu_limit") if is_moe_layer else None,
            "use_grouped_mm": text_config.get("use_grouped_mm") if is_moe_layer else None,
        }
    )


def _expert_dtype(text_config: Dict[str, Any]) -> Optional[str]:
    if str(text_config.get("model_type") or "").lower().startswith("kimi_k2"):
        return "int4-packed"
    if str(text_config.get("model_type") or "").lower().startswith("gpt_oss"):
        return "mxfp4"
    if str(text_config.get("model_type") or "").lower().startswith("deepseek_v4"):
        return "fp4"
    if str(text_config.get("model_type") or "").lower().startswith("hy_v3"):
        return "bfloat16"
    quantization = text_config.get("quantization_config")
    if isinstance(quantization, dict) and quantization.get("quant_method"):
        method = str(quantization.get("quant_method"))
        fmt = quantization.get("fmt")
        return "%s-%s" % (method, fmt) if fmt else method
    explicit = text_config.get("expert_dtype") or text_config.get("dtype") or text_config.get("torch_dtype")
    if explicit:
        return str(explicit)
    return None


def _shared_expert_summary(text_config: Dict[str, Any]) -> Dict[str, Any]:
    return _drop_none(
        {
            "type": "shared_dense",
            "experts": _shared_expert_count(text_config),
            "intermediate_size": _shared_expert_intermediate(text_config),
            "activation": text_config.get("hidden_act") or text_config.get("activation_function"),
            "swiglu_limit": text_config.get("swiglu_limit"),
            "enable_moe_fp32_combine": text_config.get("enable_moe_fp32_combine"),
        }
    )


def _shared_expert_count(text_config: Dict[str, Any]) -> Optional[int]:
    explicit = _pick_int(text_config, "num_shared_experts", "n_shared_experts")
    if explicit is not None:
        return explicit
    if str(text_config.get("model_type") or "").lower().startswith("llama4"):
        return 1
    return None


def _shared_expert_intermediate(text_config: Dict[str, Any]) -> Optional[int]:
    intermediate = _pick_int(
        text_config,
        "moe_intermediate_size",
        "expert_intermediate_size",
        "intermediate_size",
        "ffn_dim",
        "n_inner",
    )
    shared_experts = _shared_expert_count(text_config)
    if str(text_config.get("model_type") or "").lower().startswith("hy_v3") and intermediate is not None and shared_experts:
        return intermediate * shared_experts
    return intermediate


def _router_scoring_func(text_config: Dict[str, Any]) -> Optional[str]:
    if str(text_config.get("model_type") or "").lower().startswith("hy_v3"):
        return "sigmoid" if text_config.get("moe_router_use_sigmoid") is not False else "softmax"
    return text_config.get("scoring_func") or text_config.get("score_func")


def _router_topk_method(text_config: Dict[str, Any]) -> Optional[str]:
    if str(text_config.get("model_type") or "").lower().startswith("hy_v3"):
        return "sigmoid_bias_topk" if text_config.get("moe_router_enable_expert_bias") else "topk"
    return text_config.get("topk_method")


def _router_route_scale(text_config: Dict[str, Any]) -> Any:
    if str(text_config.get("model_type") or "").lower().startswith("hy_v3"):
        return text_config.get("router_scaling_factor")
    return text_config.get("routed_scaling_factor") or text_config.get("route_scale")


def _router_norm_topk(text_config: Dict[str, Any]) -> Any:
    if str(text_config.get("model_type") or "").lower().startswith("hy_v3"):
        return text_config.get("route_norm")
    return text_config.get("norm_topk_prob")


def _layer_type(model_type: str, text_config: Dict[str, Any]) -> str:
    text_model_type = str(text_config.get("model_type") or model_type or "").lower()
    if text_model_type.startswith("bloom"):
        return "BloomBlock"
    if text_model_type.startswith("gpt_oss"):
        return "GptOssDecoderLayer"
    if text_model_type.startswith("deepseek_v4"):
        return "DeepseekV4Block"
    if text_model_type.startswith("kimi_k2") or str(model_type or "").lower().startswith("kimi_k25"):
        return "KimiK25DeepseekDecoderLayer"
    if "deepseek" in text_model_type:
        return "DeepseekDecoderLayer"
    if text_model_type.startswith("glm_moe_dsa"):
        return "GlmMoeDsaDecoderLayer"
    if text_model_type.startswith("hy_v3"):
        return "HYV3DecoderLayer"
    if text_model_type.startswith("mimo_v2"):
        return "MiMoV2DecoderLayer"
    if "glm" in text_model_type:
        return "GlmDecoderLayer"
    if "llama4" in text_model_type or "llama-4" in text_model_type:
        return "Llama4TextDecoderLayer"
    if "llama" in text_model_type:
        return "LlamaDecoderLayer"
    if text_model_type.startswith("qwen3_5_moe"):
        return "Qwen3_5MoeDecoderLayer"
    if "qwen" in text_model_type:
        return "QwenDecoderLayer"
    if text_model_type.startswith("mistral3") or text_model_type.startswith("ministral3") or str(model_type or "").lower().startswith("mistral3"):
        return "Mistral3DecoderLayer"
    if "mistral" in text_model_type:
        return "MistralDecoderLayer"
    if "mixtral" in text_model_type:
        return "MixtralDecoderLayer"
    if "gemma" in text_model_type:
        return "GemmaDecoderLayer"
    if "phi" in text_model_type:
        return "PhiDecoderLayer"
    return "DecoderLayer"


def _layer_type_for_index(
    base_layer_type: str,
    text_config: Dict[str, Any],
    layer_index: int,
    moe_layers: List[int],
) -> str:
    experts = _expert_count(text_config)
    if experts is not None and experts > 1:
        suffix = "MoE" if layer_index in moe_layers else "DenseMLP"
        return "%s[%s]" % (base_layer_type, suffix)
    return base_layer_type


def _norm_type(config: Dict[str, Any]) -> str:
    if str(config.get("model_type") or "").lower().startswith("mimo_v2"):
        return "RMSNorm"
    if "rms_norm_eps" in config:
        return "RMSNorm"
    if "layer_norm_eps" in config or "layer_norm_epsilon" in config:
        return "LayerNorm"
    return "Norm"


def _bloom_mlp_intermediate(text_config: Dict[str, Any]) -> Optional[int]:
    if not str(text_config.get("model_type") or "").lower().startswith("bloom"):
        return None
    hidden_size = _pick_int(text_config, "hidden_size", "n_embed", "n_embd")
    return 4 * hidden_size if hidden_size is not None else None


def _rope_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    rope_parameters = config.get("rope_parameters") if isinstance(config.get("rope_parameters"), dict) else {}
    has_rope = any(key in config for key in ("rope_theta", "rope_scaling", "rotary_emb_base", "rope_type", "rope_parameters"))
    no_rope_interval = _pick_int(config, "no_rope_layer_interval")
    if no_rope_interval is None and str(config.get("model_type", "")).lower().startswith("llama4"):
        no_rope_interval = 4
    return _drop_none(
        {
            "enabled": has_rope,
            "theta": config.get("rope_theta") or config.get("rotary_emb_base") or rope_parameters.get("rope_theta"),
            "scaling": config.get("rope_scaling"),
            "type": config.get("rope_type") or rope_parameters.get("rope_type"),
            "no_rope_layer_interval": no_rope_interval,
        }
    )


def _expert_count(config: Dict[str, Any]) -> Optional[int]:
    return _pick_int(
        config,
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
        "moe_num_experts",
        "num_experts_total",
    )


def _activated_experts(config: Dict[str, Any]) -> Optional[int]:
    return _pick_int(
        config,
        "num_experts_per_tok",
        "num_experts_per_token",
        "num_selected_experts",
        "moe_top_k",
        "top_k",
    )


def _moe_layers(config: Dict[str, Any], layer_count: int) -> List[int]:
    experts = _expert_count(config)
    if experts is None or experts <= 1 or layer_count <= 0:
        return []

    explicit = config.get("moe_layers")
    if isinstance(explicit, list):
        if _is_binary_layer_mask(explicit, layer_count):
            return [index for index, value in enumerate(explicit[:layer_count]) if bool(_as_int(value))]
        return sorted(index for index in (_as_int(value) for value in explicit) if index is not None)

    mlp_layer_types = config.get("mlp_layer_types")
    if isinstance(mlp_layer_types, list):
        return [
            index
            for index, value in enumerate(mlp_layer_types[:layer_count])
            if str(value).lower() in ("sparse", "moe")
        ]

    mlp_only_layers = config.get("mlp_only_layers")
    if isinstance(mlp_only_layers, list):
        dense_layers = {index for index in (_as_int(value) for value in mlp_only_layers) if index is not None}
        return [index for index in range(layer_count) if index not in dense_layers]

    moe_layer_freq_raw = config.get("moe_layer_freq")
    if isinstance(moe_layer_freq_raw, list):
        if _is_binary_layer_mask(moe_layer_freq_raw, layer_count):
            return [index for index, value in enumerate(moe_layer_freq_raw[:layer_count]) if bool(_as_int(value))]
        return sorted(index for index in (_as_int(value) for value in moe_layer_freq_raw) if index is not None)

    if _is_hy_v3(str(config.get("model_type") or ""), config):
        first_dense = _pick_int(config, "first_k_dense_replace", "first_k_dense_layers")
        if first_dense is None:
            first_dense = 1
        return list(range(max(first_dense, 0), layer_count))

    moe_layer_freq = _pick_int(config, "moe_layer_freq")
    if moe_layer_freq is not None:
        first_moe_layer = _pick_int(config, "first_k_dense_replace", "first_k_dense_layers") or 0
        if moe_layer_freq <= 0:
            return []
        return [
            index
            for index in range(layer_count)
            if index >= first_moe_layer and index % moe_layer_freq == 0
        ]

    step = _pick_int(config, "interleave_moe_layer_step")
    if step is None:
        step = 1
    if step <= 0:
        return []
    return list(range(step - 1, layer_count, step))


def _layer_uses_rope(config: Dict[str, Any], layer_index: int) -> Optional[bool]:
    explicit = config.get("no_rope_layers")
    if isinstance(explicit, list) and len(explicit) > layer_index:
        value = _as_int(explicit[layer_index])
        if value is not None:
            return bool(value)

    no_rope_interval = _pick_int(config, "no_rope_layer_interval")
    if no_rope_interval is None and str(config.get("model_type", "")).lower().startswith("llama4"):
        no_rope_interval = 4
    if no_rope_interval is not None and no_rope_interval > 0:
        return (layer_index + 1) % no_rope_interval != 0

    return None


def _modalities(config: Dict[str, Any], vision_config: Optional[Dict[str, Any]]) -> List[str]:
    modalities = ["text"]
    if vision_config:
        modalities.append("image")
    if isinstance(config.get("audio_config"), dict):
        modalities.append("audio")
    if config.get("video_token_id") is not None:
        modalities.append("video")
    return modalities


def _notes(
    config: Dict[str, Any],
    text_config: Dict[str, Any],
    vision_config: Optional[Dict[str, Any]],
) -> List[str]:
    notes = []
    if vision_config:
        notes.append("Vision config detected; architecture is multimodal with image features fused into the text model.")
    experts = _expert_count(text_config)
    if experts is not None and experts > 1:
        moe_layers = _moe_layers(text_config, _pick_int(text_config, "num_hidden_layers", "n_layer", "num_layers", "n_layers") or 0)
        if _pick_int(text_config, "interleave_moe_layer_step"):
            notes.append("MoE detected with interleaved sparse/dense layers from interleave_moe_layer_step.")
        else:
            notes.append("MoE detected from config expert fields.")
        if moe_layers:
            notes.append("MoE layer indices: %s" % _compact_indices(moe_layers))
    no_rope_layers = text_config.get("no_rope_layers")
    if isinstance(no_rope_layers, list) and not no_rope_layers and str(text_config.get("model_type", "")).lower().startswith("llama4"):
        notes.append("Llama 4 empty no_rope_layers interpreted with the Transformers default NoPE interval of 4.")
    if _is_bloom(str(config.get("model_type") or ""), text_config):
        notes.append("BLOOM detected: attention uses fused query_key_value projection and ALiBi positional bias instead of RoPE.")
    if str(text_config.get("model_type") or "").lower().startswith("gpt_oss"):
        notes.append("GPT-OSS detected: MoE expert blocks use MXFP4 quantized checkpoint tensors while attention/router stay unconverted.")
    if _is_deepseek_v4(str(config.get("model_type") or ""), text_config):
        notes.append("DeepSeek V4 source model.py detected: blocks use Hyper-Connections, MLA sparse attention, and MoE FFNs.")
        if _pick_int(text_config, "num_hash_layers"):
            notes.append("First %s layers use hash-based expert indices instead of score top-k routing." % _pick_int(text_config, "num_hash_layers"))
        if _pick_int(text_config, "num_nextn_predict_layers"):
            notes.append("MTP detected: %s next-token prediction layer(s) after the decoder stack." % _pick_int(text_config, "num_nextn_predict_layers"))
    if _is_kimi_k25(str(config.get("model_type") or ""), text_config):
        notes.append("Kimi K2.5 detected: multimodal wrapper uses a DeepSeek-V3-style text decoder plus MoonViT vision tower.")
        notes.append("Text attention uses q LoRA and compressed KV MLA; K/V are expanded into dense causal attention after RoPE.")
        notes.append("Routed MoE expert weights are represented as int4-packed logical kernels from the safetensors index; shared experts stay dense.")
    if _is_glm_moe_dsa(str(config.get("model_type") or ""), text_config):
        notes.append("GLM MoE DSA detected: attention uses MLA-style q/kv low-rank projections with dynamic sparse attention.")
        notes.append("IndexShare detected from indexer_types/index_topk_freq; shared indexer layers reuse sparse indices from full indexer layers.")
        if _pick_int(text_config, "num_nextn_predict_layers"):
            notes.append("MTP detected: %s next-token prediction layer(s) after the decoder stack." % _pick_int(text_config, "num_nextn_predict_layers"))
    if _is_hy_v3(str(config.get("model_type") or ""), text_config):
        notes.append("Hy3 detected: decoder uses split-QKV GQA with per-head Q/K RMSNorm before RoPE.")
        notes.append("Hy3 MoE defaults to one dense prefix layer followed by sparse layers with sigmoid routing and expert correction bias.")
        notes.append("Hy3 sparse FFNs combine routed experts with a shared SwiGLU MLP; router weights are normalized and scaled by router_scaling_factor.")
        if _pick_int(text_config, "num_nextn_predict_layers"):
            notes.append("MTP detected: %s next-token prediction layer(s); the upstream Transformers loader ignores model.layers.80.* unexpected keys for now." % _pick_int(text_config, "num_nextn_predict_layers"))
    if _is_mimo_v2(str(config.get("model_type") or ""), text_config):
        projection_layout = _mimo_v2_projection_layout(text_config)
        projection_note = "fused-QKV" if projection_layout == "fused_qkv" else "split-QKV"
        notes.append("MiMo V2 detected: language blocks use %s GQA with hybrid full/sliding-window attention." % projection_note)
        notes.append("MiMo V2 attention applies partial RoPE, optional attention sink bias, value scaling, and separate SWA RoPE/head settings.")
        notes.append("MiMo V2 MoE uses sigmoid noaux_tc routing with expert correction bias, group top-k selection, and normalized top-k weights.")
        if isinstance(config.get("audio_config"), dict):
            notes.append("Audio config detected; audio codes are projected into the text hidden size and fused through placeholder tokens.")
    if config.get("auto_map"):
        notes.append("auto_map detected; remote modeling code may contain additional implementation details. This tool does not execute remote code.")
    return notes


def _compact_indices(indices: List[int]) -> str:
    if len(indices) <= 16:
        return ", ".join(str(index) for index in indices)
    return "%s, ... %s" % (
        ", ".join(str(index) for index in indices[:8]),
        ", ".join(str(index) for index in indices[-4:]),
    )


def _pick_int(config: Dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = config.get(key)
        parsed = _as_int(value)
        if parsed is not None:
            return parsed
    return None


def _mimo_layer_is_swa(text_config: Dict[str, Any], layer_index: int) -> bool:
    pattern = text_config.get("hybrid_layer_pattern")
    if isinstance(pattern, list) and layer_index < len(pattern):
        return bool(_as_int(pattern[layer_index]))
    return False


def _mimo_attention_heads(text_config: Dict[str, Any], is_swa: bool) -> Optional[int]:
    if is_swa:
        return _pick_int(text_config, "swa_num_attention_heads", "num_attention_heads", "n_head", "num_heads")
    return _pick_int(text_config, "num_attention_heads", "n_head", "num_heads")


def _mimo_kv_heads(text_config: Dict[str, Any], is_swa: bool) -> Optional[int]:
    if is_swa:
        return _pick_int(text_config, "swa_num_key_value_heads", "num_key_value_heads", "num_kv_heads", "n_kv_heads")
    return _pick_int(text_config, "num_key_value_heads", "num_kv_heads", "n_kv_heads")


def _mimo_head_dim(text_config: Dict[str, Any], is_swa: bool) -> Optional[int]:
    if is_swa:
        return _pick_int(text_config, "swa_head_dim", "head_dim") or _head_dim(text_config)
    return _pick_int(text_config, "head_dim") or _head_dim(text_config)


def _mimo_v_head_dim(text_config: Dict[str, Any], is_swa: bool, head_dim: Optional[int]) -> Optional[int]:
    if is_swa:
        return _pick_int(text_config, "swa_v_head_dim", "v_head_dim") or head_dim
    return _pick_int(text_config, "v_head_dim") or head_dim


def _mimo_kv_groups(heads: Optional[int], kv_heads: Optional[int]) -> Optional[int]:
    if heads is None or kv_heads in (None, 0):
        return None
    return heads // kv_heads


def _is_binary_layer_mask(values: List[Any], layer_count: int) -> bool:
    if len(values) < layer_count:
        return False
    parsed = [_as_int(value) for value in values[:layer_count]]
    return all(value in (0, 1) for value in parsed)


def _product_int(left: Optional[int], right: Optional[int]) -> Optional[int]:
    if left is None or right is None:
        return None
    return left * right


def _sum_ints(*values: Optional[int]) -> Optional[int]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none(inner)
            for key, inner in value.items()
            if inner is not None and inner != {} and inner != []
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value if item is not None]
    return value
