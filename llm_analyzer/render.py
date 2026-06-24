import json
from typing import Any, Dict, List, Optional

from .schema import Architecture, Layer


def render_json(architecture: Architecture) -> str:
    return json.dumps(architecture.to_dict(), indent=2, sort_keys=True)


def render_summary(architecture: Architecture) -> str:
    summary = architecture.summary
    lines = [
        "Model: %s" % architecture.model_id,
        "Revision: %s" % architecture.revision,
        "Type: %s" % architecture.model_type,
        "Architectures: %s" % (", ".join(architecture.architectures) if architecture.architectures else "unknown"),
        "Modalities: %s" % ", ".join(summary.get("modalities", [])),
        "Text layers: %s x %s" % (
            architecture.text_decoder.get("layers", 0),
            architecture.text_decoder.get("layer_type", "DecoderLayer"),
        ),
    ]

    for key in ("hidden_size", "intermediate_size", "attention_heads", "kv_heads", "vocab_size", "max_position_embeddings"):
        if key in summary:
            lines.append("%s: %s" % (key, summary[key]))

    moe = summary.get("moe") or {}
    if moe.get("enabled"):
        lines.append(
            "MoE: %s experts, %s active per token, %s MoE layers, %s dense MLP layers"
            % (
                moe.get("experts", "unknown"),
                moe.get("activated_experts", "unknown"),
                moe.get("moe_layers", "unknown"),
                moe.get("dense_mlp_layers", "unknown"),
            )
        )

    rope = summary.get("rope") or {}
    if rope.get("enabled"):
        rope_bits = []
        if "theta" in rope:
            rope_bits.append("theta=%s" % rope["theta"])
        if "scaling" in rope:
            rope_bits.append("scaling=%s" % rope["scaling"])
        lines.append("RoPE: %s" % (", ".join(rope_bits) if rope_bits else "enabled"))

    attention = architecture.text_decoder.get("attention") or {}
    if attention.get("type") and attention.get("type") != "MHA/GQA":
        bits = ["type=%s" % attention["type"]]
        for key in ("q_lora_rank", "kv_lora_rank", "qk_head_dim", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "o_lora_rank", "o_groups", "sliding_window", "index_topk"):
            if key in attention:
                bits.append("%s=%s" % (key, attention[key]))
        lines.append("Attention: %s" % ", ".join(bits))

    hc = summary.get("hyper_connections") or {}
    if hc:
        lines.append(
            "Hyper-Connections: hc_mult=%s, sinkhorn_iters=%s"
            % (_value(hc.get("hc_mult")), _value(hc.get("hc_sinkhorn_iters")))
        )

    mtp = summary.get("mtp") or {}
    if mtp:
        lines.append("MTP: %s next-token prediction layer(s)" % _value(mtp.get("nextn_predict_layers")))

    if architecture.vision_encoder:
        vision = architecture.vision_encoder
        lines.append(
            "Vision encoder: %s layers, hidden_size=%s"
            % (vision.get("layers", "unknown"), vision.get("hidden_size", "unknown"))
        )

    lines.append("Metadata files downloaded: %d" % len(architecture.files))
    lines.append("Weight files skipped: %d" % len(architecture.skipped_weight_files))

    if architecture.notes:
        lines.append("")
        lines.append("Notes:")
        lines.extend("- %s" % note for note in architecture.notes)

    return "\n".join(lines)


def render_mermaid_model(architecture: Architecture) -> str:
    summary = architecture.summary
    text = architecture.text_decoder
    lines = ["flowchart TD"]

    if architecture.vision_encoder:
        vision = architecture.vision_encoder
        vision_label = "Vision encoder\\n%s layers\\nhidden=%s" % (
            _value(vision.get("layers")),
            _value(vision.get("hidden_size")),
        )
        if vision.get("patch_size") is not None:
            vision_label += "\\npatch=%s" % _value(vision.get("patch_size"))
        if vision.get("projector_type"):
            vision_label += "\\nprojector=%s" % _value(vision.get("projector_type"))
        lines.append('  image["Image input"] --> vision["%s"]' % _escape(vision_label))
        fusion_label = "Multimodal projection / early fusion"
        if vision.get("merge_type"):
            fusion_label += "\\nmerge=%s" % _value(vision.get("merge_type"))
        lines.append('  vision --> fusion["%s"]' % _escape(fusion_label))
        lines.append('  tokens["Text token IDs"] --> embed["Token embedding\\nvocab=%s\\nhidden=%s"]' % (
            _value(summary.get("vocab_size")),
            _value(summary.get("hidden_size")),
        ))
        lines.append("  fusion --> decoder_input")
        lines.append("  embed --> decoder_input")
        lines.append('  decoder_input["Decoder input sequence"] --> layers')
    else:
        lines.append('  tokens["Token IDs"] --> embed["Token embedding\\nvocab=%s\\nhidden=%s"]' % (
            _value(summary.get("vocab_size")),
            _value(summary.get("hidden_size")),
        ))
        lines.append("  embed --> layers")

    layer_label = "%s x %s" % (
        text.get("layers", 0),
        text.get("layer_type", "DecoderLayer"),
    )
    moe = summary.get("moe") or {}
    if moe.get("enabled"):
        layer_label += "\\n%s MoE layers / %s dense MLP layers" % (
            _value(moe.get("moe_layers")),
            _value(moe.get("dense_mlp_layers")),
        )
        layer_label += "\\nMoE: %s experts, top-%s" % (
            _value(moe.get("experts")),
            _value(moe.get("activated_experts")),
        )
    layer_label += "\\nheads=%s kv_heads=%s" % (
        _value(summary.get("attention_heads")),
        _value(summary.get("kv_heads")),
    )
    attention = text.get("attention") or {}
    if attention.get("type") and attention.get("type") != "MHA/GQA":
        layer_label += "\\n%s" % attention.get("type")
    hc = text.get("hyper_connections") or {}
    if hc:
        layer_label += "\\nHC x%s" % _value(hc.get("hc_mult"))
    lines.append('  layers["%s"] --> final_norm["Final norm"]' % _escape(layer_label))
    lines.append('  final_norm --> lm_head["LM head"]')
    lines.append('  lm_head --> logits["Logits"]')
    mtp = text.get("mtp") or {}
    if mtp:
        lines.append('  layers -.-> mtp["MTP\\n%s next-token layer(s)"]' % _value(mtp.get("nextn_predict_layers")))

    return "\n".join(lines) + "\n"


def render_mermaid_layer(architecture: Architecture, layer_index: int = 0) -> str:
    layer = _get_layer(architecture, layer_index)

    component_ids = []
    lines = ["flowchart TD"]
    for index, component in enumerate(layer.components):
        node_id = "n%d" % index
        component_ids.append(node_id)
        details = _detail_lines(component.details)
        label = "%s\\n%s" % (component.name, component.kind)
        if details:
            label += "\\n" + "\\n".join(details)
        lines.append('  %s["%s"]' % (node_id, _escape(label)))

    names = {component.name: component_ids[index] for index, component in enumerate(layer.components)}
    has_moe_branch = _has_moe_branch(layer)

    for left_index, right_index in zip(range(len(component_ids) - 1), range(1, len(component_ids))):
        left_component = layer.components[left_index]
        right_component = layer.components[right_index]
        if has_moe_branch and left_component.name == "expert_gate/up/down_proj" and right_component.name == "shared_expert_gate/up/down_proj":
            continue
        if has_moe_branch and left_component.name == "shared_expert_gate/up/down_proj" and right_component.name == "combine_experts":
            continue
        lines.append("  %s --> %s" % (component_ids[left_index], component_ids[right_index]))

    if has_moe_branch:
        post_norm = names.get("post_attention_layernorm") or names.get("ffn_norm")
        shared = names.get("shared_expert_gate/up/down_proj")
        if post_norm and shared:
            lines.append("  %s --> %s" % (post_norm, names["shared_expert_gate/up/down_proj"]))
        lines.append("  %s --> %s" % (names["expert_gate/up/down_proj"], names["combine_experts"]))
        if shared:
            lines.append("  %s --> %s" % (shared, names["combine_experts"]))

    lines.insert(1, '  title["%s: %s"]' % (_escape(layer.name), _escape(layer.layer_type)))
    lines.insert(2, "  title -.-> n0")
    return "\n".join(lines) + "\n"


def render_mermaid_attention(architecture: Architecture, layer_index: int = 0) -> str:
    layer = _get_layer(architecture, layer_index)
    if _component_details(layer, "mla_sparse_attention"):
        return _render_mermaid_deepseek_mla(architecture, layer)
    if _component_details(layer, "bloom_attention"):
        return _render_mermaid_bloom_attention(architecture, layer)
    if _component_details(layer, "kimi_mla_attention"):
        return _render_mermaid_kimi_mla(architecture, layer)
    if _component_details(layer, "glm_dsa_attention"):
        return _render_mermaid_glm_dsa(architecture, layer)

    summary = architecture.summary
    attention = architecture.text_decoder.get("attention", {})
    qkv = _component_details(layer, "q_proj/k_proj/v_proj")
    rope = _component_details(layer, "rotary_position_embedding")
    kernel = _component_details(layer, "attention")

    hidden_size = summary.get("hidden_size")
    heads = qkv.get("attention_heads") or summary.get("attention_heads")
    kv_heads = qkv.get("kv_heads") or summary.get("kv_heads") or heads
    head_dim = qkv.get("head_dim") or attention.get("head_dim")
    kv_groups = None
    try:
        if heads and kv_heads:
            kv_groups = int(heads) // int(kv_heads)
    except (TypeError, ValueError, ZeroDivisionError):
        kv_groups = None

    rope_kind = "RoPE" if rope.get("layer_uses_rope", rope.get("enabled", False)) else "NoPE"
    qk_norm = attention.get("use_qk_norm")

    lines = [
        "flowchart TD",
        '  title["%s attention detail"]' % _escape(layer.name),
        '  x["Layer input\\nhidden=%s"] --> norm["Input norm"]' % _value(hidden_size),
        '  norm --> qkv["Q/K/V projections\\nQ: hidden -> heads * head_dim\\nK,V: hidden -> kv_heads * head_dim\\nheads=%s kv_heads=%s head_dim=%s"]'
        % (_value(heads), _value(kv_heads), _value(head_dim)),
        '  qkv --> q["Q reshape\\n[B,T,%s,%s]"]' % (_value(heads), _value(head_dim)),
        '  qkv --> k["K reshape\\n[B,T,%s,%s]"]' % (_value(kv_heads), _value(head_dim)),
        '  qkv --> v["V reshape\\n[B,T,%s,%s]"]' % (_value(kv_heads), _value(head_dim)),
    ]

    if qk_norm:
        lines.extend(
            [
                '  q --> qnorm["QK norm on Q"]',
                '  k --> knorm["QK norm on K"]',
                "  qnorm --> pos",
                "  knorm --> pos",
            ]
        )
    else:
        lines.extend(["  q --> pos", "  k --> pos"])

    rope_label = "%s\\ntheta=%s" % (rope_kind, _value(rope.get("theta")))
    if rope.get("no_rope_layer_interval") is not None:
        rope_label += "\\nno_rope_interval=%s" % rope.get("no_rope_layer_interval")
    lines.extend(
        [
            '  pos["%s"]' % _escape(rope_label),
            '  pos --> cache["KV cache update / repeat_kv\\ngroups=%s"]' % _value(kv_groups),
            "  v --> cache",
            '  cache --> scores["scores = Q K^T * scale"]',
        ]
    )

    mask_bits = ["causal mask"]
    if kernel.get("attention_chunk_size") is not None:
        mask_bits.append("chunk=%s" % kernel.get("attention_chunk_size"))
    if kernel.get("sliding_window") is not None:
        mask_bits.append("sliding_window=%s" % kernel.get("sliding_window"))
    lines.extend(
        [
            '  scores --> mask["%s"]' % _escape("\\n".join(mask_bits)),
            '  mask --> softmax["softmax"]',
            '  softmax --> dropout["dropout=%s"]' % _value(kernel.get("attention_dropout")),
            '  dropout --> context["context = probs V"]',
            '  context --> reshape["transpose / reshape\\n[B,T,hidden]"]',
            '  reshape --> out["o_proj\\nheads * head_dim -> hidden"]',
            '  out --> residual["residual add"]',
        ]
    )

    return "\n".join(lines) + "\n"


def _render_mermaid_deepseek_mla(architecture: Architecture, layer: Layer) -> str:
    summary = architecture.summary
    details = _component_details(layer, "mla_sparse_attention")
    hidden_size = summary.get("hidden_size")
    heads = details.get("attention_heads") or summary.get("attention_heads")
    head_dim = details.get("head_dim") or architecture.text_decoder.get("attention", {}).get("head_dim")
    rope_dim = details.get("qk_rope_head_dim")
    nope_dim = details.get("qk_nope_head_dim")
    compress_ratio = details.get("compress_ratio")

    lines = [
        "flowchart TD",
        '  title["%s DeepSeek V4 MLA detail"]' % _escape(layer.name),
        '  x["HC-pre attention state\\nhidden=%s"] --> norm["attn_norm RMSNorm"]' % _value(hidden_size),
        '  norm --> wqa["wq_a\\nhidden -> q_lora_rank=%s"]' % _value(details.get("q_lora_rank")),
        '  wqa --> qnorm["q_norm RMSNorm"]',
        '  qnorm --> wqb["wq_b\\nq_lora -> heads * head_dim\\nheads=%s head_dim=%s"]'
        % (_value(heads), _value(head_dim)),
        '  wqb --> qsplit["reshape Q\\n[B,S,%s,%s]\\nNoPE=%s RoPE=%s"]'
        % (_value(heads), _value(head_dim), _value(nope_dim), _value(rope_dim)),
        '  qsplit --> qscale["RMS scale Q per head"]',
        '  qscale --> qrope["RoPE on Q tail dims"]',
        '  norm --> wkv["wkv\\nhidden -> compact KV head_dim=%s"]' % _value(head_dim),
        '  wkv --> kvnorm["kv_norm RMSNorm"]',
        '  kvnorm --> kvrope["RoPE on KV tail dims"]',
        '  kvrope --> kvquant["FP8 activation quant\\nnon-RoPE dims only"]',
        '  kvquant --> win["window KV cache\\nsliding_window=%s"]' % _value(details.get("sliding_window")),
        '  win --> topk["window top-k indices"]',
    ]
    if compress_ratio:
        lines.extend(
            [
                '  norm --> comp["Compressor\\nratio=%s\\ngated pooling over tokens"]' % _value(compress_ratio),
                '  comp --> ckv["compressed KV cache\\nmode=%s"]' % _value(details.get("compress_mode")),
                '  qnorm --> idxq["Indexer wq_b\\nindex_heads=%s index_dim=%s"]'
                % (_value(details.get("index_n_heads")), _value(details.get("index_head_dim"))),
                '  idxq --> idxscore["index_score = ReLU(Q Kc^T) * weights"]',
                '  ckv --> idxscore',
                '  idxscore --> ctopk["compressed top-k\\nindex_topk=%s"]' % _value(details.get("index_topk")),
                '  topk --> cat["concat window + compressed indices"]',
                '  ctopk --> cat',
                '  ckv --> sparse',
                '  cat --> sparse',
            ]
        )
    else:
        lines.append("  topk --> sparse")

    lines.extend(
        [
            '  qrope --> sparse["sparse_attn\\nonline softmax + attn_sink\\nscale=1/sqrt(head_dim)"]',
            '  kvquant --> sparse',
            '  sparse --> invrope["inverse RoPE on output tail dims"]',
            '  invrope --> group["group heads\\no_groups=%s"]' % _value(details.get("o_groups")),
            '  group --> woa["wo_a grouped low-rank\\nhead groups -> o_lora_rank=%s"]' % _value(details.get("o_lora_rank")),
            '  woa --> wob["wo_b\\no_lora -> hidden"]',
            '  wob --> hcpost["HC-post attention state"]',
        ]
    )
    return "\n".join(lines) + "\n"


def _render_mermaid_bloom_attention(architecture: Architecture, layer: Layer) -> str:
    summary = architecture.summary
    details = _component_details(layer, "bloom_attention")
    hidden_size = summary.get("hidden_size")
    heads = details.get("attention_heads") or summary.get("attention_heads")
    head_dim = details.get("head_dim") or architecture.text_decoder.get("attention", {}).get("head_dim")
    lines = [
        "flowchart TD",
        '  title["%s BLOOM attention detail"]' % _escape(layer.name),
        '  x["Layer input\\nhidden=%s"] --> norm["input_layernorm LayerNorm"]' % _value(hidden_size),
        '  norm --> qkv["query_key_value fused linear\\nhidden -> 3 * hidden\\nheads=%s head_dim=%s"]'
        % (_value(heads), _value(head_dim)),
        '  qkv --> split["split Q / K / V"]',
        '  split --> q["Q reshape\\n[B,T,%s,%s]"]' % (_value(heads), _value(head_dim)),
        '  split --> k["K reshape\\n[B,T,%s,%s]"]' % (_value(heads), _value(head_dim)),
        '  split --> v["V reshape\\n[B,T,%s,%s]"]' % (_value(heads), _value(head_dim)),
        '  q --> scores["scores = Q K^T / sqrt(head_dim)"]',
        '  k --> kcache["K cache"]',
        '  v --> vcache["V cache"]',
        '  kcache --> scores',
        '  scores --> alibi["add ALiBi positional bias"]',
        '  alibi --> mask["causal mask"]',
        '  mask --> softmax["softmax\\nfp32=%s"]' % _value(details.get("attention_softmax_in_fp32")),
        '  softmax --> dropout["dropout=%s"]' % _value(details.get("attention_dropout")),
        '  dropout --> ctx["context = probs V"]',
        '  vcache --> ctx',
        '  ctx --> merge["merge heads"]',
        '  merge --> out["self_attention.dense\\nheads * head_dim -> hidden"]',
        '  out --> residual["residual add"]',
    ]
    return "\n".join(lines) + "\n"


def _render_mermaid_glm_dsa(architecture: Architecture, layer: Layer) -> str:
    summary = architecture.summary
    details = _component_details(layer, "glm_dsa_attention")
    hidden_size = summary.get("hidden_size")
    heads = details.get("attention_heads") or summary.get("attention_heads")
    qk_dim = details.get("qk_head_dim") or details.get("head_dim")
    v_dim = details.get("v_head_dim")
    indexer_type = details.get("indexer_type")

    lines = [
        "flowchart TD",
        '  title["%s GLM DSA attention detail"]' % _escape(layer.name),
        '  x["Layer input\\nhidden=%s"] --> norm["input_layernorm RMSNorm"]' % _value(hidden_size),
        '  norm --> qa["q_a_proj\\nhidden -> q_lora_rank=%s"]' % _value(details.get("q_lora_rank")),
        '  qa --> qan["q_a_layernorm"]',
        '  qan --> qb["q_b_proj\\nq_lora -> heads * qk_head_dim\\nheads=%s qk=%s"]'
        % (_value(heads), _value(qk_dim)),
        '  qb --> qsplit["split Q\\nNoPE=%s RoPE=%s"]'
        % (_value(details.get("qk_nope_head_dim")), _value(details.get("qk_rope_head_dim"))),
        '  qsplit --> qrope["RoPE Q\\ninterleave=%s"]' % _value(details.get("rope_interleave")),
        '  norm --> kva["kv_a_proj_with_mqa\\nhidden -> kv_lora + RoPE key\\nkv_lora=%s"]'
        % _value(details.get("kv_lora_rank")),
        '  kva --> kvlat["split KV latent"]',
        '  kva --> krope["split MQA RoPE key"]',
        '  kvlat --> kvnorm["kv_a_layernorm"]',
        '  kvnorm --> kvb["kv_b_proj\\nkv_lora -> heads * (NoPE key + V)\\nV=%s"]' % _value(v_dim),
        '  kvb --> ksplit["split K NoPE and V"]',
        '  krope --> krope2["RoPE K + expand across heads"]',
        '  ksplit --> kcat["concat K NoPE + K RoPE"]',
        '  krope2 --> kcat',
        '  ksplit --> vcache["V cache"]',
        '  kcat --> kcache["K cache"]',
    ]
    if indexer_type == "shared":
        lines.extend(
            [
                '  qan --> idx["IndexShare reuse\\nindexer_type=shared\\nindex_topk=%s"]' % _value(details.get("index_topk")),
                '  idx --> dsa',
            ]
        )
    else:
        lines.extend(
            [
                '  qan --> iq["indexer wq_b\\nindex_heads=%s index_dim=%s"]'
                % (_value(details.get("index_n_heads")), _value(details.get("index_head_dim"))),
                '  norm --> ik["indexer wk + k_norm"]',
                '  norm --> iw["indexer weights_proj"]',
                '  iq --> iscore["index_score = ReLU(Q K^T) * weights"]',
                '  ik --> iscore',
                '  iw --> iscore',
                '  iscore --> idx["top-k sparse indices\\nindex_topk=%s"]' % _value(details.get("index_topk")),
                '  idx --> dsa',
            ]
        )
    lines.extend(
        [
            '  qrope --> dsa["dynamic sparse attention\\nQK dim=%s V dim=%s"]' % (_value(qk_dim), _value(v_dim)),
            '  kcache --> dsa',
            '  vcache --> dsa',
            '  dsa --> merge["merge heads\\nheads * V -> packed"]',
            '  merge --> out["o_proj\\npacked -> hidden"]',
            '  out --> residual["residual add"]',
        ]
    )
    return "\n".join(lines) + "\n"


def _render_mermaid_kimi_mla(architecture: Architecture, layer: Layer) -> str:
    summary = architecture.summary
    details = _component_details(layer, "kimi_mla_attention")
    hidden_size = summary.get("hidden_size")
    heads = details.get("attention_heads") or summary.get("attention_heads")
    qk_dim = details.get("qk_head_dim") or details.get("head_dim")
    v_dim = details.get("v_head_dim")

    lines = [
        "flowchart TD",
        '  title["%s Kimi K2.5 MLA attention detail"]' % _escape(layer.name),
        '  x["Layer input\\nhidden=%s"] --> norm["input_layernorm RMSNorm"]' % _value(hidden_size),
        '  norm --> qa["q_a_proj\\nhidden -> q_lora_rank=%s"]' % _value(details.get("q_lora_rank")),
        '  qa --> qan["q_a_layernorm"]',
        '  qan --> qb["q_b_proj\\nq_lora -> heads * qk_head_dim\\nheads=%s qk=%s"]'
        % (_value(heads), _value(qk_dim)),
        '  qb --> qsplit["split Q\\nNoPE=%s RoPE=%s"]'
        % (_value(details.get("qk_nope_head_dim")), _value(details.get("qk_rope_head_dim"))),
        '  qsplit --> qrope["RoPE on Q tail\\ntheta=%s"]' % _value((details.get("rope") or {}).get("theta")),
        '  qrope --> qcat["concat Q NoPE + Q RoPE"]',
        '  norm --> kva["kv_a_proj_with_mqa\\nhidden -> kv_lora + RoPE key\\nkv_lora=%s"]'
        % _value(details.get("kv_lora_rank")),
        '  kva --> kvlat["split KV latent"]',
        '  kva --> krope["split single-head RoPE key"]',
        '  kvlat --> kvnorm["kv_a_layernorm"]',
        '  kvnorm --> kvb["kv_b_proj\\nkv_lora -> heads * (NoPE key + V)\\nV=%s"]' % _value(v_dim),
        '  kvb --> ksplit["split K NoPE and V"]',
        '  krope --> krope2["RoPE K + expand across heads"]',
        '  ksplit --> kcat["concat K NoPE + K RoPE"]',
        '  krope2 --> kcat',
        '  kcat --> kcache["K cache\\nQK dim=%s"]' % _value(qk_dim),
        '  ksplit --> vcache["V cache\\nV dim=%s"]' % _value(v_dim),
        '  qcat --> scores["scores = Q K^T / sqrt(qk_head_dim)"]',
        '  kcache --> scores',
        '  scores --> mask["causal mask"]',
        '  mask --> softmax["softmax fp32 then cast back"]',
        '  softmax --> dropout["dropout=%s"]' % _value(details.get("attention_dropout")),
        '  dropout --> ctx["context = probs V"]',
        '  vcache --> ctx',
        '  ctx --> merge["merge heads\\nheads * V -> packed"]',
        '  merge --> out["o_proj\\npacked -> hidden"]',
        '  out --> residual["residual add"]',
    ]
    return "\n".join(lines) + "\n"


def render_mermaid_mlp(architecture: Architecture, layer_index: int = 0) -> str:
    layer = _get_layer(architecture, layer_index)
    hidden_size = architecture.summary.get("hidden_size")
    is_moe = _has_moe_branch(layer)

    if is_moe:
        routed = _component_details(layer, "expert_gate/up/down_proj")
        shared = _component_details(layer, "shared_expert_gate/up/down_proj")
        lines = [
            "flowchart TD",
            '  title["%s MLP primitive detail"]' % _escape(layer.name),
            '  x["Post-attention hidden states\\nhidden=%s"] --> split["Used by routed experts%s"]'
            % (_value(hidden_size), " and shared expert" if shared else ""),
            '  split --> routed_gate["Routed expert gate/up projection\\nexperts=%s\\nintermediate=%s"]'
            % (_value(routed.get("experts")), _value(routed.get("intermediate_size"))),
            '  routed_gate --> routed_act["%s(gate)"]' % _escape(_value(routed.get("activation"))),
            '  routed_gate --> routed_up["up branch"]',
            '  routed_act --> routed_mul["elementwise multiply"]',
            '  routed_up --> routed_mul',
            '  routed_mul --> routed_down["routed expert down projection\\nintermediate -> hidden"]',
        ]
        if shared:
            lines.extend(
                [
                    '  split --> shared_gate["Shared expert gate/up projection\\nintermediate=%s"]'
                    % _value(shared.get("intermediate_size")),
                    '  shared_gate --> shared_act["%s(gate)"]' % _escape(_value(shared.get("activation"))),
                    '  shared_gate --> shared_up["up branch"]',
                    '  shared_act --> shared_mul["elementwise multiply"]',
                    '  shared_up --> shared_mul',
                    '  shared_mul --> shared_down["shared expert down projection\\nintermediate -> hidden"]',
                    '  routed_down --> combine["combine with router weights"]',
                    '  shared_down --> combine',
                ]
            )
        else:
            lines.append('  routed_down --> combine["combine routed experts with router weights"]')
        lines.extend(
            [
                '  combine --> output["MLP output"]',
            ]
        )
        return "\n".join(lines) + "\n"

    dense = _component_details(layer, "gate_proj/up_proj")
    activation = _component_kind(layer, "activation")
    intermediate = dense.get("intermediate_size")
    lines = [
        "flowchart TD",
        '  title["%s dense MLP detail"]' % _escape(layer.name),
        '  x["Post-attention hidden states\\nhidden=%s"] --> gate["gate_proj\\nhidden -> %s"]'
        % (_value(hidden_size), _value(intermediate)),
        '  x --> up["up_proj\\nhidden -> %s"]' % _value(intermediate),
        '  gate --> act["%s(gate)"]' % _escape(_value(activation)),
        '  act --> mul["elementwise multiply"]',
        '  up --> mul',
        '  mul --> down["down_proj\\n%s -> hidden"]' % _value(intermediate),
        '  down --> residual["residual add"]',
    ]
    return "\n".join(lines) + "\n"


def render_mermaid_moe(architecture: Architecture, layer_index: int = 0) -> str:
    layer = _get_layer(architecture, layer_index)
    if not _has_moe_branch(layer):
        raise ValueError("Layer %d is not an MoE layer; use --level mlp or choose a sparse layer." % layer_index)

    hidden_size = architecture.summary.get("hidden_size")
    router = _component_details(layer, "router")
    routed = _component_details(layer, "expert_gate/up/down_proj")
    shared = _component_details(layer, "shared_expert_gate/up/down_proj")

    weight_label = "sigmoid / routing weights"
    if router.get("scoring_func"):
        weight_label = "%s / normalize / scale=%s" % (
            router.get("scoring_func"),
            _value(router.get("route_scale")),
        )
    topk_label = "top-k select\\nk=%s" % _value(router.get("activated_experts"))
    if router.get("hash_routing"):
        topk_label = "hash tid2eid lookup\\nk=%s" % _value(router.get("activated_experts"))

    input_label = "Post-attention hidden states"
    output_label = "residual add"
    if _component_details(layer, "hc_ffn_post"):
        input_label = "FFN-normalized HC-pre states"
        output_label = "HC-post FFN combine"

    lines = [
        "flowchart TD",
        '  title["%s MoE routing detail"]' % _escape(layer.name),
        '  x["%s\\n[B,T,%s]"] --> flat["flatten tokens\\n[B*T,hidden]"]' % (_escape(input_label), _value(hidden_size)),
        '  flat --> router["router linear\\nhidden -> %s experts"]' % _value(router.get("experts")),
        '  router --> topk["%s"]' % _escape(topk_label),
        '  topk --> score["%s"]' % _escape(weight_label),
        '  score --> dispatch["dispatch or scale token copies"]',
        '  dispatch --> experts["routed expert MLPs\\nexperts=%s\\nintermediate=%s\\nactivation=%s\\ndtype=%s"]'
        % (
            _value(routed.get("experts")),
            _value(routed.get("intermediate_size")),
            _value(routed.get("activation")),
            _value(routed.get("expert_dtype")),
        ),
        '  experts --> reduce["sum routed expert outputs"]',
        '  reduce --> add["add routed + shared"]',
        '  add --> output["reshape to [B,T,hidden]"]',
        '  output --> residual["%s"]' % _escape(output_label),
    ]
    if shared:
        lines.insert(
            -3,
            '  flat --> shared["shared expert MLP\\nintermediate=%s\\nactivation=%s"]'
            % (_value(shared.get("intermediate_size")), _value(shared.get("activation"))),
        )
        lines.insert(-3, '  shared --> add')
    else:
        lines[9] = '  experts --> add["routed expert weighted sum"]'
    if router.get("router_aux_loss_coef") is not None:
        lines.append('  router -.-> aux["router aux loss coef=%s"]' % router.get("router_aux_loss_coef"))
    return "\n".join(lines) + "\n"


def _get_layer(architecture: Architecture, layer_index: int) -> Layer:
    if not architecture.layers:
        raise ValueError("Architecture has no decoder layers.")
    if layer_index < 0 or layer_index >= len(architecture.layers):
        raise IndexError("Layer index %d out of range 0..%d" % (layer_index, len(architecture.layers) - 1))
    return architecture.layers[layer_index]


def _has_moe_branch(layer: Layer) -> bool:
    names = {component.name for component in layer.components}
    return all(
        name in names
        for name in (
            "router",
            "expert_gate/up/down_proj",
            "combine_experts",
        )
    )


def _component_details(layer: Layer, name: str) -> Dict[str, Any]:
    for component in layer.components:
        if component.name == name:
            return component.details or {}
    return {}


def _component_kind(layer: Layer, name: str) -> Optional[str]:
    for component in layer.components:
        if component.name == name:
            return component.kind
    return None


def _detail_lines(details: Dict[str, Any]) -> List[str]:
    if not details:
        return []
    lines = []
    for key in sorted(details.keys()):
        value = details[key]
        if isinstance(value, dict):
            rendered = ", ".join("%s=%s" % (inner_key, value[inner_key]) for inner_key in sorted(value.keys()))
        else:
            rendered = str(value)
        lines.append("%s=%s" % (key, rendered))
    return lines


def _value(value: Optional[Any]) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _escape(value: str) -> str:
    return value.replace('"', "'")
