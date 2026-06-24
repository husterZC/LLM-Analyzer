from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import Architecture, Layer


@dataclass
class TensorSpec:
    name: str
    shape: List[Any]
    dtype: str = "float32"
    description: str = ""


@dataclass
class KernelNode:
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelGraph:
    name: str
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]
    tensors: Dict[str, TensorSpec]
    nodes: List[KernelNode]


class KernelGraphBuilder:
    def __init__(self, name: str, hidden_size: Optional[int]) -> None:
        self.name = name
        self.hidden_size = hidden_size or "hidden"
        self.tensors: Dict[str, TensorSpec] = {}
        self.nodes: List[KernelNode] = []
        self.inputs: List[TensorSpec] = []
        self.outputs: List[TensorSpec] = []

    def tensor(
        self,
        name: str,
        shape: Optional[List[Any]] = None,
        dtype: str = "float32",
        description: str = "",
    ) -> str:
        if shape is None:
            shape = ["batch", "sequence", self.hidden_size]
        if name not in self.tensors:
            self.tensors[name] = TensorSpec(name=name, shape=shape, dtype=dtype, description=description)
        else:
            spec = self.tensors[name]
            if shape is not None:
                spec.shape = shape
            if dtype:
                spec.dtype = dtype
            if description:
                spec.description = description
        return name

    def add_input(
        self,
        name: str,
        shape: Optional[List[Any]] = None,
        dtype: str = "float32",
        description: str = "",
    ) -> str:
        tensor_name = self.tensor(name, shape, dtype=dtype, description=description)
        self.inputs.append(self.tensors[tensor_name])
        return tensor_name

    def add_output(self, name: str, shape: Optional[List[Any]] = None, description: str = "") -> str:
        tensor_name = self.tensor(name, shape, description=description)
        self.outputs.append(self.tensors[tensor_name])
        return tensor_name

    def node(
        self,
        name: str,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        output_shapes: Optional[List[List[Any]]] = None,
        output_dtypes: Optional[List[str]] = None,
        output_descriptions: Optional[List[str]] = None,
        formula: Optional[str] = None,
        description: Optional[str] = None,
        **attrs: Any,
    ) -> str:
        default_shape = self.tensors[inputs[0]].shape if inputs and inputs[0] in self.tensors else None
        for index, output in enumerate(outputs):
            shape = _pick_list(output_shapes, index) or default_shape
            dtype = _pick_list(output_dtypes, index) or "float32"
            tensor_description = _pick_list(output_descriptions, index) or _default_tensor_description(output, op_type)
            self.tensor(output, shape=shape, dtype=dtype, description=tensor_description)
        metadata = self._node_metadata(op_type, inputs, outputs, attrs, formula, description)
        self.nodes.append(
            KernelNode(
                name=name,
                op_type=op_type,
                inputs=inputs,
                outputs=outputs,
                attrs={key: value for key, value in {**attrs, **metadata}.items() if value is not None},
            )
        )
        return outputs[0]

    def _node_metadata(
        self,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attrs: Dict[str, Any],
        formula: Optional[str],
        description: Optional[str],
    ) -> Dict[str, Any]:
        input_links = [self._tensor_link(name) for name in inputs]
        output_links = [self._tensor_link(name) for name in outputs]
        return {
            "formula": formula or _formula_for_op(op_type, attrs),
            "description": description or _description_for_op(op_type),
            "input_links": input_links,
            "output_links": output_links,
            "input_dims": {link["name"]: link["shape"] for link in input_links},
            "output_dims": {link["name"]: link["shape"] for link in output_links},
        }

    def _tensor_link(self, name: str) -> Dict[str, Any]:
        spec = self.tensors.get(name)
        if spec is None:
            spec = TensorSpec(name=name, shape=["unknown"])
            self.tensors[name] = spec
        return {
            "name": spec.name,
            "shape": spec.shape,
            "dtype": spec.dtype,
            "description": spec.description,
        }

    def build(self) -> KernelGraph:
        return KernelGraph(
            name=self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            tensors=self.tensors,
            nodes=self.nodes,
        )


def build_kernel_graph(architecture: Architecture, level: str, layer_index: int) -> KernelGraph:
    if level == "model":
        raise ValueError("ONNX kernel-flow export supports --level layer, attention, mlp, or moe.")

    layer = _get_layer(architecture, layer_index)
    hidden_size = architecture.summary.get("hidden_size")
    builder = KernelGraphBuilder(
        name="%s_%s_layer_%d_%s" % (
            _sanitize(architecture.model_id),
            _sanitize(layer.layer_type),
            layer_index,
            level,
        ),
        hidden_size=hidden_size,
    )

    if level == "layer":
        _build_layer_flow(builder, architecture, layer)
    elif level == "attention":
        _build_attention_flow(builder, architecture, layer, standalone=True)
    elif level == "mlp":
        _build_mlp_flow(builder, architecture, layer, "mlp_input", standalone=True)
    elif level == "moe":
        if not _has_moe_branch(layer):
            raise ValueError("Layer %d is not an MoE layer." % layer.index)
        _build_moe_flow(builder, architecture, layer, "moe_input", standalone=True)
    else:
        raise ValueError("Unsupported ONNX kernel-flow level: %s" % level)

    return builder.build()


def _build_layer_flow(builder: KernelGraphBuilder, architecture: Architecture, layer: Layer) -> None:
    if _is_deepseek_v4_arch(architecture, layer):
        _build_deepseek_v4_layer_flow(builder, architecture, layer)
        return

    layer_input = builder.add_input(
        "layer_input",
        ["batch", "sequence", builder.hidden_size],
        description="Decoder layer input hidden states.",
    )
    norm0 = builder.node(
        "input_layernorm",
        _norm_op(layer),
        [layer_input],
        ["input_layernorm_out"],
        output_descriptions=["Normalized hidden states before attention."],
    )
    attention_out = _build_attention_core(builder, architecture, layer, norm0)
    residual0 = builder.node(
        "residual_add_attention",
        "ResidualAdd",
        [layer_input, attention_out],
        ["post_attention_residual"],
        output_descriptions=["Attention residual stream: layer_input + attention_output."],
    )
    norm1 = builder.node(
        "post_attention_layernorm",
        _norm_op(layer),
        [residual0],
        ["post_attention_layernorm_out"],
        output_descriptions=["Normalized hidden states before dense MLP or MoE FFN."],
    )
    if _has_moe_branch(layer):
        ffn_out = _build_moe_core(builder, architecture, layer, norm1)
    else:
        ffn_out = _build_dense_mlp_core(builder, architecture, layer, norm1)
    layer_output = builder.node(
        "residual_add_ffn",
        "ResidualAdd",
        [residual0, ffn_out],
        ["layer_output"],
        output_descriptions=["Decoder layer output hidden states."],
    )
    builder.add_output(layer_output)


def _build_deepseek_v4_layer_flow(builder: KernelGraphBuilder, architecture: Architecture, layer: Layer) -> None:
    hc = _component_details(layer, "hc_attn_pre") or architecture.summary.get("hyper_connections", {})
    hc_mult = hc.get("hc_mult") or "hc"
    layer_input = builder.add_input(
        "layer_input",
        ["batch", "sequence", hc_mult, builder.hidden_size],
        description="DeepSeek V4 block input with Hyper-Connection copies.",
    )
    attn_pre = _build_hc_pre(builder, layer_input, "attn", hc)
    attn_norm = builder.node(
        "attn_norm",
        _component_kind(layer, "attn_norm") or _norm_op(layer),
        [attn_pre],
        ["attn_norm_out"],
        output_shapes=[["batch", "sequence", builder.hidden_size]],
        output_descriptions=["Normalized single hidden stream before MLA attention."],
    )
    attention_out = _build_attention_core(builder, architecture, layer, attn_norm)
    attn_post = _build_hc_post(builder, attention_out, layer_input, "attn", hc)

    ffn_pre = _build_hc_pre(builder, attn_post, "ffn", hc)
    ffn_norm = builder.node(
        "ffn_norm",
        _component_kind(layer, "ffn_norm") or _norm_op(layer),
        [ffn_pre],
        ["ffn_norm_out"],
        output_shapes=[["batch", "sequence", builder.hidden_size]],
        output_descriptions=["Normalized single hidden stream before DeepSeek MoE FFN."],
    )
    if _has_moe_branch(layer):
        ffn_out = _build_moe_core(builder, architecture, layer, ffn_norm)
    else:
        ffn_out = _build_dense_mlp_core(builder, architecture, layer, ffn_norm)
    layer_output = _build_hc_post(builder, ffn_out, attn_post, "ffn", hc, output_name="layer_output")
    builder.add_output(layer_output)


def _build_hc_pre(builder: KernelGraphBuilder, input_name: str, branch: str, hc: Dict[str, Any]) -> str:
    return builder.node(
        "hc_%s_pre" % branch,
        "HyperConnectionPre",
        [input_name],
        ["hc_%s_pre_out" % branch],
        output_shapes=[["batch", "sequence", builder.hidden_size]],
        output_descriptions=["Hyper-Connection pre-mix output for the %s branch." % branch],
        hc_mult=hc.get("hc_mult"),
        sinkhorn_iters=hc.get("hc_sinkhorn_iters"),
        eps=hc.get("hc_eps"),
    )


def _build_hc_post(
    builder: KernelGraphBuilder,
    branch_output: str,
    residual: str,
    branch: str,
    hc: Dict[str, Any],
    output_name: Optional[str] = None,
) -> str:
    hc_mult = hc.get("hc_mult") or "hc"
    return builder.node(
        "hc_%s_post" % branch,
        "HyperConnectionPost",
        [branch_output, residual],
        [output_name or "hc_%s_post_out" % branch],
        output_shapes=[["batch", "sequence", hc_mult, builder.hidden_size]],
        output_descriptions=["Hyper-Connection post-mix output for the %s branch." % branch],
        hc_mult=hc.get("hc_mult"),
    )


def _build_attention_flow(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    standalone: bool = False,
) -> str:
    input_name = (
        builder.add_input(
            "attention_input",
            ["batch", "sequence", builder.hidden_size],
            description="Attention-detail graph input hidden states.",
        )
        if standalone
        else "attention_input"
    )
    norm = builder.node(
        "input_layernorm",
        _norm_op(layer),
        [input_name],
        ["input_layernorm_out"],
        output_descriptions=["Normalized hidden states before attention projections."],
    )
    attention_out = _build_attention_core(builder, architecture, layer, norm)
    if standalone:
        builder.add_output(attention_out)
    return attention_out


def _build_attention_core(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
) -> str:
    if _has_deepseek_mla(layer):
        return _build_deepseek_mla_core(builder, architecture, layer, input_name)
    if _has_bloom_attention(layer):
        return _build_bloom_attention_core(builder, architecture, layer, input_name)
    if _has_kimi_mla(layer):
        return _build_kimi_mla_core(builder, architecture, layer, input_name)
    if _has_glm_dsa(layer):
        return _build_glm_dsa_core(builder, architecture, layer, input_name)

    summary = architecture.summary
    attention = architecture.text_decoder.get("attention", {})
    qkv = _component_details(layer, "q_proj/k_proj/v_proj")
    rope = _component_details(layer, "rotary_position_embedding")
    kernel = _component_details(layer, "attention")

    hidden_size = summary.get("hidden_size")
    heads = qkv.get("attention_heads") or summary.get("attention_heads")
    kv_heads = qkv.get("kv_heads") or summary.get("kv_heads") or heads
    head_dim = qkv.get("head_dim") or attention.get("head_dim")
    q_shape = ["batch", "sequence", heads or "heads", head_dim or "head_dim"]
    kv_shape = ["batch", "sequence", kv_heads or "kv_heads", head_dim or "head_dim"]

    q = builder.node(
        "q_proj",
        "Linear",
        [input_name],
        ["q_proj_out"],
        output_shapes=[["batch", "sequence", _product(heads, head_dim) or "heads*head_dim"]],
        output_descriptions=["Packed query projection output before head reshape."],
        in_features=hidden_size,
        out_features=_product(heads, head_dim),
        logical_weight="q_proj.weight",
    )
    k = builder.node(
        "k_proj",
        "Linear",
        [input_name],
        ["k_proj_out"],
        output_shapes=[["batch", "sequence", _product(kv_heads, head_dim) or "kv_heads*head_dim"]],
        output_descriptions=["Packed key projection output before head reshape."],
        in_features=hidden_size,
        out_features=_product(kv_heads, head_dim),
        logical_weight="k_proj.weight",
    )
    v = builder.node(
        "v_proj",
        "Linear",
        [input_name],
        ["v_proj_out"],
        output_shapes=[["batch", "sequence", _product(kv_heads, head_dim) or "kv_heads*head_dim"]],
        output_descriptions=["Packed value projection output before head reshape."],
        in_features=hidden_size,
        out_features=_product(kv_heads, head_dim),
        logical_weight="v_proj.weight",
    )

    builder.tensor("q_heads", q_shape, description="Query heads as [batch, sequence, attention_heads, head_dim].")
    builder.tensor("k_heads", kv_shape, description="Key heads as [batch, sequence, kv_heads, head_dim].")
    builder.tensor("v_heads", kv_shape, description="Value heads as [batch, sequence, kv_heads, head_dim].")
    q_heads = builder.node(
        "reshape_q",
        "ReshapeHeads",
        [q],
        ["q_heads"],
        output_shapes=[q_shape],
        heads=heads,
        head_dim=head_dim,
    )
    k_heads = builder.node(
        "reshape_k",
        "ReshapeHeads",
        [k],
        ["k_heads"],
        output_shapes=[kv_shape],
        heads=kv_heads,
        head_dim=head_dim,
    )
    v_heads = builder.node(
        "reshape_v",
        "ReshapeHeads",
        [v],
        ["v_heads"],
        output_shapes=[kv_shape],
        heads=kv_heads,
        head_dim=head_dim,
    )

    q_pos, k_pos = q_heads, k_heads
    uses_rope = rope.get("layer_uses_rope", rope.get("enabled", False))
    if uses_rope:
        q_pos = builder.node(
            "apply_rope_q",
            "RoPE",
            [q_heads],
            ["q_rope"],
            output_shapes=[q_shape],
            output_descriptions=["Query heads after rotary position embedding."],
            theta=rope.get("theta"),
            no_rope_layer_interval=rope.get("no_rope_layer_interval"),
        )
        k_pos = builder.node(
            "apply_rope_k",
            "RoPE",
            [k_heads],
            ["k_rope"],
            output_shapes=[kv_shape],
            output_descriptions=["Key heads after rotary position embedding."],
            theta=rope.get("theta"),
            no_rope_layer_interval=rope.get("no_rope_layer_interval"),
        )
    else:
        q_pos = builder.node("nope_q", "NoPE", [q_heads], ["q_nope"], output_shapes=[q_shape])
        k_pos = builder.node("nope_k", "NoPE", [k_heads], ["k_nope"], output_shapes=[kv_shape])

    if attention.get("use_qk_norm"):
        q_pos = builder.node("qk_norm_q", "L2Norm", [q_pos], ["q_norm"], output_shapes=[q_shape])
        k_pos = builder.node("qk_norm_k", "L2Norm", [k_pos], ["k_norm"], output_shapes=[kv_shape])

    cache_shape = ["batch", "kv_sequence", kv_heads or "kv_heads", head_dim or "head_dim"]
    cache_k = builder.node("kv_cache_update_k", "KVCacheUpdate", [k_pos], ["k_cache"], output_shapes=[cache_shape])
    cache_v = builder.node("kv_cache_update_v", "KVCacheUpdate", [v_heads], ["v_cache"], output_shapes=[cache_shape])

    repeated_shape = ["batch", "kv_sequence", heads or "heads", head_dim or "head_dim"]
    repeated_k = builder.node(
        "repeat_k",
        "RepeatKV",
        [cache_k],
        ["k_repeated"],
        output_shapes=[repeated_shape],
        groups=_kv_groups(heads, kv_heads),
    )
    repeated_v = builder.node(
        "repeat_v",
        "RepeatKV",
        [cache_v],
        ["v_repeated"],
        output_shapes=[repeated_shape],
        groups=_kv_groups(heads, kv_heads),
    )
    scores = builder.node(
        "attention_scores",
        "MatMulQK",
        [q_pos, repeated_k],
        ["attention_scores"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
        scale="1/sqrt(head_dim)",
    )
    masked = builder.node(
        "causal_mask",
        "CausalMask",
        [scores],
        ["masked_scores"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
        attention_chunk_size=kernel.get("attention_chunk_size"),
        sliding_window=kernel.get("sliding_window"),
    )
    probs = builder.node(
        "softmax",
        "Softmax",
        [masked],
        ["attention_probs"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
        axis=-1,
    )
    dropped = builder.node(
        "attention_dropout",
        "Dropout",
        [probs],
        ["attention_probs_dropout"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
        ratio=kernel.get("attention_dropout"),
    )
    context = builder.node(
        "attention_context",
        "MatMulPV",
        [dropped, repeated_v],
        ["attention_context"],
        output_shapes=[["batch", "sequence", heads or "heads", head_dim or "head_dim"]],
    )
    merged = builder.node(
        "merge_heads",
        "MergeHeads",
        [context],
        ["attention_merged"],
        output_shapes=[["batch", "sequence", _product(heads, head_dim) or "heads*head_dim"]],
        hidden_size=hidden_size,
    )
    return builder.node(
        "o_proj",
        "Linear",
        [merged],
        ["attention_output"],
        output_shapes=[["batch", "sequence", hidden_size or "hidden"]],
        in_features=_product(heads, head_dim),
        out_features=hidden_size,
        logical_weight="o_proj.weight",
    )


def _build_bloom_attention_core(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
) -> str:
    summary = architecture.summary
    attention = _component_details(layer, "bloom_attention") or architecture.text_decoder.get("attention", {})
    hidden_size = summary.get("hidden_size")
    heads = attention.get("attention_heads") or summary.get("attention_heads")
    head_dim = attention.get("head_dim") or architecture.text_decoder.get("attention", {}).get("head_dim")
    qkv_shape = ["batch", "sequence", hidden_size or "hidden"]
    head_shape = ["batch", "sequence", heads or "heads", head_dim or "head_dim"]

    qkv = builder.node(
        "query_key_value",
        "Linear",
        [input_name],
        ["qkv_packed"],
        output_shapes=[["batch", "sequence", _product(3, hidden_size) or "3*hidden"]],
        output_descriptions=["BLOOM fused Q/K/V projection output."],
        in_features=hidden_size,
        out_features=_product(3, hidden_size),
        logical_weight="h.*.self_attention.query_key_value.weight",
    )
    q = builder.node(
        "split_qkv",
        "SplitQKV",
        [qkv],
        ["q_proj_out", "k_proj_out", "v_proj_out"],
        output_shapes=[qkv_shape, qkv_shape, qkv_shape],
    )
    k = "k_proj_out"
    v = "v_proj_out"
    q_heads = builder.node(
        "reshape_q",
        "ReshapeHeads",
        [q],
        ["q_heads"],
        output_shapes=[head_shape],
        heads=heads,
        head_dim=head_dim,
    )
    k_heads = builder.node(
        "reshape_k",
        "ReshapeHeads",
        [k],
        ["k_heads"],
        output_shapes=[head_shape],
        heads=heads,
        head_dim=head_dim,
    )
    v_heads = builder.node(
        "reshape_v",
        "ReshapeHeads",
        [v],
        ["v_heads"],
        output_shapes=[head_shape],
        heads=heads,
        head_dim=head_dim,
    )
    k_cache = builder.node(
        "kv_cache_update_k",
        "KVCacheUpdate",
        [k_heads],
        ["k_cache"],
        output_shapes=[["batch", "kv_sequence", heads or "heads", head_dim or "head_dim"]],
    )
    v_cache = builder.node(
        "kv_cache_update_v",
        "KVCacheUpdate",
        [v_heads],
        ["v_cache"],
        output_shapes=[["batch", "kv_sequence", heads or "heads", head_dim or "head_dim"]],
    )
    scores = builder.node(
        "attention_scores",
        "MatMulQK",
        [q_heads, k_cache],
        ["attention_scores"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
        scale="1/sqrt(head_dim)",
    )
    alibi = builder.node(
        "alibi_bias",
        "ALiBiBias",
        [scores],
        ["attention_scores_alibi"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
        heads=heads,
    )
    masked = builder.node(
        "causal_mask",
        "CausalMask",
        [alibi],
        ["masked_scores"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
    )
    probs = builder.node(
        "softmax",
        "Softmax",
        [masked],
        ["attention_probs"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
        axis=-1,
        fp32=attention.get("attention_softmax_in_fp32"),
    )
    dropped = builder.node(
        "attention_dropout",
        "Dropout",
        [probs],
        ["attention_probs_dropout"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
        ratio=attention.get("attention_dropout"),
    )
    context = builder.node(
        "attention_context",
        "MatMulPV",
        [dropped, v_cache],
        ["attention_context"],
        output_shapes=[head_shape],
    )
    merged = builder.node(
        "merge_heads",
        "MergeHeads",
        [context],
        ["attention_merged"],
        output_shapes=[["batch", "sequence", hidden_size or "hidden"]],
        hidden_size=hidden_size,
    )
    return builder.node(
        "self_attention_dense",
        "Linear",
        [merged],
        ["attention_output"],
        output_shapes=[["batch", "sequence", hidden_size or "hidden"]],
        in_features=hidden_size,
        out_features=hidden_size,
        logical_weight="h.*.self_attention.dense.weight",
    )


def _build_deepseek_mla_core(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
) -> str:
    summary = architecture.summary
    attention = _component_details(layer, "mla_sparse_attention") or architecture.text_decoder.get("attention", {})
    hidden_size = summary.get("hidden_size")
    heads = attention.get("attention_heads") or summary.get("attention_heads")
    head_dim = attention.get("head_dim") or architecture.text_decoder.get("attention", {}).get("head_dim")
    q_rank = attention.get("q_lora_rank")
    rope_dim = attention.get("qk_rope_head_dim")
    nope_dim = attention.get("qk_nope_head_dim")
    o_groups = attention.get("o_groups")
    o_rank = attention.get("o_lora_rank")
    compress_ratio = attention.get("compress_ratio")
    window = attention.get("sliding_window")
    q_shape = ["batch", "sequence", heads or "heads", head_dim or "head_dim"]
    kv_shape = ["batch", "sequence", head_dim or "head_dim"]
    sparse_kv_shape = ["batch", "sparse_kv_sequence", head_dim or "head_dim"]
    sparse_topk = _sparse_topk_dim(window, compress_ratio, attention)

    q_latent = builder.node(
        "wq_a",
        "Linear",
        [input_name],
        ["q_latent"],
        output_shapes=[["batch", "sequence", q_rank or "q_rank"]],
        output_descriptions=["Low-rank query activation from DeepSeek wq_a."],
        in_features=hidden_size,
        out_features=q_rank,
        logical_weight="attn.wq_a.weight",
    )
    q_norm = builder.node(
        "q_norm",
        "RMSNorm",
        [q_latent],
        ["q_latent_norm"],
        output_shapes=[["batch", "sequence", q_rank or "q_rank"]],
        output_descriptions=["RMSNorm over the query LoRA rank."],
    )
    q_packed = builder.node(
        "wq_b",
        "ColumnParallelLinear",
        [q_norm],
        ["q_packed"],
        output_shapes=[["batch", "sequence", _product(heads, head_dim) or "heads*head_dim"]],
        in_features=q_rank,
        out_features=_product(heads, head_dim),
        logical_weight="attn.wq_b.weight",
    )
    q_heads = builder.node(
        "reshape_q",
        "ReshapeHeads",
        [q_packed],
        ["q_heads"],
        output_shapes=[q_shape],
        heads=heads,
        head_dim=head_dim,
    )
    q_scaled = builder.node(
        "q_head_rms_scale",
        "HeadRMSScale",
        [q_heads],
        ["q_scaled"],
        output_shapes=[q_shape],
        eps=architecture.summary.get("rms_norm_eps"),
    )
    q_for_sparse = builder.node(
        "apply_rope_q_tail",
        "PartialRoPE",
        [q_scaled],
        ["q_sparse"],
        output_shapes=[q_shape],
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        theta=(attention.get("rope") or {}).get("theta"),
    )

    kv = builder.node(
        "wkv",
        "Linear",
        [input_name],
        ["kv_compact"],
        output_shapes=[kv_shape],
        output_descriptions=["Compact DeepSeek KV vector containing NoPE and RoPE tail dimensions."],
        in_features=hidden_size,
        out_features=head_dim,
        logical_weight="attn.wkv.weight",
    )
    kv_norm = builder.node(
        "kv_norm",
        "RMSNorm",
        [kv],
        ["kv_norm"],
        output_shapes=[kv_shape],
    )
    kv_pos = builder.node(
        "apply_rope_kv_tail",
        "PartialRoPE",
        [kv_norm],
        ["kv_pos"],
        output_shapes=[kv_shape],
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        theta=(attention.get("rope") or {}).get("theta"),
    )
    kv_quant = builder.node(
        "kv_non_rope_quant",
        "ActQuant",
        [kv_pos],
        ["kv_quant"],
        output_shapes=[kv_shape],
        quantized_dims="non_rope",
        block_size=64,
        scale_format="ue8m0",
    )
    kv_window = builder.node(
        "window_kv_cache_update",
        "RingKVCacheUpdate",
        [kv_quant],
        ["kv_window_cache"],
        output_shapes=[["batch", window or "window", head_dim or "head_dim"]],
        window_size=window,
    )
    window_topk = builder.node(
        "window_topk_indices",
        "WindowTopKIndices",
        [kv_window],
        ["window_topk_indices"],
        output_shapes=[["batch", "sequence", window or "window"]],
        output_dtypes=["int64"],
        window_size=window,
    )

    kv_for_sparse = kv_window
    topk_indices = window_topk
    if compress_ratio:
        compressed_kv, compressed_topk = _build_deepseek_compression_path(
            builder,
            architecture,
            layer,
            input_name,
            q_norm,
            attention,
        )
        kv_for_sparse = builder.node(
            "concat_window_compressed_kv",
            "ConcatKVCache",
            [kv_quant, compressed_kv],
            ["sparse_kv"],
            output_shapes=[sparse_kv_shape],
            window_size=window,
            compress_ratio=compress_ratio,
        )
        topk_indices = builder.node(
            "concat_sparse_indices",
            "ConcatTopKIndices",
            [window_topk, compressed_topk],
            ["sparse_topk_indices"],
            output_shapes=[["batch", "sequence", sparse_topk]],
            output_dtypes=["int64"],
        )

    attn_sink = _ensure_graph_input(
        builder,
        "attn_sink",
        [heads or "heads"],
        "float32",
        "Learned DeepSeek attention sink bias per head.",
    )
    context = builder.node(
        "sparse_attn",
        "SparseAttention",
        [q_for_sparse, kv_for_sparse, topk_indices, attn_sink],
        ["attention_context"],
        output_shapes=[q_shape],
        scale="1/sqrt(head_dim)",
        head_dim=head_dim,
        topk=sparse_topk,
    )
    context = builder.node(
        "inverse_rope_output_tail",
        "InversePartialRoPE",
        [context],
        ["attention_context_invrope"],
        output_shapes=[q_shape],
        rope_dim=rope_dim,
    )
    grouped = builder.node(
        "group_attention_heads",
        "GroupHeads",
        [context],
        ["attention_groups"],
        output_shapes=[
            [
                "batch",
                "sequence",
                o_groups or "groups",
                _group_width(heads, head_dim, o_groups) or "group_width",
            ]
        ],
        groups=o_groups,
    )
    low_rank = builder.node(
        "wo_a_grouped_low_rank",
        "GroupedOutputLowRank",
        [grouped],
        ["attention_o_lora"],
        output_shapes=[["batch", "sequence", o_groups or "groups", o_rank or "o_rank"]],
        groups=o_groups,
        in_features=_group_width(heads, head_dim, o_groups),
        out_features=o_rank,
        logical_weight="attn.wo_a.weight",
    )
    merged = builder.node(
        "flatten_o_groups",
        "FlattenGroups",
        [low_rank],
        ["attention_o_lora_flat"],
        output_shapes=[["batch", "sequence", _product(o_groups, o_rank) or "groups*o_rank"]],
    )
    return builder.node(
        "wo_b",
        "RowParallelLinear",
        [merged],
        ["attention_output"],
        output_shapes=[["batch", "sequence", hidden_size or "hidden"]],
        in_features=_product(o_groups, o_rank),
        out_features=hidden_size,
        logical_weight="attn.wo_b.weight",
    )


def _build_kimi_mla_core(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
) -> str:
    summary = architecture.summary
    attention = _component_details(layer, "kimi_mla_attention") or architecture.text_decoder.get("attention", {})
    hidden_size = summary.get("hidden_size")
    heads = attention.get("attention_heads") or summary.get("attention_heads")
    q_rank = attention.get("q_lora_rank")
    kv_rank = attention.get("kv_lora_rank")
    qk_nope_dim = attention.get("qk_nope_head_dim")
    qk_rope_dim = attention.get("qk_rope_head_dim")
    qk_dim = attention.get("qk_head_dim") or _sum_dims(qk_nope_dim, qk_rope_dim) or attention.get("head_dim")
    v_dim = attention.get("v_head_dim") or attention.get("head_dim")

    q_latent = builder.node(
        "q_a_proj",
        "Linear",
        [input_name],
        ["q_latent"],
        output_shapes=[["batch", "sequence", q_rank or "q_rank"]],
        output_descriptions=["Kimi query LoRA activation before q_a_layernorm."],
        in_features=hidden_size,
        out_features=q_rank,
        logical_weight="self_attn.q_a_proj.weight",
    )
    q_norm = builder.node(
        "q_a_layernorm",
        "RMSNorm",
        [q_latent],
        ["q_latent_norm"],
        output_shapes=[["batch", "sequence", q_rank or "q_rank"]],
        output_descriptions=["RMSNorm over Kimi query LoRA rank."],
    )
    q_packed = builder.node(
        "q_b_proj",
        "Linear",
        [q_norm],
        ["q_packed"],
        output_shapes=[["batch", "sequence", _product(heads, qk_dim) or "heads*qk_dim"]],
        in_features=q_rank,
        out_features=_product(heads, qk_dim),
        logical_weight="self_attn.q_b_proj.weight",
    )
    q_heads = builder.node(
        "reshape_q",
        "ReshapeHeads",
        [q_packed],
        ["q_heads"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_dim or "qk_dim"]],
        heads=heads,
        head_dim=qk_dim,
    )
    q_nope = builder.node(
        "split_q_nope",
        "SplitNoPE",
        [q_heads],
        ["q_nope"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_nope_dim or "qk_nope_dim"]],
        nope_dim=qk_nope_dim,
    )
    q_rope = builder.node(
        "split_q_rope",
        "SplitRoPE",
        [q_heads],
        ["q_rope_in"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_rope_dim or "qk_rope_dim"]],
        rope_dim=qk_rope_dim,
    )
    q_rope = builder.node(
        "apply_rope_q",
        "PartialRoPE",
        [q_rope],
        ["q_rope"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_rope_dim or "qk_rope_dim"]],
        rope_dim=qk_rope_dim,
        theta=(attention.get("rope") or {}).get("theta"),
    )
    q_dense = builder.node(
        "concat_q_nope_rope",
        "ConcatQKParts",
        [q_nope, q_rope],
        ["q_dense"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_dim or "qk_dim"]],
    )

    kv_a = builder.node(
        "kv_a_proj_with_mqa",
        "Linear",
        [input_name],
        ["kv_a_packed"],
        output_shapes=[["batch", "sequence", _sum_dims(kv_rank, qk_rope_dim) or "kv_rank+qk_rope"]],
        output_descriptions=["Packed Kimi compressed KV latent plus single RoPE key tail."],
        in_features=hidden_size,
        out_features=_sum_dims(kv_rank, qk_rope_dim),
        logical_weight="self_attn.kv_a_proj_with_mqa.weight",
    )
    kv_latent = builder.node(
        "split_kv_latent",
        "SplitKVLatent",
        [kv_a],
        ["kv_latent"],
        output_shapes=[["batch", "sequence", kv_rank or "kv_rank"]],
        kv_rank=kv_rank,
    )
    k_rope = builder.node(
        "split_k_rope",
        "SplitRoPE",
        [kv_a],
        ["k_rope_in"],
        output_shapes=[["batch", "sequence", 1, qk_rope_dim or "qk_rope_dim"]],
        rope_dim=qk_rope_dim,
    )
    kv_norm = builder.node(
        "kv_a_layernorm",
        "RMSNorm",
        [kv_latent],
        ["kv_latent_norm"],
        output_shapes=[["batch", "sequence", kv_rank or "kv_rank"]],
    )
    kv_b = builder.node(
        "kv_b_proj",
        "Linear",
        [kv_norm],
        ["kv_b_packed"],
        output_shapes=[["batch", "sequence", _product(heads, _sum_dims(qk_nope_dim, v_dim)) or "heads*(qk_nope+v)"]],
        in_features=kv_rank,
        out_features=_product(heads, _sum_dims(qk_nope_dim, v_dim)),
        logical_weight="self_attn.kv_b_proj.weight",
    )
    kv_heads = builder.node(
        "reshape_kv_b",
        "ReshapeHeads",
        [kv_b],
        ["kv_heads"],
        output_shapes=[["batch", "sequence", heads or "heads", _sum_dims(qk_nope_dim, v_dim) or "qk_nope+v"]],
        heads=heads,
        head_dim=_sum_dims(qk_nope_dim, v_dim),
    )
    k_nope = builder.node(
        "split_k_nope",
        "SplitNoPE",
        [kv_heads],
        ["k_nope"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_nope_dim or "qk_nope_dim"]],
        nope_dim=qk_nope_dim,
    )
    v_heads = builder.node(
        "split_v",
        "SplitValue",
        [kv_heads],
        ["v_heads"],
        output_shapes=[["batch", "sequence", heads or "heads", v_dim or "v_dim"]],
        v_dim=v_dim,
    )
    k_rope = builder.node(
        "apply_rope_k",
        "PartialRoPE",
        [k_rope],
        ["k_rope"],
        output_shapes=[["batch", "sequence", 1, qk_rope_dim or "qk_rope_dim"]],
        rope_dim=qk_rope_dim,
        theta=(attention.get("rope") or {}).get("theta"),
    )
    k_rope = builder.node(
        "expand_k_rope",
        "ExpandKVHeads",
        [k_rope],
        ["k_rope_expanded"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_rope_dim or "qk_rope_dim"]],
        heads=heads,
    )
    k_dense = builder.node(
        "concat_k_nope_rope",
        "ConcatQKParts",
        [k_nope, k_rope],
        ["k_dense"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_dim or "qk_dim"]],
    )
    k_cache = builder.node(
        "k_cache_update",
        "KVCacheUpdate",
        [k_dense],
        ["k_cache"],
        output_shapes=[["batch", "kv_sequence", heads or "heads", qk_dim or "qk_dim"]],
    )
    v_cache = builder.node(
        "v_cache_update",
        "KVCacheUpdate",
        [v_heads],
        ["v_cache"],
        output_shapes=[["batch", "kv_sequence", heads or "heads", v_dim or "v_dim"]],
    )
    scores = builder.node(
        "attention_scores",
        "MatMulQK",
        [q_dense, k_cache],
        ["attention_scores"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
        scale="1/sqrt(qk_head_dim)",
        qk_dim=qk_dim,
    )
    masked = builder.node(
        "causal_mask",
        "CausalMask",
        [scores],
        ["masked_scores"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
    )
    probs = builder.node(
        "softmax",
        "Softmax",
        [masked],
        ["attention_probs"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
        axis=-1,
    )
    dropped = builder.node(
        "attention_dropout",
        "Dropout",
        [probs],
        ["attention_probs_dropout"],
        output_shapes=[["batch", heads or "heads", "sequence", "kv_sequence"]],
        ratio=attention.get("attention_dropout"),
    )
    context = builder.node(
        "attention_context",
        "MatMulPV",
        [dropped, v_cache],
        ["attention_context"],
        output_shapes=[["batch", "sequence", heads or "heads", v_dim or "v_dim"]],
        v_dim=v_dim,
    )
    merged = builder.node(
        "merge_heads",
        "MergeHeads",
        [context],
        ["attention_merged"],
        output_shapes=[["batch", "sequence", _product(heads, v_dim) or "heads*v_dim"]],
        hidden_size=hidden_size,
    )
    return builder.node(
        "o_proj",
        "Linear",
        [merged],
        ["attention_output"],
        output_shapes=[["batch", "sequence", hidden_size or "hidden"]],
        in_features=_product(heads, v_dim),
        out_features=hidden_size,
        logical_weight="self_attn.o_proj.weight",
    )


def _build_glm_dsa_core(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
) -> str:
    summary = architecture.summary
    attention = _component_details(layer, "glm_dsa_attention") or architecture.text_decoder.get("attention", {})
    hidden_size = summary.get("hidden_size")
    heads = attention.get("attention_heads") or summary.get("attention_heads")
    q_rank = attention.get("q_lora_rank")
    kv_rank = attention.get("kv_lora_rank")
    qk_dim = attention.get("qk_head_dim") or attention.get("head_dim")
    qk_nope_dim = attention.get("qk_nope_head_dim")
    qk_rope_dim = attention.get("qk_rope_head_dim")
    v_dim = attention.get("v_head_dim") or attention.get("head_dim")
    indexer_type = attention.get("indexer_type")
    index_topk = attention.get("index_topk")

    q_latent = builder.node(
        "q_a_proj",
        "Linear",
        [input_name],
        ["q_latent"],
        output_shapes=[["batch", "sequence", q_rank or "q_rank"]],
        in_features=hidden_size,
        out_features=q_rank,
        logical_weight="self_attn.q_a_proj.weight",
    )
    q_norm = builder.node(
        "q_a_layernorm",
        "RMSNorm",
        [q_latent],
        ["q_latent_norm"],
        output_shapes=[["batch", "sequence", q_rank or "q_rank"]],
    )
    q_packed = builder.node(
        "q_b_proj",
        "Linear",
        [q_norm],
        ["q_packed"],
        output_shapes=[["batch", "sequence", _product(heads, qk_dim) or "heads*qk_dim"]],
        in_features=q_rank,
        out_features=_product(heads, qk_dim),
        logical_weight="self_attn.q_b_proj.weight",
    )
    q_heads = builder.node(
        "reshape_q",
        "ReshapeHeads",
        [q_packed],
        ["q_heads"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_dim or "qk_dim"]],
        heads=heads,
        head_dim=qk_dim,
    )
    q_nope = builder.node(
        "split_q_nope",
        "SplitNoPE",
        [q_heads],
        ["q_nope"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_nope_dim or "qk_nope_dim"]],
        nope_dim=qk_nope_dim,
    )
    q_rope = builder.node(
        "split_q_rope",
        "SplitRoPE",
        [q_heads],
        ["q_rope_in"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_rope_dim or "qk_rope_dim"]],
        rope_dim=qk_rope_dim,
    )
    q_rope = builder.node(
        "apply_rope_q",
        "InterleavedRoPE" if attention.get("rope_interleave") else "PartialRoPE",
        [q_rope],
        ["q_rope"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_rope_dim or "qk_rope_dim"]],
        rope_dim=qk_rope_dim,
        theta=(attention.get("rope") or {}).get("theta"),
    )
    q_for_sparse = builder.node(
        "concat_q_nope_rope",
        "ConcatQKParts",
        [q_nope, q_rope],
        ["q_sparse"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_dim or "qk_dim"]],
    )

    kv_a = builder.node(
        "kv_a_proj_with_mqa",
        "Linear",
        [input_name],
        ["kv_a_packed"],
        output_shapes=[["batch", "sequence", _sum_dims(kv_rank, qk_rope_dim) or "kv_rank+qk_rope"]],
        in_features=hidden_size,
        out_features=_sum_dims(kv_rank, qk_rope_dim),
        logical_weight="self_attn.kv_a_proj_with_mqa.weight",
    )
    kv_latent = builder.node(
        "split_kv_latent",
        "SplitKVLatent",
        [kv_a],
        ["kv_latent"],
        output_shapes=[["batch", "sequence", kv_rank or "kv_rank"]],
        kv_rank=kv_rank,
    )
    k_rope = builder.node(
        "split_k_rope",
        "SplitRoPE",
        [kv_a],
        ["k_rope_in"],
        output_shapes=[["batch", "sequence", 1, qk_rope_dim or "qk_rope_dim"]],
        rope_dim=qk_rope_dim,
    )
    kv_latent = builder.node(
        "kv_a_layernorm",
        "RMSNorm",
        [kv_latent],
        ["kv_latent_norm"],
        output_shapes=[["batch", "sequence", kv_rank or "kv_rank"]],
    )
    kv_b = builder.node(
        "kv_b_proj",
        "Linear",
        [kv_latent],
        ["kv_b_packed"],
        output_shapes=[["batch", "sequence", _product(heads, _sum_dims(qk_nope_dim, v_dim)) or "heads*(qk_nope+v)"]],
        in_features=kv_rank,
        out_features=_product(heads, _sum_dims(qk_nope_dim, v_dim)),
        logical_weight="self_attn.kv_b_proj.weight",
    )
    kv_heads = builder.node(
        "reshape_kv_b",
        "ReshapeHeads",
        [kv_b],
        ["kv_heads"],
        output_shapes=[["batch", "sequence", heads or "heads", _sum_dims(qk_nope_dim, v_dim) or "qk_nope+v"]],
        heads=heads,
        head_dim=_sum_dims(qk_nope_dim, v_dim),
    )
    k_nope = builder.node(
        "split_k_nope",
        "SplitNoPE",
        [kv_heads],
        ["k_nope"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_nope_dim or "qk_nope_dim"]],
        nope_dim=qk_nope_dim,
    )
    v_heads = builder.node(
        "split_v",
        "SplitValue",
        [kv_heads],
        ["v_heads"],
        output_shapes=[["batch", "sequence", heads or "heads", v_dim or "v_dim"]],
        v_dim=v_dim,
    )
    k_rope = builder.node(
        "apply_rope_k",
        "InterleavedRoPE" if attention.get("rope_interleave") else "PartialRoPE",
        [k_rope],
        ["k_rope"],
        output_shapes=[["batch", "sequence", 1, qk_rope_dim or "qk_rope_dim"]],
        rope_dim=qk_rope_dim,
        theta=(attention.get("rope") or {}).get("theta"),
    )
    k_rope = builder.node(
        "expand_k_rope",
        "ExpandKVHeads",
        [k_rope],
        ["k_rope_expanded"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_rope_dim or "qk_rope_dim"]],
        heads=heads,
    )
    k_for_sparse = builder.node(
        "concat_k_nope_rope",
        "ConcatQKParts",
        [k_nope, k_rope],
        ["k_sparse"],
        output_shapes=[["batch", "sequence", heads or "heads", qk_dim or "qk_dim"]],
    )
    k_cache = builder.node(
        "k_cache_update",
        "KVCacheUpdate",
        [k_for_sparse],
        ["k_cache"],
        output_shapes=[["batch", "kv_sequence", heads or "heads", qk_dim or "qk_dim"]],
    )
    v_cache = builder.node(
        "v_cache_update",
        "KVCacheUpdate",
        [v_heads],
        ["v_cache"],
        output_shapes=[["batch", "kv_sequence", heads or "heads", v_dim or "v_dim"]],
    )

    sparse_indices = _build_glm_indexer_path(builder, architecture, layer, input_name, q_norm, attention)
    context = builder.node(
        "glm_dsa_attention",
        "DynamicSparseAttention",
        [q_for_sparse, k_cache, v_cache, sparse_indices],
        ["attention_context"],
        output_shapes=[["batch", "sequence", heads or "heads", v_dim or "v_dim"]],
        qk_dim=qk_dim,
        v_dim=v_dim,
        topk=index_topk,
    )
    merged = builder.node(
        "merge_heads",
        "MergeHeads",
        [context],
        ["attention_merged"],
        output_shapes=[["batch", "sequence", _product(heads, v_dim) or "heads*v_dim"]],
        hidden_size=hidden_size,
    )
    return builder.node(
        "o_proj",
        "Linear",
        [merged],
        ["attention_output"],
        output_shapes=[["batch", "sequence", hidden_size or "hidden"]],
        in_features=_product(heads, v_dim),
        out_features=hidden_size,
        logical_weight="self_attn.o_proj.weight",
        indexer_type=indexer_type,
    )


def _build_glm_indexer_path(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
    q_latent_norm: str,
    attention: Dict[str, Any],
) -> str:
    indexer_type = str(attention.get("indexer_type") or "full")
    index_topk = attention.get("index_topk")
    if indexer_type == "shared":
        return builder.node(
            "indexshare_reuse",
            "IndexShareReuse",
            [q_latent_norm],
            ["sparse_topk_indices"],
            output_shapes=[["batch", "sequence", index_topk or "index_topk"]],
            output_dtypes=["int64"],
            index_topk_freq=attention.get("index_topk_freq"),
            index_skip_topk_offset=attention.get("index_skip_topk_offset"),
        )

    hidden_size = architecture.summary.get("hidden_size")
    index_heads = attention.get("index_n_heads")
    index_dim = attention.get("index_head_dim")
    q_rank = attention.get("q_lora_rank")
    q = builder.node(
        "indexer_wq_b",
        "Linear",
        [q_latent_norm],
        ["index_q_packed"],
        output_shapes=[["batch", "sequence", _product(index_heads, index_dim) or "idx_heads*idx_dim"]],
        in_features=q_rank,
        out_features=_product(index_heads, index_dim),
        logical_weight="self_attn.indexer.wq_b.weight",
    )
    q = builder.node(
        "reshape_index_q",
        "ReshapeHeads",
        [q],
        ["index_q_heads"],
        output_shapes=[["batch", "sequence", index_heads or "idx_heads", index_dim or "idx_dim"]],
        heads=index_heads,
        head_dim=index_dim,
    )
    q = builder.node(
        "index_q_rope",
        "InterleavedRoPE" if attention.get("indexer_rope_interleave") else "PartialRoPE",
        [q],
        ["index_q_rope"],
        output_shapes=[["batch", "sequence", index_heads or "idx_heads", index_dim or "idx_dim"]],
        rope_dim=attention.get("qk_rope_head_dim"),
    )
    k = builder.node(
        "indexer_wk",
        "Linear",
        [input_name],
        ["index_k"],
        output_shapes=[["batch", "kv_sequence", index_dim or "idx_dim"]],
        in_features=hidden_size,
        out_features=index_dim,
        logical_weight="self_attn.indexer.wk.weight",
    )
    k = builder.node(
        "indexer_k_norm",
        "RMSNorm",
        [k],
        ["index_k_norm"],
        output_shapes=[["batch", "kv_sequence", index_dim or "idx_dim"]],
        logical_weight="self_attn.indexer.k_norm.weight",
    )
    weights = builder.node(
        "indexer_weights_proj",
        "Linear",
        [input_name],
        ["index_weights"],
        output_shapes=[["batch", "sequence", index_heads or "idx_heads"]],
        in_features=hidden_size,
        out_features=index_heads,
        logical_weight="self_attn.indexer.weights_proj.weight",
    )
    scores = builder.node(
        "glm_index_score",
        "SparseIndexScore",
        [q, k, weights],
        ["sparse_index_scores"],
        output_shapes=[["batch", "sequence", "kv_sequence"]],
        index_heads=index_heads,
        index_dim=index_dim,
    )
    return builder.node(
        "glm_index_topk",
        "TopK",
        [scores],
        ["sparse_topk_indices", "sparse_topk_scores"],
        output_shapes=[
            ["batch", "sequence", index_topk or "index_topk"],
            ["batch", "sequence", index_topk or "index_topk"],
        ],
        output_dtypes=["int64", "float32"],
        k=index_topk,
    )


def _build_deepseek_compression_path(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
    q_latent_norm: str,
    attention: Dict[str, Any],
) -> List[str]:
    hidden_size = architecture.summary.get("hidden_size")
    head_dim = attention.get("head_dim") or architecture.text_decoder.get("attention", {}).get("head_dim")
    rope_dim = attention.get("qk_rope_head_dim")
    compress_ratio = attention.get("compress_ratio")
    compress_mode = attention.get("compress_mode")
    coff = 2 if compress_ratio == 4 else 1
    compressed_shape = ["batch", "compressed_sequence", head_dim or "head_dim"]
    expanded_dim = _product(coff, head_dim) or ("2*head_dim" if coff == 2 else head_dim or "head_dim")

    comp_kv = builder.node(
        "compress_wkv",
        "Linear",
        [input_name],
        ["compress_kv_raw"],
        output_shapes=[["batch", "sequence", expanded_dim]],
        in_features=hidden_size,
        out_features=expanded_dim,
        logical_weight="attn.compressor.wkv.weight",
    )
    comp_gate = builder.node(
        "compress_wgate",
        "Linear",
        [input_name],
        ["compress_gate_raw"],
        output_shapes=[["batch", "sequence", expanded_dim]],
        in_features=hidden_size,
        out_features=expanded_dim,
        logical_weight="attn.compressor.wgate.weight",
    )
    pooled = builder.node(
        "compress_gated_pool",
        "GatedKVCompression",
        [comp_kv, comp_gate],
        ["compressed_kv_pool"],
        output_shapes=[compressed_shape],
        compress_ratio=compress_ratio,
        overlap=int(compress_ratio == 4),
    )
    norm = builder.node(
        "compress_norm",
        "RMSNorm",
        [pooled],
        ["compressed_kv_norm"],
        output_shapes=[compressed_shape],
    )
    pos = builder.node(
        "compress_rope_tail",
        "PartialRoPE",
        [norm],
        ["compressed_kv_pos"],
        output_shapes=[compressed_shape],
        rope_dim=rope_dim,
        theta=attention.get("compress_rope_theta"),
    )
    quant = builder.node(
        "compress_kv_quant",
        "ActQuant",
        [pos],
        ["compressed_kv_quant"],
        output_shapes=[compressed_shape],
        quantized_dims="non_rope",
        block_size=64,
    )
    cache = builder.node(
        "compressed_kv_cache_update",
        "CompressedKVCacheUpdate",
        [quant],
        ["compressed_kv_cache"],
        output_shapes=[compressed_shape],
        compress_ratio=compress_ratio,
        mode=compress_mode,
    )

    if compress_ratio == 4:
        topk = _build_deepseek_indexer_path(builder, architecture, layer, input_name, q_latent_norm, cache, attention)
    else:
        topk = builder.node(
            "deterministic_compress_topk",
            "CompressTopKIndices",
            [cache],
            ["compressed_topk_indices"],
            output_shapes=[["batch", "sequence", "compressed_sequence"]],
            output_dtypes=["int64"],
            compress_ratio=compress_ratio,
        )
    return [cache, topk]


def _build_deepseek_indexer_path(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
    q_latent_norm: str,
    compressed_kv_cache: str,
    attention: Dict[str, Any],
) -> str:
    hidden_size = architecture.summary.get("hidden_size")
    index_heads = attention.get("index_n_heads")
    index_dim = attention.get("index_head_dim")
    index_topk = attention.get("index_topk")
    rope_dim = attention.get("qk_rope_head_dim")
    q_rank = attention.get("q_lora_rank")
    index_shape = ["batch", "sequence", index_heads or "idx_heads", index_dim or "idx_dim"]

    index_q = builder.node(
        "indexer_wq_b",
        "ColumnParallelLinear",
        [q_latent_norm],
        ["index_q_packed"],
        output_shapes=[["batch", "sequence", _product(index_heads, index_dim) or "idx_heads*idx_dim"]],
        in_features=q_rank,
        out_features=_product(index_heads, index_dim),
        logical_weight="attn.indexer.wq_b.weight",
    )
    index_q = builder.node(
        "reshape_index_q",
        "ReshapeHeads",
        [index_q],
        ["index_q_heads"],
        output_shapes=[index_shape],
        heads=index_heads,
        head_dim=index_dim,
    )
    index_q = builder.node(
        "index_q_rope_tail",
        "PartialRoPE",
        [index_q],
        ["index_q_rope"],
        output_shapes=[index_shape],
        rope_dim=rope_dim,
    )
    index_q = builder.node(
        "index_q_hadamard",
        "HadamardRotate",
        [index_q],
        ["index_q_rotated"],
        output_shapes=[index_shape],
    )
    index_q = builder.node(
        "index_q_fp4_quant",
        "FP4ActQuant",
        [index_q],
        ["index_q_quant"],
        output_shapes=[index_shape],
        block_size=32,
    )
    weights = builder.node(
        "index_weights_proj",
        "ColumnParallelLinear",
        [input_name],
        ["index_weights"],
        output_shapes=[["batch", "sequence", index_heads or "idx_heads"]],
        in_features=hidden_size,
        out_features=index_heads,
        logical_weight="attn.indexer.weights_proj.weight",
    )
    scores = builder.node(
        "index_score",
        "SparseIndexScore",
        [index_q, compressed_kv_cache, weights],
        ["compressed_index_scores"],
        output_shapes=[["batch", "sequence", "compressed_sequence"]],
        index_heads=index_heads,
        index_dim=index_dim,
    )
    return builder.node(
        "index_topk",
        "TopK",
        [scores],
        ["compressed_topk_indices", "compressed_topk_scores"],
        output_shapes=[
            ["batch", "sequence", index_topk or "index_topk"],
            ["batch", "sequence", index_topk or "index_topk"],
        ],
        output_dtypes=["int64", "float32"],
        k=index_topk,
    )


def _build_mlp_flow(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
    standalone: bool = False,
) -> str:
    if standalone:
        input_name = builder.add_input(
            input_name,
            ["batch", "sequence", builder.hidden_size],
            description="MLP-detail graph input hidden states.",
        )
    if _has_moe_branch(layer):
        output = _build_moe_core(builder, architecture, layer, input_name)
    else:
        output = _build_dense_mlp_core(builder, architecture, layer, input_name)
    if standalone:
        builder.add_output(output)
    return output


def _build_dense_mlp_core(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
) -> str:
    if _has_bloom_mlp(layer):
        return _build_bloom_mlp_core(builder, architecture, layer, input_name)

    hidden_size = architecture.summary.get("hidden_size")
    dense = _component_details(layer, "gate_proj/up_proj")
    activation = _component_kind(layer, "activation") or "activation"
    intermediate = dense.get("intermediate_size")
    gate = builder.node(
        "gate_proj",
        "Linear",
        [input_name],
        ["mlp_gate"],
        output_shapes=[["batch", "sequence", intermediate or "intermediate"]],
        in_features=hidden_size,
        out_features=intermediate,
        logical_weight="gate_proj.weight",
    )
    up = builder.node(
        "up_proj",
        "Linear",
        [input_name],
        ["mlp_up"],
        output_shapes=[["batch", "sequence", intermediate or "intermediate"]],
        in_features=hidden_size,
        out_features=intermediate,
        logical_weight="up_proj.weight",
    )
    activated = builder.node(
        "mlp_activation",
        _activation_op(activation),
        [gate],
        ["mlp_gate_activated"],
        output_shapes=[["batch", "sequence", intermediate or "intermediate"]],
    )
    multiplied = builder.node(
        "mlp_gate_up_mul",
        "Mul",
        [activated, up],
        ["mlp_product"],
        output_shapes=[["batch", "sequence", intermediate or "intermediate"]],
    )
    return builder.node(
        "down_proj",
        "Linear",
        [multiplied],
        ["mlp_output"],
        output_shapes=[["batch", "sequence", hidden_size or "hidden"]],
        in_features=intermediate,
        out_features=hidden_size,
        logical_weight="down_proj.weight",
    )


def _build_bloom_mlp_core(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
) -> str:
    hidden_size = architecture.summary.get("hidden_size")
    dense = _component_details(layer, "dense_h_to_4h")
    activation = _component_kind(layer, "activation") or "gelu"
    intermediate = dense.get("intermediate_size") or _product(4, hidden_size)
    up = builder.node(
        "dense_h_to_4h",
        "Linear",
        [input_name],
        ["mlp_up"],
        output_shapes=[["batch", "sequence", intermediate or "4*hidden"]],
        in_features=hidden_size,
        out_features=intermediate,
        logical_weight="h.*.mlp.dense_h_to_4h.weight",
    )
    activated = builder.node(
        "mlp_activation",
        _activation_op(activation),
        [up],
        ["mlp_gate_activated"],
        output_shapes=[["batch", "sequence", intermediate or "4*hidden"]],
    )
    return builder.node(
        "dense_4h_to_h",
        "Linear",
        [activated],
        ["mlp_output"],
        output_shapes=[["batch", "sequence", hidden_size or "hidden"]],
        in_features=intermediate,
        out_features=hidden_size,
        logical_weight="h.*.mlp.dense_4h_to_h.weight",
    )


def _build_moe_flow(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
    standalone: bool = False,
) -> str:
    if standalone:
        input_name = builder.add_input(
            input_name,
            ["batch", "sequence", builder.hidden_size],
            description="MoE-detail graph input hidden states.",
        )
    output = _build_moe_core(builder, architecture, layer, input_name)
    if standalone:
        builder.add_output(output)
    return output


def _build_moe_core(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
) -> str:
    if _is_deepseek_v4_arch(architecture, layer):
        return _build_deepseek_moe_core(builder, architecture, layer, input_name)
    if _is_kimi_k25_arch(architecture, layer):
        return _build_kimi_moe_core(builder, architecture, layer, input_name)
    if _is_glm_moe_dsa_arch(architecture, layer):
        return _build_glm_moe_core(builder, architecture, layer, input_name)

    hidden_size = architecture.summary.get("hidden_size")
    router = _component_details(layer, "router")
    routed = _component_details(layer, "expert_gate/up/down_proj")
    shared = _component_details(layer, "shared_expert_gate/up/down_proj")

    flat = builder.node(
        "flatten_tokens",
        "FlattenTokens",
        [input_name],
        ["tokens_flat"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
    )
    logits = builder.node(
        "router",
        "RouterLinear",
        [flat],
        ["router_logits"],
        output_shapes=[["tokens", router.get("experts") or "experts"]],
        in_features=hidden_size,
        experts=router.get("experts"),
        logical_weight="router.weight",
        router_aux_loss_coef=router.get("router_aux_loss_coef"),
    )
    topk = builder.node(
        "router_topk",
        "TopK",
        [logits],
        ["router_topk_indices", "router_topk_scores"],
        output_shapes=[
            ["tokens", router.get("activated_experts") or "top_k"],
            ["tokens", router.get("activated_experts") or "top_k"],
        ],
        output_dtypes=["int64", "float32"],
        k=router.get("activated_experts"),
    )
    scores = builder.node(
        "router_sigmoid",
        "Sigmoid",
        ["router_topk_scores"],
        ["router_weights"],
        output_shapes=[["tokens", router.get("activated_experts") or "top_k"]],
    )
    dispatched = builder.node(
        "dispatch_tokens",
        "DispatchTokens",
        [flat, "router_topk_indices", scores],
        ["routed_tokens"],
        output_shapes=[["tokens", router.get("activated_experts") or "top_k", hidden_size or "hidden"]],
        experts=router.get("experts"),
        top_k=router.get("activated_experts"),
    )

    grouped_op = _grouped_gemm_op(routed)
    routed_gate_up = builder.node(
        "routed_expert_gate_up_proj",
        grouped_op,
        [dispatched],
        ["routed_gate_up"],
        output_shapes=[
            [
                "tokens",
                router.get("activated_experts") or "top_k",
                2 * int(routed.get("intermediate_size")) if routed.get("intermediate_size") else "2*intermediate",
            ]
        ],
        experts=routed.get("experts"),
        in_features=hidden_size,
        out_features=routed.get("intermediate_size"),
        logical_weight="experts.gate_up_proj",
        expert_dtype=routed.get("expert_dtype"),
    )
    routed_shape = ["tokens", router.get("activated_experts") or "top_k", routed.get("intermediate_size") or "intermediate"]
    routed_gate = builder.node("split_routed_gate", "SplitGate", [routed_gate_up], ["routed_gate"], output_shapes=[routed_shape])
    routed_up = builder.node("split_routed_up", "SplitUp", [routed_gate_up], ["routed_up"], output_shapes=[routed_shape])
    routed_act = builder.node(
        "routed_activation",
        _activation_op(routed.get("activation")),
        [routed_gate],
        ["routed_gate_activated"],
        output_shapes=[routed_shape],
    )
    routed_mul = builder.node(
        "routed_gate_up_mul",
        "Mul",
        [routed_act, routed_up],
        ["routed_product"],
        output_shapes=[routed_shape],
    )
    routed_down = builder.node(
        "routed_expert_down_proj",
        grouped_op,
        [routed_mul],
        ["routed_expert_output"],
        output_shapes=[["tokens", router.get("activated_experts") or "top_k", hidden_size or "hidden"]],
        experts=routed.get("experts"),
        in_features=routed.get("intermediate_size"),
        out_features=hidden_size,
        logical_weight="experts.down_proj",
        expert_dtype=routed.get("expert_dtype"),
    )
    routed_sum = builder.node(
        "combine_routed_experts",
        "WeightedExpertSum",
        [routed_down, "router_topk_indices", scores],
        ["routed_output"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
    )

    combined = routed_sum
    if shared:
        shared_gate = builder.node(
            "shared_expert_gate_proj",
            "Linear",
            [flat],
            ["shared_gate"],
            output_shapes=[["tokens", shared.get("intermediate_size") or "intermediate"]],
            in_features=hidden_size,
            out_features=shared.get("intermediate_size"),
            logical_weight="shared_expert.gate_proj.weight",
        )
        shared_up = builder.node(
            "shared_expert_up_proj",
            "Linear",
            [flat],
            ["shared_up"],
            output_shapes=[["tokens", shared.get("intermediate_size") or "intermediate"]],
            in_features=hidden_size,
            out_features=shared.get("intermediate_size"),
            logical_weight="shared_expert.up_proj.weight",
        )
        shared_act = builder.node(
            "shared_activation",
            _activation_op(shared.get("activation")),
            [shared_gate],
            ["shared_gate_activated"],
            output_shapes=[["tokens", shared.get("intermediate_size") or "intermediate"]],
        )
        shared_mul = builder.node(
            "shared_gate_up_mul",
            "Mul",
            [shared_act, shared_up],
            ["shared_product"],
            output_shapes=[["tokens", shared.get("intermediate_size") or "intermediate"]],
        )
        shared_down = builder.node(
            "shared_expert_down_proj",
            "Linear",
            [shared_mul],
            ["shared_output"],
            output_shapes=[["tokens", hidden_size or "hidden"]],
            in_features=shared.get("intermediate_size"),
            out_features=hidden_size,
            logical_weight="shared_expert.down_proj.weight",
        )
        combined = builder.node(
            "add_routed_shared",
            "Add",
            [routed_sum, shared_down],
            ["moe_flat_output"],
            output_shapes=[["tokens", hidden_size or "hidden"]],
        )
    return builder.node(
        "unflatten_tokens",
        "UnflattenTokens",
        [combined],
        ["moe_output"],
        output_shapes=[["batch", "sequence", hidden_size or "hidden"]],
    )


def _build_deepseek_moe_core(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
) -> str:
    hidden_size = architecture.summary.get("hidden_size")
    router = _component_details(layer, "router")
    routed = _component_details(layer, "expert_gate/up/down_proj")
    shared = _component_details(layer, "shared_expert_gate/up/down_proj")
    intermediate = routed.get("intermediate_size") or routed.get("moe_intermediate_size")
    top_k = router.get("activated_experts")

    flat = builder.node(
        "flatten_tokens",
        "FlattenTokens",
        [input_name],
        ["tokens_flat"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
    )
    logits = builder.node(
        "router",
        "RouterLinear",
        [flat],
        ["router_logits"],
        output_shapes=[["tokens", router.get("experts") or "experts"]],
        in_features=hidden_size,
        experts=router.get("experts"),
        logical_weight="ffn.gate.weight",
        scoring_func=router.get("scoring_func"),
    )
    scores = builder.node(
        "router_score_activation",
        _router_score_op(router),
        [logits],
        ["router_scores"],
        output_shapes=[["tokens", router.get("experts") or "experts"]],
        scoring_func=router.get("scoring_func"),
    )
    if router.get("hash_routing"):
        input_ids = _ensure_graph_input(
            builder,
            "input_ids",
            ["batch", "sequence"],
            "int64",
            "Token IDs used by DeepSeek hash-router layers.",
        )
        indices = builder.node(
            "hash_route_lookup",
            "HashRouteLookup",
            [input_ids],
            ["router_topk_indices"],
            output_shapes=[["tokens", top_k or "top_k"]],
            output_dtypes=["int64"],
            vocab_size=architecture.summary.get("vocab_size"),
            top_k=top_k,
            logical_weight="ffn.gate.tid2eid",
        )
    else:
        biased_scores = builder.node(
            "router_bias_add",
            "RouterBiasAdd",
            [scores],
            ["router_scores_biased"],
            output_shapes=[["tokens", router.get("experts") or "experts"]],
            logical_weight="ffn.gate.bias",
        )
        indices = builder.node(
            "router_topk",
            "TopK",
            [biased_scores],
            ["router_topk_indices", "router_topk_scores"],
            output_shapes=[
                ["tokens", top_k or "top_k"],
                ["tokens", top_k or "top_k"],
            ],
            output_dtypes=["int64", "float32"],
            k=top_k,
        )

    gathered = builder.node(
        "gather_router_scores",
        "GatherRouterScores",
        [scores, indices],
        ["router_selected_scores"],
        output_shapes=[["tokens", top_k or "top_k"]],
    )
    weights = builder.node(
        "router_normalize_scale",
        "RouterNormalizeScale",
        [gathered],
        ["router_weights"],
        output_shapes=[["tokens", top_k or "top_k"]],
        scoring_func=router.get("scoring_func"),
        route_scale=router.get("route_scale"),
        norm_topk_prob=router.get("norm_topk_prob"),
    )
    dispatched = builder.node(
        "dispatch_tokens",
        "DispatchTokens",
        [flat, indices, weights],
        ["routed_tokens"],
        output_shapes=[["tokens", top_k or "top_k", hidden_size or "hidden"]],
        experts=router.get("experts"),
        top_k=top_k,
    )

    routed_gate = builder.node(
        "routed_expert_w1",
        "FP4GroupedGEMM",
        [dispatched],
        ["routed_gate"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
        experts=routed.get("experts"),
        in_features=hidden_size,
        out_features=intermediate,
        logical_weight="ffn.experts.*.w1.weight",
        expert_dtype=routed.get("expert_dtype"),
    )
    routed_up = builder.node(
        "routed_expert_w3",
        "FP4GroupedGEMM",
        [dispatched],
        ["routed_up"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
        experts=routed.get("experts"),
        in_features=hidden_size,
        out_features=intermediate,
        logical_weight="ffn.experts.*.w3.weight",
        expert_dtype=routed.get("expert_dtype"),
    )
    if routed.get("swiglu_limit"):
        clipped_gate = builder.node(
            "routed_swiglu_clamp",
            "SwiGLUClamp",
            [routed_gate, routed_up],
            ["routed_gate_clamped", "routed_up_clamped"],
            output_shapes=[
                ["tokens", top_k or "top_k", intermediate or "intermediate"],
                ["tokens", top_k or "top_k", intermediate or "intermediate"],
            ],
            swiglu_limit=routed.get("swiglu_limit"),
        )
        routed_gate = clipped_gate
        routed_up = "routed_up_clamped"
    routed_act = builder.node(
        "routed_activation",
        _activation_op(routed.get("activation")),
        [routed_gate],
        ["routed_gate_activated"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
    )
    routed_mul = builder.node(
        "routed_gate_up_mul",
        "Mul",
        [routed_act, routed_up],
        ["routed_product"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
    )
    routed_weighted = builder.node(
        "apply_router_weights",
        "ApplyRouterWeights",
        [routed_mul, weights],
        ["routed_weighted_product"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
    )
    routed_down = builder.node(
        "routed_expert_w2",
        "FP4GroupedGEMM",
        [routed_weighted],
        ["routed_expert_output"],
        output_shapes=[["tokens", top_k or "top_k", hidden_size or "hidden"]],
        experts=routed.get("experts"),
        in_features=intermediate,
        out_features=hidden_size,
        logical_weight="ffn.experts.*.w2.weight",
        expert_dtype=routed.get("expert_dtype"),
    )
    routed_sum = builder.node(
        "reduce_routed_experts",
        "ExpertReduce",
        [routed_down, indices],
        ["routed_output"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
        experts=routed.get("experts"),
    )

    shared_intermediate = shared.get("intermediate_size") or intermediate
    shared_gate = builder.node(
        "shared_expert_w1",
        "QuantizedLinear",
        [flat],
        ["shared_gate"],
        output_shapes=[["tokens", shared_intermediate or "intermediate"]],
        in_features=hidden_size,
        out_features=shared_intermediate,
        logical_weight="ffn.shared_experts.w1.weight",
    )
    shared_up = builder.node(
        "shared_expert_w3",
        "QuantizedLinear",
        [flat],
        ["shared_up"],
        output_shapes=[["tokens", shared_intermediate or "intermediate"]],
        in_features=hidden_size,
        out_features=shared_intermediate,
        logical_weight="ffn.shared_experts.w3.weight",
    )
    if shared.get("swiglu_limit"):
        clipped_shared = builder.node(
            "shared_swiglu_clamp",
            "SwiGLUClamp",
            [shared_gate, shared_up],
            ["shared_gate_clamped", "shared_up_clamped"],
            output_shapes=[
                ["tokens", shared_intermediate or "intermediate"],
                ["tokens", shared_intermediate or "intermediate"],
            ],
            swiglu_limit=shared.get("swiglu_limit"),
        )
        shared_gate = clipped_shared
        shared_up = "shared_up_clamped"
    shared_act = builder.node(
        "shared_activation",
        _activation_op(shared.get("activation")),
        [shared_gate],
        ["shared_gate_activated"],
        output_shapes=[["tokens", shared_intermediate or "intermediate"]],
    )
    shared_mul = builder.node(
        "shared_gate_up_mul",
        "Mul",
        [shared_act, shared_up],
        ["shared_product"],
        output_shapes=[["tokens", shared_intermediate or "intermediate"]],
    )
    shared_down = builder.node(
        "shared_expert_w2",
        "QuantizedLinear",
        [shared_mul],
        ["shared_output"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
        in_features=shared_intermediate,
        out_features=hidden_size,
        logical_weight="ffn.shared_experts.w2.weight",
    )
    combined = builder.node(
        "add_routed_shared",
        "Add",
        [routed_sum, shared_down],
        ["moe_flat_output"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
    )
    return builder.node(
        "unflatten_tokens",
        "UnflattenTokens",
        [combined],
        ["moe_output"],
        output_shapes=[["batch", "sequence", hidden_size or "hidden"]],
    )


def _build_kimi_moe_core(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
) -> str:
    hidden_size = architecture.summary.get("hidden_size")
    router = _component_details(layer, "router")
    routed = _component_details(layer, "expert_gate/up/down_proj")
    shared = _component_details(layer, "shared_expert_gate/up/down_proj")
    intermediate = routed.get("intermediate_size") or routed.get("moe_intermediate_size")
    top_k = router.get("activated_experts")

    flat = builder.node(
        "flatten_tokens",
        "FlattenTokens",
        [input_name],
        ["tokens_flat"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
    )
    logits = builder.node(
        "router",
        "RouterLinear",
        [flat],
        ["router_logits"],
        output_shapes=[["tokens", router.get("experts") or "experts"]],
        in_features=hidden_size,
        experts=router.get("experts"),
        logical_weight="language_model.model.layers.*.mlp.gate.weight",
        scoring_func=router.get("scoring_func"),
    )
    scores = builder.node(
        "router_sigmoid",
        "Sigmoid",
        [logits],
        ["router_scores"],
        output_shapes=[["tokens", router.get("experts") or "experts"]],
    )
    biased_scores = builder.node(
        "router_score_correction_bias",
        "RouterBiasAdd",
        [scores],
        ["router_scores_biased"],
        output_shapes=[["tokens", router.get("experts") or "experts"]],
        logical_weight="language_model.model.layers.*.mlp.gate.e_score_correction_bias",
        topk_method=router.get("topk_method"),
        n_group=router.get("n_group"),
        topk_group=router.get("topk_group"),
    )
    indices = builder.node(
        "router_topk",
        "TopK",
        [biased_scores],
        ["router_topk_indices", "router_topk_scores"],
        output_shapes=[
            ["tokens", top_k or "top_k"],
            ["tokens", top_k or "top_k"],
        ],
        output_dtypes=["int64", "float32"],
        k=top_k,
    )
    gathered = builder.node(
        "gather_router_scores",
        "GatherRouterScores",
        [scores, indices],
        ["router_selected_scores"],
        output_shapes=[["tokens", top_k or "top_k"]],
    )
    weights = builder.node(
        "router_normalize_scale",
        "RouterNormalizeScale",
        [gathered],
        ["router_weights"],
        output_shapes=[["tokens", top_k or "top_k"]],
        scoring_func=router.get("scoring_func"),
        route_scale=router.get("route_scale"),
        norm_topk_prob=router.get("norm_topk_prob"),
    )
    dispatched = builder.node(
        "dispatch_tokens",
        "DispatchTokens",
        [flat, indices, weights],
        ["routed_tokens"],
        output_shapes=[["tokens", top_k or "top_k", hidden_size or "hidden"]],
        experts=router.get("experts"),
        top_k=top_k,
    )
    routed_gate = builder.node(
        "routed_expert_gate_proj",
        "Int4GroupedGEMM",
        [dispatched],
        ["routed_gate"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
        experts=routed.get("experts"),
        in_features=hidden_size,
        out_features=intermediate,
        logical_weight="language_model.model.layers.*.mlp.experts.*.gate_proj.weight_packed",
        logical_scale="language_model.model.layers.*.mlp.experts.*.gate_proj.weight_scale",
        logical_shape="language_model.model.layers.*.mlp.experts.*.gate_proj.weight_shape",
        expert_dtype=routed.get("expert_dtype"),
    )
    routed_up = builder.node(
        "routed_expert_up_proj",
        "Int4GroupedGEMM",
        [dispatched],
        ["routed_up"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
        experts=routed.get("experts"),
        in_features=hidden_size,
        out_features=intermediate,
        logical_weight="language_model.model.layers.*.mlp.experts.*.up_proj.weight_packed",
        logical_scale="language_model.model.layers.*.mlp.experts.*.up_proj.weight_scale",
        logical_shape="language_model.model.layers.*.mlp.experts.*.up_proj.weight_shape",
        expert_dtype=routed.get("expert_dtype"),
    )
    routed_act = builder.node(
        "routed_activation",
        _activation_op(routed.get("activation")),
        [routed_gate],
        ["routed_gate_activated"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
    )
    routed_mul = builder.node(
        "routed_gate_up_mul",
        "Mul",
        [routed_act, routed_up],
        ["routed_product"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
    )
    routed_down = builder.node(
        "routed_expert_down_proj",
        "Int4GroupedGEMM",
        [routed_mul],
        ["routed_expert_output"],
        output_shapes=[["tokens", top_k or "top_k", hidden_size or "hidden"]],
        experts=routed.get("experts"),
        in_features=intermediate,
        out_features=hidden_size,
        logical_weight="language_model.model.layers.*.mlp.experts.*.down_proj.weight_packed",
        logical_scale="language_model.model.layers.*.mlp.experts.*.down_proj.weight_scale",
        logical_shape="language_model.model.layers.*.mlp.experts.*.down_proj.weight_shape",
        expert_dtype=routed.get("expert_dtype"),
    )
    routed_sum = builder.node(
        "combine_routed_experts",
        "WeightedExpertSum",
        [routed_down, indices, weights],
        ["routed_output"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
    )

    shared_intermediate = shared.get("intermediate_size") or intermediate
    shared_gate = builder.node(
        "shared_expert_gate_proj",
        "Linear",
        [flat],
        ["shared_gate"],
        output_shapes=[["tokens", shared_intermediate or "intermediate"]],
        in_features=hidden_size,
        out_features=shared_intermediate,
        logical_weight="language_model.model.layers.*.mlp.shared_experts.gate_proj.weight",
    )
    shared_up = builder.node(
        "shared_expert_up_proj",
        "Linear",
        [flat],
        ["shared_up"],
        output_shapes=[["tokens", shared_intermediate or "intermediate"]],
        in_features=hidden_size,
        out_features=shared_intermediate,
        logical_weight="language_model.model.layers.*.mlp.shared_experts.up_proj.weight",
    )
    shared_act = builder.node(
        "shared_activation",
        _activation_op(shared.get("activation")),
        [shared_gate],
        ["shared_gate_activated"],
        output_shapes=[["tokens", shared_intermediate or "intermediate"]],
    )
    shared_mul = builder.node(
        "shared_gate_up_mul",
        "Mul",
        [shared_act, shared_up],
        ["shared_product"],
        output_shapes=[["tokens", shared_intermediate or "intermediate"]],
    )
    shared_down = builder.node(
        "shared_expert_down_proj",
        "Linear",
        [shared_mul],
        ["shared_output"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
        in_features=shared_intermediate,
        out_features=hidden_size,
        logical_weight="language_model.model.layers.*.mlp.shared_experts.down_proj.weight",
    )
    combined = builder.node(
        "add_routed_shared",
        "Add",
        [routed_sum, shared_down],
        ["moe_flat_output"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
    )
    return builder.node(
        "unflatten_tokens",
        "UnflattenTokens",
        [combined],
        ["moe_output"],
        output_shapes=[["batch", "sequence", hidden_size or "hidden"]],
    )


def _build_glm_moe_core(
    builder: KernelGraphBuilder,
    architecture: Architecture,
    layer: Layer,
    input_name: str,
) -> str:
    hidden_size = architecture.summary.get("hidden_size")
    router = _component_details(layer, "router")
    routed = _component_details(layer, "expert_gate/up/down_proj")
    shared = _component_details(layer, "shared_expert_gate/up/down_proj")
    intermediate = routed.get("intermediate_size") or routed.get("moe_intermediate_size")
    top_k = router.get("activated_experts")

    flat = builder.node(
        "flatten_tokens",
        "FlattenTokens",
        [input_name],
        ["tokens_flat"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
    )
    logits = builder.node(
        "router",
        "RouterLinear",
        [flat],
        ["router_logits"],
        output_shapes=[["tokens", router.get("experts") or "experts"]],
        in_features=hidden_size,
        experts=router.get("experts"),
        logical_weight="mlp.gate.weight",
        scoring_func=router.get("scoring_func"),
    )
    scores = builder.node(
        "router_sigmoid",
        "Sigmoid",
        [logits],
        ["router_scores"],
        output_shapes=[["tokens", router.get("experts") or "experts"]],
    )
    biased_scores = builder.node(
        "router_score_correction_bias",
        "RouterBiasAdd",
        [scores],
        ["router_scores_biased"],
        output_shapes=[["tokens", router.get("experts") or "experts"]],
        logical_weight="mlp.gate.e_score_correction_bias",
        topk_method=router.get("topk_method"),
    )
    indices = builder.node(
        "router_topk",
        "TopK",
        [biased_scores],
        ["router_topk_indices", "router_topk_scores"],
        output_shapes=[
            ["tokens", top_k or "top_k"],
            ["tokens", top_k or "top_k"],
        ],
        output_dtypes=["int64", "float32"],
        k=top_k,
    )
    gathered = builder.node(
        "gather_router_scores",
        "GatherRouterScores",
        [scores, indices],
        ["router_selected_scores"],
        output_shapes=[["tokens", top_k or "top_k"]],
    )
    weights = builder.node(
        "router_normalize_scale",
        "RouterNormalizeScale",
        [gathered],
        ["router_weights"],
        output_shapes=[["tokens", top_k or "top_k"]],
        scoring_func=router.get("scoring_func"),
        route_scale=router.get("route_scale"),
        norm_topk_prob=router.get("norm_topk_prob"),
    )
    dispatched = builder.node(
        "dispatch_tokens",
        "DispatchTokens",
        [flat, indices, weights],
        ["routed_tokens"],
        output_shapes=[["tokens", top_k or "top_k", hidden_size or "hidden"]],
        experts=router.get("experts"),
        top_k=top_k,
    )
    routed_gate = builder.node(
        "routed_expert_gate_proj",
        "GroupedGEMM",
        [dispatched],
        ["routed_gate"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
        experts=routed.get("experts"),
        in_features=hidden_size,
        out_features=intermediate,
        logical_weight="mlp.experts.*.gate_proj.weight",
    )
    routed_up = builder.node(
        "routed_expert_up_proj",
        "GroupedGEMM",
        [dispatched],
        ["routed_up"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
        experts=routed.get("experts"),
        in_features=hidden_size,
        out_features=intermediate,
        logical_weight="mlp.experts.*.up_proj.weight",
    )
    routed_act = builder.node(
        "routed_activation",
        _activation_op(routed.get("activation")),
        [routed_gate],
        ["routed_gate_activated"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
    )
    routed_mul = builder.node(
        "routed_gate_up_mul",
        "Mul",
        [routed_act, routed_up],
        ["routed_product"],
        output_shapes=[["tokens", top_k or "top_k", intermediate or "intermediate"]],
    )
    routed_down = builder.node(
        "routed_expert_down_proj",
        "GroupedGEMM",
        [routed_mul],
        ["routed_expert_output"],
        output_shapes=[["tokens", top_k or "top_k", hidden_size or "hidden"]],
        experts=routed.get("experts"),
        in_features=intermediate,
        out_features=hidden_size,
        logical_weight="mlp.experts.*.down_proj.weight",
    )
    routed_sum = builder.node(
        "combine_routed_experts",
        "WeightedExpertSum",
        [routed_down, indices, weights],
        ["routed_output"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
    )

    shared_intermediate = shared.get("intermediate_size") or intermediate
    shared_gate = builder.node(
        "shared_expert_gate_proj",
        "Linear",
        [flat],
        ["shared_gate"],
        output_shapes=[["tokens", shared_intermediate or "intermediate"]],
        in_features=hidden_size,
        out_features=shared_intermediate,
        logical_weight="mlp.shared_experts.gate_proj.weight",
    )
    shared_up = builder.node(
        "shared_expert_up_proj",
        "Linear",
        [flat],
        ["shared_up"],
        output_shapes=[["tokens", shared_intermediate or "intermediate"]],
        in_features=hidden_size,
        out_features=shared_intermediate,
        logical_weight="mlp.shared_experts.up_proj.weight",
    )
    shared_act = builder.node(
        "shared_activation",
        _activation_op(shared.get("activation")),
        [shared_gate],
        ["shared_gate_activated"],
        output_shapes=[["tokens", shared_intermediate or "intermediate"]],
    )
    shared_mul = builder.node(
        "shared_gate_up_mul",
        "Mul",
        [shared_act, shared_up],
        ["shared_product"],
        output_shapes=[["tokens", shared_intermediate or "intermediate"]],
    )
    shared_down = builder.node(
        "shared_expert_down_proj",
        "Linear",
        [shared_mul],
        ["shared_output"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
        in_features=shared_intermediate,
        out_features=hidden_size,
        logical_weight="mlp.shared_experts.down_proj.weight",
    )
    combined = builder.node(
        "add_routed_shared",
        "Add",
        [routed_sum, shared_down],
        ["moe_flat_output"],
        output_shapes=[["tokens", hidden_size or "hidden"]],
    )
    return builder.node(
        "unflatten_tokens",
        "UnflattenTokens",
        [combined],
        ["moe_output"],
        output_shapes=[["batch", "sequence", hidden_size or "hidden"]],
    )


def _get_layer(architecture: Architecture, layer_index: int) -> Layer:
    if not architecture.layers:
        raise ValueError("Architecture has no decoder layers.")
    if layer_index < 0 or layer_index >= len(architecture.layers):
        raise IndexError("Layer index %d out of range 0..%d" % (layer_index, len(architecture.layers) - 1))
    return architecture.layers[layer_index]


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


def _norm_op(layer: Layer) -> str:
    for component in layer.components:
        if component.name in ("input_layernorm", "attn_norm", "ffn_norm", "post_attention_layernorm"):
            return component.kind
    return "Norm"


def _activation_op(name: Optional[str]) -> str:
    value = str(name or "activation").lower()
    if value in ("silu", "swish"):
        return "SiLU"
    if value == "gelu":
        return "GELU"
    if value == "relu":
        return "ReLU"
    return str(name or "Activation")


def _router_score_op(router: Dict[str, Any]) -> str:
    value = str(router.get("scoring_func") or "sigmoid").lower()
    if value == "softmax":
        return "Softmax"
    if value == "sqrtsoftplus":
        return "SqrtSoftplus"
    if value == "sigmoid":
        return "Sigmoid"
    return "RouterScoreActivation"


def _grouped_gemm_op(routed: Dict[str, Any]) -> str:
    dtype = str(routed.get("expert_dtype") or "").lower()
    if dtype == "mxfp4":
        return "MXFP4GroupedGEMM"
    return "GroupedGEMM"


def _is_deepseek_v4_arch(architecture: Architecture, layer: Optional[Layer] = None) -> bool:
    if str(architecture.model_type).lower().startswith("deepseek_v4"):
        return True
    if layer is None:
        return False
    return _has_deepseek_mla(layer)


def _is_glm_moe_dsa_arch(architecture: Architecture, layer: Optional[Layer] = None) -> bool:
    if str(architecture.model_type).lower().startswith("glm_moe_dsa"):
        return True
    if layer is None:
        return False
    return _has_glm_dsa(layer)


def _is_kimi_k25_arch(architecture: Architecture, layer: Optional[Layer] = None) -> bool:
    if str(architecture.model_type).lower().startswith("kimi_k25"):
        return True
    if str(architecture.text_decoder.get("model_type") or "").lower().startswith("kimi_k2"):
        return True
    if layer is None:
        return False
    return _has_kimi_mla(layer)


def _has_deepseek_mla(layer: Layer) -> bool:
    return bool(_component_details(layer, "mla_sparse_attention"))


def _has_bloom_attention(layer: Layer) -> bool:
    return bool(_component_details(layer, "bloom_attention"))


def _has_bloom_mlp(layer: Layer) -> bool:
    return bool(_component_details(layer, "dense_h_to_4h"))


def _has_kimi_mla(layer: Layer) -> bool:
    return bool(_component_details(layer, "kimi_mla_attention"))


def _has_glm_dsa(layer: Layer) -> bool:
    return bool(_component_details(layer, "glm_dsa_attention"))


def _ensure_graph_input(
    builder: KernelGraphBuilder,
    name: str,
    shape: List[Any],
    dtype: str,
    description: str,
) -> str:
    builder.tensor(name, shape=shape, dtype=dtype, description=description)
    if not any(spec.name == name for spec in builder.inputs):
        builder.inputs.append(builder.tensors[name])
    return name


def _kv_groups(heads: Any, kv_heads: Any) -> Optional[int]:
    try:
        return int(heads) // int(kv_heads)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _group_width(heads: Any, head_dim: Any, groups: Any) -> Optional[int]:
    try:
        return int(heads) * int(head_dim) // int(groups)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _sparse_topk_dim(window: Any, compress_ratio: Any, attention: Dict[str, Any]) -> Any:
    if not compress_ratio:
        return window or "window"
    ratio = _as_int(compress_ratio)
    if ratio == 4:
        return "%s+%s" % (window or "window", attention.get("index_topk") or "index_topk")
    return "%s+compressed_sequence" % (window or "window")


def _product(left: Any, right: Any) -> Optional[int]:
    try:
        return int(left) * int(right)
    except (TypeError, ValueError):
        return None


def _sum_dims(left: Any, right: Any) -> Optional[int]:
    try:
        return int(left) + int(right)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pick_list(values: Optional[List[Any]], index: int) -> Optional[Any]:
    if values is None or index >= len(values):
        return None
    return values[index]


def _default_tensor_description(name: str, op_type: str) -> str:
    return "Tensor '%s' produced by %s." % (name, op_type)


def _description_for_op(op_type: str) -> str:
    descriptions = {
        "RMSNorm": "Root-mean-square normalization over the hidden dimension.",
        "LayerNorm": "Layer normalization over the hidden dimension.",
        "Linear": "Affine projection represented without real weights.",
        "ColumnParallelLinear": "Output-sharded affine projection represented without real weights.",
        "RowParallelLinear": "Input-sharded affine projection followed by an all-reduce in tensor parallel runs.",
        "QuantizedLinear": "Affine projection with quantized checkpoint weights represented without real weights.",
        "ReshapeHeads": "Reshape packed projections into per-head tensors.",
        "SplitQKV": "Split a fused QKV projection into query, key, and value tensors.",
        "RoPE": "Apply rotary position embedding to query or key heads.",
        "PartialRoPE": "Apply rotary position embedding only to the RoPE tail dimensions.",
        "InterleavedRoPE": "Apply rotary position embedding with GLM interleaved even/odd layout.",
        "InversePartialRoPE": "Apply inverse rotary embedding to the output tail dimensions.",
        "NoPE": "Pass-through for layers that skip rotary position embedding.",
        "SplitNoPE": "Extract the non-rotary Q/K dimensions.",
        "SplitRoPE": "Extract the rotary Q/K dimensions.",
        "SplitKVLatent": "Extract compressed KV latent dimensions from a packed MLA projection.",
        "SplitValue": "Extract value dimensions from a packed KV projection.",
        "ExpandKVHeads": "Broadcast MQA-style rotary key dimensions across attention heads.",
        "ConcatQKParts": "Concatenate non-rotary and rotary Q/K dimensions.",
        "L2Norm": "L2-normalize query/key vectors.",
        "HeadRMSScale": "Per-head RMS scaling used by DeepSeek MLA queries.",
        "ActQuant": "Activation quantization metadata node.",
        "FP4ActQuant": "FP4 activation quantization metadata node.",
        "HadamardRotate": "Randomized Hadamard rotation before quantized indexing.",
        "KVCacheUpdate": "Append or update key/value cache state.",
        "RingKVCacheUpdate": "Update the fixed-size sliding-window KV ring buffer.",
        "CompressedKVCacheUpdate": "Update compressed KV cache entries.",
        "RepeatKV": "Repeat grouped-query key/value heads to match attention heads.",
        "MatMulQK": "Compute scaled attention logits.",
        "ALiBiBias": "Add BLOOM ALiBi positional slopes to attention logits.",
        "CausalMask": "Apply causal and optional local/chunk mask.",
        "Softmax": "Normalize logits into attention probabilities.",
        "Dropout": "Drop attention probabilities during training.",
        "MatMulPV": "Compute weighted value context.",
        "SparseAttention": "DeepSeek sparse attention kernel with indexed KV gather and online softmax.",
        "DynamicSparseAttention": "GLM dynamic sparse attention using top-k selected KV positions.",
        "WindowTopKIndices": "Construct sliding-window attention indices.",
        "CompressTopKIndices": "Construct deterministic compressed-cache attention indices.",
        "SparseIndexScore": "Learned compressed-cache index scoring.",
        "IndexShareReuse": "Reuse sparse top-k indices from a shared GLM indexer group.",
        "ConcatKVCache": "Concatenate current/window KV and compressed KV streams.",
        "ConcatTopKIndices": "Concatenate sparse-attention index sets.",
        "MergeHeads": "Merge attention heads back to hidden dimension.",
        "GroupHeads": "Group attention heads before DeepSeek grouped output projection.",
        "GroupedOutputLowRank": "Per-output-group low-rank projection used by DeepSeek wo_a.",
        "FlattenGroups": "Flatten grouped low-rank output activations.",
        "ResidualAdd": "Add a residual branch to a block output.",
        "HyperConnectionPre": "DeepSeek Hyper-Connection pre-mix from multiple hidden-state copies.",
        "HyperConnectionPost": "DeepSeek Hyper-Connection post-mix that restores hidden-state copies.",
        "SiLU": "Silu activation.",
        "SqrtSoftplus": "DeepSeek router scoring activation sqrt(softplus(x)).",
        "GELU": "GELU activation.",
        "ReLU": "ReLU activation.",
        "Mul": "Elementwise multiplication.",
        "FlattenTokens": "Flatten batch and sequence dimensions into a token dimension.",
        "UnflattenTokens": "Restore token dimension to batch and sequence dimensions.",
        "RouterLinear": "Compute MoE router logits over experts.",
        "RouterBiasAdd": "Add expert-selection bias before top-k routing.",
        "HashRouteLookup": "Look up predetermined expert IDs from token IDs in hash-routed layers.",
        "GatherRouterScores": "Gather original router scores at selected expert indices.",
        "RouterNormalizeScale": "Normalize and scale selected router weights.",
        "TopK": "Select top-k experts per token.",
        "Sigmoid": "Convert selected router logits to routing weights.",
        "DispatchTokens": "Dispatch token copies to selected experts.",
        "GroupedGEMM": "Expert-parallel grouped matrix multiplication.",
        "MXFP4GroupedGEMM": "Expert-parallel grouped matrix multiplication with MXFP4 expert weights.",
        "FP4GroupedGEMM": "Expert-parallel grouped matrix multiplication with FP4 expert weights.",
        "Int4GroupedGEMM": "Expert-parallel grouped matrix multiplication with int4-packed expert weights.",
        "SplitGate": "Extract gate half from a packed gate/up projection.",
        "SplitUp": "Extract up half from a packed gate/up projection.",
        "SwiGLUClamp": "Apply DeepSeek's optional SwiGLU clipping limits before activation.",
        "ApplyRouterWeights": "Scale routed expert intermediate states by selected router weights.",
        "ExpertReduce": "Accumulate per-expert outputs back into token order.",
        "WeightedExpertSum": "Combine routed expert outputs with router weights.",
        "Add": "Elementwise addition.",
        "GatedKVCompression": "Compress KV blocks through learned score softmax pooling.",
    }
    return descriptions.get(op_type, "Metadata-only kernel node.")


def _formula_for_op(op_type: str, attrs: Dict[str, Any]) -> str:
    hidden = attrs.get("in_features", "in")
    out = attrs.get("out_features", "out")
    top_k = attrs.get("k", attrs.get("top_k", "K"))
    experts = attrs.get("experts", "E")
    groups = attrs.get("groups", "G")
    route_scale = attrs.get("route_scale", "scale")
    rope_dim = attrs.get("rope_dim", "Dr")
    nope_dim = attrs.get("nope_dim", "Dn")
    kv_rank = attrs.get("kv_rank", "Rkv")
    ratio = attrs.get("compress_ratio", "R")
    formulas = {
        "RMSNorm": "y = x / sqrt(mean(x^2, axis=-1) + eps) * gamma",
        "LayerNorm": "y = (x - mean(x, axis=-1)) / sqrt(var(x, axis=-1) + eps) * gamma + beta",
        "Norm": "y = norm(x)",
        "Linear": "y[..., %s] = x[..., %s] @ W[%s,%s]^T + b" % (out, hidden, out, hidden),
        "ColumnParallelLinear": "y_part = x @ W_part^T; full y is concatenated across tensor-parallel ranks",
        "RowParallelLinear": "y = all_reduce(x_part @ W_part^T) + b",
        "QuantizedLinear": "xq, sx = quantize(x); y = dequant_gemm(xq, Wq, sx, sw)",
        "ReshapeHeads": "y[B,S,H,D] = reshape(x[B,S,H*D])",
        "SplitQKV": "q, k, v = split(fused_qkv, 3, axis=-1)",
        "RoPE": "y_even = x_even*cos(pos,theta) - x_odd*sin(pos,theta); y_odd = x_even*sin(pos,theta) + x_odd*cos(pos,theta)",
        "PartialRoPE": "y[..., -%s:] = RoPE(x[..., -%s:]); y[..., :-%s] = x[..., :-%s]" % (rope_dim, rope_dim, rope_dim, rope_dim),
        "InterleavedRoPE": "y[..., 0::2], y[..., 1::2] = rotate_interleaved_pairs(x, cos, sin)",
        "InversePartialRoPE": "y[..., -%s:] = inverse_RoPE(x[..., -%s:]); y[..., :-%s] = x[..., :-%s]" % (rope_dim, rope_dim, rope_dim, rope_dim),
        "NoPE": "y = x",
        "SplitNoPE": "nope = x[..., :%s]" % nope_dim,
        "SplitRoPE": "rope = x[..., -%s:]" % rope_dim,
        "SplitKVLatent": "kv_latent = packed[..., :%s]" % kv_rank,
        "SplitValue": "v = packed[..., -V:]",
        "ExpandKVHeads": "y[B,S,H,D] = repeat(x[B,S,1,D], repeats=H, axis=head)",
        "ConcatQKParts": "qk = concat(nope, rope, axis=-1)",
        "L2Norm": "y = x / max(||x||_2, eps)",
        "HeadRMSScale": "y = x * rsqrt(mean(x^2, axis=-1) + eps)",
        "ActQuant": "xq, scale = quantize_activation(x, block_size)",
        "FP4ActQuant": "xq, scale = fp4_quantize_activation(x, block_size=32)",
        "HadamardRotate": "y = hadamard(x) / sqrt(width)",
        "KVCacheUpdate": "cache[:, past:past+S, :, :] = x",
        "RingKVCacheUpdate": "cache[:, position % window] = kv",
        "CompressedKVCacheUpdate": "compressed_cache[:, floor(position/%s)] = compressed_kv" % ratio,
        "RepeatKV": "y = repeat_interleave(x, repeats=%s, axis=head)" % groups,
        "MatMulQK": "scores[B,H,S,QK] = (Q[B,H,S,D] @ K[B,H,QK,D]^T) / sqrt(D)",
        "ALiBiBias": "scores = scores + slope[head] * relative_position",
        "CausalMask": "masked_scores = scores + causal_or_local_mask",
        "Softmax": "probs = exp(x - max(x)) / sum(exp(x - max(x)), axis=-1)",
        "Dropout": "y = dropout(x, p)",
        "MatMulPV": "context[B,H,S,D] = probs[B,H,S,QK] @ V[B,H,QK,D]",
        "SparseAttention": "o = online_softmax(q @ gather(kv, topk)^T * scale, attn_sink) @ gather(kv, topk)",
        "DynamicSparseAttention": "o = softmax(q @ gather(k, topk)^T / sqrt(Dqk)) @ gather(v, topk)",
        "WindowTopKIndices": "idx[b,s,:] = valid positions in sliding window ending at token s",
        "CompressTopKIndices": "idx[b,s,:] = valid compressed-cache positions before token s",
        "SparseIndexScore": "score[b,s,t] = sum_h(relu(q[b,s,h] @ kc[b,t]) * w[b,s,h])",
        "IndexShareReuse": "idx = cached_sparse_indices_from_shared_indexer_group",
        "ConcatKVCache": "kv_sparse = concat(window_kv, compressed_kv, axis=sequence)",
        "ConcatTopKIndices": "idx_sparse = concat(window_indices, compressed_indices, axis=-1)",
        "MergeHeads": "y[B,S,H*D] = reshape(transpose(context[B,H,S,D]))",
        "GroupHeads": "y[B,S,G,(H/G)*D] = reshape(context[B,S,H,D])",
        "GroupedOutputLowRank": "y[b,s,g,r] = x[b,s,g,:] @ W[g,r,:]^T",
        "FlattenGroups": "y[B,S,G*R] = reshape(x[B,S,G,R])",
        "ResidualAdd": "y = residual + branch",
        "HyperConnectionPre": "y[B,S,D] = sum_hc(pre[B,S,hc] * residual[B,S,hc,D])",
        "HyperConnectionPost": "y[B,S,hc,D] = post[B,S,hc] * branch[B,S,D] + sum_j(comb[B,S,hc,j] * residual[B,S,j,D])",
        "SiLU": "y = x * sigmoid(x)",
        "SqrtSoftplus": "y = sqrt(softplus(x))",
        "GELU": "y = GELU(x)",
        "ReLU": "y = max(x, 0)",
        "Mul": "y = a * b",
        "FlattenTokens": "y[B*S,H] = reshape(x[B,S,H])",
        "UnflattenTokens": "y[B,S,H] = reshape(x[B*S,H])",
        "RouterLinear": "router_logits[T,%s] = tokens[T,H] @ W_router[%s,H]^T" % (experts, experts),
        "RouterBiasAdd": "selection_scores = router_scores + expert_bias",
        "HashRouteLookup": "indices[T,K] = tid2eid[input_ids[T], :K]",
        "GatherRouterScores": "selected_scores[T,K] = router_scores[T, indices[T,K]]",
        "RouterNormalizeScale": "weights = %s * selected_scores / sum(selected_scores, axis=-1)" % route_scale,
        "TopK": "indices, scores = top_k(router_logits, k=%s, axis=experts)" % top_k,
        "Sigmoid": "weights = 1 / (1 + exp(-scores))",
        "DispatchTokens": "routed_tokens[T,K,H] = tokens[T,H] selected by top_k expert indices",
        "GroupedGEMM": "y[t,k,:] = x[t,k,:] @ W_expert[e=tokens_expert[t,k]]^T",
        "MXFP4GroupedGEMM": "y[t,k,:] = dequant_mxfp4_gemm(x[t,k,:], W_mxfp4[expert=tokens_expert[t,k]])",
        "FP4GroupedGEMM": "y[t,k,:] = dequant_fp4_gemm(x[t,k,:], W4[expert=tokens_expert[t,k]])",
        "Int4GroupedGEMM": "y[t,k,:] = dequant_int4_gemm(x[t,k,:], W4_packed[expert=tokens_expert[t,k]], scale)",
        "SplitGate": "gate = packed[..., :intermediate]",
        "SplitUp": "up = packed[..., intermediate:]",
        "SwiGLUClamp": "gate = clamp(gate, max=limit); up = clamp(up, min=-limit, max=limit)",
        "ApplyRouterWeights": "y[T,K,I] = router_weight[T,K] * expert_mid[T,K,I]",
        "ExpertReduce": "y[T,H] = sum_k(expert_output[T,K,H])",
        "WeightedExpertSum": "y[t,H] = sum_k(router_weight[t,k] * expert_output[t,k,H])",
        "Add": "y = a + b",
        "GatedKVCompression": "kv_c = sum_r(softmax(score + ape)[r] * kv[r]) over each compression block",
    }
    return formulas.get(op_type, "y = %s(inputs)" % op_type)


def _sanitize(value: str) -> str:
    return value.replace("/", "_").replace("[", "_").replace("]", "").replace("-", "_")
