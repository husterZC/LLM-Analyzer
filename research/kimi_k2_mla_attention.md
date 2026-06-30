# Kimi K2/K2.5 Attention Mechanism

## Summary

Kimi K2 uses **Multi-head Latent Attention (MLA)** for its language model attention mechanism. This is the same attention family used by DeepSeek-V3, and Kimi K2's public configuration even uses the `DeepseekV3ForCausalLM` architecture wrapper.

Kimi K2.5 is described as being built on top of Kimi K2, with additional multimodal training and a vision/video encoder path. Based on the public reports and model configuration, the language decoder backbone should therefore be treated as **MLA-based**, not GQA-based.

The important nuance is that **Kimi K2 is not shape-identical to DeepSeek-V3**. It uses the same MLA latent ranks and per-head dimensions, but it halves the number of attention heads.

## Kimi K2 vs DeepSeek-V3 MLA Hyperparameters

| Parameter | Kimi K2 | DeepSeek-V3 | Same? |
| --- | ---: | ---: | --- |
| `hidden_size` | 7168 | 7168 | yes |
| `num_hidden_layers` | 61 | 61 | yes |
| `num_attention_heads` | 64 | 128 | no |
| `num_key_value_heads` | 64 | 128 | no |
| `q_lora_rank` | 1536 | 1536 | yes |
| `kv_lora_rank` | 512 | 512 | yes |
| `qk_nope_head_dim` | 128 | 128 | yes |
| `qk_rope_head_dim` | 64 | 64 | yes |
| `v_head_dim` | 128 | 128 | yes |

## Shape Implications

The compressed MLA KV cache shape per token is effectively the same:

```text
kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
```

However, the expanded attention tensor dimensions differ because Kimi K2 has 64 heads while DeepSeek-V3 has 128 heads.

| Derived dimension | Kimi K2 | DeepSeek-V3 |
| --- | ---: | ---: |
| Expanded query dimension, `num_heads * (qk_nope_head_dim + qk_rope_head_dim)` | `64 * 192 = 12288` | `128 * 192 = 24576` |
| Expanded KV dimension, `num_heads * (qk_nope_head_dim + v_head_dim)` | `64 * 256 = 16384` | `128 * 256 = 32768` |
| Attention output input dimension, `num_heads * v_head_dim` | `64 * 128 = 8192` | `128 * 128 = 16384` |

## Conclusion

Kimi K2/K2.5 is close to DeepSeek-V3 in attention design because it uses MLA with the same key latent-rank and per-head settings:

- `q_lora_rank = 1536`
- `kv_lora_rank = 512`
- `qk_nope_head_dim = 128`
- `qk_rope_head_dim = 64`
- `v_head_dim = 128`

But it is **not exactly the same attention shape** as DeepSeek-V3 because Kimi K2 uses **64 attention heads** instead of **128**. In practical terms, Kimi K2 keeps the same compressed KV cache size per token but reduces the expanded attention compute/projection dimensions relative to DeepSeek-V3.

## Sources

- Kimi K2 config: https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json
- DeepSeek-V3 config: https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
- Kimi K2.5 report: https://arxiv.org/abs/2602.02276
- Kimi K2 report: https://arxiv.org/abs/2507.20534
