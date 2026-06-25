# LLM-Analyzer

Metadata-only tools for inspecting open-source LLM architectures.

The first tool is an architecture viewer for Hugging Face models. It downloads
configuration, tokenizer metadata, model cards, and modeling/configuration source
files while skipping actual weight files.

## First-Time Setup

Use Python 3.8 or newer. Python 3.11 is recommended.

On this workspace, the system `python3` is too old. Use `python3.11` or the
explicit Anaconda path:

```bash
python3.11 --version
/usr/local/anaconda3/bin/python3.11 --version
```

After cloning:

```bash
git clone <repo-url>
cd LLM-Analyzer

# Optional but recommended: create an isolated local environment.
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode so the `llm-analyzer` command is available.
python -m pip install -e .

# Optional: install ONNX export support for Netron-compatible kernel graphs.
python -m pip install -e ".[onnx]"
```

The current tool uses only the Python standard library, so there are no required
runtime packages to install. Editable install is only needed for the convenient
`llm-analyzer` command; direct `python -m llm_analyzer ...` usage also works.

## Hugging Face Access

Public models do not require a Hugging Face token. Gated or private models do.
For gated models, first accept the model license on Hugging Face, then provide a
token with one of these methods:

```bash
# Option 1: environment variable
export HF_TOKEN=hf_xxx

# Option 2: local token file
printf '%s\n' 'hf_xxx' > .hf_token.txt
chmod 600 .hf_token.txt

# Option 3: command-line argument
python -m llm_analyzer inspect meta-llama/Llama-4-Maverick-17B-128E --hf-token hf_xxx
```

`.hf_token.txt` is ignored by git. Do not commit tokens.

The tool checks credentials in this order:

1. `--hf-token`
2. `HF_TOKEN`
3. `HUGGINGFACE_HUB_TOKEN`
4. `.hf_token.txt`

## Usage

The simplest entrypoint is the Makefile:

```bash
make help
make inspect
make arch
make arch MODEL=Qwen/Qwen2.5-7B-Instruct
make layer-diagram MODEL=meta-llama/Llama-4-Maverick-17B-128E LAYER=1
make attention-diagram LAYER=0
make mlp-diagram LAYER=0
make moe-detail-diagram MOE_LAYER=1
make setup-onnx
make onnx-graphs
make analyze-list MODEL_LIST=model_lists/evaluated_models.txt
make reproduce-evaluated
```

By default, `make` uses `.venv/bin/python` when a local virtual environment
exists, otherwise `/usr/local/anaconda3/bin/python3.11`:

```text
PYTHON=.venv/bin/python
MODEL=meta-llama/Llama-4-Maverick-17B-128E
MODEL_LIST=model_lists/evaluated_models.txt
REVISION=main
MAX_FILE_MB=50.0
LAYER=0
MOE_LAYER=1
OUT_DIR=outputs
CACHE_DIR=.llm_analyzer_cache
```

Run directly from the repo:

```bash
/usr/local/anaconda3/bin/python3.11 -m llm_analyzer inspect meta-llama/Llama-4-Maverick-17B-128E
/usr/local/anaconda3/bin/python3.11 -m llm_analyzer arch meta-llama/Llama-4-Maverick-17B-128E --level model --format mermaid --out outputs/meta-llama_Llama-4-Maverick-17B-128E/overview/model.mmd
/usr/local/anaconda3/bin/python3.11 -m llm_analyzer arch meta-llama/Llama-4-Maverick-17B-128E --level layer --layer 0 --format mermaid --out outputs/meta-llama_Llama-4-Maverick-17B-128E/layers/layer_0/block.mmd
/usr/local/anaconda3/bin/python3.11 -m llm_analyzer arch meta-llama/Llama-4-Maverick-17B-128E --level attention --layer 0 --format mermaid --out outputs/meta-llama_Llama-4-Maverick-17B-128E/details/layer_0/attention.mmd
/usr/local/anaconda3/bin/python3.11 -m llm_analyzer arch meta-llama/Llama-4-Maverick-17B-128E --level mlp --layer 0 --format mermaid --out outputs/meta-llama_Llama-4-Maverick-17B-128E/details/layer_0/mlp.mmd
/usr/local/anaconda3/bin/python3.11 -m llm_analyzer arch meta-llama/Llama-4-Maverick-17B-128E --level moe --layer 1 --format mermaid --out outputs/meta-llama_Llama-4-Maverick-17B-128E/details/layer_1/moe.mmd
/usr/local/anaconda3/bin/python3.11 -m llm_analyzer arch meta-llama/Llama-4-Maverick-17B-128E --level layer --layer 0 --format onnx --out outputs/meta-llama_Llama-4-Maverick-17B-128E/onnx/layer_0/kernels.onnx
/usr/local/anaconda3/bin/python3.11 -m llm_analyzer arch meta-llama/Llama-4-Maverick-17B-128E --format json --out outputs/meta-llama_Llama-4-Maverick-17B-128E/ir/architecture.json
```

If installed as a package, the equivalent command is:

```bash
llm-analyzer inspect Qwen/Qwen2.5-7B-Instruct
```

Useful commands:

```bash
# Download metadata only.
llm-analyzer fetch Qwen/Qwen2.5-7B-Instruct

# Print a compact architecture summary.
llm-analyzer inspect Qwen/Qwen2.5-7B-Instruct

# Save top-level Mermaid diagram.
llm-analyzer arch Qwen/Qwen2.5-7B-Instruct --level model --format mermaid --out outputs/Qwen_Qwen2.5-7B-Instruct/overview/model.mmd

# Save one decoder-layer Mermaid diagram.
llm-analyzer arch Qwen/Qwen2.5-7B-Instruct --level layer --layer 0 --format mermaid --out outputs/Qwen_Qwen2.5-7B-Instruct/layers/layer_0/block.mmd

# Save detailed attention and MLP diagrams.
llm-analyzer arch Qwen/Qwen2.5-7B-Instruct --level attention --layer 0 --format mermaid --out outputs/Qwen_Qwen2.5-7B-Instruct/details/layer_0/attention.mmd
llm-analyzer arch Qwen/Qwen2.5-7B-Instruct --level mlp --layer 0 --format mermaid --out outputs/Qwen_Qwen2.5-7B-Instruct/details/layer_0/mlp.mmd

# Save detailed MoE routing diagram for a sparse layer.
llm-analyzer arch meta-llama/Llama-4-Maverick-17B-128E --level moe --layer 1 --format mermaid --out outputs/meta-llama_Llama-4-Maverick-17B-128E/details/layer_1/moe.mmd

# Save a flat ONNX kernel-flow graph for Netron.
llm-analyzer arch meta-llama/Llama-4-Maverick-17B-128E --level layer --layer 0 --format onnx --out outputs/meta-llama_Llama-4-Maverick-17B-128E/onnx/layer_0/kernels.onnx
llm-analyzer arch meta-llama/Llama-4-Maverick-17B-128E --level layer --layer 1 --format onnx --out outputs/meta-llama_Llama-4-Maverick-17B-128E/onnx/layer_1/kernels.onnx

# Save normalized architecture IR.
llm-analyzer arch Qwen/Qwen2.5-7B-Instruct --format json --out outputs/Qwen_Qwen2.5-7B-Instruct/ir/architecture.json

# Analyze a list of models and write the full selected output set.
llm-analyzer batch model_lists/evaluated_models.txt --out-dir outputs
```

## Batch Model Lists

Use `make analyze-list` when the input is a file of model IDs instead of a
single `MODEL` value:

```bash
make setup-onnx
make analyze-list MODEL_LIST=model_lists/evaluated_models.txt
```

The included evaluated-model manifest can regenerate the checked-in output tree:

```bash
make reproduce-evaluated
```

The list format is pipe-delimited:

```text
model|layers|attention_layers|mlp_layers|moe_layers
meta-llama/Llama-4-Maverick-17B-128E|0,1|0|0|1
bigscience/bloom|0|0|0|
```

Layer lists are comma-separated. Empty fields are allowed, for example dense
models leave `moe_layers` empty. The batch command writes `model.mmd`,
`architecture.json`, layer block diagrams, attention/MLP/MoE detail diagrams,
and ONNX kernel-flow graphs for the selected layers.

The Makefile writes outputs in this hierarchy:

```text
outputs/
  <model_slug>/
    overview/
      model.mmd
    layers/
      layer_<N>/
        block.mmd
    details/
      layer_<N>/
        attention.mmd
        mlp.mmd
        moe.mmd
    onnx/
      layer_<N>/
        kernels.onnx
        attention.onnx
        mlp.onnx
        moe.onnx
    ir/
      architecture.json
```

For example, the default model writes to:

```text
outputs/meta-llama_Llama-4-Maverick-17B-128E/
```

## Evaluated Models

The repository includes generated metadata-only outputs for representative
state-of-the-art open-source architectures. Each output directory contains a
normalized JSON IR, Mermaid diagrams, and ONNX kernel-flow graphs for one or
more representative layers.

The manifest used to reproduce these outputs is
`model_lists/evaluated_models.txt`; run `make reproduce-evaluated` after
installing ONNX support.

| Model | Detected architecture | Output directory |
| --- | --- | --- |
| `meta-llama/Llama-4-Maverick-17B-128E` | Llama 4 multimodal MoE | `outputs/meta-llama_Llama-4-Maverick-17B-128E/` |
| `deepseek-ai/DeepSeek-V4-Pro` | DeepSeek V4 Hyper-Connection + sparse MLA MoE | `outputs/deepseek-ai_DeepSeek-V4-Pro/` |
| `zai-org/GLM-5.2` | GLM MoE DSA/MLA with IndexShare | `outputs/zai-org_GLM-5.2/` |
| `moonshotai/Kimi-K2.5` | Kimi multimodal wrapper with DeepSeek-V3-style MLA/MoE text stack | `outputs/moonshotai_Kimi-K2.5/` |
| `bigscience/bloom` | BLOOM fused-QKV attention with ALiBi and dense GELU MLP | `outputs/bigscience_bloom/` |
| `openai/gpt-oss-120b` | GPT-OSS GQA + sliding-window RoPE + MXFP4 MoE experts | `outputs/openai_gpt-oss-120b/` |
| `mistralai/Mistral-Medium-3.5-128B` | Mistral3 multimodal dense decoder with Pixtral vision tower | `outputs/mistralai_Mistral-Medium-3.5-128B/` |
| `Qwen/Qwen3.5-122B-A10B` | Qwen3.5 multimodal MoE, 256 routed experts top-8 | `outputs/Qwen_Qwen3.5-122B-A10B/` |
| `Qwen/Qwen3.5-397B-A17B` | Qwen3.5 multimodal MoE, 512 routed experts top-10 | `outputs/Qwen_Qwen3.5-397B-A17B/` |
| `zai-org/GLM-5` | GLM MoE DSA/MLA with dense prefix layers and MoE layers | `outputs/zai-org_GLM-5/` |
| `XiaomiMiMo/MiMo-V2.5-Base` | MiMo V2 omnimodal MoE with hybrid full/SWA fused-QKV attention | `outputs/XiaomiMiMo_MiMo-V2.5-Base/` |
| `XiaomiMiMo/MiMo-V2-Flash` | MiMo V2 text MoE with hybrid full/SWA split-QKV attention | `outputs/XiaomiMiMo_MiMo-V2-Flash/` |
| `tencent/Hy3-preview-Base` | Hy3 MoE with Q/K RMSNorm GQA, sigmoid-bias router, shared MLP, and MTP | `outputs/tencent_Hy3-preview-Base/` |

Recent model-specific lowering includes:

- BLOOM: fused `query_key_value`, `SplitQKV`, ALiBi bias, and dense GELU MLP.
- GPT-OSS: GQA attention with RoPE/sliding-window metadata and `MXFP4GroupedGEMM` routed expert kernels.
- Qwen3.5 MoE: optional shared-expert handling so graphs do not invent shared expert branches when absent.
- Kimi K2.5: DeepSeek-V3-style q-LoRA/compressed-KV MLA and int4-packed routed experts.
- GLM-5/5.2: MLA-style projections, dynamic sparse attention, IndexShare, and MoE router correction bias.
- DeepSeek V4: Hyper-Connection blocks, sparse MLA compression/indexing, hash-routed early layers, and FP4 expert kernels.
- MiMo V2/2.5: fused or split QKV GQA, partial RoPE, full/sliding-window hybrid attention, attention sink bias, value scaling, and noaux_tc sigmoid MoE routing.
- Hy3 preview: split-QKV GQA with per-head Q/K RMSNorm, RoPE, sigmoid correction-bias top-k routing, routed expert SwiGLU, shared SwiGLU MLP, and MTP metadata.

## Local Cache

Downloaded metadata is cached under:

```text
.llm_analyzer_cache/
```

The cache is ignored by git. It may contain model cards, config files, tokenizer
metadata, and source files, but the downloader skips model weights.

To use a different cache location:

```bash
llm-analyzer inspect Qwen/Qwen2.5-7B-Instruct --cache-dir /tmp/llm_analyzer_cache
```

## Output Formats

- `--format summary`: human-readable architecture summary
- `--format mermaid`: Mermaid flowchart text
- `--format json`: normalized architecture IR
- `--format onnx`: flat kernel-flow graph with custom metadata-only ONNX ops

Diagram levels:

- `--level model`: top-level model flow
- `--level layer`: compact decoder-layer flow
- `--level attention`: attention internals for one layer
- `--level mlp`: dense MLP or expert/shared MLP internals for one layer
- `--level moe`: MoE router, expert dispatch, shared expert, and combine path

ONNX export is intended for graph inspection in tools such as Netron. The ONNX
files contain custom `llm_analyzer` ops and tensor edges, but no real weights.
They are metadata-only graphs, not executable inference models.

Each ONNX kernel node includes metadata attributes:

- `formula`: math formula for the kernel
- `input_links`: input tensor names, shapes, dtypes, and descriptions
- `output_links`: output tensor names, shapes, dtypes, and descriptions
- `input_dims`: compact input tensor dimension map
- `output_dims`: compact output tensor dimension map

Each ONNX `ValueInfoProto` also has a readable tensor name and doc string where
available, so graph inspectors can show tensor link metadata without adding
extra nodes to the kernel flow.

The ONNX exporter names tensor edges with compact shape-bearing labels, for
example:

```text
q[B,S,5120]
kh[B,S,8,128]
score[B,40,S,KV]
gate[B,S,16384]
rlogit[T,128]
rtok[T,1,5120]
```

Legend: `B` = batch, `S` = sequence, `KV` = cached key/value sequence length,
and `T` = flattened token count.

Mermaid files can be viewed in GitHub Markdown, many editors, or rendered by
pasting the `.mmd` content into [Mermaid Live Editor](https://mermaid.live).

## Development Checks

Run the current test suite with:

```bash
/usr/local/anaconda3/bin/python3.11 -m unittest discover -s tests
/usr/local/anaconda3/bin/python3.11 -m py_compile llm_analyzer/*.py
```

## What It Downloads

Included:

- `config.json`
- tokenizer and generation config JSON
- model card and license text
- `modeling_*.py`, `configuration_*.py`, and related source files
- tokenizer metadata files such as `.model`, `.vocab`, `.merges`, `.tiktoken`

Skipped:

- `.safetensors`
- `.bin`
- `.pt`, `.pth`, `.ckpt`
- `.gguf`, `.ggml`
- `.onnx`
- other common serialized weight formats
