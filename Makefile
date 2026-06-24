PYTHON ?= /usr/local/anaconda3/bin/python3.11
MODEL ?= meta-llama/Llama-4-Maverick-17B-128E
LAYER ?= 0
MOE_LAYER ?= 1
OUT_DIR ?= outputs
CACHE_DIR ?= .llm_analyzer_cache

MODEL_SLUG := $(subst /,_,$(MODEL))
MODEL_OUT_DIR := $(OUT_DIR)/$(MODEL_SLUG)
OVERVIEW_DIR := $(MODEL_OUT_DIR)/overview
LAYERS_DIR := $(MODEL_OUT_DIR)/layers
DETAILS_DIR := $(MODEL_OUT_DIR)/details
IR_DIR := $(MODEL_OUT_DIR)/ir
ONNX_DIR := $(MODEL_OUT_DIR)/onnx

.PHONY: help setup setup-onnx test compile fetch inspect arch model-diagram layer-diagram attention-diagram mlp-diagram moe-detail-diagram detail-diagrams moe-layer-diagram json kernel-onnx moe-kernel-onnx attention-onnx mlp-onnx moe-onnx onnx-graphs all clean-cache

help:
	@echo "LLM-Analyzer targets"
	@echo ""
	@echo "  make setup              Create .venv and install editable package"
	@echo "  make setup-onnx         Create .venv and install editable package with ONNX export"
	@echo "  make test               Run unit tests and py_compile"
	@echo "  make fetch              Download metadata only, skipping weights"
	@echo "  make inspect            Print architecture summary"
	@echo "  make model-diagram      Write top-level Mermaid diagram"
	@echo "  make layer-diagram      Write layer Mermaid diagram, default LAYER=0"
	@echo "  make attention-diagram  Write detailed attention diagram, default LAYER=0"
	@echo "  make mlp-diagram        Write detailed MLP diagram, default LAYER=0"
	@echo "  make moe-detail-diagram Write detailed MoE routing diagram, default MOE_LAYER=1"
	@echo "  make detail-diagrams    Write attention, MLP, and MoE detail diagrams"
	@echo "  make moe-layer-diagram  Write compact Llama-4 Maverick MoE layer diagram, default layer 1"
	@echo "  make json               Write normalized architecture JSON"
	@echo "  make kernel-onnx        Write flat layer kernel-flow ONNX graph, default LAYER=0"
	@echo "  make moe-kernel-onnx    Write flat MoE layer kernel-flow ONNX graph, default MOE_LAYER=1"
	@echo "  make attention-onnx     Write attention kernel-flow ONNX graph, default LAYER=0"
	@echo "  make mlp-onnx           Write MLP kernel-flow ONNX graph, default LAYER=0"
	@echo "  make moe-onnx           Write MoE kernel-flow ONNX graph, default MOE_LAYER=1"
	@echo "  make onnx-graphs        Write layer, attention, MLP, and MoE ONNX graphs"
	@echo "  make arch               Write model, layer, and JSON outputs"
	@echo "  make all                Run test and arch"
	@echo "  make clean-cache        Remove local metadata cache"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  MODEL=$(MODEL)"
	@echo "  LAYER=$(LAYER)"
	@echo "  MOE_LAYER=$(MOE_LAYER)"
	@echo "  OUT_DIR=$(OUT_DIR)"
	@echo "  MODEL_OUT_DIR=$(MODEL_OUT_DIR)"
	@echo "  CACHE_DIR=$(CACHE_DIR)"
	@echo ""
	@echo "Example:"
	@echo "  make arch MODEL=Qwen/Qwen2.5-7B-Instruct"

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -e .

setup-onnx:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -e ".[onnx]"

test:
	$(PYTHON) -m unittest discover -s tests
	$(PYTHON) -m py_compile llm_analyzer/*.py

compile:
	$(PYTHON) -m py_compile llm_analyzer/*.py

fetch:
	$(PYTHON) -m llm_analyzer fetch "$(MODEL)" --cache-dir "$(CACHE_DIR)"

inspect:
	$(PYTHON) -m llm_analyzer inspect "$(MODEL)" --cache-dir "$(CACHE_DIR)"

model-diagram:
	mkdir -p "$(OVERVIEW_DIR)"
	$(PYTHON) -m llm_analyzer arch "$(MODEL)" --cache-dir "$(CACHE_DIR)" --level model --format mermaid --out "$(OVERVIEW_DIR)/model.mmd"

layer-diagram:
	mkdir -p "$(LAYERS_DIR)/layer_$(LAYER)"
	$(PYTHON) -m llm_analyzer arch "$(MODEL)" --cache-dir "$(CACHE_DIR)" --level layer --layer "$(LAYER)" --format mermaid --out "$(LAYERS_DIR)/layer_$(LAYER)/block.mmd"

attention-diagram:
	mkdir -p "$(DETAILS_DIR)/layer_$(LAYER)"
	$(PYTHON) -m llm_analyzer arch "$(MODEL)" --cache-dir "$(CACHE_DIR)" --level attention --layer "$(LAYER)" --format mermaid --out "$(DETAILS_DIR)/layer_$(LAYER)/attention.mmd"

mlp-diagram:
	mkdir -p "$(DETAILS_DIR)/layer_$(LAYER)"
	$(PYTHON) -m llm_analyzer arch "$(MODEL)" --cache-dir "$(CACHE_DIR)" --level mlp --layer "$(LAYER)" --format mermaid --out "$(DETAILS_DIR)/layer_$(LAYER)/mlp.mmd"

moe-detail-diagram:
	mkdir -p "$(DETAILS_DIR)/layer_$(MOE_LAYER)"
	$(PYTHON) -m llm_analyzer arch "$(MODEL)" --cache-dir "$(CACHE_DIR)" --level moe --layer "$(MOE_LAYER)" --format mermaid --out "$(DETAILS_DIR)/layer_$(MOE_LAYER)/moe.mmd"

detail-diagrams: attention-diagram mlp-diagram moe-detail-diagram

moe-layer-diagram:
	$(MAKE) layer-diagram LAYER=1

json:
	mkdir -p "$(IR_DIR)"
	$(PYTHON) -m llm_analyzer arch "$(MODEL)" --cache-dir "$(CACHE_DIR)" --format json --out "$(IR_DIR)/architecture.json"

kernel-onnx:
	mkdir -p "$(ONNX_DIR)/layer_$(LAYER)"
	$(PYTHON) -m llm_analyzer arch "$(MODEL)" --cache-dir "$(CACHE_DIR)" --level layer --layer "$(LAYER)" --format onnx --out "$(ONNX_DIR)/layer_$(LAYER)/kernels.onnx"

moe-kernel-onnx:
	mkdir -p "$(ONNX_DIR)/layer_$(MOE_LAYER)"
	$(PYTHON) -m llm_analyzer arch "$(MODEL)" --cache-dir "$(CACHE_DIR)" --level layer --layer "$(MOE_LAYER)" --format onnx --out "$(ONNX_DIR)/layer_$(MOE_LAYER)/kernels.onnx"

attention-onnx:
	mkdir -p "$(ONNX_DIR)/layer_$(LAYER)"
	$(PYTHON) -m llm_analyzer arch "$(MODEL)" --cache-dir "$(CACHE_DIR)" --level attention --layer "$(LAYER)" --format onnx --out "$(ONNX_DIR)/layer_$(LAYER)/attention.onnx"

mlp-onnx:
	mkdir -p "$(ONNX_DIR)/layer_$(LAYER)"
	$(PYTHON) -m llm_analyzer arch "$(MODEL)" --cache-dir "$(CACHE_DIR)" --level mlp --layer "$(LAYER)" --format onnx --out "$(ONNX_DIR)/layer_$(LAYER)/mlp.onnx"

moe-onnx:
	mkdir -p "$(ONNX_DIR)/layer_$(MOE_LAYER)"
	$(PYTHON) -m llm_analyzer arch "$(MODEL)" --cache-dir "$(CACHE_DIR)" --level moe --layer "$(MOE_LAYER)" --format onnx --out "$(ONNX_DIR)/layer_$(MOE_LAYER)/moe.onnx"

onnx-graphs: kernel-onnx moe-kernel-onnx attention-onnx mlp-onnx moe-onnx

arch: model-diagram layer-diagram json

all: test arch

clean-cache:
	rm -rf "$(CACHE_DIR)"
