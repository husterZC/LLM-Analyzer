import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .extract import extract_from_snapshot
from .hf import HuggingFaceError, HuggingFaceMetadataClient, load_token
from .onnx_export import OnnxExportError, export_onnx_kernel_graph
from .render import (
    render_json,
    render_mermaid_attention,
    render_mermaid_layer,
    render_mermaid_mlp,
    render_mermaid_model,
    render_mermaid_moe,
    render_summary,
)


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help(sys.stderr)
        return 2

    try:
        if args.command == "fetch":
            return _cmd_fetch(args)
        if args.command == "inspect":
            return _cmd_inspect(args)
        if args.command == "arch":
            return _cmd_arch(args)
        if args.command == "batch":
            return _cmd_batch(args)
        parser.error("unknown command %s" % args.command)
        return 2
    except HuggingFaceError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    except OnnxExportError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    except Exception as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-analyzer",
        description="Metadata-only tools for inspecting Hugging Face LLM architectures.",
    )
    subparsers = parser.add_subparsers(dest="command")

    fetch = subparsers.add_parser("fetch", help="Download metadata/config/code files, excluding weights.")
    _add_hf_args(fetch)

    inspect = subparsers.add_parser("inspect", help="Fetch metadata and print a compact architecture summary.")
    _add_hf_args(inspect)

    arch = subparsers.add_parser("arch", help="Fetch metadata and render an architecture diagram or JSON IR.")
    _add_hf_args(arch)
    arch.add_argument("--level", choices=("model", "layer", "attention", "mlp", "moe"), default="model")
    arch.add_argument("--layer", type=int, default=0, help="Decoder layer index for layer/detail diagrams.")
    arch.add_argument("--format", choices=("mermaid", "json", "summary", "onnx"), default="mermaid")
    arch.add_argument("--out", help="Output file. Prints to stdout when omitted.")

    batch = subparsers.add_parser("batch", help="Analyze every model in a pipe-delimited model list.")
    batch.add_argument("model_list", help="Model list file: model|layers|attention_layers|mlp_layers|moe_layers")
    batch.add_argument("--out-dir", default="outputs", help="Output directory root, default: outputs")
    _add_hf_options(batch)

    return parser


def _add_hf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Hugging Face model id or URL.")
    _add_hf_options(parser)


def _add_hf_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default=".llm_analyzer_cache")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Prefer env vars or .hf_token.txt.")
    parser.add_argument("--token-file", default=".hf_token.txt", help="Token file path, default: .hf_token.txt")
    parser.add_argument("--max-file-mb", type=float, default=50.0, help="Maximum metadata file size to download.")


def _cmd_fetch(args) -> int:
    result = _fetch(args)
    print("Snapshot: %s" % result.snapshot_dir)
    print("Metadata files downloaded: %d" % len(result.downloaded_files))
    print("Weight files skipped: %d" % len(result.skipped_weight_files))
    if result.skipped_large_files:
        print("Large metadata files skipped: %d" % len(result.skipped_large_files))
    return 0


def _cmd_inspect(args) -> int:
    result = _fetch(args)
    architecture = extract_from_snapshot(
        snapshot_dir=result.snapshot_dir,
        model_id=result.model_id,
        revision=result.revision,
        files=result.downloaded_files,
        skipped_weight_files=result.skipped_weight_files,
    )
    print(render_summary(architecture))
    return 0


def _cmd_arch(args) -> int:
    result = _fetch(args)
    architecture = extract_from_snapshot(
        snapshot_dir=result.snapshot_dir,
        model_id=result.model_id,
        revision=result.revision,
        files=result.downloaded_files,
        skipped_weight_files=result.skipped_weight_files,
    )

    if args.format == "json":
        output = render_json(architecture)
    elif args.format == "onnx":
        if not args.out:
            raise ValueError("--format onnx requires --out path/to/file.onnx")
        path = Path(args.out)
        export_onnx_kernel_graph(
            architecture=architecture,
            output_path=path,
            level=args.level,
            layer_index=args.layer,
        )
        print("Wrote %s" % path)
        return 0
    elif args.format == "summary":
        output = render_summary(architecture)
    elif args.level == "layer":
        output = render_mermaid_layer(architecture, args.layer)
    elif args.level == "attention":
        output = render_mermaid_attention(architecture, args.layer)
    elif args.level == "mlp":
        output = render_mermaid_mlp(architecture, args.layer)
    elif args.level == "moe":
        output = render_mermaid_moe(architecture, args.layer)
    else:
        output = render_mermaid_model(architecture)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
        print("Wrote %s" % path)
    else:
        print(output)
    return 0


@dataclass
class BatchPlan:
    model: str
    layers: List[int]
    attention_layers: List[int]
    mlp_layers: List[int]
    moe_layers: List[int]


def _cmd_batch(args) -> int:
    plans = _read_model_list(Path(args.model_list))
    if not plans:
        raise ValueError("model list has no model entries: %s" % args.model_list)

    for plan in plans:
        print("Analyzing %s" % plan.model)
        fetch_args = argparse.Namespace(
            model=plan.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            hf_token=args.hf_token,
            token_file=args.token_file,
            max_file_mb=args.max_file_mb,
        )
        result = _fetch(fetch_args)
        architecture = extract_from_snapshot(
            snapshot_dir=result.snapshot_dir,
            model_id=result.model_id,
            revision=result.revision,
            files=result.downloaded_files,
            skipped_weight_files=result.skipped_weight_files,
        )

        output_root = Path(args.out_dir) / _model_slug(result.model_id)
        written = 0
        written += _write_text(output_root / "overview" / "model.mmd", render_mermaid_model(architecture))
        written += _write_text(output_root / "ir" / "architecture.json", render_json(architecture))

        for layer_index in plan.layers:
            written += _write_text(
                output_root / "layers" / ("layer_%d" % layer_index) / "block.mmd",
                render_mermaid_layer(architecture, layer_index),
            )
            written += _write_onnx(
                architecture,
                output_root / "onnx" / ("layer_%d" % layer_index) / "kernels.onnx",
                level="layer",
                layer_index=layer_index,
            )

        for layer_index in plan.attention_layers:
            written += _write_text(
                output_root / "details" / ("layer_%d" % layer_index) / "attention.mmd",
                render_mermaid_attention(architecture, layer_index),
            )
            written += _write_onnx(
                architecture,
                output_root / "onnx" / ("layer_%d" % layer_index) / "attention.onnx",
                level="attention",
                layer_index=layer_index,
            )

        for layer_index in plan.mlp_layers:
            written += _write_text(
                output_root / "details" / ("layer_%d" % layer_index) / "mlp.mmd",
                render_mermaid_mlp(architecture, layer_index),
            )
            written += _write_onnx(
                architecture,
                output_root / "onnx" / ("layer_%d" % layer_index) / "mlp.onnx",
                level="mlp",
                layer_index=layer_index,
            )

        for layer_index in plan.moe_layers:
            written += _write_text(
                output_root / "details" / ("layer_%d" % layer_index) / "moe.mmd",
                render_mermaid_moe(architecture, layer_index),
            )
            written += _write_onnx(
                architecture,
                output_root / "onnx" / ("layer_%d" % layer_index) / "moe.onnx",
                level="moe",
                layer_index=layer_index,
            )

        print("  Wrote %d files under %s" % (written, output_root))
    return 0


def _write_text(path: Path, text: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return 1


def _write_onnx(architecture, path: Path, level: str, layer_index: int) -> int:
    export_onnx_kernel_graph(
        architecture=architecture,
        output_path=path,
        level=level,
        layer_index=layer_index,
    )
    return 1


def _read_model_list(path: Path) -> List[BatchPlan]:
    if not path.exists():
        raise ValueError("model list does not exist: %s" % path)

    plans = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 5:
            raise ValueError(
                "%s:%d expected 5 pipe-delimited fields: "
                "model|layers|attention_layers|mlp_layers|moe_layers" % (path, line_number)
            )

        model, layers, attention_layers, mlp_layers, moe_layers = fields
        if not model:
            raise ValueError("%s:%d model field is empty" % (path, line_number))
        plans.append(
            BatchPlan(
                model=model,
                layers=_parse_layer_list(layers, path, line_number, "layers"),
                attention_layers=_parse_layer_list(attention_layers, path, line_number, "attention_layers"),
                mlp_layers=_parse_layer_list(mlp_layers, path, line_number, "mlp_layers"),
                moe_layers=_parse_layer_list(moe_layers, path, line_number, "moe_layers"),
            )
        )
    return plans


def _parse_layer_list(value: str, path: Path, line_number: int, field_name: str) -> List[int]:
    layers = []
    seen = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            layer = int(item)
        except ValueError as exc:
            raise ValueError("%s:%d invalid %s layer: %s" % (path, line_number, field_name, item)) from exc
        if layer < 0:
            raise ValueError("%s:%d %s layer must be non-negative: %d" % (path, line_number, field_name, layer))
        if layer not in seen:
            seen.add(layer)
            layers.append(layer)
    return layers


def _model_slug(model_id: str) -> str:
    return _normalize_model_id(model_id).replace("/", "_")


def _fetch(args):
    model_id = _normalize_model_id(args.model)
    token = load_token(explicit_token=args.hf_token, token_file=args.token_file)
    client = HuggingFaceMetadataClient(token=token, cache_dir=args.cache_dir)
    return client.fetch_metadata(
        model_id=model_id,
        revision=args.revision,
        max_file_mb=args.max_file_mb,
    )


def _normalize_model_id(value: str) -> str:
    prefix = "https://huggingface.co/"
    if value.startswith(prefix):
        value = value[len(prefix):]
    return value.strip().strip("/")


if __name__ == "__main__":
    raise SystemExit(main())
