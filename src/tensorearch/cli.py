from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import graph_from_oscillator_trace, graph_from_transformer_trace, graph_from_family, graph_from_source_file
from .compare import comparison_report, comparison_report_json
from .demo import demo_payload, demo_report, demo_report_json
from .diagnose import analyze_logic_file, diagnose_report, diagnose_report_json
from .intervention import apply_intervention
from .io import load_graph_from_json
from .report import export_comparison_report, export_inspect_report, export_payload
from .schema import Intervention
from .space import analyze_source_file, space_report, space_report_json


def _load_graph(path: str):
    return load_graph_from_json(path)


def _build_from_adapter(adapter: str, payload: dict, family: str = "", input_path: str = ""):
    if adapter == "transformer":
        return graph_from_transformer_trace(payload)
    if adapter == "oscillator":
        return graph_from_oscillator_trace(payload)
    if adapter == "family":
        if not family:
            raise ValueError("--family is required for --adapter family")
        return graph_from_family(payload, family)
    if adapter == "source":
        return graph_from_source_file(input_path)
    raise ValueError(f"unknown adapter: {adapter}")


def _load_payload(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tensorearch",
        description="Architecture inspection toolkit for bottlenecks, compliance, and intelligence metrics",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print extra execution details")
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect_p = sub.add_parser("inspect", help="inspect a single trace and print a report")
    inspect_p.add_argument("trace")
    inspect_p.add_argument("--output", default="")
    inspect_p.add_argument("--json", action="store_true", help="emit machine-readable JSON output")

    compare_p = sub.add_parser("compare", help="compare two traces and print a delta report")
    compare_p.add_argument("left")
    compare_p.add_argument("right")
    compare_p.add_argument("--output", default="")
    compare_p.add_argument("--json", action="store_true", help="emit machine-readable JSON output")

    ablate_p = sub.add_parser("ablate", help="apply one intervention and compare before/after")
    ablate_p.add_argument("trace")
    ablate_p.add_argument("--kind", required=True)
    ablate_p.add_argument("--target", required=True)
    ablate_p.add_argument("--value", type=float, default=0.0)
    ablate_p.add_argument("--json", action="store_true", help="emit machine-readable JSON output")

    export_p = sub.add_parser("export", help="export inspect or compare results to a file")
    export_p.add_argument("--mode", choices=["inspect", "compare"], required=True)
    export_p.add_argument("--left", required=True, help="trace path for inspect, or left trace for compare")
    export_p.add_argument("--right", default="", help="right trace for compare mode")
    export_p.add_argument("--output", required=True)
    export_p.add_argument("--json", action="store_true", help="write machine-readable JSON output")

    adapt_p = sub.add_parser("adapt", help="convert a high-level architecture payload into a trace JSON")
    adapt_p.add_argument("--adapter", choices=["transformer", "oscillator", "family", "source"], required=True)
    adapt_p.add_argument("--input", required=True, help="adapter payload JSON or source file (for --adapter source)")
    adapt_p.add_argument("--output", required=True, help="output trace JSON path")
    adapt_p.add_argument("--family", default="", help="family name (for --adapter family)")

    space_p = sub.add_parser("space", help="analyze a source file and project it into the quadrupole space")
    space_p.add_argument("--source-file", required=True)
    space_p.add_argument("--output", default="")
    space_p.add_argument("--json", action="store_true", help="emit machine-readable JSON output")

    diagnose_p = sub.add_parser("diagnose", help="diagnose source-level logic smells in Python or shell scripts")
    diagnose_p.add_argument("--source-file", required=True)
    diagnose_p.add_argument("--output", default="")
    diagnose_p.add_argument("--json", action="store_true", help="emit machine-readable JSON output")

    sub.add_parser("help", help="show detailed usage guide")

    args = parser.parse_args()

    if args.cmd == "inspect":
        graph = _load_graph(args.trace)
        if args.verbose:
            print(f"[verbose] inspecting trace={args.trace}")
        report = demo_report_json(graph) if args.json else demo_report(graph)
        print(report)
        if args.output:
            export_inspect_report(graph, args.output, as_json=args.json)
            if args.verbose:
                print(f"[verbose] wrote report={args.output}")
        return

    if args.cmd == "compare":
        left = _load_graph(args.left)
        right = _load_graph(args.right)
        if args.verbose:
            print(f"[verbose] comparing left={args.left} right={args.right}")
        report = comparison_report_json(left, right) if args.json else comparison_report(left, right)
        print(report)
        if args.output:
            export_comparison_report(left, right, args.output, as_json=args.json)
            if args.verbose:
                print(f"[verbose] wrote report={args.output}")
        return

    if args.cmd == "ablate":
        graph = _load_graph(args.trace)
        if args.verbose:
            print(f"[verbose] ablating trace={args.trace} kind={args.kind} target={args.target} value={args.value}")
        altered = apply_intervention(
            graph,
            Intervention(kind=args.kind, target=args.target, value=args.value),
        )
        print(comparison_report_json(graph, altered) if args.json else comparison_report(graph, altered))
        return

    if args.cmd == "export":
        if args.mode == "inspect":
            graph = _load_graph(args.left)
            export_inspect_report(graph, args.output, as_json=args.json)
            if args.verbose:
                print(f"[verbose] exported inspect report={args.output}")
            return
        left = _load_graph(args.left)
        right = _load_graph(args.right)
        export_comparison_report(left, right, args.output, as_json=args.json)
        if args.verbose:
            print(f"[verbose] exported comparison report={args.output}")
        return

    if args.cmd == "adapt":
        if args.adapter == "source":
            payload = {}
            graph = _build_from_adapter(args.adapter, payload, input_path=args.input)
        else:
            payload = _load_payload(args.input)
            graph = _build_from_adapter(args.adapter, payload, family=args.family)
        export_payload(demo_payload(graph), args.output)
        if args.verbose:
            print(f"[verbose] adapted adapter={args.adapter} input={args.input} output={args.output}")
        return

    if args.cmd == "space":
        payload = analyze_source_file(args.source_file)
        if args.verbose:
            print(f"[verbose] source_file={args.source_file}")
        report = space_report_json(payload) if args.json else space_report(payload)
        print(report)
        if args.output:
            Path(args.output).write_text(report, encoding="utf-8")
            if args.verbose:
                print(f"[verbose] wrote report={args.output}")
        return

    if args.cmd == "diagnose":
        payload = analyze_logic_file(args.source_file)
        if args.verbose:
            print(f"[verbose] source_file={args.source_file}")
        report = diagnose_report_json(payload) if args.json else diagnose_report(payload)
        print(report)
        if args.output:
            Path(args.output).write_text(report, encoding="utf-8")
            if args.verbose:
                print(f"[verbose] wrote report={args.output}")
        return

    if args.cmd == "help":
        _print_help()
        return


def _print_help() -> None:
    print("""Tensorearch — Architecture Inspection Toolkit

COMMANDS
  inspect   Inspect a model trace and report bottlenecks
            tensorearch inspect trace.json [--json] [--output out.txt]

  compare   Compare two traces side by side
            tensorearch compare left.json right.json [--json]

  ablate    Simulate an intervention and show before/after delta
            tensorearch ablate trace.json --kind <kind> --target <slice> [--value N] [--json]
            kinds: remove_slice, mask_edge, scale_edge_bandwidth,
                   set_write_magnitude, set_read_sensitivity, set_doi_alignment, swap_topology

  adapt     Convert source code or architecture description into a trace
            tensorearch adapt --adapter source --input model.py --output trace.json
            tensorearch adapt --adapter transformer --input desc.json --output trace.json
            tensorearch adapt --adapter oscillator --input desc.json --output trace.json
            tensorearch adapt --adapter family --family diffusion_unet --input desc.json --output trace.json

  space     Classify source code into one of 15 model families
            tensorearch space --source-file model.py [--json]

  diagnose  Audit source-level logic: entropy clusters, modular flow, mutation tracking
            tensorearch diagnose --source-file script.py [--json]

  export    Write inspect or compare results to a file
            tensorearch export --mode inspect --left trace.json --output out.json [--json]
            tensorearch export --mode compare --left a.json --right b.json --output out.json [--json]

MODEL FAMILIES (space classification)
  baseline residual          Conv/Dense networks (e.g. MNIST, ResNet)
  latent-attention dominant  Transformers (e.g. GPT, BERT, LLaMA)
  diffusion-unet dominant    Diffusion models (e.g. Stable Diffusion)
  adapterization dominant    LoRA, PEFT, hypernetworks
  runtime-wrapper dominant   Triton kernels, quantization wrappers
  video-temporal dominant    Video generation (e.g. AnimateDiff, UNet3D)
  audio-spectral dominant    Audio/music (e.g. mel spectrograms, vocoders)
  3d-generative dominant     3D (e.g. NeRF, Gaussian splatting)
  speech-language dominant   ASR/TTS (e.g. Whisper, GPT-SoVITS)
  world-model dominant       Environment simulators (e.g. MDRNN, GameGAN)
  multimodal-alignment       VLMs (e.g. BLIP-2, Q-Former)
  graph-message-passing      GNNs (e.g. GCN, GAT, PyG)
  vision-detection dominant  Object detection (e.g. Detectron2, RCNN)
  bio-sequence dominant      Protein/bio (e.g. ESM, Evoformer)
  propagation dominant       Phase/oscillation-based models

GLOBAL FLAGS
  -v, --verbose   Print extra execution details
  --json          Emit machine-readable JSON output
  --output PATH   Write output to file

EXAMPLES
  # Classify a model source file
  tensorearch space --source-file model.py --json

  # Full pipeline: source → trace → inspect → ablate
  tensorearch adapt --adapter source --input model.py --output trace.json
  tensorearch inspect trace.json
  tensorearch ablate trace.json --kind remove_slice --target blk0.conv

  # Compare two architectures
  tensorearch compare transformer.json oscillator.json --json

  # Diagnose scoring logic
  tensorearch diagnose --source-file scorer.py --json
""")
