from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import graph_from_oscillator_trace, graph_from_transformer_trace, graph_from_family, graph_from_source_file
from .compare import comparison_report, comparison_report_json
from .demo import demo_payload, demo_report, demo_report_json
from .diagnose import analyze_logic_file, diagnose_report, diagnose_report_json
from .forecast import forecast_report, forecast_report_json
from .intervention import apply_intervention
from .io import load_graph_from_json, load_training_trace_from_json
from .report import export_comparison_report, export_forecast_report, export_inspect_report, export_payload
from .schema import Intervention
from .space import analyze_source_file, space_report, space_report_json
from .temporal import analyze_time_series_file, temporal_report, temporal_report_json
from .temporal_balance import (
    BalanceOperator, BalanceSpec, StaticForcingSpec,
    analyze_temporal_balance_file,
    load_spec_from_file,
    temporal_balance_report, temporal_balance_report_json,
)
from .temporal_couple import analyze_temporal_couple_file, temporal_couple_report, temporal_couple_report_json
from .temporal_radio import analyze_temporal_radio_file, temporal_radio_report, temporal_radio_report_json


def _load_graph(path: str):
    return load_graph_from_json(path)


def _load_training_trace(path: str):
    return load_training_trace_from_json(path)


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

    forecast_p = sub.add_parser("forecast", help="predict final training outcome from an early training trace prefix")
    forecast_p.add_argument("trace")
    forecast_p.add_argument("--output", default="")
    forecast_p.add_argument("--json", action="store_true", help="emit machine-readable JSON output")

    temporal_p = sub.add_parser("temporal",
        help="detect CFL/dispersion instabilities in a (T, *spatial) tensor time series")
    temporal_p.add_argument("--input", required=True,
        help=".npz, .npy, or .json file containing a time series of shape (T, *spatial)")
    temporal_p.add_argument("--key", default="",
        help="array key inside .npz/.json (default: first array)")
    temporal_p.add_argument("--dt", type=float, default=None,
        help="physical timestep (default: read from file or 1.0)")
    temporal_p.add_argument("--growth", type=float, default=1.01,
        help="amplitude growth threshold per step (default 1.01 = 1%%)")
    temporal_p.add_argument("--phase-tol", type=float, default=0.15,
        help="phase-match tolerance in radians (default 0.15 ≈ 8.6°)")
    temporal_p.add_argument("--output", default="")
    temporal_p.add_argument("--json", action="store_true", help="emit machine-readable JSON output")

    tradio_p = sub.add_parser(
        "temporal-radio",
        help="scan vector-field anomalies and lock onto the strongest temporal channel",
    )
    tradio_p.add_argument("--input", required=True, help=".npz or .json file containing u/v and optional refs")
    tradio_p.add_argument("--dt", type=float, default=None, help="physical timestep (default: read from file)")
    tradio_p.add_argument("--case-id", default="", help="override case identifier")
    tradio_p.add_argument("--time-bins", type=int, default=8, help="number of coarse temporal bins")
    tradio_p.add_argument("--y-bins", type=int, default=12, help="number of coarse spatial bins in y")
    tradio_p.add_argument("--x-bins", type=int, default=16, help="number of coarse spatial bins in x")
    tradio_p.add_argument("--output", default="")
    tradio_p.add_argument("--json", action="store_true", help="emit machine-readable JSON output")

    tcouple_p = sub.add_parser(
        "temporal-couple",
        help="diagnose h->uv coupling (geostrophic balance) on a rollout; pair-specific follow-up to temporal-radio",
    )
    tcouple_p.add_argument("--input", required=True, help=".npz or .json file containing h/u/v and optional bg/obs refs")
    tcouple_p.add_argument("--dt", type=float, default=None, help="physical timestep (default: read from file)")
    tcouple_p.add_argument("--case-id", default="", help="override case identifier")
    tcouple_p.add_argument("--time-bins", type=int, default=8, help="number of coarse temporal bins")
    tcouple_p.add_argument("--y-bins", type=int, default=12, help="number of coarse spatial bins in y")
    tcouple_p.add_argument("--x-bins", type=int, default=16, help="number of coarse spatial bins in x")
    tcouple_p.add_argument("--output", default="")
    tcouple_p.add_argument("--json", action="store_true", help="emit machine-readable JSON output")

    tbal_p = sub.add_parser(
        "temporal-balance",
        help="generic potential/response/static-forcing balance diagnostic (not TD-specific)",
    )
    tbal_p.add_argument("--input", required=True, help=".npz or .json file with potential/response/static arrays")
    tbal_p.add_argument("--spec", default="", help="optional trbalance.v1 JSON spec; other CLI flags override")
    tbal_p.add_argument("--potential", default="h", help="array key for the potential scalar field")
    tbal_p.add_argument("--response-u", default="u", help="array key for response u-component")
    tbal_p.add_argument("--response-v", default="v", help="array key for response v-component")
    tbal_p.add_argument(
        "--static",
        action="append",
        default=[],
        metavar="KEY:WEIGHT",
        help="static forcing field and its weight, e.g. --static topo:0.18 (repeatable)",
    )
    tbal_p.add_argument("--operator", default="rotated_gradient",
                        choices=["gradient", "rotated_gradient"],
                        help="balance operator applied to each scalar field")
    tbal_p.add_argument("--operator-scale", type=float, default=1.0,
                        help="multiplicative scale on the operator output (e.g. g/f for geostrophic)")
    tbal_p.add_argument("--dx", type=float, default=1.0, help="grid spacing in x")
    tbal_p.add_argument("--dy", type=float, default=1.0, help="grid spacing in y")
    tbal_p.add_argument("--dt", type=float, default=1.0, help="physical timestep (informational)")
    tbal_p.add_argument("--case-id", default="")
    tbal_p.add_argument("--time-bins", type=int, default=8)
    tbal_p.add_argument("--y-bins", type=int, default=12)
    tbal_p.add_argument("--x-bins", type=int, default=16)
    tbal_p.add_argument("--output", default="")
    tbal_p.add_argument("--json", action="store_true", help="emit machine-readable JSON output")

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

    if args.cmd == "forecast":
        trace = _load_training_trace(args.trace)
        if args.verbose:
            print(f"[verbose] forecasting trace={args.trace} run_id={trace.run_id}")
        report = forecast_report_json(trace) if args.json else forecast_report(trace)
        print(report)
        if args.output:
            export_forecast_report(trace, args.output, as_json=args.json)
            if args.verbose:
                print(f"[verbose] wrote report={args.output}")
        return

    if args.cmd == "temporal":
        if args.verbose:
            print(f"[verbose] temporal input={args.input} key={args.key or '(auto)'} "
                  f"dt={args.dt} growth={args.growth} phase_tol={args.phase_tol}")
        report = analyze_time_series_file(
            args.input, key=args.key, dt=args.dt,
            growth_threshold=args.growth, phase_threshold_rad=args.phase_tol,
        )
        text = temporal_report_json(report) if args.json else temporal_report(report)
        print(text)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            if args.verbose:
                print(f"[verbose] wrote report={args.output}")
        return

    if args.cmd == "temporal-radio":
        if args.verbose:
            print(
                f"[verbose] temporal-radio input={args.input} dt={args.dt} case_id={args.case_id or '(auto)'} "
                f"time_bins={args.time_bins} y_bins={args.y_bins} x_bins={args.x_bins}"
            )
        report = analyze_temporal_radio_file(
            args.input,
            dt=args.dt,
            case_id=args.case_id,
            time_bins=args.time_bins,
            y_bins=args.y_bins,
            x_bins=args.x_bins,
        )
        text = temporal_radio_report_json(report) if args.json else temporal_radio_report(report)
        print(text)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            if args.verbose:
                print(f"[verbose] wrote report={args.output}")
        return

    if args.cmd == "temporal-couple":
        if args.verbose:
            print(
                f"[verbose] temporal-couple input={args.input} dt={args.dt} case_id={args.case_id or '(auto)'} "
                f"time_bins={args.time_bins} y_bins={args.y_bins} x_bins={args.x_bins}"
            )
        report = analyze_temporal_couple_file(
            args.input,
            dt=args.dt,
            case_id=args.case_id,
            time_bins=args.time_bins,
            y_bins=args.y_bins,
            x_bins=args.x_bins,
        )
        text = temporal_couple_report_json(report) if args.json else temporal_couple_report(report)
        print(text)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            if args.verbose:
                print(f"[verbose] wrote report={args.output}")
        return

    if args.cmd == "temporal-balance":
        # Compose spec: start from --spec file if given, then let explicit
        # CLI flags override. A bare --input run without --spec uses all
        # CLI defaults (h / u / v, rotated_gradient, unit scale).
        if args.spec:
            spec = load_spec_from_file(args.spec)
        else:
            spec = BalanceSpec()
        # CLI overrides applied (only when the user actually passed them —
        # but since argparse always sets defaults, we just accept the
        # CLI-provided values as the authoritative choice).
        spec.case_id       = args.case_id or spec.case_id
        spec.dt            = args.dt
        spec.dx            = args.dx
        spec.dy            = args.dy
        spec.potential_key = args.potential
        spec.response_u_key = args.response_u
        spec.response_v_key = args.response_v
        spec.operator = BalanceOperator(kind=args.operator, scale=args.operator_scale)
        spec.analysis.time_bins = args.time_bins
        spec.analysis.y_bins    = args.y_bins
        spec.analysis.x_bins    = args.x_bins
        if args.static:
            # `--static` flags replace any static list from --spec so the
            # CLI stays authoritative when both are given.
            spec.static_forcings = []
            for entry in args.static:
                if ":" in entry:
                    key, w = entry.split(":", 1)
                    spec.static_forcings.append(StaticForcingSpec(key=key, weight=float(w)))
                else:
                    spec.static_forcings.append(StaticForcingSpec(key=entry, weight=1.0))
        if args.verbose:
            print(
                f"[verbose] temporal-balance input={args.input} op={args.operator} "
                f"potential={spec.potential_key} statics="
                f"{[(s.key, s.weight) for s in spec.static_forcings]}"
            )
        report = analyze_temporal_balance_file(args.input, spec)
        text = temporal_balance_report_json(report) if args.json else temporal_balance_report(report)
        print(text)
        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
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

  forecast  Predict whether training outcome is already visible from an early prefix
            tensorearch forecast training_trace.json [--json]

  temporal  Detect CFL/dispersion instabilities (2Δt checkerboard, growth) in a
            numerical-simulation (T, *spatial) tensor via symmetric cyclic numbers
            tensorearch temporal --input u.npz [--key u] [--dt 60] [--json]

  temporal-radio
            Scan vector-field anomalies, lock onto the strongest channel, and emit
            reversible gate coordinates for follow-up
            tensorearch temporal-radio --input rollout_uv.npz [--json]

  temporal-couple
            Targeted h->uv coupling diagnosis (geostrophic balance). Use after
            temporal-radio flags a wind-direction anomaly to attribute it to
            h-gradient decoupling vs anti-geostrophic flow vs weak coupling.
            tensorearch temporal-couple --input rollout_huv.npz [--json]

  temporal-balance
            Generic potential/response/static-forcing balance diagnostic. Given
            a potential scalar, a response vector field, and optional static
            forcings, compares three theoretical balance modes (potential_only,
            static_only, combined) and flags when adding a static term makes
            the response LESS consistent (static_forcing_overrides_potential_balance).
            tensorearch temporal-balance --input probe.npz \\
                --potential h --response-u u --response-v v \\
                --static topo:0.18 --operator rotated_gradient --json

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

  # Predict whether a run can be stopped early
  tensorearch forecast training_trace.json --json
""")
