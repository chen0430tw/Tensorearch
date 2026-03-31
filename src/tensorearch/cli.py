from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import graph_from_oscillator_trace, graph_from_transformer_trace
from .compare import comparison_report, comparison_report_json
from .demo import demo_payload, demo_report, demo_report_json
from .intervention import apply_intervention
from .io import load_graph_from_json
from .report import export_comparison_report, export_inspect_report, export_payload
from .schema import Intervention


def _load_graph(path: str):
    return load_graph_from_json(path)


def _build_from_adapter(adapter: str, payload: dict):
    if adapter == "transformer":
        return graph_from_transformer_trace(payload)
    if adapter == "oscillator":
        return graph_from_oscillator_trace(payload)
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
    adapt_p.add_argument("--adapter", choices=["transformer", "oscillator"], required=True)
    adapt_p.add_argument("--input", required=True, help="high-level adapter payload JSON")
    adapt_p.add_argument("--output", required=True, help="output trace JSON path")

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
        payload = _load_payload(args.input)
        graph = _build_from_adapter(args.adapter, payload)
        export_payload(demo_payload(graph), args.output)
        if args.verbose:
            print(f"[verbose] adapted adapter={args.adapter} input={args.input} output={args.output}")
        return
