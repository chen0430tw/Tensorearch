from __future__ import annotations

import json
from pathlib import Path

from .compare import comparison_report, comparison_report_json
from .demo import demo_report, demo_report_json
from .forecast import forecast_report, forecast_report_json
from .graph import ArchitectureGraph
from .schema import TrainingTrace


def export_inspect_report(graph: ArchitectureGraph, output_path: str | Path, as_json: bool = False) -> Path:
    path = Path(output_path)
    path.write_text(demo_report_json(graph) if as_json else demo_report(graph), encoding="utf-8")
    return path


def export_comparison_report(left: ArchitectureGraph, right: ArchitectureGraph, output_path: str | Path, as_json: bool = False) -> Path:
    path = Path(output_path)
    path.write_text(comparison_report_json(left, right) if as_json else comparison_report(left, right), encoding="utf-8")
    return path


def export_forecast_report(trace: TrainingTrace, output_path: str | Path, as_json: bool = False) -> Path:
    path = Path(output_path)
    path.write_text(forecast_report_json(trace) if as_json else forecast_report(trace), encoding="utf-8")
    return path


def export_payload(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
