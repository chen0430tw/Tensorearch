from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiagnosticItem:
    severity: str
    kind: str
    message: str
    line: int = 0
    symbol: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "severity": self.severity,
            "kind": self.kind,
            "message": self.message,
            "line": self.line,
            "symbol": self.symbol,
        }


def _load_text(path: str | Path) -> str:
    source = Path(path)
    raw = source.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _shannon_entropy(counts: dict[str, int]) -> float:
    total = sum(v for v in counts.values() if v > 0)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        if value <= 0:
            continue
        p = value / total
        entropy -= p * math.log2(p)
    return entropy


def _entropy_bucket(entropy: float) -> str:
    if entropy < 1.0:
        return "low_entropy"
    if entropy < 1.8:
        return "medium_entropy"
    return "high_entropy"


def _dominant_signal(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _logic_labels(name: str, text: str, counts: dict[str, int], language: str) -> list[str]:
    """Assign logic labels based on function/script name and content.

    For Python function-level scopes, match against the function name first
    (high confidence), then fall back to body-text matching only when the
    scope is a single function (not module-level).  Module-level labels are
    derived from name only, preventing the "everything matches" problem
    caused by full-file text search.
    """
    labels: list[str] = []
    name_l = name.lower()
    is_module = name_l in ("<module>", "<script>")

    # For module scope, only match on the file/module name to avoid over-labeling.
    # For function scope, match on both name and body text.
    search_text = "" if is_module else text.lower()
    search_target = name_l + " " + search_text

    if language == "python":
        _SCORING_TOKENS = ("score", "scoring", "attention", "weight", "metric")
        _RANKING_TOKENS = ("rank", "sort", "top_k", "top-k", "priority", "winner")
        _GATING_TOKENS = ("gate", "gated", "penalty", "cap", "clamp", "threshold")
        _PIPELINE_TOKENS = ("pipeline", "stage", "runner", "dispatch", "submit", "bridge")

        if any(token in search_target for token in _SCORING_TOKENS):
            labels.append("scoring_logic")
        if any(token in search_target for token in _RANKING_TOKENS):
            labels.append("ranking_logic")
        if any(token in search_target for token in _GATING_TOKENS):
            labels.append("gating_logic")
        if any(token in search_target for token in _PIPELINE_TOKENS):
            labels.append("pipeline_logic")
    else:
        if counts.get("pipeline", 0) > 0:
            labels.append("pipeline_logic")
        if counts.get("conditional", 0) >= 2 and counts.get("assign", 0) >= 2:
            labels.append("gating_logic")
        if re.search(r"\bsort\b|\brank\b|\btop\b", search_target):
            labels.append("ranking_logic")
        if re.search(r"\bscore\b|\bweight\b|\bmetric\b", search_target):
            labels.append("scoring_logic")

    if not labels:
        labels.append("general_logic")
    return labels


class _PythonLogicVisitor(ast.NodeVisitor):
    def __init__(self, text: str):
        self.text = text
        self.findings: list[DiagnosticItem] = []
        self.strengths: list[DiagnosticItem] = []
        self._score_mutations: dict[str, list[tuple[str, int]]] = {}
        self._function_entropy_clusters: list[dict[str, object]] = []
        self._module_counts = self._empty_counts()

    @staticmethod
    def _empty_counts() -> dict[str, int]:
        return {
            "assign": 0,
            "aug_assign": 0,
            "call": 0,
            "if": 0,
            "loop": 0,
            "return": 0,
            "compare": 0,
            "boolop": 0,
        }

    def visit_Assign(self, node: ast.Assign) -> None:
        self._module_counts["assign"] += 1
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target == "attention_map":
                self._inspect_attention_map(node)
            if isinstance(node.value, ast.Call):
                fn = self._call_name(node.value.func)
                if fn in {"min", "max"}:
                    self.strengths.append(
                        DiagnosticItem(
                            severity="info",
                            kind="bounded_assignment",
                            message=f"'{target}' uses explicit bounding via {fn}().",
                            line=node.lineno,
                            symbol=target,
                        )
                    )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._module_counts["aug_assign"] += 1
        if isinstance(node.target, ast.Name):
            target = node.target.id
            op = type(node.op).__name__
            self._score_mutations.setdefault(target, []).append((op, node.lineno))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._module_counts["call"] += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._module_counts["if"] += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._module_counts["loop"] += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._module_counts["loop"] += 1
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self._module_counts["return"] += 1
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        self._module_counts["compare"] += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self._module_counts["boolop"] += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        text = ast.get_source_segment(self.text, node) or ""
        has_softmax = "softmax" in text
        has_bounding = "min(" in text or "max(" in text or "clamp" in text
        has_score_return = bool(re.search(r"return\s+.*score|return\s+.*result|return\s+.*softmax", text))
        if has_bounding and (has_softmax or has_score_return):
            self.strengths.append(
                DiagnosticItem(
                    severity="info",
                    kind="score_normalization",
                    message=f"Function '{node.name}' combines normalization with explicit score bounds.",
                    line=node.lineno,
                    symbol=node.name,
                )
            )
        self._function_entropy_clusters.append(self._build_function_cluster(node))
        self.generic_visit(node)

    def finalize(self) -> None:
        for symbol, ops in self._score_mutations.items():
            mutating_lines = [line for _, line in ops]
            op_names = {op for op, _ in ops}
            _SCORE_NAMES = {"score", "s", "result", "out", "output", "total", "weighted", "raw_score", "final_score"}
            is_score_var = symbol in _SCORE_NAMES or "score" in symbol.lower()
            if is_score_var and len(mutating_lines) >= 4:
                self.findings.append(
                    DiagnosticItem(
                        severity="warning",
                        kind="multi_stage_mutation",
                        message=(
                            f"'{symbol}' is mutated across {len(mutating_lines)} stages. "
                            "Layered gates can make final rankings hard to reason about."
                        ),
                        line=mutating_lines[0],
                        symbol=symbol,
                    )
                )
            if is_score_var and {"Add", "Mult"} <= op_names:
                self.findings.append(
                    DiagnosticItem(
                        severity="warning",
                        kind="mixed_additive_multiplicative_logic",
                        message=(
                            f"'{symbol}' mixes additive and multiplicative updates. "
                            "This often hides which rule actually dominates the outcome."
                        ),
                        line=mutating_lines[0],
                        symbol=symbol,
                    )
                )

        # Check for fallback reconstruction: "fallback" must appear within
        # 5 lines of a downstream output reference to reduce false positives.
        fb_pattern = re.compile(r"fallback", re.IGNORECASE)
        downstream_pattern = re.compile(r"C_end|dtheta_end|elapsed")
        text_lines = self.text.splitlines()
        fb_lines = [i for i, ln in enumerate(text_lines) if fb_pattern.search(ln)]
        ds_lines = [i for i, ln in enumerate(text_lines) if downstream_pattern.search(ln)]
        if fb_lines and ds_lines and any(
            abs(fb - ds) <= 5 for fb in fb_lines for ds in ds_lines
        ):
            self.findings.append(
                DiagnosticItem(
                    severity="warning",
                    kind="derived_parameter_fallback",
                    message=(
                        "The file contains fallback reconstruction from downstream outputs "
                        "(for example C_end / dtheta_end / elapsed). Reconstructed inputs can distort intent."
                    ),
                    line=fb_lines[0] + 1,
                    symbol="fallback",
                )
            )

    def entropy_clusters(self) -> list[dict[str, object]]:
        module_entropy = _shannon_entropy(self._module_counts)
        return [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(module_entropy),
                "entropy": round(module_entropy, 4),
                "dominant_signal": _dominant_signal(self._module_counts),
                "logic_labels": _logic_labels("<module>", self.text, self._module_counts, "python"),
                "counts": dict(self._module_counts),
            },
            *self._function_entropy_clusters,
        ]

    def _inspect_attention_map(self, node: ast.Assign) -> None:
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            return
        by_td: dict[str, list[tuple[str, float | None, int]]] = {}
        for elt in node.value.elts:
            if not isinstance(elt, ast.Tuple) or len(elt.elts) < 4:
                continue
            td = self._const_str(elt.elts[0])
            cand = self._const_str(elt.elts[1])
            weight = self._const_num(elt.elts[2])
            mode = self._const_str(elt.elts[3])
            if td and cand and mode:
                by_td.setdefault(td, []).append((mode, weight, elt.lineno))

        for td, rules in by_td.items():
            modes = {mode for mode, _, _ in rules}
            hi_weights = [weight for _, weight, _ in rules if weight is not None and weight >= 2.5]
            if "align" in modes and "oppose" in modes:
                self.findings.append(
                    DiagnosticItem(
                        severity="warning",
                        kind="conflicting_signal",
                        message=(
                            f"Feature '{td}' drives both align and oppose rules. "
                            "Without an explicit priority rule, the scoring logic may self-cancel or oscillate."
                        ),
                        line=rules[0][2],
                        symbol=td,
                    )
                )
            if len(hi_weights) >= 2:
                self.findings.append(
                    DiagnosticItem(
                        severity="warning",
                        kind="high_weight_cluster",
                        message=(
                            f"Feature '{td}' has multiple high-weight rules (>= 2.5). "
                            "A single input may dominate the whole ranking."
                        ),
                        line=rules[0][2],
                        symbol=td,
                    )
                )

        if by_td:
            self.strengths.append(
                DiagnosticItem(
                    severity="info",
                    kind="explicit_mapping",
                    message="The file exposes an explicit feature-to-feature mapping table, which helps auditability.",
                    line=node.lineno,
                    symbol="attention_map",
                )
            )

    def _build_function_cluster(self, node: ast.FunctionDef) -> dict[str, object]:
        counts = self._empty_counts()
        text = ast.get_source_segment(self.text, node) or ""
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                counts["assign"] += 1
            elif isinstance(child, ast.AugAssign):
                counts["aug_assign"] += 1
            elif isinstance(child, ast.Call):
                counts["call"] += 1
            elif isinstance(child, ast.If):
                counts["if"] += 1
            elif isinstance(child, (ast.For, ast.While)):
                counts["loop"] += 1
            elif isinstance(child, ast.Return):
                counts["return"] += 1
            elif isinstance(child, ast.Compare):
                counts["compare"] += 1
            elif isinstance(child, ast.BoolOp):
                counts["boolop"] += 1
        entropy = _shannon_entropy(counts)
        return {
            "scope": "function",
            "name": node.name,
            "cluster": _entropy_bucket(entropy),
            "entropy": round(entropy, 4),
            "dominant_signal": _dominant_signal(counts),
            "logic_labels": _logic_labels(node.name, text, counts, "python"),
            "counts": counts,
            "line": node.lineno,
        }

    @staticmethod
    def _const_str(node: ast.AST) -> str | None:
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    @staticmethod
    def _const_num(node: ast.AST) -> float | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        return None

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""


def _diagnose_python(path: str | Path, text: str) -> dict[str, object]:
    tree = ast.parse(text, filename=str(path))
    visitor = _PythonLogicVisitor(text)
    visitor.visit(tree)
    visitor.finalize()
    return {
        "language": "python",
        "source_file": str(path),
        "findings": [item.to_dict() for item in visitor.findings],
        "strengths": [item.to_dict() for item in visitor.strengths],
        "entropy_clusters": visitor.entropy_clusters(),
    }


def _diagnose_shell(path: str | Path, text: str) -> dict[str, object]:
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()

    assignment_lines: dict[str, list[int]] = {}
    assign_re = re.compile(r"^\s*(?:\$)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
    pipe_re = re.compile(r"\|")
    danger_re = re.compile(r"\brm\b|\bdel\b|\bRemove-Item\b", re.IGNORECASE)
    counts = {
        "assign": 0,
        "pipeline": 0,
        "conditional": 0,
        "destructive": 0,
        "command": 0,
    }

    for idx, line in enumerate(lines, start=1):
        if line.strip():
            counts["command"] += 1
        m = assign_re.search(line)
        if m:
            assignment_lines.setdefault(m.group(1), []).append(idx)
            counts["assign"] += 1
        if pipe_re.search(line):
            counts["pipeline"] += 1
        if "if " in line or line.strip().startswith("if"):
            counts["conditional"] += 1
        if danger_re.search(line):
            counts["destructive"] += 1
            findings.append(
                DiagnosticItem(
                    severity="warning",
                    kind="destructive_command",
                    message="The script contains a destructive command. Validation of targets should be explicit.",
                    line=idx,
                    symbol="delete",
                )
            )

    # Emit one strength per kind (not per line) to avoid flooding
    if counts["pipeline"] > 0:
        strengths.append(
            DiagnosticItem(
                severity="info",
                kind="command_pipeline",
                message=f"The script uses {counts['pipeline']} explicit command pipeline(s), which makes dataflow visible.",
                line=0,
                symbol="pipeline",
            )
        )
    if counts["conditional"] > 0:
        strengths.append(
            DiagnosticItem(
                severity="info",
                kind="conditional_branch",
                message=f"The script contains {counts['conditional']} explicit branch(es) rather than hidden control flow.",
                line=0,
                symbol="if",
            )
        )

    for name, hit_lines in assignment_lines.items():
        if len(hit_lines) >= 3:
            findings.append(
                DiagnosticItem(
                    severity="warning",
                    kind="repeated_overwrite",
                    message=(
                        f"Variable '{name}' is overwritten {len(hit_lines)} times. "
                        "Repeated reassignment can hide the effective computation path."
                    ),
                    line=hit_lines[0],
                    symbol=name,
                )
            )

    entropy = _shannon_entropy(counts)
    return {
        "language": "shell",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "script",
                "name": "<script>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(counts),
                "logic_labels": _logic_labels("<script>", text, counts, "shell"),
                "counts": counts,
                "line": 0,
            }
        ],
    }


def analyze_logic_file(path: str | Path) -> dict[str, object]:
    source = Path(path)
    text = _load_text(source)
    suffix = source.suffix.lower()

    if suffix == ".py":
        payload = _diagnose_python(source, text)
    elif suffix in {".sh", ".bash", ".zsh", ".ps1", ".cmd", ".bat"}:
        payload = _diagnose_shell(source, text)
    else:
        payload = {
            "language": "unknown",
            "source_file": str(source),
            "findings": [
                DiagnosticItem(
                    severity="warning",
                    kind="unsupported_language",
                    message="No language-specific diagnostic rules exist for this file type yet.",
                ).to_dict()
            ],
            "strengths": [],
            "entropy_clusters": [],
        }

    payload["summary"] = {
        "n_findings": len(payload["findings"]),
        "n_strengths": len(payload["strengths"]),
        "n_entropy_clusters": len(payload.get("entropy_clusters", [])),
        "overall_assessment": _overall_assessment(payload["findings"]),
    }
    return payload


def _overall_assessment(findings: list[dict[str, object]]) -> str:
    if not findings:
        return "no_major_logic_smells_detected"
    severities = {item.get("severity") for item in findings}
    if "error" in severities:
        return "high_risk_logic"
    return "needs_review"


def diagnose_report(payload: dict[str, object]) -> str:
    lines = [
        f"source_file={payload['source_file']}",
        f"language={payload['language']}",
        f"assessment={payload['summary']['overall_assessment']}",
        (
            f"findings={payload['summary']['n_findings']} "
            f"strengths={payload['summary']['n_strengths']} "
            f"entropy_clusters={payload['summary']['n_entropy_clusters']}"
        ),
        "",
        "entropy_clusters",
    ]

    clusters = payload.get("entropy_clusters", [])
    if clusters:
        for cluster in clusters:
            where = f" line={cluster['line']}" if cluster.get("line") else ""
            labels = ",".join(cluster.get("logic_labels", []))
            lines.append(
                f"- {cluster['scope']} name={cluster['name']}{where} "
                f"cluster={cluster['cluster']} entropy={cluster['entropy']:.4f} "
                f"dominant_signal={cluster['dominant_signal']} labels={labels}"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("findings")
    findings = payload["findings"]
    if findings:
        for item in findings:
            where = f" line={item['line']}" if item.get("line") else ""
            symbol = f" symbol={item['symbol']}" if item.get("symbol") else ""
            lines.append(f"- [{item['severity']}] {item['kind']}{where}{symbol} :: {item['message']}")
    else:
        lines.append("- none")

    lines.append("")
    lines.append("strengths")
    strengths = payload["strengths"]
    if strengths:
        for item in strengths:
            where = f" line={item['line']}" if item.get("line") else ""
            symbol = f" symbol={item['symbol']}" if item.get("symbol") else ""
            lines.append(f"- [{item['severity']}] {item['kind']}{where}{symbol} :: {item['message']}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def diagnose_report_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)
