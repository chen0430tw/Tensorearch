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


# ---------------------------------------------------------------------------
# Line pre-processing: strip comments and string literals before regex
# ---------------------------------------------------------------------------

# Comment prefix patterns per language family
_COMMENT_STYLES: dict[str, list[str]] = {
    "c_family":  ["//"],           # Go, Rust, JS, TS, Java, C#, C++, C, Zig, PHP
    "hash":      ["#"],            # Python, Shell, Ruby, YAML, Dockerfile, PHP
    "sql":       ["--"],           # SQL
    "basic":     ["'", "REM "],    # VB/VBA/VBScript
    "lua":       ["--"],           # Lua
    "epl":       [],               # EPL has no standard line comments
}

# Map file suffix → comment style key
_SUFFIX_TO_COMMENT: dict[str, str] = {
    ".py": "hash", ".sh": "hash", ".bash": "hash", ".zsh": "hash",
    ".rb": "hash", ".yaml": "hash", ".yml": "hash",
    ".go": "c_family", ".rs": "c_family", ".js": "c_family",
    ".jsx": "c_family", ".mjs": "c_family", ".ts": "c_family",
    ".tsx": "c_family", ".java": "c_family", ".cs": "c_family",
    ".cpp": "c_family", ".cc": "c_family", ".cxx": "c_family",
    ".hpp": "c_family", ".c": "c_family", ".h": "c_family",
    ".zig": "c_family",
    ".php": "c_family",  # PHP also supports # but // is more common
    ".sql": "sql",
    ".lua": "lua",
    ".bas": "basic", ".vb": "basic", ".vbs": "basic", ".frm": "basic",
    ".e": "epl", ".ec": "epl",
    ".ps1": "hash", ".cmd": "hash", ".bat": "hash",
    ".pseudo": "c_family", ".ppc": "c_family",
}

# Regex to strip quoted strings: matches "..." or '...' (non-greedy, handles escaped quotes)
_DOUBLE_QUOTE_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_SINGLE_QUOTE_RE = re.compile(r"'(?:[^'\\]|\\.)*'")
# Backtick strings (JS/TS template literals, Go raw strings)
_BACKTICK_RE = re.compile(r'`(?:[^`\\]|\\.)*`')


def _strip_line(line: str, suffix: str) -> str:
    """Remove string literals and comments from a line for cleaner regex matching.

    Returns the 'logic skeleton' of the line — only code structure remains.
    This is applied before regex counting to avoid false positives from
    keywords inside strings or comments.

    Example:
        input:  '# if this is important'  (suffix=".py")
        output: ''  (entire line is a comment)

        input:  'x = "if (true)"'  (suffix=".py")
        output: 'x = ""'  (string content stripped)

        input:  'if x > 0:  // check boundary'  (suffix=".go")
        output: 'if x > 0:'  (comment stripped)
    """
    # Step 1: Strip string literals (replace content with empty string, keep quotes)
    stripped = _DOUBLE_QUOTE_RE.sub('""', line)
    stripped = _SINGLE_QUOTE_RE.sub("''", stripped)
    if suffix in {".js", ".jsx", ".mjs", ".ts", ".tsx", ".go"}:
        stripped = _BACKTICK_RE.sub('``', stripped)

    # Step 2: Strip line comments
    style_key = _SUFFIX_TO_COMMENT.get(suffix, "")
    if style_key:
        prefixes = _COMMENT_STYLES.get(style_key, [])
        for prefix in prefixes:
            # Find comment start (not inside remaining quotes)
            idx = stripped.find(prefix)
            if idx >= 0:
                # For Basic, REM must be at line start or after whitespace
                if prefix == "REM " and idx > 0 and not stripped[:idx].strip() == "":
                    continue
                stripped = stripped[:idx]

    return stripped


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


def _normalized_entropy(counts: dict[int, int] | dict[str, int]) -> float:
    total = sum(v for v in counts.values() if v > 0)
    active = sum(1 for v in counts.values() if v > 0)
    if total <= 0 or active <= 1:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        if value <= 0:
            continue
        p = value / total
        entropy -= p * math.log2(p)
    return entropy / math.log2(active)


def _modular_flow_profile(
    event_lines: list[int],
    span_start: int,
    span_end: int,
) -> dict[str, object]:
    if not event_lines:
        return {
            "modulus": 0,
            "event_count": 0,
            "modular_uniformity": 0.0,
            "topological_uniformity": 0.0,
            "modular_shrinking_number": 0.0,
            "assessment": "insufficient_signal",
            "hotspots": [],
        }

    unique_lines = sorted(set(line for line in event_lines if line > 0))
    event_count = len(unique_lines)
    if event_count < 4:
        return {
            "modulus": 0,
            "event_count": event_count,
            "modular_uniformity": 0.0,
            "topological_uniformity": 0.0,
            "modular_shrinking_number": 0.0,
            "assessment": "insufficient_signal",
            "hotspots": [],
        }
    modulus = max(3, min(11, int(round(math.sqrt(event_count + 1)))))

    modular_counts = {idx: 0 for idx in range(modulus)}
    for line in unique_lines:
        modular_counts[line % modulus] += 1
    modular_uniformity = _normalized_entropy(modular_counts)

    span = max(span_end - span_start + 1, 1)
    n_bins = max(3, min(8, int(round(math.sqrt(event_count))) or 1))
    topological_counts = {idx: 0 for idx in range(n_bins)}
    for line in unique_lines:
        rel = (line - span_start) / span
        idx = min(n_bins - 1, max(0, int(rel * n_bins)))
        topological_counts[idx] += 1
    topological_uniformity = _normalized_entropy(topological_counts)

    modular_shrinking_number = round(
        max(0.0, min(1.0, 1.0 - 0.5 * (modular_uniformity + topological_uniformity))),
        4,
    )

    expected_mod = event_count / max(modulus, 1)
    expected_top = event_count / max(n_bins, 1)
    hotspots: list[str] = []
    for residue, count in modular_counts.items():
        if count >= max(2, math.ceil(expected_mod * 1.6)):
            hotspots.append(f"mod:{residue}")
    for topo_bin, count in topological_counts.items():
        if count >= max(2, math.ceil(expected_top * 1.6)):
            hotspots.append(f"topo:{topo_bin}")

    if modular_shrinking_number >= 0.6:
        assessment = "concentrated_flow"
    elif modular_uniformity >= 0.8 and topological_uniformity >= 0.8:
        assessment = "uniform_flow"
    else:
        assessment = "mixed_flow"

    return {
        "modulus": modulus,
        "event_count": event_count,
        "modular_uniformity": round(modular_uniformity, 4),
        "topological_uniformity": round(topological_uniformity, 4),
        "modular_shrinking_number": modular_shrinking_number,
        "assessment": assessment,
        "hotspots": hotspots[:6],
    }


def _python_function_bucket(counts: dict[str, int], entropy: float) -> str:
    if _is_short_boolean_helper(counts):
        return "medium_entropy" if entropy >= 1.0 else "low_entropy"
    return _entropy_bucket(entropy)


def _is_short_boolean_helper(counts: dict[str, int]) -> bool:
    control_nodes = counts.get("if", 0) + counts.get("loop", 0)
    comparison_nodes = counts.get("compare", 0) + counts.get("boolop", 0)
    dataflow_nodes = counts.get("assign", 0) + counts.get("aug_assign", 0) + counts.get("call", 0)
    returns = counts.get("return", 0)
    return (
        control_nodes <= 1
        and returns >= 1
        and comparison_nodes >= 1
        and dataflow_nodes <= 4
        and counts.get("loop", 0) == 0
        and counts.get("aug_assign", 0) == 0
    )


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
        self._module_event_lines: list[int] = []

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
        self._module_event_lines.append(node.lineno)
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
        self._module_event_lines.append(node.lineno)
        if isinstance(node.target, ast.Name):
            target = node.target.id
            op = type(node.op).__name__
            self._score_mutations.setdefault(target, []).append((op, node.lineno))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._module_counts["call"] += 1
        self._module_event_lines.append(node.lineno)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._module_counts["if"] += 1
        self._module_event_lines.append(node.lineno)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._module_counts["loop"] += 1
        self._module_event_lines.append(node.lineno)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._module_counts["loop"] += 1
        self._module_event_lines.append(node.lineno)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self._module_counts["return"] += 1
        self._module_event_lines.append(node.lineno)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        self._module_counts["compare"] += 1
        self._module_event_lines.append(node.lineno)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self._module_counts["boolop"] += 1
        self._module_event_lines.append(node.lineno)
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
        n_lines = max(len(self.text.splitlines()), 1)
        return [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(module_entropy),
                "entropy": round(module_entropy, 4),
                "dominant_signal": _dominant_signal(self._module_counts),
                "logic_labels": _logic_labels("<module>", self.text, self._module_counts, "python"),
                "counts": dict(self._module_counts),
                "modular_flow": _modular_flow_profile(self._module_event_lines, 1, n_lines),
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
        event_lines: list[int] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                counts["assign"] += 1
                event_lines.append(child.lineno)
            elif isinstance(child, ast.AugAssign):
                counts["aug_assign"] += 1
                event_lines.append(child.lineno)
            elif isinstance(child, ast.Call):
                counts["call"] += 1
                event_lines.append(child.lineno)
            elif isinstance(child, ast.If):
                counts["if"] += 1
                event_lines.append(child.lineno)
            elif isinstance(child, (ast.For, ast.While)):
                counts["loop"] += 1
                event_lines.append(child.lineno)
            elif isinstance(child, ast.Return):
                counts["return"] += 1
                event_lines.append(child.lineno)
            elif isinstance(child, ast.Compare):
                counts["compare"] += 1
                event_lines.append(child.lineno)
            elif isinstance(child, ast.BoolOp):
                counts["boolop"] += 1
                event_lines.append(child.lineno)
        entropy = _shannon_entropy(counts)
        span_end = getattr(node, "end_lineno", node.lineno)
        return {
            "scope": "function",
            "name": node.name,
            "cluster": _python_function_bucket(counts, entropy),
            "entropy": round(entropy, 4),
            "dominant_signal": _dominant_signal(counts),
            "logic_labels": _logic_labels(node.name, text, counts, "python"),
            "counts": counts,
            "line": node.lineno,
            "modular_flow": _modular_flow_profile(event_lines, node.lineno, span_end),
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
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

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
            event_lines.append(idx)
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
                "modular_flow": _modular_flow_profile(event_lines, 1, max(len(lines), 1)),
            }
        ],
    }


def _diagnose_go(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose Go source via regex pattern matching (no AST required)."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    # Per-function tracking
    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "compare": 0, "switch": 0, "defer": 0,
        "goroutine": 0, "error_check": 0,
    }

    # Regex patterns for Go
    func_re = re.compile(r"^func\s+(?:\(.*?\)\s*)?(\w+)\s*\(")
    assign_re = re.compile(r"(?::=|[^=!<>]=[^=])")
    call_re = re.compile(r"\w+\.\w+\(|\w+\(")
    if_re = re.compile(r"^\s*if\s+")
    for_re = re.compile(r"^\s*for\s+")
    return_re = re.compile(r"^\s*return\s")
    switch_re = re.compile(r"^\s*switch\s")
    defer_re = re.compile(r"^\s*defer\s")
    go_re = re.compile(r"^\s*go\s+\w")
    err_check_re = re.compile(r"if\s+err\s*!=\s*nil")
    nil_return_re = re.compile(r"return\s+.*,\s*nil|return\s+nil\s*,")
    mutex_re = re.compile(r"\.(?:Lock|Unlock|RLock|RUnlock)\(\)")
    atomic_re = re.compile(r"atomic\.\w+")
    panic_re = re.compile(r"\bpanic\(")

    # Track variable overwrites
    var_assignments: dict[str, list[int]] = {}
    var_assign_re = re.compile(r"^\s*(\w+)\s*(?::?=)\s*")

    # Current function tracking
    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    brace_depth = 0
    in_func = False

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("//"):
            continue

        # Track brace depth
        brace_depth += stripped.count("{") - stripped.count("}")

        # Detect function boundaries
        fm = func_re.match(stripped)
        if fm:
            # Close previous function
            if in_func and cur_func_name:
                func_clusters.append(_build_go_cluster(
                    cur_func_name, cur_func_start, idx - 1,
                    cur_func_counts, cur_func_events, text,
                ))
            cur_func_name = fm.group(1)
            cur_func_start = idx
            cur_func_counts = {k: 0 for k in module_counts}
            cur_func_events = []
            in_func = True
            continue

        # Count patterns
        if assign_re.search(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)
            vm = var_assign_re.match(stripped)
            if vm:
                var_assignments.setdefault(vm.group(1), []).append(idx)

        if call_re.search(stripped):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if for_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if switch_re.match(stripped):
            module_counts["switch"] += 1
            if in_func:
                cur_func_counts["switch"] = cur_func_counts.get("switch", 0) + 1

        if defer_re.match(stripped):
            module_counts["defer"] += 1
            if in_func:
                cur_func_counts["defer"] = cur_func_counts.get("defer", 0) + 1

        if go_re.match(stripped):
            module_counts["goroutine"] += 1
            if in_func:
                cur_func_counts["goroutine"] = cur_func_counts.get("goroutine", 0) + 1

        if err_check_re.search(stripped):
            module_counts["error_check"] += 1
            if in_func:
                cur_func_counts["error_check"] = cur_func_counts.get("error_check", 0) + 1

        # Detect findings
        if panic_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="panic_call",
                message="panic() found — prefer returning error in library code.",
                line=idx, symbol="panic",
            ))

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_go_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Variable overwrite detection
    for name, hit_lines in var_assignments.items():
        if len(hit_lines) >= 5 and name not in ("err", "_", "i", "j", "k", "n"):
            findings.append(DiagnosticItem(
                severity="warning", kind="repeated_overwrite",
                message=f"Variable '{name}' is assigned {len(hit_lines)} times. May hide effective computation path.",
                line=hit_lines[0], symbol=name,
            ))

    # Strengths
    if module_counts["error_check"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="explicit_error_handling",
            message=f"{module_counts['error_check']} explicit 'if err != nil' checks found.",
            line=0, symbol="error_check",
        ))
    if module_counts["defer"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="defer_cleanup",
            message=f"{module_counts['defer']} defer statement(s) for cleanup.",
            line=0, symbol="defer",
        ))
    if module_counts["goroutine"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="goroutine_concurrency",
            message=f"{module_counts['goroutine']} goroutine launch(es) detected.",
            line=0, symbol="goroutine",
        ))

    # Module-level entropy
    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "go",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "go"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


def _build_go_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    """Build an entropy cluster for a Go function."""
    entropy = _shannon_entropy(counts)
    # Extract function text for label inference
    text_lines = full_text.splitlines()
    func_text = "\n".join(text_lines[start - 1:end]) if start > 0 else ""
    return {
        "scope": "function",
        "name": name,
        "cluster": _entropy_bucket(entropy),
        "entropy": round(entropy, 4),
        "dominant_signal": _dominant_signal(counts),
        "logic_labels": _logic_labels(name, func_text, counts, "go"),
        "counts": dict(counts),
        "line": start,
        "modular_flow": _modular_flow_profile(event_lines, start, end),
    }


def _diagnose_c_pseudo(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose C-like pseudocode (e.g., from PPM PseudoCodeGenerator output).

    Extracts control flow and data flow patterns from C-style pseudocode
    without requiring a real C parser. Works on PPM's output format as well
    as hand-written C pseudocode.
    """
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "compare": 0, "goto": 0, "bitmask": 0,
    }

    # Per-function tracking
    func_clusters: list[dict[str, object]] = []
    func_re = re.compile(r"^(?:void|NTSTATUS|int|PVOID|BOOLEAN|HANDLE|LONG)\s+(\w+)\s*\(")
    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False

    # API call tracking for architecture inference
    api_calls: list[tuple[str, int]] = []
    goto_count = 0
    bitmask_count = 0

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("//"):
            # Check for function header in comments
            if "Function at" in stripped:
                continue
            continue

        # Detect function boundaries
        fm = func_re.match(stripped)
        if fm:
            if in_func and cur_func_name:
                func_clusters.append(_build_pseudo_cluster(
                    cur_func_name, cur_func_start, idx - 1,
                    cur_func_counts, cur_func_events, text,
                ))
            cur_func_name = fm.group(1)
            cur_func_start = idx
            cur_func_counts = {k: 0 for k in counts}
            cur_func_events = []
            in_func = True
            continue

        # Assignment: "reg = value;" or "dst = src;"
        if "=" in stripped and "==" not in stripped and "!=" not in stripped and not stripped.startswith("if"):
            if re.search(r"\w+\s*[+\-|&^]*=\s*", stripped):
                counts["assign"] += 1
                if in_func:
                    cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
                event_lines.append(idx)
                if in_func:
                    cur_func_events.append(idx)

        # Bitmask operations
        if "&=" in stripped or "|=" in stripped:
            counts["bitmask"] += 1
            bitmask_count += 1
            if in_func:
                cur_func_counts["bitmask"] = cur_func_counts.get("bitmask", 0) + 1

        # Function calls: "ApiName(args);"
        call_match = re.search(r"(\w+)\s*\(", stripped)
        if call_match and not stripped.startswith("if") and not stripped.startswith("for"):
            fn_name = call_match.group(1)
            if fn_name not in ("if", "for", "while", "switch", "void", "int", "return"):
                counts["call"] += 1
                api_calls.append((fn_name, idx))
                if in_func:
                    cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
                event_lines.append(idx)
                if in_func:
                    cur_func_events.append(idx)

        # Conditionals
        if stripped.startswith("if ") or stripped.startswith("if("):
            counts["if"] += 1
            counts["compare"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
                cur_func_counts["compare"] = cur_func_counts.get("compare", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        # Loops
        if stripped.startswith("for ") or stripped.startswith("while ") or stripped.startswith("for(") or stripped.startswith("while("):
            counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        # Return
        if stripped.startswith("return ") or stripped == "return;":
            counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        # Goto (common in decompiled code)
        if stripped.startswith("goto "):
            counts["goto"] += 1
            goto_count += 1
            if in_func:
                cur_func_counts["goto"] = cur_func_counts.get("goto", 0) + 1

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_pseudo_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Findings for pseudocode
    if goto_count >= 3:
        findings.append(DiagnosticItem(
            severity="warning", kind="excessive_goto",
            message=f"{goto_count} goto statements — complex control flow, likely from decompilation.",
            line=0, symbol="goto",
        ))
    if bitmask_count >= 3:
        strengths.append(DiagnosticItem(
            severity="info", kind="bitmask_operations",
            message=f"{bitmask_count} bitmask operations detected — likely flag manipulation or permission checks.",
            line=0, symbol="bitmask",
        ))

    # Detect known dangerous API patterns
    dangerous_apis = {"ZwTerminateProcess", "KeInsertQueueApc", "ZwAllocateVirtualMemory"}
    protection_apis = {"ObRegisterCallbacks", "CmRegisterCallbackEx", "PsSetCreateProcessNotifyRoutine"}
    found_dangerous = [(name, ln) for name, ln in api_calls if name in dangerous_apis]
    found_protection = [(name, ln) for name, ln in api_calls if name in protection_apis]

    if found_dangerous:
        for name, ln in found_dangerous:
            findings.append(DiagnosticItem(
                severity="warning", kind="dangerous_api",
                message=f"Potentially dangerous API call: {name}",
                line=ln, symbol=name,
            ))
    if found_protection:
        for name, ln in found_protection:
            strengths.append(DiagnosticItem(
                severity="info", kind="protection_callback",
                message=f"Protection/monitoring callback registration: {name}",
                line=ln, symbol=name,
            ))

    entropy = _shannon_entropy(counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "c_pseudo",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(counts),
                "logic_labels": _logic_labels("<module>", text, counts, "c_pseudo"),
                "counts": dict(counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


def _build_pseudo_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    """Build an entropy cluster for a pseudocode function."""
    entropy = _shannon_entropy(counts)
    text_lines = full_text.splitlines()
    func_text = "\n".join(text_lines[start - 1:end]) if start > 0 else ""
    return {
        "scope": "function",
        "name": name,
        "cluster": _entropy_bucket(entropy),
        "entropy": round(entropy, 4),
        "dominant_signal": _dominant_signal(counts),
        "logic_labels": _logic_labels(name, func_text, counts, "c_pseudo"),
        "counts": dict(counts),
        "line": start,
        "modular_flow": _modular_flow_profile(event_lines, start, end),
    }


def _build_rust_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    """Build an entropy cluster for a Rust function."""
    entropy = _shannon_entropy(counts)
    text_lines = full_text.splitlines()
    func_text = "\n".join(text_lines[start - 1:end]) if start > 0 else ""
    return {
        "scope": "function",
        "name": name,
        "cluster": _entropy_bucket(entropy),
        "entropy": round(entropy, 4),
        "dominant_signal": _dominant_signal(counts),
        "logic_labels": _logic_labels(name, func_text, counts, "rust"),
        "counts": dict(counts),
        "line": start,
        "modular_flow": _modular_flow_profile(event_lines, start, end),
    }


def _diagnose_rust(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose Rust source via regex pattern matching (no AST required)."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "match_arm": 0, "unsafe": 0, "lifetime": 0,
        "macro_call": 0, "error_propagation": 0,
    }

    # Regex patterns for Rust
    func_re = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*[<(]")
    assign_re = re.compile(r"(?:let\s+(?:mut\s+)?\w+\s*(?::\s*\S+\s*)?=|[^=!<>]=[^=])")
    call_re = re.compile(r"\w+\s*\(")
    if_re = re.compile(r"^\s*(?:if\s+|else\s+if\s+)")
    loop_re = re.compile(r"^\s*(?:for\s+|while\s+|loop\s*\{)")
    return_re = re.compile(r"^\s*return\s")
    match_arm_re = re.compile(r"^\s*(?:\w+|_)\s*(?:\(.*?\)\s*)?(?:\|.*?)?=>\s*")
    unsafe_re = re.compile(r"\bunsafe\s*\{")
    lifetime_re = re.compile(r"[&<]'[a-z]\w*")
    macro_call_re = re.compile(r"\w+!\s*[\(\[\{]")
    error_prop_re = re.compile(r"\?\s*;|\?\s*$")
    unwrap_re = re.compile(r"\.unwrap\(\)")
    panic_re = re.compile(r"\bpanic!\s*\(")
    result_option_re = re.compile(r"->\s*(?:Result|Option)\b")

    # Variable overwrite tracking
    var_assignments: dict[str, list[int]] = {}
    var_assign_re = re.compile(r"^\s*(?:let\s+(?:mut\s+)?)?(\w+)\s*(?::\s*\S+\s*)?=\s*")

    # Current function tracking
    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("//"):
            continue

        # Detect function boundaries
        fm = func_re.match(line)
        if fm:
            if in_func and cur_func_name:
                func_clusters.append(_build_rust_cluster(
                    cur_func_name, cur_func_start, idx - 1,
                    cur_func_counts, cur_func_events, text,
                ))
            cur_func_name = fm.group(1)
            cur_func_start = idx
            cur_func_counts = {k: 0 for k in module_counts}
            cur_func_events = []
            in_func = True
            continue

        # --- Count patterns ---
        if assign_re.search(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)
            vm = var_assign_re.match(stripped)
            if vm:
                var_assignments.setdefault(vm.group(1), []).append(idx)

        if call_re.search(stripped):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if loop_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if match_arm_re.match(stripped):
            module_counts["match_arm"] += 1
            if in_func:
                cur_func_counts["match_arm"] = cur_func_counts.get("match_arm", 0) + 1

        if unsafe_re.search(stripped):
            module_counts["unsafe"] += 1
            if in_func:
                cur_func_counts["unsafe"] = cur_func_counts.get("unsafe", 0) + 1
            findings.append(DiagnosticItem(
                severity="warning", kind="unsafe_block",
                message="unsafe block found — verify memory safety invariants are upheld.",
                line=idx, symbol="unsafe",
            ))

        if lifetime_re.search(stripped):
            module_counts["lifetime"] += 1
            if in_func:
                cur_func_counts["lifetime"] = cur_func_counts.get("lifetime", 0) + 1

        if macro_call_re.search(stripped):
            module_counts["macro_call"] += 1
            if in_func:
                cur_func_counts["macro_call"] = cur_func_counts.get("macro_call", 0) + 1

        if error_prop_re.search(stripped):
            module_counts["error_propagation"] += 1
            if in_func:
                cur_func_counts["error_propagation"] = cur_func_counts.get("error_propagation", 0) + 1

        # --- Findings ---
        if unwrap_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="unwrap_call",
                message=".unwrap() can panic at runtime — prefer pattern matching or ? operator.",
                line=idx, symbol="unwrap",
            ))

        if panic_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="panic_macro",
                message="panic!() found — prefer returning Result in library code.",
                line=idx, symbol="panic",
            ))

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_rust_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Variable overwrite detection
    for name, hit_lines in var_assignments.items():
        if len(hit_lines) >= 5 and name not in ("_", "i", "j", "k", "n", "idx"):
            findings.append(DiagnosticItem(
                severity="warning", kind="repeated_overwrite",
                message=f"Variable '{name}' is assigned {len(hit_lines)} times. May hide effective computation path.",
                line=hit_lines[0], symbol=name,
            ))

    # Strengths
    if module_counts["error_propagation"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="error_propagation",
            message=f"{module_counts['error_propagation']} error propagation(s) via ? operator.",
            line=0, symbol="error_propagation",
        ))

    # Scan for Result/Option return types (strengths)
    result_option_count = sum(1 for line in lines if result_option_re.search(line))
    if result_option_count > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="result_option_handling",
            message=f"{result_option_count} function(s) return Result/Option for explicit error handling.",
            line=0, symbol="Result",
        ))

    if module_counts["match_arm"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="pattern_matching",
            message=f"{module_counts['match_arm']} match arm(s) detected — exhaustive pattern matching.",
            line=0, symbol="match",
        ))

    if module_counts["lifetime"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="lifetime_annotations",
            message=f"{module_counts['lifetime']} lifetime annotation(s) for explicit ownership tracking.",
            line=0, symbol="lifetime",
        ))

    # Module-level entropy
    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "rust",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "rust"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


# ===================================================================
#  JavaScript Analyzer
# ===================================================================

def _build_js_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    """Build an entropy cluster for a JavaScript function."""
    entropy = _shannon_entropy(counts)
    text_lines = full_text.splitlines()
    func_text = "\n".join(text_lines[start - 1:end]) if start > 0 else ""
    return {
        "scope": "function",
        "name": name,
        "cluster": _entropy_bucket(entropy),
        "entropy": round(entropy, 4),
        "dominant_signal": _dominant_signal(counts),
        "logic_labels": _logic_labels(name, func_text, counts, "javascript"),
        "counts": dict(counts),
        "line": start,
        "modular_flow": _modular_flow_profile(event_lines, start, end),
    }


def _diagnose_javascript(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose JavaScript source via regex pattern matching (no AST required)."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "async_await": 0, "arrow_fn": 0, "try_catch": 0,
        "promise": 0, "callback": 0,
    }

    # Regex patterns for JavaScript
    # function foo(, const foo = (, const foo = async (
    func_re = re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\("
        r"|^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\("
        r"|^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\w*\s*=>\s*"
    )
    assign_re = re.compile(r"(?:(?:const|let|var)\s+\w+\s*=|[^=!<>]=[^=])")
    call_re = re.compile(r"\w+\s*\(")
    if_re = re.compile(r"^\s*(?:if\s*\(|else\s+if\s*\()")
    loop_re = re.compile(r"^\s*(?:for\s*\(|while\s*\(|do\s*\{)")
    return_re = re.compile(r"^\s*return[\s;]")
    async_await_re = re.compile(r"\bawait\s+|\basync\s+")
    arrow_fn_re = re.compile(r"=>\s*[\{(]|=>\s*\w")
    try_catch_re = re.compile(r"^\s*(?:try\s*\{|catch\s*\()")
    promise_re = re.compile(r"new\s+Promise\b|\.then\s*\(|\.catch\s*\(|Promise\.\w+\s*\(")
    callback_re = re.compile(r"\w+\s*\(\s*(?:function\s*\(|(?:\w+|\([^)]*\))\s*=>)")
    eval_re = re.compile(r"\beval\s*\(")
    var_re = re.compile(r"^\s*var\s+")
    console_re = re.compile(r"\bconsole\.\w+\s*\(")
    destructure_re = re.compile(r"(?:const|let|var)\s+[\{\[].+[\}\]]\s*=")

    # Variable overwrite tracking
    var_assignments: dict[str, list[int]] = {}
    var_assign_re = re.compile(r"^\s*(?:(?:const|let|var)\s+)?(\w+)\s*=\s*")

    # Current function tracking
    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False

    # Callback nesting tracking
    callback_nesting = 0
    max_callback_nesting = 0

    is_test_file = bool(re.search(r"\.(?:test|spec)\.", str(path)))

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("//"):
            continue

        # Detect function boundaries
        fm = func_re.match(line)
        if fm:
            func_name = fm.group(1) or fm.group(2) or fm.group(3)
            if func_name:
                if in_func and cur_func_name:
                    func_clusters.append(_build_js_cluster(
                        cur_func_name, cur_func_start, idx - 1,
                        cur_func_counts, cur_func_events, text,
                    ))
                cur_func_name = func_name
                cur_func_start = idx
                cur_func_counts = {k: 0 for k in module_counts}
                cur_func_events = []
                in_func = True

        # --- Count patterns ---
        if assign_re.search(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)
            vm = var_assign_re.match(stripped)
            if vm:
                var_assignments.setdefault(vm.group(1), []).append(idx)

        if call_re.search(stripped):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if loop_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if async_await_re.search(stripped):
            module_counts["async_await"] += 1
            if in_func:
                cur_func_counts["async_await"] = cur_func_counts.get("async_await", 0) + 1

        if arrow_fn_re.search(stripped):
            module_counts["arrow_fn"] += 1
            if in_func:
                cur_func_counts["arrow_fn"] = cur_func_counts.get("arrow_fn", 0) + 1

        if try_catch_re.match(stripped):
            module_counts["try_catch"] += 1
            if in_func:
                cur_func_counts["try_catch"] = cur_func_counts.get("try_catch", 0) + 1

        if promise_re.search(stripped):
            module_counts["promise"] += 1
            if in_func:
                cur_func_counts["promise"] = cur_func_counts.get("promise", 0) + 1

        if callback_re.search(stripped):
            module_counts["callback"] += 1
            callback_nesting += 1
            if in_func:
                cur_func_counts["callback"] = cur_func_counts.get("callback", 0) + 1
        else:
            # Reset nesting on non-callback lines (simplified heuristic)
            if callback_nesting > max_callback_nesting:
                max_callback_nesting = callback_nesting
            callback_nesting = 0

        # --- Findings ---
        if eval_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="eval_usage",
                message="eval() is a security risk — prefer safer alternatives.",
                line=idx, symbol="eval",
            ))

        if var_re.match(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="var_usage",
                message="'var' has function-scoped hoisting — prefer 'let' or 'const'.",
                line=idx, symbol="var",
            ))

        if console_re.search(stripped) and not is_test_file:
            findings.append(DiagnosticItem(
                severity="warning", kind="console_in_production",
                message="console.log in non-test file — consider removing or using a logger.",
                line=idx, symbol="console",
            ))

    # Check max callback nesting
    if callback_nesting > max_callback_nesting:
        max_callback_nesting = callback_nesting
    if max_callback_nesting >= 3:
        findings.append(DiagnosticItem(
            severity="warning", kind="callback_hell",
            message=f"{max_callback_nesting} nested callbacks detected — consider refactoring to async/await.",
            line=0, symbol="callback",
        ))

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_js_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Variable overwrite detection
    for name, hit_lines in var_assignments.items():
        if len(hit_lines) >= 5 and name not in ("_", "i", "j", "k", "n", "idx"):
            findings.append(DiagnosticItem(
                severity="warning", kind="repeated_overwrite",
                message=f"Variable '{name}' is assigned {len(hit_lines)} times. May hide effective computation path.",
                line=hit_lines[0], symbol=name,
            ))

    # Strengths
    if module_counts["async_await"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="async_await_usage",
            message=f"{module_counts['async_await']} async/await usage(s) for readable asynchronous code.",
            line=0, symbol="async",
        ))

    destructure_count = sum(1 for line in lines if destructure_re.search(line))
    if destructure_count > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="destructuring",
            message=f"{destructure_count} destructuring assignment(s) for concise data extraction.",
            line=0, symbol="destructure",
        ))

    if module_counts["arrow_fn"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="arrow_functions",
            message=f"{module_counts['arrow_fn']} arrow function(s) for concise function expressions.",
            line=0, symbol="arrow",
        ))

    # Module-level entropy
    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "javascript",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "javascript"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


# ===================================================================
#  TypeScript Analyzer
# ===================================================================

def _build_ts_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    """Build an entropy cluster for a TypeScript function."""
    entropy = _shannon_entropy(counts)
    text_lines = full_text.splitlines()
    func_text = "\n".join(text_lines[start - 1:end]) if start > 0 else ""
    return {
        "scope": "function",
        "name": name,
        "cluster": _entropy_bucket(entropy),
        "entropy": round(entropy, 4),
        "dominant_signal": _dominant_signal(counts),
        "logic_labels": _logic_labels(name, func_text, counts, "typescript"),
        "counts": dict(counts),
        "line": start,
        "modular_flow": _modular_flow_profile(event_lines, start, end),
    }


def _diagnose_typescript(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose TypeScript source via regex pattern matching (no AST required).

    Reuses JavaScript logic and adds TypeScript-specific counts, findings,
    and strengths (type_annotation, interface, generic, type_assertion, any, ts-ignore).
    """
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        # JS base counts
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "async_await": 0, "arrow_fn": 0, "try_catch": 0,
        "promise": 0, "callback": 0,
        # TS-specific counts
        "type_annotation": 0, "interface": 0, "generic": 0, "type_assertion": 0,
    }

    # --- JS regex patterns (same as _diagnose_javascript) ---
    func_re = re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[<(]"
        r"|^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\("
        r"|^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\w*\s*=>\s*"
    )
    assign_re = re.compile(r"(?:(?:const|let|var)\s+\w+\s*(?::\s*\S+\s*)?=|[^=!<>]=[^=])")
    call_re = re.compile(r"\w+\s*\(")
    if_re = re.compile(r"^\s*(?:if\s*\(|else\s+if\s*\()")
    loop_re = re.compile(r"^\s*(?:for\s*\(|while\s*\(|do\s*\{)")
    return_re = re.compile(r"^\s*return[\s;]")
    async_await_re = re.compile(r"\bawait\s+|\basync\s+")
    arrow_fn_re = re.compile(r"=>\s*[\{(]|=>\s*\w")
    try_catch_re = re.compile(r"^\s*(?:try\s*\{|catch\s*\()")
    promise_re = re.compile(r"new\s+Promise\b|\.then\s*\(|\.catch\s*\(|Promise\.\w+\s*\(")
    callback_re = re.compile(r"\w+\s*\(\s*(?:function\s*\(|(?:\w+|\([^)]*\))\s*=>)")
    eval_re = re.compile(r"\beval\s*\(")
    var_re = re.compile(r"^\s*var\s+")
    console_re = re.compile(r"\bconsole\.\w+\s*\(")
    destructure_re = re.compile(r"(?:const|let|var)\s+[\{\[].+[\}\]]\s*(?::\s*\S+\s*)?=")

    # --- TS-specific regex patterns ---
    type_annotation_re = re.compile(r":\s*(?:string|number|boolean|void|null|undefined|never|any|unknown|\w+(?:<[^>]+>)?)\b")
    interface_re = re.compile(r"^\s*(?:export\s+)?interface\s+(\w+)")
    generic_re = re.compile(r"<\s*\w+(?:\s+extends\s+\w+)?\s*(?:,\s*\w+(?:\s+extends\s+\w+)?\s*)*>")
    type_assertion_re = re.compile(r"\bas\s+\w+|<\w+>\s*\w+")
    any_re = re.compile(r":\s*any\b")
    ts_ignore_re = re.compile(r"@ts-ignore|@ts-nocheck")

    # Variable overwrite tracking
    var_assignments: dict[str, list[int]] = {}
    var_assign_re = re.compile(r"^\s*(?:(?:const|let|var)\s+)?(\w+)\s*(?::\s*\S+\s*)?=\s*")

    # Current function tracking
    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False

    callback_nesting = 0
    max_callback_nesting = 0

    is_test_file = bool(re.search(r"\.(?:test|spec)\.", str(path)))

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("//"):
            # Still check for ts-ignore in comments
            if ts_ignore_re.search(stripped):
                findings.append(DiagnosticItem(
                    severity="warning", kind="ts_ignore",
                    message="@ts-ignore suppresses type checking — fix the underlying type error instead.",
                    line=idx, symbol="ts-ignore",
                ))
            continue

        # Detect function boundaries
        fm = func_re.match(line)
        if fm:
            func_name = fm.group(1) or fm.group(2) or fm.group(3)
            if func_name:
                if in_func and cur_func_name:
                    func_clusters.append(_build_ts_cluster(
                        cur_func_name, cur_func_start, idx - 1,
                        cur_func_counts, cur_func_events, text,
                    ))
                cur_func_name = func_name
                cur_func_start = idx
                cur_func_counts = {k: 0 for k in module_counts}
                cur_func_events = []
                in_func = True

        # --- JS base counts ---
        if assign_re.search(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)
            vm = var_assign_re.match(stripped)
            if vm:
                var_assignments.setdefault(vm.group(1), []).append(idx)

        if call_re.search(stripped):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if loop_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if async_await_re.search(stripped):
            module_counts["async_await"] += 1
            if in_func:
                cur_func_counts["async_await"] = cur_func_counts.get("async_await", 0) + 1

        if arrow_fn_re.search(stripped):
            module_counts["arrow_fn"] += 1
            if in_func:
                cur_func_counts["arrow_fn"] = cur_func_counts.get("arrow_fn", 0) + 1

        if try_catch_re.match(stripped):
            module_counts["try_catch"] += 1
            if in_func:
                cur_func_counts["try_catch"] = cur_func_counts.get("try_catch", 0) + 1

        if promise_re.search(stripped):
            module_counts["promise"] += 1
            if in_func:
                cur_func_counts["promise"] = cur_func_counts.get("promise", 0) + 1

        if callback_re.search(stripped):
            module_counts["callback"] += 1
            callback_nesting += 1
            if in_func:
                cur_func_counts["callback"] = cur_func_counts.get("callback", 0) + 1
        else:
            if callback_nesting > max_callback_nesting:
                max_callback_nesting = callback_nesting
            callback_nesting = 0

        # --- TS-specific counts ---
        if type_annotation_re.search(stripped):
            module_counts["type_annotation"] += 1
            if in_func:
                cur_func_counts["type_annotation"] = cur_func_counts.get("type_annotation", 0) + 1

        if interface_re.match(stripped):
            module_counts["interface"] += 1
            if in_func:
                cur_func_counts["interface"] = cur_func_counts.get("interface", 0) + 1

        if generic_re.search(stripped):
            module_counts["generic"] += 1
            if in_func:
                cur_func_counts["generic"] = cur_func_counts.get("generic", 0) + 1

        if type_assertion_re.search(stripped):
            module_counts["type_assertion"] += 1
            if in_func:
                cur_func_counts["type_assertion"] = cur_func_counts.get("type_assertion", 0) + 1

        # --- JS Findings ---
        if eval_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="eval_usage",
                message="eval() is a security risk — prefer safer alternatives.",
                line=idx, symbol="eval",
            ))

        if var_re.match(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="var_usage",
                message="'var' has function-scoped hoisting — prefer 'let' or 'const'.",
                line=idx, symbol="var",
            ))

        if console_re.search(stripped) and not is_test_file:
            findings.append(DiagnosticItem(
                severity="warning", kind="console_in_production",
                message="console.log in non-test file — consider removing or using a logger.",
                line=idx, symbol="console",
            ))

        # --- TS-specific Findings ---
        if any_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="any_type",
                message="'any' type bypasses type checking — prefer explicit types or 'unknown'.",
                line=idx, symbol="any",
            ))

        if ts_ignore_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="ts_ignore",
                message="@ts-ignore suppresses type checking — fix the underlying type error instead.",
                line=idx, symbol="ts-ignore",
            ))

    # Callback hell check
    if callback_nesting > max_callback_nesting:
        max_callback_nesting = callback_nesting
    if max_callback_nesting >= 3:
        findings.append(DiagnosticItem(
            severity="warning", kind="callback_hell",
            message=f"{max_callback_nesting} nested callbacks detected — consider refactoring to async/await.",
            line=0, symbol="callback",
        ))

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_ts_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Variable overwrite detection
    for name, hit_lines in var_assignments.items():
        if len(hit_lines) >= 5 and name not in ("_", "i", "j", "k", "n", "idx"):
            findings.append(DiagnosticItem(
                severity="warning", kind="repeated_overwrite",
                message=f"Variable '{name}' is assigned {len(hit_lines)} times. May hide effective computation path.",
                line=hit_lines[0], symbol=name,
            ))

    # --- JS Strengths ---
    if module_counts["async_await"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="async_await_usage",
            message=f"{module_counts['async_await']} async/await usage(s) for readable asynchronous code.",
            line=0, symbol="async",
        ))

    destructure_count = sum(1 for line in lines if destructure_re.search(line))
    if destructure_count > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="destructuring",
            message=f"{destructure_count} destructuring assignment(s) for concise data extraction.",
            line=0, symbol="destructure",
        ))

    if module_counts["arrow_fn"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="arrow_functions",
            message=f"{module_counts['arrow_fn']} arrow function(s) for concise function expressions.",
            line=0, symbol="arrow",
        ))

    # --- TS Strengths ---
    if module_counts["type_annotation"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="explicit_types",
            message=f"{module_counts['type_annotation']} explicit type annotation(s) for type safety.",
            line=0, symbol="type_annotation",
        ))

    if module_counts["generic"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="generic_usage",
            message=f"{module_counts['generic']} generic type usage(s) for reusable typed abstractions.",
            line=0, symbol="generic",
        ))

    if module_counts["interface"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="interface_definitions",
            message=f"{module_counts['interface']} interface definition(s) for structural type contracts.",
            line=0, symbol="interface",
        ))

    # Module-level entropy
    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "typescript",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "typescript"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


# ===================================================================
#  Java Analyzer
# ===================================================================

def _build_java_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    """Build an entropy cluster for a Java method."""
    entropy = _shannon_entropy(counts)
    text_lines = full_text.splitlines()
    func_text = "\n".join(text_lines[start - 1:end]) if start > 0 else ""
    return {
        "scope": "function",
        "name": name,
        "cluster": _entropy_bucket(entropy),
        "entropy": round(entropy, 4),
        "dominant_signal": _dominant_signal(counts),
        "logic_labels": _logic_labels(name, func_text, counts, "java"),
        "counts": dict(counts),
        "line": start,
        "modular_flow": _modular_flow_profile(event_lines, start, end),
    }


def _diagnose_java(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose Java source via regex pattern matching (no AST required)."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "switch": 0, "try_catch": 0, "throw": 0,
        "annotation": 0, "synchronized": 0,
    }

    # Regex patterns for Java
    # Method: access modifier + optional static/abstract/final + return type + method name(
    method_re = re.compile(
        r"^\s*(?:(?:public|protected|private)\s+)?(?:(?:static|abstract|final|synchronized|native)\s+)*"
        r"(?:\w+(?:<[^>]*>)?(?:\[\])*)\s+(\w+)\s*\("
    )
    assign_re = re.compile(r"(?:(?:\w+(?:<[^>]*>)?(?:\[\])*)\s+\w+\s*=|[^=!<>]=[^=])")
    call_re = re.compile(r"\w+\s*\(")
    if_re = re.compile(r"^\s*(?:if\s*\(|else\s+if\s*\()")
    loop_re = re.compile(r"^\s*(?:for\s*\(|while\s*\(|do\s*\{)")
    return_re = re.compile(r"^\s*return[\s;]")
    switch_re = re.compile(r"^\s*switch\s*\(")
    try_re = re.compile(r"^\s*try\s*[\({]")
    catch_re = re.compile(r"^\s*\}\s*catch\s*\(")
    throw_re = re.compile(r"^\s*throw\s+")
    annotation_re = re.compile(r"^\s*@(\w+)")
    synchronized_re = re.compile(r"\bsynchronized\s*[\({]")
    system_exit_re = re.compile(r"System\.exit\s*\(")
    thread_sleep_re = re.compile(r"Thread\.sleep\s*\(")
    override_re = re.compile(r"^\s*@Override\b")
    try_with_resources_re = re.compile(r"^\s*try\s*\(")

    # Empty catch detection: "} catch (...) {" followed by "}" with only whitespace/comments
    # We track catch lines and check if the block is empty

    # Variable overwrite tracking
    var_assignments: dict[str, list[int]] = {}
    var_assign_re = re.compile(r"^\s*(?:(?:\w+(?:<[^>]*>)?(?:\[\])*)\s+)?(\w+)\s*=\s*")

    # Current function tracking
    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False

    is_test_file = bool(re.search(r"[Tt]est", str(path)))

    # Track catch blocks for empty-catch detection
    catch_lines: list[int] = []

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
            continue

        # Detect method boundaries
        fm = method_re.match(line)
        if fm:
            method_name = fm.group(1)
            # Exclude class/constructor-like matches (name starts with uppercase and matches class decl)
            if method_name and method_name not in ("if", "for", "while", "switch", "return", "new", "class"):
                if in_func and cur_func_name:
                    func_clusters.append(_build_java_cluster(
                        cur_func_name, cur_func_start, idx - 1,
                        cur_func_counts, cur_func_events, text,
                    ))
                cur_func_name = method_name
                cur_func_start = idx
                cur_func_counts = {k: 0 for k in module_counts}
                cur_func_events = []
                in_func = True

        # --- Count patterns ---
        if assign_re.search(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)
            vm = var_assign_re.match(stripped)
            if vm:
                var_assignments.setdefault(vm.group(1), []).append(idx)

        if call_re.search(stripped):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if loop_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if switch_re.match(stripped):
            module_counts["switch"] += 1
            if in_func:
                cur_func_counts["switch"] = cur_func_counts.get("switch", 0) + 1

        if try_re.match(stripped) or catch_re.match(stripped):
            module_counts["try_catch"] += 1
            if in_func:
                cur_func_counts["try_catch"] = cur_func_counts.get("try_catch", 0) + 1

        if catch_re.match(stripped):
            catch_lines.append(idx)

        if throw_re.match(stripped):
            module_counts["throw"] += 1
            if in_func:
                cur_func_counts["throw"] = cur_func_counts.get("throw", 0) + 1

        am = annotation_re.match(stripped)
        if am:
            module_counts["annotation"] += 1
            if in_func:
                cur_func_counts["annotation"] = cur_func_counts.get("annotation", 0) + 1

        if synchronized_re.search(stripped):
            module_counts["synchronized"] += 1
            if in_func:
                cur_func_counts["synchronized"] = cur_func_counts.get("synchronized", 0) + 1

        # --- Findings ---
        if system_exit_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="system_exit",
                message="System.exit() terminates JVM — prefer throwing an exception.",
                line=idx, symbol="System.exit",
            ))

        if thread_sleep_re.search(stripped) and not is_test_file:
            findings.append(DiagnosticItem(
                severity="warning", kind="thread_sleep",
                message="Thread.sleep() in non-test code — consider ScheduledExecutorService or async patterns.",
                line=idx, symbol="Thread.sleep",
            ))

    # Empty catch block detection
    for catch_line in catch_lines:
        # Look at the lines after the catch to see if the block is empty
        # Pattern: catch line contains "{", next non-blank line is "}"
        block_start = catch_line
        found_content = False
        for scan_idx in range(catch_line, min(catch_line + 5, len(lines))):
            scan_stripped = lines[scan_idx].strip() if scan_idx < len(lines) else ""
            # Skip the catch line itself and blank lines
            if scan_idx == catch_line - 1:
                continue
            if not scan_stripped or scan_stripped.startswith("//"):
                continue
            if scan_stripped == "}":
                # Empty catch block
                if not found_content:
                    findings.append(DiagnosticItem(
                        severity="warning", kind="empty_catch",
                        message="Empty catch block swallows exception — at minimum log the error.",
                        line=catch_line, symbol="catch",
                    ))
                break
            found_content = True

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_java_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Variable overwrite detection
    for name, hit_lines in var_assignments.items():
        if len(hit_lines) >= 5 and name not in ("_", "i", "j", "k", "n", "idx"):
            findings.append(DiagnosticItem(
                severity="warning", kind="repeated_overwrite",
                message=f"Variable '{name}' is assigned {len(hit_lines)} times. May hide effective computation path.",
                line=hit_lines[0], symbol=name,
            ))

    # Strengths
    override_count = sum(1 for line in lines if override_re.search(line))
    if override_count > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="override_annotations",
            message=f"{override_count} @Override annotation(s) for compile-time method contract verification.",
            line=0, symbol="@Override",
        ))

    try_resources_count = sum(1 for line in lines if try_with_resources_re.search(line))
    if try_resources_count > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="try_with_resources",
            message=f"{try_resources_count} try-with-resources statement(s) for automatic resource cleanup.",
            line=0, symbol="try-with-resources",
        ))

    if module_counts["synchronized"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="synchronized_blocks",
            message=f"{module_counts['synchronized']} synchronized block(s) for thread-safe access.",
            line=0, symbol="synchronized",
        ))

    # Module-level entropy
    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "java",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "java"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


def _build_func_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int],
    full_text: str, language: str,
) -> dict[str, object]:
    entropy = _shannon_entropy(counts)
    text_lines = full_text.splitlines()
    func_text = "\n".join(text_lines[start - 1:end]) if start > 0 else ""
    return {
        "scope": "function",
        "name": name,
        "cluster": _entropy_bucket(entropy),
        "entropy": round(entropy, 4),
        "dominant_signal": _dominant_signal(counts),
        "logic_labels": _logic_labels(name, func_text, counts, language),
        "counts": dict(counts),
        "line": start,
        "modular_flow": _modular_flow_profile(event_lines, start, end),
    }


# ===================================================================
# 1. Zig analyzer
# ===================================================================

def _build_zig_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    return _build_func_cluster(name, start, end, counts, event_lines, full_text, "zig")


def _diagnose_zig(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose Zig source via regex pattern matching."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "defer": 0, "comptime": 0,
        "error_union": 0, "test_block": 0,
    }

    func_re = re.compile(r"^\s*(?:pub\s+)?fn\s+(\w+)\s*\(")
    assign_re = re.compile(r"(?:const|var)\s+\w+\s*=|[^=!<>]=[^=]")
    call_re = re.compile(r"\w+\s*\(")
    if_re = re.compile(r"^\s*if\s*\(")
    loop_re = re.compile(r"^\s*(?:while|for)\s*[\(\|]")
    return_re = re.compile(r"^\s*return\s")
    defer_re = re.compile(r"^\s*(?:defer|errdefer)\s")
    comptime_re = re.compile(r"\bcomptime\b")
    error_union_re = re.compile(r"!\w+|!\{")
    test_re = re.compile(r'^\s*test\s+"')
    panic_re = re.compile(r"@panic\(")
    unreachable_re = re.compile(r"\bunreachable\b")

    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False
    brace_depth = 0

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("//"):
            continue

        brace_depth += stripped.count("{") - stripped.count("}")

        fm = func_re.match(stripped)
        if fm:
            if in_func and cur_func_name:
                func_clusters.append(_build_zig_cluster(
                    cur_func_name, cur_func_start, idx - 1,
                    cur_func_counts, cur_func_events, text,
                ))
            cur_func_name = fm.group(1)
            cur_func_start = idx
            cur_func_counts = {k: 0 for k in module_counts}
            cur_func_events = []
            in_func = True
            continue

        if assign_re.search(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if call_re.search(stripped) and not stripped.startswith("fn ") and not stripped.startswith("pub fn "):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if loop_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if defer_re.match(stripped):
            module_counts["defer"] += 1
            if in_func:
                cur_func_counts["defer"] = cur_func_counts.get("defer", 0) + 1

        if comptime_re.search(stripped):
            module_counts["comptime"] += 1
            if in_func:
                cur_func_counts["comptime"] = cur_func_counts.get("comptime", 0) + 1

        if error_union_re.search(stripped):
            module_counts["error_union"] += 1
            if in_func:
                cur_func_counts["error_union"] = cur_func_counts.get("error_union", 0) + 1

        if test_re.match(stripped):
            module_counts["test_block"] += 1

        # Findings
        if panic_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="panic_call",
                message="@panic() found — consider returning an error instead.",
                line=idx, symbol="@panic",
            ))
        if unreachable_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="unreachable",
                message="'unreachable' found — will crash at runtime if reached.",
                line=idx, symbol="unreachable",
            ))

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_zig_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Strengths
    if module_counts["defer"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="defer_cleanup",
            message=f"{module_counts['defer']} defer/errdefer statement(s) for resource cleanup.",
            line=0, symbol="defer",
        ))
    if module_counts["comptime"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="comptime_evaluation",
            message=f"{module_counts['comptime']} comptime usage(s) for compile-time evaluation.",
            line=0, symbol="comptime",
        ))
    if module_counts["error_union"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="error_unions",
            message=f"{module_counts['error_union']} error union(s) for explicit error handling.",
            line=0, symbol="error_union",
        ))

    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "zig",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "zig"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


# ===================================================================
# 2. C++ analyzer
# ===================================================================

def _build_cpp_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    return _build_func_cluster(name, start, end, counts, event_lines, full_text, "cpp")


def _diagnose_cpp(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose C++ source via regex pattern matching."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "switch": 0, "template": 0,
        "class_def": 0, "destructor": 0, "virtual": 0, "throw": 0,
    }

    # Function: return_type name( or Class::method(
    func_re = re.compile(
        r"^\s*(?:(?:static|inline|virtual|explicit|constexpr|const|unsigned|signed|long|short)\s+)*"
        r"(?:\w[\w:<>*&\s]*?)\s+(\w+)\s*\("
    )
    method_re = re.compile(r"^\s*(?:\w[\w:<>*&\s]*?)\s+(\w+::\w+)\s*\(")
    assign_re = re.compile(r"(?:[^=!<>]=[^=])|(?:\w+\s*(?:\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=))")
    call_re = re.compile(r"\w+\s*\(")
    if_re = re.compile(r"^\s*(?:if|else\s+if)\s*\(")
    loop_re = re.compile(r"^\s*(?:for|while|do)\s*[\({]")
    return_re = re.compile(r"^\s*return\s")
    switch_re = re.compile(r"^\s*switch\s*\(")
    template_re = re.compile(r"^\s*template\s*<")
    class_re = re.compile(r"^\s*(?:class|struct)\s+(\w+)")
    destructor_re = re.compile(r"~\w+\s*\(")
    virtual_re = re.compile(r"\bvirtual\b")
    throw_re = re.compile(r"\bthrow\b")
    new_re = re.compile(r"\bnew\s+\w")
    delete_re = re.compile(r"\bdelete\s")
    goto_re = re.compile(r"\bgoto\s+\w")
    reinterpret_re = re.compile(r"\breinterpret_cast\b")
    smart_ptr_re = re.compile(r"\b(?:unique_ptr|shared_ptr|make_unique|make_shared)\b")
    const_re = re.compile(r"\bconst\b")
    override_re = re.compile(r"\boverride\b")

    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False

    has_smart_ptr = False
    has_const = False
    has_override = False
    has_destructor = False

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
            continue

        # Detect function boundaries
        fm = method_re.match(stripped) or func_re.match(stripped)
        if fm and "{" in stripped:
            if in_func and cur_func_name:
                func_clusters.append(_build_cpp_cluster(
                    cur_func_name, cur_func_start, idx - 1,
                    cur_func_counts, cur_func_events, text,
                ))
            cur_func_name = fm.group(1)
            cur_func_start = idx
            cur_func_counts = {k: 0 for k in module_counts}
            cur_func_events = []
            in_func = True

        # Count patterns
        if assign_re.search(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if call_re.search(stripped) and not stripped.startswith("if") and not stripped.startswith("for"):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if loop_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if switch_re.match(stripped):
            module_counts["switch"] += 1
            if in_func:
                cur_func_counts["switch"] = cur_func_counts.get("switch", 0) + 1

        if template_re.match(stripped):
            module_counts["template"] += 1
            if in_func:
                cur_func_counts["template"] = cur_func_counts.get("template", 0) + 1

        if class_re.match(stripped):
            module_counts["class_def"] += 1

        if destructor_re.search(stripped):
            module_counts["destructor"] += 1
            has_destructor = True

        if virtual_re.search(stripped):
            module_counts["virtual"] += 1

        if throw_re.search(stripped):
            module_counts["throw"] += 1
            if in_func:
                cur_func_counts["throw"] = cur_func_counts.get("throw", 0) + 1

        # Findings
        if new_re.search(stripped) and not smart_ptr_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="raw_new",
                message="Raw 'new' detected — prefer smart pointers (unique_ptr/shared_ptr).",
                line=idx, symbol="new",
            ))
        if delete_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="raw_delete",
                message="Raw 'delete' detected — prefer smart pointers for automatic cleanup.",
                line=idx, symbol="delete",
            ))
        if goto_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="goto_usage",
                message="'goto' found — consider structured control flow.",
                line=idx, symbol="goto",
            ))
        if reinterpret_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="reinterpret_cast",
                message="reinterpret_cast found — bypasses type safety.",
                line=idx, symbol="reinterpret_cast",
            ))

        # Track strengths
        if smart_ptr_re.search(stripped):
            has_smart_ptr = True
        if const_re.search(stripped):
            has_const = True
        if override_re.search(stripped):
            has_override = True

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_cpp_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Strengths
    if has_destructor:
        strengths.append(DiagnosticItem(
            severity="info", kind="raii_destructor",
            message="Destructor(s) found — RAII pattern for resource management.",
            line=0, symbol="destructor",
        ))
    if has_smart_ptr:
        strengths.append(DiagnosticItem(
            severity="info", kind="smart_pointers",
            message="Smart pointers (unique_ptr/shared_ptr) used for memory safety.",
            line=0, symbol="smart_ptr",
        ))
    if has_const:
        strengths.append(DiagnosticItem(
            severity="info", kind="const_usage",
            message="'const' qualifiers used for immutability.",
            line=0, symbol="const",
        ))
    if has_override:
        strengths.append(DiagnosticItem(
            severity="info", kind="override_usage",
            message="'override' keyword used for virtual method safety.",
            line=0, symbol="override",
        ))

    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "cpp",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "cpp"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


# ===================================================================
# 3. YAML analyzer
# ===================================================================

def _diagnose_yaml(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose YAML files via regex pattern matching."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    counts: dict[str, int] = {
        "mapping": 0, "sequence": 0, "scalar": 0,
        "anchor": 0, "comment": 0, "nested_depth": 0,
    }

    anchor_re = re.compile(r"&(\w+)")
    alias_re = re.compile(r"\*(\w+)")
    comment_re = re.compile(r"^\s*#|(?<=\s)#")
    mapping_re = re.compile(r"^\s*[\w\-_.]+\s*:")
    sequence_re = re.compile(r"^\s*-\s")

    anchors_defined: set[str] = set()
    aliases_used: set[str] = set()
    max_depth = 0

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        # Calculate indentation depth (assume 2-space indent)
        indent = len(line) - len(line.lstrip())
        depth = indent // 2
        if depth > max_depth:
            max_depth = depth

        if comment_re.search(line):
            counts["comment"] += 1

        if mapping_re.match(line):
            counts["mapping"] += 1
            event_lines.append(idx)

        if sequence_re.match(line):
            counts["sequence"] += 1
            event_lines.append(idx)

        # Scalar: a line that is a mapping value (after :) or sequence item
        stripped = _strip_line(line.strip(), _suffix)
        if ":" in stripped and not stripped.endswith(":") and not stripped.startswith("#"):
            counts["scalar"] += 1

        am = anchor_re.search(line)
        if am:
            counts["anchor"] += 1
            anchors_defined.add(am.group(1))

        alm = alias_re.search(line)
        if alm:
            aliases_used.add(alm.group(1))

        # Very long lines
        if len(line) > 200:
            findings.append(DiagnosticItem(
                severity="warning", kind="long_line",
                message=f"Line exceeds 200 characters ({len(line)} chars) — may reduce readability.",
                line=idx, symbol="long_line",
            ))

    counts["nested_depth"] = max_depth

    # Findings
    if max_depth > 6:
        findings.append(DiagnosticItem(
            severity="warning", kind="deep_nesting",
            message=f"Deeply nested structure (depth {max_depth} > 6) — consider flattening.",
            line=0, symbol="nesting",
        ))

    orphan_anchors = anchors_defined - aliases_used
    if orphan_anchors:
        findings.append(DiagnosticItem(
            severity="warning", kind="orphan_anchors",
            message=f"Anchor(s) defined but never referenced: {', '.join(sorted(orphan_anchors))}.",
            line=0, symbol="anchor",
        ))

    # Strengths
    if counts["comment"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="comments_present",
            message=f"{counts['comment']} comment(s) found — improves maintainability.",
            line=0, symbol="comment",
        ))
    if anchors_defined and aliases_used:
        strengths.append(DiagnosticItem(
            severity="info", kind="anchors_for_dry",
            message="YAML anchors and aliases used for DRY configuration.",
            line=0, symbol="anchor",
        ))

    entropy = _shannon_entropy(counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "yaml",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(counts),
                "logic_labels": _logic_labels("<module>", text, counts, "yaml"),
                "counts": counts,
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
        ],
    }


# ===================================================================
# 4. SQL analyzer
# ===================================================================

def _diagnose_sql(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose SQL files via regex pattern matching."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    counts: dict[str, int] = {
        "select": 0, "insert": 0, "update": 0, "delete": 0,
        "join": 0, "subquery": 0, "create": 0, "alter": 0,
        "drop": 0, "transaction": 0,
    }

    select_re = re.compile(r"\bSELECT\b", re.IGNORECASE)
    select_star_re = re.compile(r"\bSELECT\s+\*", re.IGNORECASE)
    insert_re = re.compile(r"\bINSERT\s+INTO\b", re.IGNORECASE)
    update_re = re.compile(r"\bUPDATE\b", re.IGNORECASE)
    delete_re = re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE)
    join_re = re.compile(r"\bJOIN\b", re.IGNORECASE)
    create_re = re.compile(r"\bCREATE\s+(?:TABLE|INDEX|VIEW|FUNCTION|PROCEDURE)\b", re.IGNORECASE)
    alter_re = re.compile(r"\bALTER\s+TABLE\b", re.IGNORECASE)
    drop_re = re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE)
    transaction_re = re.compile(r"\b(?:BEGIN|COMMIT|ROLLBACK|START\s+TRANSACTION)\b", re.IGNORECASE)
    param_re = re.compile(r"\$\d+|\?")
    where_re = re.compile(r"\bWHERE\b", re.IGNORECASE)

    # Track subquery nesting
    full_text_upper = text.upper()
    max_subquery_depth = 0

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("--"):
            continue

        if select_re.search(stripped):
            counts["select"] += 1
            event_lines.append(idx)
        if insert_re.search(stripped):
            counts["insert"] += 1
            event_lines.append(idx)
        if update_re.search(stripped):
            counts["update"] += 1
            event_lines.append(idx)
        if delete_re.search(stripped):
            counts["delete"] += 1
            event_lines.append(idx)
        if join_re.search(stripped):
            counts["join"] += 1
            event_lines.append(idx)
        if create_re.search(stripped):
            counts["create"] += 1
            event_lines.append(idx)
        if alter_re.search(stripped):
            counts["alter"] += 1
            event_lines.append(idx)
        if drop_re.search(stripped):
            counts["drop"] += 1
            event_lines.append(idx)
        if transaction_re.search(stripped):
            counts["transaction"] += 1

        # Findings
        if select_star_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="select_star",
                message="SELECT * found — prefer explicit column lists for clarity and performance.",
                line=idx, symbol="SELECT *",
            ))
        if drop_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="drop_table",
                message="DROP TABLE found — destructive operation, ensure this is intentional.",
                line=idx, symbol="DROP TABLE",
            ))
        if delete_re.search(stripped) and not where_re.search(stripped):
            # Check if WHERE is on the next few lines (multi-line DELETE)
            context = "\n".join(lines[idx - 1:min(idx + 3, len(lines))])
            if not where_re.search(context):
                findings.append(DiagnosticItem(
                    severity="warning", kind="delete_without_where",
                    message="DELETE without WHERE clause — will delete all rows.",
                    line=idx, symbol="DELETE",
                ))

    # Count subqueries via parenthesized SELECT
    subquery_re = re.compile(r"\(\s*SELECT\b", re.IGNORECASE)
    for m in subquery_re.finditer(text):
        counts["subquery"] += 1
        # Count nesting depth at this position
        prefix = text[:m.start()]
        depth = prefix.count("(") - prefix.count(")")
        if depth > max_subquery_depth:
            max_subquery_depth = depth

    if max_subquery_depth > 2:
        findings.append(DiagnosticItem(
            severity="warning", kind="deep_subquery",
            message=f"Nested subqueries detected (depth > 2) — consider CTEs or temp tables.",
            line=0, symbol="subquery",
        ))

    # Strengths
    has_explicit_cols = False
    explicit_col_re = re.compile(r"\bSELECT\s+(?![\s*])\w", re.IGNORECASE)
    if explicit_col_re.search(text):
        has_explicit_cols = True
    if has_explicit_cols and counts["select"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="explicit_columns",
            message="Explicit column lists used in SELECT statements.",
            line=0, symbol="SELECT",
        ))
    if counts["transaction"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="transactions",
            message=f"{counts['transaction']} transaction control statement(s) for data integrity.",
            line=0, symbol="transaction",
        ))
    if param_re.search(text):
        strengths.append(DiagnosticItem(
            severity="info", kind="parameterized_queries",
            message="Parameterized queries ($N or ?) used — helps prevent SQL injection.",
            line=0, symbol="parameter",
        ))

    entropy = _shannon_entropy(counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "sql",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(counts),
                "logic_labels": _logic_labels("<module>", text, counts, "sql"),
                "counts": counts,
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
        ],
    }


# ===================================================================
# 5. Dockerfile analyzer
# ===================================================================

def _diagnose_dockerfile(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose Dockerfile via regex pattern matching."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    counts: dict[str, int] = {
        "from_layer": 0, "run": 0, "copy": 0, "env": 0,
        "expose": 0, "cmd": 0, "entrypoint": 0, "arg": 0,
        "label": 0, "healthcheck": 0,
    }

    from_re = re.compile(r"^\s*FROM\s+", re.IGNORECASE)
    from_tag_re = re.compile(r"^\s*FROM\s+(\S+)", re.IGNORECASE)
    run_re = re.compile(r"^\s*RUN\s+", re.IGNORECASE)
    copy_re = re.compile(r"^\s*(?:COPY|ADD)\s+", re.IGNORECASE)
    env_re = re.compile(r"^\s*ENV\s+", re.IGNORECASE)
    expose_re = re.compile(r"^\s*EXPOSE\s+", re.IGNORECASE)
    cmd_re = re.compile(r"^\s*CMD\s+", re.IGNORECASE)
    entrypoint_re = re.compile(r"^\s*ENTRYPOINT\s+", re.IGNORECASE)
    arg_re = re.compile(r"^\s*ARG\s+", re.IGNORECASE)
    label_re = re.compile(r"^\s*LABEL\s+", re.IGNORECASE)
    healthcheck_re = re.compile(r"^\s*HEALTHCHECK\s+", re.IGNORECASE)
    curl_bash_re = re.compile(r"curl\s+.*\|\s*(?:ba)?sh", re.IGNORECASE)

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("#"):
            continue

        if from_re.match(stripped):
            counts["from_layer"] += 1
            event_lines.append(idx)
            fm = from_tag_re.match(stripped)
            if fm:
                image = fm.group(1)
                if image.endswith(":latest") or (":" not in image and "@" not in image):
                    findings.append(DiagnosticItem(
                        severity="warning", kind="from_latest",
                        message=f"FROM uses 'latest' or untagged image '{image}' — pin a specific version.",
                        line=idx, symbol=image,
                    ))

        if run_re.match(stripped):
            counts["run"] += 1
            event_lines.append(idx)
            # Check for multiple commands not &&-chained
            run_body = stripped[4:].strip()  # after "RUN "
            if ";" in run_body and "&&" not in run_body:
                findings.append(DiagnosticItem(
                    severity="warning", kind="run_no_chain",
                    message="RUN uses ';' without '&&' — failing commands will be silently ignored.",
                    line=idx, symbol="RUN",
                ))
            if curl_bash_re.search(stripped):
                findings.append(DiagnosticItem(
                    severity="warning", kind="curl_pipe_bash",
                    message="curl|bash pattern detected — downloads and executes untrusted code.",
                    line=idx, symbol="curl|bash",
                ))

        if copy_re.match(stripped):
            counts["copy"] += 1
            event_lines.append(idx)
        if env_re.match(stripped):
            counts["env"] += 1
            event_lines.append(idx)
        if expose_re.match(stripped):
            counts["expose"] += 1
            event_lines.append(idx)
        if cmd_re.match(stripped):
            counts["cmd"] += 1
            event_lines.append(idx)
        if entrypoint_re.match(stripped):
            counts["entrypoint"] += 1
            event_lines.append(idx)
        if arg_re.match(stripped):
            counts["arg"] += 1
            event_lines.append(idx)
        if label_re.match(stripped):
            counts["label"] += 1
            event_lines.append(idx)
        if healthcheck_re.match(stripped):
            counts["healthcheck"] += 1
            event_lines.append(idx)

    # Findings
    if counts["healthcheck"] == 0 and counts["from_layer"] > 0:
        findings.append(DiagnosticItem(
            severity="warning", kind="no_healthcheck",
            message="No HEALTHCHECK instruction — container health cannot be monitored.",
            line=0, symbol="HEALTHCHECK",
        ))

    # Strengths
    if counts["from_layer"] >= 2:
        strengths.append(DiagnosticItem(
            severity="info", kind="multi_stage_build",
            message=f"{counts['from_layer']} FROM stages — multi-stage build reduces final image size.",
            line=0, symbol="FROM",
        ))
    if counts["healthcheck"] > 0:
        strengths.append(DiagnosticItem(
            severity="info", kind="healthcheck_present",
            message="HEALTHCHECK defined — enables container health monitoring.",
            line=0, symbol="HEALTHCHECK",
        ))
    # Check for specific version tags
    has_specific_tag = False
    for line in lines:
        fm = from_tag_re.match(line.strip())
        if fm:
            image = fm.group(1)
            if ":" in image and not image.endswith(":latest"):
                has_specific_tag = True
                break
    if has_specific_tag:
        strengths.append(DiagnosticItem(
            severity="info", kind="specific_version_tags",
            message="Specific version tags used in FROM — reproducible builds.",
            line=0, symbol="FROM",
        ))

    entropy = _shannon_entropy(counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "dockerfile",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(counts),
                "logic_labels": _logic_labels("<module>", text, counts, "dockerfile"),
                "counts": counts,
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
        ],
    }


# ===================================================================
# 6. Ruby analyzer
# ===================================================================

def _build_ruby_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    return _build_func_cluster(name, start, end, counts, event_lines, full_text, "ruby")


def _diagnose_ruby(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose Ruby source via regex pattern matching."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "block": 0, "rescue": 0,
        "class_def": 0, "module_def": 0, "yield": 0,
    }

    func_re = re.compile(r"^\s*def\s+(\w+[!?=]?)")
    assign_re = re.compile(r"\w+\s*(?:=(?!=)|\+=|-=|\*=|/=|%=|\|\|=|&&=)")
    call_re = re.compile(r"\w+[.(]\w")
    if_re = re.compile(r"^\s*(?:if|elsif|unless)\s+")
    loop_re = re.compile(r"^\s*(?:while|until|for|loop)\s")
    return_re = re.compile(r"^\s*return\b")
    block_re = re.compile(r"\bdo\b|\{.*\|")
    rescue_re = re.compile(r"^\s*rescue\b")
    class_re = re.compile(r"^\s*class\s+(\w+)")
    module_re = re.compile(r"^\s*module\s+(\w+)")
    yield_re = re.compile(r"\byield\b")
    eval_re = re.compile(r"\beval\s*\(")
    system_re = re.compile(r"\bsystem\s*\(")
    rescue_exception_re = re.compile(r"rescue\s+Exception\b")
    frozen_re = re.compile(r"#\s*frozen_string_literal:\s*true")
    each_re = re.compile(r"\.\s*(?:each|map|select|reject|reduce|inject)\b")

    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False

    has_frozen = False
    has_blocks = False
    has_yield = False
    has_explicit_rescue = False  # rescue with specific type (not Exception)

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("#"):
            if frozen_re.search(stripped):
                has_frozen = True
            continue

        fm = func_re.match(stripped)
        if fm:
            if in_func and cur_func_name:
                func_clusters.append(_build_ruby_cluster(
                    cur_func_name, cur_func_start, idx - 1,
                    cur_func_counts, cur_func_events, text,
                ))
            cur_func_name = fm.group(1)
            cur_func_start = idx
            cur_func_counts = {k: 0 for k in module_counts}
            cur_func_events = []
            in_func = True
            continue

        if assign_re.search(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if call_re.search(stripped) or each_re.search(stripped):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if loop_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if block_re.search(stripped):
            module_counts["block"] += 1
            has_blocks = True
            if in_func:
                cur_func_counts["block"] = cur_func_counts.get("block", 0) + 1

        if rescue_re.match(stripped):
            module_counts["rescue"] += 1
            if in_func:
                cur_func_counts["rescue"] = cur_func_counts.get("rescue", 0) + 1
            # Check for overly broad rescue
            if rescue_exception_re.search(stripped):
                findings.append(DiagnosticItem(
                    severity="warning", kind="rescue_exception",
                    message="'rescue Exception' is too broad — catches system exceptions like SignalException.",
                    line=idx, symbol="rescue",
                ))
            elif re.search(r"rescue\s+\w+", stripped):
                has_explicit_rescue = True

        if class_re.match(stripped):
            module_counts["class_def"] += 1

        if module_re.match(stripped):
            module_counts["module_def"] += 1

        if yield_re.search(stripped):
            module_counts["yield"] += 1
            has_yield = True
            if in_func:
                cur_func_counts["yield"] = cur_func_counts.get("yield", 0) + 1

        # Findings
        if eval_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="eval_usage",
                message="eval() found — executes arbitrary code, potential security risk.",
                line=idx, symbol="eval",
            ))
        if system_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="system_call",
                message="system() found — ensure input is validated to prevent command injection.",
                line=idx, symbol="system",
            ))

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_ruby_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Strengths
    if has_blocks or has_yield:
        strengths.append(DiagnosticItem(
            severity="info", kind="blocks_and_yield",
            message="Blocks and/or yield used — idiomatic Ruby for iteration and callbacks.",
            line=0, symbol="block",
        ))
    if has_explicit_rescue:
        strengths.append(DiagnosticItem(
            severity="info", kind="explicit_rescue_types",
            message="Explicit rescue types used — targeted error handling.",
            line=0, symbol="rescue",
        ))
    if has_frozen:
        strengths.append(DiagnosticItem(
            severity="info", kind="frozen_string_literal",
            message="frozen_string_literal pragma — reduces object allocations.",
            line=0, symbol="frozen_string_literal",
        ))

    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "ruby",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "ruby"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


# ===================================================================
# 7. Lua analyzer
# ===================================================================

def _build_lua_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    return _build_func_cluster(name, start, end, counts, event_lines, full_text, "lua")


def _diagnose_lua(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose Lua source via regex pattern matching."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "local": 0, "function_def": 0,
        "table_constructor": 0, "coroutine": 0,
    }

    func_re = re.compile(r"^\s*(?:local\s+)?function\s+(\w[\w.]*)\s*\(")
    assign_re = re.compile(r"\w+\s*=(?!=)")
    local_re = re.compile(r"^\s*local\s+\w")
    call_re = re.compile(r"\w+\s*[\(:]")
    if_re = re.compile(r"^\s*(?:if|elseif)\s+")
    loop_re = re.compile(r"^\s*(?:for|while|repeat)\b")
    return_re = re.compile(r"^\s*return\b")
    table_re = re.compile(r"\{")
    coroutine_re = re.compile(r"\bcoroutine\.\w+")
    loadstring_re = re.compile(r"\b(?:loadstring|dofile|loadfile)\s*\(")
    metatable_re = re.compile(r"\bsetmetatable\b|\bgetmetatable\b|__index|__newindex|__call|__add|__mul")

    # Track globals: assignments without 'local' at file scope
    global_assigns: list[tuple[str, int]] = []
    global_re = re.compile(r"^(\w+)\s*=(?!=)")  # no indent, no local

    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False

    has_coroutine = False
    has_metatable = False
    has_local_scoping = False

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("--"):
            continue

        fm = func_re.match(stripped)
        if fm:
            if in_func and cur_func_name:
                func_clusters.append(_build_lua_cluster(
                    cur_func_name, cur_func_start, idx - 1,
                    cur_func_counts, cur_func_events, text,
                ))
            cur_func_name = fm.group(1)
            cur_func_start = idx
            cur_func_counts = {k: 0 for k in module_counts}
            cur_func_events = []
            in_func = True
            module_counts["function_def"] += 1
            continue

        if local_re.match(stripped):
            module_counts["local"] += 1
            has_local_scoping = True
            if in_func:
                cur_func_counts["local"] = cur_func_counts.get("local", 0) + 1

        if assign_re.search(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

            # Check for global assignment (no local, no indentation)
            gm = global_re.match(line)
            if gm and not local_re.match(stripped) and not in_func:
                var_name = gm.group(1)
                if var_name not in ("_G", "_ENV", "_VERSION"):
                    global_assigns.append((var_name, idx))

        if call_re.search(stripped):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if loop_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if table_re.search(stripped):
            module_counts["table_constructor"] += 1
            if in_func:
                cur_func_counts["table_constructor"] = cur_func_counts.get("table_constructor", 0) + 1

        if coroutine_re.search(stripped):
            module_counts["coroutine"] += 1
            has_coroutine = True
            if in_func:
                cur_func_counts["coroutine"] = cur_func_counts.get("coroutine", 0) + 1

        if metatable_re.search(stripped):
            has_metatable = True

        # Findings
        if loadstring_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="dynamic_code_loading",
                message="loadstring/dofile/loadfile found — dynamic code loading can be a security risk.",
                line=idx, symbol="loadstring",
            ))

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_lua_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Global variable findings
    if global_assigns:
        for var_name, line_no in global_assigns[:5]:  # cap at 5 to avoid flooding
            findings.append(DiagnosticItem(
                severity="warning", kind="global_variable",
                message=f"Global variable '{var_name}' — prefer 'local' for scoping.",
                line=line_no, symbol=var_name,
            ))

    # Strengths
    if has_local_scoping:
        strengths.append(DiagnosticItem(
            severity="info", kind="local_scoping",
            message="'local' keyword used for variable scoping — reduces global namespace pollution.",
            line=0, symbol="local",
        ))
    if has_coroutine:
        strengths.append(DiagnosticItem(
            severity="info", kind="coroutine_usage",
            message="Coroutine usage detected — cooperative multitasking pattern.",
            line=0, symbol="coroutine",
        ))
    if has_metatable:
        strengths.append(DiagnosticItem(
            severity="info", kind="metatable_usage",
            message="Metatables used — enables OOP and operator overloading patterns.",
            line=0, symbol="metatable",
        ))

    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "lua",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "lua"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


def _build_csharp_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    return _build_func_cluster(name, start, end, counts, event_lines, full_text, "csharp")


def _diagnose_csharp(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose C# source via regex pattern matching."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "switch": 0, "try_catch": 0, "throw": 0,
        "using_stmt": 0, "async_await": 0, "linq": 0,
        "property": 0, "event": 0,
    }

    # Regex patterns for C#
    func_re = re.compile(
        r"^\s*(?:public|private|protected|internal|static|virtual|override|abstract|async|sealed|partial|\s)*"
        r"\s+\w[\w<>\[\],\s\?]*\s+(\w+)\s*\("
    )
    assign_re = re.compile(r"\w+\s*(?:=(?!=)|\+=|-=|\*=|/=|%=|\?\?=)")
    call_re = re.compile(r"\w+\s*[.(]\w")
    if_re = re.compile(r"^\s*(?:if|else\s+if)\s*\(")
    loop_re = re.compile(r"^\s*(?:for|foreach|while|do)\s*[\({]")
    return_re = re.compile(r"^\s*return\b")
    switch_re = re.compile(r"^\s*switch\s*\(")
    try_re = re.compile(r"^\s*try\s*\{?")
    catch_re = re.compile(r"^\s*catch\b")
    throw_re = re.compile(r"^\s*throw\b")
    using_re = re.compile(r"^\s*using\s*\(")
    async_re = re.compile(r"\basync\b|\bawait\b")
    linq_re = re.compile(r"\.\s*(?:Select|Where|OrderBy|GroupBy|Any|All|First|Last|Count|Sum|Average|Aggregate|Join|Distinct|Skip|Take)\s*\(")
    property_re = re.compile(r"^\s*(?:public|private|protected|internal)\s+\w[\w<>\[\],\s\?]*\s+\w+\s*\{\s*(?:get|set)")
    event_re = re.compile(r"^\s*(?:public|private|protected|internal)?\s*event\s+")

    # Finding patterns
    empty_catch_re = re.compile(r"catch\s*(?:\([^)]*\))?\s*\{\s*\}")
    thread_sleep_re = re.compile(r"\bThread\.Sleep\b")
    dynamic_re = re.compile(r"\bdynamic\b")
    gc_collect_re = re.compile(r"\bGC\.Collect\s*\(")

    # Strength patterns
    nullable_re = re.compile(r"\w+\?\s+\w+")  # nullable annotations

    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False

    has_using = False
    has_async = False
    has_linq = False
    has_nullable = False
    is_test_file = bool(re.search(r"[Tt]est|[Ss]pec", str(path)))

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("//"):
            continue

        # Detect function boundaries
        fm = func_re.match(stripped)
        if fm:
            if in_func and cur_func_name:
                func_clusters.append(_build_csharp_cluster(
                    cur_func_name, cur_func_start, idx - 1,
                    cur_func_counts, cur_func_events, text,
                ))
            cur_func_name = fm.group(1)
            cur_func_start = idx
            cur_func_counts = {k: 0 for k in module_counts}
            cur_func_events = []
            in_func = True
            continue

        # Count patterns
        if assign_re.search(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if call_re.search(stripped):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if loop_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if switch_re.match(stripped):
            module_counts["switch"] += 1
            if in_func:
                cur_func_counts["switch"] = cur_func_counts.get("switch", 0) + 1

        if try_re.match(stripped):
            module_counts["try_catch"] += 1
            if in_func:
                cur_func_counts["try_catch"] = cur_func_counts.get("try_catch", 0) + 1

        if throw_re.match(stripped):
            module_counts["throw"] += 1
            if in_func:
                cur_func_counts["throw"] = cur_func_counts.get("throw", 0) + 1

        if using_re.match(stripped):
            module_counts["using_stmt"] += 1
            has_using = True
            if in_func:
                cur_func_counts["using_stmt"] = cur_func_counts.get("using_stmt", 0) + 1

        if async_re.search(stripped):
            module_counts["async_await"] += 1
            has_async = True
            if in_func:
                cur_func_counts["async_await"] = cur_func_counts.get("async_await", 0) + 1

        if linq_re.search(stripped):
            module_counts["linq"] += 1
            has_linq = True
            if in_func:
                cur_func_counts["linq"] = cur_func_counts.get("linq", 0) + 1

        if property_re.match(stripped):
            module_counts["property"] += 1
            if in_func:
                cur_func_counts["property"] = cur_func_counts.get("property", 0) + 1

        if event_re.match(stripped):
            module_counts["event"] += 1
            if in_func:
                cur_func_counts["event"] = cur_func_counts.get("event", 0) + 1

        if nullable_re.search(stripped):
            has_nullable = True

        # Findings
        if empty_catch_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="empty_catch",
                message="Empty catch block swallows exception — at minimum log the error.",
                line=idx, symbol="catch",
            ))

        if thread_sleep_re.search(stripped) and not is_test_file:
            findings.append(DiagnosticItem(
                severity="warning", kind="thread_sleep",
                message="Thread.Sleep in non-test code — consider async Task.Delay or timer.",
                line=idx, symbol="Thread.Sleep",
            ))

        if dynamic_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="info", kind="dynamic_keyword",
                message="'dynamic' keyword bypasses compile-time type checking.",
                line=idx, symbol="dynamic",
            ))

        if gc_collect_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="gc_collect",
                message="GC.Collect() — premature optimization; CLR manages garbage collection.",
                line=idx, symbol="GC.Collect",
            ))

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_csharp_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Strengths
    if has_using:
        strengths.append(DiagnosticItem(
            severity="info", kind="using_statements",
            message=f"{module_counts['using_stmt']} using statement(s) — proper IDisposable resource management.",
            line=0, symbol="using",
        ))
    if has_async:
        strengths.append(DiagnosticItem(
            severity="info", kind="async_await",
            message=f"{module_counts['async_await']} async/await usage(s) — non-blocking asynchronous code.",
            line=0, symbol="async",
        ))
    if has_linq:
        strengths.append(DiagnosticItem(
            severity="info", kind="linq_queries",
            message=f"{module_counts['linq']} LINQ query operation(s) — declarative data manipulation.",
            line=0, symbol="linq",
        ))
    if has_nullable:
        strengths.append(DiagnosticItem(
            severity="info", kind="nullable_annotations",
            message="Nullable type annotations found — explicit null safety.",
            line=0, symbol="nullable",
        ))

    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "csharp",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "csharp"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


# ===================================================================
# 2. PHP analyzer
# ===================================================================

def _build_php_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    return _build_func_cluster(name, start, end, counts, event_lines, full_text, "php")


def _diagnose_php(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose PHP source via regex pattern matching."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "switch": 0, "try_catch": 0,
        "class_def": 0, "trait": 0, "namespace": 0, "use_stmt": 0,
    }

    # Regex patterns for PHP
    func_re = re.compile(
        r"^\s*(?:public|private|protected|static|\s)*function\s+(\w+)\s*\("
    )
    assign_re = re.compile(r"\$\w+\s*(?:=(?!=)|\+=|-=|\.=|\*=|/=|%=|\?\?=)")
    call_re = re.compile(r"\w+\s*\(")
    if_re = re.compile(r"^\s*(?:if|elseif|else\s+if)\s*\(")
    loop_re = re.compile(r"^\s*(?:for|foreach|while|do)\s*[\({]")
    return_re = re.compile(r"^\s*return\b")
    switch_re = re.compile(r"^\s*switch\s*\(")
    try_re = re.compile(r"^\s*try\s*\{?")
    catch_re = re.compile(r"^\s*catch\s*\(")
    class_re = re.compile(r"^\s*(?:abstract\s+|final\s+)?class\s+(\w+)")
    trait_re = re.compile(r"^\s*trait\s+(\w+)")
    namespace_re = re.compile(r"^\s*namespace\s+[\w\\]+")
    use_re = re.compile(r"^\s*use\s+[\w\\]+")

    # Finding patterns
    eval_re = re.compile(r"\beval\s*\(")
    exec_re = re.compile(r"\b(?:exec|system|passthru|shell_exec|popen|proc_open)\s*\(")
    extract_re = re.compile(r"\bextract\s*\(")
    mysql_re = re.compile(r"\bmysql_\w+\s*\(")
    varvar_re = re.compile(r"\$\$\w+")

    # Strength patterns
    type_decl_re = re.compile(
        r"(?:public|private|protected|static)?\s*function\s+\w+\s*\([^)]*(?:int|string|float|bool|array|object|callable|iterable)\s+\$"
    )
    return_type_re = re.compile(r"\)\s*:\s*(?:int|string|float|bool|array|object|void|self|static|mixed|never)\b")
    pdo_re = re.compile(r"\bPDO\b|\bnew\s+PDO\b|\$\w+->prepare\s*\(")

    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False

    has_namespace = False
    has_try_catch = False
    has_type_decl = False
    has_pdo = False

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("//") or stripped.startswith("#"):
            continue

        # Detect function boundaries
        fm = func_re.match(stripped)
        if fm:
            if in_func and cur_func_name:
                func_clusters.append(_build_php_cluster(
                    cur_func_name, cur_func_start, idx - 1,
                    cur_func_counts, cur_func_events, text,
                ))
            cur_func_name = fm.group(1)
            cur_func_start = idx
            cur_func_counts = {k: 0 for k in module_counts}
            cur_func_events = []
            in_func = True
            continue

        # Count patterns
        if assign_re.search(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if call_re.search(stripped):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if loop_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if switch_re.match(stripped):
            module_counts["switch"] += 1
            if in_func:
                cur_func_counts["switch"] = cur_func_counts.get("switch", 0) + 1

        if try_re.match(stripped):
            module_counts["try_catch"] += 1
            has_try_catch = True
            if in_func:
                cur_func_counts["try_catch"] = cur_func_counts.get("try_catch", 0) + 1

        if class_re.match(stripped):
            module_counts["class_def"] += 1

        if trait_re.match(stripped):
            module_counts["trait"] += 1

        if namespace_re.match(stripped):
            module_counts["namespace"] += 1
            has_namespace = True

        if use_re.match(stripped):
            module_counts["use_stmt"] += 1

        if type_decl_re.search(stripped) or return_type_re.search(stripped):
            has_type_decl = True

        if pdo_re.search(stripped):
            has_pdo = True

        # Findings
        if eval_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="eval_usage",
                message="eval() found — executes arbitrary PHP code, serious security risk.",
                line=idx, symbol="eval",
            ))

        if exec_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="command_injection",
                message="Shell execution function found — validate input to prevent command injection.",
                line=idx, symbol="exec",
            ))

        if extract_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="extract_usage",
                message="extract() imports variables into scope — risk of variable injection.",
                line=idx, symbol="extract",
            ))

        if mysql_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="deprecated_mysql",
                message="mysql_* functions are deprecated — use PDO or mysqli instead.",
                line=idx, symbol="mysql_",
            ))

        if varvar_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="info", kind="variable_variables",
                message="Variable variable ($$var) found — can obscure data flow.",
                line=idx, symbol="$$",
            ))

    # Check for missing namespace
    if not has_namespace and module_counts["class_def"] > 0:
        findings.append(DiagnosticItem(
            severity="info", kind="no_namespace",
            message="Class defined without namespace declaration — risk of naming conflicts.",
            line=0, symbol="namespace",
        ))

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_php_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Strengths
    if has_type_decl:
        strengths.append(DiagnosticItem(
            severity="info", kind="type_declarations",
            message="Type declarations found — improved code reliability and IDE support.",
            line=0, symbol="type_decl",
        ))
    if has_namespace:
        strengths.append(DiagnosticItem(
            severity="info", kind="namespace_usage",
            message="Namespace declarations used — proper code organization.",
            line=0, symbol="namespace",
        ))
    if has_try_catch:
        strengths.append(DiagnosticItem(
            severity="info", kind="try_catch_handling",
            message=f"{module_counts['try_catch']} try/catch block(s) — structured error handling.",
            line=0, symbol="try_catch",
        ))
    if has_pdo:
        strengths.append(DiagnosticItem(
            severity="info", kind="pdo_usage",
            message="PDO usage detected — safe parameterized database access.",
            line=0, symbol="PDO",
        ))

    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "php",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "php"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


# ===================================================================
# 3. BASIC analyzer (VB/VBA/VBScript/FreeBasic)
# ===================================================================

def _build_basic_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    return _build_func_cluster(name, start, end, counts, event_lines, full_text, "basic")


def _diagnose_basic(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose VB/VBA/VBScript/FreeBasic source via regex pattern matching."""
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "select_case": 0, "dim": 0, "redim": 0,
        "on_error": 0, "goto": 0, "class_def": 0,
    }

    # Regex patterns for BASIC (case-insensitive)
    func_re = re.compile(
        r"^\s*(?:Public\s+|Private\s+|Friend\s+|Static\s+)*(?:Sub|Function)\s+(\w+)\s*\(",
        re.IGNORECASE,
    )
    end_func_re = re.compile(r"^\s*End\s+(?:Sub|Function)\b", re.IGNORECASE)
    assign_re = re.compile(r"^\s*(?:Set\s+|Let\s+)?\w+(?:\.\w+)*\s*=(?!=)", re.IGNORECASE)
    call_re = re.compile(r"\b(?:Call\s+)?\w+(?:\.\w+)*\s*\(", re.IGNORECASE)
    if_re = re.compile(r"^\s*(?:If|ElseIf)\s+", re.IGNORECASE)
    loop_re = re.compile(r"^\s*(?:For\s|Do\s|While\s|Loop\s|Wend\b|Next\b)", re.IGNORECASE)
    return_re = re.compile(r"^\s*(?:Exit\s+(?:Sub|Function))\b", re.IGNORECASE)
    select_re = re.compile(r"^\s*Select\s+Case\b", re.IGNORECASE)
    dim_re = re.compile(r"^\s*Dim\s+", re.IGNORECASE)
    redim_re = re.compile(r"^\s*ReDim\s+", re.IGNORECASE)
    on_error_re = re.compile(r"^\s*On\s+Error\s+", re.IGNORECASE)
    goto_re = re.compile(r"\bGoTo\s+\w+", re.IGNORECASE)
    class_re = re.compile(r"^\s*Class\s+(\w+)", re.IGNORECASE)

    # Finding patterns
    on_error_resume_re = re.compile(r"^\s*On\s+Error\s+Resume\s+Next\b", re.IGNORECASE)
    on_error_goto_handler_re = re.compile(r"^\s*On\s+Error\s+GoTo\s+\w+", re.IGNORECASE)
    redim_no_preserve_re = re.compile(r"^\s*ReDim\s+(?!Preserve\b)\w+", re.IGNORECASE)
    createobject_fso_re = re.compile(
        r'CreateObject\s*\(\s*"Scripting\.FileSystemObject"\s*\)', re.IGNORECASE,
    )

    # Strength patterns
    option_explicit_re = re.compile(r"^\s*Option\s+Explicit\b", re.IGNORECASE)

    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False

    has_option_explicit = False
    has_error_handler = False
    has_select_case = False

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("'") or stripped.upper().startswith("REM "):
            continue

        if option_explicit_re.match(stripped):
            has_option_explicit = True
            continue

        # Detect function boundaries
        fm = func_re.match(stripped)
        if fm:
            if in_func and cur_func_name:
                func_clusters.append(_build_basic_cluster(
                    cur_func_name, cur_func_start, idx - 1,
                    cur_func_counts, cur_func_events, text,
                ))
            cur_func_name = fm.group(1)
            cur_func_start = idx
            cur_func_counts = {k: 0 for k in module_counts}
            cur_func_events = []
            in_func = True
            continue

        if end_func_re.match(stripped):
            if in_func and cur_func_name:
                func_clusters.append(_build_basic_cluster(
                    cur_func_name, cur_func_start, idx,
                    cur_func_counts, cur_func_events, text,
                ))
                cur_func_name = ""
                in_func = False
            continue

        # Count patterns
        if assign_re.match(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if call_re.search(stripped):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if loop_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if select_re.match(stripped):
            module_counts["select_case"] += 1
            has_select_case = True
            if in_func:
                cur_func_counts["select_case"] = cur_func_counts.get("select_case", 0) + 1

        if dim_re.match(stripped):
            module_counts["dim"] += 1
            if in_func:
                cur_func_counts["dim"] = cur_func_counts.get("dim", 0) + 1

        if redim_re.match(stripped):
            module_counts["redim"] += 1
            if in_func:
                cur_func_counts["redim"] = cur_func_counts.get("redim", 0) + 1

        if on_error_re.match(stripped):
            module_counts["on_error"] += 1
            if in_func:
                cur_func_counts["on_error"] = cur_func_counts.get("on_error", 0) + 1

        if goto_re.search(stripped) and not on_error_re.match(stripped):
            module_counts["goto"] += 1
            if in_func:
                cur_func_counts["goto"] = cur_func_counts.get("goto", 0) + 1

        if class_re.match(stripped):
            module_counts["class_def"] += 1

        # Findings
        if on_error_resume_re.match(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="on_error_resume_next",
                message="'On Error Resume Next' swallows all errors — use structured error handling.",
                line=idx, symbol="On Error Resume Next",
            ))

        if on_error_goto_handler_re.match(stripped) and not on_error_resume_re.match(stripped):
            has_error_handler = True

        if goto_re.search(stripped) and not on_error_re.match(stripped):
            findings.append(DiagnosticItem(
                severity="info", kind="goto_usage",
                message="GoTo found — consider structured control flow (loops, Select Case).",
                line=idx, symbol="GoTo",
            ))

        if redim_no_preserve_re.match(stripped):
            findings.append(DiagnosticItem(
                severity="info", kind="redim_no_preserve",
                message="ReDim without Preserve — existing array data will be lost.",
                line=idx, symbol="ReDim",
            ))

        if createobject_fso_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="info", kind="fso_createobject",
                message="CreateObject(\"Scripting.FileSystemObject\") — validate file paths before operations.",
                line=idx, symbol="FileSystemObject",
            ))

    # Close last function (if no explicit End Sub/Function)
    if in_func and cur_func_name:
        func_clusters.append(_build_basic_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Strengths
    if has_option_explicit:
        strengths.append(DiagnosticItem(
            severity="info", kind="option_explicit",
            message="Option Explicit — forces variable declaration, prevents typo bugs.",
            line=0, symbol="Option Explicit",
        ))
    if has_error_handler:
        strengths.append(DiagnosticItem(
            severity="info", kind="structured_error_handling",
            message="On Error GoTo handler — structured error handling with labeled handlers.",
            line=0, symbol="On Error GoTo",
        ))
    if has_select_case:
        strengths.append(DiagnosticItem(
            severity="info", kind="select_case_usage",
            message=f"{module_counts['select_case']} Select Case block(s) — structured branching.",
            line=0, symbol="Select Case",
        ))

    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "basic",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "basic"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
        ],
    }


# ===================================================================
# 4. EPL (易语言) analyzer
# ===================================================================

def _build_epl_cluster(
    name: str, start: int, end: int,
    counts: dict[str, int], event_lines: list[int], full_text: str,
) -> dict[str, object]:
    return _build_func_cluster(name, start, end, counts, event_lines, full_text, "epl")


def _diagnose_epl(path: str | Path, text: str) -> dict[str, object]:
    """Diagnose EPL (易语言/Easy Programming Language) source via regex pattern matching.

    EPL source files use Chinese keywords. This analyzer matches those keywords
    to extract control flow and data flow patterns.
    """
    findings: list[DiagnosticItem] = []
    strengths: list[DiagnosticItem] = []
    lines = text.splitlines()
    _suffix = Path(path).suffix.lower()
    event_lines: list[int] = []

    func_clusters: list[dict[str, object]] = []
    module_counts: dict[str, int] = {
        "assign": 0, "call": 0, "if": 0, "loop": 0,
        "return": 0, "select": 0, "variable_def": 0,
        "event_handler": 0, "dll_import": 0, "api_call": 0,
    }

    # Regex patterns for EPL (Chinese keywords)
    # .子程序 marks function/subroutine definition
    func_re = re.compile(r"^\s*\.子程序\s+(\S+)")
    # .局部变量 / .全局变量 / .参数
    local_var_re = re.compile(r"^\s*\.局部变量\s+")
    global_var_re = re.compile(r"^\s*\.全局变量\s+")
    param_re = re.compile(r"^\s*\.参数\s+")
    # Control flow
    if_re = re.compile(r"^\s*(?:如果|如果真)\s*[\(（]")
    loop_re = re.compile(r"^\s*(?:循环|计次循环|判断循环首|变量循环首)\s*[\(（]?")
    return_re = re.compile(r"^\s*返回\s*[\(（]?")
    select_re = re.compile(r"^\s*判断\s*[\(（]?")
    # Assignment: variable = value
    assign_re = re.compile(r"^\s*\S+\s*[＝=]\s*")
    # Function/method calls: name( or name（
    call_re = re.compile(r"\S+\s*[\(（]")
    # Event handler: _事件名
    event_handler_re = re.compile(r"^\s*\.子程序\s+\S+_\S+(?:事件|被单击|被选择|被改变|按下)")
    # DLL import
    dll_import_re = re.compile(r"^\s*\.DLL命令\s+|^\s*DLL命令\s+")
    # API / shell calls
    api_call_re = re.compile(r"取反|取绝对值|取整|取余数|位与|位或|位异或|位取反")
    ui_call_re = re.compile(r"信息框|输入框|文件对话框|颜色对话框")
    # Finding patterns
    shell_exec_re = re.compile(r"执行\s*[\(（]|运行\s*[\(（]")
    # Version marker
    version_re = re.compile(r"^\s*\.版本\s+")

    cur_func_name = ""
    cur_func_start = 0
    cur_func_counts: dict[str, int] = {}
    cur_func_events: list[int] = []
    in_func = False

    has_local_vars = False
    has_structured_subs = False
    dll_import_lines: list[int] = []

    for idx, line in enumerate(lines, start=1):
        stripped = _strip_line(line.strip(), _suffix)
        if not stripped or stripped.startswith("'"):
            continue

        # Detect function boundaries (.子程序)
        fm = func_re.match(stripped)
        if fm:
            if in_func and cur_func_name:
                func_clusters.append(_build_epl_cluster(
                    cur_func_name, cur_func_start, idx - 1,
                    cur_func_counts, cur_func_events, text,
                ))
            cur_func_name = fm.group(1)
            cur_func_start = idx
            cur_func_counts = {k: 0 for k in module_counts}
            cur_func_events = []
            in_func = True
            has_structured_subs = True

            # Check for event handler pattern
            if event_handler_re.match(stripped):
                module_counts["event_handler"] += 1
                if in_func:
                    cur_func_counts["event_handler"] = cur_func_counts.get("event_handler", 0) + 1
            continue

        # Variable declarations
        if local_var_re.match(stripped) or param_re.match(stripped):
            module_counts["variable_def"] += 1
            has_local_vars = True
            if in_func:
                cur_func_counts["variable_def"] = cur_func_counts.get("variable_def", 0) + 1
            continue

        if global_var_re.match(stripped):
            module_counts["variable_def"] += 1
            if in_func:
                cur_func_counts["variable_def"] = cur_func_counts.get("variable_def", 0) + 1
            continue

        # DLL command
        if dll_import_re.match(stripped):
            module_counts["dll_import"] += 1
            dll_import_lines.append(idx)
            if in_func:
                cur_func_counts["dll_import"] = cur_func_counts.get("dll_import", 0) + 1
            continue

        # Skip version markers
        if version_re.match(stripped):
            continue

        # Count patterns
        if assign_re.match(stripped) and not func_re.match(stripped):
            module_counts["assign"] += 1
            if in_func:
                cur_func_counts["assign"] = cur_func_counts.get("assign", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if call_re.search(stripped):
            module_counts["call"] += 1
            if in_func:
                cur_func_counts["call"] = cur_func_counts.get("call", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if if_re.match(stripped):
            module_counts["if"] += 1
            if in_func:
                cur_func_counts["if"] = cur_func_counts.get("if", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if loop_re.match(stripped):
            module_counts["loop"] += 1
            if in_func:
                cur_func_counts["loop"] = cur_func_counts.get("loop", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if return_re.match(stripped):
            module_counts["return"] += 1
            if in_func:
                cur_func_counts["return"] = cur_func_counts.get("return", 0) + 1
            event_lines.append(idx)
            if in_func:
                cur_func_events.append(idx)

        if select_re.match(stripped):
            module_counts["select"] += 1
            if in_func:
                cur_func_counts["select"] = cur_func_counts.get("select", 0) + 1

        if api_call_re.search(stripped) or ui_call_re.search(stripped):
            module_counts["api_call"] += 1
            if in_func:
                cur_func_counts["api_call"] = cur_func_counts.get("api_call", 0) + 1

        # Findings
        if shell_exec_re.search(stripped):
            findings.append(DiagnosticItem(
                severity="warning", kind="shell_execution",
                message="执行()/运行() found — shell execution, validate input to prevent injection.",
                line=idx, symbol="执行/运行",
            ))

    # DLL import without error handling check
    if dll_import_lines:
        # Check if there's any error handling near DLL imports
        has_error_near_dll = False
        for dll_line in dll_import_lines:
            nearby_text = "\n".join(lines[max(0, dll_line - 1):min(len(lines), dll_line + 5)])
            if re.search(r"如果|判断|错误", nearby_text):
                has_error_near_dll = True
                break
        if not has_error_near_dll and len(dll_import_lines) > 0:
            findings.append(DiagnosticItem(
                severity="info", kind="dll_no_error_handling",
                message=f"{len(dll_import_lines)} DLL命令 without nearby error handling.",
                line=dll_import_lines[0], symbol="DLL命令",
            ))

    # Close last function
    if in_func and cur_func_name:
        func_clusters.append(_build_epl_cluster(
            cur_func_name, cur_func_start, len(lines),
            cur_func_counts, cur_func_events, text,
        ))

    # Strengths
    if has_structured_subs:
        strengths.append(DiagnosticItem(
            severity="info", kind="structured_subroutines",
            message="Structured .子程序 definitions — modular code organization.",
            line=0, symbol=".子程序",
        ))
    if has_local_vars:
        strengths.append(DiagnosticItem(
            severity="info", kind="explicit_local_variables",
            message=".局部变量 declarations — explicit variable scoping.",
            line=0, symbol=".局部变量",
        ))

    entropy = _shannon_entropy(module_counts)
    n_lines = max(len(lines), 1)

    return {
        "language": "epl",
        "source_file": str(path),
        "findings": [item.to_dict() for item in findings],
        "strengths": [item.to_dict() for item in strengths],
        "entropy_clusters": [
            {
                "scope": "module",
                "name": "<module>",
                "cluster": _entropy_bucket(entropy),
                "entropy": round(entropy, 4),
                "dominant_signal": _dominant_signal(module_counts),
                "logic_labels": _logic_labels("<module>", text, module_counts, "epl"),
                "counts": dict(module_counts),
                "modular_flow": _modular_flow_profile(event_lines, 1, n_lines),
            },
            *func_clusters,
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
    elif suffix == ".go":
        payload = _diagnose_go(source, text)
    elif suffix in {".c", ".h", ".pseudo", ".ppc"}:
        payload = _diagnose_c_pseudo(source, text)
    elif suffix == ".rs":
        payload = _diagnose_rust(source, text)
    elif suffix in {".js", ".jsx", ".mjs"}:
        payload = _diagnose_javascript(source, text)
    elif suffix in {".ts", ".tsx"}:
        payload = _diagnose_typescript(source, text)
    elif suffix == ".java":
        payload = _diagnose_java(source, text)
    elif suffix == ".zig":
        payload = _diagnose_zig(source, text)
    elif suffix in {".cpp", ".cc", ".cxx", ".hpp"}:
        payload = _diagnose_cpp(source, text)
    elif suffix in {".yaml", ".yml"}:
        payload = _diagnose_yaml(source, text)
    elif suffix == ".sql":
        payload = _diagnose_sql(source, text)
    elif suffix == ".rb":
        payload = _diagnose_ruby(source, text)
    elif suffix == ".lua":
        payload = _diagnose_lua(source, text)
    elif source.name.lower() == "dockerfile" or source.name.lower().startswith("dockerfile."):
        payload = _diagnose_dockerfile(source, text)
    elif suffix == ".cs":
        payload = _diagnose_csharp(source, text)
    elif suffix == ".php":
        payload = _diagnose_php(source, text)
    elif suffix in {".bas", ".vb", ".vbs", ".frm"}:
        payload = _diagnose_basic(source, text)
    elif suffix in {".e", ".ec"}:
        payload = _diagnose_epl(source, text)
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
            flow = cluster.get("modular_flow", {})
            flow_text = (
                f" flow={flow.get('assessment','unknown')}"
                f" msn={flow.get('modular_shrinking_number', 0.0):.4f}"
                f" mod_u={flow.get('modular_uniformity', 0.0):.4f}"
                f" topo_u={flow.get('topological_uniformity', 0.0):.4f}"
            )
            lines.append(
                f"- {cluster['scope']} name={cluster['name']}{where} "
                f"cluster={cluster['cluster']} entropy={cluster['entropy']:.4f} "
                f"dominant_signal={cluster['dominant_signal']} labels={labels}{flow_text}"
            )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("modular_flow")
    flowful_clusters = [
        cluster for cluster in clusters
        if isinstance(cluster.get("modular_flow"), dict)
    ]
    if flowful_clusters:
        ranked = sorted(
            flowful_clusters,
            key=lambda cluster: cluster["modular_flow"].get("modular_shrinking_number", 0.0),
            reverse=True,
        )
        for cluster in ranked[:8]:
            flow = cluster["modular_flow"]
            hotspots = ",".join(flow.get("hotspots", [])) or "none"
            where = f" line={cluster['line']}" if cluster.get("line") else ""
            lines.append(
                f"- {cluster['scope']} name={cluster['name']}{where} "
                f"assessment={flow.get('assessment', 'unknown')} "
                f"msn={flow.get('modular_shrinking_number', 0.0):.4f} "
                f"mod_u={flow.get('modular_uniformity', 0.0):.4f} "
                f"topo_u={flow.get('topological_uniformity', 0.0):.4f} "
                f"events={flow.get('event_count', 0)} hotspots={hotspots}"
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
