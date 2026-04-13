from pathlib import Path
import subprocess
import sys


def _env_with_src():
    root = Path(__file__).resolve().parents[1]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(root / "src")
    return env


def test_cli_help():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "--help"],
        cwd=root,
        env=_env_with_src(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "inspect" in result.stdout
    assert "compare" in result.stdout
    assert "ablate" in result.stdout


def test_cli_verbose_inspect():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "-v", "inspect", "examples/sample_trace.json"],
        cwd=root,
        env=_env_with_src(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[verbose]" in result.stdout
    assert "predicted_bottleneck=" in result.stdout


def test_cli_json_inspect():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "inspect", "examples/sample_trace.json", "--json"],
        cwd=root,
        env=_env_with_src(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"summary"' in result.stdout
    assert '"predicted_bottleneck"' in result.stdout


def test_cli_export_json(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "inspect.json"
    subprocess.run(
        [sys.executable, "-m", "tensorearch", "export", "--mode", "inspect", "--left", "examples/sample_trace.json", "--output", str(out), "--json"],
        cwd=root,
        env=_env_with_src(),
        capture_output=True,
        text=True,
        check=True,
    )
    text = out.read_text(encoding="utf-8")
    assert '"summary"' in text


def test_cli_adapt_transformer(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "adapted.json"
    subprocess.run(
        [sys.executable, "-m", "tensorearch", "adapt", "--adapter", "transformer", "--input", "examples/transformer_adapter_input.json", "--output", str(out)],
        cwd=root,
        env=_env_with_src(),
        capture_output=True,
        text=True,
        check=True,
    )
    text = out.read_text(encoding="utf-8")
    assert '"predicted_bottleneck"' in text


def test_cli_space_json(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "sample_model.py"
    src.write_text(
        "\n".join(
            [
                "class MLA: pass",
                "class MoE: pass",
                "kv_cache = None",
                "n_routed_experts = 64",
                "n_shared_experts = 2",
                "def attention(x):",
                "    return x",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "space", "--source-file", str(src), "--json"],
        cwd=root,
        env=_env_with_src(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"quadrupole_projection"' in result.stdout
    assert '"classification"' in result.stdout


def test_cli_diagnose_json(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "score_logic.py"
    src.write_text(
        "\n".join(
            [
                "attention_map = [",
                "    ('seed_turbulence', 'conservatism', 3.0, 'align'),",
                "    ('seed_turbulence', 'aggressiveness', 2.5, 'oppose'),",
                "]",
                "def score(xs):",
                "    score = 0.0",
                "    score += 1.0",
                "    score *= 0.7",
                "    return max(0.0, min(1.0, score))",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "diagnose", "--source-file", str(src), "--json"],
        cwd=root,
        env=_env_with_src(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"findings"' in result.stdout
    assert '"entropy_clusters"' in result.stdout
    assert '"cluster"' in result.stdout
    assert '"logic_labels"' in result.stdout
    assert '"scoring_logic"' in result.stdout
    assert '"conflicting_signal"' in result.stdout
    assert '"score_normalization"' in result.stdout


def test_cli_diagnose_shell(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "score_logic.ps1"
    src.write_text(
        "\n".join(
            [
                "$score = 0",
                "$score = 1",
                "$score = 2",
                "Get-Content x | Select-String y",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "diagnose", "--source-file", str(src), "--json"],
        cwd=root,
        env=_env_with_src(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"language": "shell"' in result.stdout
    assert '"entropy_clusters"' in result.stdout
    assert '"logic_labels"' in result.stdout
    assert '"pipeline_logic"' in result.stdout
    assert '"repeated_overwrite"' in result.stdout
