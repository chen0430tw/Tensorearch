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
