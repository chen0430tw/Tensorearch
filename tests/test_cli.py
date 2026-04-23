import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from tensorearch.forecast import forecast_trace
from tensorearch.io import load_training_trace_from_dict


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
    assert '"space_family_projection"' in result.stdout


def test_cli_space_detects_diffusion_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "diffusion_model.py"
    src.write_text(
        "\n".join(
            [
                "class UNetModel: pass",
                "def get_timestep_embedding(timestep):",
                "    return timestep",
                "class DDIMSampler: pass",
                "def denoise_step(latent, sigma, timestep):",
                "    noise_pred = latent",
                "    return noise_pred",
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
    payload = json.loads(result.stdout)
    assert payload["classification"] == "diffusion-unet dominant"
    assert payload["space_family_projection"]["extended_axes"]["D_diffusion_denoising"] > 0
    assert payload["space_family_projection"]["extended_axes"]["U_multiscale_unet"] > 0


def test_cli_space_detects_adapter_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "adapter_model.py"
    src.write_text(
        "\n".join(
            [
                "class LoRALinear: pass",
                "def merge_adapter(target_modules, rank):",
                "    low_rank = rank",
                "    return low_rank",
                "def apply_lora_adapter(attn_q_proj):",
                "    return attn_q_proj",
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
    payload = json.loads(result.stdout)
    assert payload["classification"] == "adapterization dominant"
    assert payload["space_family_projection"]["extended_axes"]["A_adapterization"] > 0


def test_cli_space_detects_runtime_wrapper_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "runtime_wrapper.py"
    src.write_text(
        "\n".join(
            [
                "def build_blackwell_runtime_config():",
                "    return {'topology': 'nvlink'}",
                "def apply_blackwell_wrapper(kernel, cache):",
                "    quant_tensor = kernel",
                "    return quant_tensor, cache",
                "def export_runtime_bridge():",
                "    return 'bridge'",
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
    payload = json.loads(result.stdout)
    assert payload["classification"] == "runtime-wrapper dominant"
    assert payload["space_family_projection"]["extended_axes"]["R_runtime_wrapper"] > 0


def test_cli_space_detects_video_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "video_model.py"
    src.write_text(
        "\n".join(
            [
                "class UNet3DConditionModel: pass",
                "def process_video_frames(frames, num_frames, motion):",
                "    temporal = num_frames",
                "    return frames, temporal, motion",
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
    payload = json.loads(result.stdout)
    assert payload["classification"] == "video-temporal dominant"
    assert payload["space_family_projection"]["extended_axes"]["V_temporal_video"] > 0


def test_cli_space_detects_audio_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "audio_model.py"
    src.write_text(
        "\n".join(
            [
                "def mel_spectrogram(waveform):",
                "    return waveform",
                "class AudioDiffusion: pass",
                "def denoise_audio(stft, vocoder):",
                "    return vocoder(stft)",
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
    payload = json.loads(result.stdout)
    assert payload["classification"] == "audio-spectral dominant"
    assert payload["space_family_projection"]["extended_axes"]["O_audio_spectral"] > 0


def test_cli_space_detects_3d_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "threed_model.py"
    src.write_text(
        "\n".join(
            [
                "class TriplaneNeRF: pass",
                "def render_mesh(point_cloud, camera_pose):",
                "    radiance_field = point_cloud",
                "    return radiance_field, camera_pose",
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
    payload = json.loads(result.stdout)
    assert payload["classification"] == "3d-generative dominant"
    assert payload["space_family_projection"]["extended_axes"]["G_3d_generative"] > 0


def test_cli_space_detects_speech_language_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "speech_model.py"
    src.write_text(
        "\n".join(
            [
                "class AudioEncoder: pass",
                "class TextDecoder: pass",
                "def transcribe(mel, token_embedding, logits):",
                "    return logits",
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
    payload = json.loads(result.stdout)
    assert payload["classification"] == "speech-language dominant"
    assert payload["space_family_projection"]["extended_axes"]["S_speech_language"] > 0


def test_cli_space_detects_world_model_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "world_model.py"
    src.write_text(
        "\n".join(
            [
                "class MDRNN: pass",
                "def rollout_world_model(actions, latents, reward, done):",
                "    hidden = actions",
                "    return latents, reward, done, hidden",
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
    payload = json.loads(result.stdout)
    assert payload["classification"] == "world-model dominant"
    assert payload["space_family_projection"]["extended_axes"]["M_world_model"] > 0


def test_cli_space_detects_multimodal_alignment_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "vlm_model.py"
    src.write_text(
        "\n".join(
            [
                "class QFormer: pass",
                "def project_image_text(image_embeds, text_embeds, query_tokens):",
                "    return image_embeds, text_embeds, query_tokens",
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
    payload = json.loads(result.stdout)
    assert payload["classification"] == "multimodal-alignment dominant"
    assert payload["space_family_projection"]["extended_axes"]["L_multimodal_alignment"] > 0


def test_cli_space_detects_graph_message_passing_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "graph_model.py"
    src.write_text(
        "\n".join(
            [
                "class GCNConv: pass",
                "def propagate(edge_index, node_features):",
                "    return edge_index, node_features",
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
    payload = json.loads(result.stdout)
    assert payload["classification"] == "graph-message-passing dominant"
    assert payload["space_family_projection"]["extended_axes"]["H_graph_message_passing"] > 0


def test_cli_space_detects_vision_detection_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "vision_model.py"
    src.write_text(
        "\n".join(
            [
                "class MaskRCNN: pass",
                "def detect_masks(pixel_values, boxes, roi):",
                "    return pixel_values, boxes, roi",
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
    payload = json.loads(result.stdout)
    assert payload["classification"] == "vision-detection dominant"
    assert payload["space_family_projection"]["extended_axes"]["I_vision_detection"] > 0


def test_cli_space_detects_bio_sequence_family(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "bio_model.py"
    src.write_text(
        "\n".join(
            [
                "class ESM2:",
                "    num_residues = 512",
                "    amino_vocab = 33",
                "def evoformer(msa, pairwise_rep, contact_map):",
                "    return esm2_model(msa, chain_id=0)",
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
    payload = json.loads(result.stdout)
    assert payload["classification"] == "bio-sequence dominant"
    assert payload["space_family_projection"]["extended_axes"]["B_bio_sequence"] > 0


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


def test_cli_temporal_radio_json(tmp_path):
    root = Path(__file__).resolve().parents[1]
    T, H, W = 12, 6, 6
    u = np.ones((T, H, W), dtype=float)
    v = np.zeros((T, H, W), dtype=float)
    h = np.zeros((T, H, W), dtype=float)
    bg_u = np.ones((H, W), dtype=float)
    bg_v = np.zeros((H, W), dtype=float)
    obs_u = np.zeros((H, W), dtype=float)
    obs_v = np.ones((H, W), dtype=float)
    for t in range(1, T):
        u[t] = u[t - 1]
        v[t] = v[t - 1]
        if t >= 4:
            u[t, :3, :3] = 1.25 * u[t - 1, :3, :3]
        h[t] = h[t - 1]
        h[t] += np.linspace(0.0, 1.0, W)[None, :]
    probe = tmp_path / "radio_probe.npz"
    np.savez(
        probe,
        dt=np.asarray(1.0),
        u=u,
        v=v,
        h=h,
        bg_u=bg_u,
        bg_v=bg_v,
        obs_u=obs_u,
        obs_v=obs_v,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorearch",
            "temporal-radio",
            "--input",
            str(probe),
            "--time-bins",
            "4",
            "--y-bins",
            "2",
            "--x-bins",
            "2",
            "--json",
        ],
        cwd=root,
        env=_env_with_src(),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["version"] == "tradio.v1"
    assert payload["lock"]["field"] == "uv"
    assert payload["lock"]["spatial_window"]["y_bin"] == 0
    assert payload["lock"]["spatial_window"]["x_bin"] == 0
    assert payload["vector_scores"]["background_alignment_mean"] is not None
    assert payload["coupled_scores"]["h_uv_coupling_mean"] is not None


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


def test_cli_diagnose_short_boolean_helper_not_high_entropy(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "helpers.py"
    src.write_text(
        "\n".join(
            [
                "def _is_payload_dict(candidate):",
                "    if not isinstance(candidate, dict):",
                "        return False",
                "    return 'gamma_pcm' in candidate",
                "",
                "def _is_extreme_candidate(cand_features):",
                "    return cand_features.get('aggressiveness', 0.5) > 0.9 or cand_features.get('conservatism', 0.5) > 0.9",
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
    payload = json.loads(result.stdout)
    clusters = {c["name"]: c for c in payload["entropy_clusters"] if c["scope"] == "function"}
    assert clusters["_is_payload_dict"]["cluster"] != "high_entropy"
    assert clusters["_is_extreme_candidate"]["cluster"] != "high_entropy"


def test_cli_diagnose_modular_flow_profile(tmp_path):
    root = Path(__file__).resolve().parents[1]
    src = tmp_path / "flow_logic.py"
    src.write_text(
        "\n".join(
            [
                "def front_loaded(x):",
                "    total = 0",
                "    total += x",
                "    total += 1",
                "    total += 2",
                "    if total > 0:",
                "        total += 3",
                "    return total",
                "",
                "def spread_out(x):",
                "    total = x",
                "    if x > 0:",
                "        total += 1",
                "",
                "    helper = total * 2",
                "",
                "    if helper > 4:",
                "        helper -= 1",
                "",
                "    return helper",
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
    payload = json.loads(result.stdout)
    clusters = {c["name"]: c for c in payload["entropy_clusters"] if c["scope"] == "function"}
    assert "modular_flow" in clusters["front_loaded"]
    assert "modular_shrinking_number" in clusters["front_loaded"]["modular_flow"]
    assert clusters["front_loaded"]["modular_flow"]["assessment"] in {"concentrated_flow", "mixed_flow", "uniform_flow"}
    assert clusters["spread_out"]["modular_flow"]["assessment"] in {"uniform_flow", "mixed_flow"}


def test_cli_forecast_json(tmp_path):
    root = Path(__file__).resolve().parents[1]
    trace = tmp_path / "training_trace.json"
    trace.write_text(
        json.dumps(
            {
                "run_id": "exp_001",
                "checkpoint_path": str(tmp_path / "exp_001.pt"),
                "target_metric": "val_top1",
                "steps": [
                    {"step": 100, "train_loss": 1.8, "val_metric": 0.36, "grad_norm": 1.4, "curvature": 0.30, "direction_consistency": 0.58},
                    {"step": 200, "train_loss": 1.4, "val_metric": 0.48, "grad_norm": 1.3, "curvature": 0.24, "direction_consistency": 0.66},
                    {"step": 300, "train_loss": 1.2, "val_metric": 0.58, "grad_norm": 1.2, "curvature": 0.22, "direction_consistency": 0.73},
                    {"step": 400, "train_loss": 1.0, "val_metric": 0.66, "grad_norm": 1.1, "curvature": 0.18, "direction_consistency": 0.78},
                    {"step": 500, "train_loss": 0.92, "val_metric": 0.71, "grad_norm": 1.0, "curvature": 0.15, "direction_consistency": 0.82},
                    {"step": 600, "train_loss": 0.88, "val_metric": 0.74, "grad_norm": 0.95, "curvature": 0.13, "direction_consistency": 0.86},
                    {"step": 700, "train_loss": 0.85, "val_metric": 0.755, "grad_norm": 0.92, "curvature": 0.12, "direction_consistency": 0.88},
                    {"step": 800, "train_loss": 0.83, "val_metric": 0.763, "grad_norm": 0.91, "curvature": 0.11, "direction_consistency": 0.89}
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    env = _env_with_src()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "forecast", str(trace), "--json"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["run_id"] == "exp_001"
    assert "predicted_final_score" in payload
    assert "earliest_decision_step" in payload
    assert "continue_training_recommended" in payload


def test_cli_zombie_json(tmp_path):
    root = Path(__file__).resolve().parents[1]
    trace = tmp_path / "nan_trace.json"
    trace.write_text(
        json.dumps(
            {
                "run_id": "nan_run",
                "checkpoint_path": "",
                "target_metric": "val_metric",
                "steps": [
                    {"step": 1, "train_loss": 1.5, "val_metric": 0.30, "grad_norm": 1.2, "curvature": 0.20, "direction_consistency": 0.60},
                    {"step": 2, "train_loss": 1.3, "val_metric": 0.40, "grad_norm": 1.1, "curvature": 0.18, "direction_consistency": 0.65},
                    {"step": 3, "train_loss": float("nan"), "val_metric": 0.42, "grad_norm": 1.0, "curvature": 0.16, "direction_consistency": 0.70},
                ],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    env = _env_with_src()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, "-m", "tensorearch", "zombie", str(trace), "--json"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["severity"] == "ZOMBIE"
    assert payload["zombie_class"] == "NAN_HOST"
    assert payload["infection_step"] == 3


def test_zombie_high_preclip_with_improving_loss_is_suspect_not_infected():
    """The pre-Codex bug: pre-clip grad_norm > 50 immediately classified INFECTED.
    Now: high pre-clip alone with no spike + with improving loss → SUSPECT only."""
    from tensorearch.zombie import assess_zombie
    # 6 steps with grad_norm = 200 throughout (steady, not spike), loss decreasing
    trace = load_training_trace_from_dict({
        "run_id": "high_preclip_healthy",
        "steps": [
            {"step": i, "train_loss": 2.5 - 0.1 * i, "val_metric": 0.0,
             "grad_norm": 200.0, "curvature": 0.0, "direction_consistency": 1.0,
             "grad_norm_kind": "pre_clip"}
            for i in range(1, 7)
        ],
    })
    report = assess_zombie(trace)
    # legacy single-point heuristic still surfaces something when pre-clip
    # exceeds the (now raised) inf_grad_threshold; but loss is improving and
    # the value is well below the abs_threshold (1000), so it must be SUSPECT
    # at most, never INFECTED.
    assert report.severity in ("ALIVE", "SUSPECT"), report
    assert report.severity != "INFECTED"


def test_zombie_healthy_clipper_with_high_preclip_returns_alive_not_suspect():
    """If trace records post_clip and clipper is doing its job (post_clip <= clip*factor),
    high pre-clip alone must NOT trigger SUSPECT — it's just normal warmup behavior."""
    from tensorearch.zombie import assess_zombie
    trace = load_training_trace_from_dict({
        "run_id": "healthy_clipper_high_preclip",
        "steps": [
            {"step": i, "train_loss": 11.0 - 0.05 * i, "val_metric": 0.0,
             "grad_norm": 5000.0,                  # huge pre-clip
             "post_clip_grad_norm": 1.0,            # but clipper neutralized it
             "gradient_clip": 1.0,
             "curvature": 0.01, "direction_consistency": 1.0,
             "grad_norm_kind": "pre_clip"}
            for i in range(1, 8)
        ],
    })
    report = assess_zombie(trace)
    # Clipper is healthy (post_clip == clip == 1.0), loss is improving — no zombie behavior.
    assert report.severity == "ALIVE", report
    assert report.zombie_class == "HEALTHY"


def test_zombie_post_clip_combined_rule_triggers_infected():
    """When trace records post-clip + clip threshold, sustained breach → INFECTED."""
    from tensorearch.zombie import assess_zombie
    trace = load_training_trace_from_dict({
        "run_id": "broken_clipper",
        "steps": [
            {"step": i, "train_loss": 5.0 + 0.5 * i, "val_metric": 0.0,
             "grad_norm": 5.0, "post_clip_grad_norm": 5.0, "gradient_clip": 1.0,
             "curvature": 0.0, "direction_consistency": 0.0,
             "grad_norm_kind": "pre_clip"}
            for i in range(1, 7)
        ],
    })
    report = assess_zombie(trace)
    assert report.severity == "INFECTED", report
    assert report.zombie_class == "INF_HOST"
    assert report.evidence.get("rule") == "post_clip_combined"


def test_trace_contract_detects_display_smoothing():
    """train_loss_kind='display_smoothed' -> CRITICAL warning + valid_for_forecast=False."""
    from tensorearch.training_contract import validate_trace
    trace = load_training_trace_from_dict({
        "run_id": "smoothed",
        "steps": [
            {"step": i, "train_loss": 1.5 - 0.05 * i, "val_metric": 0.0,
             "grad_norm": 1.0, "curvature": 0.0, "direction_consistency": 1.0,
             "train_loss_kind": "display_smoothed"}
            for i in range(1, 8)
        ],
    })
    report = validate_trace(trace)
    assert report.valid_for_forecast is False
    codes = {w.code for w in report.warnings}
    assert "DISPLAY_SMOOTHED_LOSS" in codes


def test_forecast_returns_stop_window_when_stable():
    """When the heuristic early-decision criteria fire, all three window fields are non-zero."""
    # Construct a very stable, high-confidence trace
    trace = load_training_trace_from_dict({
        "run_id": "stable_run",
        "steps": [
            {"step": i, "train_loss": 0.5, "val_metric": 0.95,
             "grad_norm": 0.5, "curvature": 0.001, "direction_consistency": 1.0,
             "train_loss_kind": "raw_step_mean", "val_metric_observed": True,
             "grad_norm_kind": "pre_clip", "gradient_clip": 1.0}
            for i in range(1, 16)
        ],
    })
    result = forecast_trace(trace)
    assert result.continue_training_recommended is False
    assert result.recommended_stop_step > 0
    assert result.decision_window_start > 0
    assert result.decision_window_end >= result.decision_window_start


def test_forecast_signal_poor_floors_confidence():
    trace = load_training_trace_from_dict(
        {
            "run_id": "signal_poor_run",
            "checkpoint_path": "",
            "steps": [
                {"step": i, "train_loss": 10.0 - 0.05 * i, "val_metric": 0.0, "grad_norm": 0.0, "curvature": 0.0, "direction_consistency": 1.0}
                for i in range(1, 12)
            ],
        }
    )
    result = forecast_trace(trace)
    assert result.metadata["signal_quality"]["poor"] is True
    assert set(result.metadata["signal_quality"]["missing"]) == {"val_metric", "grad_norm", "curvature"}
    assert result.confidence <= 0.25
    assert "signal-poor" in result.reason


def test_forecast_trace_resets_between_runs():
    trace_a = load_training_trace_from_dict(
        {
            "run_id": "run_a",
            "checkpoint_path": "A.pt",
            "steps": [
                {"step": 1, "train_loss": 1.6, "val_metric": 0.35, "grad_norm": 1.5, "curvature": 0.35, "direction_consistency": 0.55},
                {"step": 2, "train_loss": 1.3, "val_metric": 0.48, "grad_norm": 1.3, "curvature": 0.26, "direction_consistency": 0.64},
                {"step": 3, "train_loss": 1.0, "val_metric": 0.61, "grad_norm": 1.1, "curvature": 0.18, "direction_consistency": 0.73},
                {"step": 4, "train_loss": 0.88, "val_metric": 0.70, "grad_norm": 0.98, "curvature": 0.14, "direction_consistency": 0.81},
                {"step": 5, "train_loss": 0.82, "val_metric": 0.75, "grad_norm": 0.90, "curvature": 0.12, "direction_consistency": 0.86},
                {"step": 6, "train_loss": 0.79, "val_metric": 0.77, "grad_norm": 0.88, "curvature": 0.10, "direction_consistency": 0.89},
            ],
        }
    )
    trace_b = load_training_trace_from_dict(
        {
            "run_id": "run_b",
            "checkpoint_path": "B.pt",
            "steps": [
                {"step": 1, "train_loss": 2.2, "val_metric": 0.18, "grad_norm": 2.0, "curvature": 0.60, "direction_consistency": 0.35},
                {"step": 2, "train_loss": 2.0, "val_metric": 0.22, "grad_norm": 1.9, "curvature": 0.55, "direction_consistency": 0.38},
                {"step": 3, "train_loss": 1.9, "val_metric": 0.24, "grad_norm": 1.8, "curvature": 0.50, "direction_consistency": 0.40},
                {"step": 4, "train_loss": 1.85, "val_metric": 0.25, "grad_norm": 1.8, "curvature": 0.48, "direction_consistency": 0.41},
                {"step": 5, "train_loss": 1.82, "val_metric": 0.255, "grad_norm": 1.78, "curvature": 0.47, "direction_consistency": 0.42},
                {"step": 6, "train_loss": 1.80, "val_metric": 0.258, "grad_norm": 1.77, "curvature": 0.46, "direction_consistency": 0.43},
            ],
        }
    )
    result_a = forecast_trace(trace_a)
    result_b = forecast_trace(trace_b)
    assert result_a.predicted_final_score > result_b.predicted_final_score
    assert result_a.run_id == "run_a"
    assert result_b.run_id == "run_b"


def test_cli_temporal_couple_json(tmp_path):
    root = Path(__file__).resolve().parents[1]
    T, H, W = 16, 8, 8
    rng = np.random.default_rng(0)
    h = np.zeros((T, H, W), dtype=float)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    yy, xx = np.mgrid[0:H, 0:W]
    bump = np.exp(-((yy - 1.0) ** 2 + (xx - 1.0) ** 2) / 2.5)
    for t in range(T):
        h[t] = 0.05 * t * bump + 0.01 * rng.standard_normal(h[t].shape)
        gy_t, gx_t = np.gradient(h[t])
        mag = np.sqrt(gx_t ** 2 + gy_t ** 2) + 1e-6
        mask = (yy < 3) & (xx < 3)
        u[t] = np.where(mask,  gy_t / mag, 0.01 * rng.standard_normal(h[t].shape))
        v[t] = np.where(mask, -gx_t / mag, 0.01 * rng.standard_normal(h[t].shape))
    probe = tmp_path / "couple_probe.npz"
    np.savez(probe, dt=np.asarray(1.0), h=h, u=u, v=v)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorearch",
            "temporal-couple",
            "--input",
            str(probe),
            "--time-bins",
            "4",
            "--y-bins",
            "2",
            "--x-bins",
            "2",
            "--json",
        ],
        cwd=root,
        env=_env_with_src(),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["version"] == "trcouple.v1"
    assert payload["meta"]["pair"] == ["h", "uv"]
    # Lock should land in the top-left bump region.
    assert payload["lock"]["spatial_window"]["y_bin"] == 0
    assert payload["lock"]["spatial_window"]["x_bin"] == 0
    assert payload["lock"]["verdict"]["kind"] == "anti_geostrophic"
    # Core coupling metrics must be present and non-null.
    cs = payload["coupling_scores"]
    for key in ("h_uv_coupling_mean", "h_uv_anti_geo_fraction",
                "grad_growth_mean", "geostrophic_coherence_mean"):
        assert cs[key] is not None, f"coupling_scores.{key} is None"


def test_cli_temporal_balance_json(tmp_path):
    root = Path(__file__).resolve().parents[1]
    T, H, W = 8, 6, 8
    # Potential = +x ramp; response aligned with rotated_grad → verdict
    # should be potential_consistent.
    x = np.arange(W, dtype=float)
    y = np.arange(H, dtype=float)
    YY, XX = np.meshgrid(y, x, indexing="ij")
    pot = np.broadcast_to(XX[None, :, :], (T, H, W)).copy()
    gy = np.gradient(pot, axis=1)
    gx = np.gradient(pot, axis=2)
    u = -gy
    v =  gx
    probe = tmp_path / "balance_probe.npz"
    np.savez(probe, h=pot, u=u, v=v)

    result = subprocess.run(
        [
            sys.executable, "-m", "tensorearch",
            "temporal-balance",
            "--input", str(probe),
            "--potential", "h",
            "--response-u", "u",
            "--response-v", "v",
            "--operator", "rotated_gradient",
            "--operator-scale", "1.0",
            "--time-bins", "4",
            "--y-bins", "3",
            "--x-bins", "4",
            "--json",
        ],
        cwd=root, env=_env_with_src(),
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["version"] == "trbalance_report.v1"
    g = payload["global"]
    for key in ("alignment_potential_only_mean", "alignment_static_only_mean",
                "alignment_combined_mean", "speed_potential_only_mean",
                "speed_static_only_mean", "speed_combined_mean",
                "consistency_gap_mean", "best_mode", "verdict"):
        assert key in g, f"global.{key} missing"
    assert g["verdict"] == "potential_consistent"
    assert g["alignment_potential_only_mean"] > 0.95
