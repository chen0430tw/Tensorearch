import json
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
