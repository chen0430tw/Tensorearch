from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


BASELINE_DENSITY = {
    "residual": 0.3812296177637549,
    "attention": 0.11233611144671586,
    "ffn": 0.34391218143014296,
    "expert": 0.0,
    "propagation": 0.0,
    "kv": 0.16252208935938614,
    "diffusion": 0.0,
    "timestep": 0.0,
    "unet": 0.0,
    "adapter": 0.0,
    "runtime": 0.0,
    "video": 0.0,
    "audio": 0.0,
    "threed": 0.0,
    "speech": 0.0,
    "world": 0.0,
    "multimodal": 0.0,
    "graph": 0.0,
    "vision": 0.0,
    "bio": 0.0,
}

# Pre-compiled per-dimension regexes — one pass per dimension instead of N passes
_RE_RESIDUAL = re.compile(
    r"\bresidual\b|\bx\s*\+\s*|\bskip\b|\bshortcut\b|\bln1\b|\bln2\b|\brmsnorm\b|\blayernorm\b",
    re.IGNORECASE,
)
_RE_ATTENTION = re.compile(
    r"\battention\b|\battn\b|\bq_proj\b|\bk_proj\b|\bv_proj\b|\bwo\b|\bwq\b|\bwk\b|\bwv\b"
    r"|\bsoftmax\b|\brope\b|\bmla\b|\bq_lora_rank\b|\bqk_",
    re.IGNORECASE,
)
_RE_FFN = re.compile(
    r"\bffn\b|\bmlp\b|\binter_dim\b|\bintermediate_size\b|\bup_proj\b|\bdown_proj\b"
    r"|\bgate_proj\b|\bsilu\b|\bgelu\b",
    re.IGNORECASE,
)
_RE_EXPERT = re.compile(
    r"\bexpert\b|\bmoe\b|\brouter\b|\bgate\b|\bn_routed_experts\b|\bn_shared_experts\b"
    r"|\bn_activated_experts\b|\bshared_experts\b|\btopk\b",
    re.IGNORECASE,
)
_RE_PROPAGATION = re.compile(
    r"\bprop\b|\bpropagation\b|\boscillat|\bstate[_ ]field\b|\bdynamics\b|\bphase\b|\bcoupl",
    re.IGNORECASE,
)
_RE_KV = re.compile(
    r"\bkv\b|\bkv_cache\b|\bcache\b|\bkey[_ ]value\b|\bwkv\b|\bkv_lora_rank\b|\blatent\b|\btransport\b",
    re.IGNORECASE,
)
_RE_DIFFUSION = re.compile(
    r"\bdiffusion\b|\bdenois\w*\b|\bddpm\b|\bddim\b|\bplms\b|\bscheduler\b|\bsampler\b|\bsigma\b"
    r"|\bnoise_pred\b|\beps(?:ilon)?\b|\bq_sample\b|\bp_sample\b|\bstochastic_encode\b|\balpha(?:s|_bar)?\b|\bbeta(?:s)?\b",
    re.IGNORECASE,
)
_RE_TIMESTEP = re.compile(
    r"\btimestep(?:s)?\b|\btime_embed(?:ding)?\b|\btemb\b|\bsinusoidal\b|\bpositional\b|\btime_cond\b"
    r"|\bget_timestep_embedding\b|\btime_proj\b|\btime_mlp\b",
    re.IGNORECASE,
)
_RE_UNET = re.compile(
    r"\bunet\w*\b|\bdown_block(?:s)?\b|\bup_block(?:s)?\b|\bdownsample(?:r|d)?\b|\bupsample(?:r|d)?\b"
    r"|\bskip_connection\b|\bresnet_block\b|\bresblock\b|\bmid_block\b|\bconv_in\b|\bconv_out\b",
    re.IGNORECASE,
)
_RE_ADAPTER = re.compile(
    r"\blora\b|\blow[_-]?rank\b|\brank[_ ]?decomp\w*\b|\badapter\b|\bpeft\b|\bhypernetwork\b|\bdelta[_ ]weight\b"
    r"|\bmerge_(?:weights|adapter)\b|\bunmerge\b|\btarget_modules?\b",
    re.IGNORECASE,
)
_RE_RUNTIME = re.compile(
    r"\bruntime\b|\bwrapper\b|\bbridge\b|\bconfig\b|\bexport\b|\bdetect\b|\btopology\b|\btriton\b|\bkernel\b"
    r"|\bquant(?:ize|_group|_tensor)?\b|\bcached\b|\bbuild_[a-z0-9_]+\b|\bapply_[a-z0-9_]+\b",
    re.IGNORECASE,
)
_RE_VIDEO = re.compile(
    r"\bvideo\b|\btemporal\b|\bspatiotemporal\b|\bmotion\b|\bclip_length\b"
    r"|\bnum_frames\b|\bvideo_unet\b|\bunet3d\b|\banimate(?:diff)?\b|\bcausal_3d\b",
    re.IGNORECASE,
)
_RE_AUDIO = re.compile(
    r"\baudio\b|\bwaveform\b|\bmel(?:_spectrogram)?\b|\bspectrogram\b|\bstft\b|\bistft\b|\bvocoder\b"
    r"|\bcodec\b|\bconformer\b|\bwhisper\b|\btts\b|\basr\b|\baudio_diffusion\b|\bdenoise_audio\b",
    re.IGNORECASE,
)
_RE_THREED = re.compile(
    r"\b3d\b|\bmesh\b|\bpoint[_ ]cloud\b|\bpointnet\b|\bvoxel\b|\btriplane\b|\bnerf\b|\bradiance[_ ]field\b"
    r"|\bsdf\b|\boccupancy\b|\bgaussian[_ ]splat(?:ting)?\b|\bview(?:point|_cond)\b|\bcamera_pose\b",
    re.IGNORECASE,
)
_RE_SPEECH = re.compile(
    r"\basr\b|\btranscribe\b|\bwhisper\b|\bspeech\b|\bn_audio_\w+\b|\bn_text_\w+\b|\baudioencoder\b|\btextdecoder\b"
    r"|\btoken_embedding\b|\blogits\b|\bdetect_language\b|\btimestamp\b|\bdecode\w*\b",
    re.IGNORECASE,
)
_RE_WORLD = re.compile(
    r"\bworld[_-]?model\b|\bmdrnn\b|\bdream\w*\b|\brollout\b|\benvironment\b|\benv\b|\blatent(?:s)?\b|\baction(?:s)?\b"
    r"|\breward\b|\bterminality\b|\bdone\b|\bgaussian(?:s)?\b|\bgmm\b|\blstmcell\b|\bhidden\b",
    re.IGNORECASE,
)
_RE_MULTIMODAL = re.compile(
    r"\bmultimodal\b|\bvision_language\b|\bimage_text\b|\bqformer\b|\bq-former\b|\bquery_tokens?\b|\bvision_model\b"
    r"|\blanguage_model\b|\btext_config\b|\bvision_config\b|\bprojector\b|\bimage_embeds?\b|\btext_embeds?\b|\bcross_modal\b",
    re.IGNORECASE,
)
_RE_GRAPH = re.compile(
    r"\bgraph\b|\bmessage[_ ]passing\b|\bedge_index\b|\badj(?:acency)?\b|\bnode(?:_feat(?:ures)?)?\b|\bneighbors?\b"
    r"|\bgcn\b|\bgat\b|\bgraphsage\b|\bhetero\b|\bscatter\b|\bpropagate\b|\baggregation\b",
    re.IGNORECASE,
)
_RE_VISION = re.compile(
    r"\bvision\b|\bimage\b|\bmask\b|\bsegmentation\b|\bdetection\b|\bbox(?:es)?\b|\broi\b|\brpn\b|\bfpn\b|\bbackbone\b"
    r"|\bproposal(?:s)?\b|\bpixel_values\b|\bmaskrcnn\b|\bdetectron\b|\bpanoptic\b|\bsemantic_seg(?:mentation)?\b",
    re.IGNORECASE,
)
_RE_BIO = re.compile(
    r"\bprotein\b|\bamino[_ ]acid\b|\bresidue(?:_index|s_mask)\b|\bmsa\b|\bevoformer\b"
    r"|\bfold(?:ing)?_model\b|\besm[0-9]\b|\besm_model\b|\bcontact_map\b|\bpairwise_rep\b"
    r"|\bchain_id\b|\btoken_dropout\b|\bnum_residues\b|\bamino_vocab\b",
    re.IGNORECASE,
)


@dataclass
class DensityVector:
    residual: float
    attention: float
    ffn: float
    expert: float
    propagation: float
    kv: float
    diffusion: float
    timestep: float
    unet: float
    adapter: float
    runtime: float
    video: float
    audio: float
    threed: float
    speech: float
    world: float
    multimodal: float
    graph: float
    vision: float
    bio: float


def _keyword_hits(text: str, patterns: list[str]) -> int:
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, text, flags=re.IGNORECASE))
    return total


def _normalize_density(density: DensityVector) -> dict[str, float]:
    data = asdict(density)
    total = sum(data.values())
    if total <= 0:
        return {k: 0.0 for k in data}
    return {k: v / total for k, v in data.items()}


def _delta_density(current: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {k: current[k] - baseline[k] for k in current}


def infer_density_from_source_text(text: str) -> DensityVector:
    return DensityVector(
        residual=1.0 + len(_RE_RESIDUAL.findall(text)) / 12.0,
        attention=len(_RE_ATTENTION.findall(text)) / 10.0,
        ffn=len(_RE_FFN.findall(text)) / 10.0,
        expert=len(_RE_EXPERT.findall(text)) / 10.0,
        propagation=len(_RE_PROPAGATION.findall(text)) / 8.0,
        kv=len(_RE_KV.findall(text)) / 8.0,
        diffusion=len(_RE_DIFFUSION.findall(text)) / 10.0,
        timestep=len(_RE_TIMESTEP.findall(text)) / 8.0,
        unet=len(_RE_UNET.findall(text)) / 8.0,
        adapter=len(_RE_ADAPTER.findall(text)) / 8.0,
        runtime=len(_RE_RUNTIME.findall(text)) / 10.0,
        video=len(_RE_VIDEO.findall(text)) / 8.0,
        audio=len(_RE_AUDIO.findall(text)) / 8.0,
        threed=len(_RE_THREED.findall(text)) / 8.0,
        speech=len(_RE_SPEECH.findall(text)) / 8.0,
        world=len(_RE_WORLD.findall(text)) / 8.0,
        multimodal=len(_RE_MULTIMODAL.findall(text)) / 8.0,
        graph=len(_RE_GRAPH.findall(text)) / 8.0,
        vision=len(_RE_VISION.findall(text)) / 8.0,
        bio=len(_RE_BIO.findall(text)) / 8.0,
    )


def build_quadrupole_projection(normalized: dict[str, float]) -> dict[str, object]:
    delta = _delta_density(normalized, BASELINE_DENSITY)
    x = normalized["residual"]
    y = normalized["attention"] + normalized["kv"]
    z = normalized["kv"]
    w = normalized["propagation"]

    _axes = {"X": x, "Y": y, "Z": z, "W": w}
    dominant_axis = max(_axes, key=_axes.__getitem__)
    _classifications = {
        "X": "baseline residual",
        "Y": "latent-attention dominant",
        "Z": "kv-transport dominant",
        "W": "propagation dominant",
    }
    classification = _classifications[dominant_axis]

    return {
        "axes": {
            "X_residual": x,
            "Y_latent_attention": y,
            "Z_kv_transport": z,
            "W_propagation": w,
        },
        "extension": {
            "expert_extension": normalized["expert"],
            "ffn_extension": normalized["ffn"],
        },
        "delta_from_baseline": delta,
        "dominant_axis": dominant_axis,
        "classification": classification,
    }


def build_space_family_projection(normalized: dict[str, float]) -> dict[str, object]:
    legacy = build_quadrupole_projection(normalized)
    video_diffusion_overlap = min(
        normalized["video"],
        normalized["diffusion"] + normalized["unet"] + normalized["timestep"],
    )
    audio_generative_overlap = min(
        normalized["audio"],
        normalized["attention"] + normalized["diffusion"] + normalized["unet"] + normalized["ffn"],
    )
    specialization_mass = (
        normalized["attention"]
        + normalized["kv"]
        + normalized["diffusion"]
        + normalized["timestep"]
        + normalized["unet"]
        + normalized["adapter"]
        + normalized["runtime"]
        + normalized["video"]
        + normalized["audio"]
        + normalized["threed"]
        + normalized["speech"]
        + normalized["world"]
        + normalized["multimodal"]
        + normalized["graph"]
        + normalized["vision"]
        + normalized["bio"]
        + normalized["propagation"]
    )
    family_scores = {
        "baseline_residual": max(normalized["residual"] - 0.45 * specialization_mass, 0.0),
        "latent_attention": 1.2 * normalized["attention"] + 1.1 * normalized["kv"] + 0.35 * normalized["ffn"],
        "diffusion_unet": 1.8 * normalized["diffusion"] + 1.8 * normalized["unet"] + 1.4 * normalized["timestep"] + 0.25 * normalized["attention"],
        "adapterization": 2.2 * normalized["adapter"] + 0.25 * normalized["attention"] + 0.15 * normalized["ffn"],
        "runtime_wrapper": 1.8 * normalized["runtime"] + 0.2 * normalized["propagation"] + 0.15 * normalized["kv"],
        "video_temporal": 2.2 * normalized["video"] + 0.8 * normalized["diffusion"] + 0.45 * normalized["attention"] + 0.35 * normalized["unet"] + 0.2 * normalized["timestep"] + 5.0 * video_diffusion_overlap,
        "audio_spectral": 2.0 * normalized["audio"] + 0.6 * normalized["diffusion"] + 0.2 * normalized["attention"] + 3.5 * audio_generative_overlap,
        "threed_generative": 2.0 * normalized["threed"] + 0.35 * normalized["attention"] + 0.25 * normalized["propagation"],
        "speech_language": 2.4 * normalized["speech"] + 1.0 * normalized["audio"] + 0.8 * normalized["attention"] + 0.8 * normalized["kv"],
        "world_model": 2.3 * normalized["world"] + 0.8 * normalized["propagation"] + 0.6 * normalized["kv"],
        "multimodal_alignment": 2.2 * normalized["multimodal"] + 0.8 * normalized["attention"] + 0.6 * normalized["vision"] + 0.4 * normalized["speech"],
        "graph_message_passing": 2.5 * normalized["graph"] + 0.3 * normalized["attention"] + 0.2 * normalized["propagation"],
        "vision_detection": 2.2 * normalized["vision"] + 0.4 * normalized["attention"] + 0.3 * normalized["runtime"],
        "bio_sequence": 2.3 * normalized["bio"] + 0.5 * normalized["attention"] + 0.2 * normalized["kv"],
        "propagation": 1.2 * normalized["propagation"],
    }
    dominant_family = max(family_scores, key=family_scores.__getitem__)
    family_labels = {
        "baseline_residual": "baseline residual",
        "latent_attention": "latent-attention dominant",
        "diffusion_unet": "diffusion-unet dominant",
        "adapterization": "adapterization dominant",
        "runtime_wrapper": "runtime-wrapper dominant",
        "video_temporal": "video-temporal dominant",
        "audio_spectral": "audio-spectral dominant",
        "threed_generative": "3d-generative dominant",
        "speech_language": "speech-language dominant",
        "world_model": "world-model dominant",
        "multimodal_alignment": "multimodal-alignment dominant",
        "graph_message_passing": "graph-message-passing dominant",
        "vision_detection": "vision-detection dominant",
        "bio_sequence": "bio-sequence dominant",
        "propagation": "propagation dominant",
    }
    diffusion_axes = {
        "D_diffusion_denoising": normalized["diffusion"],
        "T_timestep_conditioning": normalized["timestep"],
        "U_multiscale_unet": normalized["unet"],
        "A_adapterization": normalized["adapter"],
        "R_runtime_wrapper": normalized["runtime"],
        "V_temporal_video": normalized["video"],
        "O_audio_spectral": normalized["audio"],
        "G_3d_generative": normalized["threed"],
        "S_speech_language": normalized["speech"],
        "M_world_model": normalized["world"],
        "L_multimodal_alignment": normalized["multimodal"],
        "H_graph_message_passing": normalized["graph"],
        "I_vision_detection": normalized["vision"],
        "B_bio_sequence": normalized["bio"],
    }
    return {
        "legacy_quadrupole": legacy,
        "space_family_scores": family_scores,
        "extended_axes": diffusion_axes,
        "dominant_family": dominant_family,
        "classification": family_labels[dominant_family],
    }


def analyze_source_file(path: str | Path) -> dict[str, object]:
    source = Path(path)
    raw_bytes = source.read_bytes()
    text = None
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw_bytes.decode("utf-8", errors="ignore")

    raw_density = infer_density_from_source_text(text)
    normalized = _normalize_density(raw_density)
    quad = build_quadrupole_projection(normalized)
    family = build_space_family_projection(normalized)
    return {
        "mode": "source_file_auto",
        "source_file": str(source),
        "raw_density": asdict(raw_density),
        "normalized_density": normalized,
        "baseline_density": BASELINE_DENSITY,
        "quadrupole_projection": quad,
        "space_family_projection": family,
        "classification": family["classification"],
    }


def space_report(payload: dict[str, object]) -> str:
    quad = payload["quadrupole_projection"]
    family = payload["space_family_projection"]
    axes = quad["axes"]
    ext = quad["extension"]
    ext_axes = family["extended_axes"]
    return (
        f"classification={payload['classification']}\n"
        f"dominant_axis={quad['dominant_axis']}\n"
        f"dominant_family={family['dominant_family']}\n"
        f"X_residual={axes['X_residual']:.4f}\n"
        f"Y_latent_attention={axes['Y_latent_attention']:.4f}\n"
        f"Z_kv_transport={axes['Z_kv_transport']:.4f}\n"
        f"W_propagation={axes['W_propagation']:.4f}\n"
        f"D_diffusion_denoising={ext_axes['D_diffusion_denoising']:.4f}\n"
        f"T_timestep_conditioning={ext_axes['T_timestep_conditioning']:.4f}\n"
        f"U_multiscale_unet={ext_axes['U_multiscale_unet']:.4f}\n"
        f"A_adapterization={ext_axes['A_adapterization']:.4f}\n"
        f"R_runtime_wrapper={ext_axes['R_runtime_wrapper']:.4f}\n"
        f"V_temporal_video={ext_axes['V_temporal_video']:.4f}\n"
        f"O_audio_spectral={ext_axes['O_audio_spectral']:.4f}\n"
        f"G_3d_generative={ext_axes['G_3d_generative']:.4f}\n"
        f"S_speech_language={ext_axes['S_speech_language']:.4f}\n"
        f"M_world_model={ext_axes['M_world_model']:.4f}\n"
        f"L_multimodal_alignment={ext_axes['L_multimodal_alignment']:.4f}\n"
        f"H_graph_message_passing={ext_axes['H_graph_message_passing']:.4f}\n"
        f"I_vision_detection={ext_axes['I_vision_detection']:.4f}\n"
        f"B_bio_sequence={ext_axes['B_bio_sequence']:.4f}\n"
        f"expert_extension={ext['expert_extension']:.4f}\n"
        f"ffn_extension={ext['ffn_extension']:.4f}"
    )


def space_report_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)
