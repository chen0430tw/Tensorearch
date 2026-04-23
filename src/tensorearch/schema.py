from dataclasses import dataclass, field
from typing import Any


@dataclass
class SystemTrace:
    name: str = "unknown"
    model_arch: str = "unknown"
    num_layers: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_heads: int = 0
    kv_heads: int = 0
    batch_size: int = 0
    seq_len: int = 0
    dtype: str = "unknown"
    measured_latency_ms: float = 0.0
    measured_tokens_per_sec: float = 0.0
    device: str = "unknown"
    tp_degree: int = 1
    pp_degree: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SliceState:
    slice_id: str
    kind: str
    layer_index: int = -1
    op_type: str = "unknown"
    hidden_size: int = 0
    intermediate_size: int = 0
    num_heads: int = 0
    kv_heads: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    flops: float = 0.0
    activation_bytes: float = 0.0
    memory_bytes: float = 0.0
    weight_bytes: float = 0.0
    kv_bytes: float = 0.0
    comm_bytes: float = 0.0
    sync_cost: float = 0.0
    kernel_time_ms: float = 0.0
    stall_ms: float = 0.0
    measured_latency_ms: float = 0.0
    write_magnitude: float = 1.0
    read_sensitivity: float = 1.0
    doi_alignment: float = 1.0
    obedience_target: float = 1.0
    local_vector_space: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SliceEdge:
    src: str
    dst: str
    weight: float = 0.0
    kind: str = "dataflow"
    edge_type: str = "dataflow"
    edge_bytes: float = 0.0
    transport_scale: float = 1.0
    collective: str = "none"
    same_device: bool = True
    same_stage: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Intervention:
    kind: str
    target: str
    value: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingStep:
    step: int
    train_loss: float = 0.0
    val_metric: float = 0.0
    grad_norm: float = 0.0
    curvature: float = 0.0
    direction_consistency: float = 0.0
    # Contract metadata (Codex review v4): the trace writer should describe what
    # kind of value it put into train_loss / grad_norm so downstream consumers
    # (zombie / forecast / contract validator) don't get fooled by display
    # smoothing or pre-clip explosions that the optimizer already neutralized.
    train_loss_kind: str = "unknown"        # "raw_step_mean" | "display_smoothed" | "unknown"
    val_metric_observed: bool = False       # False if val_metric is a placeholder/stale
    grad_norm_kind: str = "unknown"         # "pre_clip" | "post_clip" | "unknown"
    post_clip_grad_norm: float = 0.0        # 0 if not captured
    gradient_clip: float = 0.0              # clip threshold in effect this step (0 = unknown)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingTrace:
    run_id: str
    checkpoint_path: str = ""
    target_metric: str = "val_metric"
    steps: list[TrainingStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastResult:
    run_id: str
    predicted_final_score: float
    uncertainty: float
    confidence: float
    earliest_decision_step: int
    continue_training_recommended: bool
    stability: float
    growth_fitness: float
    growth_gain: float
    reason: str
    # Stop-window outputs (Codex review v4): rather than committing to a
    # single "stop here" step, the forecaster also reports a window of
    # acceptable stop points and a recommended one inside that window.
    # All three are 0 when the forecaster is not yet confident enough to
    # recommend stopping (in that case the caller should keep training).
    decision_window_start: int = 0   # earliest step where stop is safe
    decision_window_end: int = 0     # latest step where further training is unlikely to improve
    recommended_stop_step: int = 0   # forecaster's pick inside the window
    metadata: dict[str, Any] = field(default_factory=dict)
