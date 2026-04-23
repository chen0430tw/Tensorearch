from __future__ import annotations

"""
Zombie detection for training runs.

A zombie is a training run that has died but keeps moving — producing
numbers that look like progress but no longer carry signal. Classical fitness
metrics mistake zombies for healthy runs because NaN propagation can bypass
clamps and produce artificially "perfect" scores.

This module explicitly hunts for zombies. Unlike the forecast module (which
asks "will this run succeed?"), the zombie module asks "is this run still
alive at all?"

Inspired by the black-humor observation that in naive forecast code, a run
with NaN loss gets predicted_final_score = 1.0 — the deadest individual tops
the fitness ranking. That is not a successful run. That is a zombie.

Zombie classes:
  NAN_HOST           训练信号被 NaN 污染（经典 zombie，尸体发光）
  INF_HOST           梯度爆炸后未处理（尸体还在抽搐）
  FROZEN_LIMB        梯度消失且 loss 不动（冻僵的肢体）
  FAKE_PLATEAU       指标微动但方向一致性完全坍塌（行走的尸体）
  DRIFT_PHASE        loss 和 metric 反向（灵魂和身体分离）
  HEALTHY            还活着

严重度：
  ALIVE     - 正常运行
  SUSPECT   - 疑似感染，需观察
  INFECTED  - 已感染，建议隔离
  ZOMBIE    - 确认死亡但仍在产出数字，必须终止
"""

import math
from dataclasses import dataclass, field
from typing import Literal

from .schema import TrainingStep, TrainingTrace
from .training_contract import validate_trace


ZombieClass = Literal[
    "HEALTHY",
    "NAN_HOST",
    "INF_HOST",
    "FROZEN_LIMB",
    "FAKE_PLATEAU",
    "DRIFT_PHASE",
]

Severity = Literal["ALIVE", "SUSPECT", "INFECTED", "ZOMBIE"]


@dataclass(frozen=True)
class ZombieConfig:
    # Loss 变化 < 此阈值且持续多步 = 冻僵
    frozen_delta_threshold: float = 1e-5
    frozen_min_steps: int = 4

    # Legacy 单点阈值: pre-clip grad norm > 此值即触发的旧规则。新版默认抬高
    # 到 1000 + 配合下面的组合规则; 设这么高相当于"几乎不会单独触发"。
    inf_grad_threshold: float = 1000.0

    # 梯度范数过小且 loss 不动 = 冻僵肢体
    dead_grad_threshold: float = 1e-6

    # 方向一致性崩塌 = 行走的尸体
    direction_collapse_threshold: float = 0.15

    # Loss 降而 metric 也降（或反向）= 灵魂分离
    drift_correlation_threshold: float = -0.3

    # 至少多少步才能做丧尸判定
    min_observation_steps: int = 3

    # ── 组合式爆炸检测 (Codex review) ────────────────────────────────────
    # pre-clip grad norm 单独大不算 INFECTED。要 INFECTED 需要：
    #   * 优先路径 (有 post-clip + gradient_clip)：post_clip 大于 clip * factor
    #     持续 N 步 + 同时 loss 没改善 → INFECTED
    #   * 退化路径 (只有 pre-clip)：pre_clip > 绝对阈值 + > spike_ratio × 近期中位
    #     持续 N 步 + loss 没改善 → INFECTED
    # 其他情况：高 pre-clip 只升级到 SUSPECT, 不直接 INFECTED。
    explosive_preclip_abs_threshold: float = 1000.0
    explosive_preclip_spike_ratio: float = 4.0
    explosive_persist_steps: int = 3
    explosive_postclip_factor: float = 1.2
    explosive_loss_window: int = 5  # 检查最近 N 步 loss 是否在改善


DEFAULT_ZOMBIE_CONFIG = ZombieConfig()


@dataclass
class ZombieReport:
    run_id: str
    severity: Severity
    zombie_class: ZombieClass
    infection_step: int             # 第一次检测到感染的 step
    reason: str
    evidence: dict = field(default_factory=dict)
    recommendation: str = ""
    contract: dict = field(default_factory=dict)   # trace_contract validator output


def _is_nan(x: float) -> bool:
    return math.isnan(x) if isinstance(x, float) else False


def _is_inf(x: float) -> bool:
    return math.isinf(x) if isinstance(x, float) else False


def _is_finite(x: float) -> bool:
    return math.isfinite(x) if isinstance(x, float) else True


def _scan_nan_inf(steps: list[TrainingStep]) -> tuple[int, str, dict]:
    """扫描 NaN / Inf 感染。返回 (infection_step, class, evidence)。"""
    for s in steps:
        fields = {
            "train_loss": s.train_loss,
            "val_metric": s.val_metric,
            "grad_norm": s.grad_norm,
            "curvature": s.curvature,
            "direction_consistency": s.direction_consistency,
        }
        nan_fields = [k for k, v in fields.items() if _is_nan(v)]
        inf_fields = [k for k, v in fields.items() if _is_inf(v)]

        if nan_fields:
            return s.step, "NAN_HOST", {"nan_fields": nan_fields, "at_step": s.step}
        if inf_fields:
            return s.step, "INF_HOST", {"inf_fields": inf_fields, "at_step": s.step}
    return -1, "HEALTHY", {}


def _detect_frozen(steps: list[TrainingStep], config: ZombieConfig) -> tuple[int, dict] | None:
    """检测冻僵肢体：loss 不动 + 梯度接近零。"""
    if len(steps) < config.frozen_min_steps:
        return None

    for i in range(config.frozen_min_steps, len(steps) + 1):
        window = steps[i - config.frozen_min_steps:i]
        losses = [s.train_loss for s in window if _is_finite(s.train_loss)]
        grads = [s.grad_norm for s in window if _is_finite(s.grad_norm)]
        if len(losses) < config.frozen_min_steps:
            continue

        loss_range = max(losses) - min(losses)
        avg_grad = sum(grads) / len(grads) if grads else 0.0

        if loss_range < config.frozen_delta_threshold and avg_grad < config.dead_grad_threshold:
            return window[-1].step, {
                "loss_range": round(loss_range, 8),
                "avg_grad_norm": round(avg_grad, 8),
                "window_size": len(window),
            }
    return None


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _loss_improving(losses: list[float]) -> bool:
    """True if losses show improvement over the window (last <= mean of first half)."""
    if len(losses) < 2:
        return True  # not enough data, assume OK
    half = len(losses) // 2
    if half == 0:
        return True
    early = losses[:half]
    late = losses[half:]
    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)
    return late_mean < early_mean  # decreasing is improving


def _detect_explosive(steps: list[TrainingStep], config: ZombieConfig) -> tuple[int, str, dict] | None:
    """组合式爆炸检测。

    返回 (infection_step, severity, evidence) 或 None。severity 可能是
    "INFECTED" (满足完整组合规则) 或 "SUSPECT" (单点高 pre-clip 但未通过组合判定)。

    优先路径 (post-clip 已知)：
      post_clip_grad_norm > gradient_clip * postclip_factor 持续 N 步 +
      最近 N 步 loss 没改善 → INFECTED

    退化路径 (只有 pre-clip)：
      grad_norm > abs_threshold AND > spike_ratio × recent_median 持续 N 步 +
      最近 loss 没改善 → INFECTED

    单点高 pre-clip：legacy inf_grad_threshold 命中但其他条件不齐 → SUSPECT
    """
    n = len(steps)
    if n == 0:
        return None

    persist = max(1, config.explosive_persist_steps)
    win = max(2, config.explosive_loss_window)

    # 优先路径：post-clip + clip threshold 都有
    for end in range(persist, n + 1):
        window = steps[end - persist:end]
        post_clip_vals = [s.post_clip_grad_norm for s in window]
        clip_vals = [s.gradient_clip for s in window]
        # 必须每步都有 post-clip + clip > 0
        if all(v > 0 for v in clip_vals) and all(_is_finite(v) for v in post_clip_vals):
            triggered = all(
                pc > clip * config.explosive_postclip_factor
                for pc, clip in zip(post_clip_vals, clip_vals)
            )
            if triggered:
                # 检查 loss 是否在恶化
                loss_window = [
                    s.train_loss for s in steps[max(0, end - win):end]
                    if _is_finite(s.train_loss)
                ]
                if not _loss_improving(loss_window):
                    last = window[-1]
                    return last.step, "INFECTED", {
                        "rule": "post_clip_combined",
                        "post_clip_grad_norm": round(last.post_clip_grad_norm, 4),
                        "gradient_clip": round(last.gradient_clip, 4),
                        "threshold_factor": config.explosive_postclip_factor,
                        "persist_steps": persist,
                        "loss_improving": False,
                    }

    # 退化路径：只有 pre-clip。需要 abs > threshold + > ratio × median + persist + loss not improving
    finite_grads = [s.grad_norm for s in steps if _is_finite(s.grad_norm)]
    for end in range(persist, n + 1):
        window = steps[end - persist:end]
        pre_clip_vals = [s.grad_norm for s in window]
        if not all(_is_finite(v) for v in pre_clip_vals):
            continue
        if not all(v > config.explosive_preclip_abs_threshold for v in pre_clip_vals):
            continue
        # spike 比对：跟全程 median 比 (避免被自己拉高)
        med = _median(finite_grads[:max(1, end - persist)] or finite_grads)
        if med <= 0 or any(v < med * config.explosive_preclip_spike_ratio for v in pre_clip_vals):
            continue
        loss_window = [
            s.train_loss for s in steps[max(0, end - win):end]
            if _is_finite(s.train_loss)
        ]
        if _loss_improving(loss_window):
            continue  # loss 在改善, 即使梯度大也不算 zombie
        last = window[-1]
        return last.step, "INFECTED", {
            "rule": "pre_clip_combined",
            "grad_norm": round(last.grad_norm, 4),
            "abs_threshold": config.explosive_preclip_abs_threshold,
            "recent_median": round(med, 4),
            "spike_ratio": config.explosive_preclip_spike_ratio,
            "persist_steps": persist,
            "loss_improving": False,
        }

    # SUSPECT 路径：legacy 单点超 inf_grad_threshold 但组合判定没过。
    # 但若 trace 已经写了 post_clip 且 clipper 工作正常 (post_clip 未爆 clip 阈值),
    # 那 high pre-clip 只是 clipper 在做它该做的事情, 不需要 SUSPECT 提醒——直接放行。
    for s in steps:
        if not (_is_finite(s.grad_norm) and s.grad_norm > config.inf_grad_threshold):
            continue
        # 健康 clipper 检查：如果该步有真实 post_clip + clip threshold,
        # 且 post_clip 在 clip × postclip_factor 之内, 视为 "clipper 在工作", 跳过。
        has_post_clip = (
            _is_finite(s.post_clip_grad_norm)
            and s.post_clip_grad_norm > 0
            and s.gradient_clip > 0
        )
        if has_post_clip and s.post_clip_grad_norm <= s.gradient_clip * config.explosive_postclip_factor:
            continue  # clipper 健康, pre-clip 高是正常 warmup 现象
        # 否则保留 SUSPECT, 但根据 post_clip 是否存在给不同 note
        if has_post_clip:
            note = (
                f"高 pre-clip ({round(s.grad_norm, 2)}) 配上 post_clip "
                f"({round(s.post_clip_grad_norm, 4)}) > clip × {config.explosive_postclip_factor}, "
                "clipper 处于压力区。建议降低 LR 或检查 clip 阈值。"
            )
        else:
            note = (
                "高 pre-clip 但未触发组合规则; trace 缺 post_clip_grad_norm + "
                "gradient_clip, zombie 无法判定 clipper 是否在工作。建议训练侧补这两个字段。"
            )
        return s.step, "SUSPECT", {
            "rule": "legacy_single_point",
            "grad_norm": round(s.grad_norm, 4),
            "threshold": config.inf_grad_threshold,
            "post_clip_grad_norm": round(s.post_clip_grad_norm, 4) if has_post_clip else None,
            "gradient_clip": round(s.gradient_clip, 4) if has_post_clip else None,
            "note": note,
        }
    return None


def _detect_fake_plateau(steps: list[TrainingStep], config: ZombieConfig) -> tuple[int, dict] | None:
    """检测假平台：metric 看起来在动但方向一致性已崩塌。"""
    if len(steps) < 3:
        return None

    # 只看能用的步骤
    finite_steps = [s for s in steps if _is_finite(s.direction_consistency) and _is_finite(s.val_metric)]
    if len(finite_steps) < 3:
        return None

    recent = finite_steps[-3:]
    avg_direction = sum(s.direction_consistency for s in recent) / len(recent)

    # metric 仍有微弱变动
    metrics = [s.val_metric for s in recent]
    metric_range = max(metrics) - min(metrics)

    if avg_direction < config.direction_collapse_threshold and metric_range > 1e-4:
        return recent[-1].step, {
            "avg_direction_consistency": round(avg_direction, 4),
            "metric_range": round(metric_range, 6),
        }
    return None


def _detect_drift_phase(steps: list[TrainingStep], config: ZombieConfig) -> tuple[int, dict] | None:
    """检测灵魂分离：loss 和 metric 反向（loss 下降但 metric 也下降）。"""
    if len(steps) < 4:
        return None

    finite = [s for s in steps if _is_finite(s.train_loss) and _is_finite(s.val_metric)]
    if len(finite) < 4:
        return None

    losses = [s.train_loss for s in finite]
    metrics = [s.val_metric for s in finite]

    # 简单相关系数
    n = len(losses)
    mean_l = sum(losses) / n
    mean_m = sum(metrics) / n
    num = sum((losses[i] - mean_l) * (metrics[i] - mean_m) for i in range(n))
    den_l = math.sqrt(sum((l - mean_l) ** 2 for l in losses))
    den_m = math.sqrt(sum((m - mean_m) ** 2 for m in metrics))
    if den_l < 1e-12 or den_m < 1e-12:
        return None

    corr = num / (den_l * den_m)

    # 理想：loss↓ metric↑，correlation 应为负
    # 变灵异：loss↓ metric↓（正相关）或两者都朝坏方向走
    # 所以我们要找 correlation > threshold（本该反向的信号现在同向）
    if corr > -config.drift_correlation_threshold:
        return finite[-1].step, {
            "loss_metric_correlation": round(corr, 4),
            "expected": "negative",
        }
    return None


def assess_zombie(trace: TrainingTrace, config: ZombieConfig = DEFAULT_ZOMBIE_CONFIG) -> ZombieReport:
    """评估训练 run 的生命状态。同时把 trace_contract 验证结果挂到 report.contract。"""
    contract = validate_trace(trace).to_dict()
    report = _assess_zombie_inner(trace, config)
    report.contract = contract
    return report


def _assess_zombie_inner(trace: TrainingTrace, config: ZombieConfig) -> ZombieReport:
    """评估训练 run 的生命状态 (不含 contract 检查, 由 wrapper 注入)。

    按严重度依次检测：
      1. NaN / Inf 污染 → ZOMBIE
      2. 梯度爆炸 → INFECTED (组合规则) / SUSPECT (单点 legacy)
      3. 冻僵 → INFECTED
      4. 假平台 → SUSPECT
      5. 灵魂分离 → SUSPECT
      6. 健康 → ALIVE
    """
    if not trace.steps:
        return ZombieReport(
            run_id=trace.run_id,
            severity="SUSPECT",
            zombie_class="HEALTHY",
            infection_step=0,
            reason="empty trace — cannot assess liveness",
        )

    # NaN/Inf 检测优先于观察步数门槛：1 步就能判定死亡
    infection_step, zombie_class, evidence = _scan_nan_inf(trace.steps)
    if zombie_class == "NAN_HOST":
        return ZombieReport(
            run_id=trace.run_id,
            severity="ZOMBIE",
            zombie_class="NAN_HOST",
            infection_step=infection_step,
            reason=f"NaN 感染扩散中 — 训练信号已死，但 run 仍在产出数字",
            evidence=evidence,
            recommendation="立即终止训练。丧尸的分数不代表任何东西，它只是数学夹子的副产品。",
        )
    if zombie_class == "INF_HOST":
        return ZombieReport(
            run_id=trace.run_id,
            severity="ZOMBIE",
            zombie_class="INF_HOST",
            infection_step=infection_step,
            reason=f"Inf 感染 — 信号尺度已脱离定义域",
            evidence=evidence,
            recommendation="立即终止。降低学习率或启用梯度裁剪后重启。",
        )

    # NaN/Inf 之外的检测需要足够观察步数
    if len(trace.steps) < config.min_observation_steps:
        return ZombieReport(
            run_id=trace.run_id,
            severity="SUSPECT",
            zombie_class="HEALTHY",
            infection_step=0,
            reason=f"insufficient observation ({len(trace.steps)} < {config.min_observation_steps})",
            evidence={"steps_observed": len(trace.steps)},
        )

    # 2. 梯度爆炸 (组合规则)
    result = _detect_explosive(trace.steps, config)
    if result:
        step, severity, ev = result
        if severity == "INFECTED":
            rule = ev.get("rule", "combined")
            if rule == "post_clip_combined":
                reason = (
                    f"post-clip 梯度持续超过 clip × {ev['threshold_factor']} "
                    f"({ev['post_clip_grad_norm']} > {ev['gradient_clip']} × {ev['threshold_factor']}) "
                    f"持续 {ev['persist_steps']} 步且 loss 未改善"
                )
                rec = "梯度裁剪没起作用。检查 clip 阈值, 或考虑降低 LR / 重启训练。"
            else:
                reason = (
                    f"pre-clip 梯度持续异常 ({ev['grad_norm']} > {ev['abs_threshold']} 且 "
                    f"> {ev['spike_ratio']} × 近期中位 {ev['recent_median']}) "
                    f"持续 {ev['persist_steps']} 步且 loss 未改善"
                )
                rec = "看起来是真发散, 不只是 transformer 启动期高梯度。建议启用 grad clip 并降 LR。"
            return ZombieReport(
                run_id=trace.run_id,
                severity="INFECTED",
                zombie_class="INF_HOST",
                infection_step=step,
                reason=reason,
                evidence=ev,
                recommendation=rec,
            )
        # SUSPECT — 单点 legacy 命中, 组合判定不齐
        return ZombieReport(
            run_id=trace.run_id,
            severity="SUSPECT",
            zombie_class="INF_HOST",
            infection_step=step,
            reason=(
                f"单点高 pre-clip grad_norm={ev['grad_norm']} > {ev['threshold']}, "
                f"但未通过持续/改善组合判定。"
            ),
            evidence=ev,
            recommendation="如需更精确判定, 训练侧 trace 应同时写 post_clip_grad_norm + gradient_clip。",
        )

    # 3. 冻僵肢体
    result = _detect_frozen(trace.steps, config)
    if result:
        step, ev = result
        return ZombieReport(
            run_id=trace.run_id,
            severity="INFECTED",
            zombie_class="FROZEN_LIMB",
            infection_step=step,
            reason="梯度消失 + loss 不动 — 训练已僵死，表面看起来还在 epoch",
            evidence=ev,
            recommendation="重启训练。学习率可能太小，或模型初始化有问题。",
        )

    # 4. 假平台
    result = _detect_fake_plateau(trace.steps, config)
    if result:
        step, ev = result
        return ZombieReport(
            run_id=trace.run_id,
            severity="SUSPECT",
            zombie_class="FAKE_PLATEAU",
            infection_step=step,
            reason=f"方向一致性已崩塌（{ev['avg_direction_consistency']} < {config.direction_collapse_threshold}）但 metric 仍有微动",
            evidence=ev,
            recommendation="观察 — metric 的微动可能只是噪声。继续训练 10 步确认。",
        )

    # 5. 灵魂分离
    result = _detect_drift_phase(trace.steps, config)
    if result:
        step, ev = result
        return ZombieReport(
            run_id=trace.run_id,
            severity="SUSPECT",
            zombie_class="DRIFT_PHASE",
            infection_step=step,
            reason=f"loss 和 metric 反向（corr={ev['loss_metric_correlation']}）— train/val 脱钩",
            evidence=ev,
            recommendation="检查数据集是否错配，或模型是否在训练集上过拟合。",
        )

    # 健康
    return ZombieReport(
        run_id=trace.run_id,
        severity="ALIVE",
        zombie_class="HEALTHY",
        infection_step=-1,
        reason="未检测到丧尸特征，run 仍在正常进化",
        evidence={"observed_steps": len(trace.steps)},
        recommendation="继续训练。",
    )


def zombie_payload(trace: TrainingTrace, config: ZombieConfig = DEFAULT_ZOMBIE_CONFIG) -> dict:
    from dataclasses import asdict
    return asdict(assess_zombie(trace, config))


def zombie_report(trace: TrainingTrace, config: ZombieConfig = DEFAULT_ZOMBIE_CONFIG) -> str:
    result = assess_zombie(trace, config)
    severity_icon = {
        "ALIVE": "[OK]",
        "SUSPECT": "[?]",
        "INFECTED": "[!]",
        "ZOMBIE": "[X]",
    }.get(result.severity, "")

    lines = [
        f"Tensorearch zombie assessment",
        f"run_id={result.run_id}",
        f"severity={result.severity} {severity_icon}",
        f"zombie_class={result.zombie_class}",
        f"infection_step={result.infection_step}",
        f"reason={result.reason}",
    ]
    if result.evidence:
        lines.append(f"evidence={result.evidence}")
    if result.recommendation:
        lines.append(f"recommendation={result.recommendation}")
    return "\n".join(lines)


def zombie_report_json(trace: TrainingTrace, config: ZombieConfig = DEFAULT_ZOMBIE_CONFIG) -> str:
    import json
    return json.dumps(zombie_payload(trace, config), ensure_ascii=False, indent=2)
