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

    # 梯度范数过大 = 爆炸尸体
    inf_grad_threshold: float = 50.0

    # 梯度范数过小且 loss 不动 = 冻僵肢体
    dead_grad_threshold: float = 1e-6

    # 方向一致性崩塌 = 行走的尸体
    direction_collapse_threshold: float = 0.15

    # Loss 降而 metric 也降（或反向）= 灵魂分离
    drift_correlation_threshold: float = -0.3

    # 至少多少步才能做丧尸判定
    min_observation_steps: int = 3


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


def _detect_explosive(steps: list[TrainingStep], config: ZombieConfig) -> tuple[int, dict] | None:
    """检测爆炸尸体：梯度爆炸超过阈值。"""
    for s in steps:
        if _is_finite(s.grad_norm) and s.grad_norm > config.inf_grad_threshold:
            return s.step, {
                "grad_norm": s.grad_norm,
                "threshold": config.inf_grad_threshold,
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
    """评估训练 run 的生命状态。

    按严重度依次检测：
      1. NaN / Inf 污染 → ZOMBIE
      2. 梯度爆炸 → INFECTED
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

    # 2. 梯度爆炸
    result = _detect_explosive(trace.steps, config)
    if result:
        step, ev = result
        return ZombieReport(
            run_id=trace.run_id,
            severity="INFECTED",
            zombie_class="INF_HOST",
            infection_step=step,
            reason=f"梯度爆炸（grad_norm={ev['grad_norm']:.2f} > {config.inf_grad_threshold}）",
            evidence=ev,
            recommendation="启用梯度裁剪。继续跑可能变 ZOMBIE。",
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
