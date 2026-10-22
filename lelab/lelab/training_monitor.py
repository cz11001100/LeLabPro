# Copyright 2025 The HuggingFace Inc. library.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training monitoring and overfitting detection for lelab.

Analyzes training metrics (loss, lr, grad_norm) to detect:
- Loss plateau (no improvement)
- Loss divergence (potential overfitting)
- Gradient instability
- Training convergence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingHealthReport:
    """Summary of training health based on metrics analysis."""

    status: str  # "healthy", "warning", "critical"
    loss_trend: str  # "decreasing", "plateau", "increasing", "oscillating"
    gradient_health: str  # "stable", "spiking", "vanishing"
    convergence_progress: float  # 0.0 to 1.0
    overfitting_risk: str  # "low", "medium", "high"
    recommendations: list[str]
    details: dict


def analyze_training_health(
    steps: list[int],
    losses: list[float | None],
    grad_norms: list[float | None],
    lrs: list[float | None],
    total_steps: int,
) -> TrainingHealthReport:
    """Analyze training metrics and return a health report.

    Args:
        steps: List of step numbers
        losses: List of loss values (None for missing)
        grad_norms: List of gradient norm values
        lrs: List of learning rate values
        total_steps: Total training steps (for progress calculation)

    Returns:
        TrainingHealthReport with status, trends, and recommendations
    """
    # Filter out None values
    valid_points = [(s, l, g, lr) for s, l, g, lr in zip(steps, losses, grad_norms, lrs) if l is not None]

    if len(valid_points) < 5:
        return TrainingHealthReport(
            status="healthy",
            loss_trend="insufficient_data",
            gradient_health="insufficient_data",
            convergence_progress=0.0,
            overfitting_risk="low",
            recommendations=["Continue training, collecting more data points"],
            details={"points_analyzed": len(valid_points)},
        )

    recent_losses = [l for _, l, _, _ in valid_points[-20:]]
    recent_grads = [g for _, _, g, _ in valid_points[-20:] if g is not None]

    # Analyze loss trend
    loss_trend = _analyze_loss_trend(recent_losses)

    # Analyze gradient health
    gradient_health = _analyze_gradient_health(recent_grads)

    # Detect overfitting risk (simplified - without val loss, we look for loss increase)
    overfitting_risk = _detect_overfitting_risk([l for _, l, _, _ in valid_points])

    # Calculate convergence progress
    initial_loss = valid_points[0][1] if valid_points[0][1] else 1.0
    current_loss = valid_points[-1][1] if valid_points[-1][1] else initial_loss
    progress = _calculate_convergence_progress(initial_loss, current_loss, len(valid_points), total_steps)

    # Generate recommendations
    recommendations = _generate_recommendations(
        loss_trend, gradient_health, overfitting_risk, progress, current_loss
    )

    # Overall status
    status = _determine_status(loss_trend, gradient_health, overfitting_risk)

    return TrainingHealthReport(
        status=status,
        loss_trend=loss_trend,
        gradient_health=gradient_health,
        convergence_progress=progress,
        overfitting_risk=overfitting_risk,
        recommendations=recommendations,
        details={
            "points_analyzed": len(valid_points),
            "initial_loss": initial_loss,
            "current_loss": current_loss,
            "min_loss": min(l for l in losses if l is not None) if any(l is not None for l in losses) else None,
            "avg_recent_loss": sum(recent_losses) / len(recent_losses) if recent_losses else None,
            "avg_recent_grad": sum(recent_grads) / len(recent_grads) if recent_grads else None,
        },
    )


def _analyze_loss_trend(losses: list[float]) -> str:
    """Determine the trend of recent losses."""
    if len(losses) < 5:
        return "insufficient_data"

    # Split into first half and second half
    mid = len(losses) // 2
    first_half_avg = sum(losses[:mid]) / mid
    second_half_avg = sum(losses[mid:]) / (len(losses) - mid)

    # Calculate variance in second half (oscillation detection)
    second_half_var = sum((l - second_half_avg) ** 2 for l in losses[mid:]) / (len(losses) - mid)
    relative_var = second_half_var / (second_half_avg ** 2) if second_half_avg > 0 else 0

    # Oscillating: high variance
    if relative_var > 0.1:
        return "oscillating"

    # Increasing: second half significantly worse
    if second_half_avg > first_half_avg * 1.1:
        return "increasing"

    # Plateau: minimal change
    if abs(second_half_avg - first_half_avg) / first_half_avg < 0.02:
        return "plateau"

    # Decreasing: second half better
    return "decreasing"


def _analyze_gradient_health(grad_norms: list[float]) -> str:
    """Analyze gradient norm stability."""
    if len(grad_norms) < 5:
        return "insufficient_data"

    avg_grad = sum(grad_norms) / len(grad_norms)

    # Vanishing gradients
    if avg_grad < 1e-5:
        return "vanishing"

    # Spiking: any gradient > 10x average
    max_grad = max(grad_norms)
    if max_grad > avg_grad * 10:
        return "spiking"

    return "stable"


def _detect_overfitting_risk(losses: list[float]) -> str:
    """Detect overfitting risk based on loss behavior.

    Without validation loss, we look for:
    - Loss increasing after initial decrease
    - Loss plateau at a high value
    """
    if len(losses) < 20:
        return "low"

    # Find minimum loss point
    min_loss = min(losses)
    min_idx = losses.index(min_loss)

    # If minimum was early and loss has been increasing since
    if min_idx < len(losses) * 0.3:
        recent_avg = sum(losses[-10:]) / 10
        if recent_avg > min_loss * 1.2:
            return "high"

    # If loss plateau at high value
    recent_losses = losses[-20:]
    recent_avg = sum(recent_losses) / len(recent_losses)
    if recent_avg > 1.0 and len(set(round(l, 2) for l in recent_losses)) < 5:
        return "medium"

    return "low"


def _calculate_convergence_progress(
    initial_loss: float, current_loss: float, points_seen: int, total_steps: int
) -> float:
    """Calculate how close training is to convergence (0.0 to 1.0)."""
    # Progress based on steps completed
    step_progress = points_seen / total_steps if total_steps > 0 else 0

    # Progress based on loss reduction
    if initial_loss > 0:
        loss_reduction = (initial_loss - current_loss) / initial_loss
        loss_progress = min(max(loss_reduction, 0), 1.0)
    else:
        loss_progress = 0

    # Combine: 60% weight on steps, 40% on loss reduction
    return 0.6 * step_progress + 0.4 * loss_progress


def _generate_recommendations(
    loss_trend: str,
    gradient_health: str,
    overfitting_risk: str,
    progress: float,
    current_loss: float,
) -> list[str]:
    """Generate actionable recommendations based on training health."""
    recs = []

    if loss_trend == "plateau":
        recs.append("Loss plateau detected. Consider: increasing learning rate, adding data augmentation, or training longer")

    if loss_trend == "increasing":
        recs.append("Loss is increasing. Consider: reducing learning rate, checking data quality, or reducing batch size")

    if loss_trend == "oscillating":
        recs.append("Loss is oscillating. Consider: reducing learning rate, increasing batch size, or using gradient clipping")

    if gradient_health == "spiking":
        recs.append("Gradient spikes detected. Enable gradient clipping (grad_clip_norm=1.0)")

    if gradient_health == "vanishing":
        recs.append("Gradients are vanishing. Check for dead neurons, consider increasing learning rate")

    if overfitting_risk == "high":
        recs.append("High overfitting risk. Consider: adding more data, stronger augmentation, or early stopping")

    if overfitting_risk == "medium":
        recs.append("Medium overfitting risk. Monitor closely, consider adding validation set")

    if progress > 0.9 and current_loss < 0.1:
        recs.append("Training appears converged. Consider testing the model or training longer for marginal gains")

    if progress < 0.3 and current_loss > 1.0:
        recs.append("Early training stage. Loss should decrease significantly in the next 20-30% of training")

    if not recs:
        recs.append("Training is progressing normally")

    return recs


def _determine_status(loss_trend: str, gradient_health: str, overfitting_risk: str) -> str:
    """Determine overall training status."""
    if gradient_health == "spiking" or overfitting_risk == "high" or loss_trend == "increasing":
        return "critical"

    if gradient_health == "vanishing" or overfitting_risk == "medium" or loss_trend in ("plateau", "oscillating"):
        return "warning"

    return "healthy"
