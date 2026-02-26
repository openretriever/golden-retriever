#!/usr/bin/env python3
"""
Visualize gradient-based optimization of initial velocity through Retriever pipeline.

Three-panel figure:
  Top:    Loss landscape L(θ) with gradient descent paths from B starting points
  Middle: Loss convergence curves (one line per θ_i) over optimization steps
  Bottom: Ball trajectories before (grey batch) and after (colored) optimization

Usage:
    python plot_optimization.py                    # default B=6, steps=80
    python plot_optimization.py --batch-size 8 --steps 120
    python plot_optimization.py --output results/optimization.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from physics import PhysicsConfig, finite_difference_gradient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bouncing_ball_backprop import optimize_batch


# =============================================================================
# Helpers
# =============================================================================

def simulate_trajectory(theta: float, cfg: PhysicsConfig) -> Tuple[List[float], List[float]]:
    """Return (time, height) arrays for a single θ."""
    x, v = cfg.x_init, theta
    times, positions = [0.0], [x]
    for _ in range(1, cfg.T):
        v_pred = v - cfg.g * cfg.dt
        x_pred = x + v_pred * cfg.dt
        if x_pred < 0.0:
            x, v = 0.0, -cfg.e * v_pred
        else:
            x, v = x_pred, v_pred
        times.append(len(times) * cfg.dt)
        positions.append(x)
    return times, positions


def compute_loss_landscape(theta_range: np.ndarray, cfg: PhysicsConfig) -> np.ndarray:
    """Compute L(θ) for each θ in theta_range using finite simulation."""
    return np.array([finite_difference_gradient(th, cfg, eps=1e-8)[1] for th in theta_range])


# =============================================================================
# Plotting
# =============================================================================

def plot_optimization(
    cfg: PhysicsConfig,
    theta_init: List[float],
    lr: float = 0.15,
    steps: int = 80,
    output_path: str = None,
):
    # --- Run optimization ---
    print("Running batch gradient descent...")
    result = optimize_batch(theta_init, cfg, lr=lr, steps=steps, print_every=steps + 1)
    theta_history = np.array(result["theta_history"])   # [steps+1, B]
    loss_history = np.array(result["loss_history"])     # [steps, B]
    theta_final = result["final_theta"]

    print(f"Final θ values: {[f'{t:.4f}' for t in theta_final]}")

    # --- Loss landscape ---
    theta_scan = np.linspace(0.5, max(theta_init) + 1.5, 300)
    print("Computing loss landscape...")
    L_scan = compute_loss_landscape(theta_scan, cfg)

    # =========================================================================
    # Figure layout: 3 rows
    # =========================================================================
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1,
        figsize=(3.5, 6.8),
        gridspec_kw={"height_ratios": [1.0, 0.75, 0.9]},
    )
    fig.subplots_adjust(hspace=0.50)

    B = len(theta_init)
    cmap = plt.cm.plasma
    colors = [cmap(0.15 + 0.65 * i / max(B - 1, 1)) for i in range(B)]

    # =========================================================================
    # TOP PANEL: Loss landscape + gradient descent paths
    # =========================================================================
    ax = ax_top

    ax.plot(theta_scan, L_scan, color="#1e293b", linewidth=1.5, zorder=2)
    ax.fill_between(theta_scan, L_scan, alpha=0.08, color="#1e293b")

    for i in range(B):
        th_path = theta_history[:, i]
        # Reconstruct loss along the path
        loss_path = np.array(
            [loss_history[s, i] if s < len(loss_history) else loss_history[-1, i]
             for s in range(len(th_path) - 1)]
        )
        _, final_loss = finite_difference_gradient(th_path[-1], cfg, eps=1e-8)
        loss_path = np.append(loss_path, final_loss)

        ax.plot(th_path, loss_path, color=colors[i], linewidth=0.8, alpha=0.6, zorder=3)

        # Start marker
        _, l0 = finite_difference_gradient(theta_init[i], cfg, eps=1e-8)
        ax.scatter(theta_init[i], l0, color=colors[i], s=22, zorder=5,
                   marker="o", edgecolors="white", linewidths=0.4)

        # End arrow
        if len(th_path) >= 3:
            ax.annotate(
                "", xy=(th_path[-1], loss_path[-1]),
                xytext=(th_path[-3], loss_path[-3]),
                arrowprops=dict(arrowstyle="->", color=colors[i],
                                lw=0.8, mutation_scale=8),
                zorder=6,
            )

    # Mark global minimum of scanned landscape
    idx_min = int(np.argmin(L_scan))
    theta_star = theta_scan[idx_min]
    ax.axvline(theta_star, color="#dc2626", linewidth=1.0, linestyle="--",
               alpha=0.7, zorder=2)
    ax.scatter(theta_star, L_scan[idx_min], color="#dc2626", s=50, zorder=7,
               marker="*", label=f"$\\theta^*\\approx{theta_star:.2f}$")

    ax.set_xlabel(r"Initial velocity $\theta$", fontsize=8)
    ax.set_ylabel(r"Loss $L(\theta)$", fontsize=8)
    ax.set_title("Loss landscape & gradient descent", fontsize=8, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(theta_scan[0], theta_scan[-1])
    ax.set_ylim(bottom=-0.02 * L_scan.max())

    # =========================================================================
    # MIDDLE PANEL: Loss convergence per θ_i
    # =========================================================================
    ax = ax_mid

    for i in range(B):
        loss_curve = loss_history[:, i]   # [steps]
        ax.plot(range(len(loss_curve)), loss_curve,
                color=colors[i], linewidth=0.9, alpha=0.85, zorder=3)

    # Convergence threshold guide
    ax.axhline(1e-4, color="#94a3b8", linewidth=0.7, linestyle=":", zorder=1)
    ax.text(steps * 0.98, 1e-4 * 1.8, "1e-4", color="#94a3b8",
            fontsize=6, ha="right", va="bottom")

    ax.set_xlabel("Gradient step", fontsize=8)
    ax.set_ylabel(r"Loss $L(\theta_i)$", fontsize=8)
    ax.set_title("Loss convergence per initial $\\theta$", fontsize=8, fontweight="bold")

    # Use log scale only if losses span multiple orders of magnitude
    min_loss = loss_history.min()
    max_loss = loss_history.max()
    if max_loss > 0 and min_loss > 0 and (max_loss / min_loss) > 100:
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(min_loss * 0.5, 1e-8))
    ax.set_xlim(0, steps - 1)
    ax.tick_params(labelsize=7)

    # =========================================================================
    # BOTTOM PANEL: Ball trajectories (before → after)
    # =========================================================================
    ax = ax_bot

    # Compute actual max height across all trajectories for y-axis
    all_positions = []
    for th in list(theta_init) + list(theta_final):
        _, pos = simulate_trajectory(th, cfg)
        all_positions.extend(pos)
    y_max = max(all_positions) * 1.12

    # Ground line
    ax.axhline(0, color="#374151", linewidth=0.8, zorder=1)

    # Target line
    ax.axhline(cfg.x_target, color="#dc2626", linewidth=1.0,
               linestyle="--", alpha=0.8, zorder=2,
               label=f"target $h={cfg.x_target}$")

    # Initial trajectories (grey)
    for th in theta_init:
        times, pos = simulate_trajectory(th, cfg)
        ax.plot(times, pos, color="#94a3b8", linewidth=0.6, alpha=0.45, zorder=3)

    # "initial batch" label — anchor at a time where grey trajectories are dense
    t_anchor = cfg.T * cfg.dt * 0.15
    label_heights = []
    for th in theta_init:
        times_th, pos_th = simulate_trajectory(th, cfg)
        idx = min(int(0.15 * cfg.T), len(pos_th) - 1)
        label_heights.append(pos_th[idx])
    label_h = float(np.mean(label_heights))
    ax.text(t_anchor, label_h * 1.08, "initial\nbatch",
            color="#64748b", fontsize=6.5, ha="center", va="bottom",
            style="italic")

    # Optimized trajectories (colored)
    for i, th in enumerate(theta_final):
        times, pos = simulate_trajectory(th, cfg)
        ax.plot(times, pos, color=colors[i], linewidth=1.0, alpha=0.85, zorder=4)

    # Mark final heights of optimized trajectories
    for i, th in enumerate(theta_final):
        times_f, pos_f = simulate_trajectory(th, cfg)
        ax.scatter([times_f[-1]], [pos_f[-1]], color=colors[i],
                   s=18, zorder=6, marker="o", edgecolors="none")

    ax.set_xlabel(r"Time $t$ (s)", fontsize=8)
    ax.set_ylabel(r"Height $x(t)$", fontsize=8)
    ax.set_title("Trajectories: grey=initial, colored=optimized", fontsize=8,
                 fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    ax.tick_params(labelsize=7)
    ax.set_ylim(-0.05, y_max)
    ax.set_xlim(0, cfg.T * cfg.dt)

    # =========================================================================
    # Save
    # =========================================================================
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
        if output_path.endswith(".png"):
            pdf_path = output_path.replace(".png", ".pdf")
            plt.savefig(pdf_path, bbox_inches="tight")
            print(f"Saved: {pdf_path}")
    else:
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/optimization.png", dpi=300, bbox_inches="tight")
        print("Saved: results/optimization.png")

    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot gradient-based optimization")
    parser.add_argument("--batch-size", "-B", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--output", default="results/optimization.png")
    args = parser.parse_args()

    cfg = PhysicsConfig(g=9.81, e=0.8, dt=0.01, T=100, x_target=0.5, x_init=1.0)
    theta_init = [2.0 + i * 1.0 for i in range(args.batch_size)]

    plot_optimization(cfg, theta_init, lr=args.lr, steps=args.steps,
                      output_path=args.output)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
