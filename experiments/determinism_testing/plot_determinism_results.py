#!/usr/bin/env python3
"""
Plot Determinism Benchmark Results

Generates publication-quality figures from the determinism benchmark CSV results.

Usage:
    pixi run -e torch determinism-plot
    pixi run -e torch determinism-plot --input results/determinism_results.csv
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def load_results(csv_path: str) -> tuple[list[float], list[float], list[float], list[float]]:
    """Load results from CSV file."""
    retriever_gradients = []
    retriever_losses = []
    pubsub_gradients = []
    pubsub_losses = []

    with open(csv_path, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue

            executor = parts[0]
            gradient = float(parts[2]) if parts[2] != 'nan' else float('nan')
            loss = float(parts[3])

            if executor == 'retriever':
                retriever_gradients.append(gradient)
                retriever_losses.append(loss)
            elif executor == 'pubsub':
                pubsub_gradients.append(gradient)
                pubsub_losses.append(loss)

    return retriever_gradients, retriever_losses, pubsub_gradients, pubsub_losses


def compute_analytical_gradient(g=9.81, e=0.8, dt=0.01, T=100, x_init=1.0, x_target=0.5, theta=3.0, eps=1e-6):
    """Compute analytical gradient using finite differences."""
    def simulate_loss(theta_val: float) -> float:
        x = x_init
        v = theta_val
        for t in range(T):
            v_pred = v - g * dt
            x_pred = x + v_pred * dt
            if x_pred < 0.0:
                x = 0.0
                v = -e * v_pred
            else:
                x = x_pred
                v = v_pred
        return (x - x_target) ** 2

    loss_center = simulate_loss(theta)
    loss_plus = simulate_loss(theta + eps)
    gradient = (loss_plus - loss_center) / eps
    return gradient


def plot_gradient_histogram(
    retriever_gradients: list[float],
    pubsub_gradients: list[float],
    output_path: str = None,
    show: bool = True
):
    """Plot single histogram comparing gradient distributions."""
    # Single column figure size for RSS paper (3.5 inches width is typical)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Filter NaN values
    ret_grads = [g for g in retriever_gradients if not np.isnan(g)]
    ps_grads = [g for g in pubsub_gradients if not np.isnan(g)]

    # Compute statistics
    ret_mean = np.mean(ret_grads) if ret_grads else 0
    ret_std = np.std(ret_grads) if ret_grads else 0
    ps_mean = np.mean(ps_grads) if ps_grads else 0
    ps_std = np.std(ps_grads) if ps_grads else 0

    ret_unique = len(set(f"{g:.10f}" for g in ret_grads))
    ps_unique = len(set(f"{g:.10f}" for g in ps_grads))

    # Compute analytical/true gradient
    true_grad = compute_analytical_gradient()

    # Pub/Sub histogram with larger bins for better distribution
    ps_bins = min(ps_unique, 60)  # Use more bins to show distribution
    ax.hist(ps_grads, bins=ps_bins, color='#f97316', alpha=0.7, edgecolor='white',
            label=f'Pub/Sub: μ={ps_mean:.3f}, σ={ps_std:.3f}')

    # Pub/Sub mean line
    ax.axvline(ps_mean, color='#c2410c', linestyle='--', linewidth=2, alpha=0.8)

    # Analytical/true gradient line (red for visibility)
    ax.axvline(true_grad, color='#dc2626', linestyle='--', linewidth=2.5,
               label=f'True: {true_grad:.4f}', zorder=15, alpha=0.95)

    # Retriever: Solid line for deterministic value
    if ret_unique <= 2:
        # Draw a prominent vertical line for the single value
        y_max = ax.get_ylim()[1]
        ax.axvline(ret_mean, color='#2563eb', linestyle='-', linewidth=4,
                   label=f'Retriever: μ={ret_mean:.4f}, σ={ret_std:.1e}', zorder=10)

        # Add annotation showing all runs converged
        ax.annotate(f'{len(ret_grads)} runs\n(identical)',
                   xy=(ret_mean, y_max * 0.7),
                   xytext=(ret_mean + 0.05, y_max * 0.7),
                   fontsize=7, color='#2563eb', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#2563eb', linewidth=1.5),
                   arrowprops=dict(arrowstyle='->', color='#2563eb', lw=1.5))

        # Show error from true gradient
        error = abs(ret_mean - true_grad)
        ax.text(0.05, 0.88, f'Error: {error:.1e}',
                transform=ax.transAxes, fontsize=6.5,
                bbox=dict(boxstyle='round', facecolor='#fee2e2', alpha=0.85, edgecolor='#dc2626'))
    else:
        # Use histogram if there are multiple unique values
        ax.hist(ret_grads, bins=min(ret_unique, 20), color='#2563eb', alpha=0.7, edgecolor='white',
                label=f'Retriever (μ={ret_mean:.4f}, σ={ret_std:.2e}, n={ret_unique} unique)')
        ax.axvline(ret_mean, color='#1e40af', linestyle='--', linewidth=2)

    ax.set_title('Path Gradient: Retriever vs. Pub/Sub vs. Analytic', fontsize=9, fontweight='bold', pad=8)
    ax.set_xlabel(r'$\partial L / \partial \theta$', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc='upper right', fontsize=6, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved histogram to: {output_path}")

        # Also save PDF for publication
        pdf_path = output_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved PDF to: {pdf_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_combined_figure(
    retriever_gradients: list[float],
    pubsub_gradients: list[float],
    output_path: str = None,
    show: bool = True
):
    """Create a single combined figure suitable for a paper."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Filter NaN values
    ret_grads = [g for g in retriever_gradients if not np.isnan(g)]
    ps_grads = [g for g in pubsub_gradients if not np.isnan(g)]

    # Compute statistics
    ret_mean = np.mean(ret_grads) if ret_grads else 0
    ret_std = np.std(ret_grads) if ret_grads else 0
    ps_mean = np.mean(ps_grads) if ps_grads else 0
    ps_std = np.std(ps_grads) if ps_grads else 0

    # Determine bin range
    all_grads = ret_grads + ps_grads
    if all_grads:
        min_grad = min(all_grads)
        max_grad = max(all_grads)
        bins = np.linspace(min_grad - 0.1 * abs(max_grad - min_grad),
                          max_grad + 0.1 * abs(max_grad - min_grad),
                          31)
    else:
        bins = 30

    # Plot overlapping histograms
    ax.hist(ret_grads, bins=bins, color='#2563eb', alpha=0.7, edgecolor='white',
            label=f'Retriever (std={ret_std:.2e})')
    ax.hist(ps_grads, bins=bins, color='#f97316', alpha=0.7, edgecolor='white',
            label=f'Pub/Sub (std={ps_std:.4f})')

    # Add mean lines
    ax.axvline(ret_mean, color='#1e40af', linestyle='--', linewidth=2)
    ax.axvline(ps_mean, color='#c2410c', linestyle='--', linewidth=2)

    ax.set_xlabel(r'Path Gradient $\partial L / \partial \theta$', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Gradient Distribution: Retriever vs Pub/Sub', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined figure to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_summary_table(
    retriever_gradients: list[float],
    retriever_losses: list[float],
    pubsub_gradients: list[float],
    pubsub_losses: list[float]
):
    """Print summary statistics table."""
    ret_grads = [g for g in retriever_gradients if not np.isnan(g)]
    ps_grads = [g for g in pubsub_gradients if not np.isnan(g)]

    ret_unique = len(set(f"{g:.10f}" for g in ret_grads))
    ps_unique = len(set(f"{g:.10f}" for g in ps_grads))

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Metric':<25} {'Retriever':<15} {'Pub/Sub':<15}")
    print("-" * 60)
    print(f"{'Runs':<25} {len(ret_grads):<15} {len(ps_grads):<15}")
    print(f"{'Unique Gradients':<25} {ret_unique:<15} {ps_unique:<15}")
    print(f"{'Gradient Mean':<25} {np.mean(ret_grads):<15.6f} {np.mean(ps_grads):<15.6f}")
    print(f"{'Gradient Std':<25} {np.std(ret_grads):<15.6e} {np.std(ps_grads):<15.6f}")
    print(f"{'Loss Mean':<25} {np.mean(retriever_losses):<15.6f} {np.mean(pubsub_losses):<15.6f}")
    print(f"{'Loss Std':<25} {np.std(retriever_losses):<15.6e} {np.std(pubsub_losses):<15.6f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot determinism benchmark results")
    parser.add_argument("--input", "-i", type=str,
                        default="experiments/determinism_testing/results/determinism_results.csv",
                        help="Input CSV file path")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="experiments/determinism_testing/results",
                        help="Output directory for plots")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display plots (just save)")
    parser.add_argument("--combined", action="store_true",
                        help="Generate combined overlay figure")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print("Run the benchmark first: python bouncing_ball_benchmark.py")
        sys.exit(1)

    # Load results
    ret_grads, ret_losses, ps_grads, ps_losses = load_results(args.input)

    if not ret_grads or not ps_grads:
        print("Error: No data found in CSV file")
        sys.exit(1)

    print(f"Loaded {len(ret_grads)} Retriever runs, {len(ps_grads)} Pub/Sub runs")

    # Print summary
    print_summary_table(ret_grads, ret_losses, ps_grads, ps_losses)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot side-by-side histogram
    hist_path = os.path.join(args.output_dir, "gradient_histogram.png")
    plot_gradient_histogram(ret_grads, ps_grads, hist_path, show=not args.no_show)

    # Always plot combined figure for better distribution visualization
    combined_path = os.path.join(args.output_dir, "gradient_combined.png")
    plot_combined_figure(ret_grads, ps_grads, combined_path, show=not args.no_show)


if __name__ == "__main__":
    main()
