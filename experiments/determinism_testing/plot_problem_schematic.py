#!/usr/bin/env python3
"""
Generate schematic diagram of bouncing ball problem.

Creates a trajectory plot showing height vs time with bouncing pattern.
"""

import matplotlib.pyplot as plt
import numpy as np


def simulate_trajectory(theta, g=9.81, e=0.8, dt=0.01, T=100, x_init=1.0):
    """Simulate one trajectory."""
    x, v = x_init, theta
    times, positions = [0], [x]

    for t in range(1, T):
        v_pred = v - g * dt
        x_pred = x + v_pred * dt

        if x_pred < 0.0:
            x = 0.0
            v = -e * v_pred
        else:
            x = x_pred
            v = v_pred

        times.append(t * dt)
        positions.append(x)

    return np.array(times), np.array(positions)


def plot_problem_schematic():
    """Create trajectory schematic for RSS paper."""
    # Figure size for RSS paper (~1/3 column width)
    fig, ax = plt.subplots(figsize=(2.2, 1.6))

    # Multiple trajectories with different initial velocities (grey)
    thetas = [2.0, 2.5, 3.0, 3.5, 4.0]
    for theta in thetas:
        times, positions = simulate_trajectory(theta=theta, T=100)
        ax.plot(times, positions, color='#9ca3af', linewidth=0.8,
                alpha=0.5, zorder=3)

    # Target line
    ax.axhline(0.5, color='#dc2626', linewidth=1.5, linestyle='--',
               alpha=0.8, zorder=5)

    # Ground
    ax.axhline(0, color='#1f2937', linewidth=2.5, zorder=10)

    # Ball at initial position
    ax.plot(0, 1.0, 'o', color='#374151', markersize=10, zorder=15)

    # Velocity arrows on a few trajectories
    arrow_times = [0.15, 0.3, 0.45]
    for i, theta in enumerate([2.5, 3.0, 3.5]):
        times, positions = simulate_trajectory(theta=theta, T=100)
        for t_arrow in arrow_times:
            idx = int(t_arrow / 0.01)
            if idx < len(times) - 5:
                dx = times[idx + 5] - times[idx]
                dy = positions[idx + 5] - positions[idx]
                ax.arrow(times[idx], positions[idx], dx * 3, dy * 3,
                        head_width=0.03, head_length=0.04, fc='#6b7280',
                        ec='#6b7280', alpha=0.6, linewidth=0.8, zorder=4)

    # Labels
    ax.text(0.98, 0.52, 'x*', fontsize=8, color='#dc2626',
            ha='right', va='bottom', fontweight='bold')
    ax.text(0.02, 1.02, 'x₀', fontsize=8, color='#374151',
            ha='left', va='bottom', fontweight='bold')

    # Styling
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.05, 2.0)
    ax.set_xlabel('time', fontsize=8)
    ax.set_ylabel('h', fontsize=8, rotation=0, labelpad=10)
    ax.tick_params(labelsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.3)

    # Save
    output_path = 'experiments/determinism_testing/results/problem_schematic.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")

    output_pdf = 'experiments/determinism_testing/results/problem_schematic.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Saved to: {output_pdf}")

    plt.close()


if __name__ == "__main__":
    plot_problem_schematic()
