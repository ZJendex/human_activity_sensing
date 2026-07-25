"""Render the quantitative and conceptual figures for the model comparison page."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "model_comparison"
OUT.mkdir(parents=True, exist_ok=True)


def save_mve_chart() -> None:
    """Compare radar-only HMR methods under the same M4Human protocols."""
    methods = [
        "mmMesh (RPC)",
        "P4Transformer (RPC)",
        "RT-Pose (RT)",
        "RETR (RT)",
        "RT-Mesh (RT)",
    ]
    values = {
        "S1 random": [132.7, 90.4, 100.7, 97.1, 90.9],
        "S2 cross-subject": [170.1, 140.8, 148.1, 169.7, 135.1],
        "S3 cross-action": [173.8, 147.8, 152.8, 163.1, 143.1],
    }
    colors = ["#2F6F9F", "#D98E32", "#667085"]

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=150)
    y = np.arange(len(methods))
    height = 0.22
    offsets = [-height, 0, height]

    for (label, series), color, offset in zip(values.items(), colors, offsets):
        bars = ax.barh(
            y + offset,
            series,
            height=height * 0.86,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9, color="#27313A")

    ax.set_yticks(y, methods, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 190)
    ax.set_xlabel("Mean vertex error (mm) — lower is better", fontsize=10.5)
    ax.set_title(
        "Radar-only human mesh reconstruction on M4Human",
        loc="left",
        fontsize=17,
        fontweight="bold",
        color="#1F2933",
        pad=34,
    )
    ax.text(
        0,
        1.035,
        "Same dataset and all-action protocols; errors are measured in the world frame without root/Procrustes alignment.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#52606D",
        va="bottom",
    )
    ax.legend(
        loc="lower right",
        frameon=False,
        ncol=3,
        fontsize=9.5,
        bbox_to_anchor=(1, -0.16),
    )
    ax.grid(axis="x", color="#D9E2EC", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, colors="#334E68")
    fig.text(
        0.012,
        0.012,
        "Source: M4Human, Table 2. RT = radar tensor; RPC = radar point cloud.",
        fontsize=8.5,
        color="#6B7280",
    )
    fig.subplots_adjust(left=0.23, right=0.96, top=0.82, bottom=0.19)
    fig.savefig(OUT / "m4human_same_benchmark_mve.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _box(ax, x, y, w, h, title, body, face, edge):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.035, title, fontsize=11.5, fontweight="bold", color="#1F2933", va="top")
    ax.text(x + 0.018, y + h - 0.085, body, fontsize=9.1, color="#405261", va="top", linespacing=1.35)


def _arrow(ax, start, end, color="#7B8794"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.5,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def save_design_map() -> None:
    """Show the design logic that links the eight works."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.96,
        "How radar human models evolved",
        fontsize=19,
        fontweight="bold",
        color="#1F2933",
        va="top",
    )
    ax.text(
        0.02,
        0.91,
        "The key design choice is what information survives preprocessing—and where the model spends spatial and temporal compute.",
        fontsize=10.5,
        color="#52606D",
        va="top",
    )

    columns = [0.03, 0.355, 0.68]
    widths = [0.285, 0.285, 0.285]
    headers = [
        ("1 · Dense projections", "RF-Pose · MMVR", "#E8F1FA", "#2F6F9F"),
        ("2 · Sparse point clouds", "mRI · MM-Fi · mmMesh · Argus", "#FFF2E2", "#D98E32"),
        ("3 · Rich tensor / fusion", "MVDoppler-Pose · M4Human", "#EAF5EE", "#37805B"),
    ]
    for x, w, (title, names, face, edge) in zip(columns, widths, headers):
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.12),
                w,
                0.72,
                boxstyle="round,pad=0.014,rounding_size=0.025",
                linewidth=1.2,
                edgecolor=edge,
                facecolor="#FFFFFF",
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.72),
                w,
                0.12,
                boxstyle="round,pad=0.014,rounding_size=0.025",
                linewidth=0,
                facecolor=face,
            )
        )
        ax.text(x + 0.018, 0.805, title, fontsize=12, fontweight="bold", color="#1F2933", va="top")
        ax.text(x + 0.018, 0.76, names, fontsize=9.2, color=edge, va="top")

    _box(ax, 0.055, 0.55, 0.235, 0.12, "Input", "Horizontal + vertical\nrange/angle heatmaps", "#F5F9FD", "#8BB8D8")
    _box(ax, 0.055, 0.36, 0.235, 0.12, "Encoder", "3D spatiotemporal CNNs;\ntwo views concatenate", "#F5F9FD", "#8BB8D8")
    _box(ax, 0.055, 0.17, 0.235, 0.12, "Output", "2D confidence maps + PAF;\nscore with OKS/AP", "#F5F9FD", "#8BB8D8")
    _arrow(ax, (0.172, 0.55), (0.172, 0.49))
    _arrow(ax, (0.172, 0.36), (0.172, 0.30))

    _box(ax, 0.38, 0.55, 0.235, 0.12, "Input", "CFAR-selected RPC: xyz +\nrange/Doppler/intensity", "#FFF8EF", "#E2AF6C")
    _box(ax, 0.38, 0.36, 0.235, 0.12, "Encoder", "PointTransformer/PointNet++;\nattention + LSTM over time", "#FFF8EF", "#E2AF6C")
    _box(ax, 0.38, 0.17, 0.235, 0.12, "Output", "3D joints or SMPL mesh;\nscore with MPJPE/MVE", "#FFF8EF", "#E2AF6C")
    _arrow(ax, (0.497, 0.55), (0.497, 0.49))
    _arrow(ax, (0.497, 0.36), (0.497, 0.30))

    _box(ax, 0.705, 0.55, 0.235, 0.12, "Input", "Range + Doppler views, or\nfull 3D radar tensor", "#F1F8F4", "#74A98A")
    _box(ax, 0.705, 0.36, 0.235, 0.12, "Spend compute selectively", "Cross-modal/view attention, or\n2D localization → local 3D RoI", "#F1F8F4", "#74A98A")
    _box(ax, 0.705, 0.17, 0.235, 0.12, "Output", "3D pose or SMPL-X mesh +\nglobal translation", "#F1F8F4", "#74A98A")
    _arrow(ax, (0.822, 0.55), (0.822, 0.49))
    _arrow(ax, (0.822, 0.36), (0.822, 0.30))

    _arrow(ax, (0.315, 0.48), (0.35, 0.48), color="#A0AEC0")
    _arrow(ax, (0.64, 0.48), (0.675, 0.48), color="#A0AEC0")
    ax.text(0.332, 0.505, "less dense", fontsize=8.3, color="#7B8794", ha="center")
    ax.text(0.657, 0.505, "recover detail", fontsize=8.3, color="#7B8794", ha="center")

    ax.text(
        0.5,
        0.055,
        "M4Human’s intuition: use a cheap bird’s-eye view to find the person, then preserve the local 3D tensor for mesh detail.",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color="#2E5E46",
    )
    fig.savefig(OUT / "model_design_evolution.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    save_mve_chart()
    save_design_map()
