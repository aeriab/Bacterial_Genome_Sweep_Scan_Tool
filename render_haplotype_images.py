#!/usr/bin/env python3
"""
Render haplotype PNGs for the strain-filtered C. difficile windows.

Two figure types:
  1. filtered grid  -- N windows spread evenly across the genome, drawn from
     the dense w400 strain-filtered scan array (results/dense_w400/).
  2. before/after    -- matched windows (same genomic start, same seed) with
     no strain filter vs. with the per-window relatedness cap + MAF>=0.2
     clustering key, from results/real_w400_{nofix,fixed}_cmp/.

Channel 0 (allele state) is shown: -1 major, 0 missing, 1 minor.
Encoding per project convention (dual-channel int8 (N, samples, sites, 2)).
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

BASE = "/u/project/ngarud/baeria/Research/Summer26_Sims/Cdiff_Bromii_Clonality_Fixes"

# allele-state colours: major / missing / minor
CH0_COLORS = ["#2f3640", "#c4ccd2", "#3498db"]
CH0_CMAP = ListedColormap(CH0_COLORS)
CH0_LEGEND = [
    Patch(facecolor="#2f3640", edgecolor="#888", label="major allele"),
    Patch(facecolor="#c4ccd2", edgecolor="#888", label="missing"),
    Patch(facecolor="#3498db", edgecolor="#888", label="minor allele"),
]


def show(ax, img_ch0, title):
    ax.imshow(img_ch0, aspect="auto", cmap=CH0_CMAP, vmin=-1.5, vmax=1.5,
              interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def filtered_grid(out_path, n=12, seed=0):
    arr = np.load(f"{BASE}/results/dense_w400/real_windows.npy", mmap_mode="r")
    sm = np.load(f"{BASE}/results/dense_w400/sitemap_001.npy", mmap_mode="r")
    idx = np.linspace(0, arr.shape[0] - 1, n).astype(int)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows))
    axes = np.atleast_2d(axes)
    for k, i in enumerate(idx):
        ax = axes[k // cols, k % cols]
        mid_bp = int((int(sm[i].min()) + int(sm[i].max())) / 2)
        show(ax, np.asarray(arr[i, :, :, 0]), f"~{mid_bp/1e6:.2f} Mb")
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].axis("off")
    fig.legend(handles=CH0_LEGEND, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("C. difficile haplotypes after closely-related-strain filtering "
                 "(window_h=400, 120 samples/window)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def before_after(out_path, n=5, seed=1):
    nofix = np.load(f"{BASE}/results/real_w400_nofix_cmp/real_windows.npy", mmap_mode="r")
    fixed = np.load(f"{BASE}/results/real_w400_fixed_cmp/real_windows.npy", mmap_mode="r")
    sm = np.load(f"{BASE}/results/real_w400_nofix_cmp/sitemap_001.npy", mmap_mode="r")
    m = min(nofix.shape[0], fixed.shape[0])
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(m, n, replace=False))
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 6.2))
    for c, i in enumerate(idx):
        mid_bp = int((int(sm[i].min()) + int(sm[i].max())) / 2)
        show(axes[0, c], np.asarray(nofix[i, :, :, 0]), f"~{mid_bp/1e6:.2f} Mb")
        show(axes[1, c], np.asarray(fixed[i, :, :, 0]), "")
    axes[0, 0].set_ylabel("no strain filter", fontsize=10)
    axes[1, 0].set_ylabel("strain-filtered", fontsize=10)
    fig.legend(handles=CH0_LEGEND, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Effect of the closely-related-strain filter on matched "
                 "C. difficile windows", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def singles(out_dir, n=4):
    arr = np.load(f"{BASE}/results/real_w400_fixed_cmp/real_windows.npy", mmap_mode="r")
    sm = np.load(f"{BASE}/results/real_w400_fixed_cmp/sitemap_001.npy", mmap_mode="r")
    idx = np.linspace(0, arr.shape[0] - 1, n).astype(int)
    ch1_colors = ["#e1b12c", "#2f3640", "#44bd32"]
    ch1_cmap = ListedColormap(ch1_colors)
    ch1_legend = [
        Patch(facecolor="#e1b12c", edgecolor="#888", label="synonymous"),
        Patch(facecolor="#2f3640", edgecolor="#888", label="major / missing"),
        Patch(facecolor="#44bd32", edgecolor="#888", label="non-synonymous"),
    ]
    for j, i in enumerate(idx):
        img = np.asarray(arr[i])
        mid_bp = int((int(sm[i].min()) + int(sm[i].max())) / 2)
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
        ax[0].imshow(img[:, :, 0], aspect="auto", cmap=CH0_CMAP, vmin=-1.5, vmax=1.5,
                     interpolation="nearest")
        ax[0].set_title("allele state"); ax[0].set_xlabel("sites"); ax[0].set_ylabel("samples")
        ax[0].legend(handles=CH0_LEGEND, loc="upper right", fontsize=8, framealpha=0.9)
        ax[1].imshow(img[:, :, 1], aspect="auto", cmap=ch1_cmap, vmin=-1.5, vmax=1.5,
                     interpolation="nearest")
        ax[1].set_title("mutation type"); ax[1].set_xlabel("sites")
        ax[1].legend(handles=ch1_legend, loc="upper right", fontsize=8, framealpha=0.9)
        fig.suptitle(f"C. difficile strain-filtered window ~{mid_bp/1e6:.2f} Mb", fontsize=11)
        fig.tight_layout()
        p = os.path.join(out_dir, f"cdiff_filtered_window_{j+1:02d}.png")
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    filtered_grid(os.path.join(args.out_dir, "cdiff_filtered_grid.png"))
    before_after(os.path.join(args.out_dir, "cdiff_filter_before_after.png"))
    singles(args.out_dir)
