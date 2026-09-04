#!/usr/bin/env python3
"""
Generate a small, curated set of haplotype-window PNGs per species for the
Bacterial Genome Sweep Scan Tool, plus a per-species JSON index the web app
loads into the "Haplotype snapshots" panel below the scan.

Per species, up to 9 windows are rendered:
  - 3 "baseline" windows at 1/4, 1/2, 3/4 of the genome (the tool's
    concatenated x-axis), to show what ordinary / mostly-neutral structure
    looks like for that species.
  - up to 3 windows at the longest consecutive runs of Hard-sweep calls.
  - up to 3 windows at the longest consecutive runs of Soft-sweep calls.

Caching whole-genome haplotype images (one per sliding window) would be
hundreds of thousands of PNGs; this keeps it to <= 9 per species.

Each rendered window is REBUILT from the species' haplotype CSV(s) with the
same per-window recipe the scan pipeline used (recode -> pick the
target_samples least-missing genomes -> 2-channel image -> cluster-sort rows
-> per-window major/minor flip -> middle-window sort). It is a faithful,
deterministic reconstruction of that genomic window rather than the exact
(randomly sampled) tensor the CNN scored, which is not reproducible.

Outputs:
  haplotype_images/sites/<species>/<key>.png
  data/<species>_haplotype_sites.json
  data/haplotype_sites_index.json   (list of species that have a sites file)
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.abspath(__file__))
BSO = "/u/project/ngarud/baeria/Research/Summer26_Sims/Big_Sims_Official"
DEFAULT_SCANS = os.path.join(BSO, "plot_inference/genome_scans")
DEFAULT_HAPS = os.path.join(BSO, "haplotypes")
DEFAULT_MODELS = os.path.join(BSO, "species_cnn_training")

LABEL_CODE = {"Neutral": 0, "Hard_sweep": 1, "Soft_sweep": 2}
CODE_NAME = {0: "Neutral", 1: "Hard sweep", 2: "Soft sweep"}

CH0_COLORS = ["#2f3640", "#c4ccd2", "#3498db"]   # major / missing / minor
CH1_COLORS = ["#e1b12c", "#2f3640", "#44bd32"]   # synonymous / (major|missing) / non-synonymous
CH0_CMAP = ListedColormap(CH0_COLORS)
CH1_CMAP = ListedColormap(CH1_COLORS)
CH0_LEGEND = [Patch(facecolor=c, edgecolor="#888", label=l) for c, l in
              zip(CH0_COLORS, ["major allele", "missing", "minor allele"])]
CH1_LEGEND = [Patch(facecolor=c, edgecolor="#888", label=l) for c, l in
              zip(CH1_COLORS, ["synonymous", "major / missing", "non-synonymous"])]

MIN_RUN = 4          # ignore hard/soft runs shorter than this many windows
TOP_RUNS_PER_CLASS = 3


# --------------------------------------------------------------------------
# scan-order reconstruction (must match export_scan_data.py exactly so the
# web app's x-axis and these marker positions line up)
# --------------------------------------------------------------------------
def extract_num(path):
    m = re.search(r"(\d+)\.txt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def ordered_contig_files(species_dir):
    files = [f for f in glob.glob(os.path.join(species_dir, "predictions_*.txt"))
             if extract_num(f) != -1]
    lengths_path = os.path.join(species_dir, "contig_lengths.tsv")
    if os.path.isfile(lengths_path):
        ldf = pd.read_csv(lengths_path, sep="\t", dtype={"contig_num": str})
        by_num = dict(zip(ldf["contig_num"], ldf["length_bp"]))
        files.sort(key=lambda f: by_num.get(f"{extract_num(f):03d}", -1), reverse=True)
    else:
        files.sort(key=extract_num)
    return files


def build_scan_table(species_dir):
    """Return a DataFrame of every scan window in the tool's plotting order:
    columns x (concatenated axis), code (0/1/2), contig_num (str), center_bp,
    win_lo_bp, win_hi_bp."""
    rows = []
    x_offset = 0.0
    for pf in ordered_contig_files(species_dir):
        num = extract_num(pf)
        df = pd.read_csv(pf)
        sm = np.load(os.path.join(species_dir, f"sitemap_{num:03d}.npy"), allow_pickle=True)
        sm = np.asarray(sm)
        n = min(len(sm), len(df))
        sm, df = sm[:n], df.iloc[:n].copy()
        center = (sm.min(axis=1) + sm.max(axis=1)) / 2.0
        df["center_bp"] = center
        df["win_lo_bp"] = sm.min(axis=1)
        df["win_hi_bp"] = sm.max(axis=1)
        df = df.sort_values("center_bp", kind="mergesort").reset_index(drop=True)
        code = df["Predicted_Label"].map(LABEL_CODE).fillna(0).astype(int).to_numpy()
        xs = df["center_bp"].to_numpy() + x_offset
        for k in range(len(df)):
            rows.append((xs[k], int(code[k]), f"{num:03d}",
                         float(df["center_bp"].iloc[k]),
                         float(df["win_lo_bp"].iloc[k]), float(df["win_hi_bp"].iloc[k])))
        x_offset = float(xs.max()) + 1.0 if len(df) else x_offset
    return pd.DataFrame(rows, columns=["x", "code", "contig_num", "center_bp",
                                       "win_lo_bp", "win_hi_bp"]), x_offset


def longest_runs(codes, target, k=TOP_RUNS_PER_CLASS, min_len=MIN_RUN):
    runs = []
    i = 0
    n = len(codes)
    while i < n:
        if codes[i] == target:
            j = i
            while j + 1 < n and codes[j + 1] == target:
                j += 1
            if (j - i + 1) >= min_len:
                runs.append((i, j, j - i + 1))
            i = j + 1
        else:
            i += 1
    runs.sort(key=lambda r: r[2], reverse=True)
    return runs[:k]


# --------------------------------------------------------------------------
# per-window image reconstruction from the haplotype CSV
# --------------------------------------------------------------------------
def load_csv_sitepos(csv_path):
    return pd.read_csv(csv_path, usecols=["site_pos"])["site_pos"].to_numpy(np.int64)


def match_contig_csvs(species, contig_nums, species_dir, haps_dir):
    """contig_num (str) -> csv path, by checking that the contig's sitemap bp
    values are a subset of the CSV's site_pos."""
    csvs = sorted(glob.glob(os.path.join(haps_dir, species, "*_haplotypes.csv")))
    pos_by_csv = {}
    out = {}
    for num in contig_nums:
        sm = np.asarray(np.load(os.path.join(species_dir, f"sitemap_{int(num):03d}.npy"),
                                allow_pickle=True))
        probe = set(np.unique(sm[[0, len(sm) // 2, -1]]).tolist())
        for c in csvs:
            if c not in pos_by_csv:
                pos_by_csv[c] = set(load_csv_sitepos(c).tolist())
            if probe <= pos_by_csv[c]:
                out[num] = c
                break
    return out


def build_window_image(csv_path, win_lo_bp, win_hi_bp, window_h, target_samples,
                        complete_thresh=0.7):
    df = pd.read_csv(csv_path)
    fsc = df.columns.get_loc("site_type") + 1
    site_pos = df["site_pos"].to_numpy(np.int64)
    raw_all = df.iloc[:, fsc:].fillna(-1).to_numpy(np.int8)          # sites x samples
    st_all = df["site_type"].map({"syn": 0, "nonsyn": 1}).fillna(0).to_numpy(np.int8)

    lo = int(np.searchsorted(site_pos, win_lo_bp, "left"))
    lo = min(lo, len(site_pos) - window_h)
    lo = max(lo, 0)
    hi = lo + window_h
    win_raw = raw_all[lo:hi, :]                                      # window_h x n_samples
    win_st = st_all[lo:hi]

    missing_rate = (win_raw == -1).mean(axis=0)
    order = np.argsort(missing_rate, kind="stable")                 # least-missing first
    good = order[missing_rate[order] <= (1 - complete_thresh)]
    if len(good) < target_samples:
        good = order[:target_samples]                               # fall back
    chosen = np.sort(good[:target_samples])

    block = win_raw[:, chosen].T                                    # samples x sites
    ch0 = np.zeros_like(block, dtype=np.int8)
    ch0[block == 0] = -1
    ch0[block == 1] = 1
    tiled = np.tile(win_st, (block.shape[0], 1))
    ch1 = np.zeros_like(block, dtype=np.int8)
    ch1[(block == 1) & (tiled == 0)] = -1
    ch1[(block == 1) & (tiled == 1)] = 1
    img = np.stack([ch0, ch1], axis=-1)

    _cluster_sort_rows(img)
    _remajor(img)
    _middle_sort_rows(img)
    lo_bp = int(site_pos[lo]); hi_bp = int(site_pos[hi - 1])
    return img, lo_bp, hi_bp


def _cluster_sort_rows(img, min_maf=0.2):
    geno = img[:, :, 0]
    is_missing = geno == 0
    n_called = (~is_missing).sum(axis=0)
    n_minor = (geno == 1).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        maf = np.where(n_called > 0, n_minor / np.maximum(n_called, 1), 0.0)
    mask = maf >= min_maf
    key = geno[:, mask] if mask.sum() else geno
    n = key.shape[0]
    assigned = -np.ones(n, dtype=int)
    clusters = []
    for i in range(n):
        if assigned[i] != -1:
            continue
        members = [i]
        assigned[i] = len(clusters)
        for j in range(i + 1, n):
            if assigned[j] != -1:
                continue
            valid = (key[i] != 0) & (key[j] != 0)
            if valid.any() and not np.any((key[i] != key[j]) & valid):
                assigned[j] = len(clusters)
                members.append(j)
        clusters.append(members)
    clusters.sort(key=len, reverse=True)
    order = [idx for c in clusters for idx in c]
    img[:, :, :] = img[order, :, :]


def _remajor(img):
    geno = img[:, :, 0]
    color = img[:, :, 1]
    n_samples = geno.shape[0]
    for col in range(geno.shape[1]):
        if np.sum(geno[:, col] == 1) > n_samples / 2:
            st = 0
            if np.any(color[:, col] == -1):
                st = -1
            elif np.any(color[:, col] == 1):
                st = 1
            new = np.where(geno[:, col] == 1, -1, 1)
            geno[:, col] = new
            nc = np.zeros_like(color[:, col])
            nc[new == 1] = st
            color[:, col] = nc


def _middle_sort_rows(img):
    n_snps = img.shape[1]
    w = min(n_snps, max(100, int(n_snps * 0.5)))
    s = (n_snps - w) // 2
    sl = img[:, s:s + w, 0]
    uniq, counts = np.unique(sl, axis=0, return_counts=True)
    order = []
    for gi in np.argsort(-counts):
        order.extend(np.where((sl == uniq[gi]).all(axis=1))[0].tolist())
    img[:, :, :] = img[order, :, :]


def render_png(img, out_path, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].imshow(img[:, :, 0], aspect="auto", cmap=CH0_CMAP, vmin=-1.5, vmax=1.5,
                 interpolation="nearest")
    ax[0].set_title("allele state", fontsize=10)
    ax[0].set_xlabel("polymorphic sites"); ax[0].set_ylabel("sampled genomes")
    ax[0].legend(handles=CH0_LEGEND, loc="upper right", fontsize=7.5, framealpha=0.9)
    ax[1].imshow(img[:, :, 1], aspect="auto", cmap=CH1_CMAP, vmin=-1.5, vmax=1.5,
                 interpolation="nearest")
    ax[1].set_title("mutation type", fontsize=10)
    ax[1].set_xlabel("polymorphic sites")
    ax[1].legend(handles=CH1_LEGEND, loc="upper right", fontsize=7.5, framealpha=0.9)
    for a in ax:
        a.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def model_shape(models_dir, species):
    j = os.path.join(models_dir, species, "direct_fixed_training",
                     f"{species}_direct_fixed_model.json")
    if not os.path.isfile(j):
        cand = glob.glob(os.path.join(models_dir, species, "*_model.json"))
        j = cand[0] if cand else None
    if not j:
        return None
    m = json.load(open(j))
    for layer in m["config"]["layers"]:
        sh = layer["config"].get("batch_shape") or layer["config"].get("batch_input_shape")
        if sh and len(sh) == 4:
            return int(sh[1]), int(sh[2])   # target_samples, window_h
    return None


def process_species(species, scans_dir, haps_dir, models_dir, out_img_root, out_data_dir,
                    pretty):
    species_dir = os.path.join(scans_dir, species)
    shape = model_shape(models_dir, species)
    if shape is None:
        print(f"  {species}: no model shape, skip")
        return None
    target_samples, window_h = shape

    table, x_max = build_scan_table(species_dir)
    if table.empty:
        print(f"  {species}: empty scan table, skip")
        return None
    codes = table["code"].to_numpy()

    xs = table["x"].to_numpy()
    picks = []   # (key, type, label, row_index)
    for q, name in [(0.25, "¼ genome"), (0.5, "½ genome"), (0.75, "¾ genome")]:
        target_x = q * x_max
        # nearest window to the q mark, but prefer a Neutral one within +/-3%
        near = np.where(np.abs(xs - target_x) <= 0.03 * x_max)[0]
        neutral_near = near[codes[near] == 0]
        pool = neutral_near if len(neutral_near) else np.array([np.abs(xs - target_x).argmin()])
        ri = int(pool[np.abs(xs[pool] - target_x).argmin()])
        picks.append((f"baseline_{int(q*100)}", "baseline", name, ri))
    for tgt, tname, tkey in [(1, "Hard-sweep run", "hard"), (2, "Soft-sweep run", "soft")]:
        rank = 0
        for (i0, i1, ln) in longest_runs(codes, tgt, k=TOP_RUNS_PER_CLASS * 3):
            ri = (i0 + i1) // 2
            r = table.iloc[ri]
            # skip if it overlaps an already-picked window on the same contig
            dup = any(p_r["contig_num"] == r["contig_num"]
                      and not (r["win_hi_bp"] < p_r["win_lo_bp"] or r["win_lo_bp"] > p_r["win_hi_bp"])
                      for _, _, _, pri in picks for p_r in [table.iloc[pri]])
            if dup:
                continue
            rank += 1
            picks.append((f"{tkey}_run{rank}", tkey, f"{tname} ({ln} windows)", ri))
            if rank >= TOP_RUNS_PER_CLASS:
                break

    contig_nums = sorted({table["contig_num"].iloc[ri] for _, _, _, ri in picks})
    csv_of = match_contig_csvs(species, contig_nums, species_dir, haps_dir)

    out_img_dir = os.path.join(out_img_root, species)
    os.makedirs(out_img_dir, exist_ok=True)
    sites = []
    for key, typ, label, ri in picks:
        r = table.iloc[ri]
        cn = r["contig_num"]
        csv_path = csv_of.get(cn)
        if not csv_path:
            print(f"  {species}/{key}: no CSV for contig {cn}, skip")
            continue
        try:
            img, lo_bp, hi_bp = build_window_image(
                csv_path, r["win_lo_bp"], r["win_hi_bp"], window_h, target_samples)
        except Exception as e:
            print(f"  {species}/{key}: build failed ({e}), skip")
            continue
        multi = len(ordered_contig_files(species_dir)) > 1
        loc = (f"contig {cn} · " if multi else "") + f"{lo_bp/1e6:.3f}–{hi_bp/1e6:.3f} Mb"
        title = f"{pretty} — {label} — {loc}"
        png_rel = f"haplotype_images/sites/{species}/{key}.png"
        render_png(img, os.path.join(REPO, png_rel), title)
        sites.append({
            "key": key, "type": typ, "label": label,
            "x": round(float(r["x"]), 2),
            "contig": cn, "bp_lo": lo_bp, "bp_hi": hi_bp,
            "call_here": CODE_NAME[int(r["code"])],
            "image": png_rel,
        })
        print(f"  {species}/{key}: {loc}  [{CODE_NAME[int(r['code'])]}]")

    if not sites:
        return None
    payload = {"species": species, "pretty": pretty, "window_h": window_h,
               "target_samples": target_samples, "x_max": round(float(x_max), 2),
               "sites": sites}
    with open(os.path.join(out_data_dir, f"{species}_haplotype_sites.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return species


def pretty_name(key):
    m = re.match(r"^(.*)_(\d+)$", key)
    if not m:
        return key.replace("_", " ")
    return f"{m.group(1).replace('_', ' ')} ({m.group(2)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans-dir", default=DEFAULT_SCANS)
    ap.add_argument("--haplotypes-dir", default=DEFAULT_HAPS)
    ap.add_argument("--models-dir", default=DEFAULT_MODELS)
    ap.add_argument("--species", nargs="*", default=None)
    args = ap.parse_args()

    out_img_root = os.path.join(REPO, "haplotype_images", "sites")
    out_data_dir = os.path.join(REPO, "data")
    os.makedirs(out_img_root, exist_ok=True)

    species_list = args.species or sorted(
        d for d in os.listdir(args.scans_dir)
        if os.path.isdir(os.path.join(args.scans_dir, d)))

    done = []
    for sp in species_list:
        print(f"{sp} ...")
        try:
            r = process_species(sp, args.scans_dir, args.haplotypes_dir, args.models_dir,
                                 out_img_root, out_data_dir, pretty_name(sp))
            if r:
                done.append(r)
        except Exception as e:
            import traceback
            print(f"  {sp}: FAILED {e}")
            traceback.print_exc()

    with open(os.path.join(out_data_dir, "haplotype_sites_index.json"), "w") as f:
        json.dump(sorted(done), f)
    print(f"\n{len(done)} species with haplotype sites -> {out_data_dir}/haplotype_sites_index.json")


if __name__ == "__main__":
    main()
