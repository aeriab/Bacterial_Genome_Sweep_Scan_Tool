#!/usr/bin/env python3
"""
Generate a curated set of haplotype-window PNGs per species for the Bacterial
Genome Sweep Scan Tool, plus a per-species JSON index the web app loads into
the "Haplotype snapshots" panel below the scan.

Sweep-window selection matches what a user sees when they slide the pooling
control in the tool. For a ladder of pooling window sizes we reproduce the
web app's pooled per-window classification exactly (per-contig centered rolling
mean of P_Neutral/P_Hard/P_Soft, then the same tie-break rules as app.js
getPooled), find every stretch of many consecutive Hard or Soft calls, and
cluster those stretches across the whole pooling ladder. A region that stays a
run at many different pooling scales is a prominent sweep; we render the
haplotype window at the MIDDLE of each such region. Baseline neutral windows
are sampled evenly along the genome.

Per species we render up to:
  - N_BASELINE evenly spaced neutral windows,
  - up to MAX_PER_SWEEP of the most prominent Hard-sweep runs,
  - up to MAX_PER_SWEEP of the most prominent Soft-sweep runs.

Each rendered window is REBUILT from the species' haplotype CSV(s) with the
same per-window recipe the scan pipeline used (recode -> pick the
target_samples least-missing genomes -> 2-channel image -> cluster-sort rows
-> per-window major/minor flip -> middle-window sort). It is a faithful,
deterministic reconstruction of that genomic window rather than the exact
(randomly sampled) tensor the CNN scored, which is not reproducible.

The two re-run scan variants in the tool (Clostridioides difficile
strain-filtered, R. bromii re-sorted) have no species dir under
plot_inference/genome_scans; their scans live under Cdiff_Bromii_Clonality_Fixes
and their haplotype windows are drawn from the base species' CSVs.

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
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.abspath(__file__))
BSO = "/u/project/ngarud/baeria/Research/Summer26_Sims/Big_Sims_Official"
CBF = "/u/project/ngarud/baeria/Research/Summer26_Sims/Cdiff_Bromii_Clonality_Fixes"
DEFAULT_SCANS = os.path.join(BSO, "plot_inference/genome_scans")
DEFAULT_HAPS = os.path.join(BSO, "haplotypes")
DEFAULT_MODELS = os.path.join(BSO, "species_cnn_training")

# Re-run scan variants: name -> (scan dir, base species for CSVs/model, pretty).
VARIANTS = {
    "Clostridioides_difficile_strain_filtered": {
        "scan_dir": os.path.join(CBF, "results/dense_w400"),
        "base": "Clostridioides_difficile",
        "pretty": "Clostridioides difficile (strain-filtered)",
    },
    "Ruminococcus_bromii_62047_resorted": {
        "scan_dir": os.path.join(CBF, "results/scan_export/Ruminococcus_bromii_62047_resorted"),
        "base": "Ruminococcus_bromii_62047",
        "pretty": "Ruminococcus bromii (62047, re-sorted)",
    },
}

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

# --- sweep-run detection knobs ----------------------------------------------
# Pooling window sizes to sweep over (matches the tool's "Pooling N windows"
# control). A region that stays a run across many of these is prominent.
POOL_LADDER = (12, 16, 20, 25, 30, 40, 50, 65, 80, 100, 125, 160, 200, 250)
MIN_RUN_WINDOWS = 20         # a "run" is at least this many consecutive calls
MIN_RUN_BP_FRAC = 0.0006     # ...and spans at least this fraction of the genome
MIN_RUN_BP_FLOOR = 800       # ...but never require more than a hard floor either
MIN_SUPPORT = 3              # ...and must survive as a run at >= this many pooling scales
MIN_SEP_FRAC = 0.004         # keep same-class picks at least this far apart (of the genome)
MIN_SEP_FLOOR = 8000         # ...floored at this many bp
MAX_PER_SWEEP = 12           # cap of Hard runs, and of Soft runs, per species
N_BASELINE = 6               # evenly spaced neutral windows
BASELINE_POOL = 50           # pooling scale used to confirm a window reads neutral


# --------------------------------------------------------------------------
# scan-order reconstruction (must match export_scan_data.py exactly so the
# web app's x-axis and these picks line up)
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
    """Return (DataFrame, x_max). Columns: x (concatenated axis), code (0/1/2),
    p_neutral/p_hard/p_soft, contig_num (str), center_bp, win_lo_bp, win_hi_bp.
    Rows are in the tool's plotting order (longest contig first, ascending
    center within contig); the index is a clean 0..N-1 range."""
    rows = []
    x_offset = 0.0
    for pf in ordered_contig_files(species_dir):
        num = extract_num(pf)
        df = pd.read_csv(pf)
        sm = np.asarray(np.load(os.path.join(species_dir, f"sitemap_{num:03d}.npy"),
                                allow_pickle=True))
        n = min(len(sm), len(df))
        sm, df = sm[:n], df.iloc[:n].copy()
        df["center_bp"] = (sm.min(axis=1) + sm.max(axis=1)) / 2.0
        df["win_lo_bp"] = sm.min(axis=1)
        df["win_hi_bp"] = sm.max(axis=1)
        df = df.sort_values("center_bp", kind="mergesort").reset_index(drop=True)
        code = df["Predicted_Label"].map(LABEL_CODE).fillna(0).astype(int).to_numpy()
        xs = df["center_bp"].to_numpy() + x_offset
        pn = df["P_Neutral"].to_numpy(); ph = df["P_Hard"].to_numpy(); ps = df["P_Soft"].to_numpy()
        for k in range(len(df)):
            rows.append((xs[k], int(code[k]), float(pn[k]), float(ph[k]), float(ps[k]),
                         f"{num:03d}", float(df["center_bp"].iloc[k]),
                         float(df["win_lo_bp"].iloc[k]), float(df["win_hi_bp"].iloc[k])))
        x_offset = float(xs.max()) + 1.0 if len(df) else x_offset
    table = pd.DataFrame(rows, columns=["x", "code", "p_neutral", "p_hard", "p_soft",
                                        "contig_num", "center_bp", "win_lo_bp", "win_hi_bp"])
    return table, x_offset


def contig_bounds(table):
    """[(start_row, end_row_exclusive), ...] for each contig block, in order."""
    out = []
    for _, g in table.groupby("contig_num", sort=False):
        idx = g.index.to_numpy()
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


# --------------------------------------------------------------------------
# pooled classification -- reproduces app.js getPooled() / rollingMeanInto()
# --------------------------------------------------------------------------
def rolling_mean_app(v, w):
    """pandas .rolling(window=w, center=True, min_periods=1).mean() with the
    same even-window centering app.js uses (extra element before the center)."""
    n = len(v)
    if n == 0:
        return np.asarray(v, dtype=np.float64)
    before = w // 2
    after = w - 1 - before
    pre = np.concatenate(([0.0], np.cumsum(v, dtype=np.float64)))
    i = np.arange(n)
    lo = np.clip(i - before, 0, n - 1)
    hi = np.clip(i + after, 0, n - 1)
    return (pre[hi + 1] - pre[lo]) / (hi - lo + 1)


def pooled_colorcode(pn, ph, ps, bounds, w):
    """0 neutral / 1 hard / 2 soft per window, exactly as app.js colors the dots."""
    n = len(pn)
    a = np.empty(n); b = np.empty(n); c = np.empty(n)
    for s, e in bounds:
        a[s:e] = rolling_mean_app(pn[s:e], w)
        b[s:e] = rolling_mean_app(ph[s:e], w)
        c[s:e] = rolling_mean_app(ps[s:e], w)
    neutral = (a > 0.01) | ((a >= c) & (a >= b))
    soft = (~neutral) & (c >= b)
    cc = np.zeros(n, np.int8)
    cc[soft] = 2
    cc[(~neutral) & (~soft)] = 1
    return cc


def find_runs(cc, target, bounds, min_win):
    out = []
    for s, e in bounds:
        i = s
        while i < e:
            if cc[i] == target:
                j = i
                while j + 1 < e and cc[j + 1] == target:
                    j += 1
                if j - i + 1 >= min_win:
                    out.append((i, j))
                i = j + 1
            else:
                i += 1
    return out


def pick_sweep_sites(table, bounds, x_max):
    """Return {1: [cluster, ...], 2: [...]} plus the list of taken genomic
    intervals. Each cluster: dict(cls, i0, i1, mid, bp_lo, bp_hi, support,
    max_win) where `mid` is the row index at the middle of the run region."""
    pn = table["p_neutral"].to_numpy(np.float64)
    ph = table["p_hard"].to_numpy(np.float64)
    ps = table["p_soft"].to_numpy(np.float64)
    win_lo = table["win_lo_bp"].to_numpy()
    win_hi = table["win_hi_bp"].to_numpy()
    contig = table["contig_num"].to_numpy()
    xs = table["x"].to_numpy()

    min_bp = max(MIN_RUN_BP_FLOOR, MIN_RUN_BP_FRAC * x_max)
    min_sep = max(MIN_SEP_FLOOR, MIN_SEP_FRAC * x_max)

    raw = {1: [], 2: []}
    for w in POOL_LADDER:
        cc = pooled_colorcode(pn, ph, ps, bounds, w)
        for tgt in (1, 2):
            for i0, i1 in find_runs(cc, tgt, bounds, MIN_RUN_WINDOWS):
                raw[tgt].append((i0, i1, w))

    clusters = []
    for cls in (1, 2):
        cur = None
        for i0, i1, w in sorted(raw[cls]):
            if cur and i0 <= cur["i1"]:
                cur["i0"] = min(cur["i0"], i0)
                cur["i1"] = max(cur["i1"], i1)
                cur["ws"].add(w)
                cur["max_win"] = max(cur["max_win"], i1 - i0 + 1)
            else:
                cur = {"cls": cls, "i0": i0, "i1": i1, "ws": {w}, "max_win": i1 - i0 + 1}
                clusters.append(cur)

    cand = []
    for cl in clusters:
        i0, i1 = cl["i0"], cl["i1"]
        lo = float(min(win_lo[i0], win_lo[i1]))
        hi = float(max(win_hi[i0], win_hi[i1]))
        cl["support"] = len(cl["ws"])
        if hi - lo < min_bp or cl["support"] < MIN_SUPPORT:
            continue
        cl["bp_lo"], cl["bp_hi"] = lo, hi
        cl["mid"] = (i0 + i1) // 2
        cand.append(cl)

    # prominent = a run that holds at many pooling scales, then a long run
    cand.sort(key=lambda c: (c["support"], c["max_win"]), reverse=True)

    chosen = {1: [], 2: []}
    taken = []
    picked_x = {1: [], 2: []}
    for c in cand:
        cn = contig[c["mid"]]
        if len(chosen[c["cls"]]) >= MAX_PER_SWEEP:
            continue
        if any(cn == tc and not (c["bp_hi"] < tlo or c["bp_lo"] > thi)
               for tc, tlo, thi in taken):
            continue
        cx = xs[c["mid"]]
        if any(abs(cx - px) < min_sep for px in picked_x[c["cls"]]):
            continue
        chosen[c["cls"]].append(c)
        taken.append((cn, c["bp_lo"], c["bp_hi"]))
        picked_x[c["cls"]].append(cx)
    return chosen, taken


def _overlaps(cn, lo, hi, taken):
    return any(cn == tc and not (hi < tlo or lo > thi) for tc, tlo, thi in taken)


def pick_baseline_sites(table, bounds, x_max, taken):
    xs = table["x"].to_numpy()
    contig = table["contig_num"].to_numpy()
    win_lo = table["win_lo_bp"].to_numpy()
    win_hi = table["win_hi_bp"].to_numpy()
    raw_code = table["code"].to_numpy()
    cc = pooled_colorcode(table["p_neutral"].to_numpy(np.float64),
                          table["p_hard"].to_numpy(np.float64),
                          table["p_soft"].to_numpy(np.float64), bounds, BASELINE_POOL)
    picks = []
    for k in range(1, N_BASELINE + 1):
        target_x = k / (N_BASELINE + 1) * x_max
        ri = None
        for radius in (0.02, 0.05, 0.12):
            near = np.where(np.abs(xs - target_x) <= radius * x_max)[0]
            ok = [int(i) for i in near if cc[i] == 0 and raw_code[i] == 0
                  and not _overlaps(contig[i], win_lo[i], win_hi[i], taken)]
            if ok:
                ri = min(ok, key=lambda i: abs(xs[i] - target_x))
                break
        if ri is None:
            near = np.where(np.abs(xs - target_x) <= 0.12 * x_max)[0]
            pool = near[cc[near] == 0] if len(near) and (cc[near] == 0).any() else near
            if not len(pool):
                pool = np.array([int(np.abs(xs - target_x).argmin())])
            ri = int(pool[np.abs(xs[pool] - target_x).argmin()])
        cn = contig[ri]; lo = float(win_lo[ri]); hi = float(win_hi[ri])
        picks.append((f"neutral_{k}", ri, lo, hi))
        taken.append((cn, lo, hi))
    return picks


# --------------------------------------------------------------------------
# per-window image reconstruction from the haplotype CSV
# --------------------------------------------------------------------------
_CSV_CACHE = {}


def load_haplotype_csv(csv_path):
    hit = _CSV_CACHE.get(csv_path)
    if hit is None:
        df = pd.read_csv(csv_path)
        fsc = df.columns.get_loc("site_type") + 1
        hit = {
            "site_pos": df["site_pos"].to_numpy(np.int64),
            "raw": df.iloc[:, fsc:].fillna(-1).to_numpy(np.int8),               # sites x samples
            "st": df["site_type"].map({"syn": 0, "nonsyn": 1}).fillna(0).to_numpy(np.int8),
        }
        _CSV_CACHE[csv_path] = hit
    return hit


def load_csv_sitepos(csv_path):
    return pd.read_csv(csv_path, usecols=["site_pos"])["site_pos"].to_numpy(np.int64)


def match_contig_csvs(haps_species, contig_nums, species_dir, haps_dir):
    """contig_num (str) -> csv path, by checking that the contig's sitemap bp
    values are a subset of the CSV's site_pos."""
    csvs = sorted(glob.glob(os.path.join(haps_dir, haps_species, "*_haplotypes.csv")))
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
    csv = load_haplotype_csv(csv_path)
    site_pos, raw_all, st_all = csv["site_pos"], csv["raw"], csv["st"]

    lo = int(np.searchsorted(site_pos, win_lo_bp, "left"))
    lo = min(lo, len(site_pos) - window_h)
    lo = max(lo, 0)
    hi = lo + window_h
    win_raw = raw_all[lo:hi, :]
    win_st = st_all[lo:hi]

    missing_rate = (win_raw == -1).mean(axis=0)
    order = np.argsort(missing_rate, kind="stable")
    good = order[missing_rate[order] <= (1 - complete_thresh)]
    if len(good) < target_samples:
        good = order[:target_samples]
    chosen = np.sort(good[:target_samples])

    block = win_raw[:, chosen].T
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
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
    ax[0].imshow(img[:, :, 0], aspect="auto", cmap=CH0_CMAP, vmin=-1.5, vmax=1.5,
                 interpolation="nearest")
    ax[0].set_title("allele state", fontsize=10)
    ax[0].set_xlabel("polymorphic sites"); ax[0].set_ylabel("sampled genomes")
    ax[0].legend(handles=CH0_LEGEND, loc="upper right", fontsize=7, framealpha=0.9)
    ax[1].imshow(img[:, :, 1], aspect="auto", cmap=CH1_CMAP, vmin=-1.5, vmax=1.5,
                 interpolation="nearest")
    ax[1].set_title("mutation type", fontsize=10)
    ax[1].set_xlabel("polymorphic sites")
    ax[1].legend(handles=CH1_LEGEND, loc="upper right", fontsize=7, framealpha=0.9)
    for a in ax:
        a.set_yticks([])
    fig.suptitle(title, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
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


def variant_window_h(scan_dir):
    for p in sorted(glob.glob(os.path.join(scan_dir, "sitemap_*.npy"))):
        return int(np.asarray(np.load(p, allow_pickle=True)).shape[1])
    return None


def pretty_name(key):
    m = re.match(r"^(.*)_(\d+)$", key)
    if not m:
        return key.replace("_", " ")
    return f"{m.group(1).replace('_', ' ')} ({m.group(2)})"


def process_species(species, scans_dir, haps_dir, models_dir, out_img_root, out_data_dir):
    v = VARIANTS.get(species)
    scan_dir = v["scan_dir"] if v else os.path.join(scans_dir, species)
    haps_species = v["base"] if v else species
    model_species = v["base"] if v else species
    pretty = v["pretty"] if v else pretty_name(species)

    if not os.path.isdir(scan_dir):
        print(f"  {species}: no scan dir ({scan_dir}), skip")
        return None

    shape = model_shape(models_dir, model_species)
    if shape is None:
        print(f"  {species}: no model shape, skip")
        return None
    target_samples, window_h = shape
    if v:
        window_h = variant_window_h(scan_dir) or window_h

    table, x_max = build_scan_table(scan_dir)
    if table.empty:
        print(f"  {species}: empty scan table, skip")
        return None
    bounds = contig_bounds(table)

    chosen, taken = pick_sweep_sites(table, bounds, x_max)
    baseline = pick_baseline_sites(table, bounds, x_max, taken)

    n_pool = len(POOL_LADDER)
    picks = []   # (key, type, label, row_index, call_here)
    for key, ri, _lo, _hi in baseline:
        cbp = float(table["center_bp"].iloc[ri])
        picks.append((key, "baseline", f"Neutral · {cbp / 1e6:.2f} Mb", ri, "Neutral"))
    for cls, tname, tkey in [(1, "Hard", "hard"), (2, "Soft", "soft")]:
        for n, c in enumerate(sorted(chosen[cls], key=lambda d: d["bp_lo"]), 1):
            span_kb = (c["bp_hi"] - c["bp_lo"]) / 1e3
            label = (f"{tname}-sweep run — consecutive at {c['support']}/{n_pool} "
                     f"pooling scales, ~{span_kb:.0f} kb wide")
            picks.append((f"{tkey}_{n}", tkey, label, c["mid"], CODE_NAME[cls]))

    if not picks:
        print(f"  {species}: nothing to render, skip")
        return None

    contig_nums = sorted({table["contig_num"].iloc[ri] for _, _, _, ri, _ in picks})
    csv_of = match_contig_csvs(haps_species, contig_nums, scan_dir, haps_dir)

    out_img_dir = os.path.join(out_img_root, species)
    if os.path.isdir(out_img_dir):
        shutil.rmtree(out_img_dir)
    os.makedirs(out_img_dir, exist_ok=True)

    multi = len(contig_bounds(table)) > 1
    sites = []
    for key, typ, label, ri, call_here in picks:
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
        loc = (f"contig {cn} · " if multi else "") + f"{lo_bp / 1e6:.3f}–{hi_bp / 1e6:.3f} Mb"
        png_rel = f"haplotype_images/sites/{species}/{key}.png"
        render_png(img, os.path.join(REPO, png_rel), f"{pretty} — {label} — {loc}")
        sites.append({
            "key": key, "type": typ, "label": label,
            "x": round(float(r["x"]), 2),
            "contig": cn, "bp_lo": lo_bp, "bp_hi": hi_bp,
            "call_here": call_here,
            "image": png_rel,
        })
        print(f"  {species}/{key}: {loc}  [{call_here}]")

    if not sites:
        shutil.rmtree(out_img_dir, ignore_errors=True)
        return None

    payload = {"species": species, "pretty": pretty, "window_h": window_h,
               "target_samples": target_samples, "x_max": round(float(x_max), 2),
               "sites": sites}
    with open(os.path.join(out_data_dir, f"{species}_haplotype_sites.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return species


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

    base_species = sorted(
        d for d in os.listdir(args.scans_dir)
        if os.path.isdir(os.path.join(args.scans_dir, d)))
    all_species = base_species + [s for s in VARIANTS if s not in base_species]
    species_list = args.species or all_species

    done = []
    for sp in species_list:
        print(f"{sp} ...")
        _CSV_CACHE.clear()
        try:
            r = process_species(sp, args.scans_dir, args.haplotypes_dir, args.models_dir,
                                 out_img_root, out_data_dir)
            if r:
                done.append(r)
        except Exception as e:
            import traceback
            print(f"  {sp}: FAILED {e}")
            traceback.print_exc()

    # drop stale per-species image dirs / json for species we no longer emit
    if not args.species:
        keep = set(done)
        for d in glob.glob(os.path.join(out_img_root, "*")):
            if os.path.isdir(d) and os.path.basename(d) not in keep:
                shutil.rmtree(d, ignore_errors=True)
        for j in glob.glob(os.path.join(out_data_dir, "*_haplotype_sites.json")):
            if os.path.basename(j)[:-len("_haplotype_sites.json")] not in keep:
                os.remove(j)

    with open(os.path.join(out_data_dir, "haplotype_sites_index.json"), "w") as f:
        json.dump(sorted(done), f)
    print(f"\n{len(done)} species with haplotype sites -> {out_data_dir}/haplotype_sites_index.json")


if __name__ == "__main__":
    main()
