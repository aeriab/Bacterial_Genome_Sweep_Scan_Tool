"""
Export genome-scan prediction data (plot_inference/genome_scans/<species>/) into
compact static binary + JSON files for the client-side genome scan browser
(webapp/index.html).

Mirrors the exact per-window "Center" bp-position calculation and contig
ordering used by plot_inference/plot_inference_results_code/
multi_plot_showcase_no_annotation.py (longest-contig-first, Center = midpoint
of (min site, max site) in each window's sitemap row), so the web app's
bin_size=30 default view matches the existing genome_wide_scan_<species>.png
plots exactly. Pooling/annotation beyond that happens client-side in JS.

Output per species (all under webapp/data/):
  <species>.bin        flat binary: N x float32 position, N x float32 P_Neutral,
                        N x float32 P_Hard, N x float32 P_Soft, N x uint8 label
                        (0=Neutral, 1=Hard, 2=Soft), windows ordered exactly as
                        they are concatenated for plotting (contig by contig,
                        longest contig first, ascending position within contig).
  <species>_manifest.json   contig boundaries/metadata + n_windows + x_max

Also writes webapp/data/species_index.json listing all exported species.

Usage:
    python export_scan_data.py [--scans-dir DIR] [--out-dir DIR] [--species NAME ...]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

DEFAULT_SCANS_DIR = "/u/project/ngarud/baeria/Research/Summer26_Sims/Big_Sims_Official/plot_inference/genome_scans"
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

LABEL_CODE = {"Neutral": 0, "Hard_sweep": 1, "Soft_sweep": 2}


def extract_num(filepath):
    basename = os.path.basename(filepath)
    match = re.search(r"(\d+)\.txt$", basename)
    return int(match.group(1)) if match else -1


def ordered_contig_files(species_dir):
    """Return prediction file paths ordered exactly like the plotting script:
    longest contig first via contig_lengths.tsv, else numeric-suffix order."""
    all_files = glob.glob(os.path.join(species_dir, "predictions_*.txt"))
    valid_files = [f for f in all_files if extract_num(f) != -1]

    lengths_path = os.path.join(species_dir, "contig_lengths.tsv")
    if os.path.isfile(lengths_path):
        lengths_df = pd.read_csv(lengths_path, sep="\t", dtype={"contig_num": str})
        length_by_num = dict(zip(lengths_df["contig_num"], lengths_df["length_bp"]))

        def contig_length(filepath):
            num = f"{extract_num(filepath):03d}"
            return length_by_num.get(num, -1)

        valid_files.sort(key=contig_length, reverse=True)
    else:
        valid_files.sort(key=extract_num)
    return valid_files


def export_species(species_dir, out_dir):
    species = os.path.basename(species_dir.rstrip("/"))
    files = ordered_contig_files(species_dir)
    if not files:
        print(f"  SKIP {species}: no prediction files found")
        return None

    positions = []
    p_neutral = []
    p_hard = []
    p_soft = []
    labels = []
    contig_meta = []
    current_x_offset = 0.0

    for filepath in files:
        num = extract_num(filepath)
        df = pd.read_csv(filepath)
        expected_cols = {"Image_Index", "Predicted_Label", "P_Neutral", "P_Hard", "P_Soft"}
        if not expected_cols.issubset(df.columns):
            print(f"  WARNING: {os.path.basename(filepath)} missing expected columns, skipping")
            continue

        sitemap_path = os.path.join(species_dir, f"sitemap_{num:03d}.npy")
        if not os.path.isfile(sitemap_path):
            print(f"  ERROR: no sitemap for {os.path.basename(filepath)}, skipping contig")
            continue

        site_map = np.load(sitemap_path, allow_pickle=True)
        if len(site_map) != len(df):
            n = min(len(site_map), len(df))
            site_map = site_map[:n]
            df = df.iloc[:n]

        site_map = np.asarray(site_map)
        center = (site_map.min(axis=1) + site_map.max(axis=1)) / 2.0
        df = df.copy()
        df["Center"] = center
        df = df.sort_values("Center", kind="mergesort").reset_index(drop=True)

        unknown = set(df["Predicted_Label"].unique()) - set(LABEL_CODE)
        if unknown:
            print(f"  WARNING: unknown labels {unknown} in {os.path.basename(filepath)}")
        label_codes = df["Predicted_Label"].map(LABEL_CODE).fillna(0).astype(np.uint8)

        n_windows = len(df)
        contig_positions = df["Center"].to_numpy(dtype=np.float64) + current_x_offset

        start_index = sum(len(a) for a in positions)
        x_start = current_x_offset
        x_end = float(contig_positions.max()) if n_windows else current_x_offset

        positions.append(contig_positions.astype(np.float32))
        p_neutral.append(df["P_Neutral"].to_numpy(dtype=np.float32))
        p_hard.append(df["P_Hard"].to_numpy(dtype=np.float32))
        p_soft.append(df["P_Soft"].to_numpy(dtype=np.float32))
        labels.append(label_codes.to_numpy())

        contig_meta.append({
            "contig_num": f"{num:03d}",
            "n_windows": int(n_windows),
            "start_index": int(start_index),
            "end_index": int(start_index + n_windows),
            "x_start": x_start,
            "x_end": x_end,
        })

        current_x_offset = x_end + 1.0

    if not positions:
        print(f"  SKIP {species}: no usable contigs")
        return None

    pos_arr = np.concatenate(positions)
    pn_arr = np.concatenate(p_neutral)
    ph_arr = np.concatenate(p_hard)
    ps_arr = np.concatenate(p_soft)
    lb_arr = np.concatenate(labels)
    n_total = len(pos_arr)

    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, f"{species}.bin")
    with open(bin_path, "wb") as f:
        f.write(pos_arr.tobytes())
        f.write(pn_arr.tobytes())
        f.write(ph_arr.tobytes())
        f.write(ps_arr.tobytes())
        f.write(lb_arr.tobytes())

    manifest = {
        "species": species,
        "n_windows": int(n_total),
        "x_max": float(current_x_offset),
        "contigs": contig_meta,
        "layout": ["position_f32", "p_neutral_f32", "p_hard_f32", "p_soft_f32", "label_u8"],
    }
    manifest_path = os.path.join(out_dir, f"{species}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    bin_size_mb = os.path.getsize(bin_path) / 1e6
    print(f"  OK {species}: {n_total} windows, {len(contig_meta)} contigs, {bin_size_mb:.2f} MB")
    return {
        "species": species,
        "n_windows": int(n_total),
        "n_contigs": len(contig_meta),
        "x_max": float(current_x_offset),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans-dir", default=DEFAULT_SCANS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--species", nargs="*", default=None, help="Subset of species dir names to export")
    args = parser.parse_args()

    if args.species:
        species_dirs = [os.path.join(args.scans_dir, s) for s in args.species]
    else:
        species_dirs = sorted(
            os.path.join(args.scans_dir, d)
            for d in os.listdir(args.scans_dir)
            if os.path.isdir(os.path.join(args.scans_dir, d))
        )

    index = []
    for species_dir in species_dirs:
        if not os.path.isdir(species_dir):
            print(f"  MISSING dir: {species_dir}")
            continue
        print(f"Exporting {os.path.basename(species_dir)}...")
        result = export_species(species_dir, args.out_dir)
        if result:
            index.append(result)

    index.sort(key=lambda r: r["species"])
    index_path = os.path.join(args.out_dir, "species_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    total_mb = sum(os.path.getsize(os.path.join(args.out_dir, f"{r['species']}.bin")) for r in index) / 1e6
    print(f"\nExported {len(index)} species, {total_mb:.2f} MB total -> {args.out_dir}")


if __name__ == "__main__":
    main()
