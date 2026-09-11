#!/usr/bin/env python3
"""Build data/peaks.json from the curated H12 peak table.

Source: data/H12_Peaks/cropped_s4.tsv  -- one row per curated peak, with
  species, peak (number), is_hard, left_coord, right_coord, accession, and
  contig_num_LABEL_in_webtool (the contig's number as shown on the web tool's
  x-axis; verified against each species manifest here).

left_coord / right_coord are contig-relative bp (0 = contig start), which is
exactly what the web tool's band renderer expects (it adds contig.x_start
itself -- see geneAnnotationPixels in app.js).

Output: data/peaks.json
  { "<species>": [ { peak, contigNum, startBp, endBp, kind, text, color }, ... ] }

Every peak is validated against data/<species>_manifest.json: the contig must
exist and the region must fit inside it. Anything that fails is skipped and
listed in the summary printed to stderr.
"""

import csv
import json
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
SRC_TSV = os.path.join(DATA_DIR, "H12_Peaks", "cropped_s4.tsv")
OUT_JSON = os.path.join(DATA_DIR, "peaks.json")

# One distinct band color for every curated peak. Deliberately NOT the red/blue
# of the CNN hard/soft dots -- an amber that stays legible over both -- since the
# hard/soft split is already carried in each peak's label and tooltip.
PEAK_COLOR = "#e8710a"

_manifest_cache = {}


def load_manifest(species):
    """Return {contig_num: length_bp} for a species, or None if no manifest."""
    if species in _manifest_cache:
        return _manifest_cache[species]
    path = os.path.join(DATA_DIR, f"{species}_manifest.json")
    if not os.path.isfile(path):
        _manifest_cache[species] = None
        return None
    with open(path) as f:
        manifest = json.load(f)
    lengths = {
        c["contig_num"]: round(c["x_end"] - c["x_start"])
        for c in manifest["contigs"]
    }
    _manifest_cache[species] = lengths
    return lengths


def parse_bool(s):
    return str(s).strip().lower() in ("true", "1", "yes", "t")


def main():
    if not os.path.isfile(SRC_TSV):
        sys.exit(f"ERROR: source table not found: {SRC_TSV}")

    peaks_by_species = defaultdict(list)
    problems = []
    n_rows = 0

    with open(SRC_TSV, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            species = row["species"].strip()
            if not species:
                continue
            n_rows += 1

            peak_no = int(float(row["peak"]))
            label_num = int(float(row["contig_num_LABEL_in_webtool"]))
            contig_num = f"{label_num:03d}"
            start_bp = int(float(row["left_coord"]))
            end_bp = int(float(row["right_coord"]))
            is_hard = parse_bool(row["is_hard"])
            kind = "hard" if is_hard else "soft"

            where = f"{species} peak {peak_no} (contig {contig_num}, acc {row['accession']})"

            lengths = load_manifest(species)
            if lengths is None:
                problems.append(f"{where}: no manifest data/{species}_manifest.json")
                continue
            if contig_num not in lengths:
                problems.append(
                    f"{where}: contig {contig_num} not in manifest "
                    f"(has {', '.join(sorted(lengths))})"
                )
                continue
            if not (0 <= start_bp < end_bp):
                problems.append(f"{where}: bad coords {start_bp}-{end_bp}")
                continue
            if end_bp > lengths[contig_num]:
                problems.append(
                    f"{where}: region {start_bp}-{end_bp} overruns contig "
                    f"{contig_num} length {lengths[contig_num]}"
                )
                continue

            peaks_by_species[species].append({
                "peak": peak_no,
                "contigNum": contig_num,
                "startBp": start_bp,
                "endBp": end_bp,
                "kind": kind,
                "text": f"Peak {peak_no} · {kind}",
                "color": PEAK_COLOR,
            })

    out = {}
    for species in sorted(peaks_by_species):
        out[species] = sorted(
            peaks_by_species[species],
            key=lambda p: (p["contigNum"], p["startBp"]),
        )

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")

    n_written = sum(len(v) for v in out.values())
    print(f"read {n_rows} rows from {os.path.relpath(SRC_TSV, SCRIPT_DIR)}", file=sys.stderr)
    print(f"wrote {n_written} peaks across {len(out)} species -> "
          f"{os.path.relpath(OUT_JSON, SCRIPT_DIR)}", file=sys.stderr)
    for species in sorted(out):
        print(f"  {species}: {len(out[species])}", file=sys.stderr)
    if problems:
        print(f"\nskipped {len(problems)} row(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
