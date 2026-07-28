"""
merge.py
======================================================================
Fuse the two halves of the pipeline into ONE per-frame table:

  * RITnet metrics  (ritnet_*.csv)  -> gaze / pupil / iris  (640x400 mask space)
  * ocular pipeline (ocular_*.csv)  -> torsion              (original video space)

Joined on FRAME INDEX. RITnet's coordinates are transformed back to ORIGINAL
video space using the extraction metadata (_frames_meta.json), so every column
in the output is in the same coordinate system and aligned to the same frame.

OUTPUT: combined_<name>.csv with columns:
    frame, time,
    pupil_x, pupil_y, pupil_diam,        (RITnet, original coords)
    iris_x, iris_y, iris_diam,           (RITnet, original coords)
    pupil_found,                          (RITnet blink/quality signal)
    torsion_deg, torsion_inner_deg, torsion_outer_deg,   (irisometry)
    n_features, blink                     (irisometry)

This combined.csv is the integrated deliverable: gaze+pupil (deep learning) and
torsion (classical feature tracking) for the same eye, frame by frame.

Usage:
    python merge.py --ritnet ritnet_8.csv --ocular output/ocular_8.csv \\
                    --meta frames_8_full/_frames_meta.json --out combined_8.csv
"""
import argparse, csv, json
import numpy as np


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ritnet", required=True)
    ap.add_argument("--ocular", required=True)
    ap.add_argument("--meta", required=True, help="_frames_meta.json from extraction")
    ap.add_argument("--out", default="combined.csv")
    args = ap.parse_args()

    # --- coordinate transform from meta (mask 640x400 -> original video) ---
    with open(args.meta) as f:
        meta = json.load(f)
    if "scale" in meta:                      # shrink / letterbox style
        sx = sy = meta["scale"]
        px, py = meta["pad_x"], meta["pad_y"]
    else:                                     # older per-axis style
        sx, sy = 1.0 / meta["scale_x"], 1.0 / meta["scale_y"]
        px = py = 0.0

    def to_orig_x(mx):
        return (mx - px) / sx if not np.isnan(mx) else np.nan

    def to_orig_y(my):
        return (my - py) / sy if not np.isnan(my) else np.nan

    def to_orig_len(d):
        return d / sx if not np.isnan(d) else np.nan   # diameters scale by sx

    # --- index RITnet rows by frame ---
    ritnet = {}
    for r in load_csv(args.ritnet):
        fr = int(fnum(r["frame"]))
        ritnet[fr] = r

    # --- walk ocular rows (these define the master frame list & time) ---
    ocular_rows = load_csv(args.ocular)

    out_cols = ["frame", "time",
                "pupil_x", "pupil_y", "pupil_diam",
                "iris_x", "iris_y", "iris_diam",
                "pupil_found",
                "torsion_deg", "torsion_inner_deg", "torsion_outer_deg",
                "n_features", "blink"]

    n_matched = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols)
        w.writeheader()
        for o in ocular_rows:
            fr = int(fnum(o["frame"]))
            row = {c: "" for c in out_cols}
            row["frame"] = fr
            row["time"] = o.get("time", "")
            # irisometry side (already original coords)
            row["torsion_deg"] = o.get("torsion_deg", "")
            row["torsion_inner_deg"] = o.get("torsion_inner_deg", "")
            row["torsion_outer_deg"] = o.get("torsion_outer_deg", "")
            row["n_features"] = o.get("n_features", "")
            row["blink"] = o.get("blink", "")
            # RITnet side (transform coords to original space)
            if fr in ritnet:
                n_matched += 1
                rr = ritnet[fr]
                pf = int(fnum(rr.get("pupil_found", 0)))
                row["pupil_found"] = pf
                px_ = to_orig_x(fnum(rr.get("pupil_x")))
                py_ = to_orig_y(fnum(rr.get("pupil_y")))
                pd_ = to_orig_len(fnum(rr.get("pupil_diam")))
                ix_ = to_orig_x(fnum(rr.get("iris_x")))
                iy_ = to_orig_y(fnum(rr.get("iris_y")))
                id_ = to_orig_len(fnum(rr.get("iris_diam")))
                row["pupil_x"] = "" if np.isnan(px_) else round(px_, 2)
                row["pupil_y"] = "" if np.isnan(py_) else round(py_, 2)
                row["pupil_diam"] = "" if np.isnan(pd_) else round(pd_, 2)
                row["iris_x"] = "" if np.isnan(ix_) else round(ix_, 2)
                row["iris_y"] = "" if np.isnan(iy_) else round(iy_, 2)
                row["iris_diam"] = "" if np.isnan(id_) else round(id_, 2)
            w.writerow(row)

    print("Merged %d ocular frames; %d matched a RITnet frame." %
          (len(ocular_rows), n_matched))
    print("Wrote:", args.out)
    print("Coordinate transform applied: x=(mx-%g)/%g, y=(my-%g)/%g"
          % (px, sx, py, sy))


if __name__ == "__main__":
    main()