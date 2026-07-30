"""
get_aoi.py
Read RITnet's per-frame metrics CSV + the frame-extraction meta, and compute the
AOI (iris circle) in ORIGINAL video coordinates -- to seed ocular.py automatically
instead of typing it by hand. This is the RITnet -> irisometry handoff.

Uses the FIRST few frames where the iris was confidently found, takes the median
iris centre + radius (robust to a bad first frame), and maps from 640x400 mask
space back to original video space using _frames_meta.json.

Usage:
    python get_aoi.py --ritnet ..\\ritnet_8.csv --meta ..\\frames_8_full\\_frames_meta.json
"""
import argparse, csv, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ritnet", required=True, help="RITnet metrics CSV")
    ap.add_argument("--meta", required=True, help="_frames_meta.json from extraction")
    ap.add_argument("--n", type=int, default=0,
                    help="use only the first N valid frames (0 = use the whole "
                         "recording, which is the default and what you want)")
    ap.add_argument("--pad", type=float, default=1.0,
                    help="multiply the estimated radius by this before use")
    args = ap.parse_args()

    with open(args.meta) as f:
        meta = json.load(f)

    # support both meta formats (letterbox/shrink store scale + pad; older stores scale_x/y)
    if "scale" in meta:           # shrink/letterbox style: single scale + pads
        sx = sy = meta["scale"]
        px, py = meta["pad_x"], meta["pad_y"]
    else:                          # older style: per-axis scale, no pad
        sx, sy = 1.0 / meta["scale_x"], 1.0 / meta["scale_y"]
        px = py = 0.0

    def to_orig(mx, my):
        return (mx - px) / sx, (my - py) / sy

    xs, ys, rs = [], [], []
    with open(args.ritnet) as f:
        for row in csv.DictReader(f):
            if row.get("iris_found", "0") not in ("1", "1.0"):
                continue
            try:
                ix = float(row["iris_x"]); iy = float(row["iris_y"])
                idiam = float(row["iris_diam"])
            except (ValueError, KeyError):
                continue
            ox, oy = to_orig(ix, iy)
            xs.append(ox); ys.append(oy)
            rs.append((idiam / 2.0) / sx)   # radius scaled to original space
            if args.n and len(xs) >= args.n:
                break

    if not xs:
        print("No valid iris detections found in", args.ritnet)
        return

    xs = np.asarray(xs); ys = np.asarray(ys); rs = np.asarray(rs)

    # The AOI is locked for the whole run, so a bad estimate contaminates every
    # frame. Estimating from the first N frames gambles the recording on how the
    # video happens to open. Use all of it, trimmed to frames whose iris diameter
    # is near the modal value so partially-occluded frames drop out.
    keep = np.ones(len(rs), bool)
    if len(rs) > 50:
        med_r = float(np.median(rs))
        mad_r = float(np.median(np.abs(rs - med_r))) + 1e-9
        keep = np.abs(rs - med_r) < 3.0 * 1.4826 * mad_r

    cx = float(np.median(xs[keep]))
    cy = float(np.median(ys[keep]))
    r = float(np.median(rs[keep])) * args.pad

    print("Estimated AOI in ORIGINAL video coordinates")
    print("  frames with a confident iris   %d" % len(xs))
    print("  kept after diameter trimming   %d" % int(keep.sum()))
    print("  centre = (%.0f, %.0f)   radius = %.0f" % (cx, cy, r))
    print("  centre spread (IQR)  x %.1f px   y %.1f px"
          % (np.subtract(*np.percentile(xs[keep], [75, 25])),
             np.subtract(*np.percentile(ys[keep], [75, 25]))))
    print()
    print("Use with ocular.py:")
    print('  --aoi %.0f,%.0f,%.0f' % (cx, cy, r))


if __name__ == "__main__":
    main()