#!/usr/bin/env python3
"""
reliability.py
======================================================================
How much of the measured torsion signal is real?

THE METHOD
----------
Every other diagnostic here is internal -- within-segment SD, drift, jitter --
and none of them separates a trace that is tracking the eye from one that is
smoothly wrong. Weak features slide SMOOTHLY under Lucas-Kanade, improving
jitter while accumulating error, so a smoother trace can be a worse measurement.

Split-half reliability settles it without ground truth. Split the tracked
features into two random halves, compute torsion independently from each half of
the SAME frame, and correlate the two series within segment. A real common
rotation appears in both halves; per-feature noise does not. Spearman-Brown then
converts the half-set correlation into the reliability of the full estimate:

    reliability = 2r / (1 + r)

and the variance splits as SD_total^2 = SD_signal^2 + SD_noise^2, with
SD_signal = SD_total * sqrt(reliability).

INTERPRETATION
--------------
This is an upper bound, not an estimate. Anything common to both halves inflates
it -- iris deformation under pupil dilation, mask-boundary changes, a global LK
bias -- and none of those are eye rotation. A low value is conclusive; a high
value is necessary but not sufficient.

Reference measurement (video 8, frame-to-frame LK chaining, circular median):

    split-half r  0.302     reliability  0.486
    signal SD     0.26 deg  noise SD     0.39 deg
    rigid-fit residual rose 1.98 px -> 10.34 px across a segment

Usage:
    python src/analysis/reliability.py --features data/video_8/features_8.npz
    python src/analysis/reliability.py --features new.npz --baseline old.npz
    python src/analysis/reliability.py --features f.npz --assert-min 0.55

Sohil Ananth, MSc Bioinformatics & CS, University of Leicester
"""
import os
import sys
import json
import math
import argparse

import numpy as np


# ----------------------------------------------------------------------
def procrustes_deg(ref, cur):
    """Least-squares rigid rotation ref -> cur, in degrees, plus median residual.

    Kept deliberately independent of ocular.py: this is a measuring instrument,
    and it should not silently change when the thing being measured changes.
    """
    if len(ref) < 3:
        return np.nan, np.nan
    a = ref - ref.mean(axis=0)
    b = cur - cur.mean(axis=0)
    S = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())
    C = float((a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum())
    if S == 0.0 and C == 0.0:
        return np.nan, np.nan
    th = math.atan2(S, C)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    resid = float(np.median(np.linalg.norm(b - a @ R.T, axis=1)))
    return math.degrees(th), resid


def within_centre(v, seg):
    """Subtract each segment's own mean (torsion is re-referenced per segment)."""
    out = np.full(len(v), np.nan)
    for s in np.unique(seg):
        m = seg == s
        if m.sum() > 1:
            out[m] = v[m] - np.nanmean(v[m])
    return out


# ----------------------------------------------------------------------
def measure(npz_path, min_seg=25, min_feat=20, half_min=10, seed=0):
    z = np.load(npz_path)
    XY, OK, SEG = z["feat_xy"], z["feat_ok"], z["seg_id"]
    rng = np.random.default_rng(seed)
    half = rng.random(XY.shape[1]) < 0.5

    A, B, F, S, R, K = [], [], [], [], [], []
    for s in np.unique(SEG[SEG >= 0]):
        idx = np.flatnonzero(SEG == s)
        if len(idx) < min_seg:
            continue
        ref, refok = XY[idx[0]], OK[idx[0]]
        for k, f in enumerate(idx):
            sel = OK[f] & refok
            if sel.sum() < min_feat:
                continue
            full, resid = procrustes_deg(ref[sel], XY[f][sel])
            ha = sel & half
            hb = sel & ~half
            a = procrustes_deg(ref[ha], XY[f][ha])[0] if ha.sum() >= half_min else np.nan
            b = procrustes_deg(ref[hb], XY[f][hb])[0] if hb.sum() >= half_min else np.nan
            A.append(a); B.append(b); F.append(full)
            R.append(resid); S.append(s); K.append(k / max(1, len(idx) - 1))

    A, B, F = np.array(A), np.array(B), np.array(F)
    S, R, K = np.array(S), np.array(R), np.array(K)
    if len(A) < 100:
        raise SystemExit("Too few usable frames in %s (%d)" % (npz_path, len(A)))

    Ac, Bc, Fc = within_centre(A, S), within_centre(B, S), within_centre(F, S)
    m = np.isfinite(Ac) & np.isfinite(Bc)
    r = float(np.corrcoef(Ac[m], Bc[m])[0, 1])
    rel = 2 * r / (1 + r) if r > -1 else np.nan
    sd = float(np.nanstd(Fc))

    # residual by position within segment -- the LK-slide signature
    prof = []
    for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
        sel = (K >= lo) & (K < hi) & np.isfinite(R)
        prof.append(float(np.median(R[sel])) if sel.sum() > 10 else np.nan)

    return dict(
        path=npz_path, n_frames=int(len(A)), n_segments=int(len(np.unique(S))),
        split_r=r, reliability=rel,
        sd_total=sd,
        sd_signal=sd * math.sqrt(max(rel, 0.0)),
        sd_noise=sd * math.sqrt(max(1.0 - max(rel, 0.0), 0.0)),
        resid_median=float(np.nanmedian(R)),
        resid_profile=prof,
        resid_growth=(prof[-1] / prof[0]) if (prof[0] and np.isfinite(prof[0])
                                              and np.isfinite(prof[-1])) else np.nan,
    )


def report(res, label=""):
    L = []
    L.append("  %-22s %s" % ("file", os.path.basename(res["path"])))
    L.append("  %-22s %d frames, %d segments"
             % ("usable", res["n_frames"], res["n_segments"]))
    L.append("  %-22s %+.3f" % ("split-half r", res["split_r"]))
    L.append("  %-22s %.3f" % ("reliability (S-B)", res["reliability"]))
    L.append("  %-22s %.3f deg total = %.3f signal + %.3f noise"
             % ("torsion SD", res["sd_total"], res["sd_signal"], res["sd_noise"]))
    L.append("  %-22s %.2f px" % ("rigid-fit residual", res["resid_median"]))
    L.append("  %-22s %s px"
             % ("  start -> end of seg",
                "  ".join("%.1f" % v if np.isfinite(v) else "--"
                          for v in res["resid_profile"])))
    L.append("  %-22s %.2fx  %s"
             % ("  growth across seg", res["resid_growth"],
                "(flat is good; >1.5 means LK slide is accumulating)"))
    return ("\n".join(L)) if not label else (label + "\n" + "\n".join(L))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Split-half reliability of the torsion estimate")
    ap.add_argument("--features", required=True, help="features_*.npz")
    ap.add_argument("--baseline", default=None,
                    help="a second features_*.npz to compare against")
    ap.add_argument("--min-seg", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write results as JSON here")
    ap.add_argument("--assert-min", type=float, default=None,
                    help="exit non-zero if reliability falls below this. Use in "
                         "a regression check so a change that smooths the trace "
                         "while degrading the measurement cannot pass silently.")
    args = ap.parse_args()

    print("=" * 68)
    print("TORSION RELIABILITY  (split-half, within-segment)")
    print("=" * 68)
    print()
    cur = measure(args.features, args.min_seg, seed=args.seed)
    print(report(cur, "CURRENT"))

    if args.baseline:
        print()
        base = measure(args.baseline, args.min_seg, seed=args.seed)
        print(report(base, "BASELINE"))
        print()
        print("CHANGE")
        print("  reliability        %.3f -> %.3f   (%+.3f)"
              % (base["reliability"], cur["reliability"],
                 cur["reliability"] - base["reliability"]))
        print("  noise SD           %.3f -> %.3f deg   (%+.1f%%)"
              % (base["sd_noise"], cur["sd_noise"],
                 100 * (cur["sd_noise"] / base["sd_noise"] - 1)))
        print("  residual growth    %.2fx -> %.2fx"
              % (base["resid_growth"], cur["resid_growth"]))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(cur, f, indent=2)
        print("\nWrote:", args.json)

    if args.assert_min is not None:
        ok = cur["reliability"] >= args.assert_min
        print("\nASSERT reliability >= %.3f : %s"
              % (args.assert_min, "PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
