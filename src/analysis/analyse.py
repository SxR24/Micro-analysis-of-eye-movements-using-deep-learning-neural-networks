#!/usr/bin/env python3
"""
analyse.py
======================================================================
Test whether measured ocular torsion varies with gaze direction in the way
Listing's Law predicts.

THE PREDICTION
--------------
Listing's Law states that the eye's orientation, from primary position, is
always a rotation about an axis lying in a single plane (Listing's plane). The
torsional component of the ROTATION VECTOR is therefore ~0 for all gaze
directions.

Torsion measured in the image plane is not the rotation-vector component,
though. In Fick coordinates the eye still shows systematic "false torsion":

    psi  ~=  -(theta_h * theta_v) / 2          (angles in radians)

Converting to degrees throughout, the predicted regression slope of torsion on
the PRODUCT of horizontal and vertical gaze angle is

    beta_predicted = -1 / (2 * 180/pi) = -0.00873 deg per deg^2

The magnitude is what this script tests. The SIGN depends on the coordinate
convention (Fick vs Helmholtz) and on the camera's handedness -- image y runs
downward, and the eye may be imaged directly or via a mirror -- so the sign
must be checked against the physical setup rather than assumed.

The distinguishing feature is that torsion should depend on the PRODUCT, and be
near zero along the cardinal axes (pure horizontal or pure vertical gaze). Two
control regressions test that.

TWO PROPERTIES OF THIS DATA THAT CONSTRAIN THE ANALYSIS
-------------------------------------------------------
1. Torsion is re-referenced to zero after every blink. Absolute values are
   therefore NOT comparable across inter-blink segments. The regression
   includes a per-segment intercept (implemented as within-segment centring,
   equivalent to segment fixed effects) so that only WITHIN-segment covariation
   contributes.

2. Samples at 50 fps are heavily autocorrelated, so ordinary standard errors
   would be far too small. Confidence intervals come from a bootstrap that
   resamples whole SEGMENTS, preserving within-segment correlation.

CALIBRATION
-----------
No calibration recording is required. The horizontal visible iris is used as a
physical ruler (~11.7 mm in adults; Rufer et al. 2005, Cornea 24:259), giving
px/mm from the measured iris diameter. Gaze angle then follows from pupil
displacement on a sphere of radius ~12 mm:

    theta = arcsin(delta_pupil_px / R_eye_px)

Both constants are adjustable; they scale the slope, so they matter for
comparing against the predicted magnitude but not for the shape of the
relationship or the significance of the test.

Usage:
    python src/analysis/analyse.py --csv data/video_8/combined_8.csv \
                                   --out data/video_8/analysis
"""
import os
import argparse

import numpy as np
import pandas as pd


# Physical constants (overridable)
IRIS_MM = 11.7      # horizontal visible iris diameter, adult
R_EYE_MM = 12.0     # eyeball radius
DEG = 180.0 / np.pi
BETA_LISTING = -1.0 / (2.0 * DEG)   # deg of torsion per deg^2 of product


# ----------------------------------------------------------------------
def load(csv_path, torsion_col):
    d = pd.read_csv(csv_path)
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if torsion_col not in d:
        raise SystemExit("No column %r in %s" % (torsion_col, csv_path))

    # Segment index BEFORE filtering, so blinks delimit segments correctly.
    d["segment"] = (d["blink"].fillna(0) == 1).cumsum()

    ok = (d["blink"].fillna(0) != 1)
    if "pupil_found" in d:
        ok &= (d["pupil_found"].fillna(0) == 1)
    ok &= d[torsion_col].notna() & d["pupil_x"].notna() & d["pupil_y"].notna()
    return d[ok].copy(), len(d)


def calibrate(d, iris_mm, r_eye_mm):
    """Pupil pixels -> gaze angle in degrees, using the iris as a ruler."""
    iris_px = float(d["iris_diam"].median())
    px_per_mm = iris_px / iris_mm
    r_eye_px = r_eye_mm * px_per_mm

    # Primary position: the median gaze over all tracked frames. This is an
    # assumption -- true primary position is defined physiologically, not
    # statistically -- but with a long recording the median is a reasonable
    # stand-in, and an offset in primary position adds a linear term rather
    # than changing the product's slope.
    x0 = float(d["pupil_x"].median())
    y0 = float(d["pupil_y"].median())

    th = np.degrees(np.arcsin(np.clip((d["pupil_x"] - x0) / r_eye_px, -1, 1)))
    tv = np.degrees(np.arcsin(np.clip((d["pupil_y"] - y0) / r_eye_px, -1, 1)))
    return th, tv, dict(iris_px=iris_px, px_per_mm=px_per_mm,
                        r_eye_px=r_eye_px, x0=x0, y0=y0)


def within_segment_centre(v, seg):
    """Subtract each segment's own mean: segment fixed effects."""
    return v - v.groupby(seg).transform("mean")


def fe_slope(y, x, seg):
    """Fixed-effects OLS slope of y on x with per-segment intercepts."""
    yc = within_segment_centre(y, seg)
    xc = within_segment_centre(x, seg)
    m = np.isfinite(yc) & np.isfinite(xc)
    yc, xc = yc[m], xc[m]
    denom = float((xc ** 2).sum())
    if denom <= 0:
        return np.nan, np.nan, 0
    beta = float((yc * xc).sum() / denom)
    resid = yc - beta * xc
    ss_tot = float((yc ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    return beta, r2, int(m.sum())


def bootstrap_slope(y, x, seg, n_boot=2000, seed=0):
    """Segment-level bootstrap CI.

    Resampling individual frames would treat 50 fps samples as independent and
    produce absurdly tight intervals. Whole segments are the independent unit.
    """
    rng = np.random.default_rng(seed)
    segs = seg.unique()
    df = pd.DataFrame({"y": y.values, "x": x.values, "s": seg.values})
    groups = {s: g for s, g in df.groupby("s")}
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(segs, size=len(segs), replace=True)
        parts = []
        for j, s in enumerate(pick):
            g = groups[s].copy()
            g["s"] = j                      # re-label so repeats stay distinct
            parts.append(g)
        r = pd.concat(parts, ignore_index=True)
        bta, _, _ = fe_slope(r["y"], r["x"], r["s"])
        out[b] = bta
    return np.nanpercentile(out, [2.5, 97.5]), out


# ----------------------------------------------------------------------
def coverage_report(th, tv, product, lines):
    """Can this recording identify a product term at all?

    The product theta_h*theta_v is only identifiable if BOTH angles vary over a
    useful range and are not collinear. If the eye moved almost entirely along
    one axis, or along a single diagonal, the product is nearly a rescaled copy
    of one main effect and the test cannot separate them.
    """
    h_sd, v_sd = float(th.std()), float(tv.std())
    r_hv = float(np.corrcoef(th, tv)[0, 1])
    quad = {}
    for hs, hn in ((1, "right"), (-1, "left")):
        for vs, vn in ((1, "down"), (-1, "up")):
            quad["%s-%s" % (hn, vn)] = float(
                ((np.sign(th) == hs) & (np.sign(tv) == vs)).mean())
    minq = min(quad.values())

    lines.append("GAZE COVERAGE")
    lines.append("  horizontal SD      %6.2f deg   (p1..p99 %+.1f .. %+.1f)"
                 % (h_sd, th.quantile(.01), th.quantile(.99)))
    lines.append("  vertical   SD      %6.2f deg   (p1..p99 %+.1f .. %+.1f)"
                 % (v_sd, tv.quantile(.01), tv.quantile(.99)))
    lines.append("  corr(h, v)         %+6.3f" % r_hv)
    lines.append("  product SD         %6.2f deg^2" % float(product.std()))
    lines.append("  quadrant occupancy " + "  ".join(
        "%s %.0f%%" % (k, 100 * v) for k, v in quad.items()))
    lines.append("")

    warn = []
    if h_sd < 2.0:
        warn.append("horizontal gaze varies by only %.2f deg (SD). The product "
                    "term is then close to a rescaled vertical main effect, so "
                    "the Listing prediction cannot be cleanly separated from a "
                    "simple linear dependence on vertical gaze." % h_sd)
    if v_sd < 2.0:
        warn.append("vertical gaze varies by only %.2f deg (SD)." % v_sd)
    if abs(r_hv) > 0.5:
        warn.append("horizontal and vertical gaze are strongly correlated "
                    "(r=%+.2f): the eye moved mostly along one diagonal, so the "
                    "product is confounded with the squared main effect."
                    % r_hv)
    if minq < 0.05:
        warn.append("one gaze quadrant holds only %.1f%% of frames; the product "
                    "term is estimated from very few observations of that sign."
                    % (100 * minq))
    return warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/video_8/combined_8.csv")
    ap.add_argument("--out", default="data/video_8/analysis")
    ap.add_argument("--torsion-col", default="torsion_outer_deg",
                    help="outer iris is the cleaner estimator; inner-ring "
                         "features sit nearer the pupil boundary")
    ap.add_argument("--iris-mm", type=float, default=IRIS_MM)
    ap.add_argument("--eye-mm", type=float, default=R_EYE_MM)
    ap.add_argument("--min-seg", type=int, default=25,
                    help="ignore inter-blink segments shorter than this")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d, n_all = load(args.csv, args.torsion_col)

    # drop very short segments: their within-segment variance is mostly noise
    sizes = d.groupby("segment")["frame"].transform("size")
    d = d[sizes >= args.min_seg].copy()

    th, tv, cal = calibrate(d, args.iris_mm, args.eye_mm)
    d["th"], d["tv"] = th, tv
    d["product"] = d["th"] * d["tv"]
    tor = d[args.torsion_col]

    L = []
    L.append("=" * 72)
    L.append("LISTING'S LAW ANALYSIS -- %s" % os.path.basename(args.csv))
    L.append("=" * 72)
    L.append("")
    L.append("DATA")
    L.append("  frames in file            %d" % n_all)
    L.append("  tracked & usable          %d" % len(d))
    L.append("  inter-blink segments      %d (>= %d frames)"
             % (d["segment"].nunique(), args.min_seg))
    L.append("  torsion column            %s" % args.torsion_col)
    L.append("")
    L.append("CALIBRATION  (iris %.1f mm, eyeball radius %.1f mm)"
             % (args.iris_mm, args.eye_mm))
    L.append("  iris diameter             %.1f px" % cal["iris_px"])
    L.append("  scale                     %.2f px/mm" % cal["px_per_mm"])
    L.append("  eyeball radius            %.0f px" % cal["r_eye_px"])
    L.append("  primary position          (%.1f, %.1f) px" % (cal["x0"], cal["y0"]))
    L.append("")

    warn = coverage_report(d["th"], d["tv"], d["product"], L)

    # ---- main model + cardinal-axis controls ----
    L.append("REGRESSION  (torsion ~ predictor, with per-segment intercepts)")
    L.append("")
    L.append("  %-22s %10s %9s %9s" % ("predictor", "slope", "R2", "n"))
    L.append("  " + "-" * 54)
    results = {}
    for name, key in (("theta_h * theta_v", "product"),
                      ("theta_h  (control)", "th"),
                      ("theta_v  (control)", "tv")):
        b, r2, n = fe_slope(tor, d[key], d["segment"])
        results[key] = (b, r2, n)
        L.append("  %-22s %10.5f %9.4f %9d" % (name, b, r2, n))
    L.append("")

    beta, r2, _ = results["product"]
    (lo, hi), dist = bootstrap_slope(tor, d["product"], d["segment"],
                                     n_boot=args.boot)
    L.append("  Segment-level bootstrap (%d resamples):" % args.boot)
    L.append("    slope            %+.5f deg/deg^2" % beta)
    L.append("    95%% CI           [%+.5f, %+.5f]" % (lo, hi))
    L.append("    Listing predicts %+.5f  (magnitude %.5f)"
             % (BETA_LISTING, abs(BETA_LISTING)))
    consistent = (lo <= abs(BETA_LISTING) <= hi) or (lo <= BETA_LISTING <= hi)
    excludes0 = not (lo <= 0 <= hi)
    L.append("    CI excludes zero:            %s" % ("YES" if excludes0 else "no"))
    L.append("    CI contains Listing value:   %s" % ("YES" if consistent else "no"))
    L.append("")

    if warn:
        L.append("!" * 72)
        L.append("POWER WARNINGS -- read before interpreting the slope above")
        L.append("!" * 72)
        for w in warn:
            L.append("  * " + w)
        L.append("")
        L.append("  A slope estimated under these conditions may be precise but")
        L.append("  not meaningful: the design cannot distinguish the Listing")
        L.append("  product term from simpler alternatives. Recordings covering")
        L.append("  a wider, less correlated range of gaze positions are needed")
        L.append("  before treating this as a test of Listing's Law.")
        L.append("")

    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(args.out, "listing_report.txt"), "w") as f:
        f.write(txt + "\n")

    d[["frame", "time", "th", "tv", "product", args.torsion_col, "segment"]].to_csv(
        os.path.join(args.out, "listing_data.csv"), index=False)

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("(figures skipped:", e, ")")
        return

    tor_c = within_segment_centre(tor, d["segment"])
    prod_c = within_segment_centre(d["product"], d["segment"])

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # gaze coverage
    a = ax[0, 0]
    a.hexbin(d["th"], d["tv"], gridsize=45, cmap="viridis", bins="log")
    a.axhline(0, lw=.6, color="w"); a.axvline(0, lw=.6, color="w")
    a.set_xlabel("horizontal gaze (deg)"); a.set_ylabel("vertical gaze (deg)")
    a.set_title("Gaze coverage")

    # binned torsion vs product
    a = ax[0, 1]
    q = pd.qcut(prod_c, 18, duplicates="drop")
    g = pd.DataFrame({"p": prod_c, "t": tor_c}).groupby(q, observed=True)
    mu, se, xc = g["t"].mean(), g["t"].sem(), g["p"].mean()
    a.errorbar(xc, mu, yerr=se, fmt="o", ms=4, lw=1, color="#1f77b4")
    xs = np.linspace(prod_c.min(), prod_c.max(), 50)
    a.plot(xs, beta * xs, "-", color="#d62728",
           label="fitted %.5f" % beta)
    a.plot(xs, BETA_LISTING * xs, "--", color="#2ca02c",
           label="Listing %.5f" % BETA_LISTING)
    a.axhline(0, lw=.6, color="k"); a.axvline(0, lw=.6, color="k")
    a.set_xlabel(r"$\theta_h \cdot \theta_v$ (deg$^2$, segment-centred)")
    a.set_ylabel("torsion (deg, segment-centred)")
    a.set_title("Listing's Law test")
    a.legend(fontsize=8)

    # controls
    for a, key, lab in ((ax[1, 0], "th", "horizontal gaze (deg)"),
                        (ax[1, 1], "tv", "vertical gaze (deg)")):
        xc_ = within_segment_centre(d[key], d["segment"])
        qq = pd.qcut(xc_, 18, duplicates="drop")
        gg = pd.DataFrame({"x": xc_, "t": tor_c}).groupby(qq, observed=True)
        a.errorbar(gg["x"].mean(), gg["t"].mean(), yerr=gg["t"].sem(),
                   fmt="o", ms=4, lw=1, color="#7f7f7f")
        b_, r2_, _ = results[key]
        a.plot(np.sort(xc_), b_ * np.sort(xc_), "-", color="#d62728",
               label="slope %.4f" % b_)
        a.axhline(0, lw=.6, color="k")
        a.set_xlabel(lab + "  (segment-centred)")
        a.set_ylabel("torsion (deg)")
        a.set_title("Control: should be flat if Listing holds")
        a.legend(fontsize=8)

    fig.tight_layout()
    p = os.path.join(args.out, "listing_analysis.png")
    fig.savefig(p, dpi=150)
    print("Wrote:", p)
    print("Wrote:", os.path.join(args.out, "listing_report.txt"))


if __name__ == "__main__":
    main()
