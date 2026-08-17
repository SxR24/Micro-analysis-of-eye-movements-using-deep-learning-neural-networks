#!/usr/bin/env python3
"""
compare_openiris.py
======================================================================
Agreement between this pipeline (RITnet + irisometry) and OpenIris on the
same frames of the same recording.

WHY AN EXTERNAL COMPARISON
--------------------------
reliability.py measures split-half reliability, which is an UPPER BOUND: it
cannot separate a real rotation from an artefact common to both halves of the
iris. An independent implementation, with a different algorithm, run on the
same frames, is the only check available here that speaks to VALIDITY rather
than to internal consistency.

The two methods are genuinely independent:

    this pipeline   Shi-Tomasi corners + Lucas-Kanade tracked from a
                    reference frame, rigid rotation fitted to the
                    displacements, features gated to RITnet's iris class
    OpenIris (JOM)  polar unwrap of the iris annulus, cross-correlated
                    against a reference template, no lid exclusion by default

Nothing is shared between them except the video.

FOUR ARMS
---------
1. PUPIL. Both estimate pupil centre and size. Deep-learning segmentation
   against classical blob detection with ellipse fitting. This arm works
   whether or not torsion does.
2. TORSION. Both estimate rotation about the line of sight.
3. BAND. A near-zero correlation between a smooth series and a noisy one does
   not by itself show the noisy one carries no signal, since a buried
   low-frequency signal would give the same result. The same low-pass filter is
   applied to both traces across a range of cutoffs and the correlation
   recomputed, which separates "measured badly" from "not measured".
4. CONTROL. Agreement is also computed after averaging into progressively
   longer blocks, for the pupil as well as for torsion. The pupil serves as an
   internal control on frame pairing: if the two files were misaligned no
   channel could agree, so pupil agreement that rises with averaging while
   torsion agreement does not locates the failure in the torsion channel
   rather than in the comparison itself.

WHAT MUST BE HANDLED
--------------------
* Reference offset. Torsion is relative to whichever frame each method took as
  its zero, so absolute values are not comparable. Both are centred within
  segment before correlating.
* Sign convention. Image y runs downward and the eye may be viewed via a
  mirror, so the sign is not guaranteed to match. --sign flips OpenIris.
* Failed frames. OpenIris emits a frame-centre fallback (pupil width equal to
  the frame width) when it finds no pupil. Those are not measurements and are
  excluded, as are blinks and untracked frames on our side.
* Temporal offset. Cross-correlation over a range of lags checks the two are
  actually aligned frame for frame before agreement is interpreted.

Usage:
    python src/analysis/compare_openiris.py \
        --ours data/video_8/combined_8.csv \
        --openiris data/openiris/8-PostProc-.../8-PostProc-....txt \
        --out data/video_8/analysis

Sohil Ananth, MSc Bioinformatics & CS, University of Leicester
"""
import os
import glob
import time
import hashlib
import argparse

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------
def load_openiris(path, eye="Left", frame_w=None, frame_h=None):
    """Read an OpenIris session .txt and return frame, pupil and torsion.

    OpenIris writes a frame-centre fallback rather than a NaN when it fails to
    find a pupil: pupil x,y sit at the image centre and pupil width equals the
    image width. Those rows look like data and must be removed explicitly.
    """
    d = pd.read_csv(path, sep=r"\s+")
    need = [eye + c for c in ("FrameNumber", "PupilX", "PupilY",
                              "PupilWidth", "PupilHeight", "Torsion")]
    miss = [c for c in need if c not in d.columns]
    if miss:
        raise SystemExit("Missing columns in %s: %s" % (path, miss))

    out = pd.DataFrame({
        "frame": pd.to_numeric(d[eye + "FrameNumber"], errors="coerce"),
        "oi_pupil_x": pd.to_numeric(d[eye + "PupilX"], errors="coerce"),
        "oi_pupil_y": pd.to_numeric(d[eye + "PupilY"], errors="coerce"),
        "oi_pupil_w": pd.to_numeric(d[eye + "PupilWidth"], errors="coerce"),
        "oi_pupil_h": pd.to_numeric(d[eye + "PupilHeight"], errors="coerce"),
        "oi_torsion": pd.to_numeric(d[eye + "Torsion"], errors="coerce"),
    })

    if frame_w is None:
        # the fallback width is the modal value equal to the largest width seen
        frame_w = out["oi_pupil_w"].max()
    fallback = np.isclose(out["oi_pupil_w"], frame_w, atol=1.0)
    zeros = (out["oi_pupil_x"] == 0) & (out["oi_pupil_y"] == 0)
    # Plausibility bounds. Removing the frame-centre fallback is not enough:
    # OpenIris also emits diverged ellipse fits with centres far outside the
    # image (values beyond -260,000 px occur on this recording). A handful of
    # those is sufficient to drag a Pearson correlation to near zero, so they
    # must be excluded before any agreement statistic. These are the same
    # bounds used to count usable pupils elsewhere, so the reported detection
    # rate and the reported agreement refer to the same set of frames.
    implausible = ~(out["oi_pupil_x"].between(200, 750)
                    & out["oi_pupil_y"].between(150, 600)
                    & out["oi_pupil_w"].between(120, 320))
    out["oi_valid"] = (~(fallback | zeros | implausible)
                       & out["oi_pupil_x"].notna())
    out.loc[~out["oi_valid"], ["oi_pupil_x", "oi_pupil_y",
                               "oi_pupil_w", "oi_torsion"]] = np.nan
    out["frame"] = out["frame"].astype("Int64")
    return out, int(fallback.sum()), int(zeros.sum())


def load_ours(path):
    d = pd.read_csv(path)
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "seg" in d.columns and d["seg"].notna().any():
        d["segment"] = d["seg"]
    else:
        d["segment"] = (d["blink"].fillna(0) == 1).cumsum()
    d["ours_valid"] = ((d["blink"].fillna(1) != 1)
                       & (d["pupil_found"].fillna(0) == 1)
                       & d["torsion_deg"].notna())
    if "seg" in d.columns:
        d["ours_valid"] &= d["seg"].fillna(-1) >= 0
    d["frame"] = d["frame"].astype("Int64")
    return d


def within_centre(v, seg):
    return v - v.groupby(seg).transform("mean")


def bland_altman(a, b):
    """Bias and 95% limits of agreement for two paired measurements."""
    d = a - b
    bias = float(np.nanmean(d))
    sd = float(np.nanstd(d))
    return bias, sd, bias - 1.96 * sd, bias + 1.96 * sd


def lowpass(x, fc, fs=50.0, order=4):
    """Zero-phase Butterworth low-pass.

    filtfilt runs the filter forwards and backwards, so there is no phase lag
    to confound a correlation between two series. The same filter is applied to
    BOTH traces: filtering one and not the other would change their frequency
    content relative to each other and the comparison would stop being
    like-for-like.
    """
    from scipy.signal import butter, filtfilt
    b, a = butter(order, fc / (fs / 2.0), btype="low")
    return filtfilt(b, a, x)


def filter_sweep(d, ours_col, oi_col, fs=50.0,
                 cutoffs=(0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 24),
                 min_len=40):
    """Within-segment correlation as a function of low-pass cutoff.

    WHY THIS TEST EXISTS
    --------------------
    A near-zero correlation between a smooth series and a noisy one does not by
    itself show that the noisy one carries no signal. Real ocular torsion is a
    low-frequency quantity; the eye cannot rotate torsionally at 25 Hz. If a
    method were measuring torsion badly rather than not at all, its signal would
    sit under high-frequency noise and a correlation computed across the whole
    spectrum would be near zero either way.

    Removing the noise band and correlating again separates the two cases. A
    correlation that rises substantially means the signal was present and
    buried; one that stays low means it was not there to begin with.
    """
    d = d[np.isfinite(d[ours_col]) & np.isfinite(d[oi_col])]
    rows = []
    for fc in cutoffs:
        A, B = [], []
        for _, g in d.groupby("segment"):
            if len(g) < min_len:
                continue
            try:
                a = lowpass(g[ours_col].values.astype(float), fc, fs)
                b = lowpass(g[oi_col].values.astype(float), fc, fs)
            except Exception:
                continue
            A.append(a - a.mean())
            B.append(b - b.mean())
        if not A:
            continue
        A, B = np.concatenate(A), np.concatenate(B)
        rows.append(dict(cutoff_hz=fc, r=float(np.corrcoef(A, B)[0, 1]),
                         n=len(A), sd_ours=float(A.std()), sd_oi=float(B.std())))
    return pd.DataFrame(rows)


def averaging_sweep(d, ours_col, oi_col, within_segment,
                    windows=(1, 10, 25, 50, 100, 250)):
    """Correlation after averaging into blocks of increasing length.

    A second, filter-free way of asking the same question. Torsion must be
    centred within segment because it is reset at every segment boundary; pupil
    position must not be, since it is an absolute quantity that carries across
    boundaries and centring it would discard most of its variance.
    """
    rows = []
    for w in windows:
        t = d.copy()
        t["_blk"] = t["frame"] // w
        keys = ["segment", "_blk"] if within_segment else ["_blk"]
        g = t.groupby(keys).agg(a=(ours_col, "mean"), b=(oi_col, "mean"),
                                n=("frame", "size"))
        g = g[g["n"] >= max(1, w * 0.8)].dropna()
        if len(g) < 30:
            continue
        if within_segment:
            c = lambda v: v - v.groupby(level=0).transform("mean")
            ga, gb = c(g["a"]), c(g["b"])
        else:
            ga, gb = g["a"], g["b"]
        rows.append(dict(window_frames=w, window_s=w / 50.0,
                         r=float(ga.corr(gb)), n=len(g)))
    return pd.DataFrame(rows)


def bootstrap_r(d, ours_col, oi_col, fc, fs=50.0, n_boot=2000, seed=0,
                min_len=40):
    """Segment-level bootstrap CI for the filtered correlation.

    Whole segments are the independent unit; frames within one are heavily
    autocorrelated and resampling them would give an absurdly tight interval.
    """
    d = d[np.isfinite(d[ours_col]) & np.isfinite(d[oi_col])]
    segs = {}
    for s, g in d.groupby("segment"):
        if len(g) < min_len:
            continue
        try:
            a = lowpass(g[ours_col].values.astype(float), fc, fs)
            b = lowpass(g[oi_col].values.astype(float), fc, fs)
        except Exception:
            continue
        segs[s] = (a - a.mean(), b - b.mean())
    if len(segs) < 5:
        return np.nan, np.nan, np.nan, 0
    keys = list(segs)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        A = np.concatenate([segs[keys[j]][0] for j in pick])
        B = np.concatenate([segs[keys[j]][1] for j in pick])
        out[i] = np.corrcoef(A, B)[0, 1]
    A = np.concatenate([v[0] for v in segs.values()])
    B = np.concatenate([v[1] for v in segs.values()])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(np.corrcoef(A, B)[0, 1]), float(lo), float(hi), len(segs)


def lag_profile(a, b, max_lag=10):
    """Correlation as a function of integer frame lag.

    If the peak is not at zero the two series are misaligned, and any agreement
    at lag 0 understates the true correspondence.
    """
    out = []
    for L in range(-max_lag, max_lag + 1):
        x = a.shift(L)
        m = np.isfinite(x) & np.isfinite(b)
        out.append((L, float(np.corrcoef(x[m], b[m])[0, 1]) if m.sum() > 100
                    else np.nan))
    return out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Agreement between this pipeline and OpenIris")
    ap.add_argument("--ours", default="data/video_8/combined_8.csv")
    ap.add_argument("--openiris", default=None,
                    help="OpenIris session .txt (default: newest under "
                         "data/openiris/)")
    ap.add_argument("--out", default="data/video_8/analysis")
    ap.add_argument("--eye", default="Left", choices=("Left", "Right"))
    ap.add_argument("--torsion-col", default="torsion_deg")
    ap.add_argument("--sign", type=int, default=1, choices=(1, -1),
                    help="flip the sign of the OpenIris torsion")
    ap.add_argument("--min-seg", type=int, default=25)
    ap.add_argument("--max-lag", type=int, default=10)
    args = ap.parse_args()

    if args.openiris is None:
        cand = sorted(glob.glob(os.path.join("data", "openiris", "*", "*.txt")))
        cand = [c for c in cand if not c.endswith("-log.log")]
        if not cand:
            raise SystemExit("No OpenIris .txt found under data/openiris/")
        args.openiris = max(cand, key=os.path.getmtime)

    os.makedirs(args.out, exist_ok=True)
    oi, n_fallback, n_zero = load_openiris(args.openiris, args.eye)
    ours = load_ours(args.ours)
    m = ours.merge(oi, on="frame", how="inner")

    L = []
    L.append("=" * 72)
    L.append("METHOD COMPARISON -- this pipeline vs OpenIris")
    L.append("=" * 72)
    L.append("")
    L.append("INPUTS")
    L.append("  ours       %s  [%s]" % (os.path.basename(args.ours),
                                        sha(args.ours)))
    L.append("  openiris   %s  [%s]" % (os.path.basename(args.openiris),
                                        sha(args.openiris)))
    L.append("  eye        %s      generated %s"
             % (args.eye, time.strftime("%Y-%m-%d %H:%M:%S")))
    L.append("")
    L.append("FRAME COVERAGE")
    L.append("  frames in common                %d" % len(m))
    L.append("  OpenIris frame-centre fallback  %d  (no pupil found)" % n_fallback)
    L.append("  OpenIris all-zero rows          %d" % n_zero)
    L.append("  OpenIris usable                 %d (%.0f%%)"
             % (m["oi_valid"].sum(), 100 * m["oi_valid"].mean()))
    L.append("  ours usable                     %d (%.0f%%)"
             % (m["ours_valid"].sum(), 100 * m["ours_valid"].mean()))
    both = m["oi_valid"] & m["ours_valid"]
    L.append("  BOTH usable                     %d (%.0f%%)"
             % (both.sum(), 100 * both.mean()))
    L.append("")

    if both.sum() < 100:
        L.append("Too few frames measured by both methods to compare.")
        txt = "\n".join(L)
        print(txt)
        open(os.path.join(args.out, "openiris_comparison.txt"), "w").write(txt)
        return

    b = m[both].copy()

    # ---------------- arm 1: pupil ----------------
    L.append("-" * 72)
    L.append("ARM 1 -- PUPIL  (segmentation vs blob + ellipse fitting)")
    L.append("-" * 72)
    L.append("")
    L.append("  %-14s %9s %9s %9s %9s %9s"
             % ("measure", "r", "bias", "SD diff", "LoA lo", "LoA hi"))
    L.append("  " + "-" * 64)
    pupil_rows = [("pupil x", "pupil_x", "oi_pupil_x"),
                  ("pupil y", "pupil_y", "oi_pupil_y"),
                  ("pupil diam", "pupil_diam", "oi_pupil_w")]
    for name, ca, cb in pupil_rows:
        if ca not in b or cb not in b:
            continue
        x, y = b[ca], b[cb]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 100:
            continue
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        bias, sd, lo, hi = bland_altman(x[ok], y[ok])
        L.append("  %-14s %9.3f %9.2f %9.2f %9.2f %9.2f"
                 % (name, r, bias, sd, lo, hi))
    L.append("")
    L.append("  Bias is ours minus OpenIris, in pixels of the original video.")
    L.append("  A large correlation with a constant bias means the two agree on")
    L.append("  the movement but differ on the absolute definition of centre,")
    L.append("  which is expected: a segmentation centroid and an ellipse-fit")
    L.append("  centre are not the same quantity.")
    L.append("")

    # ---------------- arm 2: torsion ----------------
    L.append("-" * 72)
    L.append("ARM 2 -- TORSION  (feature tracking vs iris cross-correlation)")
    L.append("-" * 72)
    L.append("")
    sizes = b.groupby("segment")["frame"].transform("size")
    t = b[sizes >= args.min_seg].copy()
    L.append("  segments >= %d frames            %d"
             % (args.min_seg, t["segment"].nunique()))
    L.append("  frames used                      %d" % len(t))
    L.append("")

    if len(t) > 100:
        t["ours_c"] = within_centre(t[args.torsion_col], t["segment"])
        t["oi_c"] = within_centre(args.sign * t["oi_torsion"], t["segment"])
        ok = np.isfinite(t["ours_c"]) & np.isfinite(t["oi_c"])
        r = float(np.corrcoef(t.loc[ok, "ours_c"], t.loc[ok, "oi_c"])[0, 1])
        bias, sd, lo, hi = bland_altman(t.loc[ok, "ours_c"], t.loc[ok, "oi_c"])
        L.append("  within-segment correlation       %+.3f  (sign %+d)"
                 % (r, args.sign))
        L.append("  bias                             %+.3f deg" % bias)
        L.append("  95%% limits of agreement          [%+.3f, %+.3f] deg"
                 % (lo, hi))
        L.append("  SD, ours                         %.3f deg"
                 % float(t.loc[ok, "ours_c"].std()))
        L.append("  SD, OpenIris                     %.3f deg"
                 % float(t.loc[ok, "oi_c"].std()))
        L.append("")

        L.append("  lag-1 autocorrelation (is each trace even self-consistent?)")
        L.append("    ours       %+.3f" % t["ours_c"].autocorr(1))
        L.append("    OpenIris   %+.3f" % t["oi_c"].autocorr(1))
        L.append("    A trace with autocorrelation near zero at 50 fps is")
        L.append("    frame-to-frame noise; a real rotation is smooth.")
        L.append("")

        lp = lag_profile(t["ours_c"], t["oi_c"], args.max_lag)
        best = max((v for v in lp if np.isfinite(v[1])),
                   key=lambda v: abs(v[1]), default=(0, np.nan))
        L.append("  cross-correlation vs frame lag")
        L.append("    " + "  ".join("%+d:%+.2f" % (l, c)
                                    for l, c in lp if abs(l) <= 5))
        L.append("    peak at lag %+d (r=%+.3f)" % (best[0], best[1]))
        if best[0] != 0 and abs(best[1]) > abs(r) + 0.05:
            L.append("    NOTE: peak is not at lag 0, so the two series may be")
            L.append("    misaligned in time. Check frame indexing.")
        L.append("")

        if "torsion_resid_px" in t.columns:
            dif = (t["ours_c"] - t["oi_c"]).abs()
            rr = t["torsion_resid_px"]
            okr = np.isfinite(dif) & np.isfinite(rr)
            if okr.sum() > 100:
                cr = float(np.corrcoef(dif[okr], rr[okr])[0, 1])
                L.append("  corr(|disagreement|, our rigid-fit residual)  %+.3f"
                         % cr)
                L.append("    Positive means our own quality channel predicts")
                L.append("    where the two methods disagree, which validates")
                L.append("    torsion_resid_px as a confidence measure.")
                L.append("")

        # ---------------- arm 3: is the disagreement a band effect? ----------
        L.append("-" * 72)
        L.append("ARM 3 -- DOES THE DISAGREEMENT SURVIVE REMOVING THE NOISE BAND?")
        L.append("-" * 72)
        L.append("")
        L.append("  A low correlation between a smooth series and a noisy one does not")
        L.append("  on its own show the noisy one carries no signal. Torsion is a")
        L.append("  low-frequency quantity, so a buried signal would give a low")
        L.append("  correlation across the whole spectrum either way. The same")
        L.append("  low-pass filter is applied to BOTH traces and the correlation")
        L.append("  recomputed.")
        L.append("")
        sw = filter_sweep(t, args.torsion_col, "oi_torsion")
        if len(sw):
            L.append("  %-11s %8s %10s %12s %10s"
                     % ("cutoff Hz", "r", "n frames", "SD ours deg", "SD OI deg"))
            L.append("  " + "-" * 55)
            for _, r_ in sw.iterrows():
                L.append("  %-11.1f %+8.3f %10d %12.3f %10.3f"
                         % (r_.cutoff_hz, r_.r, r_.n, r_.sd_ours, r_.sd_oi))
            sw_ok = sw[np.isfinite(sw["r"])]
            best = sw_ok.loc[sw_ok["r"].abs().idxmax()] if len(sw_ok) else None
        if len(sw) and best is not None:
            rb, lo_b, hi_b, nseg_b = bootstrap_r(
                t, args.torsion_col, "oi_torsion", float(best.cutoff_hz))
            L.append("")
            L.append("  peak at %.1f Hz: r = %+.3f, 95%% CI [%+.3f, %+.3f] over %d segments"
                     % (best.cutoff_hz, rb, lo_b, hi_b, nseg_b))
            L.append("  shared variance at the peak: %.1f%%" % (100 * rb ** 2))
            L.append("  SD ratio at the peak: OpenIris / ours = %.0fx"
                     % (best.sd_oi / best.sd_ours if best.sd_ours else np.nan))
            L.append("")

        # arm 4: the pupil channel as an internal control on frame pairing
        L.append("  PUPIL AS AN INTERNAL CONTROL")
        L.append("  If the two files were misaligned, no channel would agree. Pupil")
        L.append("  position is absolute and carries across segment boundaries, so it")
        L.append("  is pooled over the whole recording; torsion is reset at every")
        L.append("  boundary and must be centred within segment.")
        L.append("")
        pu = averaging_sweep(b, "pupil_y", "oi_pupil_y", within_segment=False)
        to = averaging_sweep(t, args.torsion_col, "oi_torsion", within_segment=True)
        merged = pu.merge(to, on="window_frames", how="outer",
                          suffixes=("_pupil", "_torsion")).sort_values("window_frames")
        L.append("  %-16s %12s %12s" % ("window", "pupil r", "torsion r"))
        L.append("  " + "-" * 42)
        for _, r_ in merged.iterrows():
            wp = "%d fr (%.1f s)" % (r_.window_frames, r_.window_frames / 50.0)
            fp = ("%+.3f" % r_.r_pupil) if pd.notna(r_.get("r_pupil")) else "   --"
            ft = ("%+.3f" % r_.r_torsion) if pd.notna(r_.get("r_torsion")) else "   --"
            L.append("  %-16s %12s %12s" % (wp, fp, ft))
        L.append("")
        L.append("  Pupil agreement rising with averaging while torsion does not is")
        L.append("  evidence that the frames are correctly paired and that the")
        L.append("  failure is specific to the torsion channel.")
        L.append("")

        L.append("  INTERPRETATION")
        if abs(r) > 0.5:
            L.append("    The two methods agree substantially. Since they share")
            L.append("    no code and no algorithm, this is evidence that both")
            L.append("    are tracking real ocular torsion.")
        elif abs(r) > 0.2:
            L.append("    Modest agreement. Both may be partly tracking torsion")
            L.append("    with substantial independent noise.")
        else:
            L.append("    No meaningful agreement on torsion. Three things above")
            L.append("    bear on how that should be read:")
            L.append("      - each trace's own lag-1 autocorrelation, which says")
            L.append("        whether either is self-consistent at all;")
            L.append("      - the filter sweep, which says whether agreement")
            L.append("        appears once the noise band is removed;")
            L.append("      - the pupil control, which says whether the frames")
            L.append("        are paired correctly in the first place.")
            L.append("    Agreement on the pupil channel alongside disagreement on")
            L.append("    torsion locates the failure in the torsion measurement")
            L.append("    rather than in the alignment or the configuration.")
        L.append("")

    txt = "\n".join(L)
    print(txt)
    open(os.path.join(args.out, "openiris_comparison.txt"), "w").write(txt + "\n")
    b.to_csv(os.path.join(args.out, "openiris_merged.csv"), index=False)

    # ---------------- figures ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("(figures skipped:", e, ")")
        return

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    a = ax[0, 0]
    ok = np.isfinite(b["pupil_x"]) & np.isfinite(b["oi_pupil_x"])
    a.scatter(b.loc[ok, "pupil_x"], b.loc[ok, "oi_pupil_x"], s=2, alpha=.2)
    lim = [min(b.loc[ok, "pupil_x"].min(), b.loc[ok, "oi_pupil_x"].min()),
           max(b.loc[ok, "pupil_x"].max(), b.loc[ok, "oi_pupil_x"].max())]
    a.plot(lim, lim, "k--", lw=1)
    a.set_xlabel("ours: pupil x (px)")
    a.set_ylabel("OpenIris: pupil x (px)")
    a.set_title("Pupil position agreement")

    a = ax[0, 1]
    if len(t) > 100:
        d_ = t["ours_c"] - t["oi_c"]
        mn = (t["ours_c"] + t["oi_c"]) / 2
        okb = np.isfinite(d_) & np.isfinite(mn)
        a.scatter(mn[okb], d_[okb], s=2, alpha=.2)
        bias, sd, lo, hi = bland_altman(t["ours_c"], t["oi_c"])
        for yv, st, lab in ((bias, "-", "bias %.2f" % bias),
                            (lo, "--", "LoA"), (hi, "--", None)):
            a.axhline(yv, color="#d62728", ls=st, lw=1, label=lab)
        a.set_xlabel("mean of the two (deg)")
        a.set_ylabel("difference, ours - OpenIris (deg)")
        a.set_title("Bland-Altman, torsion")
        a.legend(fontsize=8)

    a = ax[1, 0]
    if len(t) > 100:
        big = t.groupby("segment").size().idxmax()
        S = t[t["segment"] == big]
        a.plot(S["time"], S["ours_c"], lw=.8, label="ours")
        a.plot(S["time"], S["oi_c"], lw=.8, alpha=.8, label="OpenIris")
        a.set_xlabel("time (s)")
        a.set_ylabel("torsion, segment-centred (deg)")
        a.set_title("Longest common segment (#%d)" % big)
        a.legend(fontsize=8)

    a = ax[1, 1]
    if len(t) > 100:
        lp = lag_profile(t["ours_c"], t["oi_c"], args.max_lag)
        a.plot([l for l, _ in lp], [c for _, c in lp], "o-", ms=3)
        a.axvline(0, color="k", lw=.6)
        a.axhline(0, color="k", lw=.6)
        a.set_xlabel("frame lag")
        a.set_ylabel("correlation")
        a.set_title("Cross-correlation vs lag (peak should be at 0)")

    fig.tight_layout()
    p = os.path.join(args.out, "openiris_comparison.png")
    fig.savefig(p, dpi=150)
    print("Wrote:", p)
    print("Wrote:", os.path.join(args.out, "openiris_comparison.txt"))
    print("Wrote:", os.path.join(args.out, "openiris_merged.csv"))


if __name__ == "__main__":
    main()
