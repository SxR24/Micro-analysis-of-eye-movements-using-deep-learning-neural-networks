#!/usr/bin/env python3
"""
okn.py
======================================================================
Torsional optokinetic nystagmus: does measured torsion follow the stimulus?

THE PARADIGM
------------
A pattern rotating about the line of sight drives a reflexive torsional
response with two alternating phases:

    slow phase   the eye rolls with the stimulus
    quick phase  a fast flick back to re-centre

Repeated, this traces a sawtooth. The quantity of interest is slow-phase
velocity (SPV), and its ratio to the stimulus velocity is the GAIN. Torsional
OKN gain is low -- typically 0.1 to 0.2, against 0.8 to 0.9 for horizontal --
so the expected response is small and the measurement noise floor matters.

WHY THIS AND NOT LISTING'S LAW
------------------------------
Listing's Law describes how torsion varies with GAZE DIRECTION, and testing it
needs gaze excursion across a 2D range. An OKN paradigm holds gaze roughly
fixed and rotates the stimulus instead, so there is nothing for that test to
regress against. analyse.py is retained as a negative control: torsion should
show NO dependence on gaze direction here, and confirming that is evidence the
estimate is free of gaze-related translation artefact.

METHOD
------
Torsion is re-referenced to zero at every segment boundary, so absolute values
cannot be pooled across segments. Slow-phase VELOCITY can be, because it is a
derivative and is unaffected by the reference reset. Everything below is
therefore computed within segments and pooled at the velocity level.

  1. differentiate torsion within each segment (Savitzky-Golay, so the
     derivative is smooth without lagging the signal)
  2. mark quick phases where |velocity| exceeds a robust threshold
  3. take the runs between them as slow phases
  4. fit a robust slope to each slow phase -> one SPV per slow phase
  5. pool: is the SPV distribution shifted away from zero, and in one
     direction? Bootstrap over slow phases, since samples within one are
     heavily autocorrelated.

Passing --stimulus-deg-s converts SPV into gain.

Usage:
    python src/analysis/okn.py --csv data/video_8/combined_8.csv \
                               --out data/video_8/analysis
    python src/analysis/okn.py --csv ... --stimulus-deg-s 20 --sign -1

Sohil Ananth, MSc Bioinformatics & CS, University of Leicester
"""
import os
import time
import hashlib
import argparse

import numpy as np
import pandas as pd


def provenance(path):
    """Hash and mtime of the input, stamped into the report."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    st = os.stat(path)
    return dict(path=os.path.abspath(path), sha256=h.hexdigest()[:16],
                mtime=time.strftime("%Y-%m-%d %H:%M:%S",
                                    time.localtime(st.st_mtime)),
                generated=time.strftime("%Y-%m-%d %H:%M:%S"))


# ----------------------------------------------------------------------
def load(csv_path, torsion_col, max_resid=None):
    d = pd.read_csv(csv_path)
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if torsion_col not in d:
        raise SystemExit("No column %r in %s" % (torsion_col, csv_path))

    if "seg" in d.columns and d["seg"].notna().any():
        d["segment"] = d["seg"]
    else:
        # Older CSVs predate the explicit seg column. A cumsum over blink is a
        # poor substitute -- it merges the spans either side of a re-seed --
        # but it is better than refusing to run.
        d["segment"] = (d["blink"].fillna(0) == 1).cumsum()
        print("  WARNING: no `seg` column; inferring segments from blinks.")

    ok = (d["blink"].fillna(0) != 1) & d[torsion_col].notna()
    if "seg" in d.columns:
        ok &= (d["seg"].fillna(-1) >= 0)
    if max_resid is not None and "torsion_resid_px" in d.columns:
        # Frames where the tracked set is not behaving like a rigid body give a
        # meaningless rotation however smooth the trace looks.
        ok &= (d["torsion_resid_px"].fillna(np.inf) <= max_resid)
    return d[ok].copy(), len(d)


def savgol(v, win, poly=2, deriv=0, dt=1.0):
    """Savitzky-Golay filter, implemented directly to avoid a scipy dependency
    for one call. Returns NaN where the window does not fit."""
    win = int(win) | 1                       # force odd
    if len(v) < win or win < poly + 2:
        return np.full(len(v), np.nan)
    half = win // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, poly + 1, increasing=True)
    # row of the pseudo-inverse giving the requested derivative coefficient
    coef = np.linalg.pinv(A)[deriv]
    from math import factorial
    coef = coef * factorial(deriv) / (dt ** deriv)
    out = np.full(len(v), np.nan)
    out[half:len(v) - half] = np.convolve(v, coef[::-1], mode="valid")
    return out


def slow_phases(seg_df, torsion_col, fps, smooth_ms, qp_thresh,
                min_slow_ms, edge_ms):
    """Split one segment into slow phases and return a row per slow phase."""
    v = seg_df[torsion_col].values.astype(float)
    n = len(v)
    win = max(5, int(round(smooth_ms * 1e-3 * fps)) | 1)
    if n < win + 4:
        return []

    vel = savgol(v, win, poly=2, deriv=1, dt=1.0 / fps)      # deg/s
    finite = np.isfinite(vel)
    if finite.sum() < 10:
        return []

    quick = np.zeros(n, bool)
    quick[finite] = np.abs(vel[finite]) > qp_thresh

    # Widen around each quick phase: the smoothed derivative is contaminated
    # for roughly half a window either side of a fast transient.
    pad = max(1, int(round(edge_ms * 1e-3 * fps)))
    if quick.any():
        idx = np.flatnonzero(quick)
        for i in idx:
            quick[max(0, i - pad):min(n, i + pad + 1)] = True

    usable = finite & ~quick
    rows = []
    min_len = max(5, int(round(min_slow_ms * 1e-3 * fps)))

    # contiguous runs of usable samples
    edges = np.flatnonzero(np.diff(np.r_[0, usable.view(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < min_len:
            continue
        t = seg_df["time"].values[a:b]
        y = v[a:b]
        m = np.isfinite(y) & np.isfinite(t)
        if m.sum() < min_len:
            continue
        # robust slope: Theil-Sen on a subsample is overkill here; least
        # squares on an already quick-phase-free run is adequate and fast
        slope = float(np.polyfit(t[m], y[m], 1)[0])
        rows.append(dict(seg=int(seg_df["segment"].iloc[0]),
                         t_start=float(t[m][0]), dur=float(t[m][-1] - t[m][0]),
                         n=int(m.sum()), spv=slope,
                         excursion=float(y[m][-1] - y[m][0])))
    return rows


def bootstrap_mean(x, w=None, n_boot=2000, seed=0):
    """CI for the (optionally duration-weighted) mean, resampling slow phases.

    Slow phases are the independent unit: samples within one are a fitted line
    and carry no extra information about whether a drive exists.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    out = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        out[b] = np.average(x[i], weights=w[i])
    return np.percentile(out, [2.5, 97.5]), out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Torsional OKN: slow-phase velocity and gain")
    ap.add_argument("--csv", default="data/video_8/combined_8.csv")
    ap.add_argument("--out", default="data/video_8/analysis")
    ap.add_argument("--torsion-col", default="torsion_deg",
                    help="all features pooled is the better estimator; the "
                         "inner/outer split is a diagnostic, not a choice")
    ap.add_argument("--fps", type=float, default=None,
                    help="default: inferred from the time column")
    ap.add_argument("--stimulus-deg-s", type=float, default=None,
                    help="stimulus rotation velocity. Given this, SPV is "
                         "reported as a gain.")
    ap.add_argument("--sign", type=int, default=1, choices=(1, -1),
                    help="+1 if positive torsion in image axes follows a "
                         "positive stimulus rotation, -1 if the camera views "
                         "the eye via a mirror or is inverted")
    ap.add_argument("--smooth-ms", type=float, default=60.0,
                    help="Savitzky-Golay window for the velocity estimate")
    ap.add_argument("--qp-deg-s", type=float, default=None,
                    help="quick-phase velocity threshold. Default: robust, "
                         "median + 4 MAD of |velocity| over the recording.")
    ap.add_argument("--min-slow-ms", type=float, default=200.0)
    ap.add_argument("--edge-ms", type=float, default=40.0,
                    help="samples discarded either side of a quick phase")
    ap.add_argument("--max-resid", type=float, default=None,
                    help="drop frames whose torsion_resid_px exceeds this")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        import glob as _glob
        found = sorted(_glob.glob(os.path.join("data", "video_*",
                                               "combined_*.csv")))
        msg = ["Not found: %s" % args.csv]
        if found:
            msg.append("")
            msg.append("Processed videos available:")
            msg += ["    --csv " + p for p in found]
        else:
            msg.append("")
            msg.append("No combined_*.csv anywhere under data/. Run the "
                       "pipeline first (see README).")
        raise SystemExit("\n".join(msg))

    os.makedirs(args.out, exist_ok=True)
    prov = provenance(args.csv)
    d, n_all = load(args.csv, args.torsion_col, args.max_resid)

    fps = args.fps
    if fps is None:
        dt = np.median(np.diff(d["time"].values))
        fps = 1.0 / dt if dt > 0 else 50.0

    # ---- robust quick-phase threshold, if not given ----
    qp = args.qp_deg_s
    if qp is None:
        allv = []
        for _, s in d.groupby("segment"):
            if len(s) < 12:
                continue
            w = max(5, int(round(args.smooth_ms * 1e-3 * fps)) | 1)
            vv = savgol(s[args.torsion_col].values.astype(float), w,
                        deriv=1, dt=1.0 / fps)
            allv.append(vv[np.isfinite(vv)])
        allv = np.concatenate(allv) if allv else np.array([0.0])
        med = np.median(np.abs(allv))
        mad = np.median(np.abs(np.abs(allv) - med)) + 1e-9
        qp = float(med + 4 * 1.4826 * mad)

    # ---- slow phases ----
    rows = []
    for _, s in d.groupby("segment"):
        rows += slow_phases(s, args.torsion_col, fps, args.smooth_ms, qp,
                            args.min_slow_ms, args.edge_ms)
    SP = pd.DataFrame(rows)

    L = []
    L.append("=" * 72)
    L.append("TORSIONAL OKN ANALYSIS -- %s" % os.path.basename(args.csv))
    L.append("=" * 72)
    L.append("")
    L.append("PROVENANCE")
    L.append("  input                     %s" % prov["path"])
    L.append("  input sha256[:16]         %s" % prov["sha256"])
    L.append("  input last modified       %s" % prov["mtime"])
    L.append("  report generated          %s" % prov["generated"])
    L.append("")
    L.append("DATA")
    L.append("  frames in file            %d" % n_all)
    L.append("  tracked & usable          %d" % len(d))
    L.append("  frame rate                %.2f fps" % fps)
    L.append("  torsion column            %s" % args.torsion_col)
    L.append("  quick-phase threshold     %.2f deg/s%s"
             % (qp, "" if args.qp_deg_s else "  (robust, auto)"))
    L.append("")

    if len(SP) < 5:
        L.append("Too few slow phases (%d) to analyse. Either the recording is "
                 "heavily fragmented or the" % len(SP))
        L.append("quick-phase threshold is wrong. Try --qp-deg-s explicitly.")
        txt = "\n".join(L)
        print(txt)
        with open(os.path.join(args.out, "okn_report.txt"), "w") as f:
            f.write(txt + "\n")
        return

    spv = args.sign * SP["spv"].values
    dur = SP["dur"].values

    L.append("SLOW PHASES")
    L.append("  count                     %d" % len(SP))
    L.append("  total duration            %.1f s" % dur.sum())
    L.append("  duration  median          %.2f s  (p90 %.2f)"
             % (np.median(dur), np.percentile(dur, 90)))
    L.append("  beat rate                 %.2f /s"
             % (len(SP) / max(dur.sum(), 1e-9)))
    L.append("")

    mean_spv = float(np.average(spv, weights=dur))
    (lo, hi), _ = bootstrap_mean(spv, dur, args.boot)
    pos = float((spv > 0).mean())

    L.append("SLOW-PHASE VELOCITY  (sign convention: %+d)" % args.sign)
    L.append("  duration-weighted mean    %+.4f deg/s" % mean_spv)
    L.append("  95%% CI (bootstrap)        [%+.4f, %+.4f]" % (lo, hi))
    L.append("  median                    %+.4f deg/s" % float(np.median(spv)))
    L.append("  direction split           %.0f%% positive / %.0f%% negative"
             % (100 * pos, 100 * (1 - pos)))
    L.append("  |SPV| median              %.4f deg/s" % float(np.median(np.abs(spv))))
    L.append("")

    excludes0 = not (lo <= 0 <= hi)
    onesided = abs(pos - 0.5) > 0.15

    L.append("IS THERE A TORSIONAL DRIVE?")
    L.append("  CI excludes zero          %s" % ("YES" if excludes0 else "no"))
    L.append("  direction consistent      %s" % ("YES" if onesided else "no"))
    if excludes0 and onesided:
        L.append("  -> Slow phases drift consistently in one direction. This is")
        L.append("     the signature of a torsional following response.")
    else:
        L.append("  -> No consistent directional drive. Slow-phase velocity is")
        L.append("     centred near zero with a near-balanced direction split,")
        L.append("     which is what a recording WITHOUT a rotating stimulus")
        L.append("     looks like. Check that this video is an OKN condition")
        L.append("     and that the stimulus was running.")
    L.append("")

    if args.stimulus_deg_s:
        gain = mean_spv / args.stimulus_deg_s
        gl, gh = lo / args.stimulus_deg_s, hi / args.stimulus_deg_s
        L.append("GAIN  (stimulus %.2f deg/s)" % args.stimulus_deg_s)
        L.append("  gain                      %+.4f" % gain)
        L.append("  95%% CI                    [%+.4f, %+.4f]" % (gl, gh))
        L.append("  Torsional OKN gain is typically 0.1-0.2; horizontal OKN")
        L.append("  reaches 0.8-0.9. A gain far above 0.3 here suggests the")
        L.append("  stimulus velocity or the sign convention is wrong.")
        L.append("")

    # noise floor context
    if "torsion_resid_px" in d.columns:
        L.append("MEASUREMENT CONTEXT")
        L.append("  rigid-fit residual        median %.2f px"
                 % float(d["torsion_resid_px"].median()))
        L.append("  Run reliability.py for the split-half reliability of the")
        L.append("  torsion estimate; an SPV smaller than the noise floor is")
        L.append("  not interpretable however tight its CI.")
        L.append("")

    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(args.out, "okn_report.txt"), "w") as f:
        f.write(txt + "\n")
    SP.to_csv(os.path.join(args.out, "okn_slow_phases.csv"), index=False)

    # ---- figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("(figures skipped:", e, ")")
        return

    fig, ax = plt.subplots(3, 1, figsize=(11, 9))

    # longest segment, with slow phases marked
    big = d.groupby("segment").size().idxmax()
    S = d[d["segment"] == big]
    ax[0].plot(S["time"], S[args.torsion_col], lw=0.8, color="#1f77b4")
    for _, r in SP[SP["seg"] == big].iterrows():
        ax[0].axvspan(r["t_start"], r["t_start"] + r["dur"],
                      color="#2ca02c", alpha=0.15)
        ax[0].plot([r["t_start"], r["t_start"] + r["dur"]],
                   [np.nan, np.nan])
    ax[0].set_xlabel("time (s)")
    ax[0].set_ylabel("torsion (deg)")
    ax[0].set_title("Longest segment (#%d) -- shaded spans are slow phases. "
                    "OKN would show a sawtooth." % big)

    ax[1].hist(spv, bins=40, color="#7f7f7f")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].axvline(mean_spv, color="#d62728", lw=2,
                  label="weighted mean %+.3f" % mean_spv)
    ax[1].axvspan(lo, hi, color="#d62728", alpha=0.2, label="95% CI")
    ax[1].set_xlabel("slow-phase velocity (deg/s)")
    ax[1].set_ylabel("slow phases")
    ax[1].set_title("SPV distribution -- a real drive shifts this off zero")
    ax[1].legend(fontsize=8)

    ax[2].scatter(SP["dur"], spv, s=12, alpha=0.5, color="#1f77b4")
    ax[2].axhline(0, color="k", lw=1)
    ax[2].axhline(mean_spv, color="#d62728", lw=1.5)
    ax[2].set_xlabel("slow-phase duration (s)")
    ax[2].set_ylabel("slow-phase velocity (deg/s)")
    ax[2].set_title("SPV vs duration -- a drive should not depend on duration")

    fig.tight_layout()
    p = os.path.join(args.out, "okn_analysis.png")
    fig.savefig(p, dpi=150)
    print("Wrote:", p)
    print("Wrote:", os.path.join(args.out, "okn_report.txt"))
    print("Wrote:", os.path.join(args.out, "okn_slow_phases.csv"))


if __name__ == "__main__":
    main()
