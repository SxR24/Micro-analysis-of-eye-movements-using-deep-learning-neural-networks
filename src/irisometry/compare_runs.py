#!/usr/bin/env python3
"""
compare_runs.py
======================================================================
Quantify the effect of the pipeline corrections by comparing a BEFORE and an
AFTER ocular_*.csv over the same frames.

Reports the three diagnostics that have a physical or measurable target, rather
than eyeballing whether the torsion trace "looks smoother":

  1. BLINK RECALL against RITnet.
     RITnet observes the pupil directly; classical irisometry infers blinks
     from feature-count collapse. Treating RITnet's pupil_found == 0 as ground
     truth for "eye closed", what fraction does each run catch?

  2. FEATURE STARVATION.
     detect_features() relaxes its Shi-Tomasi quality threshold in a loop until
     it scrapes past min_features. Frames sitting near that floor indicate the
     AOI is not offering enough real iris texture. Lower is better.

  3. INNER/OUTER TORSION AGREEMENT.
     The inner and outer iris undergo the SAME physical rotation, so
     |torsion_inner| / |torsion_outer| should approach 1.0. Excess is
     measurement noise -- typically inner-ring features sitting on the pupil
     boundary rather than on iris texture. This is the one diagnostic with a
     principled target value.

Usage:
    python src/irisometry/compare_runs.py \
        --before data/video_8/baseline_prefix/ocular_8_BEFORE.csv \
        --after  data/video_8/ocular_8.csv \
        --ritnet data/video_8/ritnet_8.csv \
        --out    data/video_8/comparison
"""
import os
import argparse
import numpy as np
import pandas as pd


def tracked(d):
    return d["blink"] != 1


def segment_stats(d, key="torsion_outer_deg", min_len=25):
    """Noise and drift WITHIN each inter-blink segment.

    Torsion is re-referenced to zero after every blink, so any statistic pooled
    across the whole recording mixes real measurement noise with those reference
    resets -- and runs that flag different numbers of blinks then aren't
    comparable at all. Everything meaningful has to be computed inside a segment.

    within-segment SD : scatter about that segment's own reference
    drift             : |last - first| in a segment. Tracking error accumulates,
                        so drift catches slow slippage that frame-to-frame
                        jitter completely misses. This distinction matters:
                        lowering the corner-quality threshold produces MORE
                        features and LOWER jitter while drift gets dramatically
                        worse, because weak corners slide smoothly rather than
                        jumping.
    """
    d = d.copy()
    d["_seg"] = (d["blink"] == 1).cumsum()
    sds, drifts = [], []
    for _, g in d[tracked(d)].groupby("_seg"):
        v = g[key].dropna()
        if len(v) < min_len:
            continue
        sds.append(v.std())
        drifts.append(abs(v.iloc[-1] - v.iloc[0]))
    if not sds:
        return np.nan, np.nan, 0
    return float(np.median(sds)), float(np.median(drifts)), len(sds)


def summarise(d, rt, label):
    ok = tracked(d)
    nf = d.loc[ok, "n_features"]
    closed = rt["pupil_found"] == 0
    caught = (closed & (d["blink"] == 1)).sum()
    v = d.loc[ok, "torsion_outer_deg"].dropna()
    sd, drift, nseg = segment_stats(d)

    return {
        "label": label,
        "n_frames": len(d),
        "n_blink": int((d["blink"] == 1).sum()),
        "n_tracked": int(ok.sum()),
        "n_segments": nseg,
        "blink_recall": 100.0 * caught / max(int(closed.sum()), 1),
        "nfeat_median": float(nf.median()),
        "jitter": float(v.diff().abs().median()),
        "within_seg_sd": sd,
        "drift_per_seg": drift,
        "torsion_abs_median": float(v.abs().median()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--ritnet", required=True,
                    help="RITnet metrics CSV, used as the blink reference")
    ap.add_argument("--out", default="comparison")
    args = ap.parse_args()

    bef = pd.read_csv(args.before)
    aft = pd.read_csv(args.after)
    rt = pd.read_csv(args.ritnet)

    n = min(len(bef), len(aft), len(rt))
    if not (len(bef) == len(aft) == len(rt)):
        print("NOTE: differing lengths (%d / %d / %d); comparing the first %d frames."
              % (len(bef), len(aft), len(rt), n))
    bef, aft, rt = bef.head(n), aft.head(n), rt.head(n)

    os.makedirs(args.out, exist_ok=True)

    b = summarise(bef, rt, "BEFORE")
    a = summarise(aft, rt, "AFTER")

    rows = [
        ("Frames compared",             "%.0f", b["n_frames"],     a["n_frames"],     ""),
        ("Frames flagged as blink",     "%.0f", b["n_blink"],      a["n_blink"],      ""),
        ("Inter-blink segments",        "%.0f", b["n_segments"],   a["n_segments"],   ""),
        ("Blink recall vs RITnet (%)",  "%.0f", b["blink_recall"], a["blink_recall"], "higher"),
        ("n_features median",           "%.0f", b["nfeat_median"], a["nfeat_median"], "higher"),
        ("|torsion outer| median (deg)", "%.3f", b["torsion_abs_median"], a["torsion_abs_median"], "lower"),
        ("frame-to-frame jitter (deg)", "%.4f", b["jitter"],       a["jitter"],       "lower"),
        ("within-segment SD (deg)",     "%.3f", b["within_seg_sd"], a["within_seg_sd"], "lower"),
        ("drift per segment (deg)",     "%.3f", b["drift_per_seg"], a["drift_per_seg"], "lower *"),
    ]

    line = "-" * 74
    print()
    print(line)
    print("%-32s %12s %12s   %s" % ("metric", "BEFORE", "AFTER", "better"))
    print(line)
    for name, fmt, vb, va, better in rows:
        print("%-32s %12s %12s   %s"
              % (name, fmt % vb, fmt % va, better))
    print(line)
    print("* drift is the metric to trust when tuning. Weak corners slide")
    print("  smoothly, so a change can lower jitter while making drift worse.")
    print(line)

    csv_path = os.path.join(args.out, "comparison_metrics.csv")
    pd.DataFrame([b, a]).to_csv(csv_path, index=False)
    print("Wrote:", csv_path)

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("(figure skipped:", e, ")")
        return

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

    # 1. blink recall
    ax[0].bar(["before", "after"], [b["blink_recall"], a["blink_recall"]],
              color=["#b0b7c3", "#39c5cf"])
    ax[0].axhline(100, ls="--", lw=1, color="#555")
    ax[0].set_ylim(0, 108)
    ax[0].set_ylabel("% of closed-eye frames flagged")
    ax[0].set_title("Blink recall vs RITnet")

    # 2. feature counts
    ax[1].hist(bef.loc[tracked(bef), "n_features"].dropna(), bins=40, alpha=0.6,
               label="before", color="#b0b7c3")
    ax[1].hist(aft.loc[tracked(aft), "n_features"].dropna(), bins=40, alpha=0.6,
               label="after", color="#39c5cf")
    ax[1].axvline(100, ls="--", lw=1, color="#c33",
                  label="min_features floor")
    ax[1].set_xlabel("features tracked per frame")
    ax[1].set_title("Feature availability")
    ax[1].legend(fontsize=8)

    # 3. within-segment noise and drift -- the metrics that survive the
    #    reference reset at every blink
    w = 0.35
    xs = np.arange(2)
    ax[2].bar(xs - w / 2, [b["within_seg_sd"], a["within_seg_sd"]], w,
              label="within-seg SD", color="#b0b7c3")
    ax[2].bar(xs + w / 2, [b["drift_per_seg"], a["drift_per_seg"]], w,
              label="drift/segment", color="#39c5cf")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(["before", "after"])
    ax[2].set_ylabel("degrees")
    ax[2].set_title("Within-segment noise & drift")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    png = os.path.join(args.out, "comparison.png")
    fig.savefig(png, dpi=160)
    print("Wrote:", png)


if __name__ == "__main__":
    main()
