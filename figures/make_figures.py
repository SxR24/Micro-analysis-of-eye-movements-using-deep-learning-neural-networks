#!/usr/bin/env python3
"""
make_figures.py
======================================================================
Generates every figure for the project report from the pipeline outputs.
Nothing here is drawn by hand or approximated: each panel reads the actual
CSV / NPZ / OpenIris session files and recomputes its statistics on the fly,
so a figure cannot silently disagree with the data behind it.

Outputs 300 dpi PNG (report) and PDF (vector, for resizing without loss) into
figures/output/, plus figure_stats.txt listing every number that appears in a
panel so the values quoted in the text can be checked against the plots.

Run from the project root:
    python figures/make_figures.py

Sohil Ananth, MSc Bioinformatics & CS, University of Leicester
"""
import os
import glob
import math
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# House style. Kept deliberately plain: greyscale-safe where possible,
# no chart junk, font sizes that survive being printed at column width.
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

C_OURS = "#1f4e79"      # this pipeline
C_OPEN = "#c1543a"      # OpenIris
C_BASE = "#8c8c8c"      # superseded baseline
C_ACC = "#4a7c59"       # accent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "output")
os.makedirs(OUT, exist_ok=True)
STATS = []


def log(section, **kw):
    """Record every number a figure asserts, so text and plots can be checked."""
    STATS.append("[%s]" % section)
    for k, v in kw.items():
        STATS.append("    %-38s %s" % (k, v))
    STATS.append("")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("  wrote figures/output/%s.png (+pdf)" % name)


# ----------------------------------------------------------------------
def load_combined():
    d = pd.read_csv(os.path.join(ROOT, "data", "video_8", "combined_8.csv"))
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["segment"] = d["seg"] if "seg" in d else (d["blink"] == 1).cumsum()
    d["ok"] = ((d["blink"].fillna(1) != 1) & (d["pupil_found"].fillna(0) == 1)
               & d["torsion_deg"].notna() & (d["seg"].fillna(-1) >= 0))
    return d


def procrustes(ref, cur):
    """Least-squares rigid rotation ref -> cur (deg) and median residual (px)."""
    if len(ref) < 3:
        return np.nan, np.nan
    a = ref - ref.mean(0)
    b = cur - cur.mean(0)
    S = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())
    C = float((a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum())
    if S == 0 and C == 0:
        return np.nan, np.nan
    th = math.atan2(S, C)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return math.degrees(th), float(np.median(np.linalg.norm(b - a @ R.T, axis=1)))


def splithalf(npz_path, min_seg=25, seed=0):
    """Split-half reliability + residual profile across a segment."""
    z = np.load(npz_path)
    XY, OK, SEG = z["feat_xy"], z["feat_ok"], z["seg_id"]
    rng = np.random.default_rng(seed)
    half = rng.random(XY.shape[1]) < 0.5
    A, B, R, K, S = [], [], [], [], []
    for s in np.unique(SEG[SEG >= 0]):
        idx = np.flatnonzero(SEG == s)
        if len(idx) < min_seg:
            continue
        ref, refok = XY[idx[0]], OK[idx[0]]
        for k, f in enumerate(idx):
            sel = OK[f] & refok
            if sel.sum() < 20:
                continue
            _, res = procrustes(ref[sel], XY[f][sel])
            ha, hb = sel & half, sel & ~half
            A.append(procrustes(ref[ha], XY[f][ha])[0] if ha.sum() >= 10 else np.nan)
            B.append(procrustes(ref[hb], XY[f][hb])[0] if hb.sum() >= 10 else np.nan)
            R.append(res); S.append(s); K.append(k / max(1, len(idx) - 1))
    A, B, R, K, S = map(np.asarray, (A, B, R, K, S))
    # centre within segment: torsion is re-referenced at each segment start
    Ac, Bc = np.full(len(A), np.nan), np.full(len(B), np.nan)
    for s in np.unique(S):
        m = S == s
        Ac[m] = A[m] - np.nanmean(A[m])
        Bc[m] = B[m] - np.nanmean(B[m])
    m = np.isfinite(Ac) & np.isfinite(Bc)
    r = float(np.corrcoef(Ac[m], Bc[m])[0, 1])
    prof = [float(np.nanmedian(R[(K >= lo) & (K < hi)]))
            for lo, hi in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01))]
    return dict(r=r, rel=2 * r / (1 + r), Ac=Ac[m], Bc=Bc[m],
                resid=float(np.nanmedian(R)), prof=prof, n=int(m.sum()),
                nseg=int(len(np.unique(S))))


def load_openiris(path, eye="Left"):
    d = pd.read_csv(path, sep=r"\s+")
    o = pd.DataFrame({
        "frame": pd.to_numeric(d[eye + "FrameNumber"], errors="coerce"),
        "px": pd.to_numeric(d[eye + "PupilX"], errors="coerce"),
        "py": pd.to_numeric(d[eye + "PupilY"], errors="coerce"),
        "pw": pd.to_numeric(d[eye + "PupilWidth"], errors="coerce"),
        "tor": pd.to_numeric(d[eye + "Torsion"], errors="coerce"),
    })
    # OpenIris writes a frame-centre fallback, not NaN, when no pupil is found
    o["oi_ok"] = (o.px.between(200, 750) & o.py.between(150, 600)
                  & o.pw.between(120, 320))
    return o


# ======================================================================
# Figure 1 - split-half reliability, before and after reference anchoring
# ======================================================================
def fig1():
    new = splithalf(os.path.join(ROOT, "data", "video_8", "features_8.npz"))
    old = splithalf(os.path.join(ROOT, "data", "video_8", "baseline_lkchain",
                                 "features_8_LKCHAIN.npz"))

    fig, ax = plt.subplots(1, 3, figsize=(9.2, 3.1))

    for a, res, ttl, col in ((ax[0], old, "Frame-to-frame chaining", C_BASE),
                             (ax[1], new, "Reference-anchored", C_OURS)):
        s = slice(None, None, 6)          # thin for legibility, not for effect
        a.scatter(res["Ac"][s], res["Bc"][s], s=2, alpha=.25, color=col,
                  edgecolors="none", rasterized=True)
        lim = 1.4
        a.plot([-lim, lim], [-lim, lim], "k--", lw=.7)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_aspect("equal")
        a.set_xlabel("torsion, feature half A (deg)")
        a.set_ylabel("torsion, half B (deg)")
        a.set_title("%s\nr = %.3f, reliability = %.3f"
                    % (ttl, res["r"], res["rel"]))

    a = ax[2]
    x = np.arange(5)
    a.plot(x, old["prof"], "o-", color=C_BASE, label="chained", ms=4)
    a.plot(x, new["prof"], "s-", color=C_OURS, label="anchored", ms=4)
    a.set_xticks(x)
    a.set_xticklabels(["0-20", "20-40", "40-60", "60-80", "80-100"])
    a.set_xlabel("position within segment (%)")
    a.set_ylabel("rigid-fit residual (px)")
    a.set_title("Residual growth across a segment")
    a.legend(frameon=False)
    a.set_ylim(0, None)

    fig.tight_layout()
    save(fig, "fig1_reliability")
    log("Figure 1 - reliability",
        chained_r=round(old["r"], 3), chained_reliability=round(old["rel"], 3),
        anchored_r=round(new["r"], 3), anchored_reliability=round(new["rel"], 3),
        chained_residual_profile_px=[round(v, 2) for v in old["prof"]],
        anchored_residual_profile_px=[round(v, 2) for v in new["prof"]],
        n_frames_anchored=new["n"], n_segments_anchored=new["nseg"],
        n_frames_chained=old["n"], n_segments_chained=old["nseg"])
    return new, old


# ======================================================================
# Figure 2 - pupil trajectory, this pipeline vs OpenIris
# ======================================================================
def fig2(oi_path):
    d = load_combined()
    o = load_openiris(oi_path)
    m = d.merge(o, on="frame", how="inner")
    both = m["ok"] & m["oi_ok"]

    # a 20 s window with continuous tracking from both methods
    fps = 50.0
    win = 20 * int(fps)
    best, bestscore = 0, -1
    v = both.values.astype(int)
    for st in range(0, len(v) - win, 500):
        sc = v[st:st + win].sum()
        if sc > bestscore:
            bestscore, best = sc, st
    seg = m.iloc[best:best + win]

    fig, ax = plt.subplots(2, 2, figsize=(9.2, 5.2))

    a = ax[0, 0]
    a.plot(seg["time"], seg["pupil_x"], lw=.9, color=C_OURS, label="this pipeline")
    a.plot(seg["time"], seg["px"], lw=.7, color=C_OPEN, alpha=.85, label="OpenIris")
    a.set_xlabel("time (s)"); a.set_ylabel("pupil x (px)")
    a.set_title("Horizontal pupil position, 20 s window")
    a.legend(frameon=False, loc="upper right")

    a = ax[0, 1]
    a.plot(seg["time"], seg["pupil_y"], lw=.9, color=C_OURS)
    a.plot(seg["time"], seg["py"], lw=.7, color=C_OPEN, alpha=.85)
    a.set_xlabel("time (s)"); a.set_ylabel("pupil y (px)")
    a.set_title("Vertical pupil position, same window")

    # frame-to-frame differences: the stability measure
    a = ax[1, 0]
    dd_ours = d.loc[d["ok"], "pupil_x"].diff().dropna()
    dd_open = o.loc[o["oi_ok"], "px"].diff().dropna()
    bins = np.linspace(-40, 40, 81)
    a.hist(dd_open, bins=bins, color=C_OPEN, alpha=.65, label="OpenIris", density=True)
    a.hist(dd_ours, bins=bins, color=C_OURS, alpha=.75, label="this pipeline", density=True)
    a.set_xlabel("frame-to-frame change in pupil x (px)")
    a.set_ylabel("density")
    a.set_title("Frame-to-frame variation")
    a.legend(frameon=False)

    a = ax[1, 1]
    lags = np.arange(1, 16)
    def ac(s, L):
        return [float(pd.Series(s).autocorr(l)) for l in L]
    a.plot(lags, ac(d.loc[d["ok"], "pupil_x"].values, lags), "s-",
           color=C_OURS, ms=3, label="this pipeline")
    a.plot(lags, ac(o.loc[o["oi_ok"], "px"].values, lags), "o-",
           color=C_OPEN, ms=3, label="OpenIris")
    a.axhline(0, color="k", lw=.6)
    a.set_ylim(-0.1, 1.05)
    a.set_xlabel("lag (frames)"); a.set_ylabel("autocorrelation")
    a.set_title("Temporal structure of the pupil signal")
    a.legend(frameon=False)

    fig.tight_layout()
    save(fig, "fig2_pupil_comparison")
    log("Figure 2 - pupil comparison",
        window_start_s=round(float(seg["time"].iloc[0]), 1),
        ours_pupilx_sd_px=round(float(d.loc[d["ok"], "pupil_x"].std()), 2),
        ours_pupilx_frame_to_frame_sd_px=round(float(dd_ours.std()), 2),
        ours_pupilx_lag1=round(float(d.loc[d["ok"], "pupil_x"].autocorr(1)), 3),
        openiris_pupilx_sd_px=round(float(o.loc[o["oi_ok"], "px"].std()), 2),
        openiris_pupilx_frame_to_frame_sd_px=round(float(dd_open.std()), 2),
        openiris_pupilx_lag1=round(float(o.loc[o["oi_ok"], "px"].autocorr(1)), 3),
        n_ours=int(d["ok"].sum()), n_openiris=int(o["oi_ok"].sum()))


# ======================================================================
# Figure 3 - torsion traces and their temporal structure
# ======================================================================
def fig3(oi_path):
    d = load_combined()
    o = load_openiris(oi_path)
    m = d.merge(o, on="frame", how="inner")
    use = m["ok"] & m["oi_ok"]
    t = m[use].copy()
    sizes = t.groupby("segment")["frame"].transform("size")
    t = t[sizes >= 25]
    t["ours_c"] = t["torsion_deg"] - t.groupby("segment")["torsion_deg"].transform("mean")
    t["oi_c"] = t["tor"] - t.groupby("segment")["tor"].transform("mean")

    big = t.groupby("segment").size().idxmax()
    S = t[t["segment"] == big]

    fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.9))

    a = ax[0]
    a.plot(S["time"], S["ours_c"], lw=.8, color=C_OURS)
    a.set_xlabel("time (s)"); a.set_ylabel("torsion (deg)")
    a.set_title("This pipeline\nsegment %d, %d frames" % (big, len(S)))

    a = ax[1]
    a.plot(S["time"], S["oi_c"], lw=.8, color=C_OPEN)
    a.set_xlabel("time (s)"); a.set_ylabel("torsion (deg)")
    a.set_title("OpenIris, same frames")

    a = ax[2]
    lags = np.arange(1, 16)
    a.plot(lags, [t["ours_c"].autocorr(l) for l in lags], "s-", color=C_OURS,
           ms=3, label="this pipeline")
    a.plot(lags, [t["oi_c"].autocorr(l) for l in lags], "o-", color=C_OPEN,
           ms=3, label="OpenIris")
    a.axhline(0, color="k", lw=.6)
    a.set_ylim(-0.1, 1.05)
    a.set_xlabel("lag (frames)"); a.set_ylabel("autocorrelation")
    a.set_title("Torsion temporal structure")
    a.legend(frameon=False)

    fig.tight_layout()
    save(fig, "fig3_torsion_comparison")
    log("Figure 3 - torsion comparison",
        segment_shown=int(big), frames_shown=len(S),
        ours_torsion_sd_deg=round(float(t["ours_c"].std()), 3),
        openiris_torsion_sd_deg=round(float(t["oi_c"].std()), 3),
        ours_lag1=round(float(t["ours_c"].autocorr(1)), 3),
        openiris_lag1=round(float(t["oi_c"].autocorr(1)), 3),
        within_segment_correlation=round(float(t["ours_c"].corr(t["oi_c"])), 3),
        n_frames=len(t), n_segments=int(t["segment"].nunique()))


# ======================================================================
# Figure 4 - why cross-correlation fails: iris occlusion by the eyelid
# ======================================================================
def fig4():
    import cv2
    meta = json.load(open(os.path.join(ROOT, "data", "video_8", "frames",
                                       "_frames_meta.json")))
    pad_x, pad_y = int(round(meta["pad_x"])), int(round(meta["pad_y"]))
    W, H = meta["original_width"], meta["original_height"]
    d = load_combined()

    frames = list(range(1000, 27000, 2000))
    overall, sector_profiles, used = [], [], []
    nbins = 36
    for f in frames:
        p = os.path.join(ROOT, "data", "video_8", "masks", "frame_%06d.png" % f)
        m = cv2.imread(p, cv2.IMREAD_COLOR)
        row = d[d["frame"] == f]
        if m is None or row.empty or pd.isna(row["pupil_x"].iloc[0]):
            continue
        b, g, r = m[:, :, 0], m[:, :, 1], m[:, :, 2]
        disc = ((b > 128) & (r < 100) & (g < 100)) | ((r > 128) & (g < 100) & (b < 100))
        h, w = disc.shape
        core = (disc[pad_y:h - pad_y, pad_x:w - pad_x].astype(np.uint8) * 255)
        big = cv2.resize(core, (W, H), interpolation=cv2.INTER_NEAREST) > 0
        cx, cy = float(row["pupil_x"].iloc[0]), float(row["pupil_y"].iloc[0])
        Y, X = np.ogrid[:H, :W]
        rr = np.hypot(X - cx, Y - cy)
        ann = (rr >= 121) & (rr <= 191)
        overall.append(100 * big[ann].mean())
        ang = (np.degrees(np.arctan2(Y - cy, X - cx)) + 360) % 360
        prof = []
        for i in range(nbins):
            sel = ann & (ang >= i * 360 / nbins) & (ang < (i + 1) * 360 / nbins)
            prof.append(100 * big[sel].mean() if sel.sum() else np.nan)
        sector_profiles.append(prof)
        used.append(f)

    P = np.array(sector_profiles)

    fig = plt.figure(figsize=(9.2, 3.1))
    a = fig.add_subplot(1, 3, 1)
    a.plot(np.array(used) / 50.0, overall, "o-", color=C_ACC, ms=4)
    a.set_xlabel("time (s)"); a.set_ylabel("iris within annulus (%)")
    a.set_title("Unoccluded iris in the\nsampled annulus")
    a.set_ylim(0, 100)

    a = fig.add_subplot(1, 3, 2, projection="polar")
    th = np.deg2rad(np.arange(nbins) * 360 / nbins + 180 / nbins)
    mean_prof = np.nanmean(P, axis=0)
    a.plot(np.r_[th, th[:1]], np.r_[mean_prof, mean_prof[:1]], color=C_ACC, lw=1.5)
    a.set_theta_zero_location("E")
    a.set_theta_direction(-1)     # image y runs downward
    a.set_rlim(0, 100)
    a.set_title("Iris visibility by angle\n(0 deg = temporal, 270 = up)", pad=14)

    a = fig.add_subplot(1, 3, 3)
    upper = np.nanmean(P[:, (np.arange(nbins) * 10 >= 200) & (np.arange(nbins) * 10 <= 340)], axis=1)
    lower = np.nanmean(P[:, (np.arange(nbins) * 10 >= 20) & (np.arange(nbins) * 10 <= 160)], axis=1)
    a.bar([0, 1, 2], [np.mean(overall), np.nanmean(upper), np.nanmean(lower)],
          color=[C_ACC, C_OPEN, C_OURS], width=.6)
    a.set_xticks([0, 1, 2])
    a.set_xticklabels(["whole\nannulus", "upper\nsector", "lower\nsector"])
    a.set_ylabel("iris (%)"); a.set_ylim(0, 100)
    a.set_title("Mean visibility by sector")

    fig.tight_layout()
    save(fig, "fig4_annulus_occlusion")
    log("Figure 4 - annulus occlusion",
        frames_sampled=used,
        annulus_radii_px="121-191",
        mean_iris_whole_annulus_pct=round(float(np.mean(overall)), 1),
        mean_iris_upper_sector_pct=round(float(np.nanmean(upper)), 1),
        mean_iris_lower_sector_pct=round(float(np.nanmean(lower)), 1),
        first_frame_pct=round(overall[0], 1), last_frame_pct=round(overall[-1], 1))


# ======================================================================
# Figure 5 - gaze coverage (the control analysis)
# ======================================================================
def fig5():
    d = load_combined()
    g = d[d["ok"]].copy()
    iris_px = float(g["iris_diam"].median())
    R = 12.0 * (iris_px / 11.7)          # eyeball radius in px
    x0, y0 = g["pupil_x"].median(), g["pupil_y"].median()
    g["th"] = np.degrees(np.arcsin(np.clip((g["pupil_x"] - x0) / R, -1, 1)))
    g["tv"] = np.degrees(np.arcsin(np.clip((g["pupil_y"] - y0) / R, -1, 1)))

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
    a = ax[0]
    hb = a.hexbin(g["th"], g["tv"], gridsize=45, cmap="Blues", bins="log", mincnt=1)
    a.axhline(0, lw=.6, color="k"); a.axvline(0, lw=.6, color="k")
    a.set_xlabel("horizontal gaze (deg)"); a.set_ylabel("vertical gaze (deg)")
    a.set_title("Gaze coverage")
    fig.colorbar(hb, ax=a, label="log$_{10}$ frames")

    a = ax[1]
    tc = g["torsion_deg"] - g.groupby("segment")["torsion_deg"].transform("mean")
    prod = g["th"] * g["tv"]
    pc = prod - prod.groupby(g["segment"]).transform("mean")
    q = pd.qcut(pc, 18, duplicates="drop")
    grp = pd.DataFrame({"p": pc, "t": tc}).groupby(q, observed=True)
    a.errorbar(grp["p"].mean(), grp["t"].mean(), yerr=grp["t"].sem(),
               fmt="o", ms=3.5, lw=1, color=C_OURS)
    xs = np.linspace(pc.min(), pc.max(), 50)
    beta = float((tc * pc).sum() / (pc ** 2).sum())
    a.plot(xs, beta * xs, "-", color=C_OPEN, lw=1.2,
           label="fitted %.4f" % beta)
    a.plot(xs, (-1 / (2 * 180 / np.pi)) * xs, "--", color=C_ACC, lw=1.2,
           label="Listing %.4f" % (-1 / (2 * 180 / np.pi)))
    a.axhline(0, lw=.6, color="k")
    a.set_xlabel(r"$\theta_h\cdot\theta_v$ (deg$^2$, segment-centred)")
    a.set_ylabel("torsion (deg, segment-centred)")
    a.set_title("Torsion vs gaze product")
    a.legend(frameon=False)

    fig.tight_layout()
    save(fig, "fig5_gaze_control")
    log("Figure 5 - gaze control",
        horizontal_gaze_sd_deg=round(float(g["th"].std()), 2),
        vertical_gaze_sd_deg=round(float(g["tv"].std()), 2),
        corr_h_v=round(float(np.corrcoef(g["th"], g["tv"])[0, 1]), 3),
        product_sd_deg2=round(float(prod.std()), 2),
        fitted_slope_deg_per_deg2=round(beta, 5),
        listing_prediction=round(-1 / (2 * 180 / np.pi), 5),
        iris_diameter_px=round(iris_px, 1), eyeball_radius_px=round(R, 0),
        n_frames=len(g))


# ======================================================================
def main():
    print("Generating figures from pipeline outputs...")
    oi = sorted(glob.glob(os.path.join(ROOT, "data", "openiris", "*", "*.txt")))
    oi = [p for p in oi if "-log" not in p]
    if not oi:
        raise SystemExit("No OpenIris session .txt under data/openiris/")
    oi_path = max(oi, key=os.path.getmtime)
    print("  OpenIris session: %s" % os.path.basename(oi_path))

    fig1()
    fig2(oi_path)
    fig3(oi_path)
    try:
        fig4()
    except Exception as e:
        print("  fig4 skipped (%s)" % e)
    fig5()

    p = os.path.join(OUT, "figure_stats.txt")
    with open(p, "w") as f:
        f.write("Values plotted in each figure, recomputed at generation time.\n")
        f.write("OpenIris session: %s\n\n" % os.path.basename(oi_path))
        f.write("\n".join(STATS))
    print("\nWrote %s" % p)


if __name__ == "__main__":
    main()
