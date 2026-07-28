#!/usr/bin/env python3
"""
live_view.py
======================================================================
Playback viewer for the integrated pipeline -- the irisometry "live tracking"
window from the original RunMe.py, but replaying data the pipeline already
computed instead of tracking on the fly.

Everything drawn comes from files produced by the RITnet -> irisometry handoff:

    combined_<n>.csv   pupil + iris geometry (RITnet), torsion + blink (irisometry)
    features_<n>.npz   the raw per-feature trajectories from ocular.py
    <n>.avi            the ORIGINAL video (all CSV coordinates are in its space)

Because it replays rather than recomputes, it runs at full speed and can be
scrubbed, stepped and paused freely.

Layout: video on the left, a readout sidebar on the right, in one window.

Controls
--------
    SPACE       play / pause
    . or RIGHT  step forward one frame   (works while paused)
    , or LEFT   step back one frame
    L / J       jump forward / back 100 frames
    + / -       playback speed up / down
    F           cycle feature display: valid / all / off
    T           toggle the torsion trace
    S           save a PNG of the current view
    Q or ESC    quit
    (drag the trackbar at the top to seek)

Usage
-----
    python src/review/live_view.py

    # explicit paths
    python src/review/live_view.py --video data/raw/8.avi \
        --csv data/video_8/combined_8.csv \
        --features data/video_8/features_8.npz

    # no window: render an annotated clip instead (for slides / supervisor)
    python src/review/live_view.py --export out.avi --start 1200 --count 500
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2


# ----------------------------------------------------------------------
# Palette (BGR)
# ----------------------------------------------------------------------
C_BG       = (22, 17, 14)
C_PANEL    = (34, 27, 22)
C_LINE     = (56, 44, 34)
C_INK      = (243, 237, 230)
C_MUTED    = (148, 135, 125)
C_IRIS     = (207, 197, 57)     # cyan
C_PUPIL    = (35, 166, 245)     # amber
C_FEAT     = (120, 220, 120)    # green  - tracked & inside AOI
C_FEAT_BAD = (90, 90, 200)      # dim red - lost / outside AOI
C_BLINK    = (84, 84, 245)      # red
C_ACCENT   = (207, 197, 57)

SIDEBAR_W = 400
FONT = cv2.FONT_HERSHEY_SIMPLEX
MONO = cv2.FONT_HERSHEY_DUPLEX


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_csv(path):
    """Minimal CSV -> dict of float arrays (avoids a pandas dependency)."""
    import csv as _csv
    with open(path) as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        sys.exit("Empty CSV: " + path)
    cols = {}
    for k in rows[0]:
        vals = np.empty(len(rows), np.float64)
        for i, r in enumerate(rows):
            v = r.get(k, "")
            try:
                vals[i] = float(v) if v not in ("", None) else np.nan
            except ValueError:
                vals[i] = np.nan
        cols[k] = vals
    return cols


def col(d, name, n):
    return d.get(name, np.full(n, np.nan))


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def draw_overlay(frame_bgr, i, D, feats, show_feats):
    """Draw the tracking overlay onto a copy of the video frame."""
    vis = frame_bgr.copy()

    blink = D["blink"][i] == 1

    # --- iris circle (the AOI the tracker actually used) ---
    ix, iy, idm = D["iris_x"][i], D["iris_y"][i], D["iris_diam"][i]
    if np.isfinite(ix) and np.isfinite(idm) and idm > 0:
        cv2.circle(vis, (int(ix), int(iy)), int(idm / 2),
                   C_IRIS, 1 if blink else 2, cv2.LINE_AA)

    # --- tracked features ---
    if feats is not None and show_feats != "off":
        xy = feats["feat_xy"][i]
        ok = feats["feat_ok"][i]
        sel = ok if show_feats == "valid" else np.isfinite(xy[:, 0])
        pts = xy[sel]
        good = ok[sel]
        for (x, y), g in zip(pts, good):
            if not np.isfinite(x):
                continue
            cv2.circle(vis, (int(x), int(y)), 2,
                       C_FEAT if g else C_FEAT_BAD, -1, cv2.LINE_AA)

    # --- pupil (suppressed on blinks: the estimate is a lid artefact) ---
    px, py, pdm = D["pupil_x"][i], D["pupil_y"][i], D["pupil_diam"][i]
    if not blink and np.isfinite(px) and np.isfinite(pdm):
        cv2.circle(vis, (int(px), int(py)), max(3, int(pdm / 2)),
                   C_PUPIL, 2, cv2.LINE_AA)
        cv2.drawMarker(vis, (int(px), int(py)), C_PUPIL,
                       cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)

    if blink:
        cv2.putText(vis, "BLINK", (16, 42), MONO, 1.1, C_BLINK, 2, cv2.LINE_AA)

    return vis


def put(canvas, text, x, y, scale=0.46, color=C_INK, font=FONT, thick=1):
    cv2.putText(canvas, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


ROW_H = 22          # vertical pitch of a label/value row
SEC_GAP = 10        # extra space between sections
LEGEND_H = 100      # reserved at the bottom for the controls legend
TRACE_H = 92        # height of the torsion trace box


def row(canvas, y, label, value, x0, color=C_INK):
    put(canvas, label, x0, y, 0.42, C_MUTED)
    put(canvas, value, x0 + 165, y, 0.46, color, MONO)
    return y + ROW_H


def draw_sidebar(h, i, D, feats, state):
    """Build the right-hand readout panel."""
    sb = np.full((h, SIDEBAR_W, 3), C_PANEL, np.uint8)
    x0 = 22
    y = 42
    n = len(D["frame"])
    blink = D["blink"][i] == 1

    put(sb, "OCULAR REVIEW", x0, y, 0.62, C_INK, MONO)
    y += 20
    put(sb, "replay of computed tracking", x0, y, 0.40, C_MUTED)
    y += 26
    cv2.line(sb, (x0, y), (SIDEBAR_W - x0, y), C_LINE, 1)
    y += 30

    # status chip
    label = "BLINK" if blink else "TRACKING"
    chip = C_BLINK if blink else C_ACCENT
    cv2.rectangle(sb, (x0, y - 17), (x0 + 108, y + 7), chip, -1)
    put(sb, label, x0 + 12, y, 0.46, (20, 20, 20), MONO)
    put(sb, "%s  x%.2g" % ("PLAY" if state["playing"] else "PAUSE",
                           state["speed"]), x0 + 126, y, 0.44, C_MUTED, MONO)
    y += 34

    y = row(sb, y, "frame", "%d / %d" % (i, n - 1), x0)
    y = row(sb, y, "time", "%.2f s" % D["time"][i], x0)
    y += SEC_GAP

    put(sb, "TORSION", x0, y, 0.42, C_ACCENT, MONO); y += ROW_H
    for key, lab in (("torsion_deg", "combined"),
                     ("torsion_outer_deg", "outer"),
                     ("torsion_inner_deg", "inner")):
        v = D[key][i]
        # outer is the cleanest estimator -- highlight it
        c = C_ACCENT if lab == "outer" else C_INK
        y = row(sb, y, lab, "--" if not np.isfinite(v) else "%+.3f deg" % v,
                x0, c)
    y += SEC_GAP

    put(sb, "PUPIL  (RITnet)", x0, y, 0.42, C_ACCENT, MONO); y += ROW_H
    y = row(sb, y, "centre", _xy(D["pupil_x"][i], D["pupil_y"][i]), x0)
    y = row(sb, y, "diameter", _px(D["pupil_diam"][i]), x0)
    y += SEC_GAP

    put(sb, "IRIS  (RITnet)", x0, y, 0.42, C_ACCENT, MONO); y += ROW_H
    y = row(sb, y, "centre", _xy(D["iris_x"][i], D["iris_y"][i]), x0)
    y = row(sb, y, "diameter", _px(D["iris_diam"][i]), x0)
    y += SEC_GAP

    put(sb, "TRACKING", x0, y, 0.42, C_ACCENT, MONO); y += ROW_H
    y = row(sb, y, "features", "--" if not np.isfinite(D["n_features"][i])
            else "%d" % D["n_features"][i], x0)
    if feats is not None:
        sid = feats["seg_id"][i]
        y = row(sb, y, "segment", "%d" % sid if sid >= 0 else "--", x0)
        y = row(sb, y, "shown", "%d" % int(feats["feat_ok"][i].sum()), x0)

    # ---- torsion trace, only if it fits above the legend ----
    legend_top = h - LEGEND_H
    if state["show_trace"] and (legend_top - y) > (TRACE_H + 34):
        y += 12
        put(sb, "OUTER TORSION  +/- 5 deg", x0, y, 0.38, C_MUTED)
        y += 8
        gw = SIDEBAR_W - 2 * x0
        g0 = y
        cv2.rectangle(sb, (x0, g0), (x0 + gw, g0 + TRACE_H), C_LINE, 1)
        cv2.line(sb, (x0, g0 + TRACE_H // 2), (x0 + gw, g0 + TRACE_H // 2),
                 C_LINE, 1)
        half = 400
        lo, hi = max(0, i - half), min(n, i + half)
        seg = D["torsion_outer_deg"][lo:hi]
        if seg.size > 1:
            xs = np.linspace(0, gw, seg.size)
            ys = np.clip(seg, -5, 5)
            pts = [(int(x0 + xx), int(g0 + TRACE_H / 2 - yy / 5.0 * (TRACE_H / 2)))
                   for xx, yy in zip(xs, ys) if np.isfinite(yy)]
            if len(pts) > 1:
                cv2.polylines(sb, [np.array(pts, np.int32)], False,
                              C_ACCENT, 1, cv2.LINE_AA)
        cx = int(x0 + gw * (i - lo) / max(hi - lo, 1))
        cv2.line(sb, (cx, g0), (cx, g0 + TRACE_H), C_PUPIL, 1)

    # ---- legend / controls, pinned to the bottom ----
    ly = legend_top
    cv2.line(sb, (x0, ly), (SIDEBAR_W - x0, ly), C_LINE, 1)
    ly += 20
    for line in ("SPACE play/pause     . , step",
                 "L J jump 100         + - speed",
                 "F features  T trace  S save",
                 "Q quit"):
        put(sb, line, x0, ly, 0.38, C_MUTED)
        ly += 18
    return sb


def _xy(x, y):
    if not (np.isfinite(x) and np.isfinite(y)):
        return "--"
    return "%.1f, %.1f" % (x, y)


def _px(v):
    return "--" if not np.isfinite(v) else "%.1f px" % v


def compose(frame_bgr, i, D, feats, state, disp_h):
    """Video (scaled) + sidebar -> one canvas."""
    vis = draw_overlay(frame_bgr, i, D, feats, state["show_feats"])
    h, w = vis.shape[:2]
    s = disp_h / float(h)
    vis = cv2.resize(vis, (int(w * s), disp_h), interpolation=cv2.INTER_AREA)
    sb = draw_sidebar(disp_h, i, D, feats, state)
    canvas = np.hstack([vis, sb])
    cv2.line(canvas, (vis.shape[1], 0), (vis.shape[1], disp_h), C_LINE, 1)
    return canvas


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/raw/8.avi")
    ap.add_argument("--csv", default="data/video_8/combined_8.csv")
    ap.add_argument("--features", default="data/video_8/features_8.npz",
                    help="raw feature trajectories from ocular.py (optional)")
    ap.add_argument("--meta", default="data/video_8/frames/_frames_meta.json",
                    help="used only to verify the video resolution matches "
                         "the coordinate space of the CSV")
    ap.add_argument("--height", type=int, default=820,
                    help="display height in px. The torsion trace needs ~820; "
                         "below that it is dropped automatically rather than "
                         "overlapping the controls.")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--export", default=None,
                    help="render to this video file instead of opening a window")
    ap.add_argument("--count", type=int, default=500,
                    help="frames to render when using --export")
    args = ap.parse_args()

    for p in (args.video, args.csv):
        if not os.path.isfile(p):
            sys.exit("Not found: %s\n(run this from the project root)" % p)

    D = load_csv(args.csv)
    n = len(D["frame"])
    for k in ("pupil_x", "pupil_y", "pupil_diam", "iris_x", "iris_y",
              "iris_diam", "torsion_deg", "torsion_inner_deg",
              "torsion_outer_deg", "n_features", "blink", "time"):
        D[k] = col(D, k, n)

    feats = None
    if args.features and os.path.isfile(args.features):
        z = np.load(args.features)
        feats = {k: z[k] for k in z.files}
        if len(feats["feat_xy"]) < n:
            print("NOTE: features file is shorter than the CSV; "
                  "feature dots stop early.")
        print("Loaded raw trajectories: %s" % (feats["feat_xy"].shape,))
    else:
        print("No features file -- feature dots disabled.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("Cannot open video: " + args.video)
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0

    # Same guard as the review app: CSV coordinates are in ORIGINAL video space,
    # so a padded/shrunk copy would put every marker in the wrong place.
    if args.meta and os.path.isfile(args.meta):
        m = json.load(open(args.meta))
        exp = (int(m["original_width"]), int(m["original_height"]))
        if (vw, vh) != exp:
            sys.exit("Resolution mismatch: video is %dx%d but the CSV is in "
                     "%dx%d space.\nUse the ORIGINAL video, not a padded copy."
                     % (vw, vh, exp[0], exp[1]))

    state = {"playing": True, "speed": 1.0, "show_feats": "valid",
             "show_trace": True}

    # ---------------- export mode ----------------
    if args.export:
        i0 = max(0, args.start)
        i1 = min(n, i0 + args.count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
        writer = None
        state["playing"] = False
        for i in range(i0, i1):
            ok, fr = cap.read()
            if not ok:
                break
            canvas = compose(fr, i, D, feats, state, args.height)
            if writer is None:
                writer = cv2.VideoWriter(
                    args.export, cv2.VideoWriter_fourcc(*"MJPG"), fps,
                    (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            if (i - i0) % 100 == 0:
                print("  %d/%d" % (i - i0, i1 - i0))
        if writer:
            writer.release()
        cap.release()
        print("Wrote:", args.export)
        return

    # ---------------- interactive ----------------
    win = "Ocular live view"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    seek = {"req": None, "own": -1}

    def on_track(v):
        if v != seek["own"]:
            seek["req"] = v

    cv2.createTrackbar("frame", win, 0, max(n - 1, 1), on_track)

    i = max(0, min(args.start, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    need_seek = False
    frame = None

    print("Playing. SPACE=pause  .=step  Q=quit")
    while True:
        if seek["req"] is not None:
            i = int(seek["req"]); seek["req"] = None; need_seek = True
        if need_seek:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            need_seek = False
            ok, frame = cap.read()
            if not ok:
                break
        elif state["playing"] or frame is None:
            ok, frame = cap.read()
            if not ok:
                state["playing"] = False
            else:
                i += 1
                if i >= n:
                    i = n - 1
                    state["playing"] = False

        if frame is None:
            break

        canvas = compose(frame, min(i, n - 1), D, feats, state, args.height)
        cv2.imshow(win, canvas)
        seek["own"] = min(i, n - 1)
        cv2.setTrackbarPos("frame", win, seek["own"])

        delay = max(1, int(1000.0 / (fps * state["speed"]))) if state["playing"] else 30
        k = cv2.waitKey(delay) & 0xFF

        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            state["playing"] = not state["playing"]
        elif k in (ord("."), 83):            # step forward
            state["playing"] = False
            i = min(i + 1, n - 1); need_seek = True
        elif k in (ord(","), 81):            # step back
            state["playing"] = False
            i = max(i - 1, 0); need_seek = True
        elif k == ord("l"):
            i = min(i + 100, n - 1); need_seek = True
        elif k == ord("j"):
            i = max(i - 100, 0); need_seek = True
        elif k in (ord("+"), ord("=")):
            state["speed"] = min(state["speed"] * 1.5, 8.0)
        elif k == ord("-"):
            state["speed"] = max(state["speed"] / 1.5, 0.125)
        elif k == ord("f"):
            state["show_feats"] = {"valid": "all", "all": "off",
                                   "off": "valid"}[state["show_feats"]]
        elif k == ord("t"):
            state["show_trace"] = not state["show_trace"]
        elif k == ord("s"):
            fn = "live_view_frame_%06d.png" % i
            cv2.imwrite(fn, canvas)
            print("saved", fn)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
