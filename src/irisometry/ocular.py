"""
ocular_pipeline.py
======================================================================
A single, clean-room Python pipeline for gaze + ocular-torsion estimation
from eye videos. It consolidates:

  * the ORIGINAL MATLAB irissometry algorithm (Strauch/Naber lineage):
      - starburst-style pupil detection with a QUANTIFIED circle-fit error
      - fit-error-based blink detection (far cleaner than a feature-count guess)
      - per-annulus iris feature geometry (inner vs outer iris)
  * the PYTHON port's virtue:
      - it keeps the RAW per-feature trajectories, so anything can be derived
        downstream (the MATLAB version discards them)
  * the TORSION derivation that NEITHER original computes directly:
      - centroid re-centring (removes gaze/translation) + circular-median of
        per-feature angular change about the centroid  -> degrees of torsion
      - blink-segmented and re-referenced after every blink (documented property)
  * a hook for RITnet (deep-learning segmentation) to supply pupil/iris and
    the blink signal when available, replacing the classical front-end.

OUTPUT: a single per-frame CSV with everything aligned on frame index:
    frame, time, pupil_x, pupil_y, pupil_radius, pupil_fit_err,
    n_features, blink, torsion_deg, torsion_inner_deg, torsion_outer_deg

Plus an optional quick-look PNG and an optional annotated overlay video.

Design notes
------------
- Pure inference. No training.
- Pupil detection here is a robust, self-contained classical detector (dark-blob
  + ellipse fit + residual) that mirrors the MATLAB starburst's PURPOSE (locate
  pupil, quantify fit quality for blink detection) without cloning its every line.
  If RITnet masks are supplied, they override this and are used instead.
- The torsion math is validated against synthetic known rotations (see __main__
  self-test): a known N-degree rotation under arbitrary translation recovers N.

Author: (your name), MSc Bioinformatics & CS, University of Leicester
"""

import os
import sys
import csv
import json
import math
import pickle
import argparse
import numpy as np
import cv2


# ======================================================================
# CONFIG
# ======================================================================
class Config:
    # --- feature detection / tracking (Shi-Tomasi + Lucas-Kanade) ---
    max_features = 1000          # cap on detected features
    min_features = 100           # require at least this many to call a frame valid
    min_quality = 0.005          # Shi-Tomasi quality level (starting value)
    min_distance = 10            # min pixel distance between features (starting)
    quality_floor = 0.0005       # never relax quality below this (prevents garbage)
    max_redetect = 10            # max quality-relax attempts before declaring blink

    # --- blink handling ---
    # A blink is declared when the valid-feature count collapses, OR (preferred)
    # when the pupil fit error exceeds pupil_fit_err_max (classical signal), OR
    # when a supplied RITnet mask reports no pupil.
    blink_feature_floor = 20     # < this many valid features  => blink
    pupil_fit_err_max = 0.18     # normalised ellipse-fit residual above => blink
    use_pupil_blink = False      # use classical pupil-fit-error as blink signal?
                                 # False = feature-tracking decides blinks (robust
                                 # on IR footage). Set True only if the classical
                                 # pupil detector works well on your video, or rely
                                 # on RITnet (always authoritative when supplied).
    # Frames to skip after a blink while the lid sweeps back off the iris.
    # None  = derive from the video's frame rate as ceil(0.4 * fps), i.e. ~400 ms,
    #         which is the approximate duration of a blink (and what the original
    #         irisometry implementation recommends).
    # An integer here overrides that.
    # NOTE: this was previously hardcoded to 5, which is only 100 ms at 50 fps --
    # tracking resumed while the eyelid still covered the iris, letting partially
    # occluded frames back into the torsion signal.
    frames_wait_after_blink = None
    min_segment_len = 3          # ignore inter-blink segments shorter than this

    # --- iris-mask feature gating (RITnet -> irisometry) ---
    # Pixels to erode off the iris mask before detecting features. The iris
    # class borders the pupil on the inside and the sclera/lid on the outside;
    # both edges are strong Shi-Tomasi corners but neither is iris TEXTURE, and
    # the pupil boundary in particular moves with dilation rather than with
    # rotation. Eroding keeps features on genuine stromal detail.
    mask_erode_px = 6

    # Specular highlights (IR illuminator reflections) sit inside the iris
    # region and are very strong corners, but they are fixed relative to the
    # LIGHT SOURCE, not the eye -- they do not rotate with it. Excluding
    # near-saturated pixels keeps them out of the torsion estimate. Set to 255
    # to disable.
    glint_threshold = 248
    glint_dilate_px = 3          # grow the exclusion a little past the core

    # --- pupil detection (classical fallback when RITnet not supplied) ---
    pupil_threshold = 34         # grayscale value below which is "pupil-dark"
    pupil_radius_range = (15, 160)

    # --- Lucas-Kanade params ---
    lk_win = (15, 15)
    lk_levels = 2

    # --- output ---
    # Write the RAW per-feature trajectories alongside the CSV. The CSV only
    # holds the collapsed per-frame torsion; the trajectories are what let you
    # re-derive a different estimator, a different inner/outer split, or
    # per-feature quality metrics WITHOUT re-running the tracking. This is the
    # stated advantage of the Python port over the MATLAB original, which
    # discards them.
    save_features = True
    save_overlay_video = False
    histogram_eq = None          # None | 'eqHist' | 'eqAdaptHist'
    show_frames = False          # live tracking window (set via --show)
    show_delay_ms = 1            # waitKey delay; 1 = fast, 25 = ~real-time-ish


# ======================================================================
# Geometry helpers
# ======================================================================
def cart2pol(x, y):
    rho = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    return rho, phi


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def circular_median_deg(delta_theta_rad):
    """Robust common rotation (degrees) from per-feature angle changes (radians)."""
    d = delta_theta_rad[~np.isnan(delta_theta_rad)]
    if d.size == 0:
        return np.nan
    return float(np.degrees(np.median(wrap_to_pi(d))))


# ======================================================================
# Classical pupil detector (fallback; mirrors MATLAB's purpose, not its code)
# ======================================================================
def detect_pupil(gray, cfg, prev_center=None):
    """
    Locate the pupil as the largest dark blob, fit an ellipse, and return a
    normalised fit residual that acts as the blink signal (high => poor fit =>
    occluded/blink), analogous to the MATLAB circle-fit error (eyeData col 5).

    Returns: (cx, cy, radius, fit_err, found)
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(blur, cfg.pupil_threshold, 255, cv2.THRESH_BINARY_INV)

    # morphological clean-up
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return (np.nan, np.nan, np.nan, 1.0, False)

    rmin, rmax = cfg.pupil_radius_range
    amin = np.pi * rmin * rmin
    amax = np.pi * rmax * rmax

    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < amin or area > amax:
            continue
        if area > best_area:
            best_area = area
            best = c
    if best is None or len(best) < 5:
        return (np.nan, np.nan, np.nan, 1.0, False)

    # fit ellipse; residual = how well the blob matches its fitted ellipse
    (ex, ey), (MA, ma), ang = cv2.fitEllipse(best)
    r = 0.25 * (MA + ma)  # mean radius
    ellipse_area = np.pi * (MA / 2.0) * (ma / 2.0)
    # normalised residual: 0 = perfect circle-blob match, ->1 = poor (blink-like)
    fit_err = abs(ellipse_area - best_area) / (ellipse_area + 1e-6)
    # eccentricity penalty (a blinking/occluded pupil fits as a thin ellipse)
    ecc = 1.0 - (min(MA, ma) / (max(MA, ma) + 1e-6))
    fit_err = float(min(1.0, fit_err + 0.5 * ecc))

    return (float(ex), float(ey), float(r), fit_err, True)


# ======================================================================
# Feature detection within the circular AOI
# ======================================================================
def detect_features(gray, aoi, cfg, iris_mask=None):
    """
    Shi-Tomasi corners for torsion tracking, with adaptive quality relaxation.
    Returns Nx2 float32 array (may be empty) and a 'valid' flag.

    WHY A MASK, NOT JUST A CIRCLE
    -----------------------------
    A circular AOI large enough to reach the limbus necessarily also contains
    eyelid and eyelashes, because the palpebral fissure is shorter vertically
    than the iris is wide. Lashes are high-contrast and produce far stronger
    Shi-Tomasi corners than iris texture does, so they dominate the detections.

    Measured on this footage with a circular AOI alone, only 33.7% of tracked
    features actually sat on iris tissue; 53.6% were on lid/lashes and 10.8% on
    the pupil. That is fatal for torsion for two reasons:

      * lashes move with the EYELID, not with eyeball rotation, so they inject
        lid motion into the angular measurement;
      * the torsion step re-centres on the feature centroid to cancel gaze
        translation -- if most features are on the lid, that centroid tracks
        the lid, and the "translation removal" removes the wrong thing.

    Passing iris_mask (RITnet's iris class, in ORIGINAL video coordinates)
    constrains detection to iris tissue only. This is the automated lid removal
    that the original irisometry implementation lists as an unsolved TODO
    ("Implement automated lid detection! Remove features at eye lids").

    iris_mask : uint8, nonzero where a feature is allowed. None = circle only.
    """
    cx, cy, r = aoi
    x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
    x1, y1 = min(gray.shape[1], int(cx + r)), min(gray.shape[0], int(cy + r))
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return np.empty((0, 2), np.float32), False

    roi_mask = None
    if iris_mask is not None:
        roi_mask = iris_mask[y0:y1, x0:x1]
        if roi_mask.size == 0 or not roi_mask.any():
            # no iris visible in this frame (closed eye) -> let the caller
            # treat it as a blink rather than detecting lashes instead
            return np.empty((0, 2), np.float32), False

    img = np.float32(roi)
    q = cfg.min_quality
    d = cfg.min_distance

    pts = np.empty((0, 2), np.float32)
    for _ in range(cfg.max_redetect + 1):
        corners = cv2.goodFeaturesToTrack(img, cfg.max_features, q, d,
                                          mask=roi_mask)
        if corners is None:
            pts = np.empty((0, 2), np.float32)
        else:
            pts = corners.reshape(-1, 2).astype(np.float32)
            pts[:, 0] += x0
            pts[:, 1] += y0
            # keep only those inside the AOI circle
            inside = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2 < r * r
            pts = pts[inside]

        if len(pts) > cfg.min_features:
            return pts, True
        if q <= cfg.quality_floor:
            break
        q *= 0.9
        d *= 0.9

    return pts, False


def load_iris_mask(mask_dir, fidx, shape, scale, pad_x, pad_y, erode_px):
    """RITnet iris mask for one frame, mapped into ORIGINAL video coordinates.

    Uses iris (blue) ONLY -- not the pupil -- so features cannot land on the
    pupil, whose boundary corners move with dilation rather than rotation.
    Eroding pulls features off the pupil and limbus boundaries for the same
    reason: those edges are strong corners but they are not iris texture.

    Returns uint8 mask at `shape`, or None if the mask file is missing.
    """
    p = os.path.join(mask_dir, "frame_%06d.png" % fidx)
    m = cv2.imread(p, cv2.IMREAD_COLOR)
    if m is None:
        return None
    b, g, r = m[:, :, 0], m[:, :, 1], m[:, :, 2]
    iris = ((b > 128) & (r < 100) & (g < 100)).astype(np.uint8) * 255
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * erode_px + 1, 2 * erode_px + 1))
        iris = cv2.erode(iris, k)
    if iris.sum() == 0:
        return iris_to_original(iris, shape, scale, pad_x, pad_y)
    return iris_to_original(iris, shape, scale, pad_x, pad_y)


def iris_to_original(mask, shape, scale, pad_x, pad_y):
    """Undo the letterbox: crop the padding, then resize to the original frame."""
    h, w = mask.shape[:2]
    x0, y0 = int(round(pad_x)), int(round(pad_y))
    x1, y1 = w - x0, h - y0
    if x1 <= x0 or y1 <= y0:
        core = mask
    else:
        core = mask[y0:y1, x0:x1]
    out = cv2.resize(core, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return out


# ======================================================================
# Main pipeline
# ======================================================================
class OcularPipeline:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.lk_params = dict(
            winSize=self.cfg.lk_win,
            maxLevel=self.cfg.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    # ------------------------------------------------------------------
    def run(self, video_path, aoi_seed, ritnet_csv=None, meta_path=None,
            out_dir="output", max_frames=None, mask_dir=None):
        """
        video_path : path to eye video
        aoi_seed   : [cx, cy, r] iris circle (manual, or from RITnet later)
        ritnet_csv : optional CSV with per-frame pupil from RITnet; columns
                     frame,pupil_x,pupil_y,pupil_diam[,pupil_found]. If given,
                     RITnet's pupil + blink override the classical detector.
        meta_path  : _frames_meta.json, used to map ritnet_csv coordinates from
                     mask space into original video space. See _load_ritnet.
        """
        cfg = self.cfg
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(video_path))[0]

        ritnet = self._load_ritnet(ritnet_csv, meta_path) if ritnet_csv else None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            sys.exit("Cannot open video: " + video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            nframes = min(nframes, int(max_frames))
        print("Video: %s | %d frames @ %.1f fps" % (base, nframes, fps))

        aoi = list(map(float, aoi_seed))  # static reference AOI (locked)

        # ---- iris-mask gating -------------------------------------------
        # Constrain features to RITnet's iris class instead of a bare circle.
        # See detect_features() for why this matters.
        mask_scale = mask_pad_x = mask_pad_y = None
        if mask_dir:
            if not os.path.isdir(mask_dir):
                sys.exit("Mask folder not found: " + mask_dir)
            if not meta_path:
                sys.exit("--masks requires --meta (to undo the letterbox).")
            _m = json.load(open(meta_path))
            mask_scale = float(_m.get("scale", 1.0))
            mask_pad_x = float(_m.get("pad_x", 0.0))
            mask_pad_y = float(_m.get("pad_y", 0.0))
            print("Feature gating: RITnet iris mask (erode %d px) from %s"
                  % (cfg.mask_erode_px, mask_dir))
        else:
            print("Feature gating: circular AOI only -- features may land on "
                  "eyelid/lashes. Pass --masks to restrict them to iris tissue.")

        # Blink recovery window: ~400 ms unless explicitly overridden.
        wait_frames = cfg.frames_wait_after_blink
        if wait_frames is None:
            wait_frames = int(math.ceil(0.4 * fps))
        print("Blink recovery window: %d frames (%.0f ms)"
              % (wait_frames, 1000.0 * wait_frames / fps))

        # per-frame records
        rows = []                  # dict per analysed frame
        feat_prev = None           # tracked points from previous frame (Nx2)
        ref_theta = None           # marks whether a segment reference exists
        ref_pts = None             # reference POSITIONS for the current segment
        ref_r = None               # reference radii (for inner/outer split)
        prev_gray = None

        writer = None
        if cfg.save_overlay_video:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        # ---- raw per-feature trajectory capture ----------------------------
        # Slot i is the SAME physical feature only within one inter-blink
        # segment; features are re-detected after every blink, so seg_id must be
        # used to interpret the arrays. NaN = that slot held no valid feature.
        cap_feats = cfg.save_features and nframes > 0
        if cap_feats:
            feat_xy = np.full((nframes, cfg.max_features, 2), np.nan, np.float32)
            feat_ok = np.zeros((nframes, cfg.max_features), bool)
            seg_id = np.full(nframes, -1, np.int32)
            print("Capturing raw feature trajectories (%.0f MB)"
                  % (feat_xy.nbytes / 1e6))
        cur_seg = -1

        fidx = -1
        wait_left = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fidx += 1
            if max_frames and fidx >= nframes:
                break
            t = fidx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cfg.histogram_eq == "eqHist":
                gray = cv2.equalizeHist(gray)
            elif cfg.histogram_eq == "eqAdaptHist":
                gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)

            # ---- pupil metrics + blink signal ----
            # Priority for the blink signal:
            #   1. RITnet (best): pupil mask present/absent  -> authoritative.
            #   2. Classical pupil-fit-error, but ONLY if cfg.use_pupil_blink is
            #      True AND the detector is actually finding pupils on this footage.
            #   3. Otherwise: feature-tracking decides blinks (robust on IR eye
            #      video; this is what the original Python port relied on).
            # The classical pupil detector ALWAYS runs to populate the pupil
            # columns, but its FAILURE never forces a blink on its own -- that was
            # the bug that flagged every frame as a blink on real footage.
            if ritnet is not None and fidx in ritnet:
                px, py, pr, ferr = ritnet[fidx]
                pupil_found = not np.isnan(px)
                is_blink = (not pupil_found)
            else:
                px, py, pr, ferr, pupil_found = detect_pupil(gray, cfg)
                if cfg.use_pupil_blink and pupil_found:
                    is_blink = (ferr > cfg.pupil_fit_err_max)
                else:
                    # Defer the blink decision to feature tracking below.
                    is_blink = False

            # iris mask for THIS frame, in original video coordinates
            iris_mask = None
            if mask_dir:
                iris_mask = load_iris_mask(mask_dir, fidx, gray.shape,
                                           mask_scale, mask_pad_x, mask_pad_y,
                                           cfg.mask_erode_px)
                if iris_mask is not None and cfg.glint_threshold < 255:
                    # drop specular reflections: fixed to the illuminator, so
                    # they do not rotate with the eye
                    glint = (gray >= cfg.glint_threshold).astype(np.uint8)
                    if cfg.glint_dilate_px > 0:
                        gk = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (2 * cfg.glint_dilate_px + 1,
                             2 * cfg.glint_dilate_px + 1))
                        glint = cv2.dilate(glint, gk)
                    iris_mask = cv2.bitwise_and(iris_mask,
                                                iris_mask, mask=1 - glint)

            # ---- feature tracking ----
            torsion = np.nan
            torsion_in = np.nan
            torsion_out = np.nan
            n_valid = 0

            if wait_left > 0:
                # still recovering from a blink: skip tracking this frame
                wait_left -= 1
                is_blink = True
                feat_prev = None
                ref_theta = None
                ref_pts = None
            elif is_blink:
                feat_prev = None
                ref_theta = None
                ref_pts = None
                wait_left = wait_frames
            else:
                if feat_prev is None or len(feat_prev) == 0:
                    # (re)seed features against the LOCKED aoi -> new segment
                    pts, valid = detect_features(gray, aoi, cfg, iris_mask)
                    if valid:
                        feat_prev = pts
                        # Store the reference POSITIONS, not reference angles.
                        # Angles must be recomputed each frame about the centroid
                        # of whichever features are still valid then -- see the
                        # torsion block below for why.
                        ref_pts = pts.copy()
                        ref_theta = True          # segment reference now exists
                        ref_r = None
                        torsion = 0.0
                        torsion_in = 0.0
                        torsion_out = 0.0
                        n_valid = len(pts)
                        cur_seg += 1          # new inter-blink segment starts here
                        if cap_feats:
                            m = min(len(pts), cfg.max_features)
                            feat_xy[fidx, :m] = pts[:m]
                            feat_ok[fidx, :m] = True
                            seg_id[fidx] = cur_seg
                    else:
                        is_blink = True
                        wait_left = wait_frames
                else:
                    # track existing features into this frame
                    p0 = feat_prev.reshape(-1, 1, 2)
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, gray, p0, None, **self.lk_params)
                    p0r, _, _ = cv2.calcOpticalFlowPyrLK(
                        gray, prev_gray, p1, None, **self.lk_params)
                    fb = np.abs(p0 - p0r).reshape(-1, 2).max(axis=1)
                    good = (st.reshape(-1) == 1) & (fb < 1.0)

                    p1 = p1.reshape(-1, 2)
                    # keep correspondence with the reference arrays
                    cur = p1.copy()
                    cur[~good] = np.nan

                    # in-AOI check
                    inside = ((cur[:, 0] - aoi[0]) ** 2 +
                              (cur[:, 1] - aoi[1]) ** 2) < aoi[2] ** 2

                    # Features drift. A point seeded on iris can be carried onto
                    # the lid or across the pupil boundary within a segment, so
                    # re-test membership every frame rather than trusting the
                    # seed. Points that leave the iris stop contributing to both
                    # the torsion angle AND the centroid used to cancel
                    # translation -- the latter is the important part.
                    if iris_mask is not None:
                        h_, w_ = iris_mask.shape[:2]
                        xi = np.clip(np.nan_to_num(cur[:, 0], nan=-1).astype(int),
                                     0, w_ - 1)
                        yi = np.clip(np.nan_to_num(cur[:, 1], nan=-1).astype(int),
                                     0, h_ - 1)
                        on_iris = (iris_mask[yi, xi] > 0) & np.isfinite(cur[:, 0])
                        inside = inside & on_iris

                    n_valid = int(np.nansum(good & inside))

                    if n_valid < cfg.min_features * (1 - 0.40):
                        # too few survived -> treat as blink/loss, re-seed next
                        feat_prev = None
                        ref_theta = None
                        ref_pts = None
                        is_blink = True
                        wait_left = wait_frames
                    else:
                        # ---- TORSION: centroid re-centre removes translation ----
                        #
                        # Both origins must be defined by the SAME set of
                        # features. Features are lost steadily through a segment
                        # (measured: 177 seeded, ~11 surviving to the end, i.e.
                        # 94% turnover). If the reference origin is fixed at the
                        # centroid of ALL seeded features while the current
                        # origin follows only the survivors, the two drift apart
                        # for reasons that have nothing to do with eye movement
                        # -- a spurious shift of 14.7 px median, up to 47.9 px,
                        # which correlated with torsion drift at rho = +0.68.
                        #
                        # Recomputing the reference centroid over the same subset
                        # makes any such shift common to both frames, so it
                        # cancels in the angular difference.
                        valid_mask = good & inside
                        rxr = np.nanmean(ref_pts[valid_mask, 0])
                        ryr = np.nanmean(ref_pts[valid_mask, 1])
                        cxr = np.nanmean(cur[valid_mask, 0])
                        cyr = np.nanmean(cur[valid_mask, 1])

                        ref_r, ref_theta_f = cart2pol(ref_pts[:, 0] - rxr,
                                                      ref_pts[:, 1] - ryr)
                        rr, th = cart2pol(cur[:, 0] - cxr, cur[:, 1] - cyr)
                        dth = th - ref_theta_f
                        torsion = circular_median_deg(np.where(valid_mask, dth,
                                                              np.nan))

                        # inner vs outer iris (per MATLAB inner/outer rim split).
                        # Radii come from the REFERENCE frame, so a feature keeps
                        # its ring membership for the whole segment.
                        if ref_r is not None:
                            med = np.nanmedian(np.where(valid_mask, ref_r, np.nan))
                            inner = valid_mask & (ref_r < med)
                            outer = valid_mask & (ref_r >= med)
                            torsion_in = circular_median_deg(
                                np.where(inner, dth, np.nan))
                            torsion_out = circular_median_deg(
                                np.where(outer, dth, np.nan))

                        # advance: keep only good points, and their refs, so the
                        # arrays stay aligned for the next frame
                        feat_prev = p1
                        # ref_pts stays fixed for the segment; the ANGLES
                        # about it are recomputed per frame (see above).

                        if cap_feats:
                            m = min(len(cur), cfg.max_features)
                            feat_xy[fidx, :m] = cur[:m]
                            feat_ok[fidx, :m] = valid_mask[:m]
                            seg_id[fidx] = cur_seg

            rows.append(dict(
                frame=fidx, time=t,
                pupil_x=px, pupil_y=py, pupil_radius=pr, pupil_fit_err=ferr,
                n_features=n_valid, blink=int(is_blink),
                torsion_deg=torsion,
                torsion_inner_deg=torsion_in,
                torsion_outer_deg=torsion_out,
            ))

            # ---- optional live tracking display (like the old RunMe window) ----
            if cfg.show_frames:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # AOI circle (blue)
                cv2.circle(vis, (int(aoi[0]), int(aoi[1])), int(aoi[2]),
                           (255, 0, 0), 2)
                # tracked feature points (green); only if we have current points
                if (not is_blink) and feat_prev is not None and len(feat_prev):
                    for (fx, fy) in feat_prev:
                        if not (np.isnan(fx) or np.isnan(fy)):
                            cv2.circle(vis, (int(fx), int(fy)), 2, (0, 255, 0), -1)
                # pupil (red) if detected
                if pupil_found and not np.isnan(px):
                    cv2.circle(vis, (int(px), int(py)), max(2, int(pr)),
                               (0, 0, 255), 2)
                    cv2.drawMarker(vis, (int(px), int(py)), (0, 0, 255),
                                   cv2.MARKER_CROSS, 12, 1)
                # info panel
                tor_txt = "--" if np.isnan(torsion) else "%.2f deg" % torsion
                lines = [
                    "Time: %.2fs   Frame: %d" % (t, fidx),
                    "Features: %d" % n_valid,
                    "Torsion: %s" % tor_txt,
                    "BLINK" if is_blink else "tracking",
                ]
                for i, ln in enumerate(lines):
                    cv2.putText(vis, ln, (10, 28 + 26 * i),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7,
                                (0, 255, 255) if "BLINK" not in ln else (0, 0, 255),
                                1, cv2.LINE_AA)
                # scale big frames down to fit screen
                if vis.shape[1] > 1100:
                    s = 1100.0 / vis.shape[1]
                    vis = cv2.resize(vis, None, fx=s, fy=s)
                cv2.imshow("Ocular tracking (press q to quit)", vis)
                # waitKey is REQUIRED for the window to render; 1ms = fast playback
                if (cv2.waitKey(cfg.show_delay_ms) & 0xFF) == ord("q"):
                    print("  display interrupted by user (q)")
                    break

            prev_gray = gray

            if (fidx % max(1, nframes // 10)) == 0:
                print("  %d%%" % int(100 * fidx / max(1, nframes)))

        cap.release()
        if cfg.show_frames:
            cv2.destroyAllWindows()

        # ---- write CSV ----
        out_csv = os.path.join(out_dir, "ocular_%s.csv" % base)
        cols = ["frame", "time", "pupil_x", "pupil_y", "pupil_radius",
                "pupil_fit_err", "n_features", "blink", "torsion_deg",
                "torsion_inner_deg", "torsion_outer_deg"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if (isinstance(r[k], float) and np.isnan(r[k]))
                                else r[k]) for k in cols})
        print("Wrote:", out_csv)

        # ---- raw per-feature trajectories ----
        if cap_feats:
            out_npz = os.path.join(out_dir, "features_%s.npz" % base)
            np.savez_compressed(
                out_npz,
                feat_xy=feat_xy,      # (n_frames, max_features, 2) float32, NaN-padded
                feat_ok=feat_ok,      # (n_frames, max_features) bool: tracked AND in-AOI
                seg_id=seg_id,        # (n_frames,) int32; -1 = blink/untracked
                aoi=np.asarray(aoi, np.float64),
                fps=np.float64(fps),
            )
            used = feat_ok.any(axis=0).sum()
            print("Wrote: %s  (%d segments, %d feature slots used, %.0f MB on disk)"
                  % (out_npz, cur_seg + 1, used,
                     os.path.getsize(out_npz) / 1e6))
            print("  Coordinates are ORIGINAL video space. Polar coords about the")
            print("  AOI centre are derivable: r,th = cart2pol(x-aoi[0], y-aoi[1]).")
            print("  Slot i is the same feature only WITHIN one seg_id.")

        self._plot(rows, out_dir, base, fps)
        self._summary(rows)
        return out_csv

    # ------------------------------------------------------------------
    def _load_ritnet(self, path, meta_path=None):
        """Load RITnet per-frame pupil CSV into {frame: (x,y,r,fit_err)}.

        COORDINATE SPACES -- this is the easy thing to get wrong.
        ---------------------------------------------------------
        RITnet runs on letterboxed 640x400 frames, so ritnet_*.csv is in MASK
        space. This pipeline tracks torsion on the ORIGINAL video. Feeding mask
        coordinates straight in would place the pupil tens of pixels off, which
        corrupts the AOI recentring and the pupil columns while still looking
        superficially reasonable.

        Passing meta_path (the _frames_meta.json written at extraction) applies
        the same inverse transform merge.py uses:

            original = (mask - pad) / scale

        Without meta_path the CSV is assumed to be in original space already.
        The BLINK signal (is the pupil present at all) is unaffected by the
        transform, but the coordinates very much are -- so warn loudly.
        """
        sx = sy = 1.0
        px = py = 0.0
        if meta_path:
            with open(meta_path) as f:
                meta = json.load(f)
            if "scale" in meta:                 # shrink / letterbox style
                sx = sy = float(meta["scale"])
                px, py = float(meta["pad_x"]), float(meta["pad_y"])
            else:                                # older per-axis style, no pad
                sx = 1.0 / float(meta["scale_x"])
                sy = 1.0 / float(meta["scale_y"])
            print("  RITnet coords -> original space: x=(mx-%g)/%g, y=(my-%g)/%g"
                  % (px, sx, py, sy))
        else:
            print("  WARNING: no --meta given; assuming the RITnet CSV is already "
                  "in ORIGINAL video coordinates. If it came straight from "
                  "ritnet_metrices.py it is NOT, and the pupil columns will be wrong.")

        d = {}
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                fr = int(float(row["frame"]))
                try:
                    x = float(row.get("pupil_x", "nan"))
                    y = float(row.get("pupil_y", "nan"))
                    diam = float(row.get("pupil_diam", row.get("diameter", "nan")))
                except ValueError:
                    x = y = diam = np.nan
                # An explicit pupil_found=0 means the segmentation saw no pupil
                # at all -- a closed eye. Force NaN so the blink test below
                # (isnan -> blink) fires even if stale coordinates are present.
                pf = row.get("pupil_found", None)
                if pf is not None and str(pf).strip() not in ("", "1", "1.0"):
                    x = y = diam = np.nan
                if x == x:
                    x = (x - px) / sx
                    y = (y - py) / sy
                    diam = diam / sx if diam == diam else np.nan
                d[fr] = (x, y, diam / 2.0 if diam == diam else np.nan, 0.0)
        return d

    # ------------------------------------------------------------------
    def _plot(self, rows, out_dir, base, fps):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print("  (plot skipped:", e, ")")
            return
        t = np.array([r["time"] for r in rows])
        tor = np.array([r["torsion_deg"] if r["torsion_deg"] == r["torsion_deg"]
                        else np.nan for r in rows])
        nf = np.array([r["n_features"] for r in rows])
        bl = np.array([r["blink"] for r in rows])

        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].plot(t, tor, lw=0.6)
        ax[0].axhline(0, color="k", lw=0.4)
        ax[0].set_ylabel("Torsion (deg)")
        ax[0].set_title("Ocular torsion - " + base)
        for i in range(len(t)):
            if bl[i]:
                ax[0].axvspan(t[i], t[min(i + 1, len(t) - 1)],
                              color="red", alpha=0.04)
        ax[1].plot(t, nf, lw=0.6, color="tab:green")
        ax[1].set_ylabel("Valid features")
        ax[1].set_xlabel("Time (s)")
        fig.tight_layout()
        out_png = os.path.join(out_dir, "ocular_%s.png" % base)
        fig.savefig(out_png, dpi=120)
        print("Wrote:", out_png)

    # ------------------------------------------------------------------
    def _summary(self, rows):
        tor = np.array([r["torsion_deg"] for r in rows], dtype=float)
        valid = tor[~np.isnan(tor)]
        nb = sum(r["blink"] for r in rows)
        if valid.size:
            print("Torsion: median %.2f  IQR [%.2f, %.2f]  std %.2f  "
                  "(%d valid, %d blink frames)" % (
                      np.median(valid),
                      np.percentile(valid, 25), np.percentile(valid, 75),
                      np.std(valid), valid.size, nb))
        print("done")


# ======================================================================
# Self-test of the torsion math (run: python ocular_pipeline.py --selftest)
# ======================================================================
def _selftest():
    np.random.seed(0)
    ix, iy = 440.0, 379.0
    ang = np.random.uniform(0, 2 * np.pi, 300)
    rad = np.random.uniform(40, 120, 300)
    x0 = ix + rad * np.cos(ang)
    y0 = iy + rad * np.sin(ang)
    ok = True
    for deg in [0, 1, 2.5, -3, 7, -10]:
        k = np.radians(deg)
        # rotate about centre, then apply an arbitrary translation
        xr = ix + rad * np.cos(ang + k) + np.random.uniform(-25, 25)
        yr = iy + rad * np.sin(ang + k) + np.random.uniform(-25, 25)
        cx0, cy0 = x0.mean(), y0.mean()
        cx1, cy1 = xr.mean(), yr.mean()
        _, t0 = cart2pol(x0 - cx0, y0 - cy0)
        _, t1 = cart2pol(xr - cx1, yr - cy1)
        est = circular_median_deg(t1 - t0)
        flag = "OK" if abs(est - deg) < 0.05 else "FAIL"
        if flag == "FAIL":
            ok = False
        print("  known %6.2f  ->  estimated %7.3f   [%s]" % (deg, est, flag))

    # ------------------------------------------------------------------
    # Regression test: FEATURE DROPOUT.
    #
    # The test above keeps every feature, which hides the defect that
    # mattered most in practice. Features are lost steadily through a
    # segment (measured: ~94% turnover), and they are lost ASYMMETRICALLY
    # -- the eyelid encroaches from above, so the survivors are biased
    # downward. If the reference angles are held about the centroid of all
    # ORIGINALLY seeded features while the current frame re-centres on the
    # survivors, the two origins separate and the measured angle is wrong
    # in proportion to that separation. Measured on real data this produced
    # a spurious origin shift of up to 47.9 px, correlating with torsion
    # drift at rho = +0.68.
    #
    # Both origins must therefore be computed over the SAME subset.
    # ------------------------------------------------------------------
    print()
    print("  feature-dropout regression (true rotation 3.00 deg):")
    n = 300
    th0 = np.random.uniform(0, 2 * np.pi, n)
    r0 = np.random.uniform(110, 195, n)
    pts = np.c_[ix + r0 * np.cos(th0), iy + r0 * np.sin(th0)]
    a = np.radians(3.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.array([ix, iy])
    cur = (pts - c) @ R.T + c + np.array([12.0, -7.0])
    order = np.argsort(pts[:, 1])          # lose the topmost features first
    for frac in (1.0, 0.6, 0.3, 0.15):
        sel = np.zeros(n, bool)
        sel[order[:int(n * frac)]] = True
        rx, ry = pts[sel, 0].mean(), pts[sel, 1].mean()
        cx, cy = cur[sel, 0].mean(), cur[sel, 1].mean()
        _, tr = cart2pol(pts[:, 0] - rx, pts[:, 1] - ry)
        _, tc = cart2pol(cur[:, 0] - cx, cur[:, 1] - cy)
        est = circular_median_deg(np.where(sel, tc - tr, np.nan))
        flag = "OK" if abs(est - 3.0) < 0.05 else "FAIL"
        if flag == "FAIL":
            ok = False
        print("    %3.0f%% features kept  ->  %7.3f   [%s]"
              % (100 * frac, est, flag))

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Gaze + ocular torsion pipeline")
    ap.add_argument("video", nargs="?", help="path to eye video")
    ap.add_argument("--aoi", help="iris seed as cx,cy,r (e.g. 440,379,128)")
    ap.add_argument("--ritnet", help="optional RITnet per-frame pupil CSV")
    ap.add_argument("--meta", help="_frames_meta.json from extraction. REQUIRED with "
                                   "--ritnet unless that CSV is already in original "
                                   "video coordinates (it normally is not).")
    ap.add_argument("--out", default="output", help="output directory")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after N frames (for quick validation runs)")
    ap.add_argument("--masks", default=None,
                    help="RITnet mask folder. Restricts features to iris tissue "
                         "instead of a bare circle -- this is what keeps eyelashes "
                         "out of the torsion estimate. Requires --meta.")
    ap.add_argument("--erode", type=int, default=None,
                    help="pixels to erode off the iris mask (default %d)"
                         % Config.mask_erode_px)
    ap.add_argument("--min-quality", type=float, default=None,
                    help="Shi-Tomasi quality level (default %g). WARNING: "
                         "lowering this buys more features and a smoother-LOOKING "
                         "trace, but weak corners slide progressively under LK "
                         "tracking. Measured on this footage, 0.002 cut "
                         "frame-to-frame jitter yet increased within-segment "
                         "DRIFT from ~1 deg to ~15 deg. Judge any change by "
                         "drift, not smoothness." % Config.min_quality)
    ap.add_argument("--min-distance", type=int, default=None,
                    help="minimum spacing between features (default %d)"
                         % Config.min_distance)
    ap.add_argument("--show", action="store_true",
                    help="show live tracking window (slower; press q to quit)")
    ap.add_argument("--slow", action="store_true",
                    help="with --show, play closer to real time (25ms/frame)")
    ap.add_argument("--selftest", action="store_true",
                    help="validate the torsion math and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)

    if not args.video or not args.aoi:
        ap.error("video and --aoi cx,cy,r are required (or use --selftest)")

    aoi = [float(v) for v in args.aoi.split(",")]
    if len(aoi) != 3:
        ap.error("--aoi must be cx,cy,r")

    cfg = Config()
    cfg.show_frames = args.show
    cfg.show_delay_ms = 25 if args.slow else 1
    if args.erode is not None:
        cfg.mask_erode_px = args.erode
    if args.min_quality is not None:
        cfg.min_quality = args.min_quality
    if args.min_distance is not None:
        cfg.min_distance = args.min_distance
    OcularPipeline(cfg).run(args.video, aoi, ritnet_csv=args.ritnet,
                            meta_path=args.meta, out_dir=args.out,
                            max_frames=args.max_frames, mask_dir=args.masks)


if __name__ == "__main__":
    main()
