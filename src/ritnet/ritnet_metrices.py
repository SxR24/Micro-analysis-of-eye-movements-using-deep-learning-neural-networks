#!/usr/bin/env python3
"""
ritnet_metrics.py
Reads a folder of RITnet segmentation masks (the colored PNGs produced by
infer.py) and, for each frame, fits an ellipse to the pupil and iris regions,
then writes one CSV row per frame.

Class colors expected (from infer.py PALETTE):
    background = black   (0,0,0)
    sclera     = green   (0,255,0)
    iris       = blue    (0,0,255)
    pupil      = red     (255,0,0)

Output columns:
    frame, pupil_found, pupil_x, pupil_y, pupil_major, pupil_minor,
    pupil_angle, pupil_diam, iris_found, iris_x, iris_y, iris_major,
    iris_minor, iris_diam, iris_area_diam

IRIS GEOMETRY -- why this is not just "fit an ellipse to the blue blob"
----------------------------------------------------------------------
Two properties of the segmentation make the naive measurement wrong:

  1. The blue class is an ANNULUS, not a disc. RITnet labels the pupil as a
     separate class (red), so the blue region has a hole in the middle. An
     area-equivalent diameter (2*sqrt(area/pi)) computed on blue alone
     therefore under-reports the iris badly. We measure on (blue | red).

  2. The eyelids clip the iris TOP and BOTTOM, and rarely by equal amounts.
     So the visible blob's vertical extent is not the iris diameter, and its
     vertical centre is pulled toward whichever lid occludes less. On this
     footage the upper lid/lashes cut ~38 px while the lower cuts ~14 px,
     biasing a naive centre ~12 px downward (mask space).

Consequences for how we measure:

  * DIAMETER comes from the HORIZONTAL extent only -- that axis is unoccluded.
    Taken as the median width of the widest ~10% of rows, which is robust to
    stray segmentation pixels in a way a raw bounding box is not.
  * CENTRE X comes from the same widest rows.
  * CENTRE Y is taken from the PUPIL centre when the pupil is available, since
    pupil and iris are concentric. Only if the pupil is missing do we fall back
    to the (biased) vertical centre of the visible disc.

iris_diam is therefore the true iris diameter. iris_area_diam retains the old
area-equivalent number for diagnostics/backwards comparison. iris_major and
iris_minor remain the raw ellipse-fit axes of the visible disc.

- pupil_x/y and iris_x/y are centers in pixel coordinates of the mask image.
- Frames where the region is missing/too small get *_found = 0 and NaN values.
"""

import os, glob, csv, argparse, re
import numpy as np
import cv2


def _frame_index(filename):
    """Pull an integer frame index from a filename like frame_0021.png -> 21.
    Falls back to the raw name if no number is found (kept as string)."""
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.findall(r"\d+", base)
    return int(m[-1]) if m else base


def fit_region(binmask, min_area=15):
    """Given a binary mask of one class, keep the largest blob and fit an
    ellipse. Returns a dict of measurements or None if too small."""
    if binmask.sum() == 0:
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binmask.astype(np.uint8), 8)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == largest).astype(np.uint8)
    area = int(stats[largest, cv2.CC_STAT_AREA])
    if area < min_area:
        return None

    # centroid from moments (robust)
    M = cv2.moments(blob, binaryImage=True)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # equivalent-circle diameter from area
    diam = 2.0 * np.sqrt(area / np.pi)

    # ellipse fit needs >=5 contour points
    major = minor = angle = float("nan")
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if len(c) >= 5:
            (ex, ey), (MA, ma), ang = cv2.fitEllipse(c)
            # use ellipse center when available (sub-pixel, less jagged)
            cx, cy = ex, ey
            major, minor, angle = max(MA, ma), min(MA, ma), ang

    return dict(x=cx, y=cy, major=major, minor=minor, angle=angle,
                diam=diam, area=area)


def fit_iris(disc_mask, pupil, min_area=15, widest_frac=0.10):
    """Measure the iris from the FULL disc mask (blue | red).

    See the module docstring for why the horizontal extent is used for the
    diameter and why the centre is anchored on the pupil.

    Parameters
    ----------
    disc_mask : binary mask of iris|pupil (the full iris disc)
    pupil     : the dict returned by fit_region() for the pupil, or None
    widest_frac : fraction of rows (the widest ones) used for the robust
                  horizontal measurement. 0.10 = widest 10% of occupied rows.

    Returns a dict, or None if the disc is too small (blink / lost frame).
    """
    if disc_mask.sum() == 0:
        return None

    # keep only the largest blob -- drops speculars and stray label noise
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        disc_mask.astype(np.uint8), 8)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == largest).astype(np.uint8)
    area = int(stats[largest, cv2.CC_STAT_AREA])
    if area < min_area:
        return None

    # --- robust horizontal measurement -------------------------------------
    # Per occupied row, the span from leftmost to rightmost disc pixel. The
    # widest rows lie near the vertical centre of the circle, so their width is
    # the diameter and their midpoint is the centre x.
    rows = np.flatnonzero(blob.any(axis=1))
    if rows.size == 0:
        return None
    xs_any = blob[rows].astype(bool)
    xmin = xs_any.argmax(axis=1)                       # first True per row
    xmax = xs_any.shape[1] - 1 - xs_any[:, ::-1].argmax(axis=1)
    widths = (xmax - xmin + 1).astype(float)
    mids = (xmax + xmin) / 2.0

    k = max(3, int(round(widest_frac * rows.size)))
    k = min(k, rows.size)
    top = np.argpartition(widths, -k)[-k:]              # k widest rows

    diam_h = float(np.median(widths[top]))
    cx = float(np.median(mids[top]))

    # --- centre y: pupil is concentric with the iris ------------------------
    if pupil is not None and not np.isnan(pupil["y"]):
        cy = float(pupil["y"])
        cy_src = "pupil"
    else:
        cy = float(np.median(rows[top]))
        cy_src = "disc"

    # --- ellipse fit on the disc, kept for reference ------------------------
    major = minor = angle = float("nan")
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if len(c) >= 5:
            (_ex, _ey), (MA, ma), ang = cv2.fitEllipse(c)
            major, minor, angle = max(MA, ma), min(MA, ma), ang

    return dict(x=cx, y=cy, diam=diam_h, major=major, minor=minor,
                angle=angle, area=area,
                area_diam=2.0 * np.sqrt(area / np.pi), cy_src=cy_src)


COLS = ["frame", "pupil_found", "pupil_x", "pupil_y", "pupil_major",
        "pupil_minor", "pupil_angle", "pupil_diam", "iris_found",
        "iris_x", "iris_y", "iris_major", "iris_minor", "iris_diam",
        "iris_area_diam"]


def class_masks(bgr):
    """Split a RITnet colour mask into pupil and iris binary masks.

    Single definition of the palette thresholds. run_metrics_chunked.py used to
    carry its own copy; two copies of the class decision is exactly the kind of
    thing that silently diverges when the palette changes.
    """
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    pupil = (r > 128) & (g < 100) & (b < 100)      # red
    iris = (b > 128) & (r < 100) & (g < 100)       # blue
    return pupil, iris


def measure_mask(path_or_bgr, min_area=15):
    """Full per-frame measurement of one mask image. Returns a row dict without
    the `frame` key, or None if the image could not be read.

    This is the single implementation used by both the batch driver (main) and
    the resumable chunked driver.
    """
    bgr = (cv2.imread(path_or_bgr, cv2.IMREAD_COLOR)
           if isinstance(path_or_bgr, str) else path_or_bgr)
    if bgr is None:
        return None

    pupil, iris = class_masks(bgr)
    pm = fit_region(pupil, min_area)
    # The iris is measured on the FULL disc: the blue annulus plus the red pupil
    # that sits inside it. Measuring blue alone under-reports the iris, because
    # the pupil is a hole in that region.
    im = fit_iris(iris | pupil, pm, min_area)

    row = {}
    if pm:
        row.update(pupil_found=1, pupil_x=pm["x"], pupil_y=pm["y"],
                   pupil_major=pm["major"], pupil_minor=pm["minor"],
                   pupil_angle=pm["angle"], pupil_diam=pm["diam"])
    else:
        row.update(pupil_found=0, pupil_x=np.nan, pupil_y=np.nan,
                   pupil_major=np.nan, pupil_minor=np.nan,
                   pupil_angle=np.nan, pupil_diam=np.nan)
    if im:
        row.update(iris_found=1, iris_x=im["x"], iris_y=im["y"],
                   iris_major=im["major"], iris_minor=im["minor"],
                   iris_diam=im["diam"], iris_area_diam=im["area_diam"])
    else:
        row.update(iris_found=0, iris_x=np.nan, iris_y=np.nan,
                   iris_major=np.nan, iris_minor=np.nan, iris_diam=np.nan,
                   iris_area_diam=np.nan)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="folder of RITnet mask PNGs")
    ap.add_argument("--out", default="ritnet_metrics.csv", help="output CSV path")
    ap.add_argument("--min-area", type=int, default=15,
                    help="ignore regions smaller than this many pixels")
    args = ap.parse_args()

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(args.masks, e))
        files += glob.glob(os.path.join(args.masks, e.upper()))
    files = sorted(set(files), key=_frame_index)
    if not files:
        print("No mask images found in", args.masks)
        return
    print(f"Found {len(files)} masks")

    rows = []
    for f in files:
        row = measure_mask(f, args.min_area)
        if row is None:
            continue
        row["frame"] = _frame_index(f)
        rows.append(row)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    found = sum(r["pupil_found"] for r in rows)
    print(f"Wrote {args.out}: {len(rows)} frames, pupil found in {found}")


if __name__ == "__main__":
    main()