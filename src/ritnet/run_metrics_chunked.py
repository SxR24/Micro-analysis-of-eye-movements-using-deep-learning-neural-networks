#!/usr/bin/env python3
"""
run_metrics_chunked.py
Resumable driver for ritnet_metrices.py.

Processes a mask folder in bounded chunks, appending to the output CSV and
recording progress, so a long run can be spread over several invocations
(useful when the calling shell has a wall-clock limit). Safe to re-run: it
picks up from the last frame already written.

Usage:
    python run_metrics_chunked.py --masks ../out_8_full/mask \
        --out ../ritnet_8_new.csv --seconds 35
"""
import os, sys, csv, time, argparse, importlib.util
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "rm", os.path.join(HERE, "ritnet_metrices.py"))
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)

COLS = ["frame", "pupil_found", "pupil_x", "pupil_y", "pupil_major",
        "pupil_minor", "pupil_angle", "pupil_diam", "iris_found",
        "iris_x", "iris_y", "iris_major", "iris_minor", "iris_diam",
        "iris_area_diam"]


def measure(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    pupil = ((r > 128) & (g < 100) & (b < 100)).astype(np.uint8)
    iris = ((b > 128) & (r < 100) & (g < 100)).astype(np.uint8)

    pm = rm.fit_region(pupil)
    im = rm.fit_iris((iris | pupil).astype(np.uint8), pm)

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
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, required=True,
                    help="total number of frames expected")
    ap.add_argument("--seconds", type=float, default=35.0,
                    help="stop cleanly after this much wall time")
    ap.add_argument("--pattern", default="frame_%06d.png")
    args = ap.parse_args()

    # resume point: how many rows are already written
    start = 0
    if os.path.exists(args.out):
        with open(args.out) as fh:
            start = max(0, sum(1 for _ in fh) - 1)
    else:
        with open(args.out, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=COLS).writeheader()

    t0 = time.time()
    done = 0
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        for i in range(start, args.n_frames):
            if time.time() - t0 > args.seconds:
                break
            p = os.path.join(args.masks, args.pattern % i)
            row = measure(p)
            if row is None:
                row = {c: np.nan for c in COLS}
                row["pupil_found"] = 0
                row["iris_found"] = 0
            row["frame"] = i
            w.writerow(row)
            done += 1
        fh.flush()

    end = start + done
    print("processed %d frames (%d -> %d) of %d  [%.0f%%]  in %.1fs"
          % (done, start, end, args.n_frames,
             100.0 * end / args.n_frames, time.time() - t0))
    print("DONE" if end >= args.n_frames else "MORE")


if __name__ == "__main__":
    main()
