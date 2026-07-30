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
import os, csv, time, argparse, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "rm", os.path.join(HERE, "ritnet_metrices.py"))
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)

# Columns and measurement both come from ritnet_metrices, so a batch run and a
# resumed run cannot produce different numbers from the same masks.
COLS = rm.COLS
measure = rm.measure_mask


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
