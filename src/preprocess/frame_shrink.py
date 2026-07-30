"""
frames_shrink.py
Shrink the eye inside the 640x400 frame so it occupies a smaller fraction,
mimicking OpenEDS framing (eye small, lots of dark surround). RITnet segments
poorly when the eye fills the whole frame; this gives it the proportions it
was trained on.

--fill controls how much of the frame the eye occupies (0.5 = half).

Usage:
    python frames_shrink.py 3.avi --out frames_3s --max-seconds 20 --fill 0.5
Try --fill 0.5, 0.6, 0.4 and see which segments best.
"""
import os, sys, json, argparse
import cv2, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--fill", type=float, default=0.5,
                    help="fraction of frame the eye occupies (0.4-0.6 typical)")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit("not found: " + args.video)
    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("cannot open: " + args.video)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    maxf = int(args.max_seconds * fps) if (args.max_seconds and fps) else None

    # scale the eye to occupy `fill` of the frame, centred, rest black
    scale = args.fill * min(args.width / ow, args.height / oh)
    nw, nh = int(round(ow * scale)), int(round(oh * scale))
    px, py = (args.width - nw) // 2, (args.height - nh) // 2
    print("eye %dx%d -> %dx%d (%.0f%% fill), centred in %dx%d"
          % (ow, oh, nw, nh, args.fill * 100, args.width, args.height))

    fidx = out_id = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if maxf and fidx >= maxf:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((args.height, args.width), np.uint8)
        canvas[py:py+nh, px:px+nw] = small
        cv2.imwrite(os.path.join(args.out, "frame_%06d.png" % fidx), canvas)
        out_id += 1
        fidx += 1
    cap.release()

    # inner_width/inner_height record the EXACT size of the scaled image inside
    # the canvas. pad_x = (W - nw) // 2, so when (W - nw) is odd the right pad is
    # one pixel wider than the left; anything that inverts the letterbox by
    # assuming symmetric padding is then off by a pixel. Recording nw/nh removes
    # the guesswork downstream (see ocular.iris_to_original).
    meta = dict(video=os.path.abspath(args.video), original_width=ow,
                original_height=oh, mode="shrink", fill=args.fill,
                scale=scale, pad_x=px, pad_y=py,
                inner_width=nw, inner_height=nh,
                canvas_width=args.width, canvas_height=args.height,
                fps=fps, n_frames_saved=out_id)
    with open(os.path.join(args.out, "_frames_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved %d frames. map mask->orig: x=(mx-%d)/%.4f y=(my-%d)/%.4f"
          % (out_id, px, scale, py, scale))


if __name__ == "__main__":
    main()