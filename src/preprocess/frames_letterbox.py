"""
frames_letterbox.py
Extract video frames as 640x400 grayscale PNGs for RITnet, WITHOUT distorting
the eye. Instead of stretching (which squashes the aspect ratio and confuses a
model trained on undistorted OpenEDS images), this resizes preserving aspect
ratio and pads the remainder -- so a round iris stays round.

Records the resize scale AND the pad offsets so RITnet mask coordinates can be
mapped back to original video coordinates exactly.

Usage:
    python frames_letterbox.py 3.avi --out frames_3
    python frames_letterbox.py 3.avi --out frames_3 --max-seconds 20
"""
import os, sys, json, argparse
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--pad-value", type=int, default=0,
                    help="grayscale value for padding (0=black)")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit("Video not found: " + args.video)
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("Could not open: " + args.video)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(args.max_seconds * fps) if (args.max_seconds and fps) else None

    # single uniform scale that fits the WHOLE eye inside WxH (no distortion)
    scale = min(args.width / ow, args.height / oh)
    new_w, new_h = int(round(ow * scale)), int(round(oh * scale))
    pad_x = (args.width - new_w) // 2
    pad_y = (args.height - new_h) // 2

    print("Video %s (%dx%d @ %.2f fps, %d frames)" % (args.video, ow, oh, fps, total))
    print("Aspect-preserving resize: scale %.4f -> %dx%d, pad (%d,%d)"
          % (scale, new_w, new_h, pad_x, pad_y))

    fidx = 0
    out_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and fidx >= max_frames:
            break
        if fidx % args.every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.full((args.height, args.width), args.pad_value, np.uint8)
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            cv2.imwrite(os.path.join(args.out, "frame_%06d.png" % fidx), canvas)
            out_id += 1
        fidx += 1
    cap.release()

    # To map a mask coordinate (mx,my) in 640x400 back to original video coords:
    #   orig_x = (mx - pad_x) / scale
    #   orig_y = (my - pad_y) / scale
    meta = dict(
        video=os.path.abspath(args.video),
        original_width=ow, original_height=oh,
        proc_width=args.width, proc_height=args.height,
        fps=fps, every=args.every,
        mode="letterbox",
        scale=scale, pad_x=pad_x, pad_y=pad_y,
        n_frames_saved=out_id,
    )
    with open(os.path.join(args.out, "_frames_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved %d frames (letterboxed, undistorted)." % out_id)
    print("Map mask->original:  orig_x=(mx-%d)/%.4f,  orig_y=(my-%d)/%.4f"
          % (pad_x, scale, pad_y, scale))


if __name__ == "__main__":
    main()