"""
extract_frames.py
Extract video frames as grayscale PNGs for RITnet inference.

Saves EVERY frame at 640x400 (RITnet/OpenEDS resolution) for segmentation,
and records the original resolution + scale factors to a small JSON sidecar so
that RITnet mask coordinates (640x400) can later be mapped back to the original
video space used by the irisometry/torsion pipeline.

Usage:
    python extract_frames.py 1.avi --out frames_1
    python extract_frames.py 1.avi --out frames_1 --every 1      # every frame (default)
    python extract_frames.py 1.avi --out frames_1 --max-seconds 20   # quick test
"""
import os, sys, json, argparse
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to input video (.avi/.mp4/...)")
    ap.add_argument("--out", required=True, help="output folder for frames")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--every", type=int, default=1,
                    help="save every Nth frame (1 = all; KEEP AT 1 for torsion)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="only extract the first N seconds (for quick tests)")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit("Video not found: " + args.video)
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("Could not open video: " + args.video)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(args.max_seconds * fps) if (args.max_seconds and fps) else None

    print("Video: %s  (%dx%d @ %.2f fps, %d frames)"
          % (args.video, orig_w, orig_h, fps, total))
    print("Extracting every %d frame -> %dx%d PNG in %s"
          % (args.every, args.width, args.height, args.out))

    frame_idx = 0      # index in the ORIGINAL video
    out_id = 0         # index of the saved frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        if frame_idx % args.every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (args.width, args.height),
                              interpolation=cv2.INTER_AREA)
            # IMPORTANT: name encodes the ORIGINAL frame index so the merge can
            # align RITnet output to the exact video frame, even if --every>1.
            cv2.imwrite(os.path.join(args.out, "frame_%06d.png" % frame_idx), gray)
            out_id += 1
        frame_idx += 1
    cap.release()

    # sidecar: how to map 640x400 mask coords back to original video coords
    meta = dict(
        video=os.path.abspath(args.video),
        original_width=orig_w, original_height=orig_h,
        proc_width=args.width, proc_height=args.height,
        fps=fps, every=args.every,
        scale_x=orig_w / float(args.width),
        scale_y=orig_h / float(args.height),
        n_frames_saved=out_id,
    )
    with open(os.path.join(args.out, "_frames_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved %d frames." % out_id)
    print("Coordinate mapping (mask 640x400 -> original): "
          "x*%.4f, y*%.4f  (saved to _frames_meta.json)"
          % (meta["scale_x"], meta["scale_y"]))


if __name__ == "__main__":
    main()