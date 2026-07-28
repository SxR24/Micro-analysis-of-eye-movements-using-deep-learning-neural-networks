import os, glob, argparse
import numpy as np
import cv2
import torch
from PIL import Image
from densenet import DenseNet2D   # model class from the repo

# ---- label colors (4 classes: background, sclera, iris, pupil) ----
PALETTE = np.array([
    [0,   0,   0  ],   # 0 background
    [0,   255, 0  ],   # 1 sclera  (green)
    [0,   0,   255],   # 2 iris    (blue)
    [255, 0,   0  ],   # 3 pupil   (red)
], dtype=np.uint8)


def preprocess(gray):
    # gray: uint8 HxW grayscale numpy array
    g = cv2.resize(gray, (640, 400), interpolation=cv2.INTER_AREA)
    # gamma correction (table from the repo)
    table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
    g = cv2.LUT(g.astype(np.uint8), table.astype(np.uint8))
    # CLAHE -- same params the authors used
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    g = clahe.apply(g)
    # normalize the way torchvision-style transforms do in the repo
    t = torch.from_numpy(g).float().unsqueeze(0) / 255.0   # 1xHxW
    t = (t - 0.5) / 0.5
    return t


def clean_pupil_only(pred):
    """Remove stray red (pupil) blobs ONLY. Sclera and iris are left
    exactly as the model predicted -- nothing about them is changed.

    For the pupil: keep the single red blob that sits closest to the
    iris centre (the true pupil), discarding edge/corner artifacts.
    Falls back to the largest red blob if no iris is present.
    """
    cleaned = pred.copy()   # keep sclera (1) and iris (2) untouched

    pupil = (pred == 3).astype(np.uint8)
    if pupil.sum() == 0:
        return cleaned

    # iris centroid, to anchor the real pupil
    iris = (pred == 2).astype(np.uint8)
    iris_centroid = None
    if iris.sum() > 0:
        m = cv2.moments(iris, binaryImage=True)
        if m["m00"] != 0:
            iris_centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])

    n, labels, stats, cents = cv2.connectedComponentsWithStats(pupil, 8)
    if n > 1:
        candidates = list(range(1, n))
        if iris_centroid is not None:
            def dist(i):
                dx = cents[i][0] - iris_centroid[0]
                dy = cents[i][1] - iris_centroid[1]
                return dx * dx + dy * dy
            best = min(candidates, key=dist)
        else:
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        # erase all red, then paint back only the chosen pupil blob
        cleaned[pred == 3] = 0
        cleaned[labels == best] = 3

    return cleaned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="folder of eye images")
    ap.add_argument("--output", required=True, help="folder to write results")
    ap.add_argument("--load",   default="best_model.pkl")
    ap.add_argument("--bs",     type=int, default=8)
    ap.add_argument("--no-clean", action="store_true",
                    help="disable pupil cleanup (save raw model output)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.output, "mask"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "overlay"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DenseNet2D(out_channels=4)
    state = torch.load(args.load, map_location=device)
    # the .pkl may be a full model or a state_dict -- handle both
    try:
        model.load_state_dict(state)
    except Exception:
        model = state
    model = model.to(device).eval()

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(args.input, e))
        files += glob.glob(os.path.join(args.input, e.upper()))
    files = sorted(set(files))
    print(f"Found {len(files)} images")

    with torch.no_grad():
        for i in range(0, len(files), args.bs):
            batch_files = files[i:i+args.bs]
            tensors, grays = [], []
            for f in batch_files:
                gray = np.array(Image.open(f).convert("L"))
                grays.append(cv2.resize(gray, (640, 400), interpolation=cv2.INTER_AREA))
                tensors.append(preprocess(gray))
            x = torch.stack(tensors).to(device)
            out = model(x)                       # N x 4 x H x W
            pred = torch.argmax(out, dim=1).cpu().numpy().astype(np.uint8)

            for f, p, gray in zip(batch_files, pred, grays):
                name = os.path.splitext(os.path.basename(f))[0]

                if not args.no_clean:
                    p = clean_pupil_only(p)      # remove stray RED blobs only

                color = PALETTE[p]                                  # HxWx3
                cv2.imwrite(os.path.join(args.output, "mask", name + ".png"),
                            cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
                base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                ov = cv2.addWeighted(base, 0.6,
                                     cv2.cvtColor(color, cv2.COLOR_RGB2BGR), 0.4, 0)
                cv2.imwrite(os.path.join(args.output, "overlay", name + ".png"), ov)
            print(f"  done {min(i+args.bs, len(files))}/{len(files)}")

    print("Finished. Masks + overlays in:", args.output)


if __name__ == "__main__":
    main()