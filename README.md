# Ocular Pipeline

Gaze, pupil and **ocular torsion** from eye video, by combining two independent
measurement paths over the same footage:

| | method | measures |
|---|---|---|
| **RITnet** | deep-learning segmentation (DenseNet2D) | pupil and iris geometry, gaze, blink |
| **Irisometry** | classical Shi-Tomasi + Lucas-Kanade feature tracking | ocular torsion — rotation about the line of sight |

Torsion is invisible to segmentation (a rotating iris occupies the same pixels),
and the classical tracker needs to be told where the iris is. Each supplies what
the other lacks. The two are fused per frame into a single table.

MSc Bioinformatics & Computer Science, University of Leicester.

---

## Quick start

```bash
pip install -r requirements.txt

# reproduce video 8 end to end (~15 min)
run_video8.bat            # Windows
```

Then inspect the result:

```bash
python src/review/live_view.py        # video playback with live tracking overlay
streamlit run src/review/app.py       # frame-by-frame review with signal graphs
```

Run every command **from the project root** — all paths are relative to it.

---

## Layout

```
├── src/
│   ├── preprocess/     video -> frames, with the geometry metadata
│   ├── ritnet/         segmentation, geometry metrics, AOI handoff
│   ├── irisometry/     torsion tracking, merge, run comparison
│   ├── analysis/       Listing's Law test
│   └── review/         Streamlit app + OpenCV playback viewer
├── models/
│   ├── best_model.pkl  live RITnet checkpoint
│   └── old.pkl         superseded
├── data/               (git-ignored — reproducible, ~27 GB)
│   ├── raw/            source videos 1–8.avi
│   └── video_8/
│       ├── frames/         640×400 letterboxed PNGs + _frames_meta.json
│       ├── masks/          4-class segmentation masks  ← the key artefact
│       ├── overlays/       masks painted on frames (visualisation only)
│       ├── ritnet_8.csv    pupil/iris geometry, mask space
│       ├── ocular_8.csv    torsion, original space
│       ├── combined_8.csv  ← the deliverable
│       ├── features_8.npz  raw per-feature trajectories
│       ├── baseline_*/     preserved earlier runs, for before/after comparison
│       └── analysis/       Listing's Law report and figures
└── _archive/           preprocessing trials, backups (git-ignored)
```

---

## Coordinate spaces — read this before changing anything

The single easiest thing to get wrong.

RITnet was trained on OpenEDS at 640×400, where the eye occupies a modest
fraction of the frame with periocular context around it. Close-up footage that
fills the frame edge-to-edge is **out of domain and produces garbage masks**.
`frame_shrink.py` therefore scales each frame down and **centre-pads it with
black bars** rather than stretching it — stretching would distort the eye and
degrade segmentation just as badly.

So segmentation output lives in **padded 640×400 mask space**, while torsion is
tracked on the **original video**. `_frames_meta.json` records the `scale` and
`pad_x`/`pad_y` needed to invert it:

```
original = (mask_coord − pad) / scale
```

`merge.py` applies this, so **every column in `combined_*.csv` is in original
video coordinates.**

The practical consequence: the review tools must be pointed at the *original*
video. A padded or shrunk copy puts every marker in the wrong place while still
looking plausible — this cost a debugging session once. Both `app.py` and
`live_view.py` now verify resolution against `_frames_meta.json` and refuse to
run rather than mislead you.

---

## The workflow

Every video is a different resolution, so each needs its own extraction,
metadata and AOI. Substitute `<n>` throughout.

### 1 — Extract frames

```bash
python src/preprocess/frame_shrink.py data/raw/<n>.avi --out data/video_<n>/frames --fill 0.85
```

Writes `_frames_meta.json` beside the PNGs. `--fill 0.85` leaves margin so the
eye is not flush to the frame edge; chosen empirically (the trials are in
`_archive/experiments/`).

### 2 — Segment

```bash
cd src/ritnet
python ritnet_run.py --frames ../../data/video_<n>/frames --out ../../data/video_<n> --load ../../models/best_model.pkl
cd ../..
```

Produces `masks/` (4-class label images) and `overlays/`. The **masks** are the
valuable output — overlays are ~5 GB per video and purely for looking at.

### 3 — Measure geometry

```bash
python src/ritnet/ritnet_metrices.py --masks data/video_<n>/masks --out data/video_<n>/ritnet_<n>.csv
```

For long videos `run_metrics_chunked.py` does the same job but is resumable.

### 4 — Derive the AOI

```bash
python src/ritnet/get_aoi.py --ritnet data/video_<n>/ritnet_<n>.csv --meta data/video_<n>/frames/_frames_meta.json
```

Prints `--aoi cx,cy,r` in original coordinates. This replaces the manual limbus
selection the original irisometry implementation requires.

### 5 — Track torsion

```bash
python src/irisometry/ocular.py data/raw/<n>.avi \
    --aoi <cx,cy,r> \
    --ritnet data/video_<n>/ritnet_<n>.csv \
    --meta   data/video_<n>/frames/_frames_meta.json \
    --masks  data/video_<n>/masks \
    --out    data/video_<n>
```

All four handoff flags matter — see *Why the flags matter* below. Confirm these
lines appear at startup:

```
RITnet coords -> original space: ...     --meta took effect
Feature gating: RITnet iris mask         lashes excluded
Blink recovery window: 21 frames         420 ms
Capturing raw feature trajectories       .npz export on
```

### 6 — Merge

```bash
python src/irisometry/merge.py --ritnet data/video_<n>/ritnet_<n>.csv \
    --ocular data/video_<n>/ocular_<n>.csv \
    --meta   data/video_<n>/frames/_frames_meta.json \
    --out    data/video_<n>/combined_<n>.csv
```

### 7 — Review and analyse

```bash
python src/review/live_view.py                       # playback + overlay
streamlit run src/review/app.py                      # frame-by-frame review
python src/analysis/analyse.py --csv data/video_<n>/combined_<n>.csv --out data/video_<n>/analysis
```

Restart Streamlit rather than pressing **R** after regenerating a CSV —
`@st.cache_data` keys on the file path, not its contents, so a live instance
serves stale data.

---

## Why the flags matter

Each was added after measuring a specific failure. Numbers are from video 8
(28,236 frames).

**`--masks` — restrict features to iris tissue.** A circular AOI large enough to
reach the limbus necessarily contains eyelid and lashes, because the palpebral
fissure is shorter vertically than the iris is wide. Lashes give far stronger
corners than iris texture, so they dominate detection. Measured with a circle
alone, only **33.7%** of tracked features were on iris; 53.6% were on lid and
lashes. That matters twice over: lashes move with the **eyelid**, not the
eyeball; and the torsion step re-centres on the feature centroid to cancel gaze
translation, so a lid-dominated feature set makes that centroid track the lid.
Mask gating brings feature purity to **100%** and cuts drift by 61%.

*The original irisometry implementation lists this as an unsolved TODO
("Implement automated lid detection! Remove features at eye lids"). Segmentation
solves it directly.*

**`--ritnet` — take the blink signal from segmentation.** Classical irisometry
infers blinks indirectly, from feature-count collapse. RITnet observes the pupil
directly. The two agreed on only 571 of ~1,400 blink frames; **834 closed-eye
frames** were being tracked as valid. Blink recall went 41% → 100%.

**`--meta` — transform coordinates.** Without it, RITnet's mask-space pupil
coordinates are used unchanged in original space. Verified correct against
`merge.py` to 0.003 px.

**Blink recovery window.** Derived as `ceil(0.4 × fps)` ≈ 400 ms, the approximate
duration of a blink. Previously hardcoded to 5 frames — only 100 ms at 50 fps, so
tracking resumed while the lid still covered the iris.

---

## Interpreting the output

### Columns in `combined_<n>.csv`

| column | source | notes |
|---|---|---|
| `frame`, `time` | — | |
| `pupil_x/y`, `pupil_diam` | RITnet | original coordinates |
| `iris_x/y`, `iris_diam` | RITnet | original coordinates |
| `pupil_found` | RITnet | quality flag |
| `torsion_deg` | irisometry | all features pooled |
| `torsion_outer_deg` | irisometry | **use this one** |
| `torsion_inner_deg` | irisometry | noisier — see below |
| `n_features`, `blink` | irisometry | |

Filter to `blink != 1 & pupil_found == 1` before any analysis. Blink frames still
carry a pupil estimate, but it is a lid artefact — a ~13 px "pupil" against a
~203 px median.

**Prefer `torsion_outer_deg`.** Torsion is carried by iris crypts and furrows,
concentrated in the outer iris. The inner ring sits nearer the pupil boundary,
whose corners move with dilation rather than rotation, and measures roughly twice
the noise.

### Two traps when comparing runs

**Torsion is re-referenced to zero after every blink.** Any statistic pooled
across the whole recording mixes real variation with those reference resets, and
runs that flag different numbers of blinks become incomparable. Compute
everything **within segments** — `compare_runs.py` and `analyse.py` both do.

**Judge tracking changes by drift, not smoothness.** Weak corners slide
*smoothly* under Lucas-Kanade: they lower frame-to-frame jitter while
accumulating error. Lowering the corner-quality threshold on this footage
improved jitter (0.167 → 0.148) while drift per segment went from ~1° to **15°**.
`compare_runs.py` reports both for exactly this reason.

---

## Results so far (video 8)

Cumulative effect of the corrections, measured within segments:

| metric | before | after |
|---|---|---|
| feature purity (on iris) | 33.7% | **100%** |
| blink recall vs RITnet | 41% | **100%** |
| within-segment SD | 1.073° | **0.558°** |
| drift per segment | 2.765° | **1.038°** |

**Listing's Law is not testable from video 8.** The recording is predominantly
vertical: horizontal gaze SD is 1.05°, and 80% of frames fall in two opposing
quadrants (r = +0.55). The product term θ_h·θ_v is therefore nearly a rescaled
vertical main effect. After cleaning, its confidence interval includes zero
while vertical gaze alone explains twice the variance — the earlier apparent
effect was contamination. Testing Listing's Law needs gaze sampled across a 2D
target grid, typically ±15–20° in both axes. `analyse.py` prints this power
diagnostic automatically.

---

## Status

Only **video 8** is processed end to end. Videos 1–7 are raw, and all eight are
different resolutions (600×516 to 908×620), so each needs its own extraction and
AOI.

Known issue: blink detection has occasional false negatives — a small number of
closed-eye frames pass `blink == 0, pupil_found == 1` while reporting an
implausibly small pupil. A `pupil_diam < 0.7 × median` rule would catch them.

Disk note: overlays are ~5 GB per video and are visualisation only. Skip them for
batch processing unless you intend to review each frame.
