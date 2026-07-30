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

### 8 — Check the measurement is real

```bash
python src/analysis/reliability.py --features data/video_<n>/features_<n>.npz \
    --baseline data/video_<n>/baseline_lkchain/features_<n>_LKCHAIN.npz
```

Every other diagnostic here is internal — within-segment SD, drift, jitter — and
none of them can tell a trace that is tracking the eye from one that is smoothly
wrong. This one can, without ground truth: split the tracked features into two
random halves, compute torsion independently from each half of the *same* frame,
and correlate. A real common rotation shows up in both halves; per-feature noise
does not.

Run it after **any** change to tracking. It is the only check that fails when a
change makes the trace prettier and the measurement worse — which is exactly the
failure mode this pipeline is prone to.

Reliability is an upper bound, not an estimate: anything common to both halves
inflates it, including iris deformation and mask-boundary changes, none of which
are eye rotation.

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

## How torsion is tracked

### Reference anchoring, not frame-to-frame chaining

The obvious way to track features through a segment is to chain Lucas-Kanade:
ref → f1 → f2 → … Each hop adds a little correspondence error, and the errors
**accumulate**. Measured on video 8, the median per-feature residual to a
best-fit rigid rotation grew monotonically from 1.98 px at the start of a
segment to 10.34 px at the end — a 2.8× growth. At a feature radius of ~150 px,
10 px of residual is nearly 4° of angular scatter per feature. The feature set
progressively stopped behaving like a rotating rigid body, purely as a tracking
artefact.

Each frame is now tracked **directly from the reference frame's image**, with
the chained position used only as an initial guess for the search. The intensity
anchor is always the reference, so slide cannot accumulate. The forward-backward
check runs against the reference too, which is stricter: a feature that has slid
away over fifty frames fails to map back onto its reference position and is
caught.

Two consequences to be aware of:

- The tolerance must be **looser** (`fb_max_px = 2.0`, four pyramid levels)
  because the displacement being measured is the full ref→current one, not a
  single frame's. At 1.0 px legitimate features were retired within a few frames
  and segments collapsed to a median of 4.5 frames.
- Features are retired after **three consecutive** failures rather than one. A
  single motion-blurred or saccadic frame makes LK fail transiently on perfectly
  good features; retiring them immediately emptied the feature set.

### Re-seeds are not blinks

When the surviving feature set falls below its floor, tracking starts a new
reference. That is a **re-seed**, not a blink: the eye has not closed, so it does
not set the `blink` flag and does not burn the blink recovery window. Conflating
the two inflated the blink count and discarded good frames.

This is why `combined_*.csv` carries an explicit `seg` column. Inferring
segments as a cumsum over `blink` silently merges the segments either side of a
re-seed, pooling two different zero references into one.

### The estimator

Torsion is the least-squares rigid rotation from the reference positions to the
current ones (orthogonal Procrustes), with the centroids of the *same surviving
subset* subtracted from both to cancel gaze translation, then iteratively
reweighted with a Tukey biweight.

The previous estimator took the circular median of per-feature angle changes.
That weights every feature equally, but a feature at radius *r* pins the angle
to a precision of σ/r — so the median discards the fact that outer features are
far more informative. Procrustes is the maximum-likelihood rotation under
isotropic Gaussian position noise and is implicitly r²-weighted, which is the
correct weighting, and it yields the fit residual as a free per-frame quality
channel.

Both are validated in `--selftest` against synthetic rotations, under
translation, asymmetric feature dropout, and up to 30% grossly mis-tracked
features.

---

## Interpreting the output

### Columns in `combined_<n>.csv`

| column | source | notes |
|---|---|---|
| `frame`, `time` | — | |
| `pupil_x/y`, `pupil_diam` | RITnet | original coordinates |
| `iris_x/y`, `iris_diam` | RITnet | original coordinates; `iris_y` is *copied from* `pupil_y` (they are concentric and the lids clip the iris vertically) — so it is not an independent vertical measurement |
| `pupil_found` | RITnet | quality flag; also gated on `pupil_diam ≥ 0.7 × median` |
| `torsion_deg` | irisometry | all features pooled |
| `torsion_outer_deg` | irisometry | outer annulus |
| `torsion_inner_deg` | irisometry | inner annulus — **diagnostic only**, see below |
| `torsion_resid_px` | irisometry | median per-feature distance to the fitted rigid rotation. **The quality channel** — a smooth trace with a large residual is smoothly wrong |
| `torsion_n_used` | irisometry | features surviving the robust reweighting |
| `seg` | irisometry | segment id; `-1` = not tracked. Use this rather than a cumsum over `blink` |
| `n_features`, `blink` | irisometry | |

Filter to `blink != 1 & pupil_found == 1 & seg >= 0` before any analysis. Blink
frames still carry a pupil estimate, but it is a lid artefact — a ~13 px "pupil"
against a ~203 px median. `merge.py` now demotes those to `pupil_found = 0`
automatically.

**Use `torsion_deg` (all features).** Torsion is carried by iris crypts and
furrows, concentrated in the outer iris, so `torsion_outer_deg` was previously
recommended. Treat the inner/outer split as a **diagnostic**, not a choice of
estimator: on video 8 the two annuli correlate at r = −0.008 within segment.
Two concentric annuli of the same rotating disc should agree strongly. That they
do not is a statement about the noise floor, not about which ring to prefer —
the angular precision of a feature scales with its radius, so the inner ring is
close to pure noise. If the two ever *do* agree, that is real evidence the
measurement is working.

### Trust `torsion_resid_px`

The residual is the number to look at before believing any torsion trace. It is
the median distance between where each feature actually is and where a single
rigid rotation says it should be. Low and **flat across a segment** means the
tracked set is behaving like a rotating eye. Rising across a segment means
Lucas-Kanade slide is accumulating and the trace is drifting regardless of how
clean it looks.

### Three traps when comparing runs

**Torsion is re-referenced to zero at every segment boundary.** Any statistic
pooled across the whole recording mixes real variation with those reference
resets. Compute everything **within segments** — `compare_runs.py`,
`analyse.py` and `reliability.py` all do.

**Segment ≠ inter-blink span.** Use the `seg` column, never a cumsum over
`blink`. Re-seeds start a new zero reference without a blink, so the cumsum
lumps several zero-referenced spans together. This once made `compare_runs.py`
report a drift of exactly `0.000°`, because the first and last frame of a lumped
span were both seed frames sitting at 0.0.

**Judge tracking changes by drift and reliability, not smoothness.** Weak
corners slide *smoothly* under Lucas-Kanade: they lower frame-to-frame jitter
while accumulating error. Lowering the corner-quality threshold on this footage
improved jitter (0.167 → 0.148) while drift per segment went from ~1° to **15°**.
Jitter is the metric most likely to mislead; split-half reliability is the one
that a smoothing change cannot improve spuriously.

---

## Results (video 8, 28,236 frames)

### Measurement quality

Split-half reliability over the tracked features — the check that distinguishes
a trace tracking the eye from one that is smoothly wrong:

| metric | chained LK + circular median | reference-anchored + Procrustes |
|---|---|---|
| **split-half r** | +0.321 | **+0.623** |
| **reliability (Spearman-Brown)** | 0.486 | **0.768** |
| torsion SD (signal + noise) | 0.365° = 0.254 + 0.262 | **0.219° = 0.192 + 0.106** |
| **noise SD** | 0.262° | **0.106°** (−60%) |
| rigid-fit residual | 6.39 px | **1.50 px** |
| residual across segment | 3.2 → 9.0 px (2.84×) | **1.1 → 1.7 px (1.51×)** |

Reliability is an upper bound, not an estimate: anything common to both halves
inflates it. 0.768 means roughly three-quarters of the variance in the reported
torsion is shared between independent halves of the iris — up from under half.

### Tracking behaviour

Against the previous mask-gated run, within segments:

| metric | before | after |
|---|---|---|
| feature purity (on iris) | 33.7% | 100% |
| blink recall vs RITnet | 41% | 100% |
| frame-to-frame jitter | 0.318° | **0.060°** |
| within-segment SD | 0.558° | **0.170°** |
| drift per segment | 1.038° | **0.352°** |
| segments ≥ 25 frames | 78 | 120 |
| features per frame (median) | 145 | 89 |

Fewer features, and better ones: the count falls because correspondences are
retired once genuinely lost instead of being silently readmitted.

### Listing's Law

Still **not established** from video 8, but the position is much better than it
was, and the earlier report in this repo was simply wrong (it was generated
before the input CSV was regenerated and quoted a product-term slope 23× the
true value — hence the provenance stamp `analyse.py` now writes).

```
theta_h * theta_v    slope -0.00394   95% CI [-0.01946, +0.00236]
Listing predicts           -0.00873
```

The sign is now correct and the magnitude is the right order. The CI contains
Listing's value — but it also contains zero, so this is consistent with the
prediction rather than evidence for it.

**The limit is gaze coverage, not measurement noise.** Horizontal gaze SD is
1.06°, so the product term is nearly a rescaled vertical main effect. The CI
half-width is 0.0109; resolving a slope of 0.0087 needs it below ~0.0044, and
half-width scales as 1/SD(θ_h·θ_v). SD(product) is currently 5.73 deg², so it
needs to reach ~14 deg² — about **3.8° SD in each axis, uncorrelated**, i.e. a
target grid spanning roughly **±8–10°**.

That is one recording session with a nine-point grid on a screen at 60 cm, and
it is now the limiting factor rather than the measurement noise.

---

## Status

Only **video 8** is processed end to end. Videos 1–7 are raw, and all eight are
different resolutions (600×516 to 908×620), so each needs its own extraction and
AOI.

Blink false negatives are handled: `merge.py` demotes any frame reporting a
pupil under `0.7 ×` the recording's median diameter to `pupil_found = 0` (372
frames on video 8). The threshold is relative, so it needs no per-video tuning.

**Open — the AOI is off-centre.** The locked AOI `449,380,197` was seeded from
the first 30 valid frames, and sits ~27 px above the recording's median iris
centre; `get_aoi.py` now estimates `454,407,191` from the whole recording. The
results above deliberately keep the old value so that they isolate the tracking
change. Re-running with the corrected AOI is the next single-variable test.

**Open — residual still grows 1.51× across a segment.** Down from 2.84×, but not
flat. Reference anchoring removed the accumulating component; what remains is
most likely the changing composition of the surviving feature set, and possibly
genuine iris deformation (scale correlates with pupil diameter at r = +0.47).

**Open — gaze has no slip reference.** `analyse.py` derives gaze from absolute
pupil position, so any camera or headset movement enters directly as gaze. The
usual defence is a differential signal, but `iris_y` is *copied from* `pupil_y`
in `ritnet_metrices.py`, so vertical differential gaze is not currently
available. Fitting a circle to the left and right limbus arcs only — the
unoccluded ones — would give an independent iris centre and a slip-invariant
gaze vector.

Disk note: overlays are ~5 GB per video and are visualisation only. Skip them for
batch processing unless you intend to review each frame.
