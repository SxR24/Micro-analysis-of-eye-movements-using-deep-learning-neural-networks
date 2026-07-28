"""
app.py  -  Ocular Review
======================================================================
Streamlit review tool for the integrated gaze + ocular-torsion pipeline.

Two synchronised video panels and the full signal set, driven by one frame
scrubber, all reading from combined_<name>.csv:

  LEFT  panel : RITnet segmentation overlay (from out_*/overlay/<frame>.png)
  RIGHT panel : original eye video frame with the irisometry markers drawn
                live from the CSV (iris AOI, pupil centre, torsion readout)
  GRAPHS      : torsion, gaze (x & y), pupil diameter -- each with a playhead
                at the current frame; blink frames flagged.

Run from the PROJECT ROOT (not from this folder) so the relative paths resolve:
    streamlit run src/review/app.py

Defaults assume the standard layout:
    combined CSV     : data/video_8/combined_8.csv
    original video   : data/raw/8.avi          <- ORIGINAL, not a padded copy
    RITnet overlays  : data/video_8/overlays
    frames meta      : data/video_8/frames/_frames_meta.json

To review a different video, point all four at data/video_<n>/.
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import altair as alt

# ----------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------
st.set_page_config(page_title="Ocular Review", layout="wide",
                   initial_sidebar_state="expanded")

# Instrument aesthetic: deep slate ground, single cyan signal accent, amber for
# torsion (the hero measurement), mono type for the data readouts.
st.markdown("""
<style>
  :root {
    --bg:#0e1116; --panel:#161b22; --line:#222c38;
    --ink:#e6edf3; --mut:#7d8794;
    --torsion:#f5a623; --gaze:#4ec9b0; --pupil:#a98bdc; --accent:#39c5cf;
  }
  .stApp { background:var(--bg); color:var(--ink); }
  section[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--line); }
  h1,h2,h3,h4 { color:var(--ink); font-family:"DM Sans","Inter",system-ui,sans-serif; letter-spacing:-0.01em; }
  .ocr-title { font-size:1.4rem; font-weight:600; margin:0; }
  .ocr-sub { color:var(--mut); font-size:0.82rem; margin:0 0 0.4rem 0;
             font-family:"DM Mono","SFMono-Regular",monospace; letter-spacing:0.02em; }
  .ocr-panel-label { color:var(--mut); font-family:"DM Mono",monospace;
                     font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em;
                     margin-bottom:0.25rem; }
  .ocr-readout { font-family:"DM Mono","SFMono-Regular",monospace; font-size:0.9rem; }
  .ocr-chip { display:inline-block; padding:0.15rem 0.55rem; border-radius:3px;
              font-family:"DM Mono",monospace; font-size:0.78rem; margin-right:0.4rem; }
  .chip-track { background:rgba(57,197,207,0.12); color:var(--accent); border:1px solid rgba(57,197,207,0.3); }
  .chip-blink { background:rgba(245,84,84,0.14); color:#ff6b6b; border:1px solid rgba(245,84,84,0.35); }
  .stImage img { border:1px solid var(--line); border-radius:6px; }
  hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_combined(path):
    df = pd.read_csv(path)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # gaze relative to primary position (median) so centre = 0.
    # Median is taken over TRACKED frames only -- blinks put the pupil estimate
    # somewhere arbitrary, and letting those into the reference shifts the
    # whole gaze trace.
    if "pupil_x" in df:
        ok = tracked_mask(df)
        ref_x = df.loc[ok, "pupil_x"].median()
        ref_y = df.loc[ok, "pupil_y"].median()
        df["gaze_x"] = df["pupil_x"] - ref_x
        df["gaze_y"] = df["pupil_y"] - ref_y

    # Temporally smoothed iris for the overlay. The per-frame iris estimate is
    # sound but jitters by a pixel or two frame to frame; a short rolling median
    # steadies the drawn circle without lagging real eye movement (5 frames =
    # 100 ms at 50 fps). Raw columns are left untouched for analysis.
    for c in ("iris_x", "iris_y", "iris_diam"):
        if c in df:
            s = df[c].where(tracked_mask(df))
            df[c + "_s"] = s.rolling(5, center=True, min_periods=1).median()
    return df


def tracked_mask(df):
    """Frames where the eye is actually visible and tracked."""
    ok = pd.Series(True, index=df.index)
    if "blink" in df:
        ok &= df["blink"].fillna(0) != 1
    if "pupil_found" in df:
        ok &= df["pupil_found"].fillna(0) == 1
    return ok


def frame_aoi(df, fi, fallback):
    """Iris circle for THIS frame, falling back to the session median when the
    frame is a blink or the iris was not found."""
    r = df.iloc[fi]
    x = r.get("iris_x_s", np.nan)
    y = r.get("iris_y_s", np.nan)
    d = r.get("iris_diam_s", np.nan)
    if np.isnan(x) or np.isnan(y) or np.isnan(d) or d <= 0:
        return fallback
    return (float(x), float(y), float(d) / 2.0)


@st.cache_data(show_spinner=False)
def overlay_index(folder):
    """Map frame index -> overlay png path, from filenames like frame_000123.png."""
    idx = {}
    if folder and os.path.isdir(folder):
        for p in glob.glob(os.path.join(folder, "*.png")):
            digits = "".join(ch for ch in os.path.basename(p) if ch.isdigit())
            if digits:
                idx[int(digits)] = p
    return idx


@st.cache_resource(show_spinner=False)
def open_video(path):
    if path and os.path.isfile(path):
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            return cap
    return None


def video_size(cap):
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


@st.cache_data(show_spinner=False)
def expected_size(meta_path):
    """Coordinate space the CSV is in, per the frame-extraction sidecar.

    All CSV coordinates are in ORIGINAL video space, so the video loaded here
    must be the original -- not a letterboxed or shrunk derivative. Loading a
    640x400 padded copy while the CSV holds 908x620 coordinates silently
    misplaces every marker, which is easy to mistake for a tracking failure.
    """
    try:
        with open(meta_path) as f:
            m = json.load(f)
        return int(m["original_width"]), int(m["original_height"])
    except Exception:
        return None


def read_video_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)


def draw_irisometry(gray, row, aoi):
    """Draw the irisometry markers on a grayscale frame from one CSV row."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cx, cy, r = aoi
    is_blink = row.get("blink", 0) == 1
    # AOI (iris) circle - cyan. Dashed-looking thin stroke on blinks to signal
    # that it is the session fallback, not a live measurement.
    cv2.circle(vis, (int(cx), int(cy)), int(r), (207, 197, 57),
               1 if is_blink else 2)
    # pupil - amber dot + cross (BGR). Suppressed during blinks: the detector
    # still returns a centre there, but it is a lid artefact (a ~13 px "pupil"),
    # and drawing it invites reading a closed eye as a tracked one.
    if not is_blink and not np.isnan(row.get("pupil_x", np.nan)):
        px, py = int(row["pupil_x"]), int(row["pupil_y"])
        pr = row.get("pupil_diam", np.nan)
        pr = int(pr / 2) if not np.isnan(pr) else 6
        cv2.circle(vis, (px, py), max(3, pr), (35, 166, 245), 2)
        cv2.drawMarker(vis, (px, py), (35, 166, 245), cv2.MARKER_CROSS, 14, 1)
    # torsion readout
    tor = row.get("torsion_deg", np.nan)
    txt = "torsion --" if np.isnan(tor) else "torsion %+.2f deg" % tor
    if row.get("blink", 0) == 1:
        txt = "BLINK"
    cv2.putText(vis, txt, (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (35, 166, 245) if "BLINK" not in txt else (84, 84, 245), 1, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------
# Sidebar: data sources
# ----------------------------------------------------------------------
st.sidebar.markdown("### Data sources")
csv_path = st.sidebar.text_input("Combined CSV", "data/video_8/combined_8.csv")
vid_path = st.sidebar.text_input("Original video", "data/raw/8.avi")
ovl_path = st.sidebar.text_input("RITnet overlay folder", "data/video_8/overlays")
meta_path = st.sidebar.text_input("Frames meta (_frames_meta.json)",
                                  "data/video_8/frames/_frames_meta.json")
st.sidebar.caption(
    "Paths are relative to the project root. Launch from there:  "
    "`streamlit run src/review/app.py`")

if not os.path.isfile(csv_path):
    st.error("Combined CSV not found: `%s`. Set the correct path in the sidebar." % csv_path)
    st.stop()

df = load_combined(csv_path)
ovl = overlay_index(ovl_path)
cap = open_video(vid_path)

# Guard: the CSV holds ORIGINAL-video coordinates. If a letterboxed/shrunk copy
# is loaded instead, every marker lands in the wrong place while still looking
# plausible. Catch that here rather than letting it look like bad tracking.
_exp = expected_size(meta_path)
if cap is not None and _exp is not None:
    _got = video_size(cap)
    if _got != _exp:
        st.error(
            "Resolution mismatch. `%s` is %dx%d, but the coordinates in `%s` are "
            "in %dx%d space (per `%s`). Load the ORIGINAL video, not a padded or "
            "shrunk copy - markers will be misplaced otherwise."
            % (os.path.basename(vid_path), _got[0], _got[1],
               os.path.basename(csv_path), _exp[0], _exp[1],
               os.path.basename(meta_path)))

# Fallback iris circle, used only on frames where the iris was not measured
# (blinks, dropouts). Computed over TRACKED frames so blink garbage -- the old
# median included frames with a 10 px "iris" -- cannot drag it off the eye.
_ok = tracked_mask(df)
if {"iris_x", "iris_y", "iris_diam"}.issubset(df.columns) and df.loc[_ok, "iris_x"].notna().any():
    aoi_fallback = (float(df.loc[_ok, "iris_x"].median()),
                    float(df.loc[_ok, "iris_y"].median()),
                    float(df.loc[_ok, "iris_diam"].median()) / 2.0)
else:
    aoi_fallback = (float(df.loc[_ok, "pupil_x"].median()),
                    float(df.loc[_ok, "pupil_y"].median()), 120.0)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
fps = 50.0
n = len(df)
st.markdown("<p class='ocr-title'>Ocular Review</p>", unsafe_allow_html=True)
st.markdown("<p class='ocr-sub'>gaze + ocular-torsion &mdash; %d frames @ %.0f fps &mdash; %s</p>"
            % (n, fps, os.path.basename(csv_path)), unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Scrubber: slider + exact-frame entry
# ----------------------------------------------------------------------
frame_col, jump_col = st.columns([5, 1])
with frame_col:
    fi = st.slider("Frame", 0, n - 1, 0, 1, label_visibility="collapsed")
with jump_col:
    fi = st.number_input("idx", 0, n - 1, fi, 1, label_visibility="collapsed")

row = df.iloc[fi].to_dict()
t_now = row.get("time", fi / fps)
is_blink = row.get("blink", 0) == 1

# status chips
chip = "<span class='ocr-chip chip-blink'>BLINK</span>" if is_blink \
       else "<span class='ocr-chip chip-track'>TRACKING</span>"
tor_now = row.get("torsion_deg", np.nan)
st.markdown(
    "%s <span class='ocr-readout'>t = %.2fs &nbsp;|&nbsp; frame %d &nbsp;|&nbsp; "
    "torsion %s &nbsp;|&nbsp; features %s</span>"
    % (chip, t_now, fi,
       "--" if np.isnan(tor_now) else "%+.2f deg" % tor_now,
       "--" if np.isnan(row.get("n_features", np.nan)) else "%d" % int(row["n_features"])),
    unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Two video panels
# ----------------------------------------------------------------------
L, R = st.columns(2)
with L:
    st.markdown("<p class='ocr-panel-label'>RITnet segmentation</p>", unsafe_allow_html=True)
    if fi in ovl:
        st.image(ovl[fi], use_container_width=True)
    else:
        st.info("No overlay frame for index %d.\nCheck the overlay folder path." % fi)
with R:
    st.markdown("<p class='ocr-panel-label'>Irisometry tracking</p>", unsafe_allow_html=True)
    if cap is not None:
        gray = read_video_frame(cap, fi)
        if gray is not None:
            st.image(draw_irisometry(gray, row, frame_aoi(df, fi, aoi_fallback)),
                     use_container_width=True)
        else:
            st.info("Could not read frame %d from the video." % fi)
    else:
        st.info("Original video not found.\nSet its path in the sidebar.")

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Synced graphs (torsion / gaze / pupil) with a playhead
# ----------------------------------------------------------------------
plot_df = df.copy()
plot_df["t"] = plot_df["time"]
if plot_df["t"].isna().any():
    plot_df["t"] = plot_df["t"].fillna(pd.Series(plot_df.index / fps, index=plot_df.index))
rule = alt.Chart(pd.DataFrame({"t": [t_now]})).mark_rule(
    color="#39c5cf", strokeWidth=1.5).encode(x="t:Q")


def signal_chart(y, color, title, domain=None):
    enc_y = alt.Y(y, title=title,
                  scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    base = alt.Chart(plot_df).mark_line(color=color, strokeWidth=0.8).encode(
        x=alt.X("t:Q", title=None), y=enc_y).properties(height=140)
    return (base + rule).configure_view(strokeWidth=0).configure_axis(
        labelColor="#7d8794", titleColor="#7d8794", gridColor="#1b2530",
        domainColor="#222c38")

st.markdown("<p class='ocr-panel-label'>Torsion (deg)</p>", unsafe_allow_html=True)
st.altair_chart(signal_chart("torsion_deg", "#f5a623", "deg", domain=[-15, 15]),
                use_container_width=True)

g1, g2 = st.columns(2)
with g1:
    st.markdown("<p class='ocr-panel-label'>Gaze X (px)</p>", unsafe_allow_html=True)
    st.altair_chart(signal_chart("gaze_x", "#4ec9b0", "px"), use_container_width=True)
with g2:
    st.markdown("<p class='ocr-panel-label'>Gaze Y (px)</p>", unsafe_allow_html=True)
    st.altair_chart(signal_chart("gaze_y", "#4ec9b0", "px"), use_container_width=True)

st.markdown("<p class='ocr-panel-label'>Pupil diameter (px)</p>", unsafe_allow_html=True)
st.altair_chart(signal_chart("pupil_diam", "#a98bdc", "px"), use_container_width=True)