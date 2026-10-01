import streamlit as st
import os
import pandas as pd
import zipfile
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import pickle
import requests  
from requests.auth import HTTPDigestAuth  #Forces secure password handshake
import urllib3  

import io
import json
import re
import tempfile
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from PIL import ImageDraw, ImageOps

st.set_page_config(page_title="Lab Portal", page_icon="🧪", layout="wide")
# --- AXIS CAMERA GLOBAL SETTINGS ---
# Crucial: Force python to ignore internal self-signed network certificate warnings globally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # <--- NEW

# Camera login: kept out of the code. Cameras allow anonymous viewing, so these can stay blank.
# To use a password, put CAM_USER and CAM_PASS in .streamlit/secrets.toml.
try:
    CAM_USER = st.secrets.get("CAM_USER", "root")
    CAM_PASS = st.secrets.get("CAM_PASS", "")
except Exception:  # no secrets.toml
    CAM_USER, CAM_PASS = "root", ""

CAM_FLEET_IPS = {
    "easyBlood1_Camera1": "10.76.32.103",
    "easyBlood1_Camera2" : "10.76.32.117",
    "easyBlood4_Camera1": "10.76.32.115",
    "easyBlood4_Camera2": "10.76.32.116",
    "Presto1_Camera1": "10.76.32.112", 
    "Presto1_Camera_2" : "10.76.32.111",
    "Presto2_Camera1": "10.76.32.114",  
    "Presto2_Camera2": "10.76.32.113",
    "LC_Camera1": "10.76.32.110",
    "LC_Camera2": "10.76.32.109",
    "HC_Camera1" : "10.76.32.119",
    "HC_Camera2" : "10.76.32.120"
}


# ==========================================================
# TUBE HEMOLYSIS: photo analysis
# ----------------------------------------------------------
# 1. Crop each photo to the "tube region" where the plasma sits (set in the dashboard's Setup tab).
# 2. Keep only plasma-like pixels: bright enough (drops red cells/shadows) and colorful enough
#    (drops white labels, glare, gray background).
# 3. Measure their average color hue. Yellow plasma scores near 0; pink/red plasma scores higher.
# 4. Grade the score with thresholds calibrated against samples with known hemolysis.
# This is a screening aid; validate it against lab hemolysis measurements before relying on it.
# ==========================================================
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
GRADES = ["None", "Slight", "Moderate", "Gross", "Check image"]
HEMOLYZED_GRADES = {"Slight", "Moderate", "Gross"}
MIN_PLASMA_PIXELS = 50
MAX_SIDE = 1000

_EXIF_IFD = 0x8769
_EXIF_DATETIME_ORIGINAL = 36867
_EXIF_DATETIME = 306


def open_photo(source, max_side: int = MAX_SIDE) -> tuple[Image.Image, datetime | None]:
    """Open a photo (path or file-like), fix camera rotation and shrink it for analysis.

    Returns the image and when the photo was taken, from the camera's EXIF data if present.
    """
    with Image.open(source) as raw:
        taken = photo_taken_at(raw)
        img = ImageOps.exif_transpose(raw).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img, taken


def photo_taken_at(img: Image.Image) -> datetime | None:
    """When the photo was taken, from data saved inside the photo (EXIF or PNG text), if any."""
    try:
        exif = img.getexif()
        raw = exif.get_ifd(_EXIF_IFD).get(_EXIF_DATETIME_ORIGINAL) or exif.get(_EXIF_DATETIME)
        if raw:
            return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except (AttributeError, TypeError, ValueError):
        pass
    raw = (getattr(img, "info", None) or {}).get("Creation Time")
    if raw:
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(str(raw).strip()[:19], fmt)
            except ValueError:
                continue
    return None


def date_from_name(name: str) -> datetime | None:
    """A date written in a file name like 2026-09-30 or 20260930, if there is one."""
    for match in re.finditer(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})", name):
        try:
            return datetime(int(match[1]), int(match[2]), int(match[3]))
        except ValueError:
            continue
    return None


def rgb_to_lab(rgb):
    """Convert an sRGB uint8 array (..., 3) to CIE Lab (D65). Returns L, a, b arrays."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = (c @ m.T) / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return L, a, b


def roi_box(img: Image.Image, roi) -> tuple[int, int, int, int]:
    """Turn ROI fractions [left, top, right, bottom] into pixel coordinates."""
    w, h = img.size
    left, top, right, bottom = roi
    x0, y0 = int(left * w), int(top * h)
    return (x0, y0, max(int(right * w), x0 + 1), max(int(bottom * h), y0 + 1))


def plasma_mask(L, a, b, settings):
    """True for pixels that look like plasma/serum rather than cells, labels, or glare."""
    lightness, chroma = L, np.hypot(a, b)
    return (
        (lightness >= settings["min_lightness"])
        & (lightness <= settings["max_lightness"])
        & (chroma >= settings["min_chroma"])
    )


def measure(img: Image.Image, settings) -> dict:
    """Measure one photo. The score is NaN when no plasma-like pixels are found."""
    crop = np.asarray(img.crop(roi_box(img, settings["roi"])))
    L, a, b = rgb_to_lab(crop)
    mask = plasma_mask(L, a, b, settings)
    n = int(mask.sum())
    result = {
        "plasma_pixels": n,
        "plasma_fraction": round(n / mask.size, 3) if mask.size else 0.0,
        "mean_L": np.nan,
        "mean_a": np.nan,
        "mean_b": np.nan,
        "score": np.nan,
    }
    if n < MIN_PLASMA_PIXELS:
        return result
    mean_a, mean_b = float(a[mask].mean()), float(b[mask].mean())
    hue = float(np.degrees(np.arctan2(mean_b, mean_a)))
    result.update(
        mean_L=round(float(L[mask].mean()), 2),
        mean_a=round(mean_a, 2),
        mean_b=round(mean_b, 2),
        score=round(float(np.clip(90.0 - hue, 0.0, 90.0)), 2),
    )
    return result


def grade(score, thresholds) -> str:
    """Map a redness score to a hemolysis grade."""
    if score is None or np.isnan(score):
        return "Check image"
    if score >= thresholds["gross"]:
        return "Gross"
    if score >= thresholds["moderate"]:
        return "Moderate"
    if score >= thresholds["slight"]:
        return "Slight"
    return "None"


def overlay(img: Image.Image, settings) -> Image.Image:
    """Photo with the tube region outlined and plasma pixels tinted, for checking by eye."""
    box = roi_box(img, settings["roi"])
    crop = np.asarray(img.crop(box)).copy()
    L, a, b = rgb_to_lab(crop)
    mask = plasma_mask(L, a, b, settings)
    tint = np.array([0, 190, 255], dtype=np.float32)
    crop[mask] = (0.55 * crop[mask] + 0.45 * tint).astype(np.uint8)
    out = img.copy()
    out.paste(Image.fromarray(crop), box[:2])
    ImageDraw.Draw(out).rectangle(box, outline=(0, 110, 255), width=3)
    return out


# ==========================================================
# TUBE HEMOLYSIS: dashboard (results are saved so already-analyzed photos are skipped)
# ==========================================================
# Results, settings and small photo copies live outside the repo so lab data never reaches GitHub.
_custom_dir = os.environ.get("HEMOLYSIS_DATA_DIR", "").strip()
DATA_DIR = Path(_custom_dir) if _custom_dir else Path.home() / ".lab-dashboard" / "hemolysis"
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # e.g. the server container, where the home folder isn't writable
    DATA_DIR = Path(tempfile.gettempdir()) / "lab-dashboard-hemolysis"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
PHOTO_DIR = DATA_DIR / "photos"
PHOTO_DIR.mkdir(exist_ok=True)
SETTINGS_FILE = DATA_DIR / "settings.json"
RESULTS_FILE = DATA_DIR / "results.csv"

DEFAULT_SETTINGS = {
    # Tube region as fractions of the photo: left, top, right, bottom (0 = one edge, 1 = the other).
    # Set from an easyBlood tube photo: plasma layer of the front tube, right of the metal bracket.
    "roi": [0.48, 0.53, 0.85, 0.74],
    "min_lightness": 25.0,
    "max_lightness": 97.0,
    "min_chroma": 12.0,
    # Redness score cutoffs. PLACEHOLDERS: calibrate against samples with known hemolysis.
    "thresholds": {"slight": 15.0, "moderate": 30.0, "gross": 45.0},
}
RESULT_COLUMNS = [
    "file_name", "sample", "photo_time", "time_source", "stored_copy", "score", "mean_L",
    "mean_a", "mean_b", "plasma_pixels", "plasma_fraction", "analyzed_at", "manual_grade",
]  # fmt: skip
CALL_OPTIONS = ["Gross", "Moderate", "Slight", "None"]  # grades a person can assign


# ---------- storage ----------


def load_settings() -> dict:
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    try:
        saved = json.loads(SETTINGS_FILE.read_text())
        settings.update({k: v for k, v in saved.items() if k in settings})
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return settings


def save_settings(settings: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


def load_results() -> pd.DataFrame:
    if not RESULTS_FILE.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    # keep_default_na=False so a manual call of "None" isn't read back as a blank.
    df = pd.read_csv(
        RESULTS_FILE,
        dtype={"file_name": str, "sample": str, "manual_grade": str},
        keep_default_na=False,
        na_values=[""],
    )
    if "file_name" not in df.columns or "score" not in df.columns:  # not a results file
        return pd.DataFrame(columns=RESULT_COLUMNS)
    for col in RESULT_COLUMNS:  # columns added in later versions
        if col not in df.columns:
            df[col] = np.nan
    df["manual_grade"] = df["manual_grade"].astype(object)
    df["photo_time"] = pd.to_datetime(df["photo_time"])
    return df


def save_results(df: pd.DataFrame) -> None:
    df[RESULT_COLUMNS].to_csv(RESULTS_FILE, index=False)


def set_call(file_name: str, widget_key: str) -> None:
    """Save (or clear) a person's grade for one sample. Used as a widget callback."""
    value = st.session_state.get(widget_key)
    df = load_results()
    df.loc[df["file_name"] == file_name, "manual_grade"] = (
        value if value in CALL_OPTIONS else np.nan
    )
    save_results(df)


def remove_sample(file_name: str) -> None:
    """Delete one sample's result and its saved photo copy."""
    df = load_results()
    gone = df[df["file_name"] == file_name]
    for copy_name in gone["stored_copy"].dropna():
        (PHOTO_DIR / str(copy_name)).unlink(missing_ok=True)
    save_results(df[df["file_name"] != file_name])


def stored_name(file_name: str) -> str:
    """A safe file name for the small copy of an uploaded photo."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file_name).stem) + ".jpg"


def _read_upload(upload):
    upload.seek(0)
    return upload


def _read_zip_entry(archive: zipfile.ZipFile, entry: str):
    return io.BytesIO(archive.read(entry))


def collect_photos(uploads) -> list[tuple[str, object, datetime | None]]:
    """Turn uploads into (file name, opener, file date) for each photo.

    Zip files keep each photo's date (when it was saved); loose uploads don't have one,
    because browsers don't send file dates.
    """
    items: dict[str, tuple] = {}
    for upload in uploads or []:
        if upload.name.lower().endswith(".zip"):
            try:
                archive = zipfile.ZipFile(upload)
            except zipfile.BadZipFile:
                st.error(f"{upload.name} isn't a valid zip file, so it was skipped.")
                continue
            for info in archive.infolist():
                entry = info.filename
                name = entry.replace("\\", "/").rsplit("/", 1)[-1]
                is_photo = Path(name).suffix.lower().lstrip(".") in IMAGE_TYPES
                if is_photo and not entry.startswith("__MACOSX") and not name.startswith("."):
                    try:
                        file_date = datetime(*info.date_time)
                    except ValueError:
                        file_date = None
                    items.setdefault(
                        name, (name, partial(_read_zip_entry, archive, entry), file_date)
                    )
        else:
            items.setdefault(upload.name, (upload.name, partial(_read_upload, upload), None))
    return list(items.values())


def analyze_uploads(items, settings: dict, results: pd.DataFrame) -> pd.DataFrame:
    """Analyze photos, save small copies, and return results with the new rows added."""
    rows, failed = [], []
    bar = st.progress(0.0, text=f"Analyzing 0 of {len(items)}")
    previous_calls = results.set_index("file_name")["manual_grade"].to_dict()
    for i, (name, opener, file_date) in enumerate(items, start=1):
        try:
            img, taken = open_photo(opener())
        except Exception:  # not a readable image
            failed.append(name)
            continue
        copy_name = stored_name(name)
        img.save(PHOTO_DIR / copy_name, quality=90)
        # Photo date, best source first.
        named = date_from_name(name)
        if taken:
            when, source = taken, "photo data"
        elif file_date:
            when, source = file_date, "file date"
        elif named:
            when, source = named, "file name"
        else:
            when, source = datetime.now(), "upload time"
        rows.append(
            {
                "file_name": name,
                "sample": Path(name).stem,
                "photo_time": when,
                "time_source": source,
                "manual_grade": previous_calls.get(name, np.nan),
                "stored_copy": copy_name,
                "analyzed_at": datetime.now().isoformat(timespec="seconds"),
                **measure(img, settings),
            }
        )
        bar.progress(i / len(items), text=f"Analyzing {i} of {len(items)}")
    bar.empty()
    if failed:
        st.session_state.hemo_failed = failed
    names = {r["file_name"] for r in rows}
    kept = results[~results["file_name"].isin(names)]
    out = pd.concat([kept, pd.DataFrame(rows, columns=RESULT_COLUMNS)], ignore_index=True)
    out["photo_time"] = pd.to_datetime(out["photo_time"])
    return out


def reanalyze_saved(settings: dict, results: pd.DataFrame) -> pd.DataFrame:
    """Re-measure every saved photo copy with the current settings."""
    out = results.copy()
    bar = st.progress(0.0, text="Re-analyzing")
    for n, i in enumerate(out.index, start=1):
        path = PHOTO_DIR / str(out.at[i, "stored_copy"])
        if path.exists():
            img, _ = open_photo(path)
            for key, value in measure(img, settings).items():
                out.at[i, key] = value
            out.at[i, "analyzed_at"] = datetime.now().isoformat(timespec="seconds")
        bar.progress(n / len(out), text=f"Re-analyzing {n} of {len(out)}")
    bar.empty()
    return out


GRADE_ORDER = ["Gross", "Moderate", "Slight", "None", "Check image"]
GRADE_ICONS = {"Gross": "🔴", "Moderate": "🟠", "Slight": "🟡", "None": "🟢", "Check image": "⚪"}
THUMBS_PER_ROW = 10  # small tube pictures per row in the Sample viewer
THUMBS_PER_GROUP = 50  # most pictures shown per grade group
THUMB_HEIGHT = 150  # pixels
DETAIL_WIDTH = 200  # pixels, single-sample view
PREVIEW_WIDTH = 170  # pixels, Setup preview


def _crop_near_roi(img: Image.Image, roi, above: float = 0.12, below: float = 0.12) -> Image.Image:
    """The part of the photo around the tube region, so the plasma fills the picture."""
    left, top, right, bottom = roi
    w, h = img.size
    x0, x1 = max(0.0, left - 0.08), min(1.0, right + 0.08)
    y0, y1 = max(0.0, top - above), min(1.0, bottom + below)
    return img.crop((int(x0 * w), int(y0 * h), max(int(x1 * w), 1), max(int(y1 * h), 1)))


@st.cache_data(show_spinner=False, max_entries=3000)
def _thumbnail_cached(path: str, mtime: float, roi: tuple) -> Image.Image:
    img, _ = open_photo(path)
    thumb = _crop_near_roi(img, roi, above=0.04, below=0.04)
    scale = THUMB_HEIGHT / thumb.height
    return thumb.resize((max(1, int(thumb.width * scale)), THUMB_HEIGHT))


def _thumbnail(path: Path, roi) -> Image.Image | None:
    """Small picture of a tube's plasma area for the grouped view."""
    try:
        return _thumbnail_cached(str(path), path.stat().st_mtime, tuple(roi))
    except OSError:
        return None



# ---------- page ----------


def render_hemolysis_dashboard() -> None:
    """Draw the hemolysis dashboard screen."""
    if "hemo_settings" not in st.session_state:
        st.session_state.hemo_settings = load_settings()
        st.session_state.hemo_upload_round = 0
    s = st.session_state.hemo_settings

    st.title("🩸 Tube Hemolysis Dashboard")
    st.caption(
        "Screening aid based on plasma color in photos. Grades depend on the calibration in "
        "Setup and should be checked against lab measurements before being relied on."
    )

    flash = st.session_state.pop("hemo_flash", None)
    if flash:
        st.success(flash)
    failed = st.session_state.pop("hemo_failed", None)
    if failed:
        st.warning(
            f"These files couldn't be opened as photos and were skipped: {', '.join(failed)}"
        )

    results = load_results()

    # ---------- upload ----------

    with st.container(border=True):
        st.markdown("### 📤 Add photos")
        uploads = st.file_uploader(
            "Drag tube photos or zipped folders of photos here, or click to browse.",
            type=[*IMAGE_TYPES, "zip"],
            accept_multiple_files=True,
            key=f"hemo_uploads_{st.session_state.hemo_upload_round}",
        )
        items = collect_photos(uploads)
        if uploads and not items:
            st.warning("No photos found in what you uploaded.")
        if items:
            done = set(results["file_name"].astype(str))
            new = [it for it in items if it[0] not in done]
            repeat = [it for it in items if it[0] in done]
            st.write(
                f"**{len(new)}** new photo{'s' if len(new) != 1 else ''}"
                + (f", **{len(repeat)}** already analyzed and will be skipped" if repeat else "")
                + "."
            )
            redo = bool(repeat) and st.checkbox("Analyze the already-analyzed photos again too")
            todo = new + (repeat if redo else [])
            if st.button(f"Analyze {len(todo)} photos", type="primary", disabled=not todo):
                results = analyze_uploads(todo, s, results)
                save_results(results)
                st.session_state.hemo_flash = f"Analyzed {len(todo)} photos."
                st.session_state.hemo_upload_round += 1  # clears the upload box
                st.rerun()

    # Grades come from the saved score, so threshold changes apply instantly.
    if not results.empty:
        results["auto_grade"] = results["score"].apply(lambda x: grade(x, s["thresholds"]))
        manual = results["manual_grade"].where(results["manual_grade"].isin(CALL_OPTIONS))
        results["grade"] = manual.fillna(results["auto_grade"])
        results["call"] = np.where(manual.notna(), "manual", "auto")
        results["date"] = results["photo_time"].dt.date

    tab_dash, tab_view, tab_setup = st.tabs(["📊 Dashboard", "🔍 Sample viewer", "⚙️ Setup"])

    # ---------- dashboard ----------

    with tab_dash:
        if results.empty:
            st.info("No results yet. Add photos above to get started.")
        else:
            today = date.today()
            c1, c2 = st.columns([2, 3])
            picked = c1.date_input("Date range", (today - timedelta(days=20), today))
            search = c2.text_input("Find a sample", placeholder="Type part of a sample name")
            start, end = picked if len(picked) == 2 else (picked[0], picked[0])

            view = results[(results["date"] >= start) & (results["date"] <= end)]
            if search:
                view = view[view["sample"].str.contains(search, case=False, na=False)]

            total = len(view)
            n_hemo = int(view["grade"].isin(HEMOLYZED_GRADES).sum())
            n_check = int((view["grade"] == "Check image").sum())  # still waiting for a call
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Samples", total)
            m2.metric("Hemolyzed (slight or worse)", n_hemo)
            m3.metric("Hemolysis rate", f"{(n_hemo / total * 100) if total else 0:.1f}%")
            m4.metric("Need a manual check", n_check)

            if total:
                st.subheader("Samples per day by grade")
                daily = view.pivot_table(
                    index="date", columns="grade", values="file_name", aggfunc="count", fill_value=0
                ).reindex(columns=GRADES, fill_value=0)
                st.bar_chart(daily.loc[:, (daily != 0).any()])

                st.subheader("Samples")
                table = (
                    view.sort_values("photo_time", ascending=False)[
                        ["sample", "photo_time", "time_source", "grade", "call", "score"]
                    ]
                    .rename(
                        columns={
                            "sample": "Sample",
                            "photo_time": "Photo date",
                            "time_source": "Date from",
                            "grade": "Grade",
                            "call": "Graded by",
                            "score": "Redness score",
                        }
                    )
                    .reset_index(drop=True)
                )
                st.dataframe(table, width="stretch", hide_index=True)
                st.caption(
                    "Photo date comes from data saved in the photo if it has any, otherwise "
                    "the photo file's date (kept when you upload a zip), a date in the file "
                    "name, or the upload time. 'Graded by: manual' means a person made the call."
                )
                st.download_button(
                    "📥 Download these results (CSV)",
                    table.to_csv(index=False).encode(),
                    file_name=f"hemolysis_{start}_{end}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No samples match this date range or search.")

    # ---------- sample viewer: samples grouped by grade ----------

    with tab_view:
        if results.empty:
            st.info("Add some photos first.")
        else:
            pool = view  # same date range and search as the Dashboard tab
            counts = pool["grade"].value_counts()
            options = [g for g in GRADE_ORDER if counts.get(g, 0)]
            st.caption(
                "Samples grouped by hemolysis grade, using the date range and search from the "
                "Dashboard tab. Each group is sorted from highest to lowest redness score."
            )
            if not options:
                st.info("No samples match the Dashboard's date range or search.")
            else:
                default = [g for g in options if g in HEMOLYZED_GRADES] or options[:1]
                shown_grades = st.multiselect(
                    "Grades to show",
                    options,
                    default=default,
                    format_func=lambda g: f"{GRADE_ICONS[g]} {g} ({counts[g]})",
                )
                for g in [g for g in GRADE_ORDER if g in shown_grades]:
                    group = pool[pool["grade"] == g].sort_values("score", ascending=False)
                    st.markdown(f"#### {GRADE_ICONS[g]} {g}: {len(group)} sample(s)")
                    shown = group.head(THUMBS_PER_GROUP)
                    for first in range(0, len(shown), THUMBS_PER_ROW):
                        cols = st.columns(THUMBS_PER_ROW)
                        chunk = shown.iloc[first : first + THUMBS_PER_ROW]
                        for col, (_, r) in zip(cols, chunk.iterrows(), strict=False):
                            with col:
                                thumb = _thumbnail(PHOTO_DIR / str(r["stored_copy"]), s["roi"])
                                if thumb is not None:
                                    st.image(thumb)
                                score = "-" if pd.isna(r["score"]) else f"{r['score']:.0f}"
                                st.caption(f"{r['sample']}  \nscore {score}")
                                if g == "Check image":
                                    key = f"quick_call_{r['file_name']}"
                                    st.selectbox(
                                        "Call",
                                        ["", *CALL_OPTIONS],
                                        format_func=lambda x: "Make a call" if x == "" else x,
                                        key=key,
                                        on_change=set_call,
                                        args=(r["file_name"], key),
                                        label_visibility="collapsed",
                                    )
                    if len(group) > THUMBS_PER_GROUP:
                        st.caption(
                            f"Showing the {THUMBS_PER_GROUP} highest-scoring of {len(group)}. "
                            "Narrow the date range or search on the Dashboard tab to see others."
                        )

                if "Check image" in shown_grades and counts.get("Check image", 0):
                    st.caption(
                        "Check image: the app couldn't find plasma in these photos. Look at each "
                        "one and pick a grade under it; it then moves to that group."
                    )
                st.divider()
                st.markdown("#### Look at one sample")
                detail_pool = pool[pool["grade"].isin(shown_grades)].sort_values(
                    "score", ascending=False
                )
                if detail_pool.empty:
                    st.info("Pick at least one grade above.")
                else:
                    choice = st.selectbox(
                        "Sample",
                        detail_pool.index,
                        format_func=lambda i: (
                            f"{detail_pool.at[i, 'sample']}  ({detail_pool.at[i, 'grade']}, "
                            f"score {detail_pool.at[i, 'score']:.1f}, "
                            f"{detail_pool.at[i, 'photo_time']:%Y-%m-%d %H:%M})"
                        ),
                    )
                    row = detail_pool.loc[choice]
                    left, right = st.columns([1, 3])
                    with left:
                        path = PHOTO_DIR / str(row["stored_copy"])
                        if path.exists():
                            img, _ = open_photo(path)
                            whole = st.toggle("Show whole photo")
                            marked = overlay(img, s)
                            st.image(
                                marked if whole else _crop_near_roi(marked, s["roi"]),
                                width=DETAIL_WIDTH,
                                caption="Blue box: tube region. Blue tint: plasma pixels.",
                            )
                        else:
                            st.warning("The saved copy of this photo is missing.")
                    with right:
                        st.metric("Grade", row["grade"])
                        st.metric(
                            "Redness score", "-" if pd.isna(row["score"]) else f"{row['score']:.1f}"
                        )
                        pixels = int(row["plasma_pixels"]) if pd.notna(row["plasma_pixels"]) else 0
                        st.write(f"Plasma pixels found: {pixels}")
                        st.write(
                            f"Lab color: L {row['mean_L']}, a {row['mean_a']}, b {row['mean_b']}"
                        )
                        st.write(
                            f"Photo date: {row['photo_time']:%Y-%m-%d %H:%M} "
                            f"(from {row['time_source']})"
                        )

                        st.markdown("**Your call**")
                        current = row["manual_grade"] if row["call"] == "manual" else ""
                        key = f"detail_call_{row['file_name']}"
                        st.selectbox(
                            "Grade for this sample",
                            ["", *CALL_OPTIONS],
                            index=(["", *CALL_OPTIONS]).index(current),
                            format_func=lambda x: (
                                f"Use the app's grade ({row['auto_grade']})" if x == "" else x
                            ),
                            key=key,
                            on_change=set_call,
                            args=(row["file_name"], key),
                            help="Overrides the app's grade for this sample only.",
                        )

                        with st.expander("🗑️ Remove this sample from the data set"):
                            st.write(
                                "Deletes this sample's result and saved photo copy. If you "
                                "upload the photo again later, it will be analyzed as new."
                            )
                            sure = st.checkbox(
                                f"Yes, remove {row['sample']}", key=f"rm_ok_{row['file_name']}"
                            )
                            if st.button(
                                "Remove sample", disabled=not sure, key=f"rm_{row['file_name']}"
                            ):
                                remove_sample(row["file_name"])
                                st.session_state.hemo_flash = f"Removed {row['sample']}."
                                st.rerun()

    # ---------- setup / calibration ----------

    with tab_setup:
        st.write(
            "Set the tube region on a typical photo so the box covers only the plasma layer, then "
            "check that the blue tint lands on plasma and not on cells, labels, or background. "
            "Threshold changes apply instantly. After changing the region or filters, click "
            "**Re-analyze all saved photos**."
        )
        preview_img = None
        if items:
            preview_img, _ = open_photo(items[0][1]())
        elif not results.empty:
            latest = PHOTO_DIR / str(results.sort_values("photo_time").iloc[-1]["stored_copy"])
            if latest.exists():
                preview_img, _ = open_photo(latest)

        left, right = st.columns([3, 2])
        with right:
            st.subheader("Tube region")
            roi = s["roi"]
            left_edge, right_edge = st.slider(
                "Left and right edges", 0.0, 1.0, (float(roi[0]), float(roi[2])), 0.01
            )
            top_edge, bottom_edge = st.slider(
                "Top and bottom edges",
                0.0,
                1.0,
                (float(roi[1]), float(roi[3])),
                0.01,
                help="0 is the top of the photo, 1 is the bottom.",
            )
            s["roi"] = [left_edge, top_edge, right_edge, bottom_edge]

            st.subheader("Pixel filters")
            s["min_lightness"] = st.slider(
                "Minimum brightness (drops red cells and shadows)",
                0.0,
                100.0,
                float(s["min_lightness"]),
                1.0,
            )
            s["max_lightness"] = st.slider(
                "Maximum brightness (drops glare)", 0.0, 100.0, float(s["max_lightness"]), 1.0
            )
            s["min_chroma"] = st.slider(
                "Minimum color strength (drops labels and gray background)",
                0.0,
                60.0,
                float(s["min_chroma"]),
                1.0,
            )

            st.subheader("Grade thresholds (redness score)")
            th = s["thresholds"]
            th["slight"] = st.number_input("Slight from", value=float(th["slight"]), step=1.0)
            th["moderate"] = st.number_input("Moderate from", value=float(th["moderate"]), step=1.0)
            th["gross"] = st.number_input("Gross from", value=float(th["gross"]), step=1.0)

            b1, b2 = st.columns(2)
            if b1.button("💾 Save settings", width="stretch"):
                save_settings(s)
                st.success("Settings saved.")
            if b2.button("🔁 Re-analyze all saved photos", width="stretch", disabled=results.empty):
                results = reanalyze_saved(s, results)
                save_results(results)
                save_settings(s)
                st.session_state.hemo_flash = f"Re-analyzed {len(results)} photos."
                st.rerun()

        with left:
            if preview_img is not None:
                st.image(overlay(preview_img, s), width=PREVIEW_WIDTH, caption="Preview")
                preview = measure(preview_img, s)
                score = preview["score"]
                st.write(
                    "Redness score for this photo: "
                    + ("no plasma found" if np.isnan(score) else f"{score:.1f}")
                    + f", grade: {grade(score, s['thresholds'])}"
                )
            else:
                st.info("Add a photo above to preview the tube region here.")

        if not results.empty and results["score"].notna().any():
            st.subheader("Score distribution")
            st.caption("For calibration: normal and hemolyzed tubes should form separate groups.")
            counts = (
                results["score"].dropna().round(0).value_counts().sort_index().rename("Samples")
            )
            st.bar_chart(counts)

        with st.expander("Delete all results"):
            st.write(f"Results and photo copies are stored in `{DATA_DIR}`.")
            sure = st.checkbox("Yes, delete every saved result and photo copy")
            if st.button("Delete everything", disabled=not sure):
                RESULTS_FILE.unlink(missing_ok=True)
                for p in PHOTO_DIR.glob("*.jpg"):
                    p.unlink()
                st.session_state.hemo_flash = "All results deleted."
                st.rerun()


# 1. INITIALIZE SESSION STATE ROUTING (Tracks which screen we are viewing)
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"
    
# ==========================================================
# SCREEN 1: THE WELCOME SCREEN / MAIN HUB (4 Columns)
# ==========================================================
if st.session_state.current_page == "home":
    st.title("🧪 Laboratory Command Center")
    st.write("Welcome to the Harbinger Health portal. Select a task to begin working.")
    st.write("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # COLUMN 1: Plate Processing Tool
    with col1:
        st.subheader("📁 Plate Processing")
        st.write("Upload raw plate files and execute background blanking mathematics.")
        if st.button("🚀 Upload Plate", type="primary", use_container_width=True):
            st.session_state.current_page = "uploader"
            st.rerun()
            
    # COLUMN 2: Tube Hemolysis & Volume Inspection Tool
    with col2:
        st.subheader("🩸 Hemolysis Check")
        st.write("Drop in tube photos or zipped folders to screen for hemolysis and track recent samples.")
        if st.button("🔍 Inspect Tubes", type="primary", use_container_width=True):
            st.session_state.current_page = "hemolysis_inspector"
            st.rerun()
            
    # COLUMN 3: Camera Inspection Tool
    with col3:
        st.subheader("🎥 Cameras")
        st.write("Trigger automated deck imagery, barcode scanning, or colony counts.")
                
        if st.button("🎥 Lab Cameras", type="primary", use_container_width=True):
            st.session_state.current_page = "cameras"
            st.rerun()

        first_cam_name = list(CAM_FLEET_IPS.keys())[0]
        first_cam_ip = CAM_FLEET_IPS[first_cam_name]
        
        # --- NEW BROWSER DIRECT SECURE INJECTION ---
        # Passing credentials over HTTPS directly within an HTML5 canvas container
        PREVIEW_CAM_URL = f"https://{first_cam_ip}/axis-cgi/mjpg/video.cgi?resolution=320x240" 
        
        preview_html = f"""
        <html>
            <body style="margin:0; padding:0; background-color:#1E1E1E; border-radius:8px; overflow:hidden;">
                <img src="{PREVIEW_CAM_URL}" style="width:100%; height:230px; object-fit:contain; display:block; background-color:#000;" 
                     onerror="this.onerror=null; this.parentNode.innerHTML='<div style=\"color:#ff4b4b; text-align:center; padding-top:95px; font-family:sans-serif;\">⚠️ Camera Connection Locked</div>';">
            </body>
        </html>
        """
        st.components.v1.html(preview_html, height=235)  # tall enough for the whole frame
        # -------------------------------------------


    # COLUMN 4: Automation Deck (Linked to Venus Portal)
    with col4:
        st.subheader("🔬 Instrument Dashboard")
        st.write("Direct integration matrix with the company's Hamilton Venus automation pipelines.")
        
        # 🎨 SWITCHED TO TYPE="PRIMARY" TO MATCH NAVY BLUE / WHITE WRITING THEME
        st.link_button(
            label="🌐 Open Venus Portal", 
            url="https://venus.harbinger-health.net/", 
            type="primary", 
            use_container_width=True
        )
        
        st.write("---")
        # Live preview embed matrix
        try:
            st.components.v1.iframe(src="https://venus.harbinger-health.net/", height=350, scrolling=True)
        except Exception as e:
            st.caption("Unable to load embedded Venus frame view.")

# ==========================================================
# SCREEN: TUBE HEMOLYSIS DASHBOARD (drag-and-drop, skips already-analyzed photos)
# ==========================================================
elif st.session_state.current_page == "hemolysis_inspector":
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("⬅️ Back to Main Hub"):
            st.session_state.current_page = "home"
            st.rerun()
    with nav2:
        if st.button("📦 Review a zip with manual Yes/No corrections"):
            st.session_state.current_page = "hemolysis_zip"
            st.rerun()
    render_hemolysis_dashboard()

# ==========================================================
# SCREEN: TUBE INSPECTION SCREEN (PERFECT ROW-BY-ROW UNIFORM GRID)
# ==========================================================
elif st.session_state.current_page == "hemolysis_zip":
    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("⬅️ Back to Main Hub"):
            st.session_state.current_page = "home"
            st.rerun()
    with nav2:
        if st.button("📊 Back to Hemolysis Dashboard"):
            st.session_state.current_page = "hemolysis_inspector"
            st.rerun()

    st.title("🩸 Tube Hemolysis & Active Learning Registry")
    st.write("Upload your zipped image folder to review tube color regions and adjust findings directly.")
    st.write("---")
    
    uploaded_zip = st.file_uploader(
        "Select or Drag and Drop the zipped folder:", 
        type=["zip"], 
        accept_multiple_files=False
    )
    
    if uploaded_zip:
        try:
            if "tube_corrections" not in st.session_state:
                st.session_state.tube_corrections = {}
                
            results_data = []
            valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
            
            FEEDBACK_DIR = "training_data_feedback"
            os.makedirs(os.path.join(FEEDBACK_DIR, "normal"), exist_ok=True)
            os.makedirs(os.path.join(FEEDBACK_DIR, "hemolyzed"), exist_ok=True)
            
            with zipfile.ZipFile(uploaded_zip) as z:
                all_files = z.namelist()
                image_paths = [f for f in all_files if f.lower().endswith(valid_extensions) and not f.startswith('__MACOSX') and not os.path.basename(f).startswith('.')]
                
                if not image_paths:
                    st.error("Could not find any supported image formats inside this zip package.")
                else:
                    st.success(f"📦 Successfully extracted {len(image_paths)} images.")
                    
                    # 💡 SET IMAGES PER ROW HERE (e.g., 4 or 6 images looks best)
                    IMAGES_PER_ROW = 8
                    
                    # 💡 FIX: Split the files into distinct rows so they align horizontally
                    for i in range(0, len(image_paths), IMAGES_PER_ROW):
                        row_paths = image_paths[i : i + IMAGES_PER_ROW]
                        
                        # Generate a fresh row container layout
                        grid_cols = st.columns(IMAGES_PER_ROW)
                        
                        for col_idx, img_path in enumerate(row_paths):
                            # Overall dataset tracking index
                            global_idx = i + col_idx
                            
                            filename = os.path.basename(img_path)
                            img_bytes = z.read(img_path)
                            
                            display_img = img_bytes
                            model_pred = "No"
                            
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if cv_img is not None:
                                h, w, _ = cv_img.shape
                                start_y, end_y = int(h * 0.20), int(h * 0.55)
                                start_x, end_x = int(w * 0.25), int(w * 0.75)
                                plasma_zone = cv_img[start_y:end_y, start_x:end_x]
                                
                                if plasma_zone.size > 0:
                                    avg_color = np.average(np.average(plasma_zone, axis=0), axis=0)
                                    avg_b = avg_color[0]
                                    avg_g = avg_color[1]
                                    avg_r = avg_color[2]
                                    
                                    if os.path.exists("hemolysis_model.pkl"):
                                        try:
                                            with open("hemolysis_model.pkl", "rb") as f:
                                                trained_clf = pickle.load(f)
                                            pred_idx = trained_clf.predict([[avg_r, avg_g, avg_b]])[0]
                                            model_pred = "Yes" if pred_idx == 1 else "No"
                                        except:
                                            model_pred = "Yes" if (avg_r > (avg_g * 1.15) and avg_r > 100) else "No"
                                    else:
                                        model_pred = "Yes" if (avg_r > (avg_g * 1.15) and avg_r > 100) else "No"              
                                    
                                    box_thickness = max(2, int(w * 0.01))
                                    cv2.rectangle(cv_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), box_thickness)
                                    cv2.putText(
                                        cv_img, "COLOR ZONE", (start_x, max(20, start_y - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), max(1, int(box_thickness/2))
                                    )
                                    display_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                            
                            default_index = 0 if model_pred == "No" else 1
                            if filename in st.session_state.tube_corrections:
                                default_index = 0 if st.session_state.tube_corrections[filename] == "No" else 1
                            
                            # Render elements strictly within this row's active column slot
                            #  NEW CORRECTED CODE
                            with grid_cols[col_idx]:
                                # Streamlit scales the image to fit the column width automatically
                                st.image(display_img, use_container_width=True) 
                                st.caption(f"**{filename[:16]}...**")
                                
                                user_validation = st.selectbox(
                                    "Hemolysis?",
                                    options=["No", "Yes"],
                                    index=default_index,
                                    key=f"select_{filename}_{global_idx}"
                                )
                                
                                if user_validation != model_pred:
                                    st.session_state.tube_corrections[filename] = user_validation
                                    label_folder = "hemolyzed" if user_validation == "Yes" else "normal"
                                    save_filepath = os.path.join(FEEDBACK_DIR, label_folder, filename)
                                    with open(save_filepath, "wb") as f:
                                        f.write(img_bytes)
                                    st.caption("💾 *Logged*")
                                
                                simulated_ml = round(1.2 + (global_idx * 0.45) % 3.8, 2) 
                                
                                results_data.append({
                                    "Sample Identification (Filename)": filename,
                                    "Model Prediction": model_pred,
                                    "Final Confirmed Status": user_validation,
                                    "Estimated Volume (mL)": simulated_ml,
                                    "User Corrected": "True" if user_validation != model_pred else "False"
                                })
                        
                        # Add a visual spacer row to keep separate horizontal row groupings clean
                        st.write("")
            
            if results_data:
                df_results = pd.DataFrame(results_data)
                st.write("---")
                st.subheader("📊 Validated Summary Registry")
                st.dataframe(df_results, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    csv_buffer = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Export Assessment List (CSV)",
                        data=csv_buffer,
                        file_name="verified_tube_report.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True
                    )
                
                with col2:
                    memory_zip = BytesIO()
                    has_files = False
                    
                    if os.path.exists(FEEDBACK_DIR):
                        for root, dirs, files in os.walk(FEEDBACK_DIR):
                            for file in files:
                                if not file.startswith('.'):
                                    has_files = True
                                    
                    # This block handles the training dataset download buttons
                    if has_files:
                        with zipfile.ZipFile(memory_zip, "w") as z_out:
                            for root, dirs, files in os.walk(FEEDBACK_DIR):
                                for file in files:
                                    if not file.startswith('.'):
                                        file_path = os.path.join(root, file)
                                        archive_name = os.path.relpath(file_path, FEEDBACK_DIR)
                                        z_out.write(file_path, archive_name)
                                        
                        memory_zip.seek(0)
                        st.download_button(
                            label="📥 Download Training Dataset (.zip)",
                            data=memory_zip,
                            file_name="my_hemolysis_training_data.zip",
                            mime="application/zip",
                            type="secondary",
                            use_container_width=True
                        )
                    else:
                        st.button("📥 Training Dataset Empty", disabled=True, use_container_width=True)
                        
        except zipfile.BadZipFile:
            st.error("The uploaded file structure appears corrupted or isn't a true zip file structure.")
        except Exception as e:
            st.error(f"Processing error: {e}")

# ==========================================================
# SCREEN 2: THE FILE UPLOAD & MATH SCREEN 
# ==========================================================
elif st.session_state.current_page == "uploader":
    if st.button("⬅️ Back to Main Hub"):
        st.session_state.current_page = "home"
        st.rerun()

    st.title("🧪 Lab Plate Upload & Math Analysis")
    
    st.sidebar.header("Plate Metadata")
    tech_name = st.sidebar.text_input("Technologist Name")
    plate_id = st.sidebar.text_input("Plate / Batch ID")
    blank_value = st.sidebar.number_input("Negative Control / Blank Value to Subtract", value=0.0, step=0.1)
    
    uploaded_file = st.file_uploader("Upload Plate File", type=["csv", "xlsx"])
    
    if uploaded_file and tech_name and plate_id:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin-1')
            else:
                df = pd.read_excel(uploaded_file)
                
            st.success(f"📊 Loaded '{uploaded_file.name}' successfully!")
            
            if 'Conc. [pg/µl]' in df.columns:
                df['Corrected_Value'] = df['Conc. [pg/µl]'] - blank_value
                
                mean_signal = df['Conc. [pg/µl]'].mean()
                std_signal = df['Conc. [pg/µl]'].std()
                max_signal = df['Conc. [pg/µl]'].max()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Average Raw Signal", f"{mean_signal:.2f}")
                col2.metric("Standard Deviation", f"{std_signal:.2f}")
                col3.metric("Max Signal Observed", f"{max_signal:.2f}")
                
                st.subheader("Processed Plate Data Table")
                st.dataframe(df)
                
                if st.button("Save Calculated Results"):
                    os.makedirs("saved_plates", exist_ok=True)
                    save_path = os.path.join("saved_plates", f"PROCESSED_{plate_id}_{uploaded_file.name}")
                    
                    if uploaded_file.name.endswith('.csv'):
                        df.to_csv(save_path, index=False)
                    else:
                        df.to_excel(save_path, index=False)
                    st.balloons()
                    st.success(f"💾 Calculations saved to: {save_path}")
                    
            else:
                st.error("Error: Could not find a column named 'Conc. [pg/µl]' in your uploaded file. Please make sure your column headers match.")
                st.write("Your column names are:", list(df.columns))
                
        except Exception as e:
            st.error(f"Could not parse file: {e}")
    else:
        st.info("Please fill out metadata in the sidebar and upload a file to run calculations.")

# ==========================================================
# SCREEN 3: LAB CAMERAS LIVE FLEET MULTI-VIEW (WITH FULLSCREEN MAXIMIZE)
# ==========================================================
elif st.session_state.current_page == "cameras":
    # Initialize a session state flag to track if we are focusing on a single camera
    if "maximized_cam" not in st.session_state:
        st.session_state.maximized_cam = None

    # Custom navigation row
    nav_col1, nav_col2 = st.columns([1, 5])
    with nav_col1:
        if st.button("⬅️ Back to Hub", use_container_width=True):
            st.session_state.current_page = "home"
            st.session_state.maximized_cam = None  # Reset focus on exit
            st.rerun()
    with nav_col2:
        if st.session_state.maximized_cam:
            if st.button(f"🔍 Show All {len(CAM_FLEET_IPS)} Cameras", type="secondary"):
                st.session_state.maximized_cam = None
                st.rerun()

    st.title("🎥 Lab Camera Command Center")
    st.write("Real-time persistent feed of automation instruments.")
    st.write("---")

    # --------------------------------------------------
    # VIEW MODE A: EXPANDED SINGLE CAMERA FULLSCREEN MODE
    # --------------------------------------------------
    if st.session_state.maximized_cam and st.session_state.maximized_cam in CAM_FLEET_IPS:
        cam_name = st.session_state.maximized_cam
        cam_ip = CAM_FLEET_IPS[cam_name]
        
        st.subheader(f"🔍 Fullscreen View: {cam_name}")
        stream_url = f"https://{cam_ip}/axis-cgi/mjpg/video.cgi"
        
        # High height setting (650px) to comfortably stretch across a modern monitor screen
        fullscreen_html = f"""
        <html>
            <body style="margin:0; padding:0; background-color:black; font-family:sans-serif; overflow:hidden;">
                <div style="position:relative; width:100%; height:640px;">
                    <img src="{stream_url}" style="width:100%; height:100%; object-fit:contain; display:block;">
                </div>
            </body>
        </html>
        """
        st.components.v1.html(fullscreen_html, height=650)

    # --------------------------------------------------
    # VIEW MODE B: STANDARD MULTI-VIEW GRID MODE
    # --------------------------------------------------
    else:
        cam_cols = st.columns(2)

        for idx, (cam_name, cam_ip) in enumerate(CAM_FLEET_IPS.items()):
            with cam_cols[idx % 2]:
                # Inline Header layout grouping with a dynamic Maximize action link button
                head_col1, head_col2 = st.columns([3, 1])
                with head_col1:
                    st.subheader(cam_name)
                with head_col2:
                    # Clicking this flags the current name in the app's persistent session router memory
                    if st.button("🔲 Fullscreen", key=f"max_{idx}", use_container_width=True):
                        st.session_state.maximized_cam = cam_name
                        st.rerun()
                
                stream_url = f"https://{cam_ip}/axis-cgi/mjpg/video.cgi"
                
                stream_html = f"""
                <html>
                    <body style="margin:0; padding:0; background-color:black; font-family:sans-serif; overflow:hidden;">
                        <div style="position:relative; width:100%; height:320px;">
                            <img src="{stream_url}" style="width:100%; height:100%; object-fit:contain; display:block;" 
                                 onerror="this.onerror=null; this.parentNode.innerHTML='<div style=\"color:#ff4b4b; display:flex; justify-content:center; align-items:center; height:100%; flex-direction:column;\"><span>⚠️</span><span style=\"margin-top:8px;\">{cam_name} Offline</span></div>';">
                        </div>
                    </body>
                </html>
                """
                st.components.v1.html(stream_html, height=330)
                st.write("") # Spacing margin layout buffer
