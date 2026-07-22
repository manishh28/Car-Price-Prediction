import base64
import html
import json
import re
from pathlib import Path
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).with_name("car details.csv")
VEHICLE_ASSET_PATH = Path(__file__).with_name("assets") / "vehicle-showcase.png"
CATEGORICAL_FEATURES = ["name", "fuel", "seller_type", "transmission", "owner"]
NUMERIC_FEATURES = [
    "year",
    "km_driven",
    "mileage_value",
    "engine_value",
    "max_power_value",
    "seats",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


st.set_page_config(
    page_title="Used Car Value Estimator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #16211f;
        --muted: #60706c;
        --surface: #ffffff;
        --canvas: #f4f7f6;
        --line: #d8e0de;
        --teal: #146c63;
        --teal-dark: #0f514b;
        --coral: #e7604f;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, p, label, div {
        letter-spacing: 0 !important;
    }

    .app-kicker {
        color: var(--teal);
        font-size: 0.78rem;
        font-weight: 750;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .app-title {
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3.35rem);
        font-weight: 760;
        line-height: 1.08;
        margin: 0;
        max-width: 760px;
    }

    .app-intro {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.65;
        margin: 0.9rem 0 1.5rem;
        max-width: 720px;
    }

    .accent-line {
        background: linear-gradient(90deg, var(--teal) 0 72%, var(--coral) 72% 82%, transparent 82%);
        height: 4px;
        margin-bottom: 1.9rem;
        width: 100%;
    }

    .section-heading {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 720;
        margin: 0.2rem 0 0.15rem;
    }

    .section-copy {
        color: var(--muted);
        font-size: 0.86rem;
        margin: 0 0 0.9rem;
    }

    div[data-testid="stMetric"] {
        background: transparent;
        border-left: 3px solid var(--line);
        padding: 0.15rem 0 0.15rem 0.85rem;
    }

    div[data-testid="stMetric"] label {
        color: var(--muted);
        font-size: 0.78rem;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 700;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background: var(--surface) !important;
        border-color: var(--line);
        border-radius: 6px;
        color: var(--ink) !important;
        min-height: 44px;
    }

    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: var(--ink) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    div[data-testid="stNumberInput"] input {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }

    div[data-testid="stNumberInput"] button {
        background: #edf2f1 !important;
        border-color: var(--line) !important;
        color: var(--ink) !important;
    }

    div[data-baseweb="select"] > div:focus-within,
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--teal);
        box-shadow: 0 0 0 1px var(--teal);
    }

    .stButton button,
    [data-testid="stFormSubmitButton"] button {
        background: var(--teal) !important;
        border: 1px solid var(--teal);
        border-radius: 6px;
        color: #ffffff !important;
        font-size: 0.98rem;
        font-weight: 700;
        min-height: 48px;
        transition: background 150ms ease, border-color 150ms ease;
        width: 100%;
    }

    .stButton button p,
    [data-testid="stFormSubmitButton"] button p {
        color: #ffffff !important;
    }

    .stButton button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: var(--teal-dark) !important;
        border-color: var(--teal-dark);
        color: #ffffff;
    }

    .result-panel {
        background: var(--ink);
        border-top: 5px solid var(--coral);
        color: #ffffff;
        margin: 2rem 0 1.3rem;
        padding: 1.6rem 1.8rem 1.7rem;
    }

    .result-label {
        color: #b8c8c4;
        font-size: 0.75rem;
        font-weight: 750;
        text-transform: uppercase;
    }

    .result-price {
        color: #ffffff;
        font-size: clamp(2rem, 5vw, 3.2rem);
        font-weight: 760;
        line-height: 1.15;
        margin: 0.35rem 0 0.55rem;
    }

    .result-note {
        color: #d9e2e0;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0;
    }

    .data-note {
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 0.78rem;
        line-height: 1.55;
        margin-top: 2rem;
        padding-top: 1rem;
    }

    div[data-baseweb="tab-list"] {
        border-bottom: 1px solid var(--line);
        gap: 1.25rem;
    }

    button[data-baseweb="tab"] {
        color: var(--muted);
        font-weight: 680;
        padding-left: 0;
        padding-right: 0;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--teal);
    }

    div[data-testid="stSlider"] [role="slider"] {
        background: var(--teal) !important;
    }

    .live-change-positive {
        color: var(--teal);
        font-size: 0.86rem;
        font-weight: 700;
    }

    .live-change-negative {
        color: var(--coral);
        font-size: 0.86rem;
        font-weight: 700;
    }

    .vehicle-stage {
        background: #102724;
        border-bottom: 4px solid var(--coral);
        color: #ffffff;
        display: grid;
        grid-template-columns: minmax(230px, 0.78fr) minmax(380px, 1.45fr);
        margin: 1.2rem 0 1.8rem;
        min-height: 285px;
        overflow: hidden;
        padding: 1.55rem 1.7rem 2.1rem;
        position: relative;
    }

    .vehicle-stage-copy {
        align-self: center;
        position: relative;
        z-index: 2;
    }

    .vehicle-stage-kicker {
        color: #8fc2b9;
        font-size: 0.72rem;
        font-weight: 750;
        text-transform: uppercase;
    }

    .vehicle-stage-name {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 750;
        line-height: 1.2;
        margin: 0.45rem 0 1rem;
        max-width: 360px;
    }

    .vehicle-stage-facts {
        color: #c4d5d1;
        display: grid;
        font-size: 0.78rem;
        gap: 0.45rem;
    }

    .vehicle-stage-facts strong {
        color: #ffffff;
        font-weight: 700;
    }

    .vehicle-motion-window {
        align-self: end;
        height: 230px;
        min-width: 0;
        position: relative;
        z-index: 2;
    }

    .vehicle-motion-window img {
        position: absolute;
    }

    .vehicle-motion-window:not(.is-photo) img {
        bottom: -0.15rem;
        filter: drop-shadow(0 16px 14px rgba(0, 0, 0, 0.34));
        height: auto;
        max-width: none;
        right: -1.2rem;
        width: min(760px, 112%);
        animation: vehicle-enter 850ms cubic-bezier(0.22, 1, 0.36, 1) both,
                   vehicle-idle 4.2s ease-in-out 900ms infinite;
    }

    .vehicle-motion-window.is-photo {
        align-self: stretch;
        height: auto;
    }

    .vehicle-motion-window.is-photo img {
        animation: vehicle-photo-enter 900ms cubic-bezier(0.22, 1, 0.36, 1) both,
                   vehicle-photo-idle 7s ease-in-out 1s infinite;
        bottom: -2.1rem;
        height: calc(100% + 3.65rem);
        max-width: none;
        object-fit: cover;
        object-position: center;
        opacity: 0.82;
        right: -1.7rem;
        top: -1.55rem;
        width: calc(100% + 5rem);
        -webkit-mask-image: linear-gradient(to right, transparent 0%, #000 30%);
        mask-image: linear-gradient(to right, transparent 0%, #000 30%);
    }

    .vehicle-motion-window.is-3d {
        align-self: stretch;
        height: auto;
    }

    .vehicle-3d-frame {
        animation: vehicle-viewer-enter 700ms cubic-bezier(0.22, 1, 0.36, 1) both;
        background: #0a1715;
        border: 0;
        bottom: -2.1rem;
        height: calc(100% + 3.65rem);
        position: absolute;
        right: -1.7rem;
        top: -1.55rem;
        width: calc(100% + 3.4rem);
    }

    .vehicle-image-credit {
        bottom: 0.22rem;
        color: rgba(255, 255, 255, 0.74) !important;
        font-size: 0.58rem;
        position: absolute;
        right: 0.65rem;
        text-decoration: none !important;
        z-index: 4;
    }

    .vehicle-image-credit:hover {
        color: #ffffff !important;
        text-decoration: underline !important;
    }

    .road-marker-track {
        align-items: center;
        bottom: 1.05rem;
        display: flex;
        left: 0;
        position: absolute;
        width: 220%;
        animation: road-move 2.2s linear infinite;
        z-index: 3;
    }

    .road-marker-track span {
        background: rgba(255, 255, 255, 0.42);
        display: block;
        height: 3px;
        margin-right: 46px;
        width: 68px;
    }

    @keyframes vehicle-enter {
        from { opacity: 0; transform: translateX(34%); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes vehicle-idle {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
    }

    @keyframes vehicle-photo-enter {
        from { opacity: 0; transform: translateX(18%) scale(1.05); }
        to { opacity: 0.82; transform: translateX(0) scale(1.02); }
    }

    @keyframes vehicle-photo-idle {
        0%, 100% { transform: scale(1.02); }
        50% { transform: scale(1.055); }
    }

    @keyframes vehicle-viewer-enter {
        from { opacity: 0; transform: translateX(12%); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes road-move {
        from { transform: translateX(0); }
        to { transform: translateX(-114px); }
    }

    @media (max-width: 900px) {
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: 0.8rem;
        }

        [data-testid="column"] {
            flex: 1 1 280px !important;
            min-width: 0 !important;
            width: 100% !important;
        }

        .vehicle-stage {
            grid-template-columns: 1fr;
            min-height: 445px;
        }

        .vehicle-stage-copy {
            align-self: start;
        }

        .vehicle-stage-name {
            max-width: 100%;
        }

        .vehicle-motion-window {
            height: 205px;
        }

        .vehicle-motion-window:not(.is-photo) img {
            left: 50%;
            right: auto;
            transform: translateX(-50%);
            width: min(720px, 118%);
        }

        .vehicle-motion-window.is-photo img {
            bottom: -2.1rem;
            height: calc(100% + 2.1rem);
            left: -1.7rem;
            right: -1.7rem;
            top: 0;
            width: calc(100% + 3.4rem);
            -webkit-mask-image: linear-gradient(to bottom, transparent 0%, #000 28%);
            mask-image: linear-gradient(to bottom, transparent 0%, #000 28%);
        }

        .vehicle-3d-frame {
            bottom: -2.1rem;
            height: calc(100% + 2.1rem);
            left: -1.7rem;
            right: -1.7rem;
            top: 0;
            width: calc(100% + 3.4rem);
        }

        @keyframes vehicle-enter {
            from { opacity: 0; transform: translateX(-22%); }
            to { opacity: 1; transform: translateX(-50%); }
        }

        @keyframes vehicle-idle {
            0%, 100% { transform: translate(-50%, 0); }
            50% { transform: translate(-50%, -4px); }
        }
    }

    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.4rem;
        }

        .app-title {
            font-size: 2.15rem;
        }

        .result-panel {
            padding: 1.25rem;
        }

        .vehicle-stage {
            min-height: 520px;
            padding: 1.25rem 1.1rem 1.8rem;
        }

        .vehicle-stage-name {
            font-size: 1.28rem;
        }

        .vehicle-motion-window {
            height: 280px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .vehicle-motion-window img,
        .vehicle-3d-frame,
        .road-marker-track {
            animation: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False),
        errors="coerce",
    )


@st.cache_data(show_spinner=False)
def load_vehicle_asset() -> str:
    encoded = base64.b64encode(VEHICLE_ASSET_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def model_family_name(brand: str, model: str) -> str:
    variant_markers = {
        "amt", "asta", "at", "automatic", "crdi", "dct", "diesel", "era",
        "gdi", "highline", "luxury", "lxi", "m", "magna", "manual", "mpi",
        "mt", "petrol", "prestige", "sedan", "sportline", "sportz", "tdi",
        "tsi", "vdi", "vtec", "vxi", "xline", "zdi", "zxi",
    }
    model_words = model.split()
    brand_words = brand.split()
    remaining = model_words[len(brand_words):]
    family_words: list[str] = []

    for word in remaining:
        normalized = re.sub(r"[^a-z0-9.]", "", word.lower())
        is_engine_size = bool(re.fullmatch(r"\d+(?:\.\d+)?", normalized))
        is_engine_badge = bool(
            re.fullmatch(r"\d+(?:\.\d+)?[a-z]{1,4}", normalized)
        )
        is_drive_layout = bool(re.fullmatch(r"\d+x\d+", normalized))
        is_trim_code = bool(re.fullmatch(r"[vwz]\d{1,2}", normalized))
        is_drivetrain_badge = normalized.startswith(("sdrive", "xdrive"))
        if family_words and (
            normalized in variant_markers
            or is_engine_size
            or is_engine_badge
            or is_drive_layout
            or is_trim_code
            or is_drivetrain_badge
        ):
            break
        family_words.append(word)
        if len(family_words) == 4:
            break

    return " ".join([brand, *(family_words or remaining[:1])])


def plain_metadata(value: str, fallback: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value or ""))
    cleaned = " ".join(without_tags.split())
    return cleaned[:70] or fallback


@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_vehicle_3d_model(search_name: str) -> dict[str, str] | None:
    search_tokens = set(re.findall(r"[a-z0-9]+", search_name.lower()))
    parameters = {
        "type": "models",
        "q": search_name,
        "sort_by": "-relevance",
        "count": "24",
    }
    request = urllib.request.Request(
        "https://api.sketchfab.com/v3/search?" + urllib.parse.urlencode(parameters),
        headers={
            "User-Agent": (
                "AutoValueIndia/1.0 "
                "(https://github.com/manishh28/Car-Price-Prediction)"
            )
        },
    )
    excluded_terms = {
        "badge", "bodykit", "cabriolet", "dashboard", "diagram", "emblem",
        "engine", "gauge", "interior", "logo", "mod", "offroad", "pack",
        "part", "rim", "spoiler", "traffic", "trim", "tuned", "vent", "wheel",
    }

    with urllib.request.urlopen(request, timeout=5.0) as response:
        result = json.load(response)

    matches: list[tuple[tuple[int, int, int], dict[str, str]]] = []
    for candidate in result.get("results", []):
        title = str(candidate.get("name", ""))
        title_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
        if not search_tokens.issubset(title_tokens):
            continue
        if title_tokens.intersection(excluded_terms):
            continue
        license_data = candidate.get("license") or {}
        uid = candidate.get("uid")
        if not uid or not license_data.get("label"):
            continue
        gltf_archive = (candidate.get("archives") or {}).get("gltf") or {}
        face_count = int(gltf_archive.get("faceCount") or 0)
        archive_size = int(gltf_archive.get("size") or 0)
        if (
            not face_count
            or not archive_size
            or face_count > 100_000
            or archive_size > 12_000_000
        ):
            continue
        user = candidate.get("user") or {}
        model_data = {
            "uid": str(uid),
            "name": title,
            "author": str(user.get("displayName") or user.get("username") or "Creator"),
            "license": str(license_data["label"]),
            "source_url": str(
                candidate.get("viewerUrl")
                or f"https://sketchfab.com/3d-models/{uid}"
            ),
        }
        extra_token_count = len(title_tokens - search_tokens)
        matches.append(
            ((extra_token_count, face_count, archive_size), model_data)
        )

    return min(matches, key=lambda match: match[0])[1] if matches else None


def find_vehicle_3d_model(brand: str, model: str) -> dict[str, str] | None:
    try:
        return fetch_vehicle_3d_model(model_family_name(brand, model))
    except (OSError, TimeoutError, ValueError):
        return None


@st.cache_data(ttl=86_400, show_spinner=False)
def find_vehicle_image(brand: str, model: str) -> dict[str, str] | None:
    search_name = model_family_name(brand, model)
    parameters = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{search_name} automobile",
        "gsrnamespace": "6",
        "gsrlimit": "8",
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": "1000",
        "format": "json",
        "formatversion": "2",
    }
    request = urllib.request.Request(
        "https://commons.wikimedia.org/w/api.php?"
        + urllib.parse.urlencode(parameters),
        headers={
            "User-Agent": (
                "AutoValueIndia/1.0 "
                "(https://github.com/manishh28/Car-Price-Prediction)"
            )
        },
    )
    excluded_terms = {
        "badge", "dashboard", "diagram", "engine", "interior", "logo", "wheel"
    }

    try:
        with urllib.request.urlopen(request, timeout=3.5) as response:
            result = json.load(response)
    except (OSError, TimeoutError, ValueError):
        return None

    for page in result.get("query", {}).get("pages", []):
        title = str(page.get("title", ""))
        if any(term in title.lower() for term in excluded_terms):
            continue
        image_info = page.get("imageinfo", [{}])[0]
        if image_info.get("mime") not in {"image/jpeg", "image/png", "image/webp"}:
            continue
        thumbnail_url = image_info.get("thumburl")
        source_url = image_info.get("descriptionurl")
        if not thumbnail_url or not source_url:
            continue
        metadata = image_info.get("extmetadata", {})
        return {
            "url": thumbnail_url,
            "source_url": source_url,
            "artist": plain_metadata(
                metadata.get("Artist", {}).get("value", ""), "Commons contributor"
            ),
            "license": plain_metadata(
                metadata.get("LicenseShortName", {}).get("value", ""),
                "Wikimedia Commons",
            ),
        }

    return None


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["mileage_value"] = extract_number(data["mileage"])
    data["engine_value"] = extract_number(data["engine"])
    data["max_power_value"] = extract_number(data["max_power"])
    data["brand"] = data["name"].astype(str).str.split().str[0]
    return data.dropna(subset=["name", "brand", "year", "selling_price"])


@st.cache_resource(show_spinner="Preparing the valuation model...")
def train_model() -> tuple[Pipeline, dict[str, float]]:
    data = load_data()
    X = data[FEATURES]
    y = np.log1p(data["selling_price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=8.0, solver="lsqr")),
        ]
    )

    pipeline.fit(X_train, y_train)
    test_predictions = np.expm1(pipeline.predict(X_test))
    actual_prices = np.expm1(y_test)
    metrics = {
        "mae": float(mean_absolute_error(actual_prices, test_predictions)),
        "r2": float(r2_score(actual_prices, test_predictions)),
    }

    pipeline.fit(X, y)
    return pipeline, metrics


def format_inr(value: float) -> str:
    rounded = str(max(0, int(round(value))))
    if len(rounded) <= 3:
        grouped = rounded
    else:
        grouped = rounded[-3:]
        prefix = rounded[:-3]
        while prefix:
            grouped = f"{prefix[-2:]},{grouped}"
            prefix = prefix[:-2]
    return f"INR {grouped}"


def common_value(values: pd.Series, fallback: str) -> str:
    modes = values.dropna().mode()
    return str(modes.iloc[0]) if not modes.empty else fallback


def median_value(values: pd.Series, fallback: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else fallback


def predict_price(pipeline: Pipeline, values: dict[str, object]) -> float:
    sample = pd.DataFrame([values])
    return max(0.0, float(np.expm1(pipeline.predict(sample[FEATURES])[0])))


data = load_data()
model, model_metrics = train_model()

st.markdown('<div class="app-kicker">AutoValue India</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="app-title">Used car value estimator</h1>', unsafe_allow_html=True
)
st.markdown(
    f"""
    <p class="app-intro">
        Estimate a fair resale value using {len(data):,} real used-car listings.
        Choose the exact variant, add its condition details, and compare the result
        with similar cars in the dataset.
    </p>
    <div class="accent-line"></div>
    """,
    unsafe_allow_html=True,
)

summary_columns = st.columns(3)
summary_columns[0].metric("Listings analysed", f"{len(data):,}")
summary_columns[1].metric("Makes covered", f"{data['brand'].nunique():,}")
summary_columns[2].metric(
    "Listing years", f"{int(data['year'].min())}-{int(data['year'].max())}"
)

st.markdown('<div style="height: 1.5rem"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Select your vehicle</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Start with the make, then choose the matching variant.</p>',
    unsafe_allow_html=True,
)

brand_counts = data["brand"].value_counts()
brands = sorted(brand_counts.index.tolist())
default_brand = "Toyota" if "Toyota" in brands else brands[0]

if st.session_state.get("selected_brand") not in brands:
    st.session_state["selected_brand"] = default_brand

vehicle_columns = st.columns([1, 2])
with vehicle_columns[0]:
    selected_brand = st.selectbox("Make", brands, key="selected_brand")

available_models = sorted(
    data.loc[data["brand"] == selected_brand, "name"].dropna().unique().tolist()
)
fortuner_models = [name for name in available_models if "Fortuner" in name]
default_model = fortuner_models[0] if fortuner_models else available_models[0]
if st.session_state.get("selected_model") not in available_models:
    st.session_state["selected_model"] = default_model

with vehicle_columns[1]:
    selected_model = st.selectbox(
        "Model and variant", available_models, key="selected_model"
    )

matching_cars = data[data["name"] == selected_model]

typical_specs = {
    "year": int(median_value(matching_cars["year"], 2018)),
    "km_driven": int(median_value(matching_cars["km_driven"], 50_000)),
    "mileage": median_value(matching_cars["mileage_value"], 18.0),
    "engine": int(median_value(matching_cars["engine_value"], 1200)),
    "max_power": median_value(matching_cars["max_power_value"], 80.0),
    "seats": int(median_value(matching_cars["seats"], 5)),
    "fuel": common_value(matching_cars["fuel"], "Petrol"),
    "seller_type": common_value(matching_cars["seller_type"], "Individual"),
    "transmission": common_value(matching_cars["transmission"], "Manual"),
    "owner": common_value(matching_cars["owner"], "First Owner"),
}

if st.session_state.get("specs_for_model") != selected_model:
    st.session_state.update(
        {
            "spec_year": typical_specs["year"],
            "spec_km": typical_specs["km_driven"],
            "spec_mileage": typical_specs["mileage"],
            "spec_engine": typical_specs["engine"],
            "spec_power": typical_specs["max_power"],
            "spec_seats": typical_specs["seats"],
            "spec_fuel": typical_specs["fuel"],
            "spec_seller": typical_specs["seller_type"],
            "spec_transmission": typical_specs["transmission"],
            "spec_owner": typical_specs["owner"],
            "specs_for_model": selected_model,
        }
    )

safe_brand = html.escape(selected_brand)
safe_model = html.escape(selected_model)
safe_fuel = html.escape(str(typical_specs["fuel"]))
safe_transmission = html.escape(str(typical_specs["transmission"]))
road_markers = "".join("<span></span>" for _ in range(14))
vehicle_3d_model = find_vehicle_3d_model(selected_brand, selected_model)

if vehicle_3d_model:
    model_uid = html.escape(vehicle_3d_model["uid"], quote=True)
    viewer_parameters = urllib.parse.urlencode(
        {
            "autostart": "1",
            "autospin": "0.25",
            "camera": "0",
            "dnt": "1",
            "ui_hint": "0",
            "ui_infos": "1",
            "ui_controls": "1",
        }
    )
    viewer_url = f"https://sketchfab.com/models/{model_uid}/embed?{viewer_parameters}"
    motion_window_class = "vehicle-motion-window is-3d"
    vehicle_visual_markup = (
        f'<iframe class="vehicle-3d-frame" src="{viewer_url}" '
        f'title="Interactive 3D model of {safe_model}" '
        'allow="autoplay; fullscreen; xr-spatial-tracking" allowfullscreen '
        'loading="eager"></iframe>'
    )
    image_credit_markup = ""
    road_markup = ""
else:
    vehicle_image = find_vehicle_image(selected_brand, selected_model)

if not vehicle_3d_model and vehicle_image:
    vehicle_asset_uri = html.escape(vehicle_image["url"], quote=True)
    motion_window_class = "vehicle-motion-window is-photo"
    image_alt = f"Reference photo of {safe_model}"
    vehicle_visual_markup = f'<img src="{vehicle_asset_uri}" alt="{image_alt}">'
    source_url = html.escape(vehicle_image["source_url"], quote=True)
    artist = html.escape(vehicle_image["artist"])
    license_name = html.escape(vehicle_image["license"])
    image_credit_markup = (
        f'<a class="vehicle-image-credit" href="{source_url}" target="_blank" '
        f'rel="noopener noreferrer">Photo: {artist} &middot; {license_name}</a>'
    )
    road_markup = (
        f'<div class="road-marker-track" aria-hidden="true">{road_markers}</div>'
    )
elif not vehicle_3d_model:
    vehicle_asset_uri = load_vehicle_asset()
    motion_window_class = "vehicle-motion-window"
    image_alt = f"Stylized vehicle illustration for {safe_model}"
    vehicle_visual_markup = f'<img src="{vehicle_asset_uri}" alt="{image_alt}">'
    image_credit_markup = ""
    road_markup = (
        f'<div class="road-marker-track" aria-hidden="true">{road_markers}</div>'
    )

st.markdown(
    f"""
    <div class="vehicle-stage">
        <div class="vehicle-stage-copy">
            <div class="vehicle-stage-kicker">Selected {safe_brand} profile</div>
            <div class="vehicle-stage-name">{safe_model}</div>
            <div class="vehicle-stage-facts">
                <div><strong>{typical_specs["year"]}</strong> typical listing year</div>
                <div><strong>{safe_fuel}</strong> &middot; <strong>{safe_transmission}</strong></div>
                <div><strong>{len(matching_cars):,}</strong> exact-match records</div>
            </div>
        </div>
        <div class="{motion_window_class}">
            {vehicle_visual_markup}
        </div>
        {road_markup}
        {image_credit_markup}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div style="height: 0.8rem"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-heading">Condition and specifications</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-copy">Typical values for the selected variant are filled in automatically.</p>',
    unsafe_allow_html=True,
)

context_column, reset_column = st.columns([3, 1])
context_column.caption(
    f"{len(matching_cars):,} matching listing records found for this exact variant."
)
if reset_column.button("Reset typical specs", use_container_width=True):
    st.session_state.update(
        {
            "spec_year": typical_specs["year"],
            "spec_km": typical_specs["km_driven"],
            "spec_mileage": typical_specs["mileage"],
            "spec_engine": typical_specs["engine"],
            "spec_power": typical_specs["max_power"],
            "spec_seats": typical_specs["seats"],
            "spec_fuel": typical_specs["fuel"],
            "spec_seller": typical_specs["seller_type"],
            "spec_transmission": typical_specs["transmission"],
            "spec_owner": typical_specs["owner"],
        }
    )
    st.session_state.pop("valuation", None)
    st.rerun()

with st.form("valuation_form"):
    row_one = st.columns(3)
    with row_one[0]:
        year = st.number_input(
            "Registration year", min_value=1980, max_value=2030, step=1, key="spec_year"
        )
    with row_one[1]:
        km_driven = st.number_input(
            "Kilometres driven",
            min_value=0,
            max_value=1_500_000,
            step=5_000,
            key="spec_km",
        )
    with row_one[2]:
        owner = st.selectbox(
            "Ownership",
            [
                "First Owner",
                "Second Owner",
                "Third Owner",
                "Fourth & Above Owner",
                "Test Drive Car",
            ],
            key="spec_owner",
        )

    row_two = st.columns(3)
    with row_two[0]:
        fuel = st.selectbox(
            "Fuel", ["Petrol", "Diesel", "CNG", "LPG"], key="spec_fuel"
        )
    with row_two[1]:
        transmission = st.selectbox(
            "Transmission", ["Manual", "Automatic"], key="spec_transmission"
        )
    with row_two[2]:
        seller_type = st.selectbox(
            "Seller",
            ["Individual", "Dealer", "Trustmark Dealer"],
            key="spec_seller",
        )

    row_three = st.columns(4)
    with row_three[0]:
        mileage = st.number_input(
            "Mileage (km/l)", min_value=1.0, max_value=60.0, step=0.5, key="spec_mileage"
        )
    with row_three[1]:
        engine = st.number_input(
            "Engine (cc)", min_value=300, max_value=7000, step=50, key="spec_engine"
        )
    with row_three[2]:
        max_power = st.number_input(
            "Power (bhp)", min_value=10.0, max_value=1000.0, step=5.0, key="spec_power"
        )
    with row_three[3]:
        seats = st.number_input(
            "Seats", min_value=2, max_value=14, step=1, key="spec_seats"
        )

    submitted = st.form_submit_button(
        "Estimate resale value", use_container_width=True
    )

if submitted:
    input_values = {
        "name": selected_model,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "year": year,
        "km_driven": km_driven,
        "mileage_value": mileage,
        "engine_value": engine,
        "max_power_value": max_power,
        "seats": seats,
    }
    prediction = predict_price(model, input_values)

    comparison = matching_cars
    if len(comparison) < 4:
        comparison = data[data["brand"] == selected_brand]

    st.session_state["valuation"] = {
        "prediction": prediction,
        "median": float(comparison["selling_price"].median()),
        "low": float(comparison["selling_price"].quantile(0.25)),
        "high": float(comparison["selling_price"].quantile(0.75)),
        "sample_size": int(len(comparison)),
        "model": selected_model,
        "inputs": input_values,
        "scenario_token": st.session_state.get("scenario_token", 0) + 1,
    }
    st.session_state["scenario_token"] = st.session_state["valuation"][
        "scenario_token"
    ]

    history = st.session_state.get("estimate_history", [])
    history.insert(
        0,
        {
            "Vehicle": selected_model,
            "Year": int(year),
            "Kilometres": f"{int(km_driven):,}",
            "Estimate": format_inr(prediction),
        },
    )
    st.session_state["estimate_history"] = history[:6]

valuation = st.session_state.get("valuation")
if valuation and valuation["model"] == selected_model:
    prediction = valuation["prediction"]
    market_median = valuation["median"]
    delta_percent = (
        ((prediction - market_median) / market_median) * 100 if market_median else 0
    )
    relation = "above" if delta_percent >= 0 else "below"

    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-label">Estimated resale value</div>
            <div class="result-price">{format_inr(prediction)}</div>
            <p class="result-note">
                {abs(delta_percent):.1f}% {relation} the median of the closest comparable
                listings. Use this as a negotiation benchmark, not a guaranteed sale price.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_columns = st.columns(3)
    result_columns[0].metric("Comparable median", format_inr(market_median))
    result_columns[1].metric(
        "Middle market range",
        f"{format_inr(valuation['low']).replace('INR ', '')} - {format_inr(valuation['high']).replace('INR ', '')}",
    )
    result_columns[2].metric("Comparable listings", f"{valuation['sample_size']:,}")

    st.markdown('<div style="height: 1rem"></div>', unsafe_allow_html=True)
    market_tab, scenario_tab, history_tab = st.tabs(
        ["Market view", "What-if explorer", "Recent estimates"]
    )

    with market_tab:
        st.markdown(
            '<div class="section-heading">Market position</div>',
            unsafe_allow_html=True,
        )
        st.caption("Compare the model estimate with the closest listing benchmark.")

        chart_values = [prediction, market_median]
        chart_labels = ["Your estimate", "Comparable median"]
        fig, ax = plt.subplots(figsize=(8, 2.35))
        fig.patch.set_facecolor("#f4f7f6")
        ax.set_facecolor("#f4f7f6")
        bars = ax.barh(
            chart_labels,
            chart_values,
            color=["#146c63", "#aebbb8"],
            height=0.48,
        )
        ax.invert_yaxis()
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(axis="x", colors="#60706c", labelsize=9, length=0)
        ax.tick_params(axis="y", colors="#16211f", labelsize=10, length=0, pad=10)
        ax.grid(axis="x", color="#d8e0de", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value / 100000:.1f}L")
        )
        for bar, value in zip(bars, chart_values):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f"  {format_inr(value)}",
                va="center",
                color="#16211f",
                fontsize=9,
                fontweight="bold",
            )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with scenario_tab:
        st.markdown(
            '<div class="section-heading">Explore another ownership scenario</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Move either control to see how year and kilometres change this model estimate."
        )

        original_inputs = valuation["inputs"]
        scenario_columns = st.columns(2)
        with scenario_columns[0]:
            scenario_year = st.slider(
                "Registration year",
                min_value=1983,
                max_value=2026,
                value=int(original_inputs["year"]),
                key=f"scenario_year_{valuation['scenario_token']}",
            )
        with scenario_columns[1]:
            current_km = int(original_inputs["km_driven"])
            scenario_km = st.slider(
                "Kilometres driven",
                min_value=0,
                max_value=min(1_500_000, max(300_000, current_km + 100_000)),
                value=current_km,
                step=5_000,
                key=f"scenario_km_{valuation['scenario_token']}",
            )

        scenario_inputs = dict(original_inputs)
        scenario_inputs["year"] = scenario_year
        scenario_inputs["km_driven"] = scenario_km
        scenario_prediction = predict_price(model, scenario_inputs)
        scenario_delta = scenario_prediction - prediction
        scenario_percent = (scenario_delta / prediction * 100) if prediction else 0

        live_columns = st.columns(2)
        live_columns[0].metric("Scenario estimate", format_inr(scenario_prediction))
        live_columns[1].metric(
            "Change from estimate",
            format_inr(abs(scenario_delta)).replace("INR ", ""),
            delta=f"{scenario_percent:+.1f}%",
        )
        change_class = (
            "live-change-positive" if scenario_delta >= 0 else "live-change-negative"
        )
        direction = "higher" if scenario_delta >= 0 else "lower"
        st.markdown(
            f'<div class="{change_class}">This scenario is {abs(scenario_percent):.1f}% {direction} than your submitted estimate.</div>',
            unsafe_allow_html=True,
        )

    with history_tab:
        st.markdown(
            '<div class="section-heading">Recent estimates</div>',
            unsafe_allow_html=True,
        )
        history = st.session_state.get("estimate_history", [])
        if history:
            st.dataframe(
                pd.DataFrame(history),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("Your recent estimates will appear here during this session.")

st.markdown(
    f"""
    <div class="data-note">
        Estimates are generated from the repository dataset and may not reflect
        location, service history, accident history, modifications, or current local demand.
        Holdout model score: R² {model_metrics['r2']:.2f}; median absolute error benchmark:
        {format_inr(model_metrics['mae'])}.
    </div>
    """,
    unsafe_allow_html=True,
)
