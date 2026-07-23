import html
from pathlib import Path

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

from vehicle_search import rank_matches


DATA_PATH = Path(__file__).with_name("car details.csv")
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

    header[data-testid="stHeader"] {
        display: none;
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
    div[data-testid="stTextInput"] input,
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
    div[data-testid="stTextInput"] input,
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
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--teal);
        box-shadow: 0 0 0 1px var(--teal);
    }

    .search-match {
        background: #eaf2f0;
        border-left: 3px solid var(--teal);
        color: var(--ink);
        font-size: 0.88rem;
        line-height: 1.45;
        margin: 0.25rem 0 1.25rem;
        padding: 0.7rem 0.85rem;
    }

    .search-match strong {
        color: var(--teal-dark);
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
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["mileage_value"] = extract_number(data["mileage"])
    data["engine_value"] = extract_number(data["engine"])
    data["max_power_value"] = extract_number(data["max_power"])
    names = data["name"].astype(str)
    data["brand"] = names.str.split().str[0]
    data.loc[names.str.startswith("Land Rover "), "brand"] = "Land Rover"
    data.loc[names.str.startswith("Ashok Leyland "), "brand"] = "Ashok Leyland"
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
        Search for a vehicle, add its condition details, and compare the result
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
st.markdown('<div class="section-heading">Search for your vehicle</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="section-copy">Search the brand first, then enter a model or variant.</p>',
    unsafe_allow_html=True,
)

brand_counts = data["brand"].value_counts()
brands = sorted(brand_counts.index.tolist())
default_brand = "Toyota" if "Toyota" in brands else brands[0]

def clear_model_search() -> None:
    st.session_state["vehicle_model_search"] = ""
    st.session_state.pop("valuation", None)

vehicle_columns = st.columns([1, 2])
with vehicle_columns[0]:
    brand_query = st.text_input(
        "Search car brand",
        value=default_brand,
        placeholder="e.g. Toyota, BMW, Land Rover",
        key="vehicle_brand_search",
        on_change=clear_model_search,
    )
with vehicle_columns[1]:
    model_query = st.text_input(
        "Search car model",
        value="Fortuner",
        placeholder="e.g. Fortuner, 3 Series, Swift",
        key="vehicle_model_search",
    )

brand_matches = rank_matches(brands, brand_query, brand_counts.to_dict())
if not brand_matches:
    st.warning("No matching brand found. Check the spelling and try again.")
    st.stop()

selected_brand = brand_matches[0]

available_models = sorted(
    data.loc[data["brand"] == selected_brand, "name"].dropna().unique().tolist()
)
model_counts = (
    data.loc[data["brand"] == selected_brand, "name"].value_counts().to_dict()
)

model_matches = rank_matches(available_models, model_query, model_counts)
if not model_matches:
    st.info(
        f"Enter a model name to search the {len(available_models):,} "
        f"{selected_brand} variants in the dataset."
    )
    st.stop()

selected_model = model_matches[0]
match_note = (
    f"{len(model_matches):,} matches found; showing the closest match."
    if len(model_matches) > 1
    else "1 matching variant found."
)
st.markdown(
    f"""
    <div class="search-match">
        <strong>Matched vehicle:</strong> {html.escape(selected_model)}
        &nbsp;&middot;&nbsp; {match_note}
    </div>
    """,
    unsafe_allow_html=True,
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

st.markdown(
    '<div class="section-heading">Condition and specifications</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="section-copy">Typical values for the matched vehicle are filled in automatically.</p>',
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
