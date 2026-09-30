import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ScaleWise — AI Bioprocess Copilot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 1rem;
    padding-bottom: 0.5rem;
    max-width: 1500px;
}

.main-title {
    font-size: 32px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 2px;
}

.subtitle {
    font-size: 14px;
    color: #8b96a5;
    margin-bottom: 12px;
}

.section-title {
    font-size: 18px;
    font-weight: 750;
    margin-top: 10px;
    margin-bottom: 8px;
}

.metric-card {
    padding: 10px;
    border-radius: 10px;
    border: 1px solid rgba(120,130,150,0.25);
    background: rgba(100,110,130,0.08);
    text-align: center;
    min-height: 76px;
}

.metric-title {
    font-size: 12px;
    color: #8b96a5;
}

.metric-value {
    font-size: 22px;
    font-weight: 750;
    margin-top: 4px;
}

.risk-high {
    padding: 12px;
    border-radius: 10px;
    background: rgba(210,70,70,0.14);
    border-left: 4px solid #d65c5c;
}

.risk-medium {
    padding: 12px;
    border-radius: 10px;
    background: rgba(210,160,40,0.14);
    border-left: 4px solid #d6ad36;
}

.risk-low {
    padding: 12px;
    border-radius: 10px;
    background: rgba(40,160,100,0.14);
    border-left: 4px solid #35b77a;
}

.info-card {
    padding: 12px;
    border-radius: 10px;
    background: rgba(70,120,180,0.10);
    border-left: 4px solid #4c9be8;
}

.warning-card {
    padding: 12px;
    border-radius: 10px;
    background: rgba(210,160,40,0.12);
    border-left: 4px solid #d6ad36;
}

.chat-box {
    border-radius: 10px;
    border: 1px solid rgba(100,130,170,0.30);
    padding: 8px;
}

div[data-testid="stMetric"] {
    padding: 4px;
}

div[data-testid="stMetricValue"] {
    font-size: 20px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILENAME = "Final Data for Hackathon.xlsx"

MODEL_TARGETS = [
    "VCD_million_cells_mL",
    "viability_percent",
    "growth_rate_per_h",
    "glucose_g_L",
    "lactate_g_L"
]

MODEL_FEATURES = [
    "cell_line",
    "scale_L",
    "batch_age_h",
    "rpm",
    "aeration_vvm",
    "temperature_C",
    "pH",
    "DO_percent",
    "initial_VCD_million_cells_mL",
    "feed_rate_mL_h",
    "PV_W_L",
    "kLa_per_h",
    "mixing_time_s",
    "OTR_mmol_L_h",
    "tank_diameter_m",
    "tank_height_m",
    "impeller_diameter_m",
    "reynolds_number",
    "tip_speed_m_s",
    "impeller_type"
]

PROCESS_CATEGORICAL = [
    "cell_line",
    "impeller_type"
]

PHYSICS_COLUMNS = [
    "tank_diameter_m",
    "tank_height_m",
    "impeller_diameter_m",
    "PV_W_L",
    "kLa_per_h",
    "mixing_time_s",
    "OTR_mmol_L_h",
    "reynolds_number",
    "tip_speed_m_s"
]


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_data_file():

    direct = BASE_DIR / DATA_FILENAME

    if direct.exists():
        return direct

    matches = list(
        BASE_DIR.rglob(DATA_FILENAME)
    )

    if matches:
        return matches[0]

    # Case-insensitive fallback
    for path in BASE_DIR.rglob("*"):
        if path.is_file() and path.name.lower() == DATA_FILENAME.lower():
            return path

    return None


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset():

    path = find_data_file()

    if path is None:

        raise FileNotFoundError(
            f"{DATA_FILENAME} was not found. "
            "Upload the Excel file to the same GitHub repository "
            "as app.py."
        )

    df = pd.read_excel(path)

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df, str(path)


try:

    df, data_path = load_dataset()

except Exception as e:

    st.error("### ❌ Data file error")

    st.error(str(e))

    st.info(
        "Your GitHub repository should contain:\n\n"
        "app.py\n\n"
        "Final Data for Hackathon.xlsx\n\n"
        "requirements.txt"
    )

    st.stop()


# ============================================================
# VALIDATE DATASET
# ============================================================

required_columns = sorted(
    set(MODEL_FEATURES + MODEL_TARGETS)
)

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:

    st.error(
        "The Excel dataset is missing required columns:"
    )

    st.code(
        "\n".join(missing_columns)
    )

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

for col in MODEL_FEATURES + MODEL_TARGETS:

    if col not in PROCESS_CATEGORICAL:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=MODEL_TARGETS
).copy()


# ============================================================
# SCALE INFORMATION
# ============================================================

observed_scales = sorted(
    pd.to_numeric(
        df["scale_L"],
        errors="coerce"
    )
    .dropna()
    .unique()
    .tolist()
)

observed_scales = [
    float(x)
    for x in observed_scales
]


# ============================================================
# DATA-DERIVED INPUT RANGES
# ============================================================

def percentile_range(
    column,
    lower=0.01,
    upper=0.99
):

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return 0.0, 1.0

    lo = float(
        values.quantile(lower)
    )

    hi = float(
        values.quantile(upper)
    )

    if lo == hi:
        lo = float(values.min())
        hi = float(values.max())

    return lo, hi


def safe_default(column):

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return 0.0

    return float(
        values.median()
    )


input_ranges = {}

for col in [
    "batch_age_h",
    "rpm",
    "aeration_vvm",
    "temperature_C",
    "pH",
    "DO_percent",
    "initial_VCD_million_cells_mL",
    "feed_rate_mL_h"
]:

    input_ranges[col] = percentile_range(
        col
    )


# ============================================================
# UNIQUE CATEGORIES
# ============================================================

cell_lines = sorted(
    df["cell_line"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

impeller_types = sorted(
    df["impeller_type"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


if not cell_lines:
    cell_lines = ["Chicken fibroblast"]

if not impeller_types:
    impeller_types = ["Pitched-blade"]


# ============================================================
# OPTIONAL CATEGORIES
# ============================================================

if "medium" in df.columns:

    medium_options = sorted(
        df["medium"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    medium_options = []


if "serum_condition" in df.columns:

    serum_options = sorted(
        df["serum_condition"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    serum_options = []


# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource(show_spinner="Training ScaleWise biological models...")

def train_models(data):

    X = data[
        MODEL_FEATURES
    ].copy()

    models = {}

    categorical = [
        c for c in MODEL_FEATURES
        if c in PROCESS_CATEGORICAL
    ]

    numerical = [
        c for c in MODEL_FEATURES
        if c not in categorical
    ]

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical
            )
        ],
        remainder="drop"
    )

    for target in MODEL_TARGETS:

        y = pd.to_numeric(
            data[target],
            errors="coerce"
        )

        valid = y.notna()

        X_target = X.loc[valid]
        y_target = y.loc[valid]

        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=180,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ]
        )

        model.fit(
            X_target,
            y_target
        )

        models[target] = model

    return models


with st.spinner("Preparing biological prediction engine..."):

    biological_models = train_models(df)


# ============================================================
# PHYSICS ENGINE
# ============================================================

@st.cache_data(show_spinner=False)
def build_scale_reference(data):

    numeric_physics = [
        "tank_diameter_m",
        "tank_height_m",
        "impeller_diameter_m",
        "PV_W_L",
        "kLa_per_h",
        "mixing_time_s",
        "OTR_mmol_L_h",
        "reynolds_number",
        "tip_speed_m_s",
        "rpm",
        "aeration_vvm",
        "DO_percent",
        "temperature_C",
        "pH",
        "feed_rate_mL_h"
    ]

    rows = []

    for scale in observed_scales:

        subset = data[
            np.isclose(
                data["scale_L"],
                scale
            )
        ]

        if len(subset) == 0:
            continue

        row = {
            "scale_L": scale
        }

        for col in numeric_physics:

            if col in subset.columns:

                row[col] = float(
                    pd.to_numeric(
                        subset[col],
                        errors="coerce"
                    ).median()
                )

        if "impeller_type" in subset.columns:

            modes = (
                subset["impeller_type"]
                .dropna()
                .astype(str)
            )

            if len(modes):

                row["impeller_type"] = (
                    modes.mode().iloc[0]
                )

        rows.append(row)

    return pd.DataFrame(rows)


scale_reference = build_scale_reference(
    df
)


def log_interpolate(
    target_scale,
    scales,
    values
):

    scales = np.asarray(
        scales,
        dtype=float
    )

    values = np.asarray(
        values,
        dtype=float
    )

    valid = (
        np.isfinite(scales)
        &
        np.isfinite(values)
        &
        (scales > 0)
    )

    scales = scales[valid]
    values = values[valid]

    if len(scales) == 0:
        return np.nan

    order = np.argsort(scales)

    scales = scales[order]
    values = values[order]

    if target_scale in scales:
        idx = np.where(
            scales == target_scale
        )[0][0]

        return float(
            values[idx]
        )

    if target_scale <= scales.min():

        x1 = np.log(scales[0])
        x2 = np.log(scales[1])

        y1 = values[0]
        y2 = values[1]

    elif target_scale >= scales.max():

        x1 = np.log(scales[-2])
        x2 = np.log(scales[-1])

        y1 = values[-2]
        y2 = values[-1]

    else:

        upper_idx = np.searchsorted(
            scales,
            target_scale
        )

        lower_idx = upper_idx - 1

        x1 = np.log(
            scales[lower_idx]
        )

        x2 = np.log(
            scales[upper_idx]
        )

        y1 = values[lower_idx]
        y2 = values[upper_idx]

    x = np.log(
        float(target_scale)
    )

    if x2 == x1:
        return float(y1)

    fraction = (
        (x - x1)
        /
        (x2 - x1)
    )

    return float(
        y1 + fraction * (y2 - y1)
    )


def target_scale_physics(
    target_scale,
    rpm,
    aeration_vvm,
    temperature_C,
    pH,
    DO_percent,
    feed_rate_mL_h,
    impeller_type
):

    result = {}

    result["scale_L"] = float(
        target_scale
    )

    # --------------------------------------------------------
    # Interpolate engineering reference values
    # --------------------------------------------------------

    numeric_cols = [
        "tank_diameter_m",
        "tank_height_m",
        "impeller_diameter_m",
        "PV_W_L",
        "kLa_per_h",
        "mixing_time_s",
        "OTR_mmol_L_h",
        "reynolds_number",
        "tip_speed_m_s"
    ]

    for col in numeric_cols:

        if col not in scale_reference.columns:
            result[col] = np.nan
            continue

        result[col] = log_interpolate(
            target_scale,
            scale_reference["scale_L"].values,
            scale_reference[col].values
        )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    D = result["impeller_diameter_m"]

    tank_D = result["tank_diameter_m"]

    # --------------------------------------------------------
    # Operating condition overrides
    # --------------------------------------------------------

    reference_rpm = log_interpolate(
        target_scale,
        scale_reference["scale_L"].values,
        scale_reference["rpm"].values
    )

    reference_aeration = log_interpolate(
        target_scale,
        scale_reference["scale_L"].values,
        scale_reference["aeration_vvm"].values
    )

    reference_DO = log_interpolate(
        target_scale,
        scale_reference["scale_L"].values,
        scale_reference["DO_percent"].values
    )

    reference_temp = log_interpolate(
        target_scale,
        scale_reference["scale_L"].values,
        scale_reference["temperature_C"].values
    )

    reference_pH = log_interpolate(
        target_scale,
        scale_reference["scale_L"].values,
        scale_reference["pH"].values
    )

    # --------------------------------------------------------
    # P/V scaling
    # --------------------------------------------------------

    base_pv = result["PV_W_L"]

    if (
        base_pv is not None
        and np.isfinite(base_pv)
        and reference_rpm
        and reference_rpm > 0
    ):

        rpm_ratio = (
            float(rpm)
            /
            float(reference_rpm)
        )

        result["PV_W_L"] = (
            base_pv
            *
            rpm_ratio ** 3
        )

    # --------------------------------------------------------
    # kLa scaling
    # --------------------------------------------------------

    base_kla = result["kLa_per_h"]

    if (
        np.isfinite(base_kla)
        and reference_rpm
        and reference_rpm > 0
    ):

        rpm_ratio = (
            float(rpm)
            /
            float(reference_rpm)
        )

        aeration_ratio = (
            float(aeration_vvm)
            /
            max(
                float(reference_aeration),
                1e-6
            )
        )

        result["kLa_per_h"] = (
            base_kla
            *
            max(rpm_ratio, 0.01) ** 0.5
            *
            max(aeration_ratio, 0.01) ** 0.35
        )

    # --------------------------------------------------------
    # Mixing time
    # --------------------------------------------------------

    base_mixing = result["mixing_time_s"]

    if (
        np.isfinite(base_mixing)
        and reference_rpm
        and reference_rpm > 0
    ):

        rpm_ratio = (
            float(rpm)
            /
            float(reference_rpm)
        )

        result["mixing_time_s"] = (
            base_mixing
            /
            max(
                rpm_ratio,
                0.1
            ) ** 0.5
        )

    # --------------------------------------------------------
    # Reynolds number
    # --------------------------------------------------------

    base_re = result["reynolds_number"]

    if (
        np.isfinite(base_re)
        and reference_rpm
        and reference_rpm > 0
    ):

        result["reynolds_number"] = (
            base_re
            *
            (
                float(rpm)
                /
                float(reference_rpm)
            )
        )

    # --------------------------------------------------------
    # Tip speed
    # --------------------------------------------------------

    if (
        np.isfinite(D)
        and D > 0
    ):

        result["tip_speed_m_s"] = (
            np.pi
            *
            D
            *
            float(rpm)
            /
            60.0
        )

    # --------------------------------------------------------
    # OTR
    # --------------------------------------------------------

    base_otr = result["OTR_mmol_L_h"]

    if (
        np.isfinite(base_otr)
        and np.isfinite(base_kla)
        and base_kla > 0
    ):

        kla_ratio = (
            result["kLa_per_h"]
            /
            base_kla
        )

        do_ratio = (
            float(DO_percent)
            /
            max(
                float(reference_DO),
                1e-6
            )
        )

        result["OTR_mmol_L_h"] = (
            base_otr
            *
            max(kla_ratio, 0.01)
            *
            max(do_ratio, 0.01)
        )

    # --------------------------------------------------------
    # User process variables
    # --------------------------------------------------------

    result["rpm"] = float(rpm)

    result["aeration_vvm"] = float(
        aeration_vvm
    )

    result["temperature_C"] = float(
        temperature_C
    )

    result["pH"] = float(pH)

    result["DO_percent"] = float(
        DO_percent
    )

    result["feed_rate_mL_h"] = float(
        feed_rate_mL_h
    )

    result["impeller_type"] = (
        impeller_type
    )

    # --------------------------------------------------------
    # Reference values
    # --------------------------------------------------------

    result["reference_rpm"] = reference_rpm
    result["reference_aeration_vvm"] = (
        reference_aeration
    )
    result["reference_DO_percent"] = (
        reference_DO
    )
    result["reference_temperature_C"] = (
        reference_temp
    )
    result["reference_pH"] = (
        reference_pH
    )

    return result


# ============================================================
# BIOLOGICAL PREDICTION
# ============================================================

def biological_prediction(
    physics,
    cell_line,
    batch_age_h,
    initial_vcd
):

    row = {

        "cell_line":
            cell_line,

        "scale_L":
            physics["scale_L"],

        "batch_age_h":
            batch_age_h,

        "rpm":
            physics["rpm"],

        "aeration_vvm":
            physics["aeration_vvm"],

        "temperature_C":
            physics["temperature_C"],

        "pH":
            physics["pH"],

        "DO_percent":
            physics["DO_percent"],

        "initial_VCD_million_cells_mL":
            initial_vcd,

        "feed_rate_mL_h":
            physics["feed_rate_mL_h"],

        "PV_W_L":
            physics["PV_W_L"],

        "kLa_per_h":
            physics["kLa_per_h"],

        "mixing_time_s":
            physics["mixing_time_s"],

        "OTR_mmol_L_h":
            physics["OTR_mmol_L_h"],

        "tank_diameter_m":
            physics["tank_diameter_m"],

        "tank_height_m":
            physics["tank_height_m"],

        "impeller_diameter_m":
            physics["impeller_diameter_m"],

        "reynolds_number":
            physics["reynolds_number"],

        "tip_speed_m_s":
            physics["tip_speed_m_s"],

        "impeller_type":
            physics["impeller_type"]
    }

    X = pd.DataFrame(
        [row],
        columns=MODEL_FEATURES
    )

    predictions = {}

    for target, model in biological_models.items():

        value = model.predict(X)[0]

        predictions[target] = float(
            value
        )

    # Physical sanity bounds
    predictions[
        "viability_percent"
    ] = float(
        np.clip(
            predictions["viability_percent"],
            0,
            100
        )
    )

    predictions[
        "VCD_million_cells_mL"
    ] = max(
        0.0,
        predictions[
            "VCD_million_cells_mL"
        ]
    )

    return predictions


# ============================================================
# RISK ENGINE
# ============================================================

def range_deviation(
    value,
    series
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return 0.0

    lo = float(
        values.quantile(0.05)
    )

    hi = float(
        values.quantile(0.95)
    )

    if value < lo:

        return min(
            1.0,
            (lo - value)
            /
            max(
                abs(hi - lo),
                1e-9
            )
        )

    if value > hi:

        return min(
            1.0,
            (value - hi)
            /
            max(
                abs(hi - lo),
                1e-9
            )
        )

    return 0.0


def calculate_risk(
    target_scale,
    physics,
    predictions
):

    risks = []

    warnings_list = []

    # --------------------------------------------------------
    # 1. Scale interpolation / extrapolation
    # --------------------------------------------------------

    min_scale = min(
        observed_scales
    )

    max_scale = max(
        observed_scales
    )

    if (
        target_scale < min_scale
        or target_scale > max_scale
    ):

        scale_risk = 1.0

        warnings_list.append(
            "Target scale is outside the observed dataset range."
        )

    elif target_scale not in observed_scales:

        scale_risk = 0.20

        lower = max(
            [
                s for s in observed_scales
                if s < target_scale
            ],
            default=min_scale
        )

        upper = min(
            [
                s for s in observed_scales
                if s > target_scale
            ],
            default=max_scale
        )

        warnings_list.append(
            f"Target scale is interpolated between "
            f"{lower:g} L and {upper:g} L."
        )

    else:

        scale_risk = 0.0

    risks.append(
        (
            "Scale coverage",
            scale_risk,
            0.15
        )
    )

    # --------------------------------------------------------
    # 2. Operating envelope deviations
    # --------------------------------------------------------

    process_checks = [
        (
            "RPM",
            physics["rpm"],
            df["rpm"]
        ),
        (
            "Aeration",
            physics["aeration_vvm"],
            df["aeration_vvm"]
        ),
        (
            "DO",
            physics["DO_percent"],
            df["DO_percent"]
        ),
        (
            "Temperature",
            physics["temperature_C"],
            df["temperature_C"]
        ),
        (
            "pH",
            physics["pH"],
            df["pH"]
        )
    ]

    for label, value, series in process_checks:

        deviation = range_deviation(
            value,
            series
        )

        risks.append(
            (
                label,
                deviation,
                0.10
            )
        )

        if deviation > 0.35:

            warnings_list.append(
                f"{label} is outside the central "
                f"training-data operating envelope."
            )

    # --------------------------------------------------------
    # 3. Physics deviations
    # --------------------------------------------------------

    physics_checks = [
        (
            "P/V",
            physics["PV_W_L"],
            df["PV_W_L"]
        ),
        (
            "kLa",
            physics["kLa_per_h"],
            df["kLa_per_h"]
        ),
        (
            "Mixing time",
            physics["mixing_time_s"],
            df["mixing_time_s"]
        ),
        (
            "OTR",
            physics["OTR_mmol_L_h"],
            df["OTR_mmol_L_h"]
        ),
        (
            "Reynolds number",
            physics["reynolds_number"],
            df["reynolds_number"]
        ),
        (
            "Tip speed",
            physics["tip_speed_m_s"],
            df["tip_speed_m_s"]
        )
    ]

    for label, value, series in physics_checks:

        deviation = range_deviation(
            value,
            series
        )

        risks.append(
            (
                label,
                deviation,
                0.08
            )
        )

        if deviation > 0.40:

            warnings_list.append(
                f"{label} differs substantially "
                f"from the training-data envelope."
            )

    # --------------------------------------------------------
    # 4. Oxygen transfer warning
    # --------------------------------------------------------

    oxygen_risk = 0.0

    if physics["DO_percent"] < 30:
        oxygen_risk += 0.5

    if physics["kLa_per_h"] < 5:
        oxygen_risk += 0.25

    if physics["OTR_mmol_L_h"] < 0.5:
        oxygen_risk += 0.25

    oxygen_risk = min(
        1.0,
        oxygen_risk
    )

    risks.append(
        (
            "Oxygen transfer",
            oxygen_risk,
            0.15
        )
    )

    if oxygen_risk > 0.4:

        warnings_list.append(
            "Potential oxygen-transfer limitation; "
            "verify kLa, OTR and DO response experimentally."
        )

    # --------------------------------------------------------
    # 5. Biological sanity checks
    # --------------------------------------------------------

    biology_risk = 0.0

    if predictions[
        "viability_percent"
    ] < 80:

        biology_risk += 0.5

        warnings_list.append(
            "Predicted viability is below 80%."
        )

    if predictions[
        "growth_rate_per_h"
    ] < 0:

        biology_risk += 0.25

        warnings_list.append(
            "Predicted growth rate is negative."
        )

    if predictions[
        "lactate_g_L"
    ] > float(
        df["lactate_g_L"].quantile(0.95)
    ):

        biology_risk += 0.25

        warnings_list.append(
            "Predicted lactate is above the "
            "95th percentile of the training dataset."
        )

    biology_risk = min(
        1.0,
        biology_risk
    )

    risks.append(
        (
            "Biological condition",
            biology_risk,
            0.15
        )
    )

    # --------------------------------------------------------
    # Weighted risk
    # --------------------------------------------------------

    weighted = sum(
        value * weight
        for _, value, weight in risks
    )

    weight_sum = sum(
        weight
        for _, _, weight in risks
    )

    normalized = (
        weighted
        /
        max(
            weight_sum,
            1e-9
        )
    )

    risk_score = float(
        np.clip(
            normalized * 100,
            0,
            100
        )
    )

    # --------------------------------------------------------
    # Risk classification
    # --------------------------------------------------------

    if risk_score >= 70:

        risk_level = "HIGH"

        suggested_action = (
            "Pause scale transition and investigate the "
            "identified operating/engineering deviations "
            "before proceeding."
        )

        validation = (
            "Run a controlled target-scale engineering "
            "validation focusing on mixing, oxygen transfer, "
            "DO response and biological performance."
        )

    elif risk_score >= 40:

        risk_level = "MODERATE"

        suggested_action = (
            "Proceed only with controlled validation and "
            "enhanced monitoring of the identified risk drivers."
        )

        validation = (
            "Perform a target-scale confirmation run with "
            "replicate measurements of DO, kLa/OTR, mixing "
            "and cell-performance indicators."
        )

    else:

        risk_level = "LOW"

        suggested_action = (
            "The target condition is within the broader "
            "model-supported envelope. Continue with "
            "planned validation."
        )

        validation = (
            "Confirm biological and engineering performance "
            "at target scale before production use."
        )

    # --------------------------------------------------------
    # Evidence confidence
    # --------------------------------------------------------

    if (
        target_scale >= min_scale
        and target_scale <= max_scale
    ):

        coverage_confidence = 90.0

    else:

        coverage_confidence = 55.0

    warning_penalty = min(
        30,
        len(warnings_list) * 3
    )

    confidence = float(
        np.clip(
            coverage_confidence
            - warning_penalty,
            40,
            95
        )
    )

    # Unique warnings
    warnings_list = list(
        dict.fromkeys(
            warnings_list
        )
    )

    if len(warnings_list) == 0:

        warnings_list.append(
            "No major rule-based deviation detected."
        )

    # --------------------------------------------------------
    # Category scores
    # --------------------------------------------------------

    category_scores = {

        "Scale Coverage":
            100
            -
            scale_risk * 100,

        "Process Conditions":
            100
            -
            np.mean(
                [
                    x[1]
                    for x in risks
                    if x[0] in [
                        "RPM",
                        "Aeration",
                        "DO",
                        "Temperature",
                        "pH"
                    ]
                ]
            ) * 100,

        "Oxygen Transfer":
            100
            -
            oxygen_risk * 100,

        "Hydrodynamics":
            100
            -
            np.mean(
                [
                    x[1]
                    for x in risks
                    if x[0] in [
                        "P/V",
                        "kLa",
                        "Mixing time",
                        "OTR",
                        "Reynolds number",
                        "Tip speed"
                    ]
                ]
            ) * 100,

        "Biological Condition":
            100
            -
            biology_risk * 100
    }

    return {

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "confidence":
            confidence,

        "warnings":
            warnings_list,

        "suggested_action":
            suggested_action,

        "recommended_validation":
            validation,

        "category_scores":
            category_scores
    }


# ============================================================
# AI COPILOT
# ============================================================

def build_ai_context(
    inputs,
    physics,
    predictions,
    risk
):

    return {

        "user_inputs":
            inputs,

        "physics":
            physics,

        "biological_predictions":
            predictions,

        "risk_assessment":
            risk,

        "observed_scales_L":
            observed_scales,

        "dataset_rows":
            len(df),

        "important_note":
            "The dataset contains failure_event without "
            "sufficient positive failure labels for calibrated "
            "supervised failure probability. Therefore risk_score "
            "is a model-supported risk indicator, not a calibrated "
            "probability of failure."
    }


def ai_copilot_answer(
    question,
    context
):

    api_key = (
        st.secrets.get(
            "OPENAI_API_KEY",
            None
        )
        if hasattr(st, "secrets")
        else None
    )

    if not api_key:

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

    if not api_key:

        return (
            "### 🤖 ScaleWise Copilot\n\n"
            "The live AI Copilot is not connected because "
            "`OPENAI_API_KEY` has not been configured.\n\n"
            "Add the API key under **Streamlit → Settings → "
            "Secrets** using:\n\n"
            "```toml\n"
            'OPENAI_API_KEY = "your_api_key"\n'
            "```\n\n"
            "The dashboard itself remains fully functional."
        )

    try:

        from openai import OpenAI

        client = OpenAI(
            api_key=api_key
        )

        system_prompt = """
You are ScaleWise Copilot, an AI assistant for
scalable cell-culture bioprocess development.

You are given the current user inputs, scale-up physics,
biological model predictions and risk assessment.

Answer the user's question specifically about the current
scenario.

Rules:

1. Be technically precise.
2. Explain engineering and biological reasoning clearly.
3. Do not invent experimental results.
4. Do not present the risk score as a calibrated probability
   of failure.
5. Distinguish model prediction from experimental validation.
6. If a parameter is outside the training-data envelope,
   explicitly say so.
7. Give practical suggested validation steps when appropriate.
8. Do not claim that a model prediction proves process safety.
9. Keep responses concise enough for a dashboard.
"""

        user_prompt = (
            "CURRENT SCALEWISE SCENARIO:\n\n"
            +
            json.dumps(
                context,
                indent=2,
                default=str
            )
            +
            "\n\nUSER QUESTION:\n"
            +
            question
        )

        response = client.responses.create(

            model="gpt-5.6-luna",

            instructions=system_prompt,

            input=user_prompt
        )

        return response.output_text

    except Exception as e:

        return (
            "### ⚠️ AI Copilot connection error\n\n"
            f"`{str(e)}`\n\n"
            "The ScaleWise numerical prediction and risk "
            "engine remain available."
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="main-title">
🧬 ScaleWise — AI Bioprocess Copilot
</div>
<div class="subtitle">
Scale-up prediction • Process risk • Engineering validation • Live AI Copilot
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# MAIN USER INPUT + RESULT LAYOUT
# ============================================================

left, right = st.columns(
    [1.05, 2.6],
    gap="large"
)


# ============================================================
# LEFT — USER INPUT
# ============================================================

with left:

    st.markdown(
        '<div class="section-title">⚙️ Process Input</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Input ranges are derived from the training dataset. "
        "Target scales between observed scales are interpolated."
    )

    # --------------------------------------------------------
    # Target scale
    # --------------------------------------------------------

    target_scale = st.number_input(
        "Target scale (L)",
        min_value=float(
            min(observed_scales)
        ),
        max_value=float(
            max(observed_scales)
        ),
        value=50.0
        if 50.0 >= min(observed_scales)
        and 50.0 <= max(observed_scales)
        else float(observed_scales[0]),
        step=1.0
    )

    if target_scale in observed_scales:

        st.caption(
            f"✓ {target_scale:g} L is an observed scale."
        )

    else:

        lower = max(
            [
                s for s in observed_scales
                if s < target_scale
            ],
            default=None
        )

        upper = min(
            [
                s for s in observed_scales
                if s > target_scale
            ],
            default=None
        )

        st.caption(
            f"↔ Interpolated target scale: "
            f"{lower:g} L → {upper:g} L"
            if lower is not None
            and upper is not None
            else "⚠️ Extrapolated scale"
        )

    # --------------------------------------------------------
    # Cell line
    # --------------------------------------------------------

    cell_line = st.selectbox(
        "Cell line",
        cell_lines
    )

    # --------------------------------------------------------
    # Optional medium
    # --------------------------------------------------------

    medium = None

    if medium_options:

        medium = st.selectbox(
            "Medium",
            medium_options
        )

    # --------------------------------------------------------
    # Optional serum
    # --------------------------------------------------------

    serum_condition = None

    if serum_options:

        serum_condition = st.selectbox(
            "Serum condition",
            serum_options
        )

    # --------------------------------------------------------
    # Impeller
    # --------------------------------------------------------

    impeller_type = st.selectbox(
        "Impeller type",
        impeller_types
    )

    # --------------------------------------------------------
    # Numerical controls
    # --------------------------------------------------------

    rpm_min, rpm_max = input_ranges["rpm"]

    rpm = st.slider(
        "Agitation — RPM",
        min_value=float(
            rpm_min
        ),
        max_value=float(
            rpm_max
        ),
        value=float(
            np.clip(
                safe_default("rpm"),
                rpm_min,
                rpm_max
            )
        ),
        step=max(
            1.0,
            (rpm_max - rpm_min) / 100
        )
    )

    aer_min, aer_max = input_ranges[
        "aeration_vvm"
    ]

    aeration_vvm = st.slider(
        "Aeration — vvm",
        min_value=float(
            aer_min
        ),
        max_value=float(
            aer_max
        ),
        value=float(
            np.clip(
                safe_default("aeration_vvm"),
                aer_min,
                aer_max
            )
        ),
        step=max(
            0.001,
            (aer_max - aer_min) / 100
        )
    )

    temp_min, temp_max = input_ranges[
        "temperature_C"
    ]

    temperature_C = st.slider(
        "Temperature — °C",
        min_value=float(
            temp_min
        ),
        max_value=float(
            temp_max
        ),
        value=float(
            np.clip(
                safe_default("temperature_C"),
                temp_min,
                temp_max
            )
        ),
        step=0.1
    )

    ph_min, ph_max = input_ranges["pH"]

    pH = st.slider(
        "pH",
        min_value=float(
            ph_min
        ),
        max_value=float(
            ph_max
        ),
        value=float(
            np.clip(
                safe_default("pH"),
                ph_min,
                ph_max
            )
        ),
        step=0.01
    )

    do_min, do_max = input_ranges[
        "DO_percent"
    ]

    DO_percent = st.slider(
        "Dissolved oxygen — %",
        min_value=float(
            do_min
        ),
        max_value=float(
            do_max
        ),
        value=float(
            np.clip(
                safe_default("DO_percent"),
                do_min,
                do_max
            )
        ),
        step=0.5
    )

    feed_min, feed_max = input_ranges[
        "feed_rate_mL_h"
    ]

    feed_rate = st.slider(
        "Feed rate — mL/h",
        min_value=float(
            feed_min
        ),
        max_value=float(
            feed_max
        ),
        value=float(
            np.clip(
                safe_default("feed_rate_mL_h"),
                feed_min,
                feed_max
            )
        ),
        step=max(
            0.01,
            (feed_max - feed_min) / 100
        )
    )

    vcd_min, vcd_max = input_ranges[
        "initial_VCD_million_cells_mL"
    ]

    initial_vcd = st.slider(
        "Initial VCD — M cells/mL",
        min_value=float(
            vcd_min
        ),
        max_value=float(
            vcd_max
        ),
        value=float(
            np.clip(
                safe_default(
                    "initial_VCD_million_cells_mL"
                ),
                vcd_min,
                vcd_max
            )
        ),
        step=max(
            0.001,
            (vcd_max - vcd_min) / 100
        )
    )

    age_min, age_max = input_ranges[
        "batch_age_h"
    ]

    batch_age_h = st.slider(
        "Batch age — h",
        min_value=float(
            age_min
        ),
        max_value=float(
            age_max
        ),
        value=float(
            np.clip(
                safe_default("batch_age_h"),
                age_min,
                age_max
            )
        ),
        step=1.0
    )

    st.markdown("")

    run = st.button(
        "🚀 RUN SCALE-UP ASSESSMENT",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# RUN ASSESSMENT
# ============================================================

if run:

    physics_result = target_scale_physics(

        target_scale=target_scale,

        rpm=rpm,

        aeration_vvm=aeration_vvm,

        temperature_C=temperature_C,

        pH=pH,

        DO_percent=DO_percent,

        feed_rate_mL_h=feed_rate,

        impeller_type=impeller_type
    )

    predictions = biological_prediction(

        physics=physics_result,

        cell_line=cell_line,

        batch_age_h=batch_age_h,

        initial_vcd=initial_vcd
    )

    user_inputs = {

        "target_scale_L":
            target_scale,

        "cell_line":
            cell_line,

        "medium":
            medium,

        "serum_condition":
            serum_condition,

        "impeller_type":
            impeller_type,

        "rpm":
            rpm,

        "aeration_vvm":
            aeration_vvm,

        "temperature_C":
            temperature_C,

        "pH":
            pH,

        "DO_percent":
            DO_percent,

        "feed_rate_mL_h":
            feed_rate,

        "initial_VCD_million_cells_mL":
            initial_vcd,

        "batch_age_h":
            batch_age_h
    }

    risk_result = calculate_risk(

        target_scale=target_scale,

        physics=physics_result,

        predictions=predictions
    )

    st.session_state[
        "scalewise_result"
    ] = {

        "inputs":
            user_inputs,

        "physics":
            physics_result,

        "predictions":
            predictions,

        "risk":
            risk_result
    }


# ============================================================
# LOAD CURRENT RESULT
# ============================================================

if "scalewise_result" not in st.session_state:

    # Automatically calculate initial demonstration condition

    default_target = (
        50.0
        if 50.0 >= min(observed_scales)
        and 50.0 <= max(observed_scales)
        else float(observed_scales[0])
    )

    default_physics = target_scale_physics(

        target_scale=default_target,

        rpm=safe_default("rpm"),

        aeration_vvm=safe_default(
            "aeration_vvm"
        ),

        temperature_C=safe_default(
            "temperature_C"
        ),

        pH=safe_default("pH"),

        DO_percent=safe_default(
            "DO_percent"
        ),

        feed_rate_mL_h=safe_default(
            "feed_rate_mL_h"
        ),

        impeller_type=impeller_types[0]
    )

    default_predictions = biological_prediction(

        physics=default_physics,

        cell_line=cell_lines[0],

        batch_age_h=safe_default(
            "batch_age_h"
        ),

        initial_vcd=safe_default(
            "initial_VCD_million_cells_mL"
        )
    )

    default_inputs = {

        "target_scale_L":
            default_target,

        "cell_line":
            cell_lines[0],

        "medium":
            medium_options[0]
            if medium_options
            else None,

        "serum_condition":
            serum_options[0]
            if serum_options
            else None,

        "impeller_type":
            impeller_types[0],

        "rpm":
            safe_default("rpm"),

        "aeration_vvm":
            safe_default("aeration_vvm"),

        "temperature_C":
            safe_default("temperature_C"),

        "pH":
            safe_default("pH"),

        "DO_percent":
            safe_default("DO_percent"),

        "feed_rate_mL_h":
            safe_default("feed_rate_mL_h"),

        "initial_VCD_million_cells_mL":
            safe_default(
                "initial_VCD_million_cells_mL"
            ),

        "batch_age_h":
            safe_default("batch_age_h")
    }

    default_risk = calculate_risk(

        target_scale=default_target,

        physics=default_physics,

        predictions=default_predictions
    )

    st.session_state[
        "scalewise_result"
    ] = {

        "inputs":
            default_inputs,

        "physics":
            default_physics,

        "predictions":
            default_predictions,

        "risk":
            default_risk
    }


result = st.session_state[
    "scalewise_result"
]

inputs = result["inputs"]
physics = result["physics"]
predictions = result["predictions"]
risk = result["risk"]


# ============================================================
# RIGHT SIDE — RESULTS
# ============================================================

with right:

    # --------------------------------------------------------
    # Top metrics
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📊 Scale-Up Assessment'
        '</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "Risk Score",
            f"{risk['risk_score']:.0f}/100"
        )

    with m2:

        st.metric(
            "Risk Level",
            risk["risk_level"]
        )

    with m3:

        st.metric(
            "Evidence Confidence",
            f"{risk['confidence']:.0f}%"
        )

    with m4:

        st.metric(
            "Target Scale",
            f"{inputs['target_scale_L']:g} L"
        )


    # --------------------------------------------------------
    # Risk banner
    # --------------------------------------------------------

    if risk["risk_level"] == "HIGH":

        st.markdown(
            f"""
<div class="risk-high">
<b>🔴 HIGH SCALE-UP RISK</b><br>
{risk["suggested_action"]}
</div>
""",
            unsafe_allow_html=True
        )

    elif risk["risk_level"] == "MODERATE":

        st.markdown(
            f"""
<div class="risk-medium">
<b>🟠 MODERATE SCALE-UP RISK</b><br>
{risk["suggested_action"]}
</div>
""",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
<div class="risk-low">
<b>🟢 LOWER SCALE-UP RISK</b><br>
{risk["suggested_action"]}
</div>
""",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # Biological + engineering
    # --------------------------------------------------------

    bio_col, eng_col = st.columns(2)

    with bio_col:

        st.markdown(
            '<div class="section-title">'
            '🧫 Biological Prediction'
            '</div>',
            unsafe_allow_html=True
        )

        b1, b2 = st.columns(2)

        with b1:

            st.metric(
                "VCD",
                f"{predictions['VCD_million_cells_mL']:.3f}"
            )

            st.metric(
                "Growth rate",
                f"{predictions['growth_rate_per_h']:.4f}/h"
            )

            st.metric(
                "Glucose",
                f"{predictions['glucose_g_L']:.2f} g/L"
            )

        with b2:

            st.metric(
                "Viability",
                f"{predictions['viability_percent']:.1f}%"
            )

            st.metric(
                "Lactate",
                f"{predictions['lactate_g_L']:.3f} g/L"
            )


    with eng_col:

        st.markdown(
            '<div class="section-title">'
            '🔬 Engineering / Physics'
            '</div>',
            unsafe_allow_html=True
        )

        e1, e2 = st.columns(2)

        with e1:

            st.metric(
                "P/V",
                f"{physics['PV_W_L']:.4f} W/L"
            )

            st.metric(
                "kLa",
                f"{physics['kLa_per_h']:.2f}/h"
            )

            st.metric(
                "OTR",
                f"{physics['OTR_mmol_L_h']:.3f}"
            )

            st.metric(
                "Tip speed",
                f"{physics['tip_speed_m_s']:.3f} m/s"
            )

        with e2:

            st.metric(
                "Mixing time",
                f"{physics['mixing_time_s']:.1f} s"
            )

            st.metric(
                "Reynolds",
                f"{physics['reynolds_number']:.0f}"
            )

            st.metric(
                "Tank diameter",
                f"{physics['tank_diameter_m']:.3f} m"
            )

            st.metric(
                "Tank height",
                f"{physics['tank_height_m']:.3f} m"
            )


    # --------------------------------------------------------
    # Risk scorecard
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🛡️ Process Risk Scorecard'
        '</div>',
        unsafe_allow_html=True
    )

    category_items = list(
        risk["category_scores"].items()
    )

    score_cols = st.columns(
        len(category_items)
    )

    for col, (name, score) in zip(
        score_cols,
        category_items
    ):

        with col:

            st.metric(
                name,
                f"{score:.0f}/100"
            )


    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '⚠️ Scale-Up Warnings & Actions'
        '</div>',
        unsafe_allow_html=True
    )

    warning_text = ""

    for warning in risk["warnings"]:

        warning_text += (
            f"• {warning}<br>"
        )

    st.markdown(
        f"""
<div class="warning-card">
{warning_text}
<br>
<b>Suggested action:</b><br>
{risk["suggested_action"]}
<br><br>
<b>Recommended validation:</b><br>
{risk["recommended_validation"]}
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# TECHNICAL DETAILS — COMPACT EXPANDER
# ============================================================

with st.expander(
    "🔍 Technical details — current scale-up calculation"
):

    t1, t2, t3 = st.columns(3)

    with t1:

        st.write("**User inputs**")

        st.json(
            inputs
        )

    with t2:

        st.write("**Physics engine**")

        st.json(
            physics
        )

    with t3:

        st.write("**Biological predictions**")

        st.json(
            predictions
        )


# ============================================================
# AI COPILOT
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🤖 ScaleWise Live AI Copilot'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    "Ask questions about the current prediction, risk, "
    "physics, biological response or validation strategy."
)


# ------------------------------------------------------------
# Copilot context
# ------------------------------------------------------------

ai_context = build_ai_context(

    inputs=inputs,

    physics=physics,

    predictions=predictions,

    risk=risk
)


# ------------------------------------------------------------
# Initialize chat
# ------------------------------------------------------------

if (
    "copilot_messages"
    not in st.session_state
):

    st.session_state[
        "copilot_messages"
    ] = [
        {
            "role": "assistant",
            "content":
                "Hello! 👋 I am **ScaleWise Copilot**. "
                "I have access to the current scale-up inputs, "
                "physics calculations, biological predictions "
                "and risk assessment. Ask me why the risk is high, "
                "what to validate, or how changing a parameter "
                "may affect the scale-up."
        }
    ]


# ------------------------------------------------------------
# Quick prompts
# ------------------------------------------------------------

q1, q2, q3 = st.columns(3)

quick_questions = [
    (
        "⚠️ Why is the risk score what it is?",
        "Explain the main drivers of the current scale-up risk."
    ),
    (
        "🔬 Explain the physics",
        "Explain the current P/V, kLa, mixing time, OTR and Reynolds number."
    ),
    (
        "🧪 What should I validate?",
        "What should I experimentally validate before moving to this target scale?"
    )
]

for col, (label, question) in zip(
    [q1, q2, q3],
    quick_questions
):

    with col:

        if st.button(
            label,
            use_container_width=True
        ):

            st.session_state[
                "copilot_messages"
            ].append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.spinner(
                "ScaleWise Copilot is analyzing..."
            ):

                answer = ai_copilot_answer(
                    question,
                    ai_context
                )

            st.session_state[
                "copilot_messages"
            ].append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            st.rerun()


# ------------------------------------------------------------
# Display chat
# ------------------------------------------------------------

for message in st.session_state[
    "copilot_messages"
]:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ------------------------------------------------------------
# Chat input
# ------------------------------------------------------------

question = st.chat_input(
    "Ask ScaleWise Copilot about this scale-up..."
)


if question:

    st.session_state[
        "copilot_messages"
    ].append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.spinner(
        "ScaleWise Copilot is thinking..."
    ):

        answer = ai_copilot_answer(
            question,
            ai_context
        )

    st.session_state[
        "copilot_messages"
    ].append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ScaleWise | AI-powered decision support for scalable "
    "cell-culture bioprocess development"
)

st.caption(
    "Prototype using synthetic/generated data. "
    "Risk scores and biological predictions require "
    "experimental validation before production use."
)
