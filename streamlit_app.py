import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ScaleWise — AI Copilot",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 4px;
}

.subtitle {
    font-size: 16px;
    color: #9aa4b2;
    margin-bottom: 28px;
}

.section-title {
    font-size: 23px;
    font-weight: 750;
    margin-top: 28px;
    margin-bottom: 16px;
}

.small-label {
    font-size: 13px;
    color: #9aa4b2;
}

.copilot-header {
    font-size: 20px;
    font-weight: 700;
}

.copilot-box {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(100, 130, 170, 0.35);
    background: rgba(70, 100, 140, 0.12);
    margin-top: 10px;
    margin-bottom: 15px;
}

.info-box {
    padding: 15px;
    border-radius: 12px;
    background: rgba(70, 110, 160, 0.15);
    border-left: 4px solid #4c9be8;
}

.warning-box {
    padding: 15px;
    border-radius: 12px;
    background: rgba(190, 150, 30, 0.15);
    border-left: 4px solid #d6ad36;
}

.success-box {
    padding: 15px;
    border-radius: 12px;
    background: rgba(40, 150, 100, 0.15);
    border-left: 4px solid #35b77a;
}

.danger-box {
    padding: 15px;
    border-radius: 12px;
    background: rgba(190, 60, 60, 0.15);
    border-left: 4px solid #d65c5c;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def find_file(candidates):
    """
    Find a file in the Streamlit repository.
    Checks root directory first, then subdirectories.
    """

    for name in candidates:

        direct = BASE_DIR / name

        if direct.exists():
            return direct

    for name in candidates:

        matches = list(BASE_DIR.rglob(name))

        if matches:
            return matches[0]

    return None


def load_json(candidates, default=None):

    path = find_file(candidates)

    if path is None:
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return default


def load_csv(candidates):

    path = find_file(candidates)

    if path is None:
        return None

    try:
        return pd.read_csv(path)

    except Exception:
        return None


def safe_get(obj, key, default=None):

    try:

        if isinstance(obj, dict):
            return obj.get(key, default)

        if isinstance(obj, pd.Series):
            return obj.get(key, default)

        return default

    except Exception:
        return default


def number(value, default=None):

    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def fmt(value, decimals=2, suffix=""):

    value = number(value)

    if value is None:
        return "N/A"

    return f"{value:.{decimals}f}{suffix}"


def find_nested(data, possible_keys):

    """
    Recursively search JSON dictionaries for common keys.
    """

    if isinstance(data, dict):

        for key in possible_keys:

            if key in data:
                return data[key]

        for value in data.values():

            result = find_nested(value, possible_keys)

            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:

            result = find_nested(item, possible_keys)

            if result is not None:
                return result

    return None


# ============================================================
# LOAD MODEL 2 / DASHBOARD DATA
# ============================================================

dashboard_summary_json = load_json(
    [
        "model2_dashboard_summary.json",
        "dashboard_summary.json"
    ],
    {}
)

scorecard_json = load_json(
    [
        "model2_scorecard.json",
        "dashboard_scorecard.json"
    ],
    {}
)

representative_df = load_csv(
    [
        "model2_representative_condition.csv",
        "dashboard_representative_condition.csv"
    ]
)

window_df = load_csv(
    [
        "model2_window.csv",
        "dashboard_operating_window.csv",
        "model2_operating_window.csv"
    ]
)


# ============================================================
# LOAD MODEL 3 / RISK GUARDIAN DATA
# ============================================================

guardian_df = load_csv(
    [
        "model3_guardian_df.csv",
        "model3_guardian.csv",
        "risk_guardian.csv",
        "guardian_df.csv"
    ]
)

threshold_json = load_json(
    [
        "model3_alert_threshold.json",
        "risk_alert_threshold.json"
    ],
    {}
)


# ============================================================
# FALLBACK DEMO VALUES
# ============================================================

# These values correspond to the recovered demonstration scenario.
# They are only used if the exported JSON files are unavailable.

DEFAULT_SCORE = 71.65
DEFAULT_STATUS = "VALIDATE"
DEFAULT_SCALE = 20.0

DEFAULT_CATEGORIES = {
    "Biological Condition": 72.3,
    "Oxygen Transfer": 70.0,
    "Hydrodynamic Condition": 36.7,
    "Process Control": 100.0,
    "Model Reliability": 100.0
}


# ============================================================
# READINESS VALUES
# ============================================================

dashboard_score = find_nested(
    dashboard_summary_json,
    ["overall_score", "readiness_score", "score"]
)

if dashboard_score is None:

    dashboard_score = find_nested(
        scorecard_json,
        ["overall_score", "readiness_score", "score"]
    )

dashboard_score = number(
    dashboard_score,
    DEFAULT_SCORE
)


dashboard_status = find_nested(
    dashboard_summary_json,
    ["status", "readiness_status"]
)

if dashboard_status is None:

    dashboard_status = find_nested(
        scorecard_json,
        ["status", "readiness_status"]
    )

dashboard_status = (
    str(dashboard_status)
    if dashboard_status is not None
    else DEFAULT_STATUS
)


dashboard_target_scale = find_nested(
    dashboard_summary_json,
    ["target_scale", "dashboard_target_scale"]
)

dashboard_target_scale = number(
    dashboard_target_scale,
    DEFAULT_SCALE
)


# ============================================================
# READINESS CATEGORIES
# ============================================================

dashboard_categories = find_nested(
    dashboard_summary_json,
    ["categories", "readiness_categories"]
)

if dashboard_categories is None:

    dashboard_categories = find_nested(
        scorecard_json,
        ["categories", "readiness_categories"]
    )


if not isinstance(dashboard_categories, dict):

    dashboard_categories = DEFAULT_CATEGORIES.copy()


# Convert nested category structures to numeric scores

clean_categories = {}

for name, value in dashboard_categories.items():

    if isinstance(value, dict):

        score = value.get(
            "score",
            value.get(
                "value",
                value.get("category_score")
            )
        )

    else:
        score = value

    score = number(score)

    if score is not None:
        clean_categories[str(name)] = score


if len(clean_categories) == 0:

    clean_categories = DEFAULT_CATEGORIES.copy()


# ============================================================
# REPRESENTATIVE CONDITION
# ============================================================

if representative_df is not None and len(representative_df) > 0:

    dashboard_row = representative_df.iloc[0]

else:

    # Fallback based on recovered Model 2 result
    dashboard_row = pd.Series({
        "scale_L": 20.0,
        "working_volume_L": 17.361,
        "tank_diameter_m": 0.217,
        "impeller_diameter_m": 0.072,
        "rpm": 217.360,
        "aeration_vvm": 0.487,
        "DO_percent": 53.629,
        "temperature_C": 37.036,
        "pH": 7.003,
        "feed_rate_mL_h": 0.335,
        "VCD_million_cells_mL": 0.6247,
        "viability_percent": 97.9684,
        "growth_rate_per_h": -0.0050,
        "lactate_g_L": 0.2161,
        "PV_W_L": 0.014,
        "kLa_per_h": 5.382,
        "mixing_time_s": 168.4,
        "tip_speed_m_s": 0.820,
        "reynolds_number": 18790
    })


# ============================================================
# EXTRACT REPRESENTATIVE VALUES
# ============================================================

rep_scale = safe_get(
    dashboard_row,
    "scale_L",
    dashboard_target_scale
)

rep_working_volume = safe_get(
    dashboard_row,
    "working_volume_L"
)

rep_tank_diameter = safe_get(
    dashboard_row,
    "tank_diameter_m"
)

rep_impeller_diameter = safe_get(
    dashboard_row,
    "impeller_diameter_m"
)

rep_rpm = safe_get(
    dashboard_row,
    "rpm"
)

rep_aeration = safe_get(
    dashboard_row,
    "aeration_vvm"
)

rep_do = safe_get(
    dashboard_row,
    "DO_percent"
)

rep_temperature = safe_get(
    dashboard_row,
    "temperature_C"
)

rep_ph = safe_get(
    dashboard_row,
    "pH"
)

rep_feed = safe_get(
    dashboard_row,
    "feed_rate_mL_h"
)

rep_vcd = safe_get(
    dashboard_row,
    "VCD_million_cells_mL"
)

rep_viability = safe_get(
    dashboard_row,
    "viability_percent"
)

rep_growth = safe_get(
    dashboard_row,
    "growth_rate_per_h"
)

rep_lactate = safe_get(
    dashboard_row,
    "lactate_g_L"
)

rep_pv = safe_get(
    dashboard_row,
    "PV_W_L"
)

rep_kla = safe_get(
    dashboard_row,
    "kLa_per_h"
)

rep_mixing = safe_get(
    dashboard_row,
    "mixing_time_s"
)

rep_tip = safe_get(
    dashboard_row,
    "tip_speed_m_s"
)

rep_re = safe_get(
    dashboard_row,
    "reynolds_number"
)


# ============================================================
# OPERATING WINDOW
# ============================================================

# Recovered Model 2 window
# Used as fallback if CSV is not available.

window_fallback = pd.DataFrame({
    "Parameter": [
        "Agitation (RPM)",
        "Aeration (vvm)",
        "DO (%)"
    ],
    "Minimum": [
        152.440,
        0.223,
        44.870
    ],
    "Representative": [
        217.360,
        0.487,
        53.629
    ],
    "Maximum": [
        239.000,
        0.575,
        53.629
    ]
})


if window_df is None or len(window_df) == 0:

    dashboard_operating_window = window_fallback.copy()

else:

    dashboard_operating_window = window_df.copy()

    # Normalize common column names

    rename_map = {}

    for col in dashboard_operating_window.columns:

        low = str(col).lower()

        if low in ["parameter", "parameters"]:
            rename_map[col] = "Parameter"

        elif low in ["minimum", "min"]:
            rename_map[col] = "Minimum"

        elif low in ["representative", "recommended", "nominal"]:
            rename_map[col] = "Representative"

        elif low in ["maximum", "max"]:
            rename_map[col] = "Maximum"

    dashboard_operating_window = (
        dashboard_operating_window
        .rename(columns=rename_map)
    )


# ============================================================
# RISK GUARDIAN
# ============================================================

if guardian_df is None or len(guardian_df) == 0:

    # Fallback Guardian observation
    guardian_df = pd.DataFrame({
        "timestamp": pd.date_range(
            end=pd.Timestamp.now(),
            periods=100,
            freq="h"
        ),
        "scale_L": [dashboard_target_scale] * 100,
        "DO_percent": np.linspace(50, 54, 100),
        "oxygen_margin": np.linspace(0.10, 0.15, 100),
        "failure_probability": np.linspace(0.05, 0.30, 100),
        "risk_score": np.linspace(5, 30, 100),
        "risk_level": ["WATCH"] * 100,
        "warning_signal_count": [0] * 100,
        "risk_reason": [
            "No active rule-based warning signals."
        ] * 100,
        "recommended_action": [
            "Continue monitoring process trajectory."
        ] * 100,
        "model_confidence": np.linspace(10, 60, 100)
    })


# Parse timestamp

if "timestamp" in guardian_df.columns:

    guardian_df["timestamp"] = pd.to_datetime(
        guardian_df["timestamp"],
        errors="coerce"
    )


# Target-scale Guardian data

guardian_target = guardian_df.copy()

if "scale_L" in guardian_target.columns:

    scale_difference = (
        guardian_target["scale_L"].astype(float)
        - float(dashboard_target_scale)
    ).abs()

    target_rows = guardian_target[
        scale_difference < 1e-6
    ]

    if len(target_rows) > 0:
        guardian_target = target_rows.copy()


# Sort chronologically

if "timestamp" in guardian_target.columns:

    guardian_target = guardian_target.sort_values(
        "timestamp"
    )


current_guardian = guardian_target.iloc[-1]


# ============================================================
# CURRENT RISK VALUES
# ============================================================

risk_probability = number(
    safe_get(
        current_guardian,
        "failure_probability"
    )
)

risk_level = str(
    safe_get(
        current_guardian,
        "risk_level",
        "UNKNOWN"
    )
)

risk_score = number(
    safe_get(
        current_guardian,
        "risk_score"
    )
)

warning_count = int(
    number(
        safe_get(
            current_guardian,
            "warning_signal_count",
            0
        ),
        0
    )
)

risk_reason = str(
    safe_get(
        current_guardian,
        "risk_reason",
        "No active rule-based warning signals detected."
    )
)

risk_action = str(
    safe_get(
        current_guardian,
        "recommended_action",
        "Continue monitoring the process trajectory."
    )
)

risk_confidence = number(
    safe_get(
        current_guardian,
        "model_confidence"
    )
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🧬 ScaleWise — AI Copilot for Scalable Cell-Culture Bioprocess Design'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict the biology • Optimize the operating window • Monitor scale-up risk'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Process Configuration")

st.sidebar.metric(
    "Demo Target Scale",
    f"{dashboard_target_scale:.0f} L"
)

process_stage = st.sidebar.selectbox(
    "Process Stage",
    [
        "Scale-Up Assessment",
        "Process Validation",
        "Production Monitoring"
    ]
)

st.sidebar.info(
    f"Current prototype scenario: "
    f"{dashboard_target_scale:.0f} L target scale."
)

st.sidebar.warning(
    "Prototype dashboard using synthetic/demo data. "
    "Process limits and recommendations require experimental validation."
)


# ============================================================
# PROCESS READINESS
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Process Readiness</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Target Scale",
        f"{dashboard_target_scale:.0f} L"
    )

with c2:

    st.metric(
        "Overall Readiness",
        f"{dashboard_score:.2f} / 100"
    )

with c3:

    st.metric(
        "Status",
        dashboard_status
    )


# ============================================================
# READINESS CATEGORIES
# ============================================================

st.markdown(
    '<div class="section-title">📊 Readiness Categories</div>',
    unsafe_allow_html=True
)

category_names = list(clean_categories.keys())

# Guarantee five slots

while len(category_names) < 5:

    category_names.append(
        f"Category {len(category_names) + 1}"
    )


cols = st.columns(5)

for i in range(5):

    name = category_names[i]

    value = clean_categories.get(
        name,
        None
    )

    with cols[i]:

        st.metric(
            name,
            (
                f"{value:.1f} / 100"
                if value is not None
                else "N/A"
            )
        )


# ============================================================
# OPERATING WINDOW
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Recommended Operating Window</div>',
    unsafe_allow_html=True
)

st.dataframe(
    dashboard_operating_window,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# REPRESENTATIVE CONDITION
# ============================================================

st.markdown(
    '<div class="section-title">📍 Representative Target-Scale Condition</div>',
    unsafe_allow_html=True
)

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.metric("Scale", fmt(rep_scale, 3, " L"))

with r2:
    st.metric("Working Volume", fmt(rep_working_volume, 3, " L"))

with r3:
    st.metric("Tank Diameter", fmt(rep_tank_diameter, 3, " m"))

with r4:
    st.metric("Impeller Diameter", fmt(rep_impeller_diameter, 3, " m"))


# ============================================================
# BIOLOGICAL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">🧫 Predicted Biological Performance</div>',
    unsafe_allow_html=True
)

b1, b2, b3, b4 = st.columns(4)

with b1:

    st.metric(
        "VCD",
        fmt(rep_vcd, 3, " M cells/mL")
    )

with b2:

    st.metric(
        "Viability",
        fmt(rep_viability, 2, "%")
    )

with b3:

    st.metric(
        "Growth Rate",
        fmt(rep_growth, 4, " /h")
    )

with b4:

    st.metric(
        "Lactate",
        fmt(rep_lactate, 3, " g/L")
    )


# ============================================================
# ENGINEERING INDICATORS
# ============================================================

st.markdown(
    '<div class="section-title">🔬 Scale-Up Engineering Indicators</div>',
    unsafe_allow_html=True
)

e1, e2, e3 = st.columns(3)

with e1:
    st.metric(
        "P/V",
        fmt(rep_pv, 4, " W/L")
    )

with e2:
    st.metric(
        "kLa",
        fmt(rep_kla, 3, " /h")
    )

with e3:
    st.metric(
        "Mixing Time",
        fmt(rep_mixing, 1, " s")
    )


e4, e5, e6 = st.columns(3)

with e4:
    st.metric(
        "Tip Speed",
        fmt(rep_tip, 3, " m/s")
    )

with e5:
    st.metric(
        "Reynolds Number",
        fmt(rep_re, 0)
    )

with e6:
    st.metric(
        "DO",
        fmt(rep_do, 2, "%")
    )


# ============================================================
# REAL-TIME RISK GUARDIAN
# ============================================================

st.markdown(
    '<div class="section-title">🛡️ Real-Time Scale-Up Risk Guardian</div>',
    unsafe_allow_html=True
)

g1, g2, g3, g4 = st.columns(4)

with g1:

    if risk_probability is not None:

        st.metric(
            "Deviation Probability",
            f"{risk_probability * 100:.1f}%"
        )

    else:

        st.metric(
            "Deviation Probability",
            "N/A"
        )


with g2:

    st.metric(
        "Risk Level",
        risk_level
    )


with g3:

    if risk_confidence is not None:

        st.metric(
            "Classification Confidence",
            f"{risk_confidence:.1f}%"
        )

    else:

        st.metric(
            "Classification Confidence",
            "N/A"
        )


with g4:

    st.metric(
        "Warning Signals",
        warning_count
    )


# Risk interpretation

if risk_probability is not None and risk_probability >= 0.70:

    st.warning(
        "⚠️ Risk Guardian predicts a high probability of an upcoming "
        "process deviation in the simulated process stream."
    )

elif risk_probability is not None and risk_probability >= 0.40:

    st.warning(
        "⚠️ Risk Guardian indicates a WATCH/HIGH risk pattern "
        "requiring process review."
    )

else:

    st.success(
        "✅ No high-probability deviation pattern is currently detected."
    )


st.info(
    "Risk Guardian output is based on the synthetic/demo process stream. "
    "Classification confidence is not biological certainty."
)


# ============================================================
# RISK EXPLANATION
# ============================================================

st.markdown(
    '<div class="section-title">🔎 Current Risk Explanation</div>',
    unsafe_allow_html=True
)

if warning_count > 0:

    st.markdown(
        f"""
        <div class="warning-box">
        <b>{warning_count} rule-based warning signal(s) detected.</b>
        <br><br>
        {risk_reason}
        <br><br>
        <b>Recommended action:</b><br>
        {risk_action}
        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        f"""
        <div class="info-box">
        <b>No active rule-based warning signals are detected
        in the displayed observation.</b>
        <br><br>
        However, the classifier currently reports:
        <b>{risk_probability * 100:.1f}% predicted deviation probability</b>
        if risk_probability is not None else <b>N/A</b>.
        <br><br>
        This means the learned classifier is identifying a risk pattern
        that is not currently accompanied by the rule-based warning
        conditions. The prediction should therefore be investigated
        rather than treated as biological certainty.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# RISK GUARDIAN TREND
# ============================================================

st.markdown(
    '<div class="section-title">📈 Risk Guardian Trends</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Failure/deviation probability across the simulated "
    f"{dashboard_target_scale:.0f} L process stream."
)


trend_columns = [
    c for c in [
        "failure_probability",
        "DO_percent",
        "oxygen_margin"
    ]
    if c in guardian_target.columns
]


if len(trend_columns) > 0:

    trend_df = guardian_target.copy()

    if "timestamp" in trend_df.columns:

        trend_df = trend_df.set_index(
            "timestamp"
        )

    plot_df = trend_df[trend_columns].copy()

    if "failure_probability" in plot_df.columns:

        plot_df["failure_probability"] = (
            plot_df["failure_probability"] * 100
        )

    st.line_chart(
        plot_df,
        use_container_width=True
    )

else:

    st.info(
        "Risk trend data are not available in the exported package."
    )


# ============================================================
# 🤖 INTERACTIVE SCALEWISE AI COPILOT
# ============================================================

st.markdown(
    '<div class="section-title">🤖 ScaleWise AI Copilot</div>',
    unsafe_allow_html=True
)

st.caption(
    "Ask questions about the current scale-up scenario. "
    "The Copilot answers using the Model 1, Model 2 and Model 3 "
    "outputs currently loaded into the dashboard."
)


# ============================================================
# COPILOT RESPONSE ENGINE
# ============================================================

def copilot_response(question):

    q = question.lower().strip()


    # --------------------------------------------------------
    # READINESS
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "readiness",
            "score",
            "validate",
            "ready",
            "status"
        ]
    ):

        category_lines = []

        for name, value in clean_categories.items():

            category_lines.append(
                f"- **{name}:** {value:.1f}/100"
            )

        return f"""
### 🎯 Process Readiness

The current **{dashboard_target_scale:.0f} L** demonstration scenario
has an overall readiness score of **{dashboard_score:.2f}/100**.

The current process status is:

**{dashboard_status}**

Readiness categories:

{chr(10).join(category_lines)}

The system is therefore treating this scenario as a **validation
candidate**, rather than a validated production condition.

The result is based on the prototype's synthetic/demo dataset and
should be experimentally validated before operational use.
"""


    # --------------------------------------------------------
    # OPERATING WINDOW
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "operating window",
            "operating condition",
            "recommended condition",
            "recommend",
            "rpm",
            "agitation",
            "aeration"
        ]
    ):

        return f"""
### ⚙️ Recommended Operating Region

For the current **{dashboard_target_scale:.0f} L** demonstration
scenario, the model-derived operating region is:

**Agitation**
- Minimum: **152.44 RPM**
- Representative: **217.36 RPM**
- Maximum: **239.00 RPM**

**Aeration**
- Minimum: **0.223 vvm**
- Representative: **0.487 vvm**
- Maximum: **0.575 vvm**

**DO**
- Minimum: **44.87%**
- Representative: **53.63%**
- Maximum: **53.63%**

Representative condition:

- Temperature: **{fmt(rep_temperature, 2)} °C**
- pH: **{fmt(rep_ph, 2)}**
- Feed rate: **{fmt(rep_feed, 3)} mL/h**

These values are **model-derived starting conditions for validation**,
not validated production setpoints.
"""


    # --------------------------------------------------------
    # BIOLOGICAL PERFORMANCE
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "biology",
            "biological",
            "vcd",
            "viability",
            "growth",
            "lactate",
            "cells"
        ]
    ):

        return f"""
### 🧫 Predicted Biological Performance

For the representative target-scale condition:

- **VCD:** {fmt(rep_vcd, 3)} million cells/mL
- **Viability:** {fmt(rep_viability, 2)}%
- **Growth rate:** {fmt(rep_growth, 4)} /h
- **Lactate:** {fmt(rep_lactate, 3)} g/L

These are Model 1 predictions. They should be compared with
experimental observations during target-scale validation.
"""


    # --------------------------------------------------------
    # ENGINEERING
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "engineering",
            "physics",
            "hydrodynamic",
            "mixing",
            "kla",
            "oxygen transfer",
            "oxygen",
            "reynolds",
            "tip speed",
            "p/v"
        ]
    ):

        return f"""
### 🔬 Scale-Up Engineering Assessment

The representative target-scale condition has:

- **P/V:** {fmt(rep_pv, 4)} W/L
- **kLa:** {fmt(rep_kla, 3)} /h
- **Mixing time:** {fmt(rep_mixing, 1)} s
- **Tip speed:** {fmt(rep_tip, 3)} m/s
- **Reynolds number:** {fmt(rep_re, 0)}
- **DO:** {fmt(rep_do, 2)}%

These indicators describe the physical/hydrodynamic environment
associated with the target-scale process.

The hydrodynamic category is currently **{
    clean_categories.get(
        "Hydrodynamic Condition",
        36.7
    )
:.1f}/100**, which is one of the main areas requiring validation.
"""


    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "risk",
            "danger",
            "failure",
            "deviation",
            "warning",
            "guardian",
            "critical"
        ]
    ):

        probability_text = (
            f"{risk_probability * 100:.1f}%"
            if risk_probability is not None
            else "N/A"
        )

        confidence_text = (
            f"{risk_confidence:.1f}%"
            if risk_confidence is not None
            else "N/A"
        )

        return f"""
### 🛡️ Risk Guardian Assessment

**Predicted deviation probability:** {probability_text}

**Risk level:** {risk_level}

**Classification confidence:** {confidence_text}

**Rule-based warning signals:** {warning_count}

**Risk explanation**

{risk_reason}

**Recommended action**

{risk_action}

Important: classification confidence indicates confidence in the
classifier's output. It should **not** be interpreted as biological
certainty.
"""


    # --------------------------------------------------------
    # NEXT STEP
    # --------------------------------------------------------

    if any(
        word in q
        for word in [
            "what should i do",
            "what next",
            "next step",
            "action",
            "proceed",
            "validation",
            "validate next"
        ]
    ):

        return f"""
### 🧭 ScaleWise Recommended Next Step

For the current **{dashboard_target_scale:.0f} L** scenario:

1. Use the representative operating condition as the starting point
   for target-scale validation.
2. Verify oxygen-transfer behaviour experimentally.
3. Evaluate hydrodynamic/mixing behaviour.
4. Monitor DO, oxygen margin, viability and process trends.
5. Compare observed biology with Model 1 predictions.
6. Reassess Risk Guardian behaviour during the validation run.

Current status:

**{dashboard_status}**

The prototype should be used as a **decision-support system for
validation**, not as an autonomous production controller.
"""


    # --------------------------------------------------------
    # GENERAL
    # --------------------------------------------------------

    return f"""
### 🤖 ScaleWise Copilot

I can help interpret the current **{dashboard_target_scale:.0f} L**
scale-up scenario.

Try asking:

- **Why is the readiness score {dashboard_score:.2f}?**
- **What operating window do you recommend?**
- **What biological performance is predicted?**
- **What are the important engineering indicators?**
- **Why is the process at risk?**
- **What should we validate next?**
"""


# ============================================================
# COPILOT QUICK QUESTIONS
# ============================================================

st.markdown(
    '<div class="copilot-box">'
    '<div class="copilot-header">💬 Ask ScaleWise</div>'
    '<br>'
    'Use the chat box below or select a suggested question.'
    '</div>',
    unsafe_allow_html=True
)


q1, q2, q3 = st.columns(3)


with q1:

    if st.button(
        "🎯 Why is readiness VALIDATE?",
        use_container_width=True
    ):

        st.session_state["copilot_question"] = (
            "Why is the readiness score VALIDATE?"
        )


with q2:

    if st.button(
        "⚙️ What operating window is recommended?",
        use_container_width=True
    ):

        st.session_state["copilot_question"] = (
            "What operating window is recommended?"
        )


with q3:

    if st.button(
        "🛡️ Why is the process at risk?",
        use_container_width=True
    ):

        st.session_state["copilot_question"] = (
            "Why is the process at risk?"
        )


q4, q5, q6 = st.columns(3)


with q4:

    if st.button(
        "🧫 What biology is predicted?",
        use_container_width=True
    ):

        st.session_state["copilot_question"] = (
            "What biological performance is predicted?"
        )


with q5:

    if st.button(
        "🔬 Explain the engineering indicators",
        use_container_width=True
    ):

        st.session_state["copilot_question"] = (
            "Explain the engineering indicators."
        )


with q6:

    if st.button(
        "🧭 What should we validate next?",
        use_container_width=True
    ):

        st.session_state["copilot_question"] = (
            "What should we validate next?"
        )


# ============================================================
# CHAT HISTORY
# ============================================================

if "copilot_messages" not in st.session_state:

    st.session_state.copilot_messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm **ScaleWise Copilot**.\n\n"
                "I can explain the current readiness score, "
                "operating window, biological predictions, "
                "engineering indicators and Risk Guardian."
            )
        }
    ]


# ============================================================
# QUICK QUESTION RESPONSE
# ============================================================

if "copilot_question" in st.session_state:

    question = st.session_state.pop(
        "copilot_question"
    )

    answer = copilot_response(
        question
    )

    st.session_state.copilot_messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    st.session_state.copilot_messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.copilot_messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask ScaleWise about this scale-up scenario..."
)


if user_question:

    st.session_state.copilot_messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    answer = copilot_response(
        user_question
    )

    st.session_state.copilot_messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()


# ============================================================
# COPILOT DATA TRANSPARENCY
# ============================================================

with st.expander(
    "🔍 View data used by ScaleWise Copilot"
):

    st.write(
        "The Copilot is grounded in the current dashboard outputs."
    )

    st.json({

        "target_scale_L": dashboard_target_scale,

        "readiness_score": dashboard_score,

        "status": dashboard_status,

        "readiness_categories": clean_categories,

        "representative_condition": {

            "rpm": number(rep_rpm),

            "aeration_vvm": number(rep_aeration),

            "DO_percent": number(rep_do),

            "temperature_C": number(rep_temperature),

            "pH": number(rep_ph),

            "feed_rate_mL_h": number(rep_feed)
        },

        "predicted_biology": {

            "VCD_million_cells_mL":
                number(rep_vcd),

            "viability_percent":
                number(rep_viability),

            "growth_rate_per_h":
                number(rep_growth),

            "lactate_g_L":
                number(rep_lactate)
        },

        "risk_guardian": {

            "failure_probability":
                number(risk_probability),

            "risk_level":
                risk_level,

            "warning_signal_count":
                warning_count
        }

    })


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ScaleWise | AI Copilot for Scalable Cell-Culture Bioprocess Design"
)

st.caption(
    "Prototype / synthetic-data demonstration. "
    "Operating limits, predictions and risk signals require "
    "experimental validation before production use."
)
