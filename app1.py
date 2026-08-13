# ============================================================
# 🎓 CGPA PACKAGE PREDICTOR
# 🚀 Advanced Machine Learning Deployment Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# ⚙️ PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CGPA • Package Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 🎨 ADVANCED CSS
# ============================================================

st.markdown("""
<style>

/* ---------- MAIN BACKGROUND ---------- */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.18), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(168,85,247,0.15), transparent 25%),
        radial-gradient(circle at 50% 90%, rgba(14,165,233,0.10), transparent 30%),
        #070b14;
    color: #ffffff;
}


/* ---------- REMOVE DEFAULT HEADER ---------- */

header {
    visibility: hidden;
}

.block-container {
    padding-top: 2rem;
    max-width: 1400px;
}


/* ---------- HERO ---------- */

.hero {
    padding: 45px 20px 35px;
    text-align: center;
}

.hero-badge {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 50px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(129,140,248,0.35);
    color: #a5b4fc;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1px;
}

.hero h1 {
    font-size: 58px;
    margin: 18px 0 10px;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(
        90deg,
        #ffffff,
        #a5b4fc,
        #c084fc
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #94a3b8;
    font-size: 18px;
}


/* ---------- GLASS CARD ---------- */

.glass {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 24px;
    padding: 28px;
    box-shadow:
        0 20px 50px rgba(0,0,0,0.25),
        inset 0 1px rgba(255,255,255,0.03);
    backdrop-filter: blur(16px);
}


/* ---------- INPUT CARD ---------- */

.input-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 5px;
}

.input-subtitle {
    color: #94a3b8;
    font-size: 14px;
    margin-bottom: 20px;
}


/* ---------- RESULT CARD ---------- */

.result {
    background:
        linear-gradient(
            135deg,
            rgba(79,70,229,0.25),
            rgba(147,51,234,0.18)
        );
    border: 1px solid rgba(129,140,248,0.3);
    border-radius: 25px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(79,70,229,0.15);
}

.result-label {
    color: #c4b5fd;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.result-value {
    font-size: 52px;
    font-weight: 800;
    margin: 8px 0;
    background: linear-gradient(90deg,#ffffff,#c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.result-category {
    color: #94a3b8;
    font-size: 15px;
}


/* ---------- METRIC CARDS ---------- */

.metric {
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
}

.metric-icon {
    font-size: 25px;
}

.metric-title {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 7px;
}

.metric-value {
    font-size: 24px;
    font-weight: 700;
    margin-top: 5px;
}


/* ---------- SECTION TITLE ---------- */

.section {
    font-size: 25px;
    font-weight: 750;
    margin: 35px 0 18px;
}


/* ---------- FOOTER ---------- */

.footer {
    text-align: center;
    padding: 35px;
    color: #64748b;
    font-size: 13px;
}

.footer span {
    color: #a5b4fc;
}


/* ---------- BUTTON ---------- */

.stButton > button {
    width: 100%;
    border-radius: 14px;
    height: 52px;
    border: none;
    background: linear-gradient(
        90deg,
        #4f46e5,
        #7c3aed
    );
    color: white;
    font-size: 16px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(99,102,241,0.35);
}


/* ---------- SLIDER ---------- */

.stSlider {
    padding-top: 10px;
}


/* ---------- DATAFRAME ---------- */

[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 🤖 LOAD MODEL
# ============================================================

try:
    model = joblib.load("package_model.pkl")

except FileNotFoundError:

    st.error(
        "⚠️ Model not found! Run `python train_model.py` first."
    )

    st.stop()


# ============================================================
# 🦸 HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<div class="hero-badge">
🤖 MACHINE LEARNING • CAREER ANALYTICS
</div>

<h1>CGPA → Package Predictor</h1>

<p>
Discover your estimated placement package using
Machine Learning
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# 📥 INPUT + RESULT
# ============================================================

left, right = st.columns([1, 1], gap="large")


# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

with left:

    st.markdown("""
    <div class="glass">

    <div class="input-title">
    🎓 Student Profile
    </div>

    <div class="input-subtitle">
    Enter your academic performance
    </div>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    cgpa = st.slider(
        "Your CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1
    )

    st.markdown(
        f"""
        <div style="
        text-align:center;
        font-size:42px;
        font-weight:800;
        margin:15px 0;
        ">
        {cgpa:.1f}
        <span style="
        font-size:16px;
        color:#94a3b8;
        ">
        / 10.0
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    predict = st.button(
        "🚀  PREDICT MY PACKAGE"
    )


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

with right:

    if predict:

        prediction = model.predict(
            np.array([[cgpa]])
        )[0]

        prediction = max(0, prediction)

        # Package category
        if prediction < 3:
            category = "🌱 Entry Level"
        elif prediction < 5:
            category = "📈 Growing Potential"
        elif prediction < 8:
            category = "🔥 Strong Candidate"
        else:
            category = "🚀 High Potential"

        st.markdown(
            f"""
            <div class="result">

            <div class="result-label">
            Estimated Package
            </div>

            <div class="result-value">
            ₹ {prediction:.2f} LPA
            </div>

            <div class="result-category">
            {category}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown("""
        <div class="result">

        <div class="result-label">
        Your Prediction
        </div>

        <div class="result-value">
        ₹ -- LPA
        </div>

        <div class="result-category">
        Enter your CGPA and start prediction
        </div>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 📊 ANALYTICS
# ============================================================

st.markdown(
    '<div class="section">📊 Prediction Analytics</div>',
    unsafe_allow_html=True
)


# Generate prediction curve

cgpa_values = np.linspace(5, 10, 50)

package_values = model.predict(
    cgpa_values.reshape(-1, 1)
)

chart_df = pd.DataFrame({
    "CGPA": cgpa_values,
    "Predicted Package (LPA)": package_values
})


# ------------------------------------------------------------
# CHART
# ------------------------------------------------------------

chart_col, info_col = st.columns(
    [2.2, 1],
    gap="large"
)


with chart_col:

    st.markdown(
        '<div class="glass">',
        unsafe_allow_html=True
    )

    st.line_chart(
        chart_df.set_index("CGPA"),
        height=420
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# INSIGHTS
# ------------------------------------------------------------

with info_col:

    st.markdown("""
    <div class="glass">

    <div class="input-title">
    💡 Model Insights
    </div>

    <br>

    <b>📌 Minimum CGPA</b>
    <br>
    <span style="color:#94a3b8">
    5.0
    </span>

    <br><br>

    <b>📌 Maximum CGPA</b>
    <br>
    <span style="color:#94a3b8">
    10.0
    </span>

    <br><br>

    <b>📌 Prediction Type</b>
    <br>
    <span style="color:#a5b4fc">
    Regression
    </span>

    <br><br>

    <b>📌 Algorithm</b>
    <br>
    <span style="color:#a5b4fc">
    Linear Regression
    </span>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 📈 PERFORMANCE METRICS
# ============================================================

st.markdown(
    '<div class="section">⚡ Model Information</div>',
    unsafe_allow_html=True
)


m1, m2, m3, m4 = st.columns(4)


with m1:
    st.markdown("""
    <div class="metric">
    <div class="metric-icon">🧠</div>
    <div class="metric-title">MODEL</div>
    <div class="metric-value">Linear Regression</div>
    </div>
    """, unsafe_allow_html=True)


with m2:
    st.markdown("""
    <div class="metric">
    <div class="metric-icon">🎯</div>
    <div class="metric-title">FEATURE</div>
    <div class="metric-value">CGPA</div>
    </div>
    """, unsafe_allow_html=True)


with m3:
    st.markdown("""
    <div class="metric">
    <div class="metric-icon">📊</div>
    <div class="metric-title">PROBLEM</div>
    <div class="metric-value">Regression</div>
    </div>
    """, unsafe_allow_html=True)


with m4:
    st.markdown("""
    <div class="metric">
    <div class="metric-icon">⚡</div>
    <div class="metric-title">DEPLOYMENT</div>
    <div class="metric-value">Streamlit</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 📝 PREDICTION TABLE
# ============================================================

st.markdown(
    '<div class="section">🔮 Package Prediction Range</div>',
    unsafe_allow_html=True
)


display_df = chart_df.iloc[
    ::5
].copy()

display_df["CGPA"] = display_df["CGPA"].round(1)

display_df["Predicted Package (LPA)"] = (
    display_df["Predicted Package (LPA)"]
    .round(2)
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 🦶 FOOTER
# ============================================================

st.markdown("""
<div class="footer">

🎓 CGPA Package Predictor

<br><br>

Built with
<span>Python</span> •
<span>Scikit-Learn</span> •
<span>Joblib</span> •
<span>Streamlit</span>

<br><br>

⚠️ Predictions are estimates based on the training dataset.

</div>
""", unsafe_allow_html=True)