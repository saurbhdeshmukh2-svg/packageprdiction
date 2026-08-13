import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CGPA Package Predictor",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =======================================================
   MAIN PAGE BACKGROUND
   ======================================================= */

.stApp {
    background: linear-gradient(135deg, #dcdcdc, #f2f2f2);
}


/* =======================================================
   MAIN TITLE
   ======================================================= */

.title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 5px;
    color: #222222;
}


/* =======================================================
   SUBTITLE
   ======================================================= */

.subtitle {
    text-align: center;
    color: #555555;
    font-size: 18px;
    margin-bottom: 30px;
}


/* =======================================================
   PREDICTION CARD
   ======================================================= */

.prediction-card {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    text-align: center;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.20);
}


/* =======================================================
   PREDICTION TITLE
   ======================================================= */

.prediction-title {
    font-size: 20px;
    margin-bottom: 10px;
    color: white;
}


/* =======================================================
   MAIN PACKAGE VALUE
   ======================================================= */

.package {
    font-size: 48px;
    font-weight: bold;
    color: #ff4d4d;
}


/* =======================================================
   CGPA TEXT INSIDE PREDICTION CARD
   ======================================================= */

.cgpa-text {
    font-size: 18px;
    margin-top: 10px;
    color: white;
}


/* =======================================================
   WELCOME CARD
   ======================================================= */

.info-card {
    padding: 25px;
    border-radius: 18px;
    background-color: white;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.10);
    color: #222222;
}


/* =======================================================
   STREAMLIT METRIC CARDS
   ======================================================= */

[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.10);
}


/* =======================================================
   METRIC LABEL
   ======================================================= */

[data-testid="stMetricLabel"] {
    color: #333333 !important;
}


/* =======================================================
   METRIC VALUE
   ======================================================= */

[data-testid="stMetricValue"] {
    color: #222222 !important;
    font-weight: bold !important;
}


/* =======================================================
   BUTTON
   ======================================================= */

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    font-size: 17px;
    font-weight: bold;
}


/* =======================================================
   FOOTER
   ======================================================= */

.footer {
    text-align: center;
    color: #555555;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = joblib.load("model.joblib")

except FileNotFoundError:

    st.error(
        "⚠️ model.joblib not found!\n\n"
        "Make sure model.joblib is in the same folder as app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading model: {e}"
    )

    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🎓 CGPA → Package Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Salary Package Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# MAIN COLUMNS
# =========================================================

input_col, result_col = st.columns([1, 2])


# =========================================================
# INPUT SECTION
# =========================================================

with input_col:

    st.markdown("### 📊 Enter Your Details")

    cgpa = st.number_input(
        "Enter your CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

    st.write("")

    predict_button = st.button(
        "🚀 Predict Package",
        use_container_width=True
    )

    st.write("")

    st.info(
        "📌 Enter CGPA between 0 and 10."
    )


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # PREPARE INPUT
    # -----------------------------------------------------

    input_data = np.array([[cgpa]])


    # -----------------------------------------------------
    # MAKE PREDICTION
    # -----------------------------------------------------

    try:

        prediction = model.predict(input_data)

        # Convert NumPy array to single number
        prediction = float(
            np.asarray(prediction).flatten()[0]
        )

        # Prevent negative package
        prediction = max(0, prediction)

    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )

        st.stop()


    # =====================================================
    # RESULT CARD
    # =====================================================

    with result_col:

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-title">
                    💼 Expected Package
                </div>

                <div class="package">
                    {prediction:.2f} LPA
                </div>

                <div class="cgpa-text">
                    Based on CGPA: {cgpa:.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # =====================================================
    # METRICS
    # =====================================================

    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.metric(
            "🎓 Your CGPA",
            f"{cgpa:.2f}"
        )


    with metric2:

        st.metric(
            "💼 Predicted Package",
            f"{prediction:.2f} LPA"
        )


    with metric3:

        annual_salary = prediction * 100000

        st.metric(
            "💰 Annual Salary",
            f"₹{annual_salary:,.0f}"
        )


    st.divider()


    # =====================================================
    # PREDICTION CHART
    # =====================================================

    st.markdown(
        "### 📈 CGPA vs Expected Package"
    )


    # Generate CGPA values

    cgpa_values = np.linspace(
        0,
        10,
        100
    ).reshape(-1, 1)


    # Predict package values

    package_values = model.predict(
        cgpa_values
    )


    # Convert predictions to 1D array

    package_values = np.asarray(
        package_values
    ).flatten()


    # =====================================================
    # CREATE CHART
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )


    # Prediction curve

    ax.plot(
        cgpa_values.flatten(),
        package_values,
        linewidth=3,
        label="Prediction Curve"
    )


    # User prediction

    ax.scatter(
        [cgpa],
        [prediction],
        s=180,
        label="Your Prediction",
        zorder=5
    )


    # Chart labels

    ax.set_xlabel(
        "CGPA",
        fontsize=12
    )

    ax.set_ylabel(
        "Package (LPA)",
        fontsize=12
    )


    ax.set_title(
        "CGPA vs Expected Package",
        fontsize=16,
        fontweight="bold"
    )


    ax.grid(
        True,
        alpha=0.3
    )


    ax.legend()


    st.pyplot(fig)


    # =====================================================
    # PREDICTION SUMMARY
    # =====================================================

    st.divider()

    st.markdown(
        "### 📊 Prediction Summary"
    )


    min_package = float(
        np.min(package_values)
    )


    max_package = float(
        np.max(package_values)
    )


    summary1, summary2, summary3 = st.columns(3)


    with summary1:

        st.metric(
            "Minimum Estimated",
            f"{min_package:.2f} LPA"
        )


    with summary2:

        st.metric(
            "Your Prediction",
            f"{prediction:.2f} LPA"
        )


    with summary3:

        st.metric(
            "Maximum Estimated",
            f"{max_package:.2f} LPA"
        )


# =========================================================
# WELCOME SCREEN
# =========================================================

else:

    with result_col:

        st.markdown(
            """
            <div class="info-card">

                <h2>👋 Welcome!</h2>

                <p>
                This application uses a Machine Learning model
                to estimate an expected salary package based on
                your CGPA.
                </p>

                <br>

                <h4>🤖 Model Input</h4>
                <p>CGPA</p>

                <h4>📤 Model Output</h4>
                <p>Expected Package in LPA</p>

                <h4>📈 Visualization</h4>
                <p>CGPA vs Package Prediction Curve</p>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🎓 CGPA Package Prediction |
        Built with Python + Machine Learning + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)