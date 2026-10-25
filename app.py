
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODEL_PATH = BASE_DIR / "models" / "student_model.joblib"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.joblib"
RESULTS_PATH = BASE_DIR / "models" / "model_results.csv"

# -------------------- CUSTOM DESIGN --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101426 0%, #17243c 55%, #182b3c 100%);
    color: #f4f7fb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 28px;
    border-radius: 22px;
    background: linear-gradient(120deg, #293e75, #6045a8, #176c83);
    margin-bottom: 24px;
}
.hero h1 { color: white; font-size: 2.5rem; }
.hero p { color: #e4eaff; font-size: 1.05rem; }
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 18px;
    border-radius: 16px;
}
div[data-testid="stForm"] {
    background: rgba(255,255,255,0.04);
    padding: 20px;
    border-radius: 16px;
}
h2, h3 { color: #b9c9ff; }
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(90deg, #7657df, #248ca3);
    color: white;
    border: 0;
    border-radius: 10px;
    padding: 10px 20px;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD FILES --------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, sep=";")

@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)

try:
    model = load_model()
    feature_names = joblib.load(FEATURE_PATH)
    df = load_data()
    model_results = load_results()
except Exception as exc:
    st.error("Project files could not be loaded.")
    st.write("Please confirm that training finished and the models folder contains all three saved files.")
    st.code(str(exc))
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("# 🎓 EduPredict AI")
st.sidebar.caption("Student Performance Intelligence")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Predict Score", "Data Analytics", "Model Performance", "About Project"]
)
st.sidebar.divider()
st.sidebar.info(
    "This tool provides estimates from historical data. "
    "It does not determine a student's ability or guarantee future results."
)

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
    <h1>🎓 EduPredict AI</h1>
    <p>Smart Student Performance Analytics powered by Machine Learning</p>
    <p>Explore patterns, estimate final grades and discover helpful study insights.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- OVERVIEW --------------------
if page == "Overview":
    st.subheader("📊 Project Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students in Dataset", f"{len(df):,}")
    c2.metric("Dataset Features", len(feature_names))
    c3.metric("Average Final Grade", f"{df['G3'].mean():.2f}/20")
    c4.metric("Model Candidates", len(model_results))

    st.markdown("### Student Grade Distribution")
    fig = px.histogram(
        df, x="G3", nbins=20,
        title="Distribution of Final Grades",
        labels={"G3": "Final Grade (0–20)", "count": "Students"},
        color_discrete_sequence=["#8b79ef"]
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.caption("Dataset: UCI Student Performance — Mathematics records.")

# -------------------- PREDICT SCORE --------------------
elif page == "Predict Score":
    st.subheader("✨ Estimate a Student's Final Grade")
    st.write("Enter the available details. The model estimates a grade on a 0–20 scale.")

    with st.form("student_form"):
        left, right = st.columns(2)

        with left:
            age = st.slider("Age", 15, 22, 17)
            studytime = st.select_slider(
                "Weekly study time",
                options=[1, 2, 3, 4],
                value=2,
                format_func=lambda x: {
                    1: "Under 2 hours",
                    2: "2–5 hours",
                    3: "5–10 hours",
                    4: "Over 10 hours"
                }[x]
            )
            failures = st.selectbox("Past class failures", [0, 1, 2, 3], index=0)
            absences = st.number_input("School absences", min_value=0, max_value=100, value=2)
            schoolsup = st.selectbox("Extra educational support?", ["no", "yes"])

        with right:
            internet = st.selectbox("Internet access at home?", ["yes", "no"])
            goout = st.slider("Going out with friends (1–5)", 1, 5, 3)
            health = st.slider("Self-reported health (1–5)", 1, 5, 3)
            activities = st.selectbox("Extra-curricular activities?", ["yes", "no"])
            higher = st.selectbox("Wants higher education?", ["yes", "no"])

        submitted = st.form_submit_button("🚀 Predict Final Grade", use_container_width=True)

    if submitted:
        # Create every expected input column with a training-data median/mode.
        # User-entered values override the relevant defaults.
        row = {}
        for col in feature_names:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    row[col] = df[col].median()
                else:
                    row[col] = df[col].mode(dropna=True).iloc[0]
            else:
                row[col] = 0

        user_values = {
            "age": age,
            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "schoolsup": schoolsup,
            "internet": internet,
            "goout": goout,
            "health": health,
            "activities": activities,
            "higher": higher
        }

        for col, value in user_values.items():
            if col in row:
                row[col] = value

        input_df = pd.DataFrame([row], columns=feature_names)

        try:
            prediction = float(model.predict(input_df)[0])
            prediction = float(np.clip(prediction, 0, 20))
            st.markdown("### Your Prediction")
            m1, m2 = st.columns(2)
            m1.metric("Estimated Final Grade", f"{prediction:.1f} / 20")
            m2.metric("Approximate Percentage", f"{prediction / 20 * 100:.1f}%")

            st.progress(prediction / 20)

            st.info(
                "This is a model estimate based on historical patterns, not a guaranteed result. "
                "Use it as a learning aid, not as a judgment of the student."
            )

            st.markdown("### Helpful Study Suggestions")
            suggestions = []
            if studytime <= 2:
                suggestions.append("Create a consistent study routine with short, focused sessions.")
            else:
                suggestions.append("Keep your study routine balanced and include regular revision.")
            if absences > 5:
                suggestions.append("Try to attend classes regularly and catch up on missed topics.")
            if failures > 0:
                suggestions.append("Review difficult topics and ask a teacher for support where needed.")
            if schoolsup == "no":
                suggestions.append("Consider study groups, peer support or available learning resources.")
            suggestions.append("Set small weekly learning goals and track your progress.")
            for item in suggestions:
                st.write(f"• {item}")

        except Exception as exc:
            st.error("Prediction could not be completed.")
            st.code(str(exc))

# -------------------- DATA ANALYTICS --------------------
elif page == "Data Analytics":
    st.subheader("📈 Explore Student Data")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x="studytime", color="sex", barmode="group",
            title="Study Time by Group",
            labels={"studytime": "Study Time Category", "sex": "Group"},
            color_discrete_sequence=["#40c9a2", "#9878f5"]
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df, x="schoolsup", y="G3", color="schoolsup",
            title="Final Grades and Extra Educational Support",
            labels={"schoolsup": "Extra Support", "G3": "Final Grade"},
            color_discrete_sequence=["#40c9a2", "#9878f5"]
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Grade Summary")
    st.dataframe(df[["G3", "studytime", "absences", "failures"]].describe(),
                 use_container_width=True)

    st.markdown("### Filter Students by Final Grade")
    min_grade, max_grade = st.slider(
        "Choose grade range",
        min_value=0, max_value=20, value=(0, 20)
    )
    filtered = df[df["G3"].between(min_grade, max_grade)]
    st.write(f"Matching records: {len(filtered)}")
    st.dataframe(filtered, use_container_width=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered CSV",
        data=csv,
        file_name="edupredict_filtered_students.csv",
        mime="text/csv"
    )

# -------------------- MODEL PERFORMANCE --------------------
elif page == "Model Performance":
    st.subheader("🧠 Model Evaluation")
    st.write("Models are evaluated on a held-out test split. Lower MAE and RMSE are better.")
    st.dataframe(model_results, use_container_width=True)

    if {"Model", "MAE"}.issubset(model_results.columns):
        fig = px.bar(
            model_results, x="Model", y="MAE",
            title="Mean Absolute Error by Model",
            labels={"MAE": "Mean Absolute Error (grade points)"},
            color="Model"
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.caption("MAE measures average absolute prediction error on the test set.")

# -------------------- ABOUT --------------------
elif page == "About Project":
    st.subheader("💡 About EduPredict AI")
    st.write("""
    EduPredict AI is a student performance analytics project built with Python,
    Pandas, scikit-learn, Plotly and Streamlit.

    **Main workflow**
    1. Load and inspect historical student data.
    2. Preprocess numeric and categorical features.
    3. Compare regression models against a simple baseline.
    4. Estimate final grades using available student information.
    5. Display analytics and helpful study suggestions.
    """)
    st.markdown("### Technology Stack")
    st.write("Python • Pandas • NumPy • Scikit-learn • Plotly • Streamlit • Joblib")
    st.markdown("### Dataset")
    st.write("UCI Student Performance dataset — Mathematics subject.")
    st.caption("Educational demonstration only. Predictions may be inaccurate and should not be used for high-stakes decisions.")
