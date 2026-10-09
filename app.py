from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from catboost import CatBoostClassifier, Pool


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "catboost_model.cbm"
METADATA_PATH = PROJECT_DIR / "models" / "model_metadata.joblib"
FEATURE_STORE_PATH = (
    PROJECT_DIR / "data" / "processed" / "applicant_feature_store.parquet"
)
FAVICON_PATH = PROJECT_DIR / "assets" / "favicon.png"

APP_NAME = "Credit Risk Intelligence"

st.set_page_config(
    page_title=APP_NAME,
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Explicit colours (no default Plotly palette, no purple).
RISK_ORDER = ["Lower Risk", "Moderate Risk", "Elevated Risk", "High Risk"]
RISK_COLORS = {
    "Lower Risk": "#2E7D5B",
    "Moderate Risk": "#C9A227",
    "Elevated Risk": "#D9772B",
    "High Risk": "#B3392F",
}
DIRECTION_COLORS = {
    "Pushes prediction toward default": "#B3392F",
    "Pushes prediction away from default": "#1F6FB2",
}
NEUTRAL_BAR = "#1F6FB2"


# ============================================================
# 2. CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        .main-title {
            font-size: 2.3rem;
            font-weight: 750;
            letter-spacing: -1px;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #9CA3AF;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .risk-card {
            padding: 16px 18px;
            border-radius: 8px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-left-width: 6px;
            background: rgba(128, 128, 128, 0.06);
        }

        .risk-label {
            font-size: 0.85rem;
            color: #9CA3AF;
            margin-bottom: 4px;
        }

        .risk-value {
            font-size: 1.6rem;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.06);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 16px;
            border-radius: 8px;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 6px;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    return model


@st.cache_data
def load_metadata():
    return joblib.load(METADATA_PATH)


@st.cache_data
def load_feature_store():
    df = pd.read_parquet(FEATURE_STORE_PATH)

    if "SK_ID_CURR" not in df.columns:
        raise ValueError("SK_ID_CURR is missing from the feature store.")

    if df["SK_ID_CURR"].duplicated().any():
        raise ValueError("Duplicate applicant IDs found in the feature store.")

    return df


# ============================================================
# 4. RISK CLASSIFICATION
# ============================================================

def classify_risk(probability):
    """Return (label, colour). Thresholds are illustrative."""
    if probability < 0.05:
        label = "Lower Risk"
    elif probability < 0.10:
        label = "Moderate Risk"
    elif probability < 0.20:
        label = "Elevated Risk"
    else:
        label = "High Risk"
    return label, RISK_COLORS[label]


def assign_risk_band(probability):
    return classify_risk(probability)[0]


# ============================================================
# 5. PREDICTION UTILITIES
# ============================================================

def prepare_model_input(row, feature_columns, categorical_features):
    """
    Align applicant features with the exact feature order used
    during training and convert categorical values to strings.
    """

    X = row.reindex(columns=feature_columns).copy()

    for col in categorical_features:
        if col not in X.columns:
            raise ValueError(f"Missing categorical feature: {col}")

        X[col] = (
            X[col]
            .astype("object")
            .where(X[col].notna(), "MISSING")
            .astype(str)
        )

    return X


def predict_probability(model, X):
    return model.predict_proba(X)[:, 1]


def calculate_shap_values(model, X, categorical_features):
    """
    CatBoost native SHAP values for an individual applicant.

    Positive contributions push the prediction toward default.
    Negative contributions push it away from default.
    """

    cat_indices = [
        X.columns.get_loc(col)
        for col in categorical_features
        if col in X.columns
    ]

    pool = Pool(data=X, cat_features=cat_indices)

    shap_values = model.get_feature_importance(data=pool, type="ShapValues")

    contributions = shap_values[0, :-1]

    explanation = pd.DataFrame(
        {
            "Feature": X.columns,
            "SHAP contribution": contributions,
            "Absolute contribution": np.abs(contributions),
        }
    )

    return explanation.sort_values("Absolute contribution", ascending=False)


# ============================================================
# 6. VALIDATE FILES AND LOAD RESOURCES
# ============================================================

required_files = [MODEL_PATH, METADATA_PATH, FEATURE_STORE_PATH]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    st.error("Required project files are missing:")

    for file_path in missing_files:
        st.code(file_path)

    st.info(
        "Check that app.py is in the project root and that the model "
        "and feature store were saved in the expected folders."
    )

    st.stop()


try:
    model = load_model()
    metadata = load_metadata()
    applicants = load_feature_store()

except Exception as exc:
    st.error("The application could not load its model or data.")
    st.exception(exc)
    st.stop()


FEATURE_COLUMNS = metadata["feature_columns"]
CATEGORICAL_FEATURES = metadata["categorical_features"]

missing_features = [
    col for col in FEATURE_COLUMNS if col not in applicants.columns
]

if missing_features:
    st.error("The feature store does not match the trained model.")
    st.write("Missing features:", missing_features)
    st.stop()

# Enforce the exact model feature order.
applicants = applicants[["SK_ID_CURR"] + FEATURE_COLUMNS].copy()

# Model inference needs the same categorical representation as training.
for col in CATEGORICAL_FEATURES:
    applicants[col] = (
        applicants[col]
        .astype("object")
        .where(applicants[col].notna(), "MISSING")
        .astype(str)
    )


# ============================================================
# 7. HEADER
# ============================================================

st.markdown(
    f'<div class="main-title">{APP_NAME}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Default-probability scoring and per-applicant explanations "
    "with a CatBoost model"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    f"Validation sample: {len(applicants):,} applicants from the Home Credit "
    "Default Risk dataset. These are historical records, not a live lending "
    "portfolio. Predictions are model estimates, not lending decisions."
)

st.divider()


# ============================================================
# 8. PRECOMPUTE VALIDATION-SAMPLE PREDICTIONS
# ============================================================

@st.cache_data(show_spinner="Calculating applicant risk scores...")
def calculate_portfolio_predictions(feature_df, feature_columns):
    """Generate probabilities in batches."""

    feature_data = feature_df[feature_columns]
    results = []

    batch_size = 5000

    for start in range(0, len(feature_data), batch_size):
        batch = feature_data.iloc[start:start + batch_size]
        batch_probabilities = model.predict_proba(batch)[:, 1]
        results.extend(batch_probabilities.tolist())

    output = pd.DataFrame(
        {
            "SK_ID_CURR": feature_df["SK_ID_CURR"].values,
            "Predicted Default Probability": results,
        }
    )

    output["Risk Category"] = output[
        "Predicted Default Probability"
    ].apply(assign_risk_band)

    return output


portfolio_predictions = calculate_portfolio_predictions(
    applicants,
    FEATURE_COLUMNS,
)


# ============================================================
# 9. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Validation Sample Overview",
        "Applicant Analysis",
        "Risk Explorer",
        "Model Information",
        "Privacy and Terms",
    ],
)

st.sidebar.divider()

st.sidebar.metric(
    "Validation Sample",
    f"{len(applicants):,} applicants",
)

st.sidebar.caption(
    "Model: CatBoost Classifier\n\n"
    "Validation ROC-AUC: 0.7885\n\n"
    "Validation PR-AUC: 0.2893"
)


# ============================================================
# 10. VALIDATION SAMPLE OVERVIEW
# ============================================================

if page == "Validation Sample Overview":

    st.subheader("Validation Sample Overview")

    total_applicants = len(portfolio_predictions)

    average_risk = portfolio_predictions["Predicted Default Probability"].mean()

    high_risk_count = (
        portfolio_predictions["Risk Category"] == "High Risk"
    ).sum()

    elevated_or_high = portfolio_predictions["Risk Category"].isin(
        ["Elevated Risk", "High Risk"]
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Validation Applicants", f"{total_applicants:,}")

    col2.metric("Average Predicted Risk", f"{average_risk:.2%}")

    col3.metric(
        "High Risk",
        f"{high_risk_count:,}",
        f"{high_risk_count / total_applicants:.1%} of sample",
        delta_color="off",
    )

    col4.metric(
        "Elevated + High Risk",
        f"{elevated_or_high:,}",
        f"{elevated_or_high / total_applicants:.1%} of sample",
        delta_color="off",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("#### Applicant Risk Distribution")

        risk_counts = (
            portfolio_predictions["Risk Category"]
            .value_counts()
            .reindex(RISK_ORDER, fill_value=0)
            .rename_axis("Risk Category")
            .reset_index(name="Applicants")
        )

        fig = px.bar(
            risk_counts,
            x="Risk Category",
            y="Applicants",
            color="Risk Category",
            color_discrete_map=RISK_COLORS,
            category_orders={"Risk Category": RISK_ORDER},
            text="Applicants",
            title="Applicants by Risk Category",
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Number of Applicants",
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Predicted Risk Distribution")

        fig = px.histogram(
            portfolio_predictions,
            x="Predicted Default Probability",
            nbins=40,
            title="Distribution of Predicted Default Probabilities",
            color_discrete_sequence=[NEUTRAL_BAR],
        )

        fig.update_layout(
            xaxis_title="Predicted Probability",
            yaxis_title="Number of Applicants",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("#### Highest Predicted Risk Applicants")

    highest_risk = (
        portfolio_predictions.sort_values(
            "Predicted Default Probability", ascending=False
        )
        .head(15)
        .copy()
    )

    highest_risk["Predicted Default Probability"] = highest_risk[
        "Predicted Default Probability"
    ].map(lambda x: f"{x:.2%}")

    st.dataframe(highest_risk, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Validation-Sample Predictions (CSV)",
        data=portfolio_predictions.to_csv(index=False).encode("utf-8"),
        file_name="credit_risk_predictions.csv",
        mime="text/csv",
    )


# ============================================================
# 11. APPLICANT ANALYSIS
# ============================================================

elif page == "Applicant Analysis":

    st.subheader("Individual Applicant Analysis")

    st.write(
        "Select an applicant from the validation sample to inspect the "
        "model's predicted default probability and the features driving it."
    )

    applicant_ids = applicants["SK_ID_CURR"].astype(int).tolist()

    selected_id = st.selectbox("Applicant ID", options=applicant_ids, index=0)

    selected_row = applicants.loc[
        applicants["SK_ID_CURR"] == selected_id,
        FEATURE_COLUMNS,
    ]

    if selected_row.empty:
        st.error("Applicant not found.")
        st.stop()

    X_selected = prepare_model_input(
        selected_row,
        FEATURE_COLUMNS,
        CATEGORICAL_FEATURES,
    )

    probability = float(predict_probability(model, X_selected)[0])

    risk_label, risk_color = classify_risk(probability)

    col1, col2, col3 = st.columns(3)

    col1.metric("Applicant ID", f"{selected_id}")

    col2.metric("Predicted Default Probability", f"{probability:.2%}")

    with col3:
        st.markdown(
            f'<div class="risk-card" style="border-left-color:{risk_color};">'
            '<div class="risk-label">Risk Category</div>'
            f'<div class="risk-value" style="color:{risk_color};">'
            f"{risk_label}</div></div>",
            unsafe_allow_html=True,
        )

    st.progress(
        min(max(probability, 0.0), 1.0),
        text=f"Predicted probability: {probability:.2%}",
    )

    st.caption(
        "Risk categories use illustrative thresholds: 5%, 10%, and 20%. "
        "They are not calibrated lending-policy cutoffs."
    )

    st.divider()

    st.markdown("#### Applicant Explanation")

    with st.spinner("Calculating individual feature contributions..."):
        try:
            explanation = calculate_shap_values(
                model,
                X_selected,
                CATEGORICAL_FEATURES,
            )

            top_features = explanation.head(15).sort_values(
                "SHAP contribution", ascending=True
            )

            top_features["Direction"] = np.where(
                top_features["SHAP contribution"] > 0,
                "Pushes prediction toward default",
                "Pushes prediction away from default",
            )

            fig = px.bar(
                top_features,
                x="SHAP contribution",
                y="Feature",
                color="Direction",
                color_discrete_map=DIRECTION_COLORS,
                orientation="h",
                title="Top 15 Feature Contributions",
                hover_data={
                    "SHAP contribution": ":.5f",
                    "Direction": True,
                },
            )

            fig.add_vline(x=0, line_width=1)

            fig.update_layout(
                xaxis_title="SHAP contribution (model output scale)",
                yaxis_title="",
                height=600,
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info(
                "Positive SHAP values push the model output toward default; "
                "negative values push it away. These are contributions to "
                "the model's raw output, not percentage-point changes in "
                "default probability. They describe model behavior, not "
                "causal effects."
            )

            with st.expander("View feature contribution details"):
                st.dataframe(
                    explanation[
                        ["Feature", "SHAP contribution", "Absolute contribution"]
                    ].head(20),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as exc:
            st.warning(
                "The prediction succeeded, but the feature explanation "
                "could not be calculated."
            )
            st.exception(exc)

    with st.expander("View applicant input features"):
        st.dataframe(
            X_selected.T.rename(columns={X_selected.index[0]: "Value"}),
            use_container_width=True,
        )

    applicant_download = pd.DataFrame(
        {
            "SK_ID_CURR": [selected_id],
            "Predicted Default Probability": [probability],
            "Risk Category": [risk_label],
        }
    )

    st.download_button(
        "Download Applicant Prediction",
        data=applicant_download.to_csv(index=False).encode("utf-8"),
        file_name=f"applicant_{selected_id}_prediction.csv",
        mime="text/csv",
    )


# ============================================================
# 12. RISK EXPLORER
# ============================================================

elif page == "Risk Explorer":

    st.subheader("Risk Explorer")

    st.write(
        "Filter the validation sample by predicted default probability."
    )

    minimum_risk, maximum_risk = st.slider(
        "Predicted default probability range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
        format="%.2f",
    )

    filtered = portfolio_predictions[
        portfolio_predictions["Predicted Default Probability"].between(
            minimum_risk, maximum_risk
        )
    ].copy()

    col1, col2, col3 = st.columns(3)

    col1.metric("Matching Applicants", f"{len(filtered):,}")

    col2.metric(
        "Average Predicted Risk",
        f"{filtered['Predicted Default Probability'].mean():.2%}"
        if not filtered.empty
        else "N/A",
    )

    col3.metric(
        "Median Predicted Risk",
        f"{filtered['Predicted Default Probability'].median():.2%}"
        if not filtered.empty
        else "N/A",
    )

    if filtered.empty:
        st.info("No applicants fall in the selected range.")
    else:
        fig = px.histogram(
            filtered,
            x="Predicted Default Probability",
            color="Risk Category",
            color_discrete_map=RISK_COLORS,
            category_orders={"Risk Category": RISK_ORDER},
            nbins=35,
            title="Filtered Applicant Risk Distribution",
        )

        fig.update_layout(yaxis_title="Number of Applicants")

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Filtered Applicants")

    st.dataframe(
        filtered.sort_values("Predicted Default Probability", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download Filtered Applicants",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_credit_risk_applicants.csv",
        mime="text/csv",
    )


# ============================================================
# 13. MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.subheader("Model Information")

    st.markdown("#### Validation Performance")

    metric_col1, metric_col2 = st.columns(2)

    metric_col1.metric(
        "ROC-AUC",
        "0.7885",
        help="Measures ranking discrimination across thresholds.",
    )

    metric_col2.metric(
        "PR-AUC",
        "0.2893",
        help=(
            "Average precision on the validation set. "
            "The default class is the positive class."
        ),
    )

    st.markdown("#### Model Details")

    model_details = pd.DataFrame(
        {
            "Property": [
                "Algorithm",
                "Training framework",
                "Number of model features",
                "Categorical features",
                "Validation split",
                "Validation sample size",
                "Random seed",
                "Risk thresholds",
            ],
            "Value": [
                "CatBoost Classifier",
                "CatBoost",
                str(len(FEATURE_COLUMNS)),
                str(len(CATEGORICAL_FEATURES)),
                "80% training / 20% validation",
                f"{len(applicants):,} applicants",
                "42",
                "5%, 10%, 20% (illustrative)",
            ],
        }
    )

    st.dataframe(model_details, use_container_width=True, hide_index=True)

    st.markdown("#### Model Feature Importance")

    with st.spinner("Loading model feature importance..."):
        importance_values = model.get_feature_importance(
            type="FeatureImportance"
        )

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": importance_values,
        }
    ).sort_values("Importance", ascending=False)

    fig = px.bar(
        importance_df.head(20).sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 20 Global Features",
        color_discrete_sequence=[NEUTRAL_BAR],
    )

    fig.update_layout(height=650)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View all feature importances"):
        st.dataframe(importance_df, use_container_width=True, hide_index=True)

    st.warning(
        "This application is an educational portfolio project. "
        "Its predictions should not be used as the sole basis for real "
        "credit approvals, rejections, pricing, or other consequential "
        "decisions. The model requires additional calibration, fairness "
        "testing, governance, and external validation before operational use."
    )


# ============================================================
# 14. PRIVACY AND TERMS
# ============================================================

elif page == "Privacy and Terms":

    st.subheader("Privacy and Terms")

    st.markdown("#### Privacy")

    st.markdown(
        """
        - This application has no sign-in and no form for entering personal
          information. It only displays records from a fixed dataset.
        - The dataset is the public Home Credit Default Risk dataset. Applicants
          appear only as numeric IDs; this app does not link them to named
          individuals.
        - The app does not store anything you select or filter, and it does
          not run its own analytics or set its own tracking cookies.
        - CSV downloads are generated in your browser session from the
          dataset shown on screen.
        - The hosting provider may collect standard technical logs (such as IP
          address and request time) under its own privacy policy. This app
          does not control that collection.
        """
    )

    st.markdown("#### Terms of Use")

    st.markdown(
        """
        - This is an educational portfolio demonstration, provided as is,
          without warranties of any kind.
        - Predicted probabilities and risk categories are statistical model
          outputs. They are not credit decisions, financial advice, or legal
          advice.
        - Do not use this application or its outputs to approve, reject, or
          price credit, or to make any other decision about a real person.
        - The model has not been calibrated, tested for fairness, or
          independently validated for operational use.
        - The underlying dataset is subject to its own licence and terms from
          its publisher.
        """
    )

    st.caption("Last updated: October 2026")


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.caption(
    f"{APP_NAME} | CatBoost + Streamlit | "
    "Home Credit Default Risk dataset (validation sample)"
)