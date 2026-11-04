import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import softmax

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Diabetes Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("Diabetes Readmission Prediction")
st.write(
    "Compare five machine learning classifiers "
    "using the diabetes readmission test dataset."
)

MODEL_PATHS = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "KNN": "models/knn.joblib",
    "Multinomial Naive Bayes": "models/multinomial_naive_bayes.joblib",
    "Linear SVM": "models/linear_svm.joblib"
}

CLASS_LABELS = ["<30", ">30", "NO"]

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a classification model",
    list(MODEL_PATHS.keys())
)

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.write("Dataset shape:", data.shape)
        st.dataframe(data.head())

        if "readmitted" not in data.columns:
            st.error(
                "The uploaded CSV must contain "
                "the 'readmitted' target column."
            )
            st.stop()

        X_uploaded = data.drop(columns=["readmitted"])
        y_actual = data["readmitted"]

        if y_actual.isnull().any():
            st.error("Target column contains missing values.")
            st.stop()

        if not set(y_actual.unique()).issubset(set(CLASS_LABELS)):
            st.error("Unexpected target class found in the CSV.")
            st.stop()

        # Load selected trained model
        model = joblib.load(MODEL_PATHS[selected_model])

        # Validate feature columns
        expected_columns = list(
            model.named_steps["preprocessor"].feature_names_in_
        )

        if set(X_uploaded.columns) != set(expected_columns):
            st.error(
                "Uploaded dataset features do not match "
                "the model's expected input columns."
            )
            st.write("Expected columns:", expected_columns)
            st.stop()

        # Maintain the same feature order used during training
        X_uploaded = X_uploaded[expected_columns]

        # Predict
        y_pred = model.predict(X_uploaded)

        classifier = model.named_steps["classifier"]

        # Obtain scores for AUC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_uploaded)
        else:
            decision_scores = model.decision_function(X_uploaded)
            y_score = softmax(decision_scores, axis=1)

        # Evaluation metrics
        accuracy = accuracy_score(y_actual, y_pred)

        auc = roc_auc_score(
            y_actual,
            y_score,
            labels=classifier.classes_,
            multi_class="ovr",
            average="macro"
        )

        precision = precision_score(
            y_actual, y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_actual, y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_actual, y_pred,
            average="macro",
            zero_division=0
        )

        mcc = matthews_corrcoef(y_actual, y_pred)

        st.subheader("Selected Model")
        st.success(selected_model)

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC (Macro OVR)", f"{auc:.4f}")
        col3.metric("Precision (Macro)", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall (Macro)", f"{recall:.4f}")
        col5.metric("F1 Score (Macro)", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # Confusion matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_actual,
            y_pred,
            labels=CLASS_LABELS
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        st.pyplot(fig)
        plt.close(fig)

        # Classification report
        st.subheader("Classification Report")

        report = classification_report(
            y_actual,
            y_pred,
            labels=CLASS_LABELS,
            target_names=CLASS_LABELS,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df.round(4))

        st.caption(
            "Note: Linear SVM uses softmax-normalized decision "
            "scores for AUC. These are not calibrated probabilities."
        )

    except Exception as error:
        st.error(f"An error occurred: {error}")

else:
    st.info("Upload your test_data.csv file to begin.")