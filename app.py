"""
Diabetes Readmission Prediction - Streamlit App
This script loads pre-trained machine learning models and evaluates them 
against a user-uploaded test dataset for predicting hospital readmission.
"""

import logging
from pathlib import Path

# Web app and data manipulation libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import softmax

# Machine learning evaluation metrics
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

# Define base and log directories for file management
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging to output to both console and a log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "streamlit_app.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)
logger.info("Streamlit app started from %s", BASE_DIR)

# Configure the Streamlit page layout and metadata
st.set_page_config(
    page_title="ML Assignment: Diabetes Readmission Prediction Using Machine Learning",
    page_icon="🏥",
    layout="wide"
)

# App header and description
st.title("Diabetes Readmission Prediction Using Machine Learning")
st.write(
    "Compare five machine learning classifiers "
    "using the diabetes readmission test dataset."
)

# Dictionary mapping model names to their respective saved joblib files
MODEL_PATHS = {
    "Logistic Regression": BASE_DIR / "models" / "logistic_regression.joblib",
    "Decision Tree Classifier": BASE_DIR / "models" / "decision_tree.joblib",
    "K-Nearest Neighbor Classifier": BASE_DIR / "models" / "knn.joblib",
    "Naive Bayes Classifier - Gaussian or Multinomial": BASE_DIR / "models" / "multinomial_naive_bayes.joblib",
    "Support Vector Machine": BASE_DIR / "models" / "linear_svm.joblib",
}

# The three expected target classes in the dataset
CLASS_LABELS = ["<30", ">30", "NO"]

# Set up the sidebar for user inputs
st.sidebar.header("Model Selection")

# Dropdown menu to select which pre-trained model to evaluate
selected_model = st.sidebar.selectbox(
    "Choose a classification model",
    list(MODEL_PATHS.keys())
)

# File uploader widget for the test dataset
uploaded_file = st.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"]
)

# Execute the evaluation pipeline only if a file has been uploaded
if uploaded_file is not None:

    try:
        logger.info("Reading uploaded file: %s", uploaded_file.name)
        # Read the CSV file, treating '?' as NaN based on the UCI dataset format
        data = pd.read_csv(uploaded_file, na_values=["?"], low_memory=False)

        # Display basic information about the uploaded dataset
        st.subheader("Uploaded Dataset")
        st.write("Dataset shape:", data.shape)
        st.dataframe(data.head())

        # Validate that the target column exists
        if "readmitted" not in data.columns:
            logger.error("Upload rejected: missing readmitted column")
            st.error("The uploaded CSV must contain the 'readmitted' target column.")
            st.stop()

        # Separate features (X) and target (y)
        X_uploaded = data.drop(columns=["readmitted"])
        y_actual = data["readmitted"]

        # Validate target column for missing values
        if y_actual.isnull().any():
            logger.error("Upload rejected: target column contains missing values")
            st.error("Target column contains missing values.")
            st.stop()

        # Ensure no unexpected labels exist in the target column
        unexpected_classes = sorted(set(y_actual.unique()) - set(CLASS_LABELS))
        if unexpected_classes:
            logger.error("Upload rejected: unexpected classes %s", unexpected_classes)
            st.error("Unexpected target class found in the CSV.")
            st.stop()

        # Ensure all required classes are present to avoid metric calculation errors
        missing_classes = sorted(set(CLASS_LABELS) - set(y_actual.unique()))
        if missing_classes:
            logger.error("Upload rejected: missing classes %s", missing_classes)
            st.error(
                "The uploaded dataset must contain all three target classes: "
                "<30, >30, and NO. Missing: " + ", ".join(missing_classes)
            )
            st.stop()

        # Load the selected machine learning model from disk
        logger.info("Loading model %s from %s", selected_model, MODEL_PATHS[selected_model])
        model = joblib.load(MODEL_PATHS[selected_model])

        # The original UCI dataset contains identifier and removed columns that are not model inputs.
        # Remove those automatically so the raw diabetic_data.csv can be tested directly.
        training_only_columns = {
            "encounter_id",
            "patient_nbr",
            "weight",
            "examide",
            "citoglipton",
        }
        # Find intersection of columns to drop and existing columns
        dropped_columns = sorted(set(X_uploaded.columns) & training_only_columns)
        if dropped_columns:
            logger.info("Dropping training-only columns: %s", dropped_columns)
            X_uploaded = X_uploaded.drop(columns=dropped_columns)

        # Retrieve the expected feature names from the trained pipeline's preprocessor step
        expected_columns = list(
            model.named_steps["preprocessor"].feature_names_in_
        )

        # Validate that the uploaded data features exactly match the training features
        missing_columns = sorted(set(expected_columns) - set(X_uploaded.columns))
        extra_columns = sorted(set(X_uploaded.columns) - set(expected_columns))
        if missing_columns or extra_columns:
            logger.error(
                "Feature mismatch for %s; missing=%s extra=%s",
                selected_model,
                missing_columns,
                extra_columns,
            )
            st.error("Uploaded dataset features do not match the model's expected input columns.")
            st.write("Missing columns:", missing_columns)
            st.write("Unexpected columns:", extra_columns)
            st.stop()

        # Normalize UCI unknown markers so the fitted imputers handle them consistently.
        X_uploaded = X_uploaded.replace("?", pd.NA)

        # Maintain the exact same feature column order used during model training
        X_uploaded = X_uploaded[expected_columns]

        # Generate predictions using the loaded model
        y_pred = model.predict(X_uploaded)
        logger.info("Prediction completed for model=%s rows=%d", selected_model, len(y_pred))

        # Extract the classifier step from the pipeline to access classes_
        classifier = model.named_steps["classifier"]

        # Obtain scores for AUC calculation 
        # (Linear SVM uses decision_function instead of predict_proba)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_uploaded)
        else:
            # Apply softmax to decision scores to approximate probabilities for AUC
            decision_scores = model.decision_function(X_uploaded)
            y_score = softmax(decision_scores, axis=1)

        # Calculate standard evaluation metrics
        accuracy = accuracy_score(y_actual, y_pred)

        # Macro One-vs-Rest AUC is used for multi-class evaluation
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

        # Matthews Correlation Coefficient handles class imbalances well
        mcc = matthews_corrcoef(y_actual, y_pred)

        # Display the results in the UI
        st.subheader("Selected Model")
        st.success(selected_model)

        st.subheader("Evaluation Metrics")

        # Layout metrics in two rows of three columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC (Macro OVR)", f"{auc:.4f}")
        col3.metric("Precision (Macro)", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall (Macro)", f"{recall:.4f}")
        col5.metric("F1 Score (Macro)", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # Generate and plot the Confusion matrix
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

        # Generate and display the classification report DataFrame
        st.subheader("Classification Report")

        report = classification_report(
            y_actual,
            y_pred,
            labels=CLASS_LABELS,
            target_names=CLASS_LABELS,
            output_dict=True,
            zero_division=0
        )

        # Transpose for better readability in the Streamlit dataframe
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4))

        # Add a disclaimer for the SVM's AUC approximation
        st.caption(
            "Note: Support Vector Machine uses softmax-normalized decision "
            "scores for AUC. These are not calibrated probabilities."
        )
        
        logger.info(
            "Evaluation completed: model=%s accuracy=%.4f auc=%.4f precision=%.4f recall=%.4f f1=%.4f mcc=%.4f",
            selected_model,
            accuracy,
            auc,
            precision,
            recall,
            f1,
            mcc,
        )

    except Exception as error:
        # Catch and log any unexpected errors during processing
        logger.exception("Unhandled application error: %s", error)
        st.error(f"An error occurred: {error}")

else:
    # Prompt the user to upload a dataset if none is present
    logger.info("Waiting for a test dataset upload")
    st.info("Upload your test_data.csv file to begin.")