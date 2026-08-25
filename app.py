
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from datetime import datetime
from zoneinfo import ZoneInfo
import sqlite3
import uuid

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM UI STYLING
# ============================================================

st.markdown(
    dedent("""
    <style>

/* Reduce space above the main page title */
.block-container {
    padding-top: 2rem !important;
}
    .prediction-summary-box {
        background: #151a22;
        border: 1px solid #2a313c;
        border-radius: 12px;
        padding: 16px 18px;
        margin: 8px 0 18px 0;
    }

    .prediction-summary-title {
        color: #f8fafc;
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .prediction-summary-text {
        color: #a7afbd;
        font-size: 0.84rem;
        line-height: 1.5;
    }

    .model-card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        margin-bottom: 15px;
    }

    .model-card-title {
        color: #f8fafc;
        font-size: 1.20rem;
        font-weight: 700;
        line-height: 1.25;
    }

    .selected-model-badge {
        color: #fbbf24;
        background: rgba(245, 158, 11, 0.10);
        border: 1px solid rgba(245, 158, 11, 0.30);
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 0.68rem;
        font-weight: 700;
        white-space: nowrap;
    }

.prediction-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;

    color: #f3f7a6;
    background: rgba(210, 220, 90, 0.10);
    border: 1px solid rgba(210, 220, 90, 0.28);

    font-size: 0.92rem;
    font-weight: 700;

    padding: 7px 11px;
    border-radius: 8px;

    margin-bottom: 18px;
}
.prediction-status-absent {
    display: inline-flex;
    align-items: center;
    gap: 7px;

    color: #8ed0ff;
    background: rgba(80, 170, 240, 0.10);
    border: 1px solid rgba(80, 170, 240, 0.28);

    font-size: 0.92rem;
    font-weight: 700;

    padding: 7px 11px;
    border-radius: 8px;

    margin-bottom: 18px;
}

    .probability-row {
        display: flex;
        justify-content: space-between;
        align-items: end;
        gap: 10px;
        margin-bottom: 9px;
    }

    .probability-label {
        color: #9ca3af;
        font-size: 0.77rem;
    }

    .probability-value {
        color: #ffffff;
        font-size: 1.70rem;
        font-weight: 700;
        line-height: 1;
    }

    .probability-track {
        width: 100%;
        height: 7px;
        background: #282d37;
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 8px;
    }

    .probability-fill {
        height: 100%;
        background: #2b8de5;
        border-radius: 999px;
    }

    .probability-caption {
        color: #747d8b;
        font-size: 0.70rem;
        margin-bottom: 7px;
    }

.roc-info-card {
    border: 1px solid #2a313c;
    border-radius: 10px;
    padding: 18px 20px;
    min-height: 220px;
    margin-bottom: 18px;
}

.roc-info-label {
    color: #9ca3af;
    font-size: 0.76rem;
    margin-bottom: 3px;
}

.roc-info-value {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 600;
    line-height: 1.1;
    margin-bottom: 18px;
}

.roc-info-title {
    color: #f8fafc;
    font-size: 0.92rem;
    font-weight: 700;
    margin-bottom: 7px;
}

.roc-info-text {
    color: #d0d5dd;
    font-size: 0.82rem;
    line-height: 1.55;
    margin-bottom: 10px;
}

.roc-info-note {
    color: #858e9c;
    font-size: 0.74rem;
    line-height: 1.5;
}

.roc-mini-divider {
    height: 1px;
    background: #2a313c;
    margin: 15px 0;
}
/* =========================================================
   FIXED MODEL ANALYSIS DIALOG SIZE
   ========================================================= */

div[role="dialog"] {
    height: min(760px, 86vh) !important;
    min-height: min(760px, 86vh) !important;
    max-height: min(760px, 86vh) !important;

    overflow-y: auto !important;
}
/* =========================================================
   MODEL DETAIL DIALOG HEADER
   ========================================================= */

/* Main dialog title: Model Detailed Analysis */
div[role="dialog"] h2 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    margin-bottom: 0 !important;
}

/* Model name below dialog title */
.model-dialog-name {
    color: #f8fafc;
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.2;

    margin-top: -12px;
    margin-bottom: 12px;
}

.upload-stat-card {
    border: 1px solid #2a313c;
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 125px;
}

.upload-stat-label {
    color: #f8fafc;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 10px;
}

.upload-stat-value {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 500;
    line-height: 1.1;
}

    </style>
    """),
    unsafe_allow_html=True
)
# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_ROOT / "models"

# ------------------------------------------------------------
# Prediction history storage
# ------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_HISTORY_DB_PATH = (
    DATA_DIR
    / "prediction_history.db"
)

MODEL_BUNDLE_PATH = (
    MODELS_DIR
    / "model_bundle_portable.joblib"
)

XGBOOST_PREPROCESSING_PATH = (
    MODELS_DIR
    / "xgboost_preprocessing.joblib"
)

XGBOOST_MODEL_PATH = (
    MODELS_DIR
    / "xgboost_model.ubj"
)

# ============================================================
# LOAD MODEL BUNDLE
# ============================================================

@st.cache_resource
def load_model_bundle():

    try:

        # ----------------------------------------------------
        # Load portable model bundle
        # ----------------------------------------------------

        bundle = joblib.load(
            MODEL_BUNDLE_PATH
        )

        # ----------------------------------------------------
        # Load XGBoost preprocessing pipeline
        # ----------------------------------------------------

        xgboost_preprocessing = joblib.load(
            XGBOOST_PREPROCESSING_PATH
        )

        # ----------------------------------------------------
        # Load portable XGBoost classifier
        # ----------------------------------------------------

        xgboost_classifier = XGBClassifier()

        xgboost_classifier.load_model(
            XGBOOST_MODEL_PATH
        )

        # ----------------------------------------------------
        # Rebuild complete XGBoost pipeline
        # ----------------------------------------------------

        xgboost_pipeline = Pipeline(
            steps=[
                *xgboost_preprocessing.steps,
                (
                    "classifier",
                    xgboost_classifier
                )
            ]
        )

        # ----------------------------------------------------
        # Restore XGBoost into the model bundle
        # ----------------------------------------------------

        bundle[
            "models"
        ][
            "XGBoost"
        ] = xgboost_pipeline

        return bundle

    except FileNotFoundError as error:

        st.error(
            "One or more required model files could not be found."
        )

        st.error(
            str(error)
        )

        st.stop()

    except Exception as error:

        st.error(
            "The model bundle could not be loaded."
        )

        st.error(
            str(error)
        )

        st.stop()


bundle = load_model_bundle()


# ============================================================
# LOAD MODEL INFORMATION
# ============================================================

models = bundle["models"]

preferred_model_name = (
    bundle["preferred_model_name"]
)

required_features = (
    bundle["required_features"]
)

categorical_valid_values = (
    bundle["categorical_valid_values"]
)

numerical_validation_ranges = (
    bundle["numerical_validation_ranges"]
)

class_labels = (
    bundle["class_labels"]
)

final_test_results = (
    bundle["final_test_results"]
)
selected_features_by_model = (
    bundle.get(
        "selected_features",
        {}
    )
)

roc_curve_data = (
    bundle.get(
        "roc_curve_data",
        {}
    )
)

selected_feature_counts = (
    bundle.get(
        "selected_feature_counts",
        {}
    )
)

total_transformed_features = (
    bundle.get(
        "metadata",
        {}
    ).get(
        "total_transformed_features",
        21
    )
)
# ============================================================
# PREDICTION HISTORY STORAGE
# ============================================================

def initialize_prediction_history_database():
    """
    Create the prediction-history database and table if they
    do not already exist.
    """

    with sqlite3.connect(
        PREDICTION_HISTORY_DB_PATH
    ) as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (

                prediction_id TEXT PRIMARY KEY,

                prediction_datetime TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                prediction_time TEXT NOT NULL,

                case_id TEXT,

                age INTEGER,
                sex TEXT,
                chest_pain_type TEXT,
                resting_bp REAL,
                cholesterol REAL,
                fasting_bs INTEGER,
                resting_ecg TEXT,
                max_hr REAL,
                exercise_angina TEXT,
                oldpeak REAL,
                st_slope TEXT,

                selected_model TEXT NOT NULL,

                final_predicted_class INTEGER NOT NULL,
                final_prediction TEXT NOT NULL,

                probability_no_heart_disease REAL NOT NULL,
                probability_heart_disease REAL NOT NULL,

                ann_predicted_class INTEGER,
                ann_prediction TEXT,
                ann_probability_no REAL,
                ann_probability_yes REAL,

                svm_predicted_class INTEGER,
                svm_prediction TEXT,
                svm_probability_no REAL,
                svm_probability_yes REAL,

                random_forest_predicted_class INTEGER,
                random_forest_prediction TEXT,
                random_forest_probability_no REAL,
                random_forest_probability_yes REAL,

                xgboost_predicted_class INTEGER,
                xgboost_prediction TEXT,
                xgboost_probability_no REAL,
                xgboost_probability_yes REAL,

                presence_votes INTEGER,
                absence_votes INTEGER,

                model_agreement TEXT
            )
            """
        )

        connection.commit()

def get_model_history_values(
    comparison_results,
    model_name
):
    """
    Extract the prediction and probabilities for one model.
    """

    model_rows = comparison_results[
        comparison_results["Model"] == model_name
    ]

    if model_rows.empty:

        return (
            None,
            None,
            None,
            None
        )

    model_row = model_rows.iloc[0]

    return (
        int(
            model_row[
                "Predicted_Class"
            ]
        ),

        str(
            model_row[
                "Predicted_Label"
            ]
        ),

        float(
            model_row[
                "Probability_No_Heart_Disease"
            ]
        ),

        float(
            model_row[
                "Probability_Heart_Disease"
            ]
        )
    )

def save_single_prediction_to_history(
    patient_input,
    selected_prediction_result,
    comparison_results,
    case_id=None
):
    """
    Save one successful single-patient prediction
    into the local history database.
    """

    # --------------------------------------------------------
    # Current Malaysia date and time
    # --------------------------------------------------------

    malaysia_time = datetime.now(
        ZoneInfo(
            "Asia/Kuala_Lumpur"
        )
    )

    # --------------------------------------------------------
    # Generate unique Prediction ID
    # --------------------------------------------------------

    prediction_id = (
        "HD-"
        + malaysia_time.strftime(
            "%Y%m%d-%H%M%S"
        )
        + "-"
        + uuid.uuid4().hex[
            :4
        ].upper()
    )

    # Example:
    # HD-20260820-205300-A72F

    # --------------------------------------------------------
    # Patient input
    # --------------------------------------------------------

    patient_row = (
        patient_input.iloc[0]
    )

    # --------------------------------------------------------
    # Extract four-model results
    # --------------------------------------------------------

    ann_values = (
        get_model_history_values(
            comparison_results,
            "ANN"
        )
    )

    svm_values = (
        get_model_history_values(
            comparison_results,
            "SVM"
        )
    )

    random_forest_values = (
        get_model_history_values(
            comparison_results,
            "Random Forest"
        )
    )

    xgboost_values = (
        get_model_history_values(
            comparison_results,
            "XGBoost"
        )
    )

    # --------------------------------------------------------
    # Model agreement
    # --------------------------------------------------------

    presence_votes = int(
        (
            comparison_results[
                "Predicted_Class"
            ] == 1
        ).sum()
    )

    total_models = len(
        comparison_results
    )

    absence_votes = (
        total_models
        - presence_votes
    )

    if presence_votes == total_models:

        model_agreement = (
            "All Models Predict Presence"
        )

    elif absence_votes == total_models:

        model_agreement = (
            "All Models Predict Absence"
        )

    elif presence_votes == absence_votes:

        model_agreement = (
            "Mixed Predictions"
        )

    elif presence_votes > absence_votes:

        model_agreement = (
            "Majority Predict Presence"
        )

    else:

        model_agreement = (
            "Majority Predict Absence"
        )

    # --------------------------------------------------------
    # Prepare record
    # --------------------------------------------------------

    history_record = (

        prediction_id,

        malaysia_time.isoformat(
            timespec="seconds"
        ),

        malaysia_time.strftime(
            "%Y-%m-%d"
        ),

        malaysia_time.strftime(
            "%H:%M:%S"
        ),

        (
            str(case_id).strip()
            if (
                case_id is not None
                and str(case_id).strip()
            )
            else None
        ),

        # Patient inputs

        int(
            patient_row["Age"]
        ),

        str(
            patient_row["Sex"]
        ),

        str(
            patient_row[
                "ChestPainType"
            ]
        ),

        float(
            patient_row[
                "RestingBP"
            ]
        ),

        float(
            patient_row[
                "Cholesterol"
            ]
        ),

        int(
            patient_row[
                "FastingBS"
            ]
        ),

        str(
            patient_row[
                "RestingECG"
            ]
        ),

        float(
            patient_row[
                "MaxHR"
            ]
        ),

        str(
            patient_row[
                "ExerciseAngina"
            ]
        ),

        float(
            patient_row[
                "Oldpeak"
            ]
        ),

        str(
            patient_row[
                "ST_Slope"
            ]
        ),

        # Selected final model

        str(
            selected_prediction_result[
                "model"
            ]
        ),

        int(
            selected_prediction_result[
                "predicted_class"
            ]
        ),

        str(
            selected_prediction_result[
                "predicted_label"
            ]
        ),

        float(
            selected_prediction_result[
                "probability_no_heart_disease"
            ]
        ) * 100,

        float(
            selected_prediction_result[
                "probability_heart_disease"
            ]
        ) * 100,

        # ANN
        *ann_values,

        # SVM
        *svm_values,

        # Random Forest
        *random_forest_values,

        # XGBoost
        *xgboost_values,

        # Agreement

        presence_votes,

        absence_votes,

        model_agreement
    )

    # --------------------------------------------------------
    # Save to SQLite
    # --------------------------------------------------------

    with sqlite3.connect(
        PREDICTION_HISTORY_DB_PATH
    ) as connection:

        connection.execute(
            """
            INSERT INTO prediction_history (

                prediction_id,

                prediction_datetime,
                prediction_date,
                prediction_time,

                case_id,

                age,
                sex,
                chest_pain_type,
                resting_bp,
                cholesterol,
                fasting_bs,
                resting_ecg,
                max_hr,
                exercise_angina,
                oldpeak,
                st_slope,

                selected_model,

                final_predicted_class,
                final_prediction,

                probability_no_heart_disease,
                probability_heart_disease,

                ann_predicted_class,
                ann_prediction,
                ann_probability_no,
                ann_probability_yes,

                svm_predicted_class,
                svm_prediction,
                svm_probability_no,
                svm_probability_yes,

                random_forest_predicted_class,
                random_forest_prediction,
                random_forest_probability_no,
                random_forest_probability_yes,

                xgboost_predicted_class,
                xgboost_prediction,
                xgboost_probability_no,
                xgboost_probability_yes,

                presence_votes,
                absence_votes,

                model_agreement
            )

            VALUES (

                ?, ?, ?, ?, ?,

                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,

                ?, ?, ?, ?, ?,

                ?, ?, ?, ?,

                ?, ?, ?, ?,

                ?, ?, ?, ?,

                ?, ?, ?, ?,

                ?, ?, ?
            )
            """,

            history_record
        )

        connection.commit()

    return prediction_id


# ============================================================
# SAVE BATCH PREDICTIONS TO HISTORY
# ============================================================

def save_batch_predictions_to_history(
    batch_results
):
    """
    Save every successful batch-prediction record
    into the prediction-history database.
    """

    saved_prediction_ids = []

    for _, batch_row in (
        batch_results.iterrows()
    ):

        # ----------------------------------------------------
        # Rebuild original patient input
        # ----------------------------------------------------

        patient_input = pd.DataFrame(
            [
                {
                    feature:
                        batch_row[feature]

                    for feature in required_features
                }
            ]
        )

        # ----------------------------------------------------
        # Build four-model comparison result
        # ----------------------------------------------------

        comparison_records = []

        for model_name in models.keys():

            comparison_records.append(
                {
                    "Model":
                        model_name,

                    "Predicted_Class":
                        int(
                            batch_row[
                                f"{model_name} Predicted Class"
                            ]
                        ),

                    "Predicted_Label":
                        str(
                            batch_row[
                                f"{model_name} Prediction"
                            ]
                        ),

                    "Probability_No_Heart_Disease":
                        float(
                            batch_row[
                                f"{model_name} "
                                "Absence Probability (%)"
                            ]
                        ),

                    "Probability_Heart_Disease":
                        float(
                            batch_row[
                                f"{model_name} "
                                "Presence Probability (%)"
                            ]
                        )
                }
            )

        comparison_results = pd.DataFrame(
            comparison_records
        )

        # ----------------------------------------------------
        # Selected final model result
        # ----------------------------------------------------

        selected_prediction_result = {
            "model":
                preferred_model_name,

            "predicted_class":
                int(
                    batch_row[
                        "Predicted_Class"
                    ]
                ),

            "predicted_label":
                str(
                    batch_row[
                        "Predicted_Label"
                    ]
                ),

            # save_single_prediction_to_history()
            # expects these two values as 0–1,
            # so convert the batch percentages back.
            "probability_no_heart_disease":
                float(
                    batch_row[
                        "Probability_No_Heart_Disease"
                    ]
                ) / 100,

            "probability_heart_disease":
                float(
                    batch_row[
                        "Probability_Heart_Disease"
                    ]
                ) / 100
        }

        # ----------------------------------------------------
        # Generate Case ID for this batch patient
        # ----------------------------------------------------

        malaysia_time = datetime.now(
            ZoneInfo(
                "Asia/Kuala_Lumpur"
            )
        )

        batch_case_id = (
            "CASE-"
            + malaysia_time.strftime(
                "%Y%m%d"
            )
            + "-"
            + uuid.uuid4().hex[
                :4
            ].upper()
        )

        # ----------------------------------------------------
        # Save using existing history function
        # ----------------------------------------------------

        prediction_id = (
            save_single_prediction_to_history(
                patient_input=patient_input,
                selected_prediction_result=(
                    selected_prediction_result
                ),
                comparison_results=(
                    comparison_results
                ),
                case_id=batch_case_id
            )
        )

        saved_prediction_ids.append(
            prediction_id
        )

    return saved_prediction_ids
initialize_prediction_history_database()

def delete_predictions_from_history(
    prediction_ids
):
    """
    Permanently delete selected prediction records
    from the prediction-history database.
    """

    prediction_ids = [
        str(prediction_id).strip()
        for prediction_id in prediction_ids
        if str(prediction_id).strip()
    ]

    if not prediction_ids:
        return 0

    placeholders = ",".join(
        "?"
        for _ in prediction_ids
    )

    with sqlite3.connect(
        PREDICTION_HISTORY_DB_PATH
    ) as connection:

        cursor = connection.execute(
            f"""
            DELETE FROM prediction_history
            WHERE prediction_id IN ({placeholders})
            """,
            prediction_ids
        )

        connection.commit()

        deleted_count = cursor.rowcount

    return deleted_count
def load_prediction_history():
    """
    Load all saved single-patient prediction records
    from the local SQLite database.
    """

    initialize_prediction_history_database()

    try:

        with sqlite3.connect(
            PREDICTION_HISTORY_DB_PATH
        ) as connection:

            history_data = pd.read_sql_query(
                """
                SELECT *
                FROM prediction_history
                ORDER BY prediction_datetime DESC
                """,
                connection
            )

        return history_data

    except Exception as error:

        st.error(
            "Prediction history could not be loaded."
        )

        with st.expander(
            "Show history-loading error"
        ):

            st.code(
                str(error)
            )

        return pd.DataFrame()
# ===========================================================
# INPUT VALIDATION
# ============================================================

def validate_patient_input(patient_data):

    errors = []

    if not isinstance(patient_data, pd.DataFrame):

        return None, [
            "Input must be provided as a table."
        ]

    if patient_data.empty:

        return None, [
            "No patient records were provided."
        ]

    cleaned_data = patient_data.copy()

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in required_features
        if column not in cleaned_data.columns
    ]

    if missing_columns:

        return None, [
            "Missing required column(s): "
            + ", ".join(missing_columns)
        ]

    # --------------------------------------------------------
    # Check unexpected columns
    # --------------------------------------------------------

    unexpected_columns = [
        column
        for column in cleaned_data.columns
        if column not in required_features
    ]

    if unexpected_columns:

        return None, [
            "Unexpected column(s): "
            + ", ".join(unexpected_columns)
        ]

    cleaned_data = cleaned_data[
        required_features
    ].copy()

    # --------------------------------------------------------
    # Numerical input validation
    # --------------------------------------------------------

    numeric_features = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak"
    ]

    for feature in numeric_features:

        original_values = (
            cleaned_data[feature].copy()
        )

        cleaned_data[feature] = pd.to_numeric(
            cleaned_data[feature],
            errors="coerce"
        )

        invalid_numeric = (
            original_values.notna()
            &
            cleaned_data[feature].isna()
        )

        if invalid_numeric.any():

            errors.append(
                f"{feature} contains a non-numeric value."
            )
    required_numeric_features = [
        "Age",
        "MaxHR",
        "Oldpeak"
    ]
    
    for feature in required_numeric_features:
    
        if cleaned_data[feature].isna().any():
    
            errors.append(
                f"{feature} cannot be empty."
            )
    # --------------------------------------------------------
    # FastingBS validation
    # --------------------------------------------------------

    original_fasting = (
        cleaned_data["FastingBS"].copy()
    )

    cleaned_data["FastingBS"] = pd.to_numeric(
        cleaned_data["FastingBS"],
        errors="coerce"
    )

    invalid_fasting = (
        original_fasting.notna()
        &
        cleaned_data["FastingBS"].isna()
    )
    
    if invalid_fasting.any():
    
        errors.append(
            "FastingBS must contain a numeric value."
        )
    
    elif cleaned_data["FastingBS"].isna().any():
    
        errors.append(
            "FastingBS cannot be empty."
        )
    
    elif not cleaned_data[
        "FastingBS"
    ].isin([0, 1]).all():
    
        errors.append(
            "FastingBS must be either 0 or 1."
        )

    # --------------------------------------------------------
    # Categorical validation
    # --------------------------------------------------------

    categorical_features_to_check = [
        "Sex",
        "ChestPainType",
        "RestingECG",
        "ExerciseAngina",
        "ST_Slope"
    ]

    for feature in categorical_features_to_check:

        valid_values = (
            categorical_valid_values[feature]
        )

        missing_mask = (
            cleaned_data[feature].isna()
            |
            cleaned_data[feature]
            .astype(str)
            .str.strip()
            .eq("")
        )
        
        if missing_mask.any():
        
            errors.append(
                f"{feature} cannot be empty."
            )
        
        invalid_mask = (
            ~missing_mask
            &
            ~cleaned_data[feature].isin(
                valid_values
            )
        )

        if invalid_mask.any():

            invalid_values = (
                cleaned_data.loc[
                    invalid_mask,
                    feature
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            errors.append(
                f"{feature} contains unsupported value(s): "
                + ", ".join(invalid_values)
            )

    # --------------------------------------------------------
    # Handle invalid zero values as missing
    # --------------------------------------------------------

    cleaned_data.loc[
        cleaned_data["RestingBP"] == 0,
        "RestingBP"
    ] = np.nan

    cleaned_data.loc[
        cleaned_data["Cholesterol"] == 0,
        "Cholesterol"
    ] = np.nan

    # --------------------------------------------------------
    # Numerical range validation
    # --------------------------------------------------------

    for feature, limits in (
        numerical_validation_ranges.items()
    ):

        values = (
            cleaned_data[feature]
            .dropna()
        )

        if (
            (values < limits["min"]).any()
            or
            (values > limits["max"]).any()
        ):

            errors.append(
                f"{feature} must be between "
                f"{limits['min']} and "
                f"{limits['max']}."
            )

    # --------------------------------------------------------
    # Age must be whole number
    # --------------------------------------------------------

    age_values = (
        cleaned_data["Age"]
        .dropna()
    )

    if not age_values.apply(
        lambda value: float(value).is_integer()
    ).all():

        errors.append(
            "Age must be entered as a whole number."
        )

    if errors:

        return None, errors

    cleaned_data["Age"] = (
        cleaned_data["Age"].astype(int)
    )

    cleaned_data["FastingBS"] = (
        cleaned_data["FastingBS"].astype(int)
    )

    return cleaned_data, []


# ============================================================
# SINGLE PATIENT PREDICTION
# ============================================================

def predict_patient(patient_data):

    cleaned_data, errors = validate_patient_input(
        patient_data
    )

    if errors:

        return {
            "success": False,
            "errors": errors
        }

    try:

        model = models[
            preferred_model_name
        ]

        prediction = int(
            model.predict(
                cleaned_data
            )[0]
        )

        probabilities = (
            model.predict_proba(
                cleaned_data
            )[0]
        )
        model_classes = list(
            getattr(
                model,
                "classes_",
                [0, 1]
            )
        )
        
        absence_index = (
            model_classes.index(0)
        )
        
        presence_index = (
            model_classes.index(1)
        )

        return {
            "success": True,
            "model": preferred_model_name,
            "predicted_class": prediction,
            "predicted_label": class_labels[
                prediction
            ],
            "probability_no_heart_disease": float(
                probabilities[
                    absence_index
                ]
            ),
            
            "probability_heart_disease": float(
                probabilities[
                    presence_index
                ]
            )
        }

    except Exception as error:

        return {
            "success": False,
            "errors": [
                "Prediction could not be completed: "
                + str(error)
            ]
        }

# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_batch(batch_data):
    """
    Validate uploaded records and generate predictions with all
    available models. The selected final model is also preserved
    in compatibility columns for the rest of the application.
    """

    cleaned_data, errors = validate_patient_input(
        batch_data
    )
    
    if errors:
        return {
            "success": False,
            "errors": errors
        }
    
    try:
    
        # ----------------------------------------------------
        # Preserve original uploaded values for export
        # ----------------------------------------------------
    
        original_data = (
            batch_data[
                required_features
            ]
            .copy()
            .reset_index(drop=True)
        )
    
        # Start output with the original imported patient data
        results = original_data.copy()
    
        model_prediction_arrays = {}

        for model_name, model in models.items():

            predictions = np.asarray(
                model.predict(cleaned_data),
                dtype=int
            )

            probabilities = np.asarray(
                model.predict_proba(cleaned_data),
                dtype=float
            )

            model_classes = list(
                getattr(model, "classes_", [0, 1])
            )

            absence_index = model_classes.index(0)
            presence_index = model_classes.index(1)

            absence_probabilities = (
                probabilities[:, absence_index]
            )

            presence_probabilities = (
                probabilities[:, presence_index]
            )

            model_prediction_arrays[
                model_name
            ] = predictions

            results[
                f"{model_name} Predicted Class"
            ] = predictions

            results[
                f"{model_name} Prediction"
            ] = [
                class_labels[int(value)]
                for value in predictions
            ]

            results[
                f"{model_name} Absence Probability (%)"
            ] = (
                absence_probabilities * 100
            ).round(2)

            results[
                f"{model_name} Presence Probability (%)"
            ] = (
                presence_probabilities * 100
            ).round(2)

        # ----------------------------------------------------
        # Model agreement
        # ----------------------------------------------------

        model_names = list(models.keys())

        prediction_matrix = np.column_stack([
            model_prediction_arrays[model_name]
            for model_name in model_names
        ])

        presence_votes = (
            prediction_matrix == 1
        ).sum(axis=1)

        absence_votes = (
            len(model_names) - presence_votes
        )

        results["Presence Votes"] = presence_votes
        results["Absence Votes"] = absence_votes

        agreement_status = []

        for present_votes, absent_votes in zip(
            presence_votes,
            absence_votes
        ):
            if present_votes == len(model_names):
                status = "All Models Predict Presence"
            elif absent_votes == len(model_names):
                status = "All Models Predict Absence"
            elif present_votes == absent_votes:
                status = "Mixed Predictions"
            elif present_votes > absent_votes:
                status = "Majority Predict Presence"
            else:
                status = "Majority Predict Absence"

            agreement_status.append(status)

        results["Model Agreement"] = agreement_status
        results["Selected Final Model"] = preferred_model_name

        # ----------------------------------------------------
        # Compatibility columns for selected final model
        # ----------------------------------------------------

        selected_model = models[
            preferred_model_name
        ]

        selected_predictions = np.asarray(
            selected_model.predict(cleaned_data),
            dtype=int
        )

        selected_probabilities = np.asarray(
            selected_model.predict_proba(cleaned_data),
            dtype=float
        )

        selected_classes = list(
            getattr(selected_model, "classes_", [0, 1])
        )

        selected_absence_index = (
            selected_classes.index(0)
        )

        selected_presence_index = (
            selected_classes.index(1)
        )

        results["Predicted_Class"] = (
            selected_predictions
        )

        results["Predicted_Label"] = [
            class_labels[int(value)]
            for value in selected_predictions
        ]

        results[
            "Probability_No_Heart_Disease"
        ] = (
            selected_probabilities[
                :,
                selected_absence_index
            ] * 100
        ).round(2)

        results[
            "Probability_Heart_Disease"
        ] = (
            selected_probabilities[
                :,
                selected_presence_index
            ] * 100
        ).round(2)

        return {
            "success": True,
            "model": preferred_model_name,
            "results": results
        }

    except Exception as error:

        return {
            "success": False,
            "errors": [
                "Batch prediction could not be completed: "
                + str(error)
            ]
        }


# ============================================================
# BATCH FILE LOADING
# ============================================================

def load_batch_file(uploaded_file):

    try:

        filename = uploaded_file.name.lower()

        if filename.endswith(".xlsx"):

            data = pd.read_excel(
                uploaded_file
            )

        else:

            return {
                "success": False,
                "errors": [
                    "Unsupported file format. "
                    "Please upload an XLSX file."
                ]
            }

        # ----------------------------------------------------
        # Check empty file
        # ----------------------------------------------------

        if data.empty:

            return {
                "success": False,
                "errors": [
                    "The uploaded file contains no patient records."
                ]
            }

        # ----------------------------------------------------
        # Check required columns immediately after upload
        # ----------------------------------------------------

        missing_columns = [
            column
            for column in required_features
            if column not in data.columns
        ]

        unexpected_columns = [
            column
            for column in data.columns
            if column not in required_features
        ]

        column_errors = []

        if missing_columns:

            column_errors.append(
                "Missing required column(s): "
                + ", ".join(missing_columns)
            )

        if unexpected_columns:

            column_errors.append(
                "Unexpected column(s): "
                + ", ".join(unexpected_columns)
            )

        if column_errors:

            return {
                "success": False,
                "errors": column_errors
            }

        # ----------------------------------------------------
        # File is correctly structured
        # ----------------------------------------------------

        return {
            "success": True,
            "data": data
        }

    except Exception as error:

        return {
            "success": False,
            "errors": [
                "The uploaded file could not be read: "
                + str(error)
            ]
        }
# ============================================================
# FOUR-MODEL COMPARISON
# ============================================================

def compare_models(patient_data):

    cleaned_data, errors = validate_patient_input(
        patient_data
    )

    if errors:

        return {
            "success": False,
            "errors": errors
        }

    try:

        comparison_records = []

        for model_name, model in models.items():

            prediction = int(
                model.predict(
                    cleaned_data
                )[0]
            )

            probabilities = (
                model.predict_proba(
                    cleaned_data
                )[0]
            )
            model_classes = list(
                getattr(
                    model,
                    "classes_",
                    [0, 1]
                )
            )
            
            absence_index = (
                model_classes.index(0)
            )
            
            presence_index = (
                model_classes.index(1)
            )

            comparison_records.append({
                "Model": model_name,
                "Predicted_Class": prediction,
                "Predicted_Label": class_labels[
                    prediction
                ],
                "Probability_No_Heart_Disease": round(
                    float(
                        probabilities[
                            absence_index
                        ]
                    ) * 100,
                    2
                ),
                "Probability_Heart_Disease": round(
                    float(
                        probabilities[
                            presence_index
                        ]
                    ) * 100,
                    2
                )
            })

        comparison_results = pd.DataFrame(
            comparison_records
        )

        return {
            "success": True,
            "results": comparison_results
        }

    except Exception as error:

        return {
            "success": False,
            "errors": [
                "Model comparison could not be completed: "
                + str(error)
            ]
        }



# ============================================================
# UI HELPERS
# ============================================================

MODEL_DISPLAY_NAMES = {
    "ANN": "Artificial Neural Network",
    "SVM": "Support Vector Machine",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost"
}


def find_metric_column(candidates):
    """Return the first matching final-result column."""
    return next(
        (
            column
            for column in candidates
            if column in final_test_results.columns
        ),
        None
    )


F1_COLUMN = find_metric_column(
    ["F1-Score", "F1 Score", "F1_Score", "F1"]
)

ROC_AUC_COLUMN = find_metric_column(
    ["AUC", "ROC-AUC", "ROC AUC", "ROC_AUC", "ROC-Auc", "roc_auc"]
)


def display_input_field_glossary():
    """Display a compact guide to the 11 model inputs."""

    glossary_data = pd.DataFrame([
        {
            "Input": "Age",
            "Meaning": "Patient age in years.",
            "Accepted Values": "28–77"
        },
        {
            "Input": "Sex",
            "Meaning": "Recorded sex.",
            "Accepted Values": "M = Male; F = Female"
        },
        {
            "Input": "ChestPainType",
            "Meaning": "Chest-pain presentation category.",
            "Accepted Values": "ASY, ATA, NAP, TA"
        },
        {
            "Input": "RestingBP",
            "Meaning": "Resting blood pressure in mm Hg.",
            "Accepted Values": "0 or model-supported range"
        },
        {
            "Input": "Cholesterol",
            "Meaning": "Serum cholesterol in mg/dL.",
            "Accepted Values": "0 or model-supported range"
        },
        {
            "Input": "FastingBS",
            "Meaning": "Whether fasting blood sugar exceeds 120 mg/dL.",
            "Accepted Values": "0 = No; 1 = Yes"
        },
        {
            "Input": "RestingECG",
            "Meaning": "Resting electrocardiogram result.",
            "Accepted Values": "Normal, ST, LVH"
        },
        {
            "Input": "MaxHR",
            "Meaning": "Maximum heart rate achieved.",
            "Accepted Values": "60–202"
        },
        {
            "Input": "ExerciseAngina",
            "Meaning": "Whether exercise-induced angina was recorded.",
            "Accepted Values": "N = No; Y = Yes"
        },
        {
            "Input": "Oldpeak",
            "Meaning": "ST depression induced by exercise relative to rest.",
            "Accepted Values": "-2.6–6.2"
        },
        {
            "Input": "ST_Slope",
            "Meaning": "Slope of the peak exercise ST segment.",
            "Accepted Values": "Flat, Up, Down"
        }
    ])

    with st.expander("📖 Input Field Guide"):
        st.caption(
            "The accepted ranges reflect values represented in the "
            "project dataset and are not universal clinical reference ranges."
        )
        st.dataframe(
            glossary_data,
            hide_index=True,
            width="stretch"
        )
        st.info(
            "Entering 0 for RestingBP or Cholesterol represents an "
            "unavailable value. The saved preprocessing pipeline "
            "handles these values using the training-data imputer."
        )


def build_excel_template():
    """Create an empty Excel batch template in memory."""
    template = pd.DataFrame(columns=required_features)
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        template.to_excel(
            writer,
            sheet_name="Patient Input",
            index=False
        )

    return buffer.getvalue()


def style_prediction_label(predicted_class, predicted_label):
    """Display a prediction result using consistent status styling."""
    if int(predicted_class) == 1:
        st.warning(f"**{predicted_label}**")
    else:
        st.success(f"**{predicted_label}**")

# ============================================================
# MODEL-SPECIFIC DETAILED ANALYSIS DIALOG
# ============================================================

@st.dialog(
    "Delete Prediction Records",
    width="small"
)
def show_delete_history_dialog(
    prediction_ids
):

    prediction_ids = list(
        prediction_ids
    )

    selected_count = len(
        prediction_ids
    )

    if selected_count <= 0:
        st.info(
            "No prediction records are selected."
        )
        return

    record_word = (
        "record"
        if selected_count == 1
        else "records"
    )

    st.warning(
        (
            f"You are about to permanently delete "
            f"**{selected_count} prediction {record_word}**."
        )
    )

    st.write(
        "This action cannot be undone."
    )

    # --------------------------------------------------------
    # Optional selected-record preview
    # --------------------------------------------------------

    with st.expander(
        "View selected Prediction IDs"
    ):

        for prediction_id in prediction_ids:

            st.code(
                prediction_id
            )

    cancel_col, delete_col = (
        st.columns(
            2,
            gap="small"
        )
    )

    with cancel_col:

        if st.button(
            "Cancel",
            width="stretch",
            key="cancel_history_deletion"
        ):

            st.rerun()

    with delete_col:

        if st.button(
            (
                f"Delete {selected_count} "
                f"{'Record' if selected_count == 1 else 'Records'}"
            ),
            type="primary",
            width="stretch",
            icon=":material/delete:",
            key="confirm_history_deletion"
        ):

            try:

                deleted_count = (
                    delete_predictions_from_history(
                        prediction_ids
                    )
                )

                # Clear selected records
                st.session_state[
                    "selected_history_prediction_ids"
                ] = []

                # Return to first page after deletion
                st.session_state[
                    "history_page"
                ] = 1

                # Save confirmation message for next rerun
                st.session_state[
                    "history_delete_message"
                ] = (
                    f"{deleted_count} prediction "
                    f"{'record' if deleted_count == 1 else 'records'} "
                    "deleted successfully."
                )

                st.rerun()

            except Exception as error:

                st.error(
                    "The selected prediction records "
                    "could not be deleted."
                )

                with st.expander(
                    "Show deletion error"
                ):

                    st.code(
                        str(error)
                    )
                    
@st.dialog(
    "Model Detailed Analysis",
    width="large",
)
def show_model_detail_dialog(model_name):

    comparison_output = st.session_state.get(
        "single_comparison_result"
    )

    if comparison_output is None:

        st.error(
            "Generate a single-patient prediction first."
        )

        return

    model_rows = comparison_output[
        comparison_output["Model"] == model_name
    ]

    if model_rows.empty:

        st.error(
            f"No prediction result is available for {model_name}."
        )

        return

    model_row = model_rows.iloc[0]

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------
    
    display_model_name = MODEL_DISPLAY_NAMES.get(
        model_name,
        model_name
    )
    
    st.markdown(
        f'<div class="model-dialog-name">'
        f'{display_model_name}'
        f'</div>',
        unsafe_allow_html=True
    )





    # --------------------------------------------------------
    # TABS
    # --------------------------------------------------------

    overview_tab, confusion_tab, roc_tab, features_tab = (
        st.tabs(
            [
                "Overview",
                "Confusion Matrix",
                "ROC Curve",
                "Selected Features"
            ]
        )
    )

    # ========================================================
    # TAB 1 — OVERVIEW
    # ========================================================
    
    with overview_tab:



        st.markdown(
            "### Current Patient Prediction"
        )
    
        presence_probability = float(
            model_row[
                "Probability_Heart_Disease"
            ]
        )
    
        predicted_class = int(
            model_row[
                "Predicted_Class"
            ]
        )
    
        prediction_label = (
            "Heart Disease Present"
            if predicted_class == 1
            else "Heart Disease Absent"
        )
    
    
        prediction_col, probability_col = (
            st.columns(2)
        )
    
        with prediction_col:
    
            st.metric(
                "Prediction",
                prediction_label
            )
    
        with probability_col:
    
            st.metric(
                "Heart-Disease Probability",
                f"{presence_probability:.2f}%"
            )
    
    
        # ----------------------------------------------------
        # SIMPLE PROBABILITY INDICATOR
        # ----------------------------------------------------
    
        st.progress(
            presence_probability / 100
        )
    
        if predicted_class == 1:
    
            st.caption(
                "The model assigned a probability above the "
                "classification threshold to heart-disease presence."
            )
    
        else:
    
            st.caption(
                "The model assigned a probability below the "
                "classification threshold to heart-disease presence."
            )
    
    
        st.divider()
    
    
        # ----------------------------------------------------
        # KEY TEST-SET EVIDENCE
        # ----------------------------------------------------
    
        st.markdown(
            "### Key Model Evidence"
        )
    
        test_rows = final_test_results[
            final_test_results[
                "Model"
            ] == model_name
        ]
    
    
        if test_rows.empty:
    
            st.info(
                "Final test-set results are unavailable."
            )
    
        else:
    
            test_row = test_rows.iloc[0]
    
            evidence_col1, evidence_col2, evidence_col3 = (
                st.columns(3)
            )
    
    
            # F1
            if (
                F1_COLUMN is not None
                and F1_COLUMN in test_row.index
            ):
    
                with evidence_col1:
    
                    st.metric(
                        "F1-Score",
                        (
                            f"{float(test_row[F1_COLUMN]) * 100:.2f}%"
                        )
                    )
    
    
            # Recall
            if "Recall" in test_row.index:
    
                with evidence_col2:
    
                    st.metric(
                        "Recall",
                        (
                            f"{float(test_row['Recall']) * 100:.2f}%"
                        )
                    )
    
    
            # ROC-AUC
            if (
                ROC_AUC_COLUMN is not None
                and ROC_AUC_COLUMN in test_row.index
            ):
    
                with evidence_col3:
    
                    st.metric(
                        "ROC-AUC",
                        (
                            f"{float(test_row[ROC_AUC_COLUMN]) * 100:.2f}%"
                        )
                    )
    
    

    
    

    
        # ========================================================
        # TAB 2 — CONFUSION MATRIX
        # ========================================================
    
    with confusion_tab:
        

        
            st.markdown(
                f"### Confusion Matrix — "
                f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)}"
            )

        
            st.caption(
                "Performance on the untouched final test set."
            )
        
            test_rows = final_test_results[
                final_test_results["Model"] == model_name
            ]
        
            if test_rows.empty:
        
                st.info(
                    "Final test results are unavailable."
                )
        
            else:
        
                test_row = test_rows.iloc[0]
        
                required_columns = {
                    "TN",
                    "FP",
                    "FN",
                    "TP"
                }
        
                if required_columns.issubset(
                    test_row.index
                ):
        
                    tn = int(test_row["TN"])
                    fp = int(test_row["FP"])
                    fn = int(test_row["FN"])
                    tp = int(test_row["TP"])
        
                    # ---------------------------------------------
                    # CONFUSION MATRIX DATA
                    # ---------------------------------------------
        
                    confusion_data = pd.DataFrame(
                        [
                            {
                                "Actual": "Absent (0)",
                                "Predicted": "Absent (0)",
                                "Count": tn,
                                "Outcome": "True Negative"
                            },
                            {
                                "Actual": "Absent (0)",
                                "Predicted": "Present (1)",
                                "Count": fp,
                                "Outcome": "False Positive"
                            },
                            {
                                "Actual": "Present (1)",
                                "Predicted": "Absent (0)",
                                "Count": fn,
                                "Outcome": "False Negative"
                            },
                            {
                                "Actual": "Present (1)",
                                "Predicted": "Present (1)",
                                "Count": tp,
                                "Outcome": "True Positive"
                            }
                        ]
                    )
        
                    # ---------------------------------------------
                    # COMPACT HEATMAP
                    # ---------------------------------------------
        
                    confusion_heatmap = (
                        alt.Chart(
                            confusion_data
                        )
                        .mark_rect(
                            cornerRadius=6
                        )
                        .encode(
                            x=alt.X(
                                "Predicted:N",
                                title="Predicted Class",
                                sort=[
                                    "Absent (0)",
                                    "Present (1)"
                                ],
                                axis=alt.Axis(
                                    labelAngle=0,
                                    labelFontSize=12,
                                    titleFontSize=12
                                )
                            ),
                            y=alt.Y(
                                "Actual:N",
                                title="Actual Class",
                                sort=[
                                    "Absent (0)",
                                    "Present (1)"
                                ],
                                axis=alt.Axis(
                                    labelFontSize=12,
                                    titleFontSize=12
                                )
                            ),
                            color=alt.Color(
                                "Count:Q",
                                scale=alt.Scale(
                                    scheme="blues"
                                ),
                                legend=None
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "Outcome:N",
                                    title="Outcome"
                                ),
                                alt.Tooltip(
                                    "Count:Q",
                                    title="Count"
                                )
                            ]
                        )
                        .properties(
                            width=600,
                            height=350
                        )
                    )
        
                    confusion_labels = (
                        alt.Chart(
                            confusion_data
                        )
                        .mark_text(
                            fontSize=19,
                            fontWeight="bold",
                            color="white"
                        )
                        .encode(
                            x=alt.X(
                                "Predicted:N",
                                sort=[
                                    "Absent (0)",
                                    "Present (1)"
                                ]
                            ),
                            y=alt.Y(
                                "Actual:N",
                                sort=[
                                    "Absent (0)",
                                    "Present (1)"
                                ]
                            ),
                            text="Count:Q"
                        )
                    )
        
                    chart_col1, chart_col2, chart_col3 = st.columns([1, 3, 1])
                    
                    with chart_col2:
                        matrix_col_left, matrix_col_center, matrix_col_right = st.columns(
                            [1, 2.2, 1]
                        )
                        
                        with matrix_col_center:
                            st.altair_chart(
                                (
                                    confusion_heatmap
                                    + confusion_labels
                                ).configure_view(
                                    stroke=None
                                ),
                                width="stretch"
                            )        
                    # ---------------------------------------------
                    # IMPORTANT ERRORS ONLY
                    # ---------------------------------------------
        
                    st.markdown(
                        "#### Classification Errors"
                    )
        
                    error_col1, error_col2 = st.columns(
                        2,
                        gap="small"
                    )
        
                    with error_col1:
        
                        with st.container(
                            border=True
                        ):
        
                            st.metric(
                                "False Negatives",
                                fn
                            )
        
                            st.caption(
                                "Missed heart-disease cases"
                            )
        
                    with error_col2:
        
                        with st.container(
                            border=True
                        ):
        
                            st.metric(
                                "False Positives",
                                fp
                            )
        
                            st.caption(
                                "False heart-disease alerts"
                            )
        
                    # ---------------------------------------------
                    # ONE COMPACT INTERPRETATION
                    # ---------------------------------------------
        

        
                else:
        
                    st.info(
                        "TN, FP, FN and TP values are unavailable."
                    )
        # ========================================================
        # TAB 3 — ROC CURVE
        # ========================================================
    
    with roc_tab:
        

            st.markdown(
                f"### ROC Curve — "
                f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)}"
            )

    
            st.caption(
                "Discrimination performance on the untouched final test set."
            )
    
            if model_name not in roc_curve_data:
    
                st.warning(
                    "ROC curve data is not available in the current "
                    "model bundle."
                )
    
            else:
    
                model_roc = roc_curve_data[
                    model_name
                ]
    
                roc_dataframe = pd.DataFrame(
                    {
                        "False Positive Rate": (
                            model_roc["fpr"]
                        ),
                        "True Positive Rate": (
                            model_roc["tpr"]
                        )
                    }
                )
    
                # ----------------------------------------------------
                # ROC CURVE
                # ----------------------------------------------------
    
                roc_line = (
                    alt.Chart(
                        roc_dataframe
                    )
                    .mark_line(
                        strokeWidth=3
                    )
                    .encode(
                        x=alt.X(
                            "False Positive Rate:Q",
                            title="False Positive Rate",
                            scale=alt.Scale(
                                domain=[0, 1]
                            ),
                            axis=alt.Axis(
                                format=".1f",
                                labelFontSize=11,
                                titleFontSize=12
                            )
                        ),
                        y=alt.Y(
                            "True Positive Rate:Q",
                            title="True Positive Rate",
                            scale=alt.Scale(
                                domain=[0, 1]
                            ),
                            axis=alt.Axis(
                                format=".1f",
                                labelFontSize=11,
                                titleFontSize=12
                            )
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "False Positive Rate:Q",
                                title="FPR",
                                format=".3f"
                            ),
                            alt.Tooltip(
                                "True Positive Rate:Q",
                                title="TPR",
                                format=".3f"
                            )
                        ]
                    )
                )
    
                random_line_data = pd.DataFrame(
                    {
                        "False Positive Rate": [
                            0.0,
                            1.0
                        ],
                        "True Positive Rate": [
                            0.0,
                            1.0
                        ]
                    }
                )
    
                random_line = (
                    alt.Chart(
                        random_line_data
                    )
                    .mark_line(
                        strokeDash=[6, 6],
                        opacity=0.65
                    )
                    .encode(
                        x="False Positive Rate:Q",
                        y="True Positive Rate:Q"
                    )
                )
    
                roc_chart = (
                    roc_line
                    + random_line
                ).properties(
                    height=330
                ).configure_view(
                    stroke=None
                )
    
                # ----------------------------------------------------
                # GET AUC VALUE
                # ----------------------------------------------------
    
                test_rows = final_test_results[
                    final_test_results[
                        "Model"
                    ] == model_name
                ]
    
                auc_value = None
    
                if (
                    not test_rows.empty
                    and ROC_AUC_COLUMN is not None
                    and ROC_AUC_COLUMN
                    in test_rows.columns
                ):
    
                    auc_value = float(
                        test_rows.iloc[0][
                            ROC_AUC_COLUMN
                        ]
                    )
    
                # ----------------------------------------------------
                # BALANCED TWO-COLUMN LAYOUT
                # ----------------------------------------------------
    
                # ----------------------------------------------------
                # ROC CURVE
                # ----------------------------------------------------
                
                st.altair_chart(
                    roc_chart,
                    width="stretch"
                )
                
                st.markdown("")
                
                # ----------------------------------------------------
                # ROC-AUC + INTERPRETATION BELOW
                # ----------------------------------------------------
                
                if auc_value is not None:
                    auc_display = f"{auc_value:.4f}"
                else:
                    auc_display = "Unavailable"
                
                roc_info_html = (
                    '<div class="roc-info-card">'
                
                    '<div class="roc-info-title">'
                    'ROC-AUC'
                    '</div>'
                
                    '<div class="roc-info-label">'
                    'Area Under the Curve'
                    '</div>'
                
                    '<div class="roc-info-value">'
                    f'{auc_display}'
                    '</div>'
                
                    '<div class="roc-mini-divider"></div>'
                
                    '<div class="roc-info-title">'
                    'Key Interpretation'
                    '</div>'
                
                    '<div class="roc-info-text">'
                    f'ROC-AUC {auc_display} shows how effectively this model '
                    'separates heart-disease-present and absent cases '
                    'on the final test set.'
                    '</div>'
                
                    '<div class="roc-info-note">'
                    'Dashed line = random classification.'
                    '</div>'
                
                    '</div>'
                )
                
                st.markdown(
                    roc_info_html,
                    unsafe_allow_html=True
                )
                
    #==================================
    # TAB 4 — SELECTED FEATURES
    # ========================================================

    with features_tab:

        st.markdown(
            f"### Selected Features — "
            f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)}"
        )

        selected_features = (
            selected_features_by_model.get(
                model_name,
                []
            )
        )

        if not selected_features:

            st.warning(
                "Selected feature names are unavailable."
            )

        else:

            number_selected = len(
                selected_features
            )

            st.markdown(
                f"**{number_selected} of "
                f"{total_transformed_features} "
                "transformed features retained**"
            )


            selected_feature_table = pd.DataFrame(
                {
                    "No.": range(
                        1,
                        number_selected + 1
                    ),
                    "Selected Transformed Feature":
                        selected_features
                }
            )

            st.dataframe(
                selected_feature_table,
                hide_index=True,
                width="stretch"
            )






# ------------------------------------------------------------
# SINGLE-PATIENT FORM STATE
# ------------------------------------------------------------

SINGLE_PATIENT_DEFAULTS = {
    "single_case_id": "",
    "single_age": 54,
    "single_sex": "M",
    "single_chest_pain": "ASY",
    "single_resting_bp": 130,
    "single_cholesterol": 237,
    "single_fasting_bs": 0,
    "single_resting_ecg": "Normal",
    "single_max_hr": 138,
    "single_exercise_angina": "N",
    "single_oldpeak": 0.6,
    "single_st_slope": "Flat"
}

for state_key, default_value in (
    SINGLE_PATIENT_DEFAULTS.items()
):
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


def reset_single_patient_form():
    """Restore the single-patient form defaults."""
    for state_key, default_value in (
        SINGLE_PATIENT_DEFAULTS.items()
    ):
        st.session_state[state_key] = default_value

    for state_key in [
        "single_patient_input",
        "single_prediction_result",
        "single_comparison_result",
        "current_prediction_id",
        "current_case_id"
    ]:
        st.session_state.pop(
            state_key,
            None
        )


# ------------------------------------------------------------
# RANDOM FOREST GLOBAL FEATURE IMPORTANCE
# ------------------------------------------------------------

def get_random_forest_feature_importance():
    """
    Extract and aggregate Random Forest importance values back to
    the original input features used by the current model bundle.
    """

    if "Random Forest" not in models:
        raise KeyError(
            "Random Forest is not available in the model bundle."
        )

    random_forest_pipeline = models[
        "Random Forest"
    ]

    if not hasattr(
        random_forest_pipeline,
        "named_steps"
    ):
        raise AttributeError(
            "The Random Forest object is not a fitted pipeline."
        )

    preprocessor = (
        random_forest_pipeline
        .named_steps
        .get("preprocessor")
    )

    random_forest_model = (
        random_forest_pipeline
        .named_steps
        .get("model")
    )

    # Be tolerant of different pipeline step names.
    if preprocessor is None:
        for step in (
            random_forest_pipeline
            .named_steps
            .values()
        ):
            if hasattr(
                step,
                "get_feature_names_out"
            ):
                preprocessor = step
                break

    if (
        random_forest_model is None
        or not hasattr(
            random_forest_model,
            "feature_importances_"
        )
    ):
        for step in (
            random_forest_pipeline
            .named_steps
            .values()
        ):
            if hasattr(
                step,
                "feature_importances_"
            ):
                random_forest_model = step
                break

    if preprocessor is None:
        raise KeyError(
            "The preprocessing step could not be found "
            "inside the Random Forest pipeline."
        )

    if (
        random_forest_model is None
        or not hasattr(
            random_forest_model,
            "feature_importances_"
        )
    ):
        raise AttributeError(
            "The Random Forest estimator does not expose "
            "feature_importances_."
        )
    
    transformed_feature_names = (
        preprocessor.get_feature_names_out()
    )
    
    # --------------------------------------------------------
    # Apply SelectKBest feature-selection mask
    # --------------------------------------------------------
    
    feature_selector = (
        random_forest_pipeline
        .named_steps
        .get("feature_selection")
    )
    
    if (
        feature_selector is not None
        and hasattr(
            feature_selector,
            "get_support"
        )
    ):
    
        selected_mask = (
            feature_selector.get_support()
        )
    
        transformed_feature_names = (
            np.asarray(
                transformed_feature_names
            )[selected_mask]
        )
    
    importance_values = np.asarray(
        random_forest_model.feature_importances_,
        dtype=float
    )

    if (
        len(transformed_feature_names)
        != len(importance_values)
    ):
        raise ValueError(
            "Transformed feature names and importance values "
            "do not have matching lengths."
        )

    transformed_table = pd.DataFrame({
        "Transformed Feature": (
            transformed_feature_names
        ),
        "Importance": importance_values
    })

    ordered_original_features = sorted(
        required_features,
        key=len,
        reverse=True
    )

    original_feature_names = []

    for transformed_name in transformed_table[
        "Transformed Feature"
    ]:

        cleaned_name = (
            str(transformed_name)
            .split("__", maxsplit=1)[-1]
        )

        matched_feature = None

        for original_feature in (
            ordered_original_features
        ):
            if (
                cleaned_name == original_feature
                or cleaned_name.startswith(
                    f"{original_feature}_"
                )
            ):
                matched_feature = (
                    original_feature
                )
                break

        original_feature_names.append(
            matched_feature
            if matched_feature is not None
            else cleaned_name
        )

    transformed_table["Feature"] = (
        original_feature_names
    )

    aggregated_table = (
        transformed_table
        .groupby(
            "Feature",
            as_index=False
        )["Importance"]
        .sum()
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    total_importance = float(
        aggregated_table[
            "Importance"
        ].sum()
    )

    if total_importance <= 0:
        raise ValueError(
            "Random Forest importance values sum to zero."
        )

    aggregated_table[
        "Importance Percentage"
    ] = (
        aggregated_table["Importance"]
        / total_importance
        * 100
    )

    return aggregated_table

def prepare_export_filename(
    entered_name,
    default_name,
    extension=".xlsx"
):
    """
    Prepare a safe export filename.

    Users may enter a custom filename.
    The required file extension is added automatically.
    """

    filename = str(
        entered_name
    ).strip()

    # Use default when the field is empty
    if not filename:
        filename = default_name

    # Remove extension if user already entered it
    if filename.lower().endswith(
        extension.lower()
    ):
        filename = filename[
            :-len(extension)
        ]

    # Characters not allowed in Windows filenames
    invalid_characters = [
        "<",
        ">",
        ":",
        '"',
        "/",
        "\\",
        "|",
        "?",
        "*"
    ]

    for character in invalid_characters:
        filename = filename.replace(
            character,
            "_"
        )

    # Remove unnecessary spaces and dots
    filename = (
        filename
        .strip()
        .strip(".")
    )

    # Fall back to default if nothing remains
    if not filename:
        filename = default_name

    return (
        filename
        + extension
    )
    
def build_prediction_excel(
    prediction_results
):
    """Create an Excel download for batch results."""
    buffer = BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:
        prediction_results.to_excel(
            writer,
            sheet_name="Prediction Results",
            index=False
        )

    return buffer.getvalue()

def autofit_excel_columns(writer):
    """
    Automatically adjust column widths for every worksheet
    in an Excel workbook.
    """

    for worksheet in writer.book.worksheets:

        for column_cells in worksheet.columns:

            column_letter = (
                column_cells[0].column_letter
            )

            maximum_length = 0

            for cell in column_cells:

                if cell.value is not None:

                    cell_length = len(
                        str(cell.value)
                    )

                    maximum_length = max(
                        maximum_length,
                        cell_length
                    )

            # Add some spacing while preventing
            # extremely wide columns.
            adjusted_width = min(
                max(
                    maximum_length + 2,
                    10
                ),
                40
            )

            worksheet.column_dimensions[
                column_letter
            ].width = adjusted_width

def build_selected_prediction_excel(
    selected_record
):
    """
    Create an Excel report for one selected
    historical prediction record.
    """

    buffer = BytesIO()

    


    # --------------------------------------------------------
    # Prediction summary
    # --------------------------------------------------------

    case_id_value = selected_record.get(
        "case_id"
    )

    if (
        pd.isna(case_id_value)
        or str(case_id_value).strip() == ""
    ):
        case_id_value = "Not Provided"

    prediction_summary = pd.DataFrame(
        [
            {
                "Prediction ID":
                    selected_record[
                        "prediction_id"
                    ],

                "Case ID":
                    case_id_value,

                "Prediction Date":
                    selected_record[
                        "prediction_date"
                    ],

                "Prediction Time":
                    selected_record[
                        "prediction_time"
                    ],

                "Selected Final Model":
                    selected_record[
                        "selected_model"
                    ],

                "Final Prediction":
                    selected_record[
                        "final_prediction"
                    ],

                "Heart-Disease Probability (%)":
                    round(
                        float(
                            selected_record[
                                "probability_heart_disease"
                            ]
                        ),
                        2
                    ),

                "No-Heart-Disease Probability (%)":
                    round(
                        float(
                            selected_record[
                                "probability_no_heart_disease"
                            ]
                        ),
                        2
                    ),

                "Model Agreement":
                    selected_record[
                        "model_agreement"
                    ],

                "Presence Votes":
                    int(
                        selected_record[
                            "presence_votes"
                        ]
                    ),

                "Absence Votes":
                    int(
                        selected_record[
                            "absence_votes"
                        ]
                    )
            }
        ]
    )

    # --------------------------------------------------------
    # Patient input
    # --------------------------------------------------------

    patient_input_report = pd.DataFrame(
        [
            {
                "Age":
                    selected_record[
                        "age"
                    ],

                "Sex":
                    selected_record[
                        "sex"
                    ],

                "Chest Pain Type":
                    selected_record[
                        "chest_pain_type"
                    ],

                "Resting BP":
                    selected_record[
                        "resting_bp"
                    ],

                "Cholesterol":
                    selected_record[
                        "cholesterol"
                    ],

                "Fasting BS":
                    selected_record[
                        "fasting_bs"
                    ],

                "Resting ECG":
                    selected_record[
                        "resting_ecg"
                    ],

                "Max HR":
                    selected_record[
                        "max_hr"
                    ],

                "Exercise Angina":
                    selected_record[
                        "exercise_angina"
                    ],

                "Oldpeak":
                    selected_record[
                        "oldpeak"
                    ],

                "ST Slope":
                    selected_record[
                        "st_slope"
                    ]
            }
        ]
    )

    # --------------------------------------------------------
    # Four-model comparison
    # --------------------------------------------------------

    model_results_report = pd.DataFrame(
        [
            {
                "Model": "ANN",

                "Prediction":
                    selected_record[
                        "ann_prediction"
                    ],

                "Predicted Class":
                    selected_record[
                        "ann_predicted_class"
                    ],

                "Heart-Disease Probability (%)":
                    selected_record[
                        "ann_probability_yes"
                    ],

                "No-Heart-Disease Probability (%)":
                    selected_record[
                        "ann_probability_no"
                    ]
            },

            {
                "Model": "SVM",

                "Prediction":
                    selected_record[
                        "svm_prediction"
                    ],

                "Predicted Class":
                    selected_record[
                        "svm_predicted_class"
                    ],

                "Heart-Disease Probability (%)":
                    selected_record[
                        "svm_probability_yes"
                    ],

                "No-Heart-Disease Probability (%)":
                    selected_record[
                        "svm_probability_no"
                    ]
            },

            {
                "Model": "Random Forest",

                "Prediction":
                    selected_record[
                        "random_forest_prediction"
                    ],

                "Predicted Class":
                    selected_record[
                        "random_forest_predicted_class"
                    ],

                "Heart-Disease Probability (%)":
                    selected_record[
                        "random_forest_probability_yes"
                    ],

                "No-Heart-Disease Probability (%)":
                    selected_record[
                        "random_forest_probability_no"
                    ]
            },

            {
                "Model": "XGBoost",

                "Prediction":
                    selected_record[
                        "xgboost_prediction"
                    ],

                "Predicted Class":
                    selected_record[
                        "xgboost_predicted_class"
                    ],

                "Heart-Disease Probability (%)":
                    selected_record[
                        "xgboost_probability_yes"
                    ],

                "No-Heart-Disease Probability (%)":
                    selected_record[
                        "xgboost_probability_no"
                    ]
            }
        ]
    )

    # --------------------------------------------------------
    # Write Excel workbook
    # --------------------------------------------------------

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:
    
        prediction_summary.to_excel(
            writer,
            sheet_name="Prediction Summary",
            index=False
        )
    
        patient_input_report.to_excel(
            writer,
            sheet_name="Patient Input",
            index=False
        )
    
        model_results_report.to_excel(
            writer,
            sheet_name="Model Results",
            index=False
        )
    
        autofit_excel_columns(
            writer
        )
    
    return buffer.getvalue()

def build_prediction_records_report_excel(
    history_records
):
    """
    Create a standard three-sheet Excel report
    for multiple historical prediction records.
    """

    buffer = BytesIO()

    prediction_summary_rows = []
    patient_input_rows = []
    model_results_rows = []

    # --------------------------------------------------------
    # Build report rows for every selected prediction
    # --------------------------------------------------------

    for _, record in history_records.iterrows():

        # ----------------------------------------------------
        # Case ID
        # ----------------------------------------------------

        case_id_value = record.get(
            "case_id"
        )

        if (
            pd.isna(case_id_value)
            or str(case_id_value).strip() == ""
        ):
            case_id_value = "Not Provided"

        prediction_id = record[
            "prediction_id"
        ]

        # ====================================================
        # SHEET 1: PREDICTION SUMMARY
        # ====================================================

        prediction_summary_rows.append(
            {
                "Prediction ID":
                    prediction_id,

                "Case ID":
                    case_id_value,

                "Prediction Date":
                    record[
                        "prediction_date"
                    ],

                "Prediction Time":
                    record[
                        "prediction_time"
                    ],

                "Final Model":
                    record[
                        "selected_model"
                    ],

                "Final Prediction":
                    record[
                        "final_prediction"
                    ],

                "Heart-Disease Probability (%)":
                    round(
                        float(
                            record[
                                "probability_heart_disease"
                            ]
                        ),
                        2
                    ),

                "No-Heart-Disease Probability (%)":
                    round(
                        float(
                            record[
                                "probability_no_heart_disease"
                            ]
                        ),
                        2
                    ),

                "Model Agreement":
                    record[
                        "model_agreement"
                    ],

                "Presence Votes":
                    int(
                        record[
                            "presence_votes"
                        ]
                    ),

                "Absence Votes":
                    int(
                        record[
                            "absence_votes"
                        ]
                    )
            }
        )

        # ====================================================
        # SHEET 2: PATIENT INPUT
        # ====================================================

        patient_input_rows.append(
            {
                "Prediction ID":
                    prediction_id,

                "Case ID":
                    case_id_value,

                "Age":
                    record[
                        "age"
                    ],

                "Sex":
                    record[
                        "sex"
                    ],

                "Chest Pain Type":
                    record[
                        "chest_pain_type"
                    ],

                "Resting BP":
                    record[
                        "resting_bp"
                    ],

                "Cholesterol":
                    record[
                        "cholesterol"
                    ],

                "Fasting BS":
                    record[
                        "fasting_bs"
                    ],

                "Resting ECG":
                    record[
                        "resting_ecg"
                    ],

                "Max HR":
                    record[
                        "max_hr"
                    ],

                "Exercise Angina":
                    record[
                        "exercise_angina"
                    ],

                "Oldpeak":
                    record[
                        "oldpeak"
                    ],

                "ST Slope":
                    record[
                        "st_slope"
                    ]
            }
        )

        # ====================================================
        # SHEET 3: FOUR-MODEL RESULTS
        # ====================================================
        
        model_results_rows.append(
            {
                "Prediction ID":
                    prediction_id,
        
                "Case ID":
                    case_id_value,
        
                "ANN Result":
                    record[
                        "ann_prediction"
                    ],
        
                "ANN Presence Probability (%)":
                    round(
                        float(
                            record[
                                "ann_probability_yes"
                            ]
                        ),
                        2
                    ),
        
                "SVM Result":
                    record[
                        "svm_prediction"
                    ],
        
                "SVM Presence Probability (%)":
                    round(
                        float(
                            record[
                                "svm_probability_yes"
                            ]
                        ),
                        2
                    ),
        
                "Random Forest Result":
                    record[
                        "random_forest_prediction"
                    ],
        
                "Random Forest Presence Probability (%)":
                    round(
                        float(
                            record[
                                "random_forest_probability_yes"
                            ]
                        ),
                        2
                    ),
        
                "XGBoost Result":
                    record[
                        "xgboost_prediction"
                    ],
        
                "XGBoost Presence Probability (%)":
                    round(
                        float(
                            record[
                                "xgboost_probability_yes"
                            ]
                        ),
                        2
                    )
            }
        )

    # --------------------------------------------------------
    # Convert to tables
    # --------------------------------------------------------

    prediction_summary = pd.DataFrame(
        prediction_summary_rows
    )

    patient_input_report = pd.DataFrame(
        patient_input_rows
    )

    model_results_report = pd.DataFrame(
        model_results_rows
    )

    # --------------------------------------------------------
    # Write Excel workbook
    # --------------------------------------------------------

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:
    
        prediction_summary.to_excel(
            writer,
            sheet_name="Prediction Summary",
            index=False
        )
    
        patient_input_report.to_excel(
            writer,
            sheet_name="Patient Input",
            index=False
        )
    
        model_results_report.to_excel(
            writer,
            sheet_name="Model Results",
            index=False
        )
    
        autofit_excel_columns(
            writer
        )
    
    return buffer.getvalue()
    
def build_history_records_excel(
    history_records
):
    """
    Create a simple Excel file containing
    prediction-history records.
    """

    buffer = BytesIO()

    export_data = history_records.copy()

    # --------------------------------------------------------
    # Prepare readable Date & Time
    # --------------------------------------------------------

    export_data["prediction_datetime"] = pd.to_datetime(
        export_data["prediction_datetime"],
        errors="coerce"
    )

    export_data["Date"] = (
        export_data["prediction_datetime"]
        .dt.strftime("%d %b %Y")
    )
    
    export_data["Time"] = (
        export_data["prediction_datetime"]
        .dt.strftime("%I:%M %p")
    )

    # --------------------------------------------------------
    # Make Case ID readable
    # --------------------------------------------------------

    export_data["Case ID"] = (
        export_data["case_id"]
        .fillna("Not Provided")
        .replace("", "Not Provided")
    )

    # --------------------------------------------------------
    # Prediction wording
    # --------------------------------------------------------

    export_data["Prediction Result"] = (
        export_data["final_predicted_class"]
        .map(
            {
                0: "Heart Disease Absent",
                1: "Heart Disease Present"
            }
        )
    )

    # --------------------------------------------------------
    # Final Excel columns
    # --------------------------------------------------------

    export_table = (
        export_data[
            [
                # Record information
                "prediction_id",
                "Case ID",
                "Date",
                "Time",

                # Patient input
                "age",
                "sex",
                "chest_pain_type",
                "resting_bp",
                "cholesterol",
                "fasting_bs",
                "resting_ecg",
                "max_hr",
                "exercise_angina",
                "oldpeak",
                "st_slope",

                # Four-model results
                "ann_prediction",
                "svm_prediction",
                "random_forest_prediction",
                "xgboost_prediction"
            ]
        ]
        .rename(
            columns={
                "prediction_id":
                    "Prediction ID",

                "age":
                    "Age",

                "sex":
                    "Sex",

                "chest_pain_type":
                    "Chest Pain Type",

                "resting_bp":
                    "Resting BP",

                "cholesterol":
                    "Cholesterol",

                "fasting_bs":
                    "Fasting BS",

                "resting_ecg":
                    "Resting ECG",

                "max_hr":
                    "Max HR",

                "exercise_angina":
                    "Exercise Angina",

                "oldpeak":
                    "Oldpeak",

                "st_slope":
                    "ST Slope",

                "ann_prediction":
                    "ANN Result",

                "svm_prediction":
                    "SVM Result",

                "random_forest_prediction":
                    "Random Forest Result",

                "xgboost_prediction":
                    "XGBoost Result"
            }
        )
    )

    result_columns = [
        "ANN Result",
        "SVM Result",
        "Random Forest Result",
        "XGBoost Result"
    ]

    for column in result_columns:

        export_table[column] = (
            export_table[column]
            .replace(
                {
                    "Present (1)": "Present",
                    "Absent (0)": "Not Present",
                    "Heart Disease Present": "Present",
                    "Heart Disease Absent": "Not Present"
                }
            )
        )

    # --------------------------------------------------------
    # Write Excel file
    # --------------------------------------------------------

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:

        export_table.to_excel(
            writer,
            sheet_name="Prediction Records",
            index=False
        )

    return buffer.getvalue()
# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("❤️ Heart Disease Presence Prediction")

st.caption(
    "Target classes: 0 = Heart disease absent, "
    "1 = Heart disease present"
)


# ============================================================
# MAIN NAVIGATION TABS
# ============================================================


(
    single_patient_tab,
    batch_prediction_tab,
    history_tab,
    performance_tab,
    about_system_tab
) = (
    st.tabs(
        [
            "Single Prediction",
            "Batch Prediction",
            "Prediction History & Reports",
            "Model Performance",
            "About the System"
        ]
    )
)

# ============================================================
# SINGLE-PATIENT PREDICTION TAB
# ============================================================

with single_patient_tab:

    st.subheader("Single Prediction")

    st.write(
        "Complete all fields and select "
        "**Generate Prediction with All Models**."
    )


    display_input_field_glossary()

    with st.form(
        "single_patient_form",
        clear_on_submit=False
    ):

        case_id = st.text_input(
            "Case ID",
            key="single_case_id",
            placeholder="Leave blank to auto-generate",
            help=(
                "Enter an existing Case ID if required. "
                "If left blank, the system will automatically "
                "generate a unique Case ID."
            )
        )
        
        st.caption(
            "Case ID is optional. A unique Case ID will be "
            "automatically assigned if this field is left blank."
        )

        left_column, middle_column, right_column = st.columns(3)

        with left_column:


            age = st.number_input(
                "Age",
                min_value=28,
                max_value=77,
                step=1,
                key="single_age"
            )

            sex = st.selectbox(
                "Sex",
                options=["F", "M"],
                key="single_sex",
                format_func=lambda value: (
                    "Male" if value == "M" else "Female"
                )
            )

            chest_pain_type = st.selectbox(
                "Chest Pain Type",
                options=["ASY", "ATA", "NAP", "TA"],
                key="single_chest_pain",
                format_func=lambda value: {
                    "ASY": "ASY - Asymptomatic",
                    "ATA": "ATA - Atypical Angina",
                    "NAP": "NAP - Non-Anginal Pain",
                    "TA": "TA - Typical Angina"
                }[value]
            )

            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=0,
                max_value=200,
                step=1,
                key="single_resting_bp",
                help="Enter 0 if the value is unavailable."
            )

        with middle_column:

            cholesterol = st.number_input(
                "Cholesterol (mg/dL)",
                min_value=0,
                max_value=603,
                step=1,
                key="single_cholesterol",
                help="Enter 0 if the value is unavailable."
            )

            fasting_bs = st.selectbox(
                "Fasting Blood Sugar",
                options=[0, 1],
                key="single_fasting_bs",
                format_func=lambda value: (
                    "0 - ≤ 120 mg/dL"
                    if value == 0
                    else "1 - > 120 mg/dL"
                )
            )

            resting_ecg = st.selectbox(
                "Resting ECG",
                options=["LVH", "Normal", "ST"],
                key="single_resting_ecg",
                format_func=lambda value: {
                    "Normal": "Normal",
                    "ST": "ST - ST-T Wave Abnormality",
                    "LVH": "LVH - Left Ventricular Hypertrophy"
                }[value]
            )

            max_hr = st.number_input(
                "Maximum Heart Rate",
                min_value=60,
                max_value=202,
                step=1,
                key="single_max_hr"
            )

        with right_column:

            exercise_angina = st.selectbox(
                "Exercise-Induced Angina",
                options=["N", "Y"],
                key="single_exercise_angina",
                format_func=lambda value: (
                    "No" if value == "N" else "Yes"
                )
            )

            oldpeak = st.number_input(
                "Oldpeak",
                min_value=-2.6,
                max_value=6.2,
                step=0.1,
                format="%.1f",
                key="single_oldpeak"
            )

            st_slope = st.selectbox(
                "ST Slope",
                options=["Down", "Flat", "Up"],
                key="single_st_slope",
                format_func=lambda value: {
                    "Flat": "Flat",
                    "Up": "Up - Upsloping",
                    "Down": "Down - Downsloping"
                }[value]
            )

            st.write("")
            st.write("")

            prediction_submitted = st.form_submit_button(
                "Generate Prediction with All Models",
                type="primary",
                width="stretch"
            )

    st.button(
        "Reset Form",
        on_click=reset_single_patient_form,
        width="content"
    )

    if prediction_submitted:

        patient_input = pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "ChestPainType": [chest_pain_type],
            "RestingBP": [resting_bp],
            "Cholesterol": [cholesterol],
            "FastingBS": [fasting_bs],
            "RestingECG": [resting_ecg],
            "MaxHR": [max_hr],
            "ExerciseAngina": [exercise_angina],
            "Oldpeak": [oldpeak],
            "ST_Slope": [st_slope]
        })

        selected_prediction_result = predict_patient(
            patient_input
        )

        comparison_result = compare_models(
            patient_input
        )

        if (
            selected_prediction_result["success"]
            and comparison_result["success"]
        ):

            st.session_state[
                "single_patient_input"
            ] = patient_input

            st.session_state[
                "single_prediction_result"
            ] = selected_prediction_result

            st.session_state[
                "single_comparison_result"
            ] = (
                comparison_result[
                    "results"
                ].copy()
            )
            # ------------------------------------------------
            # Assign Case ID
            # ------------------------------------------------

            if str(case_id).strip():

                effective_case_id = (
                    str(case_id).strip()
                )

            else:

                malaysia_time = datetime.now(
                    ZoneInfo(
                        "Asia/Kuala_Lumpur"
                    )
                )

                effective_case_id = (
                    "CASE-"
                    + malaysia_time.strftime(
                        "%Y%m%d"
                    )
                    + "-"
                    + uuid.uuid4().hex[
                        :4
                    ].upper()
                )
            # ------------------------------------------------
            # Save successful prediction to history
            # ------------------------------------------------

            try:

                prediction_id = (
                    save_single_prediction_to_history(
                        patient_input=patient_input,
                    
                        selected_prediction_result=(
                            selected_prediction_result
                        ),
                    
                        comparison_results=(
                            comparison_result[
                                "results"
                            ].copy()
                        ),
                    
                        case_id=effective_case_id
                    )
                )

                st.session_state[
                    "current_prediction_id"
                ] = prediction_id

                st.session_state[
                    "current_case_id"
                ] = effective_case_id

                st.toast(
                    (
                        "Prediction saved — "
                        f"Case ID: {effective_case_id}"
                    ),
                    icon="🕒"
                )

            except Exception as history_error:

                st.session_state.pop(
                    "current_prediction_id",
                    None
                )
                
                st.session_state.pop(
                    "current_case_id",
                    None
                )

                st.warning(
                    "The prediction was generated successfully, "
                    "but it could not be saved to history."
                )

                with st.expander(
                    "Show history-saving error"
                ):

                    st.code(
                        str(history_error)
                    )

        else:

            st.session_state.pop(
                "single_prediction_result",
                None
            )
            st.session_state.pop(
                "current_case_id",
                None
            )

            st.session_state.pop(
                "single_comparison_result",
                None
            )

            st.session_state.pop(
                "current_prediction_id",
                None
            )

            st.error(
                "The prediction could not be completed."
            )

            combined_errors = []

            if not selected_prediction_result[
                "success"
            ]:

                combined_errors.extend(
                    selected_prediction_result[
                        "errors"
                    ]
                )

            if not comparison_result[
                "success"
            ]:

                combined_errors.extend(
                    comparison_result[
                        "errors"
                    ]
                )

            for error in dict.fromkeys(
                combined_errors
            ):

                st.error(error)

    if (
        "single_prediction_result" in st.session_state
        and "single_comparison_result" in st.session_state
    ):

        selected_result = st.session_state[
            "single_prediction_result"
        ]

        comparison_output = st.session_state[
            "single_comparison_result"
        ].copy()

        st.divider()

        # ============================================================
        # MODEL PREDICTION COMPARISON
        # ============================================================

        st.subheader(
            "Model Prediction Comparison"
        )

        # ------------------------------------------------------------
        # Overall model agreement summary
        # ------------------------------------------------------------

        heart_disease_votes = int(
            (
                comparison_output[
                    "Predicted_Class"
                ] == 1
            ).sum()
        )

        no_heart_disease_votes = int(
            (
                comparison_output[
                    "Predicted_Class"
                ] == 0
            ).sum()
        )

        total_models = len(
            comparison_output
        )

        if heart_disease_votes == total_models:

            summary_message = (
                f"All {total_models} models predict "
                "<b>Heart Disease Present</b> for this patient."
            )

        elif no_heart_disease_votes == total_models:

            summary_message = (
                f"All {total_models} models predict "
                "<b>Heart Disease Absent</b> for this patient."
            )

        elif heart_disease_votes > no_heart_disease_votes:

            summary_message = (
                f"{heart_disease_votes} of {total_models} models "
                "predict <b>Heart Disease Present</b>."
            )

        elif no_heart_disease_votes > heart_disease_votes:

            summary_message = (
                f"{no_heart_disease_votes} of {total_models} models "
                "predict <b>Heart Disease Absent</b>."
            )

        else:

            summary_message = (
                "The models are evenly divided between "
                "<b>Heart Disease Present</b> and "
                "<b>Heart Disease Absent</b>."
            )


        summary_html = (
            '<div class="prediction-summary-box">'
            '<div class="prediction-summary-title">'
            'Prediction Summary'
            '</div>'
            '<div class="prediction-summary-text">'
            f'{summary_message} '
            'Compare the estimated heart-disease probability '
            'produced by each model below.'
            '</div>'
            '</div>'
        )
        
        st.markdown(
            summary_html,
            unsafe_allow_html=True
        )

        # ------------------------------------------------------------
        # Two-by-two model card layout
        # ------------------------------------------------------------

        first_card_row = st.columns(
            2,
            gap="medium"
        )

        second_card_row = st.columns(
            2,
            gap="medium"
        )

        model_card_columns = [
            first_card_row[0],
            first_card_row[1],
            second_card_row[0],
            second_card_row[1]
        ]


        # Put selected Random Forest first
        model_order = [
            name
            for name in [
                "Random Forest",
                "SVM",
                "XGBoost",
                "ANN"
            ]
            if name in comparison_output[
                "Model"
            ].tolist()
        ]


        for model_column, model_name in zip(
            model_card_columns,
            model_order
        ):

            model_row = (
                comparison_output[
                    comparison_output[
                        "Model"
                    ] == model_name
                ]
                .iloc[0]
            )

            probability = float(
                model_row[
                    "Probability_Heart_Disease"
                ]
            )

            predicted_class = int(
                model_row[
                    "Predicted_Class"
                ]
            )


            # --------------------------------------------------------
            # Prediction status
            # --------------------------------------------------------
            
            if predicted_class == 1:
            
                prediction_text = (
                    "● Heart Disease Present"
                )
            
                prediction_class = (
                    "prediction-status"
                )
            
            else:
            
                prediction_text = (
                    "● Heart Disease Absent"
                )
            
                prediction_class = (
                    "prediction-status-absent"
                )


            # --------------------------------------------------------
            # Selected model badge
            # --------------------------------------------------------

            if model_name == preferred_model_name:

                badge_html = (
                    '<span class="selected-model-badge">'
                    '★ Selected Model'
                    '</span>'
                )

            else:

                badge_html = ""

            with model_column:
            
                with st.container(
                    border=True
                ):
            
                    display_model_name = MODEL_DISPLAY_NAMES.get(
                        model_name,
                        model_name
                    )
            
                    card_html = (
                        '<div class="model-card-top">'
                            '<div class="model-card-title">'
                                f'{display_model_name}'
                            '</div>'
                            f'{badge_html}'
                        '</div>'
            
                        f'<div class="{prediction_class}">'
                            f'{prediction_text}'
                        '</div>'
            
                        '<div class="probability-row">'
                            '<div class="probability-label">'
                                'Heart-Disease Probability'
                            '</div>'
                            '<div class="probability-value">'
                                f'{probability:.2f}%'
                            '</div>'
                        '</div>'
            
                        '<div class="probability-track">'
                            f'<div class="probability-fill" '
                            f'style="width:{probability:.2f}%;">'
                            '</div>'
                        '</div>'
            
                        '<div class="probability-caption">'
                            'Estimated probability of heart-disease presence'
                        '</div>'
                    )
            
                    st.markdown(
                        card_html,
                        unsafe_allow_html=True
                    )
            
                    if st.button(
                        "View Analysis",
                        key=(
                            f"open_model_detail_"
                            f"{model_name}"
                        ),
                        width="stretch",
                        icon=":material/analytics:",
                        help=(
                            "Open this model's detailed "
                            "performance analysis."
                        )
                    ):
            
                        show_model_detail_dialog(
                            model_name
                        )
        st.divider()
        # ====================================================
        # DOWNLOAD CURRENT PREDICTION
        # ====================================================

        if (
            "current_prediction_id"
            in st.session_state
        ):

            current_prediction_id = (
                st.session_state[
                    "current_prediction_id"
                ]
            )
            
            current_case_id = (
                st.session_state.get(
                    "current_case_id"
                )
            )
            
            current_history_data = (
                load_prediction_history()
            )

            current_record_rows = (
                current_history_data[
                    current_history_data[
                        "prediction_id"
                    ] == current_prediction_id
                ]
            )

            if not current_record_rows.empty:

                current_record = (
                    current_record_rows.iloc[0]
                )

                st.markdown(
                    "### Download Current Prediction"
                )

                st.caption(
                    "Download the current patient's prediction, "
                    "input values and four-model results."
                )

                current_prediction_excel = (
                    build_selected_prediction_excel(
                        current_record
                    )
                )

                # ------------------------------------------------
                # Custom export filename
                # ------------------------------------------------
                
                if (
                    current_case_id is not None
                    and str(current_case_id).strip()
                ):
                
                    default_current_filename = (
                        f"{current_case_id}_Prediction_Report"
                    )
                
                else:
                
                    default_current_filename = (
                        f"{current_prediction_id}_Prediction_Report"
                    )
                
                
                # ------------------------------------------------
                # Custom report filename
                # ------------------------------------------------
                
                current_prediction_signature = str(
                    current_prediction_id
                )
                
                previous_current_prediction_signature = (
                    st.session_state.get(
                        "current_prediction_filename_signature"
                    )
                )
                
                if (
                    previous_current_prediction_signature
                    != current_prediction_signature
                ):
                
                    st.session_state[
                        "confirmed_current_prediction_filename"
                    ] = default_current_filename
                
                    st.session_state[
                        "current_prediction_filename_input"
                    ] = default_current_filename
                
                    st.session_state[
                        "current_prediction_filename_signature"
                    ] = current_prediction_signature
                
                
                current_filename_input = st.text_input(
                    "Report File Name",
                    key="current_prediction_filename_input",
                    help=(
                        "You may rename the report before downloading. "
                        "The .xlsx extension will be added automatically."
                    )
                )
                
                
                if st.button(
                    "Apply File Name",
                    width="stretch",
                    key="apply_current_prediction_filename"
                ):
                
                    st.session_state[
                        "confirmed_current_prediction_filename"
                    ] = current_filename_input
                
                    st.success(
                        "File name applied."
                    )
                
                
                current_prediction_filename = (
                    prepare_export_filename(
                        entered_name=st.session_state[
                            "confirmed_current_prediction_filename"
                        ],
                        default_name=default_current_filename,
                        extension=".xlsx"
                    )
                )
                

                st.download_button(
                    label="Download Current Prediction as Excel",
                    data=current_prediction_excel,
                    file_name=current_prediction_filename,
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    type="primary",
                    width="stretch",
                    key="download_current_prediction"
                )
                st.divider()
        
        # ============================================================
        # OPTIONAL DETAILS
        # ============================================================
                
        with st.expander(
            "View Entered Patient Data"
        ):
        
            st.markdown(
                "#### Entered Patient Data"
            )
        
            st.caption(
                "These are the patient values used to generate "
                "the predictions above."
            )
        
            st.dataframe(
                st.session_state[
                    "single_patient_input"
                ],
                hide_index=True,
                width="stretch"
            )


# ============================================================
# BATCH PREDICTION TAB
# ============================================================

with batch_prediction_tab:

    st.subheader("Excel Batch Prediction")

    st.write(
        "Upload multiple patient records and generate predictions "
        "for all records in one operation."
    )



    step_col1, step_col2, step_col3 = st.columns(3)

    with step_col1:
        with st.container(border=True):
            st.markdown("#### 1. Download")
            st.caption(
                "Download the Excel template containing "
                "the required input columns."
            )

    with step_col2:
        with st.container(border=True):
            st.markdown("#### 2. Complete")
            st.caption(
                "Enter one patient record per row without changing "
                "the column names."
            )

    with step_col3:
        with st.container(border=True):
            st.markdown("#### 3. Upload")
            st.caption(
                "Upload the completed file and run the four-model "
                "prediction comparison."
            )

    template = pd.DataFrame(
        columns=required_features
    )


    st.download_button(
        label="Download Excel Template",
        data=build_excel_template(),
        file_name="heart_disease_batch_template.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        type="primary",
        width="stretch"
    )

    st.divider()

    # ========================================================
    # BATCH FILE UPLOADER STATE
    # ========================================================

    uploaded_file = st.file_uploader(
        "Upload completed Excel file",
        type=["xlsx"],
        help=(
            "Supported format: XLSX. The file must contain "
            "exactly the required input columns."
        ),
        key="batch_file_uploader"
    )

    if uploaded_file is None:

        # Clear old batch results when the uploaded file is removed
        st.session_state.pop(
            "batch_prediction_result",
            None
        )

        st.session_state.pop(
            "batch_prediction_model",
            None
        )

        st.session_state.pop(
            "batch_saved_prediction_ids",
            None
        )

        st.info(
            "No file has been uploaded yet. Download a template, "
            "complete the patient records and upload the file here."
        )

    else:

        file_load_result = load_batch_file(
            uploaded_file
        )

        if not file_load_result["success"]:

            st.error(
                "The uploaded file could not be processed."
            )

            for error in file_load_result["errors"]:
                st.error(error)

        else:

            uploaded_data = file_load_result[
                "data"
            ]

            st.success(
                f"File loaded: {uploaded_file.name}"
            )

            upload_col1, upload_col2 = st.columns(
                2,
                gap="medium"
            )

            with upload_col1:

                st.markdown(
                    f"""
                    <div class="upload-stat-card">
                        <div class="upload-stat-label">
                            Uploaded Records
                        </div>
                        <div class="upload-stat-value">
                            {len(uploaded_data)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with upload_col2:

                st.markdown(
                    f"""
                    <div class="upload-stat-card">
                        <div class="upload-stat-label">
                            Uploaded Columns
                        </div>
                        <div class="upload-stat-value">
                            {uploaded_data.shape[1]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                "<div style='height: 12px;'></div>",
                unsafe_allow_html=True
            )

            with st.expander(
                "Preview uploaded data"
            ):

                st.dataframe(
                    uploaded_data,
                    hide_index=True,
                    width="stretch"
                )

            if st.button(
                "Run Batch Prediction with All Models",
                type="primary",
                width="stretch",
                key="run_batch_prediction"
            ):

                batch_result = predict_batch(
                    uploaded_data
                )

                if not batch_result["success"]:

                    st.error(
                        "Batch prediction could not be completed."
                    )

                    for error in batch_result["errors"]:
                        st.error(error)

                else:

                    batch_output = (
                        batch_result[
                            "results"
                        ].copy()
                    )

                    st.session_state[
                        "batch_prediction_result"
                    ] = batch_output

                    st.session_state[
                        "batch_prediction_model"
                    ] = batch_result["model"]

                    try:

                        saved_batch_prediction_ids = (
                            save_batch_predictions_to_history(
                                batch_output
                            )
                        )

                        st.session_state[
                            "batch_saved_prediction_ids"
                        ] = saved_batch_prediction_ids

                        st.toast(
                            (
                                f"{len(saved_batch_prediction_ids)} "
                                "batch prediction records saved to history."
                            ),
                            icon="🕒"
                        )

                    except Exception as history_error:

                        st.warning(
                            "The batch predictions were generated successfully, "
                            "but the records could not be saved to history."
                        )

                        with st.expander(
                            "Show batch history-saving error"
                        ):

                            st.code(
                                str(history_error)
                            )

    if "batch_prediction_result" in st.session_state:

        batch_output = st.session_state[
            "batch_prediction_result"
        ].copy()

        st.divider()

        st.subheader(
            "Prediction Summary by Model"
        )

        summary_rows = []

        for model_name in models.keys():

            class_column = (
                f"{model_name} Predicted Class"
            )

            if class_column not in batch_output.columns:
                continue

            absent_count = int(
                (
                    batch_output[class_column]
                    == 0
                ).sum()
            )

            present_count = int(
                (
                    batch_output[class_column]
                    == 1
                ).sum()
            )

            summary_rows.append({
                "Model": (
                    f"{model_name} ⭐"
                    if model_name == preferred_model_name
                    else model_name
                ),
                "Predicted Absent": absent_count,
                "Predicted Present": present_count,
                "Total Records": (
                    absent_count + present_count
                )
            })

        model_summary_table = pd.DataFrame(
            summary_rows
        )

        st.caption(
            f"⭐ {preferred_model_name} is the selected final model."
        )

        st.dataframe(
            model_summary_table,
            hide_index=True,
            width="stretch"
        )
        st.divider()
        st.markdown(
            "### Detailed Four-Model Prediction Results"
        )

        compact_results = pd.DataFrame({
            "Record": range(
                1,
                len(batch_output) + 1
            )
        })

        for model_name in models.keys():

            class_column = (
                f"{model_name} Predicted Class"
            )

            probability_column = (
                f"{model_name} "
                "Presence Probability (%)"
            )

            if (
                class_column in batch_output.columns
                and probability_column in batch_output.columns
            ):
                display_column_name = (
                    f"{model_name} ⭐"
                    if model_name == preferred_model_name
                    else model_name
                )

                compact_results[
                    display_column_name
                ] = [
                    (
                        "Present"
                        if int(predicted_class) == 1
                        else "Absent"
                    )
                    + f" ({probability:.2f}%)"
                    for predicted_class, probability
                    in zip(
                        batch_output[
                            class_column
                        ],
                        batch_output[
                            probability_column
                        ]
                    )
                ]

        compact_results[
            "Votes (P/A)"
        ] = [
            (
                f"{int(presence_votes)} / "
                f"{int(absence_votes)}"
            )
            for presence_votes, absence_votes
            in zip(
                batch_output[
                    "Presence Votes"
                ],
                batch_output[
                    "Absence Votes"
                ]
            )
        ]

        compact_results[
            "Agreement"
        ] = batch_output[
            "Model Agreement"
        ]

        st.caption(
            "Each model cell shows the prediction followed by its "
            "estimated probability of heart-disease presence. "
            "P/A means presence votes versus absence votes."
        )

        st.dataframe(
            compact_results,
            hide_index=True,
            width="stretch"
        )
        st.divider()


        st.markdown(
            "### Download Prediction Results"
        )

        # ------------------------------------------------
        # Custom batch export filename
        # ------------------------------------------------
        
        malaysia_time = datetime.now(
            ZoneInfo(
                "Asia/Kuala_Lumpur"
            )
        )
        
        default_batch_filename = (
            "Batch_Prediction_Results_"
            + malaysia_time.strftime(
                "%d%b%Y"
            )
        )
        
        
        if "confirmed_batch_export_filename" not in st.session_state:
        
            st.session_state[
                "confirmed_batch_export_filename"
            ] = default_batch_filename
        
        
        batch_filename_input = st.text_input(
            "Report File Name",
            value=default_batch_filename,
            key="batch_filename_input",
            help=(
                "You may rename the prediction results before downloading. "
                "The .xlsx extension will be added automatically."
            )
        )
        
        
        if st.button(
            "Apply File Name",
            width="stretch",
            key="apply_batch_filename"
        ):
        
            st.session_state[
                "confirmed_batch_export_filename"
            ] = batch_filename_input
        
            st.success(
                "File name applied."
            )
        
        

        
        
        batch_excel_filename = (
            prepare_export_filename(
                entered_name=st.session_state[
                    "confirmed_batch_export_filename"
                ],
                default_name=default_batch_filename,
                extension=".xlsx"
            )
        )

        # ------------------------------------------------
        # Build standard three-sheet batch report
        # ------------------------------------------------
        
        saved_batch_ids = st.session_state.get(
            "batch_saved_prediction_ids",
            []
        )
        
        if saved_batch_ids:
        
            batch_history_data = (
                load_prediction_history()
            )
        
            batch_report_records = (
                batch_history_data[
                    batch_history_data[
                        "prediction_id"
                    ].isin(
                        saved_batch_ids
                    )
                ]
                .copy()
            )
        
            # Preserve the original batch order
            batch_id_order = {
                prediction_id: index
                for index, prediction_id
                in enumerate(saved_batch_ids)
            }
        
            batch_report_records[
                "_batch_order"
            ] = (
                batch_report_records[
                    "prediction_id"
                ].map(
                    batch_id_order
                )
            )
        
            batch_report_records = (
                batch_report_records
                .sort_values(
                    "_batch_order"
                )
                .drop(
                    columns=[
                        "_batch_order"
                    ]
                )
                .reset_index(
                    drop=True
                )
            )
        
            batch_prediction_excel = (
                build_prediction_records_report_excel(
                    batch_report_records
                )
            )
        
        else:
        
            batch_prediction_excel = (
                build_prediction_excel(
                    batch_output
                )
            )
        st.download_button(
            label="Download Batch Prediction Results as Excel",
            data=batch_prediction_excel,
            file_name=batch_excel_filename,
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            width="stretch"
        )


# ============================================================
# PREDICTION HISTORY & REPORTS TAB
# ============================================================
# ============================================================
# HISTORY FILTER ACTIONS
# ============================================================

if "history_applied_search" not in st.session_state:
    st.session_state[
        "history_applied_search"
    ] = ""

if "history_page" not in st.session_state:
    st.session_state[
        "history_page"
    ] = 1

if "selected_history_prediction_ids" not in st.session_state:
    st.session_state[
        "selected_history_prediction_ids"
    ] = []
def apply_history_search():

    st.session_state[
        "history_applied_search"
    ] = (
        st.session_state.get(
            "history_search",
            ""
        ).strip()
    )

    st.session_state[
        "history_page"
    ] = 1

    st.session_state[
        "selected_history_prediction_ids"
    ] = []

def reset_history_filters():

    st.session_state[
        "history_year_filter"
    ] = "All Years"

    st.session_state[
        "history_month_filter"
    ] = "All Months"

    st.session_state[
        "history_search"
    ] = ""

    st.session_state[
        "history_applied_search"
    ] = ""

    st.session_state[
        "history_sort"
    ] = "Newest First"

    st.session_state[
        "history_page"
    ] = 1

    st.session_state[
        "selected_history_prediction_ids"
    ] = []

def reset_history_page():
    """
    Return prediction history to page 1.
    """

    st.session_state[
        "history_page"
    ] = 1


def reset_history_page_and_selection():
    """
    Return to page 1 and clear selected records.
    """

    st.session_state[
        "history_page"
    ] = 1

    st.session_state[
        "selected_history_prediction_ids"
    ] = []


def change_history_page(
    change,
    total_pages
):
    """
    Move forward or backward through prediction history.
    """

    current_page = st.session_state.get(
        "history_page",
        1
    )

    new_page = (
        current_page
        + change
    )

    st.session_state[
        "history_page"
    ] = min(
        max(
            new_page,
            1
        ),
        total_pages
    )
with history_tab:

    st.subheader(
        "Prediction History & Reports"
    )

    if (
        "history_delete_message"
        in st.session_state
    ):
    
        st.success(
            st.session_state.pop(
                "history_delete_message"
            )
        )
    st.caption(
        "Select one or more prediction records to download. "
        "The table reflects the currently applied filters."
    )

    # --------------------------------------------------------
    # Load prediction history
    # --------------------------------------------------------

    history_data = (
        load_prediction_history()
    )

    if history_data.empty:

            st.info(
                "No prediction history is available yet. "
                "Generate a single-patient or batch prediction "
                "to create prediction records."
            )

    else:



        # ----------------------------------------------------
        # Prepare compact history table
        # ----------------------------------------------------

        history_display = (
            history_data.copy()
        )

        # Convert saved date/time into a readable format
        history_display[
            "prediction_datetime"
        ] = pd.to_datetime(
            history_display[
                "prediction_datetime"
            ],
            errors="coerce"
        )

        history_display[
            "Date"
        ] = (
            history_display[
                "prediction_datetime"
            ]
            .dt.strftime(
                "%d %b %Y"
            )
        )

        history_display[
            "Time"
        ] = (
            history_display[
                "prediction_datetime"
            ]
            .dt.strftime(
                "%I:%M %p"
            )
        )

        # ----------------------------------------------------
        # Make result wording consistent
        # ----------------------------------------------------

        history_display[
            "Prediction Result"
        ] = history_display[
            "final_predicted_class"
        ].map(
            {
                0: "Not Present",
                1: "Present"
            }
        )


        # ----------------------------------------------------
        # Case ID display
        # ----------------------------------------------------

        history_display[
            "Case ID"
        ] = (
            history_display[
                "case_id"
            ]
            .fillna("—")
            .replace(
                "",
                "—"
            )
        )
         # ----------------------------------------------------
        # Date fields used for history filtering
        # ----------------------------------------------------

        history_display[
            "Year"
        ] = history_display[
            "prediction_datetime"
        ].dt.year

        history_display[
            "Month"
        ] = history_display[
            "prediction_datetime"
        ].dt.month_name()

        # ====================================================
        # HISTORY FILTERS
        # ====================================================
        
        filter_col1, filter_col2, filter_col3 = (
            st.columns(
                3,
                gap="small"
            )
        )
        
        # ----------------------------------------------------
        # Year
        # ----------------------------------------------------
        
        available_years = sorted(
            history_display[
                "Year"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist(),
            reverse=True
        )
        
        with filter_col1:
        
            selected_year = st.selectbox(
                "Year",
                options=[
                    "All Years"
                ] + available_years,
                key="history_year_filter",
                on_change=reset_history_page_and_selection
            )
        
        # ----------------------------------------------------
        # Month
        # ----------------------------------------------------
        
        month_order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ]
        
        available_months = month_order
        
        with filter_col2:
        
            selected_month = st.selectbox(
                "Month",
                options=[
                    "All Months"
                ] + available_months,
                key="history_month_filter",
                on_change=reset_history_page_and_selection
            )
        
        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------
        
        with filter_col3:
        
            history_sort = st.selectbox(
                "Sort By",
                options=[
                    "Newest First",
                    "Oldest First"
                ],
                key="history_sort",
                on_change=reset_history_page
            )
        
        
        # ====================================================
        # SEARCH
        # ====================================================
        
        search_col, search_button_col, reset_button_col = (
            st.columns(
                [5, 1.15, 1.45],
                gap="small",
                vertical_alignment="bottom"
            )
        )      
        with search_col:
        
            history_search = st.text_input(
                "Search Case ID or Prediction ID",
                placeholder=(
                    "e.g. CASE-001 or HD-20260820..."
                ),
                key="history_search"
            )
        
        with search_button_col:
        
            st.button(
                "Search",
                type="primary",
                width="stretch",
                key="history_search_button",
                on_click=apply_history_search,
                icon=":material/search:"
            )
        
        with reset_button_col:
        
            st.button(
                "Reset Filters",
                width="stretch",
                key="history_reset_button",
                on_click=reset_history_filters,
                icon=":material/restart_alt:"
            )
                    # ====================================================
        # APPLY FILTERS
        # ====================================================

        filtered_history = (
            history_display.copy()
        )

        # Year
        if selected_year != "All Years":

            filtered_history = (
                filtered_history[
                    filtered_history[
                        "Year"
                    ] == int(
                        selected_year
                    )
                ]
            )

        # Month
        if selected_month != "All Months":

            filtered_history = (
                filtered_history[
                    filtered_history[
                        "Month"
                    ] == selected_month
                ]
            )




        # ----------------------------------------------------
        # Search Case ID / Prediction ID
        # ----------------------------------------------------
        
        applied_history_search = (
            st.session_state.get(
                "history_applied_search",
                ""
            ).strip()
        )
        
        if applied_history_search:
        
            search_text = (
                applied_history_search
                .lower()
            )

            search_mask = (

                filtered_history[
                    "prediction_id"
                ]
                .astype(str)
                .str.lower()
                .str.contains(
                    search_text,
                    regex=False
                )

                |

                filtered_history[
                    "case_id"
                ]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(
                    search_text,
                    regex=False
                )
            )

            filtered_history = (
                filtered_history[
                    search_mask
                ]
            )
                    # ====================================================
        # APPLY SORTING
        # ====================================================

        if history_sort == "Newest First":

            filtered_history = (
                filtered_history.sort_values(
                    "prediction_datetime",
                    ascending=False
                )
            )

        elif history_sort == "Oldest First":

            filtered_history = (
                filtered_history.sort_values(
                    "prediction_datetime",
                    ascending=True
                )
            )



        filtered_history = (
            filtered_history
            .reset_index(
                drop=True
            )
        )
        
        
        # ====================================================
        # PAGINATION
        # ====================================================
        
        RECORDS_PER_PAGE = 100
        
        total_filtered_records = len(
            filtered_history
        )
        
        total_pages = max(
            1,
            (
                total_filtered_records
                + RECORDS_PER_PAGE
                - 1
            )
            // RECORDS_PER_PAGE
        )
        
        
        # Keep current page within valid range
        if (
            st.session_state[
                "history_page"
            ] > total_pages
        ):
        
            st.session_state[
                "history_page"
            ] = total_pages
        
        
        current_page = st.session_state[
            "history_page"
        ]
        
        
        page_start = (
            current_page - 1
        ) * RECORDS_PER_PAGE
        
        page_end = min(
            page_start
            + RECORDS_PER_PAGE,
            total_filtered_records
        )
        
        
        page_history = (
            filtered_history
            .iloc[
                page_start:
                page_end
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )
        
        
        # ----------------------------------------------------
        # Result count
        # ----------------------------------------------------
        
        if total_filtered_records > 0:
        
            if (
                total_filtered_records
                == len(history_display)
            ):
        
                st.caption(
                    f"Showing {page_start + 1}–{page_end} "
                    f"of {total_filtered_records} saved predictions."
                )
        
            else:
        
                st.caption(
                    f"Showing {page_start + 1}–{page_end} "
                    f"of {total_filtered_records} filtered predictions "
                    f"({len(history_display)} total saved)."
                )
        
        else:
        
            st.caption(
                f"Showing 0 of "
                f"{len(history_display)} saved predictions."
            )
        
        
        st.divider()
        
        # ====================================================
        # PREDICTION RECORDS
        # ====================================================
                
        all_history_excel = (
            build_prediction_records_report_excel(
                history_data
            )
        )

        st.markdown(
            "### Prediction Records"
        )
        
        st.caption(
            "Select one or more prediction records below "
            "to enable the Excel download."
        )
        
        
        # ====================================================
        # DOWNLOAD ALL PREDICTION RECORDS
        # ====================================================
        
        with st.expander(
            "Download All Prediction Records"
        ):
        
            malaysia_time = datetime.now(
                ZoneInfo(
                    "Asia/Kuala_Lumpur"
                )
            )
        
            default_all_history_filename = (
                "All_Prediction_Records_"
                + malaysia_time.strftime(
                    "%d%b%Y"
                )
            )
        
            # ------------------------------------------------
            # Custom report filename
            # ------------------------------------------------
            
            if "confirmed_all_history_filename" not in st.session_state:
                st.session_state[
                    "confirmed_all_history_filename"
                ] = default_all_history_filename
            
            
            all_history_filename_input = st.text_input(
                "Report File Name",
                value=default_all_history_filename,
                key="all_history_filename_input",
                help=(
                    "You may rename the report before downloading. "
                    "The .xlsx extension will be added automatically."
                )
            )
            
            
            if st.button(
                "Apply File Name",
                width="stretch",
                key="apply_all_history_filename"
            ):
            
                st.session_state[
                    "confirmed_all_history_filename"
                ] = all_history_filename_input
            
                st.success(
                    "File name applied."
                )
            
            
            all_history_export_filename = (
                prepare_export_filename(
                    entered_name=st.session_state[
                        "confirmed_all_history_filename"
                    ],
                    default_name=default_all_history_filename,
                    extension=".xlsx"
                )
            )
            
            
            st.download_button(
                label="Download All Prediction Records as Excel",
                data=all_history_excel,
                file_name=all_history_export_filename,
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                width="stretch",
                key="download_all_prediction_records"
            )
        
        

        if filtered_history.empty:

            st.error(
                "No prediction records match "
                "the selected filters."
            )
        else:

            # ------------------------------------------------
            # Build compact selectable table
            # ------------------------------------------------

            selected_history_ids = set(
                st.session_state.get(
                    "selected_history_prediction_ids",
                    []
                )
            )
            
            compact_history = pd.DataFrame(
                {
                    "Select": [
                        prediction_id
                        in selected_history_ids
            
                        for prediction_id
                        in page_history[
                            "prediction_id"
                        ].astype(str)
                    ],
            
                    "Prediction ID":
                        page_history[
                            "prediction_id"
                        ].values,
            
                    "Case ID":
                        page_history[
                            "Case ID"
                        ].values,
            
                    "Date":
                        page_history[
                            "Date"
                        ].values,
            
                    "Time":
                        page_history[
                            "Time"
                        ].values,
            
                    "Prediction Result":
                        page_history[
                            "Prediction Result"
                        ].values,
            
                    "ANN Result":
                        page_history[
                            "ann_predicted_class"
                        ].map(
                            {
                                0: "Not Present",
                                1: "Present"
                            }
                        ).values,
            
                    "SVM Result":
                        page_history[
                            "svm_predicted_class"
                        ].map(
                            {
                                0: "Not Present",
                                1: "Present"
                            }
                        ).values,
            
                    "Random Forest Result":
                        page_history[
                            "random_forest_predicted_class"
                        ].map(
                            {
                                0: "Not Present",
                                1: "Present"
                            }
                        ).values,
            
                    "XGBoost Result":
                        page_history[
                            "xgboost_predicted_class"
                        ].map(
                            {
                                0: "Not Present",
                                1: "Present"
                            }
                        ).values
                }
            )
            page_signature = "|".join(
                page_history[
                    "prediction_id"
                ]
                .astype(str)
                .tolist()
            )
            
            editor_signature = (
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    page_signature
                )
                .hex[
                    :8
                ]
            )

            edited_history = st.data_editor(
                compact_history,
                hide_index=True,
                width="stretch",
                key=(
                    f"prediction_history_selector_"
                    f"{current_page}_"
                    f"{editor_signature}"
                ),
                disabled=[
                    "Prediction ID",
                    "Case ID",
                    "Date",
                    "Time",
                    "Prediction Result",
                    "ANN Result",
                    "SVM Result",
                    "Random Forest Result",
                    "XGBoost Result"
                ],
                column_config={
                    "Select":
                        st.column_config.CheckboxColumn(
                            "Select",
                            help=(
                                "Select one or more prediction "
                                "records for download."
                            ),
                            default=False
                        )
                }
            )
            # ====================================================
            # PRESERVE SELECTION ACROSS PAGES
            # ====================================================
            
            current_page_ids = set(
                page_history[
                    "prediction_id"
                ]
                .astype(str)
                .tolist()
            )
            
            selected_on_current_page = set(
                edited_history.loc[
                    edited_history[
                        "Select"
                    ] == True,
                    "Prediction ID"
                ]
                .astype(str)
                .tolist()
            )
            
            stored_selected_ids = set(
                st.session_state.get(
                    "selected_history_prediction_ids",
                    []
                )
            )
            
            
            # Remove old selections belonging to this page
            stored_selected_ids.difference_update(
                current_page_ids
            )
            
            # Add the currently checked selections
            stored_selected_ids.update(
                selected_on_current_page
            )
            
            st.session_state[
                "selected_history_prediction_ids"
            ] = list(
                stored_selected_ids
            )
            
            
            selected_records = (
                filtered_history[
                    filtered_history[
                        "prediction_id"
                    ]
                    .astype(str)
                    .isin(
                        stored_selected_ids
                    )
                ]
                .copy()
            )
            
            selected_count = len(
                selected_records
            )
            
            
            # ====================================================
            # PAGINATION CONTROLS
            # ====================================================
            
            if total_pages > 1:
            
                previous_col, page_col, next_col = (
                    st.columns(
                        [1, 2, 1],
                        vertical_alignment="center"
                    )
                )
            
                with previous_col:
            
                    st.button(
                        "Previous",
                        icon=":material/chevron_left:",
                        width="stretch",
                        disabled=(
                            current_page <= 1
                        ),
                        key="history_previous_page",
                        on_click=change_history_page,
                        args=(
                            -1,
                            total_pages
                        )
                    )
            
                with page_col:
                
                    st.markdown(
                        (
                            "<div style='"
                            "text-align:center;"
                            "font-weight:600;"
                            "font-size:0.85rem;"
                            "transform:translateY(-6px);"
                            "'>"
                            f"{page_start + 1}–{page_end} "
                            f"of {total_filtered_records} records"
                            "</div>"
                        ),
                        unsafe_allow_html=True
                    )
                                            
                with next_col:
            
                    st.button(
                        "Next",
                        icon=":material/chevron_right:",
                        width="stretch",
                        disabled=(
                            current_page >= total_pages
                        ),
                        key="history_next_page",
                        on_click=change_history_page,
                        args=(
                            1,
                            total_pages
                        )
                    )
            
            
            # ====================================================
            # SELECTED RECORD EXPORT
            # ====================================================
            
            if selected_count > 0:



                # --------------------------------------------
                # Build Excel containing selected records
                # --------------------------------------------

                selected_count = len(
                    selected_records
                )
                
                # --------------------------------------------
                # Build Excel report
                # --------------------------------------------
                
                if selected_count == 1:
                
                    selected_records_excel = (
                        build_selected_prediction_excel(
                            selected_records.iloc[0]
                        )
                    )
                
                else:
                
                    selected_records_excel = (
                        build_prediction_records_report_excel(
                            selected_records
                        )
                    )

                st.caption(
                    f"{selected_count} prediction "
                    f"{'record' if selected_count == 1 else 'records'} "
                    "selected."
                )

                st.divider()
                # --------------------------------------------
                # Default selected-record export filename
                # --------------------------------------------
                
                if selected_count == 1:
                
                    selected_case_id = (
                        selected_records.iloc[0][
                            "case_id"
                        ]
                    )
                
                    if (
                        pd.notna(selected_case_id)
                        and str(selected_case_id).strip()
                    ):
                
                        default_selected_filename = (
                            f"{selected_case_id}_Prediction_Report"
                        )
                
                    else:
                
                        selected_prediction_id = (
                            selected_records.iloc[0][
                                "prediction_id"
                            ]
                        )
                
                        default_selected_filename = (
                            f"{selected_prediction_id}_Prediction_Report"
                        )
                
                else:
                
                    malaysia_time = datetime.now(
                        ZoneInfo(
                            "Asia/Kuala_Lumpur"
                        )
                    )
                
                    default_selected_filename = (
                        "Selected_Prediction_Records_"
                        + malaysia_time.strftime(
                            "%d%b%Y"
                        )
                    )
                
                
                # --------------------------------------------
                # Update filename when record selection changes
                # --------------------------------------------
                
                selection_signature = "|".join(
                    selected_records[
                        "prediction_id"
                    ]
                    .astype(str)
                    .tolist()
                )
                
                previous_selection_signature = (
                    st.session_state.get(
                        "selected_history_selection_signature"
                    )
                )
                
                # ------------------------------------------------
                # Reset filename when record selection changes
                # ------------------------------------------------
                
                if (
                    previous_selection_signature
                    != selection_signature
                ):
                
                    st.session_state[
                        "selected_history_filename_input"
                    ] = default_selected_filename
                
                    st.session_state[
                        "confirmed_selected_history_filename"
                    ] = default_selected_filename
                
                    st.session_state[
                        "selected_history_selection_signature"
                    ] = selection_signature
                
                
                # ------------------------------------------------
                # Custom report filename
                # ------------------------------------------------
                
                selected_filename_input = st.text_input(
                    "Report File Name",
                    key="selected_history_filename_input",
                    help=(
                        "You may rename the report before downloading. "
                        "The .xlsx extension will be added automatically."
                    )
                )
                
                
                if st.button(
                    "Apply File Name",
                    width="stretch",
                    key="apply_selected_history_filename"
                ):
                
                    st.session_state[
                        "confirmed_selected_history_filename"
                    ] = selected_filename_input
                
                    st.success(
                        "File name applied."
                    )
                
                
                selected_export_filename = (
                    prepare_export_filename(
                        entered_name=st.session_state[
                            "confirmed_selected_history_filename"
                        ],
                        default_name=default_selected_filename,
                        extension=".xlsx"
                    )
                )

                download_selected_col, delete_selected_col = (
                    st.columns(
                        [2, 1],
                        gap="small"
                    )
                )
                
                with download_selected_col:
                
                    st.download_button(
                        label=(
                            "Download Selected Prediction "
                            + (
                                "Record as Excel"
                                if selected_count == 1
                                else "Records as Excel"
                            )
                        ),
                        data=selected_records_excel,
                        file_name=selected_export_filename,
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        ),
                        width="stretch",
                        key="download_selected_prediction_records"
                    )
                
                with delete_selected_col:
                
                    if st.button(
                        (
                            "Delete Selected "
                            + (
                                "Record"
                                if selected_count == 1
                                else "Records"
                            )
                        ),
                        type="primary",
                        icon=":material/delete:",
                        width="stretch",
                        key="delete_selected_prediction_records"
                    ):
                
                        show_delete_history_dialog(
                            st.session_state[
                                "selected_history_prediction_ids"
                            ]
                        )
            st.divider()




# ============================================================
# MODEL PERFORMANCE TAB
# ============================================================

with performance_tab:





    preferred_results = final_test_results[
        final_test_results["Model"] == preferred_model_name
    ]

    if not preferred_results.empty:

        preferred_row = preferred_results.iloc[0]

        st.markdown(
            f"### Selected Final Model: {preferred_model_name}"
        )

        selected_metric_pairs = [
            ("Test Accuracy", "Accuracy"),
            ("Test Recall", "Recall"),
            ("Test F1-Score", F1_COLUMN),
            ("Test ROC-AUC", ROC_AUC_COLUMN),
            ("Test Specificity", "Specificity")
        ]

        metric_row1 = st.columns(3)
        metric_row2 = st.columns(3)

        metric_containers = [
            metric_row1[0],
            metric_row1[1],
            metric_row1[2],
            metric_row2[0],
            metric_row2[1],
            metric_row2[2]
        ]

        display_items = []

        for label, column in selected_metric_pairs:
        
            if (
                column is not None
                and column in preferred_row.index
            ):
                display_items.append(
                    (
                        label,
                        f"{float(preferred_row[column]) * 100:.2f}%"
                    )
                )

        if "FN" in preferred_row.index:
            display_items.append(
                (
                    "False Negatives",
                    str(
                        int(
                            preferred_row["FN"]
                        )
                    )
                )
            )

        for metric_container, (
            label,
            value
        ) in zip(
            metric_containers,
            display_items
        ):
            with metric_container:
                st.metric(
                    label,
                    value
                )


    st.divider()

    # --------------------------------------------------------
    # Grouped test-performance chart
    # --------------------------------------------------------

    st.markdown(
        "### Test-Set Performance Comparison"
    )

    chart_metric_mapping = []

    for source_column, display_name in [
        ("Accuracy", "Test Accuracy"),
        ("Precision", "Test Precision"),
        ("Recall", "Test Recall"),
        ("Specificity", "Test Specificity"),
        (F1_COLUMN, "Test F1-Score"),
        (ROC_AUC_COLUMN, "Test ROC-AUC")
    ]:
        if (
            source_column is not None
            and source_column
            in final_test_results.columns
        ):
            chart_metric_mapping.append(
                (
                    source_column,
                    display_name
                )
            )

    if chart_metric_mapping:

        performance_chart_data = (
            final_test_results[
                [
                    "Model"
                ] + [
                    source_column
                    for source_column, _
                    in chart_metric_mapping
                ]
            ]
            .copy()
        )

        performance_chart_data = (
            performance_chart_data.rename(
                columns={
                    source_column: display_name
                    for source_column, display_name
                    in chart_metric_mapping
                }
            )
        )

        display_metric_names = [
            display_name
            for _, display_name
            in chart_metric_mapping
        ]

        # Keep the saved model order whenever possible.
        preferred_display_order = [
            "ANN",
            "Random Forest",
            "SVM",
            "XGBoost"
        ]

        model_display_order = [
            model_name
            for model_name
            in preferred_display_order
            if model_name
            in performance_chart_data[
                "Model"
            ].tolist()
        ]

        model_display_order.extend([
            model_name
            for model_name
            in performance_chart_data[
                "Model"
            ].tolist()
            if model_name
            not in model_display_order
        ])

        performance_chart_long = (
            performance_chart_data.melt(
                id_vars="Model",
                value_vars=display_metric_names,
                var_name="Metric",
                value_name="Score"
            )
        )

        # Normalise only if a saved score is expressed as 0-100
        # rather than 0-1.
        if (
            performance_chart_long[
                "Score"
            ].max()
            > 1.5
        ):
            performance_chart_long[
                "Score"
            ] = (
                performance_chart_long[
                    "Score"
                ] / 100
            )

        minimum_score = float(
            performance_chart_long[
                "Score"
            ].min()
        )

        maximum_score = float(
            performance_chart_long[
                "Score"
            ].max()
        )

        y_axis_minimum = max(
            0.0,
            np.floor(
                (
                    minimum_score - 0.02
                ) * 100
            ) / 100
        )

        y_axis_maximum = min(
            1.0,
            np.ceil(
                (
                    maximum_score + 0.02
                ) * 100
            ) / 100
        )

        if (
            y_axis_maximum
            <= y_axis_minimum
        ):
            y_axis_minimum = 0.0
            y_axis_maximum = 1.0

        performance_bars = (
            alt.Chart(
                performance_chart_long
            )
            .mark_bar(
                size=34   # Increased from 28 → wider bars
            )
            .encode(
                x=alt.X(
                    "Model:N",
                    title="Classification Model",
                    sort=model_display_order,
                    axis=alt.Axis(
                        labelAngle=0,
                        labelFontSize=12,
                        titleFontSize=13
                    )
                ),
                xOffset=alt.XOffset(
                    "Metric:N",
                    sort=display_metric_names
                ),
                y=alt.Y(
                    "Score:Q",
                    title="Performance Score",
                    scale=alt.Scale(
                        domain=[
                            y_axis_minimum,
                            y_axis_maximum
                        ],
                        zero=False
                    ),
                    axis=alt.Axis(
                        format=".2f",
                        labelFontSize=11,
                        titleFontSize=13
                    )
                ),
                y2=alt.Y2(
                    datum=y_axis_minimum
                ),
                color=alt.Color(
                    "Metric:N",
                    title="Evaluation Metric",
                    sort=display_metric_names
                ),
                tooltip=[
                    alt.Tooltip(
                        "Model:N",
                        title="Model"
                    ),
                    alt.Tooltip(
                        "Metric:N",
                        title="Metric"
                    ),
                    alt.Tooltip(
                        "Score:Q",
                        title="Score",
                        format=".4f"
                    )
                ]
            )
            .properties(
                height=500
            )
        )
        
        
        # ============================================================
        # VALUE LABEL ABOVE EACH BAR
        # ============================================================
        
        performance_labels = (
            alt.Chart(
                performance_chart_long
            )
            .mark_text(
                dy=-10,
                fontSize=11,
                fontWeight="bold",
                color="white"
            )
            .encode(
                x=alt.X(
                    "Model:N",
                    sort=model_display_order
                ),
                xOffset=alt.XOffset(
                    "Metric:N",
                    sort=display_metric_names
                ),
                y=alt.Y(
                    "Score:Q",
                    scale=alt.Scale(
                        domain=[
                            y_axis_minimum,
                            y_axis_maximum
                        ],
                        zero=False
                    )
                ),
                text=alt.Text(
                    "Score:Q",
                    format=".4f"
                )
            )
        )
        
        
        performance_chart = (
            performance_bars
            + performance_labels
        ).configure_view(
            stroke=None
        ).configure_legend(
            orient="bottom",
            direction="horizontal",
            title=None,
            labelFontSize=11
        )
        
        
        st.altair_chart(
            performance_chart,
            width="stretch"
        )



        with st.expander(
            "View Exact Model Performance Values"
        ):

            comparison_columns = [
                column
                for column in [
                    "Model",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "Specificity",
                    F1_COLUMN,
                    ROC_AUC_COLUMN
                ]
                if (
                    column is not None
                    and column in final_test_results.columns
                )
            ]

            comparison_table = final_test_results[
                comparison_columns
            ].copy()

            # ------------------------------------------------
            # Convert evaluation metrics to percentage
            # ------------------------------------------------

            metric_columns = [
                column
                for column in comparison_table.columns
                if column != "Model"
            ]

            comparison_table[
                metric_columns
            ] = (
                comparison_table[
                    metric_columns
                ] * 100
            )

            # ------------------------------------------------
            # Highlight highest and lowest values
            # ------------------------------------------------

            def highlight_performance_extremes(
                column
            ):

                styles = [
                    ""
                ] * len(column)

                highest_value = column.max()
                lowest_value = column.min()

                for index, value in enumerate(
                    column
                ):

                    if value == highest_value:

                        styles[index] = (
                            "background-color: #143d2b; "
                            "color: #86efac; "
                            "font-weight: 700;"
                        )

                    elif value == lowest_value:

                        styles[index] = (
                            "background-color: #4a1d24; "
                            "color: #fca5a5; "
                            "font-weight: 700;"
                        )

                return styles

            # ------------------------------------------------
            # Apply styling
            # ------------------------------------------------

            styled_comparison_table = (
                comparison_table.style
                .apply(
                    highlight_performance_extremes,
                    subset=metric_columns,
                    axis=0
                )
                .format(
                    {
                        column: "{:.2f}%"
                        for column in metric_columns
                    }
                )
            )

            st.caption(
                "🟩 Highest value        🟥 Lowest value"
            )
            
            st.dataframe(
                styled_comparison_table,
                hide_index=True,
                width="stretch"
            )
    # --------------------------------------------------------
    # Classification error comparison
    # --------------------------------------------------------

    if (
        "FN" in final_test_results.columns
        and "FP" in final_test_results.columns
    ):

        st.divider()

        st.markdown(
            "### Classification Error Comparison"
        )

        error_table = (
            final_test_results[
                [
                    "Model",
                    "FN",
                    "FP"
                ]
            ]
            .copy()
            .rename(
                columns={
                    "FN": "False Negatives",
                    "FP": "False Positives"
                }
            )
        )

        st.dataframe(
            error_table,
            hide_index=True,
            width="stretch"
        )

    # --------------------------------------------------------
    # Overall Random Forest Feature Importance
    # --------------------------------------------------------
    st.divider()
    st.markdown(
        "### Overall Random Forest Feature Importance"
    )

    st.write(
        "This analysis shows which original patient input variables were most influential to the Random Forest model overall across the training dataset."
    )
    
    st.caption(
        "Random Forest retained 15 of the 21 transformed features. "
        "These selected features are grouped back into 9 original "
        "input variables for easier interpretation. "
        "The chart represents overall model behaviour and does not "
        "explain an individual patient's prediction."
    )
    try:

        feature_importance_table = (
            get_random_forest_feature_importance()
        )

        feature_importance_chart = (
            feature_importance_table[
                [
                    "Feature",
                    "Importance Percentage"
                ]
            ]
            .copy()
            .sort_values(
                "Importance Percentage",
                ascending=False
            )
        )

        feature_label_map = {
            "ST_Slope": "ST Slope",
            "ChestPainType": "Chest Pain",
            "ExerciseAngina": "Exercise Angina",
            "Oldpeak": "Oldpeak",
            "MaxHR": "Max HR",
            "Sex": "Sex",
            "FastingBS": "Fasting BS",
            "Cholesterol": "Cholesterol",
            "Age": "Age",
            "RestingBP": "Resting BP",
            "RestingECG": "Resting ECG"
        }

        feature_importance_chart[
            "Display Feature"
        ] = (
            feature_importance_chart[
                "Feature"
            ]
            .map(feature_label_map)
            .fillna(
                feature_importance_chart[
                    "Feature"
                ]
            )
        )

        feature_order = (
            feature_importance_chart[
                "Display Feature"
            ].tolist()
        )

        maximum_importance = float(
            feature_importance_chart[
                "Importance Percentage"
            ].max()
        )

        x_axis_maximum = float(
            np.ceil(
                maximum_importance
                * 1.15
                / 5
            ) * 5
        )

        feature_bars = (
            alt.Chart(
                feature_importance_chart
            )
            .mark_bar(
                size=34
            )
            .encode(
                y=alt.Y(
                    "Display Feature:N",
                    title="Original Input Feature",
                    sort=feature_order,
                    axis=alt.Axis(
                        labelFontSize=11,
                        titleFontSize=12
                    )
                ),
                x=alt.X(
                    "Importance Percentage:Q",
                    title="Importance (%)",
                    scale=alt.Scale(
                        domain=[
                            0,
                            x_axis_maximum
                        ],
                        zero=True
                    ),
                    axis=alt.Axis(
                        format=".0f",
                        labelFontSize=10,
                        titleFontSize=12
                    )
                ),
                tooltip=[
                    alt.Tooltip(
                        "Display Feature:N",
                        title="Feature"
                    ),
                    alt.Tooltip(
                        "Importance Percentage:Q",
                        title="Importance (%)",
                        format=".2f"
                    )
                ]
            )
            .properties(
                height=430
            )
        )
        
        feature_labels = (
            alt.Chart(
                feature_importance_chart
            )
            .mark_text(
                dx=5,
                align="left",
                fontSize=11,
                fontWeight="bold",
                color="white"
            )
            .encode(
                y=alt.Y(
                    "Display Feature:N",
                    sort=feature_order
                ),
                x=alt.X(
                    "Importance Percentage:Q",
                    scale=alt.Scale(
                        domain=[
                            0,
                            x_axis_maximum
                        ],
                        zero=True
                    )
                ),
                text=alt.Text(
                    "Importance Percentage:Q",
                    format=".2f"
                )
            )
        )

        feature_chart = (
            feature_bars
            + feature_labels
        ).configure_view(
            stroke=None
        )

        st.altair_chart(
            feature_chart,
            width="stretch"
        )
        
        feature_importance_display = (
            feature_importance_table[
                [
                    "Feature",
                    "Importance Percentage"
                ]
            ]
            .copy()
        )
        
        feature_importance_display[
            "Importance Percentage"
        ] = (
            feature_importance_display[
                "Importance Percentage"
            ].round(2)
        )
        
        with st.expander(
            "View Exact Feature Importance Values"
        ):
        
            st.dataframe(
                feature_importance_display,
                hide_index=True,
                width="stretch",
                column_config={
                    "Importance Percentage":
                        st.column_config.NumberColumn(
                            "Importance (%)",
                            format="%.2f%%"
                        )
                }
            )



    except Exception as importance_error:

        st.error(
            "The feature-importance chart is temporarily unavailable."
        )

        with st.expander(
            "Show feature-importance error"
        ):
            st.code(
                str(importance_error)
            )

# ============================================================
# ABOUT THE SYSTEM TAB
# ============================================================

with about_system_tab:

    st.subheader(
        "About the System"
    )

    st.write(
        "The Heart Disease Presence Prediction System is a "
        "supervised machine-learning prototype designed to analyse "
        "patient information and generate heart-disease presence "
        "predictions using multiple classification models. "
        "It supports both individual and batch prediction, compares "
        "results across ANN, SVM, Random Forest and XGBoost, and "
        "provides model-level analysis to help users understand the "
        "prediction outcomes."
    )


    st.divider()

    # ========================================================
    # SYSTEM CAPABILITIES
    # ========================================================
    
    st.markdown(
        "### System Capabilities"
    )
    
    st.write(
        "The system supports a complete prediction workflow, including "
        "single-patient prediction, batch processing, input validation, "
        "four-model comparison, detailed model analysis, prediction-history "
        "management and Excel reporting."
    )
    
    capability_row1 = st.columns(3)
    capability_row2 = st.columns(3)
    capability_row3 = st.columns(3)
    
    capability_columns = [
        capability_row1[0],
        capability_row1[1],
        capability_row1[2],
        capability_row2[0],
        capability_row2[1],
        capability_row2[2],
        capability_row3[0],
        capability_row3[1],
        capability_row3[2]
    ]
    
    capabilities = [
        (
            "Single-Patient Prediction",
            "Accepts the required patient information and generates "
            "prediction results for an individual record."
        ),
        (
            "Four-Model Comparison",
            "Runs ANN, SVM, Random Forest and XGBoost so their "
            "prediction outcomes can be compared."
        ),
        (
            "Input Validation",
            "Checks required fields, categorical values, numerical "
            "ranges and supported input formats before prediction."
        ),
        (
            "Batch Prediction",
            "Processes multiple patient records from Excel "
            "and returns predictions from all four models."
        ),
        (
            "Detailed Model Analysis",
            "Provides model-specific test evidence, confusion matrices, "
            "ROC curves and selected-feature information."
        ),
        (
            "Feature Importance",
            "Displays global Random Forest feature importance to show "
            "which original input variables contribute most strongly."
        ),
        (
            "Prediction History",
            "Automatically stores successful single-patient and batch "
            "predictions with their Case ID, date, time and model results."
        ),
        (
            "History Search and Filtering",
            "Allows saved records to be filtered by year and month or "
            "searched using Case ID or Prediction ID."
        ),
        (
            "Excel Reporting",
            "Exports current, selected or complete prediction records "
            "for reporting and record-keeping."
        )
    ]
    
    for capability_column, (
        capability_title,
        capability_description
    ) in zip(
        capability_columns,
        capabilities
    ):
    
        with capability_column:
    
            with st.container(
                border=True
            ):
    
                st.markdown(
                    f"#### {capability_title}"
                )
    
                st.caption(
                    capability_description
                )
    
    st.divider()
    # ========================================================
    # PREDICTION WORKFLOW
    # ========================================================

    st.markdown(
        "### Prediction Workflow"
    )

    st.write(
        "For a single-patient prediction, the system follows a "
        "structured workflow from data entry to prediction storage "
        "and reporting."
    )

    workflow_row1 = st.columns(3)
    workflow_row2 = st.columns(3)

    workflow_columns = [
        workflow_row1[0],
        workflow_row1[1],
        workflow_row1[2],
        workflow_row2[0],
        workflow_row2[1],
        workflow_row2[2]
    ]

    workflow_steps = [
        (
            "1. Enter Patient Data",
            "The user enters the required patient values. "
            "A Case ID may also be provided."
        ),
        (
            "2. Validate Input",
            "The system checks the values and prevents prediction "
            "when unsupported or invalid inputs are detected."
        ),
        (
            "3. Run Four Models",
            "ANN, SVM, Random Forest and XGBoost independently "
            "evaluate the validated patient record."
        ),
        (
            "4. Compare Results",
            "The prediction and estimated heart-disease probability "
            "from each model are presented for comparison."
        ),
        (
            "5. Save Prediction",
            "Successful single-patient predictions are automatically "
            "stored with a Prediction ID, Case ID, date and time."
        ),
        (
            "6. Review and Export",
            "Saved records can later be searched, filtered, selected "
            "and exported to Excel for reporting."
        )
    ]

    for workflow_column, (
        workflow_title,
        workflow_description
    ) in zip(
        workflow_columns,
        workflow_steps
    ):

        with workflow_column:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"#### {workflow_title}"
                )

                st.caption(
                    workflow_description
                )

    st.divider()

    # ========================================================
    # IDENTIFICATION AND HISTORY
    # ========================================================

    st.markdown(
        "### Prediction Record Management"
    )

    record_col1, record_col2 = (
        st.columns(2)
    )

    with record_col1:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### Case ID"
            )

            st.write(
                "A Case ID helps organise and retrieve prediction "
                "records. Users may enter an existing Case ID or "
                "leave the field blank for the system to generate "
                "one automatically."
            )

    with record_col2:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### Prediction ID"
            )

            st.write(
                "Each successful single-patient prediction receives "
                "a unique Prediction ID so that the individual "
                "prediction record can be identified and tracked."
            )

    st.caption(
        "Prediction history stores successful single-patient and "
        "batch prediction records. Each saved record receives a "
        "Prediction ID and Case ID for later retrieval."
    )

    st.divider()


    # ========================================================
    # SYSTEM INFORMATION
    # ========================================================
    
    st.markdown(
        "### System Information"
    )
    
    st.caption(
        "A summary of the machine-learning configuration used by "
        "the current prediction system."
    )
    
    info_col1, info_col2, info_col3, info_col4 = (
        st.columns(4)
    )
    
    with info_col1:
    
        with st.container(
            border=True,
            height=250
        ):
    
            st.markdown(
                "#### Models"
            )
    
            st.metric(
                label="Models",
                value=len(models),
                label_visibility="collapsed"
            )
    
            st.caption(
                "ANN, SVM, Random Forest and XGBoost are used "
                "to generate and compare prediction results."
            )
    
    with info_col2:
    
        with st.container(
            border=True,
            height=250
        ):
    
            st.markdown(
                "#### Input Variables"
            )
    
            st.metric(
                label="Input Variables",
                value=len(required_features),
                label_visibility="collapsed"
            )
    
            st.caption(
                "Each prediction uses 11 patient variables, including "
                "age, chest-pain type, blood pressure, cholesterol "
                "and other clinical measurements."
            )
    
    with info_col3:
    
        with st.container(
            border=True,
            height=250
        ):
    
            st.markdown(
                "#### Prediction Classes"
            )
    
            st.metric(
                label="Prediction Classes",
                value=len(class_labels),
                label_visibility="collapsed"
            )
    
            st.caption(
                "The system performs binary classification: "
                "Not Present or Present."
            )
    
    with info_col4:
    
        with st.container(
            border=True,
            height=250
        ):
    
            st.markdown(
                "#### Selected Final Model"
            )
    
            st.metric(
                label="Selected Final Model",
                value=preferred_model_name,
                label_visibility="collapsed"
            )
    
            st.caption(
                "Random Forest is used as the selected final model, "
                "while results from all four models remain available "
                "for comparison."
            )

