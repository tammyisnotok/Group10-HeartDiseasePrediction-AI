# Heart Disease Prediction Using Machine Learning

This project is a supervised machine-learning application developed to predict the presence of heart disease based on patient clinical information.

Four classification algorithms were developed and evaluated:

- Artificial Neural Network (ANN)
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

The trained models are integrated into a Streamlit-based prototype that supports individual prediction, batch prediction, four-model comparison, model performance analysis, prediction history and Excel reporting.

Random Forest is used as the selected final model, while prediction results from all four models remain available for comparison.

---

## Main Features

The prototype provides the following functions:

- Single-patient heart disease prediction
- Automatic Case ID and Prediction ID generation
- Four-model prediction comparison
- Heart-disease presence probability display
- Batch prediction using CSV or Excel files
- Batch input validation
- Prediction history storage using SQLite
- Search and filtering of prediction records
- Prediction history persistence after the application is closed
- Export of current prediction records to Excel
- Export of selected prediction-history records to Excel
- Export of complete prediction history to Excel
- Model performance comparison
- Confusion matrix analysis
- ROC curve and ROC-AUC analysis
- Selected feature display for each model
- Random Forest global feature-importance analysis

---

## Project Structure

```text
Group10-HeartDiseasePrediction-AI/
├── app.py
├── HeartDiseasePrediction.ipynb
├── requirements.txt
├── README.md
├── data/
│   ├── prediction_history.db
│   ├── raw/
│   └── outputs/
├── models/
│   ├── model_bundle.joblib
│   ├── ann_pipeline.joblib
│   ├── svm_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   └── xgboost_pipeline.joblib
└── results/
    ├── charts/
    └── tables/
```

### Main Files

- `app.py`  
  Streamlit prototype containing the user interface, prediction functions, history management and reporting functions.

- `HeartDiseasePrediction.ipynb`  
  Jupyter Notebook containing the machine-learning workflow, including data preprocessing, exploratory analysis, feature selection, model development, tuning and evaluation.

- `requirements.txt`  
  Python packages and versions required to run the prototype.

- `models/`  
  Contains the trained machine-learning pipelines and the saved model bundle used by the Streamlit application.

- `data/`  
  Contains the project dataset, batch-prediction files and the SQLite prediction-history database.

- `results/`  
  Contains generated evaluation tables, charts and model-analysis outputs.

---

## System Requirements

### Recommended Python Version

**Python 3.12 is recommended for this project.**

The prototype and its dependencies have been successfully tested using Python 3.12.

Using a different Python version may result in compatibility issues with some of the pinned machine-learning packages.

---

## Installation

### 1. Download or Clone the Repository

The repository can be downloaded directly from GitHub or cloned using Git:

```bash
git clone https://github.com/tammyisnotok/Group10-HeartDiseasePrediction-AI.git
```

Move into the project folder:

```bash
cd Group10-HeartDiseasePrediction-AI
```

---

### 2. Create a Python 3.12 Environment

Creating a separate environment is recommended to avoid package conflicts.

#### Using Conda

```bash
conda create -n heart-disease-project python=3.12 -y
```

Activate the environment:

```bash
conda activate heart-disease-project
```

Confirm the Python version:

```bash
python --version
```

The output should show Python 3.12.x.

---

### 3. Install Required Python Packages

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The main packages include:

- NumPy
- pandas
- scikit-learn
- XGBoost
- joblib
- Streamlit
- openpyxl

---

## Additional System Requirements

### macOS

XGBoost requires the OpenMP runtime on macOS.

If XGBoost cannot be loaded and an error referring to `libomp.dylib` or OpenMP appears, install `libomp` using Homebrew:

```bash
brew install libomp
```

After installation, restart the Streamlit application.

If Homebrew is not installed, install Homebrew before running the command above.

---

### Windows

XGBoost may require the Microsoft Visual C++ runtime.

If XGBoost fails to load because of a missing DLL or Visual C++ runtime error, install the latest supported **Microsoft Visual C++ Redistributable** and restart the application.

In most Windows environments where the required runtime is already installed, no additional XGBoost setup is necessary.

---

## Running the Prototype

After installing all requirements, start the application using:

```bash
streamlit run app.py
```

Streamlit should automatically open the prototype in the default web browser.

If the browser does not open automatically, access:

```text
http://localhost:8501
```

The exact port may be different if another Streamlit application is already using port 8501.

---

## Prototype Navigation

The prototype contains five main tabs:

### 1. Single-Patient Prediction

Allows the user to enter the required patient information and generate predictions using all four machine-learning models.

The system displays:

- prediction from each model
- estimated probability of heart-disease presence
- model agreement
- selected final model
- detailed model analysis

Successful predictions are automatically stored in prediction history.

---

### 2. Batch Prediction

Allows multiple patient records to be uploaded using CSV or Excel files.

Users can:

- download a CSV template
- download an Excel template
- upload multiple patient records
- validate the uploaded data
- generate predictions using all four models
- compare model results
- export batch results as CSV or Excel

Successful batch predictions are also automatically stored in prediction history.

---

### 3. Prediction History & Reports

Stores successful single-patient and batch prediction records in a local SQLite database.

Each record contains information such as:

- Prediction ID
- Case ID
- date
- time
- patient input values
- ANN result
- SVM result
- Random Forest result
- XGBoost result

Users can:

- filter records by year
- filter records by month
- search using Case ID or Prediction ID
- sort records by date
- select individual or multiple records
- download selected records as Excel
- download all prediction records as Excel

Prediction history remains available after the Streamlit application is closed and reopened from the same project folder.

---

### 4. Model Performance

Provides evaluation information for the trained machine-learning models.

The section includes:

- accuracy
- precision
- recall
- specificity
- F1 score
- ROC-AUC
- false negatives
- false positives
- model performance comparison
- Random Forest global feature importance

Detailed analysis for individual models also includes:

- confusion matrix
- ROC curve
- selected transformed features

---

### 5. About the System

Provides an overview of:

- system capabilities
- prediction workflow
- Case ID and Prediction ID management
- machine-learning configuration
- selected final model

---

## Batch Prediction Input Format

Batch prediction requires the following 11 input variables:

```text
Age
Sex
ChestPainType
RestingBP
Cholesterol
FastingBS
RestingECG
MaxHR
ExerciseAngina
Oldpeak
ST_Slope
```

The column names should not be modified.

CSV and XLSX formats are supported.

Templates can be downloaded directly from the **Batch Prediction** tab.

---

## Prediction History

Prediction history is stored locally in:

```text
data/prediction_history.db
```

The submitted database starts without test prediction records.

When a user generates a successful single-patient or batch prediction, the application automatically adds the new record to this database.

If the application is closed and later reopened using the same project folder, the previously generated prediction records remain available.

These locally generated records are not automatically uploaded back to GitHub.

---

## Machine-Learning Notebook

The complete machine-learning development workflow is available in:

```text
HeartDiseasePrediction.ipynb
```

The notebook contains the major stages of the project, including:

- dataset loading
- data understanding
- data cleaning
- exploratory data analysis
- preprocessing
- feature selection
- ANN development
- SVM development
- Random Forest development
- XGBoost development
- hyperparameter tuning
- cross-validation
- final test-set evaluation
- model comparison
- final model selection

---

## Model Selection

The four models were evaluated using an untouched final test set.

Random Forest was selected as the final model based on its overall performance across the evaluation measures while maintaining strong recall, F1 score and ROC-AUC performance.

The Streamlit prototype therefore uses Random Forest as the selected final prediction model while retaining the ANN, SVM and XGBoost outputs for comparison.

---

## Troubleshooting

### Streamlit Command Not Found

If:

```text
streamlit: command not found
```

appears, ensure the correct Python environment has been activated and run:

```bash
pip install -r requirements.txt
```

Then retry:

```bash
streamlit run app.py
```

### XGBoost Cannot Be Loaded on macOS

Install OpenMP:

```bash
brew install libomp
```

Then restart the application.

### XGBoost DLL Error on Windows

Ensure that the Microsoft Visual C++ Redistributable is installed, then restart the application.

### Model Bundle Not Found

Ensure the following file exists:

```text
models/model_bundle.joblib
```

The application should be started from the root project directory containing `app.py`, `models/`, `data/` and `results/`.

---

## Important Note

This prototype was developed for academic and educational purposes as part of a machine-learning project.

The prediction outputs represent the results generated by the trained machine-learning models and should not be treated as a substitute for professional medical diagnosis.