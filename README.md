🩺 Diabetes Risk Prediction Dashboard

    An explainable, clinically-oriented machine learning dashboard that estimates an individual’s risk of diabetes using population-level health data from the CDC Behavioral Risk Factor Surveillance System.

    This project emphasizes medical usability, interpretability, and decision support, not just model accuracy.

    ⚠️ Important: This tool is for educational and decision-support purposes only and does not replace professional medical diagnosis.

📌 Project Overview

    This application:

    - Predicts diabetes risk probability for a patient
    - Uses a recall-optimized threshold to minimize missed diagnoses
    - Explains predictions using interpretable model coefficients
    - Presents results through an interactive Streamlit dashboard

🎯 Key Objectives

    - Build an explainable ML model suitable for healthcare
    - Handle class imbalance correctly
    - Demonstrate why default thresholds (0.5) are suboptimal in medical settings
    - Apply UX principles to make ML outputs understandable to non-ML users

🧠 Model & Methodology

    Model

        - Algorithm: Logistic Regression
        - Why: Transparent, interpretable, and clinically trusted
        - Class Imbalance Handling: class_weight="balanced"

    Features

        - 21 base features + 6 medically motivated interaction terms

    Categories

        - Demographics
        - Medical history
        - Lifestyle factors
        - Healthcare access
        - General health indicators

    Feature Engineering

    Interaction terms were added to capture non-linear medical relationships:

        - Age x Heavy Alcohol Consumption
        - Age × BMI
        - Age × High Blood Pressure
        - BMI × No Physical Activity
        - BMI x High Blood Pressure
        - Alcohol × Smoking

⚖️ Threshold Strategy (Core Contribution)
❌ Why Accuracy & F1 Are Misleading

    - Dataset is highly imbalanced
    - Maximizing F1 resulted in a threshold of 1.0, which:
        - Misses 100% of diabetics
        - Is clinically useless

✅ Clinical Threshold - RECALL-BASED (Chosen Approach)

    Threshold chosen to achieve:
        ≥ 85% Recall (Sensitivity)
    This prioritizes catching diabetics, even at the cost of false positives

📌 Clinical Rationale:

    In healthcare, false negatives are more dangerous than false positives.

📊 Model Performance (Clinical Threshold = 0.40)
      Metric	                Value

      ROC AUC	                0.82 
      Recall (Diabetes)	        85%
      Precision (Diabetes)	    ~28%
      Accuracy	                ~67%
      Dataset Size	            253,680 patients

💡 Explainability (Explainable Artificial Intelligence ~ XAI)

    - The dashboard explains predictions using:
    - Logistic regression coefficients
    - Directional explanations:
        “↑ Increases risk”
        “↓ Decreases risk”
    - Top contributing factors only (cognitive load reduction)

    Example:

    High Blood Pressure ↑ increases diabetes risk
    BMI ↑ increases diabetes risk
    Physical Activity ↓ decreases diabetes risk

🖥️ Dashboard Features
    1. Patient Input Interface

        - Grouped using UX principles:
        - Demographics
        - Medical History
        - Lifestyle
        - Healthcare Access

    2. Risk Assessment Output

        Displays:

            - Risk percentage
            - Risk category (Low / Moderate / High)
            - Color-coded feedback (🟢 🟡 🔴)
            - Includes a clinical decision note explaining the threshold choice

    3. Prediction Explanation

        - Shows why the model made its decision
        - Avoids technical jargon

    4. Sidebar Model Summary

        - ROC AUC
        - Recall
        - Precision
        - Dataset size
        - Model details

🧪 Project Structure

    diabetes-risk-ai/
    ├── dashboard.py            # Streamlit dashboard
    │
    ├── data/
    │   └── raw/
    │       └── cdc_diabetes.csv
    │
    ├── src/
    │   ├── fetch_data.py        # Downloads dataset
    │   └── data_loader.py      # Preprocessing & feature engineering
    │
    ├── models/
    │   └── diabetes_model.py   # Logistic Regression wrapper
    │
    ├── experiments/
    │   └── train_logistic.py   # Model training & threshold analysis
    │
    ├── precision_recall_curve.png
    └── README.md

🚀 How to Run
    1. Install Dependencies
    pip install -r requirements.txt

    2. Fetch Dataset
    python src/fetch_data.py

    3. Train & Evaluate Model
    python experiments/train_logistic.py

    4. Launch Dashboard
    streamlit run app/dashboard.py

🧬 Data Source

    - CDC 2014
    - Retrieved via:

        - UCI Machine Learning Repository
        - ucimlrepo Python package

    Links:

    https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

    https://github.com/uci-ml-repo/ucimlrepo?tab=readme-ov-file`

🧭 UX & Design Principles Used

    Progressive disclosure
    Cognitive load reduction (chunking)
    Improves Scannability

⚠️ Medical Disclaimer

    This tool is intended for educational and decision-support use only.
    It does not provide medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare professional.

🧠 Learning Sources

    This project was developed as part of a hands-on learning process in applied machine learning.

    Core machine learning concepts (logistic regression, class imbalance, evaluation metrics, threshold tuning, and feature engineering, scaling) were learned and reinforced through:

    Google Machine Learning Crash Course
    https://developers.google.com/machine-learning/crash-course

    Practical experimentation and iterative improvement with guidance and explanations provided by OpenAI tools, used as a learning assistant and code review aid.

    All model design decisions, evaluation logic, and clinical considerations were implemented and validated by the author.

👤 Author

    Taha Baroudi
    Software Engineering Student
    Focus: Machine Learning, AI, Healthcare Applications
