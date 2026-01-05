""" Diabetes Risk Prediction Dashboard """

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diabetes_model import DiabetesModel
from src.data_loader import load_and_prepare_data


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    """Load and train the model (cached for performance)"""
    X_train, X_test, y_train, y_test = load_and_prepare_data(add_interactions=True)
    
    model = DiabetesModel()
    model.train(X_train, y_train)
    
    return model, X_train.columns.tolist(), X_test, y_test


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_risk_category(probability, threshold=0.40):
    """Categorize risk based on probability"""
    if probability < 0.30:
        return "Low Risk", "🟢", "#28a745"
    elif probability < threshold:
        return "Moderate Risk", "🟡", "#ffc107"
    else:
        return "High Risk", "🔴", "#dc3545"


def explain_prediction(model, feature_names, patient_data):
    """Generate human-readable explanation of prediction"""
    # Get feature importance for this model
    importance = model.feature_importance(feature_names, top_k=5)
    
    explanations = []
    for feature, coef in importance:
        # Skip interaction terms for simplicity in explanation
        if '_x_' in feature:
            continue
            
        if coef > 0:
            explanations.append(f"**{feature.replace('_', ' ')}** ↑ increases risk")
        else:
            explanations.append(f"**{feature.replace('_', ' ')}** ↓ decreases risk")
    
    return explanations[:5]


# ============================================================================
# MAIN DASHBOARD
# ============================================================================
def main():
    # Load model
    model, feature_names, X_test, y_test = load_model()
    
    # ========================================================================
    # 1. HEADER
    # ========================================================================
    st.title("🩺 Diabetes Risk Prediction")
    st.markdown("### Clinical Decision Support Tool")
    st.markdown(
        '<p style="color: #6c757d; font-size: 17px;">'
        '✨ Explainable ML • 📊 Recall-optimized • 👥 Population health data (CDC)'
        '</p>',
        unsafe_allow_html=True
    )
    st.divider()
    
    # ========================================================================
    # 2. PATIENT INPUT SECTION
    # ========================================================================
    st.header("📋 Patient Information")
    
    # Create three columns for inputs
    col1, col2, col3 = st.columns(3)
    
    # --- COLUMN 1: Demographics & Medical History ---
    with col1:
        with st.expander("🧠 Demographics", expanded=False):
            age = st.slider("Age Group", 1, 13, 7, help="1=18-24, 13=80+")
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            education = st.slider("Education Level", 1, 6, 4, help="1=Never attended, 6=College graduate")
            income = st.slider("Income Level", 1, 8, 5, help="1=<$10k, 8=$75k+")
        
        with st.expander("❤️ Medical History", expanded=False):
            high_bp = st.selectbox("High Blood Pressure", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            high_chol = st.selectbox("High Cholesterol", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            chol_check = st.selectbox("Cholesterol Check (last 5 years)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            stroke = st.selectbox("Stroke History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            heart_disease = st.selectbox("Heart Disease/Attack", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            diff_walk = st.selectbox("Difficulty Walking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    # --- COLUMN 2: Lifestyle ---
    with col2:
        with st.expander("🏃 Lifestyle Factors", expanded=False):
            bmi = st.number_input("BMI (Body Mass Index)", 10, 100, 25, help="kg/m²")
            phys_activity = st.selectbox("Physical Activity (past 30 days)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            smoker = st.selectbox("Smoker (100+ cigarettes lifetime)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            hvy_alcohol = st.selectbox("Heavy Alcohol Consumption", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", 
            help="Adult men: >14 drinks/week, women: >7 drinks/week")
            fruits = st.selectbox("Consume Fruits Daily", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            veggies = st.selectbox("Consume Vegetables Daily", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    # --- COLUMN 3: Healthcare Access & General Health ---
    with col3:
        with st.expander("🩺 Healthcare Access", expanded=False):
            any_healthcare = st.selectbox("Any Health Coverage", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            no_doc_cost = st.selectbox("Couldn't See Doctor (Cost)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with st.expander("💊 Health Status", expanded=False):
            gen_hlth_options = {
                "Excellent": 1,
                "Very Good": 2,
                "Good": 3,
                "Fair": 4,
                "Poor": 5
            }
            gen_hlth_text = st.select_slider("General Health", 
                                         options=list(gen_hlth_options.keys()), 
                                         value="Good")
            gen_hlth = gen_hlth_options[gen_hlth_text]
            
            ment_hlth = st.slider("Mental Health (bad days/month)", 0, 30, 0)
            phys_hlth = st.slider("Physical Health (bad days/month)", 0, 30, 0)
    
    st.divider()
    
    # ========================================================================
    # 3. PREDICTION BUTTON
    # ========================================================================
    if st.button("🔍 Calculate Risk", type="primary", use_container_width=True):
        # Create patient data dictionary (matching original feature names)
        patient_data = {
            'HighBP': high_bp,
            'HighChol': high_chol,
            'CholCheck': chol_check,
            'BMI': bmi,
            'Smoker': smoker,
            'Stroke': stroke,
            'HeartDiseaseorAttack': heart_disease,
            'PhysActivity': phys_activity,
            'Fruits': fruits,
            'Veggies': veggies,
            'HvyAlcoholConsump': hvy_alcohol,
            'AnyHealthcare': any_healthcare,
            'NoDocbcCost': no_doc_cost,
            'GenHlth': gen_hlth,
            'MentHlth': ment_hlth,
            'PhysHlth': phys_hlth,
            'DiffWalk': diff_walk,
            'Sex': sex,
            'Age': age,
            'Education': education,
            'Income': income
        }
        
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Add interaction terms (matching training)
        if 'Age' in patient_df.columns and 'HvyAlcoholConsump' in patient_df.columns:
            patient_df['Age_x_Alcohol'] = patient_df['Age'] * patient_df['HvyAlcoholConsump']
        if 'Age' in patient_df.columns and 'BMI' in patient_df.columns:
            patient_df['Age_x_BMI'] = patient_df['Age'] * patient_df['BMI']
        if 'Age' in patient_df.columns and 'HighBP' in patient_df.columns:
            patient_df['Age_x_HighBP'] = patient_df['Age'] * patient_df['HighBP']
        if 'BMI' in patient_df.columns and 'PhysActivity' in patient_df.columns:
            patient_df['BMI_x_NoActivity'] = patient_df['BMI'] * (1 - patient_df['PhysActivity'])
        if 'BMI' in patient_df.columns and 'HighBP' in patient_df.columns:
            patient_df['BMI_x_HighBP'] = patient_df['BMI'] * patient_df['HighBP']
        if 'HvyAlcoholConsump' in patient_df.columns and 'Smoker' in patient_df.columns:
            patient_df['Alcohol_x_Smoker'] = patient_df['HvyAlcoholConsump'] * patient_df['Smoker']
        
        # Reorder columns to match training features
        patient_df = patient_df[feature_names]
        
        # Get prediction probability
        probability = model.model.predict_proba(patient_df)[0, 1]
        risk_category, risk_icon, risk_color = get_risk_category(probability)
        
        # ====================================================================
        # 4. RISK PREDICTION OUTPUT
        # ====================================================================
        st.header("🎯 Risk Assessment")
        
        # Display risk with color coding
        col_a, col_b = st.columns([2, 1])
        
        with col_a:
            st.markdown(f"### {risk_icon} {risk_category}")
            st.markdown(f"<h1 style='color: {risk_color}; margin-top: -20px;'>{probability:.1%}</h1>", 
                       unsafe_allow_html=True)
            st.markdown("**Estimated Diabetes Risk Probability**")
            
            # Clinical decision note
            st.info(
                "ℹ️ **Clinical Note:** High recall threshold used (≥85%) to minimize missed diagnoses. "
                "This model prioritizes sensitivity over specificity."
            )
        
        with col_b:
            st.markdown("#### Risk Categories")
            st.markdown("🟢 **Low Risk:** <30%")
            st.markdown("🟡 **Moderate Risk:** 30-40%")
            st.markdown("🔴 **High Risk:** ≥40%")
        
        st.divider()
        
        # ====================================================================
        # 5. MODEL EXPLANATION
        # ====================================================================
        st.header("💡 Why This Prediction?")
        st.markdown("**Top Contributing Factors:**")
        
        explanations = explain_prediction(model, feature_names, patient_df)
        
        for i, explanation in enumerate(explanations, 1):
            st.markdown(f"{i}. {explanation}")
        
        st.caption("These are the most influential features in the model's decision.")
        
        st.divider()
    
    # ========================================================================
    # 6. MEDICAL DISCLAIMER (Main Page Bottom - Collapsible)
    # ========================================================================
    with st.expander("⚠️ Medical Disclaimer"):
        st.warning(
            "This tool is for educational and decision-support purposes only. "
            "It does not replace professional medical diagnosis. "
            "Please consult a healthcare provider for proper diagnosis and treatment."
        )
    
    st.caption("© 2026 Diabetes Risk Prediction Tool")
    
    # ========================================================================
    # 7. MODEL PERFORMANCE SUMMARY (Sidebar)
    # ========================================================================
    with st.sidebar:
        st.header("📊 Model Performance")
        st.metric("ROC AUC Score", "0.82")
        st.metric("Recall (Diabetes)", "85%", help="Correctly identifies 85% of diabetic patients")
        st.metric("Precision (Diabetes)", "~28%", help="28% of positive predictions are correct")
        st.metric("Dataset Size", "253,680 patients")
        
        st.divider()
        
        # ====================================================================
        # 8. ABOUT THE MODEL (Expandable)
        # ====================================================================
        with st.expander("ℹ️ About This Model"):
            st.markdown("""
            **Model Details:**
            - **Algorithm:** Logistic Regression (explainable)
            - **Features:** Standardized and scaled
            - **Class Imbalance:** Handled with balanced weights
            - **Threshold Tuning:** Optimized for clinical recall (≥85%)
            - **Interaction Terms:** Included for complex relationships
            
            **Data Source:**
            - [CDC Behavioral Risk Factor Surveillance System (BRFSS) 2014](https://www.cdc.gov/brfss/annual_data/annual_2014.html)
            - [UCI Machine Learning Repository](https://github.com/uci-ml-repo/ucimlrepo?tab=readme-ov-file)
            """)


# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
