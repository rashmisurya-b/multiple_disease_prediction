import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")

# -------------------------
# Load models
# -------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

MODEL_PATHS = {
    "Kidney Disease": "C:\\Users\\B Rashmi Surya Vetri\\Desktop\\Multiple Disease Prediction\\models\\kidney_pipeline_best.joblib",
    "Liver Disease": "C:\\Users\\B Rashmi Surya Vetri\\Desktop\\Multiple Disease Prediction\\models\\liver_pipeline_best.joblib",
    "Parkinson's Disease": "C:\\Users\\B Rashmi Surya Vetri\\Desktop\Multiple Disease Prediction\\models\\parkinsons_pipeline_best.joblib"
}

# -------------------------
# Input forms for each disease
# -------------------------
def kidney_form():
    st.subheader("💉Kidney Disease Prediction")

    age = st.number_input("Age (years)", 0, 120, 45)
    bp = st.number_input("Blood Pressure (mm/Hg)", 0, 200, 80)
    sg = st.selectbox("Specific Gravity", ["1.005", "1.010", "1.015", "1.020", "1.025"])
    al = st.number_input("Albumin", 0, 5, 0)
    su = st.number_input("Sugar", 0, 5, 0)
    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
    pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
    ba = st.selectbox("Bacteria", ["notpresent", "present"])
    bgr = st.number_input("Blood Glucose Random (mgs/dl)", 0, 500, 120)
    bu = st.number_input("Blood Urea (mgs/dl)", 0, 400, 40)
    sc = st.number_input("Serum Creatinine (mgs/dl)", 0.0, 20.0, 1.2)
    sod = st.number_input("Sodium (mEq/L)", 100.0, 200.0, 140.0)
    pot = st.number_input("Potassium (mEq/L)", 2.0, 10.0, 4.5)
    hemo = st.number_input("Hemoglobin (gms)", 3.0, 20.0, 15.0)
    pcv = st.number_input("Packed Cell Volume", 10, 60, 40)
    wc = st.number_input("White Blood Cell Count (cells/cumm)", 2000, 25000, 8000)
    rc = st.number_input("Red Blood Cell Count (millions/cmm)", 2.0, 8.0, 5.0)
    htn = st.selectbox("Hypertension", ["yes", "no"])
    dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
    cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])
    appet = st.selectbox("Appetite", ["good", "poor"])
    pe = st.selectbox("Pedal Edema", ["yes", "no"])
    ane = st.selectbox("Anemia", ["yes", "no"])

    user_input = {
        "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
        "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba,
        "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
        "hemo": hemo, "pcv": pcv, "wc": wc, "rc": rc,
        "htn": htn, "dm": dm, "cad": cad, "appet": appet,
        "pe": pe, "ane": ane
    }
    return user_input


def liver_form():
    st.subheader("💊Liver Disease Prediction")
    age = st.number_input("Age", 0, 120, 45)
    gender = st.selectbox("Gender", ["male", "female"])
    total_bilirubin = st.number_input("Total_Bilirubin", 0.0, 50.0, 1.0)
    direct_bilirubin = st.number_input("Direct_Bilirubin", 0.0, 20.0, 0.5)
    alk_phos = st.number_input("Alkaline_Phosphotase", 0, 2000, 200)
    alt = st.number_input("Alamine_Aminotransferase", 0, 2000, 30)
    ast = st.number_input("Aspartate_Aminotransferase", 0, 2000, 40)
    total_protein = st.number_input("Total_Proteins", 0.0, 10.0, 6.5)
    albumin = st.number_input("Albumin", 0.0, 6.0, 3.0)
    ag_ratio = st.number_input("Albumin_and_Globulin_Ratio", 0.0, 5.0, 1.0)

    user_input = {
        "Age": age, 
        "Gender": gender,
        "Total_Bilirubin": total_bilirubin,
        "Direct_Bilirubin": direct_bilirubin,
        "Alkaline_Phosphotase": alk_phos,
        "Alamine_Aminotransferase": alt,
        "Aspartate_Aminotransferase": ast,
        "Total_Protiens": total_protein,
        "Albumin": albumin,
        "Albumin_and_Globulin_Ratio": ag_ratio
    }
    return user_input

def parkinsons_form():
    st.subheader("👩‍⚕️Parkinson's Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", 50.0, 300.0, 150.0)
        fhi = st.number_input("MDVP:Fhi(Hz)", 100.0, 600.0, 250.0)
        flo = st.number_input("MDVP:Flo(Hz)", 50.0, 300.0, 100.0)
        jitter_percent = st.number_input("MDVP:Jitter(%)", 0.0, 1.0, 0.01, format="%.4f")
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 0.1, 0.005, format="%.5f")
        rap = st.number_input("MDVP:RAP", 0.0, 0.2, 0.01, format="%.5f")
        ppq = st.number_input("MDVP:PPQ", 0.0, 0.2, 0.01, format="%.5f")
        ddp = st.number_input("Jitter:DDP", 0.0, 0.6, 0.02, format="%.5f")
        shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.02, format="%.4f")
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0, 1.0, 0.2, format="%.4f")
        apq3 = st.number_input("Shimmer:APQ3", 0.0, 1.0, 0.02, format="%.4f")

    with col2:
        apq5 = st.number_input("Shimmer:APQ5", 0.0, 1.0, 0.03, format="%.4f")
        apq = st.number_input("MDVP:APQ", 0.0, 1.0, 0.05, format="%.4f")
        dda = st.number_input("Shimmer:DDA", 0.0, 1.0, 0.06, format="%.4f")
        nhr = st.number_input("NHR", 0.0, 0.5, 0.02, format="%.4f")
        hnr = st.number_input("HNR", 0.0, 50.0, 20.0, format="%.2f")
        rpde = st.number_input("RPDE", 0.0, 1.0, 0.5, format="%.3f")
        dfa = st.number_input("DFA", 0.0, 2.0, 0.7, format="%.3f")
        spread1 = st.number_input("spread1", -10.0, 0.0, -5.0, format="%.3f")
        spread2 = st.number_input("spread2", 0.0, 10.0, 3.0, format="%.3f")
        d2 = st.number_input("D2", 0.0, 5.0, 2.0, format="%.3f")
        ppe = st.number_input("PPE", 0.0, 1.0, 0.3, format="%.3f")

    user_input = {
        "MDVP:Fo(Hz)": fo,
        "MDVP:Fhi(Hz)": fhi,
        "MDVP:Flo(Hz)": flo,
        "MDVP:Jitter(%)": jitter_percent,
        "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP": rap,
        "MDVP:PPQ": ppq,
        "Jitter:DDP": ddp,
        "MDVP:Shimmer": shimmer,
        "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": apq5,
        "MDVP:APQ": apq,
        "Shimmer:DDA": dda,
        "NHR": nhr,
        "HNR": hnr,
        "RPDE": rpde,
        "DFA": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe
    }
    return user_input


# -------------------------
# Prediction helper
# -------------------------
def predict(model, input_dict):
    df = pd.DataFrame([input_dict])
    # Convert text columns properly
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except:
            pass
    proba = model.predict_proba(df)[:, 1][0]
    pred = int(model.predict(df)[0])
    return proba, pred

# -------------------------
# App Layout
# -------------------------
st.title("🩺 Multiple Disease Prediction System")
st.write("🏥Predict risk of *Kidney, **Liver, and **Parkinson’s* diseases using trained ML models.")

# Sidebar to select disease
disease_choice = st.sidebar.radio("Select Disease", ["Kidney Disease", "Liver Disease", "Parkinson's Disease"])
model_path = MODEL_PATHS.get(disease_choice)

if not os.path.exists(model_path):
    st.error(f"Model file not found for {disease_choice}: {model_path}")
    st.stop()

model = load_model(model_path)

# Render form
if disease_choice == "Kidney Disease":
    user_input = kidney_form()
elif disease_choice == "Liver Disease":
    user_input = liver_form()
else:
    user_input = parkinsons_form()

if st.button("🔍 Predict"):
    try:
        proba, pred = predict(model, user_input)
        st.metric("Prediction Probability (Positive)", f"{proba:.3f}")
        st.write("Predicted Class:", "Disease" if pred == 1 else "No Disease")
        if proba > 0.75:
            risk = "High"
        elif proba > 0.4:
            risk = "Moderate"
        else:
            risk = "Low"
        st.success(f"Risk Level: {risk}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")