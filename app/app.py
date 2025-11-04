import streamlit as st
import pandas as pd
from model import train_model
from eda import show_eda
from explainability import explain_model

st.set_page_config(page_title="Healthcare Risk Prediction", layout="wide")

st.title("🏥 Predictive Healthcare Risk Stratification")
st.markdown("Analyze patient data, predict risk levels, and visualize feature importance.")

# Load dataset
data = pd.read_csv("data/dataset_heart.csv")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["📊 EDA", "🤖 Model Training", "💡 Explainability"])

if page == "📊 EDA":
    show_eda(data)

elif page == "🤖 Model Training":
    st.header("Model Training and Evaluation")
    model, scaler, acc, report, X_test, y_test = train_model(data)
    st.success(f"Model trained successfully with accuracy: **{acc:.2f}**")
    st.write(pd.DataFrame(report).T)
    st.session_state["model"] = model
    st.session_state["X_test"] = X_test

elif page == "💡 Explainability":
    if "model" in st.session_state:
        explain_model(st.session_state["model"], st.session_state["X_test"])
    else:
        st.warning("Please train the model first.")
