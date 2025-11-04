import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

def show_eda(df):
    st.header("📊 Exploratory Data Analysis")

    # 🔹 Clean column names (optional but helpful)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 🔹 Auto-detect target column
    label_candidates = [c for c in df.columns if "heart" in c or "target" in c or "risk" in c]
    label_col = label_candidates[0] if label_candidates else df.columns[-1]

    st.write(f"Detected target column: **{label_col}**")

    st.write("### Dataset Overview")
    st.dataframe(df.head())

    st.write("### Basic Info")
    st.write(df.describe())

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("### Feature Distributions")
    num_cols = df.select_dtypes(include='number').columns
    feature = st.selectbox("Select Feature", num_cols)

    # 🔹 Use detected label column for color
    fig2 = px.histogram(df, x=feature, color=label_col, barmode="overlay", nbins=30)
    st.plotly_chart(fig2, use_container_width=True)
