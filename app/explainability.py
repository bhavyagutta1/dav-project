import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def explain_model(model, X_test, num_samples=100):
    st.header("💡 Model Explainability (Feature Importance)")

    # Limit the sample size for speed (optional)
    X_sample = X_test[:min(num_samples, len(X_test))]

    # --- Feature Importance from the model ---
    st.subheader("Feature Importance (From Random Forest)")

    # Extract feature importances
    try:
        importances = model.feature_importances_
        feature_names = X_sample.columns
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Bar Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
        ax.set_title("Feature Importance Based on Model Weights")
        st.pyplot(fig)

        # Table View
        st.write("### Top 10 Most Important Features")
        st.dataframe(importance_df.head(10))

    except Exception as e:
        st.error(f"⚠️ Unable to display feature importance: {e}")

    # --- Correlation Matrix for additional explainability ---
    st.subheader("Feature Correlation Heatmap")
    corr = X_sample.corr()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, cmap='coolwarm', annot=False, ax=ax2)
    st.pyplot(fig2)

    st.success("✅ Feature importance and correlations generated successfully.")
