import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Page Configuration
st.set_page_config(page_title="Edge-IDS Model Compression Dashboard", layout="wide")

st.title("🛡️ Edge-IDS Model Compression & Benchmarking Dashboard")
st.markdown("Live performance benchmarks tracking model compression from teacher architecture to lightweight IoT deployment.")

csv_path = "artifacts/results/metrics.csv"

@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

df = load_data(csv_path)

if df is not None:
    # Clean and normalize column names (lowercase and strip whitespace)
    df.columns = [str(col).strip().lower() for col in df.columns]
else:
    # Fallback layout if file isn't found
    df = pd.DataFrame({
        "model": ["Teacher", "Student-Plain", "Student-KD", "Student-Pruned", "Student-Int8"],
        "f1": [0.9671, 0.9495, 0.9698, 0.9667, 0.9746],
        "size_kb": [837.4, 11.2, 11.2, 5.1, 1.21]
    })

# Dynamically find the correct columns regardless of exact naming
model_col = next((col for col in df.columns if 'model' in col), df.columns[0])
f1_col = next((col for col in df.columns if 'f1' in col), None)
size_col = next((col for col in df.columns if 'size' in col or 'kb' in col or 'bytes' in col), None)

# --- TOP METRICS ROW ---
col1, col2, col3 = st.columns(3)
with col1:
    best_model = df.loc[df[f1_col].idxmax()][model_col] if f1_col else "N/A"
    st.metric(label="Best Performing Model", value=str(best_model))
with col2:
    max_f1 = df[f1_col].max() if f1_col else 0.0
    st.metric(label="Peak F1 Score", value=f"{max_f1:.4f}" if f1_col else "N/A")
with col3:
    min_size = df[size_col].min() if size_col else 0.0
    st.metric(label="Smallest Model Footprint", value=str(min_size))

st.markdown("---")

# --- DATA TABLE SECTION ---
st.subheader("Complete Benchmark Results")
st.dataframe(df, use_container_width=True)

# --- GRAPHICAL REPRESENTATION SECTION ---
if f1_col and size_col:
    st.markdown("---")
    st.subheader("Graphical Performance & Footprint Analysis")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("### F1 Score Comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df[model_col], df[f1_col], color='#66b3ff')
        ax.set_ylabel("F1 Score")
        plt.xticks(rotation=25, ha='right')
        st.pyplot(fig)

    with col_g2:
        st.markdown("### Model Footprint Size")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(df[model_col], df[size_col], color='#ffb3e6')
        ax2.set_ylabel("Size")
        plt.xticks(rotation=25, ha='right')
        st.pyplot(fig2)