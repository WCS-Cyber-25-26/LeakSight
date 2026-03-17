import streamlit as st
import numpy as np
import pickle
import os
import plotly.graph_objects as go

# Configure page
st.set_page_config(page_title="LeakSight Analyzer", layout="wide")

st.title("🛡️ LeakSight: Side-Channel Vulnerability Detector")
st.markdown("""
Welcome to **LeakSight**, an educational and diagnostic tool for analyzing physical signal data (like power consumption or execution timing) to detect and mitigate side-channel information leaks.
""")

# Load paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "leak_classifier.pkl"))
IMPORTANCE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "feature_importances.npy"))


@st.cache_resource
def load_assets():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    traces = np.load(os.path.join(DATA_DIR, "traces.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    importances = np.load(IMPORTANCE_PATH)
    return model, traces, labels, importances

try:
    model, traces, labels, importances = load_assets()
except Exception as e:
    st.error(f"Could not load data or model. Have you run the generation and training scripts? Error: {e}")
    st.stop()


# Sidebar controls
st.sidebar.header("Signal Selection")
st.sidebar.markdown("Sample a trace from the generated dataset to analyze its side-channel leakage.")

# Distinguish indices
vuln_indices = np.where(labels == 1)[0]
sec_indices = np.where(labels == 0)[0]

sample_type = st.sidebar.radio("Select Trace Type to Simulate", ("Vulnerable (Unprotected AES)", "Secure (Masked/Noisy AES)"))

if st.sidebar.button("Fetch Random Trace"):
    # Force a rerun when button clicked to get new random choice
    pass

if sample_type == "Vulnerable (Unprotected AES)":
    trace_idx = np.random.choice(vuln_indices)
    expected_status = "Vulnerable"
else:
    trace_idx = np.random.choice(sec_indices)
    expected_status = "Secure"

selected_trace = traces[trace_idx]

# --- Main Dashboard ---

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Signal Visualization")
    st.write("This graph displays the simulated power consumption curve of the AES device over time. The red shaded area highlights the points the Machine Learning model identified as 'highly leaky' during its training phase.")
    
    # Plotly figure
    fig = go.Figure()
    
    # Add actual trace
    fig.add_trace(go.Scatter(y=selected_trace, mode='lines', name='Power Trace', line=dict(color='blue')))
    
    # Highlight top 10 leaky points 
    threshold = np.percentile(importances, 90) # top 10%
    leaky_points = np.where(importances >= threshold)[0]
    
    # Shade vertical regions for leaky points
    for pt in leaky_points:
        fig.add_vrect(x0=pt-0.5, x1=pt+0.5, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Leak Source" if pt == leaky_points[0] else "")
        
    fig.update_layout(xaxis_title="Time (samples)", yaxis_title="Power Consumption (mV)", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Analysis Results")
    st.write("Processing trace through Random Forest Classifier...")
    
    # Make Prediction
    prediction = model.predict([selected_trace])[0]
    probs = model.predict_proba([selected_trace])[0]
    
    pred_label = "Vulnerable" if prediction == 1 else "Secure"
    confidence = probs[1] if prediction == 1 else probs[0]
    
    if pred_label == "Vulnerable":
        st.error(f"🚨 **DETECTED LIKELY LEAK!** ({pred_label})")
    else:
        st.success(f"✅ **NO LEAK DETECTED!** ({pred_label})")
        
    st.metric(label="System Confidence Level", value=f"{confidence * 100:.1f}%")
    
    st.markdown("---")
    st.markdown("### Mitigation Metrics")
    st.write("If you previously checked a vulnerable trace, notice how the secure trace successfully flattens the predictive side-channel peaks through techniques like masking or continuous noise injection. This forces the model confidence to report 'Secure'.")
    
    st.info(f"Trace ID from Dataset: `{trace_idx}`\n\nGround Truth: `{expected_status}`")
