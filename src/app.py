import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
import plotly.graph_objects as go

# Configure page
st.set_page_config(page_title="LeakSight Analyzer", layout="wide")

st.title("Leaksight")
st.subheader("side-channel vulnerability detector")
st.markdown("""
Welcome to **LeakSight**, an educational and diagnostic tool for analyzing physical signal data (like power consumption or execution timing) to detect and mitigate side-channel information leaks.
""")

# Load paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_CONFIGS = {
    "Random Forest": {
        "type": "sklearn",
        "model_path": os.path.join(MODELS_DIR, "random_forest_leak_classifier.pkl"),
        "importance_path": os.path.join(MODELS_DIR, "random_forest_feature_importances.npy"),
    },
    "Logistic Regression": {
        "type": "sklearn",
        "model_path": os.path.join(MODELS_DIR, "logistic_regression_leak_classifier.pkl"),
        "importance_path": os.path.join(MODELS_DIR, "logistic_regression_feature_importances.npy"),
    },
    "MLP": {
        "type": "sklearn",
        "model_path": os.path.join(MODELS_DIR, "mlp_leak_classifier.pkl"),
        "importance_path": os.path.join(MODELS_DIR, "mlp_feature_importances.npy"),
    },
    "1D CNN": {
        "type": "cnn",
        "model_path": os.path.join(MODELS_DIR, "cnn1d_leak_classifier.pt"),
        "importance_path": os.path.join(MODELS_DIR, "cnn1d_feature_importances.npy"),
    },
}

# Operational uncertainty floor: never report absolute certainty.
MIN_ERROR_PROB = 0.01


@st.cache_resource
def load_dataset():
    traces = np.load(os.path.join(DATA_DIR, "traces.npy"))
    labels = np.load(os.path.join(DATA_DIR, "labels.npy"))
    return traces, labels


def list_available_models():
    # Backward compatibility if only legacy Random Forest artifacts exist.
    if (
        os.path.exists(os.path.join(MODELS_DIR, "leak_classifier.pkl"))
        and os.path.exists(os.path.join(MODELS_DIR, "feature_importances.npy"))
        and not os.path.exists(MODEL_CONFIGS["Random Forest"]["model_path"])
    ):
        MODEL_CONFIGS["Random Forest"]["model_path"] = os.path.join(MODELS_DIR, "leak_classifier.pkl")
        MODEL_CONFIGS["Random Forest"]["importance_path"] = os.path.join(MODELS_DIR, "feature_importances.npy")

    available = []
    for model_name, config in MODEL_CONFIGS.items():
        if not os.path.exists(config["model_path"]):
            continue
        if config["type"] == "cnn":
            try:
                import torch  # noqa: F401
            except Exception:
                continue
        available.append(model_name)
    return available


@st.cache_resource
def load_model(model_name):
    config = MODEL_CONFIGS[model_name]
    if config["type"] == "sklearn":
        with open(config["model_path"], "rb") as f:
            model = pickle.load(f)

        # Compatibility: older/full model artifacts may be pickled from scikit-learn versions
        # where LogisticRegression stores `multi_class` differently. Ensure attribute exists.
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception:
            LogisticRegression = None

        def _ensure_multi_class(obj):
            if LogisticRegression is not None and isinstance(obj, LogisticRegression):
                if not hasattr(obj, "multi_class"):
                    obj.multi_class = "auto"

        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                _ensure_multi_class(step)
        else:
            _ensure_multi_class(model)

        return model

    import torch
    from models.cnn_architecture import CNN1DLeakDetector

    model = CNN1DLeakDetector(input_length=100)
    state_dict = torch.load(config["model_path"], map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_data
def load_importances(model_name):
    return np.load(MODEL_CONFIGS[model_name]["importance_path"])


def run_inference(model_name, model, trace):
    config = MODEL_CONFIGS[model_name]
    if config["type"] == "sklearn":
        prediction = int(model.predict([trace])[0])
        probs = model.predict_proba([trace])[0]
        confidence = float(probs[prediction])
        return prediction, confidence

    import torch

    with torch.no_grad():
        x = torch.tensor(trace, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        prob_vulnerable = torch.sigmoid(model(x)).item()
    prediction = 1 if prob_vulnerable >= 0.5 else 0
    confidence = prob_vulnerable if prediction == 1 else (1.0 - prob_vulnerable)
    return prediction, confidence


def apply_confidence_floor(raw_confidence):
    return float(np.clip(raw_confidence, MIN_ERROR_PROB, 1.0 - MIN_ERROR_PROB))


def build_trace_figure(trace, importances, show_annotations=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=trace, mode="lines", name="Power Trace", line=dict(color="blue")))

    threshold = np.percentile(importances, 90)
    leaky_points = np.where(importances >= threshold)[0]

    for pt in leaky_points:
        fig.add_vrect(
            x0=pt - 0.5,
            x1=pt + 0.5,
            fillcolor="red",
            opacity=0.2,
            line_width=0,
            annotation_text="Leak Source" if show_annotations and pt == leaky_points[0] else "",
        )

    fig.update_layout(xaxis_title="Time (samples)", yaxis_title="Power Consumption (mV)", margin=dict(l=0, r=0, t=10, b=0))
    return fig

try:
    traces, labels = load_dataset()
except Exception as e:
    st.error(f"Could not load data or model. Have you run the generation and training scripts? Error: {e}")
    st.stop()


# Sidebar controls
st.sidebar.header("Signal Selection")
st.sidebar.markdown("Sample a trace from the generated dataset to analyze its side-channel leakage.")

available_models = list_available_models()
if not available_models:
    st.error("No trained models found. Run src/models/train.py to create model files.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Select Detection Model", available_models)

# Distinguish indices
vuln_indices = np.where(labels == 1)[0]
sec_indices = np.where(labels == 0)[0]

sample_type = st.sidebar.radio("Select Trace Type to Simulate", ("Vulnerable (Unprotected AES)", "Secure (Masked/Noisy AES)"))


def pick_random_trace_idx(is_vulnerable):
    return int(np.random.choice(vuln_indices if is_vulnerable else sec_indices))


is_vulnerable_type = sample_type == "Vulnerable (Unprotected AES)"
expected_status = "Vulnerable" if is_vulnerable_type else "Secure"

if "selected_trace_idx" not in st.session_state:
    st.session_state.selected_trace_idx = pick_random_trace_idx(is_vulnerable_type)
    st.session_state.selected_trace_mode = expected_status

if st.session_state.selected_trace_mode != expected_status:
    st.session_state.selected_trace_idx = pick_random_trace_idx(is_vulnerable_type)
    st.session_state.selected_trace_mode = expected_status

if st.sidebar.button("Fetch Random Trace"):
    st.session_state.selected_trace_idx = pick_random_trace_idx(is_vulnerable_type)
    st.session_state.selected_trace_mode = expected_status

trace_idx = st.session_state.selected_trace_idx

selected_trace = traces[trace_idx]

tab_single, tab_compare = st.tabs(["Single Model Analysis", "Compare All Models"])

with tab_single:
    model = load_model(selected_model_name)
    importances = load_importances(selected_model_name)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Signal Visualization")
        st.write("This graph displays the simulated power consumption curve of the AES device over time. The red shaded area highlights the points the selected model identified as highly leaky.")

        fig = build_trace_figure(selected_trace, importances, show_annotations=True)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Analysis Results")
        st.write(f"Processing trace through {selected_model_name}...")

        prediction, confidence = run_inference(selected_model_name, model, selected_trace)
        confidence = apply_confidence_floor(confidence)
        pred_label = "Vulnerable" if prediction == 1 else "Secure"

        if pred_label == "Vulnerable":
            st.error(f"🚨 **DETECTED LIKELY LEAK!** ({pred_label})")
        else:
            st.success(f"✅ **NO LEAK DETECTED!** ({pred_label})")

        st.metric(label="System Confidence Level", value=f"{confidence * 100:.1f}%")

        st.markdown("---")
        st.markdown("### Mitigation Metrics")
        st.write("If you previously checked a vulnerable trace, notice how the secure trace can flatten or hide predictive side-channel peaks through masking or noise injection.")
        st.info(f"Trace ID from Dataset: `{trace_idx}`\n\nGround Truth: `{expected_status}`")

with tab_compare:
    st.subheader("Compare All Models")
    st.write("Run every available model on the same trace to compare leak-detection outcomes. All model graphs are synchronized to the trace selected from the sidebar button.")

    compare_rows = []
    vulnerable_votes = 0
    model_figures = []

    for model_name in available_models:
        compare_model = load_model(model_name)
        model_importances = load_importances(model_name)
        prediction, confidence = run_inference(model_name, compare_model, selected_trace)
        confidence = apply_confidence_floor(confidence)
        pred_label = "Vulnerable" if prediction == 1 else "Secure"
        if prediction == 1:
            vulnerable_votes += 1

        model_figures.append((model_name, build_trace_figure(selected_trace, model_importances, show_annotations=False)))

        compare_rows.append(
            {
                "Model": model_name,
                "Prediction": pred_label,
                "Confidence (%)": round(confidence * 100.0, 2),
            }
        )

    graph_cols = st.columns(2)
    for idx, (model_name, model_fig) in enumerate(model_figures):
        with graph_cols[idx % 2]:
            st.markdown(f"#### {model_name}")
            st.plotly_chart(model_fig, width="stretch")

    results_df = pd.DataFrame(compare_rows).sort_values(by="Confidence (%)", ascending=False)
    st.dataframe(results_df, width="stretch")

    total_models = len(available_models)
    secure_votes = total_models - vulnerable_votes
    consensus_label = "Vulnerable" if vulnerable_votes > secure_votes else "Secure"
    st.metric("Consensus", f"{consensus_label} ({vulnerable_votes}/{total_models} vulnerable votes)")
    st.info(f"Trace ID from Dataset: `{trace_idx}`\n\nGround Truth: `{expected_status}`")
