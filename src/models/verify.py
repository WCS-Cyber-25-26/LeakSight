import argparse
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

MODEL_CONFIGS = {
    "random_forest": {
        "type": "sklearn",
        "path": os.path.join(MODELS_DIR, "random_forest_leak_classifier.pkl"),
        "legacy_path": os.path.join(MODELS_DIR, "leak_classifier.pkl"),
    },
    "logistic_regression": {
        "type": "sklearn",
        "path": os.path.join(MODELS_DIR, "logistic_regression_leak_classifier.pkl"),
    },
    "mlp": {
        "type": "sklearn",
        "path": os.path.join(MODELS_DIR, "mlp_leak_classifier.pkl"),
    },
    "cnn1d": {
        "type": "cnn",
        "path": os.path.join(MODELS_DIR, "cnn1d_leak_classifier.pt"),
    },
}


def generate_blind_test_set(num_per_class=500):
    from data.generate_traces import generate_traces

    print(f"\nGenerating {num_per_class} NEW vulnerable traces and {num_per_class} NEW secure traces for blind testing...")
    new_traces_vuln, _, _, new_labels_vuln = generate_traces(num_per_class, 100, 1.0, "none")
    new_traces_sec, _, _, new_labels_sec = generate_traces(num_per_class, 100, 1.0, "masking")

    X_blind_test = np.vstack([new_traces_vuln, new_traces_sec])
    y_blind_test = np.concatenate([new_labels_vuln, new_labels_sec])
    print(f"Created blind test set of {X_blind_test.shape[0]} traces.")
    return X_blind_test, y_blind_test


def load_model(model_name):
    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]

    if model_name == "random_forest" and not os.path.exists(model_path) and os.path.exists(config["legacy_path"]):
        model_path = config["legacy_path"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if config["type"] == "sklearn":
        with open(model_path, "rb") as f:
            return pickle.load(f)

    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"PyTorch is required for cnn1d verification: {e}")

    from models.cnn_architecture import CNN1DLeakDetector

    model = CNN1DLeakDetector(input_length=100)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_labels(model_name, model, X):
    if MODEL_CONFIGS[model_name]["type"] == "sklearn":
        return model.predict(X)

    import torch

    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        probs = torch.sigmoid(model(x)).cpu().numpy().flatten()
    return (probs >= 0.5).astype(np.int32)


def verify_model(model_name, X_blind_test, y_blind_test):
    pretty_name = model_name.replace("_", " ").upper()
    print(f"\nLoading model: {pretty_name}")
    model = load_model(model_name)
    print("Model loaded successfully.")

    print("Running verification inference...")
    y_pred = predict_labels(model_name, model, X_blind_test)
    acc = accuracy_score(y_blind_test, y_pred)

    print("\n" + "=" * 60)
    print(f"{pretty_name} BLIND TEST ACCURACY: {acc * 100:.2f}%")
    print("=" * 60)
    print("\nDetailed Verification Report:")
    print(classification_report(y_blind_test, y_pred, target_names=["Secure/Mitigated (0)", "Vulnerable (1)"]))

    cm = confusion_matrix(y_blind_test, y_pred)
    print("Confusion Matrix:")
    print(f"True Secure classified as Secure:       {cm[0][0]}  (Correct)")
    print(f"True Secure classified as Vulnerable:   {cm[0][1]}  (False Positive)")
    print(f"True Vulnerable classified as Secure:   {cm[1][0]}  (False Negative)")
    print(f"True Vulnerable classified as Vulnerable: {cm[1][1]}  (Correct)")

    if acc > 0.95:
        print("✅ VERIFICATION PASSED")
    else:
        print("❌ VERIFICATION BELOW TARGET (95%)")


def parse_args():
    parser = argparse.ArgumentParser(description="Verify one or all trained leak-detection models.")
    parser.add_argument(
        "--model",
        choices=["random_forest", "logistic_regression", "mlp", "cnn1d", "all"],
        default="all",
        help="Which model to verify.",
    )
    parser.add_argument("--num_per_class", type=int, default=500, help="Number of vulnerable and secure traces to generate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    X_blind_test, y_blind_test = generate_blind_test_set(num_per_class=args.num_per_class)

    models_to_run = [args.model] if args.model != "all" else ["random_forest", "logistic_regression", "mlp", "cnn1d"]
    for model_name in models_to_run:
        try:
            verify_model(model_name, X_blind_test, y_blind_test)
        except Exception as e:
            print(f"\nSkipping {model_name}: {e}")
