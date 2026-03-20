import argparse
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_model(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{name} Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Secure/Mitigated", "Vulnerable"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def train_random_forest(X_train, y_train, X_test, y_test, output_dir):
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model("Random Forest", y_test, y_pred)

    model_path = os.path.join(output_dir, "random_forest_leak_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    importances = model.feature_importances_
    np.save(os.path.join(output_dir, "random_forest_feature_importances.npy"), importances)

    # Keep legacy filenames for backward compatibility.
    with open(os.path.join(output_dir, "leak_classifier.pkl"), "wb") as f:
        pickle.dump(model, f)
    np.save(os.path.join(output_dir, "feature_importances.npy"), importances)
    print(f"Saved: {model_path}")


def train_logistic_regression(X_train, y_train, X_test, y_test, output_dir):
    print("\nTraining Logistic Regression...")
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model("Logistic Regression", y_test, y_pred)

    model_path = os.path.join(output_dir, "logistic_regression_leak_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    importances = np.abs(model.named_steps["clf"].coef_[0])
    np.save(os.path.join(output_dir, "logistic_regression_feature_importances.npy"), importances)
    print(f"Saved: {model_path}")


def train_mlp(X_train, y_train, X_test, y_test, output_dir):
    print("\nTraining MLP Classifier...")
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    max_iter=300,
                    random_state=42,
                    early_stopping=True,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model("MLP", y_test, y_pred)

    model_path = os.path.join(output_dir, "mlp_leak_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    first_layer_weights = model.named_steps["clf"].coefs_[0]
    importances = np.mean(np.abs(first_layer_weights), axis=1)
    np.save(os.path.join(output_dir, "mlp_feature_importances.npy"), importances)
    print(f"Saved: {model_path}")


def train_cnn1d(X_train, y_train, X_test, y_test, output_dir, epochs=8, batch_size=256):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as e:
        print(f"\nSkipping 1D CNN training: PyTorch is not available ({e}).")
        return

    from cnn_architecture import CNN1DLeakDetector

    print("\nTraining 1D CNN...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = CNN1DLeakDetector(input_length=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        y_pred = (probs >= 0.5).astype(np.int32)

    evaluate_model("1D CNN", y_test, y_pred)

    model_path = os.path.join(output_dir, "cnn1d_leak_classifier.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")

    # Input saliency as approximate feature importance.
    saliency_base = torch.tensor(X_train[:512], dtype=torch.float32, device=device)
    saliency_base.requires_grad_(True)
    saliency_input = saliency_base.unsqueeze(1)
    saliency_logits = model(saliency_input)
    saliency_score = torch.sigmoid(saliency_logits).mean()
    saliency_score.backward()
    importances = saliency_base.grad.abs().mean(dim=0).detach().cpu().numpy()
    np.save(os.path.join(output_dir, "cnn1d_feature_importances.npy"), importances)


def train_models(data_dir, output_dir):
    print("Loading datasets...")
    traces = np.load(os.path.join(data_dir, "traces.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))

    X_train, X_test, y_train, y_test = train_test_split(
        traces, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training set: {X_train.shape[0]} traces")
    print(f"Testing set:  {X_test.shape[0]} traces")

    os.makedirs(output_dir, exist_ok=True)

    train_random_forest(X_train, y_train, X_test, y_test, output_dir)
    train_logistic_regression(X_train, y_train, X_test, y_test, output_dir)
    train_mlp(X_train, y_train, X_test, y_test, output_dir)
    train_cnn1d(X_train, y_train, X_test, y_test, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/raw/")
    parser.add_argument("--output_dir", type=str, default="../models/")
    args = parser.parse_args()

    train_models(args.data_dir, args.output_dir)
