import numpy as np
import os
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(data_dir, output_model_path):
    print("Loading datasets...")
    traces = np.load(os.path.join(data_dir, "traces.npy"))
    labels = np.load(os.path.join(data_dir, "labels.npy"))
    
    # Stratified split to keep label proportions
    X_train, X_test, y_train, y_test = train_test_split(
        traces, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set: {X_train.shape[0]} traces")
    print(f"Testing set:  {X_test.shape[0]} traces")
    
    # We use a Random Forest as a strong baseline that is easy to interpret and fast
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Secure/Mitigated", "Vulnerable"]))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Extract feature importances (which points in time are most leaky?)
    importances = clf.feature_importances_
    np.save(os.path.join(os.path.dirname(output_model_path), "feature_importances.npy"), importances)
    
    # Save the model
    with open(output_model_path, "wb") as f:
        pickle.dump(clf, f)
        
    print(f"\nModel saved to {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/raw/")
    parser.add_argument("--output_model", type=str, default="../models/leak_classifier.pkl")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    train_model(args.data_dir, args.output_model)
