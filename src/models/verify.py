import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def verify_model():
    """
    Loads the saved model and runs prediction on a fresh batch of synthetic data
    that it has never seen before to prove it does not overfit and gives correct results.
    """
    # Compute absolute path directly from root LeakSight directory assumed to be CWD or 1-level up
    base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    DATA_DIR = os.path.join(base_dir, "data", "raw")
    MODEL_PATH = os.path.join(base_dir, "models", "leak_classifier.pkl")
    # 1. Load the existing model
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    print("Model loaded successfully.")
    
    # 2. To absolutely guarantee we aren't testing on training data,
    # let's generate 1000 BRAND NEW test traces right now on the fly.
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    from generate_traces import generate_traces
    
    print("\nGenerating 500 NEW vulnerable traces and 500 NEW secure traces for blind testing...")
    
    new_traces_vuln, _, _, new_labels_vuln = generate_traces(500, 100, 1.0, "none")
    new_traces_sec, _, _, new_labels_sec = generate_traces(500, 100, 1.0, "masking")

    X_blind_test = np.vstack([new_traces_vuln, new_traces_sec])
    y_blind_test = np.concatenate([new_labels_vuln, new_labels_sec])
    
    print(f"Created blind test set of {X_blind_test.shape[0]} traces.")
    
    # 3. Predict and verify 
    print("\nRunning Verification Inference...")
    y_pred = model.predict(X_blind_test)
    
    acc = accuracy_score(y_blind_test, y_pred)
    
    print("\n" + "="*50)
    print(f"BLIND TEST ACCURACY: {acc * 100:.2f}%")
    print("="*50)
    
    print("\nDetailed Verification Report:")
    print(classification_report(y_blind_test, y_pred, target_names=["Secure/Mitigated (0)", "Vulnerable (1)"]))
    
    cm = confusion_matrix(y_blind_test, y_pred)
    print("\nConfusion Matrix (How many wrong predictions?):")
    print(f"True Secure classified as Secure:       {cm[0][0]}  (Correct)")
    print(f"True Secure classified as Vulnerable:   {cm[0][1]}  (False Positive - WRONG)")
    print(f"True Vulnerable classified as Secure:   {cm[1][0]}  (False Negative - WRONG)")
    print(f"True Vulnerable classified as Vuln:     {cm[1][1]}  (Correct)")
    
    if acc > 0.95:
        print("\n✅ VERIFICATION PASSED: The model is highly accurate and robust.")
    else:
        print("\n❌ VERIFICATION FAILED: The model is providing too many wrong predictions or overfitting.")

if __name__ == "__main__":
    verify_model()
