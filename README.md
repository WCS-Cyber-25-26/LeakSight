# LeakSight

LeakSight is an educational side-channel security project that simulates power leakage from cryptographic operations, trains a machine learning detector, and visualizes results in an interactive dashboard.

## Purpose

Modern systems can leak secrets through physical side channels such as timing, electromagnetic radiation, or power consumption. LeakSight focuses on the power-analysis case:

- It generates synthetic AES-style power traces.
- It simulates both vulnerable and mitigated implementations.
- It trains multiple models to classify traces as vulnerable or secure.
- It highlights where in time the leakage signal is strongest.

The project is built for learning and experimentation in a controlled lab setting.

## Intuition

You can think of side-channel leakage like overhearing a keyboard instead of reading the screen:

- The intended output is hidden.
- The physical behavior (sound, power, timing) still reveals clues.

In LeakSight, each trace is a vector of 100 power samples. Vulnerable traces contain a leakage spike around the S-Box processing window.

## AES Vulnerable vs Secure Comparison

This project compares two AES implementation conditions from a side-channel perspective.

### Vulnerable AES (Unprotected)

- Mitigation mode: `none`
- Leakage behavior: secret-dependent intermediate values are directly reflected in power consumption.
- Trace shape: clearer, structured spike around the target operation window.
- Security impact: attacker can correlate power measurements with processed secret-dependent values.
- Label in dataset: `1` (Vulnerable)

### Secure AES (Mitigated)

Secure traces come from two mitigation strategies:

1. Masking
- Mitigation mode: `masking`
- Leakage behavior: secret correlation is removed in the simulator.
- Trace shape: flatter around leakage window, mostly baseline noise.

2. Noise Injection
- Mitigation mode: `noise_injection`
- Leakage behavior: leakage is obscured by amplified random noise.
- Trace shape: noisier signal where leak patterns are harder to isolate.

Both secure variants are labeled as:

- Label in dataset: `0` (Secure/Mitigated)

### Why The Graphs Look Different

- Vulnerable traces preserve a learnable leak signal near the S-Box timing window.
- Secure traces either remove that signal (masking) or drown it out (noise injection).

This difference is exactly what the models learn to classify in LeakSight.

## Core Concepts Used

### 1) Leakage Injection in Traces

Synthetic traces are generated in [src/data/generate_traces.py](src/data/generate_traces.py). The vulnerable class embeds leakage near the middle of the trace, where secret-dependent processing occurs.

### 2) Hamming Weight Leakage Model

Leakage magnitude is linked to the Hamming weight (number of 1-bits) of intermediate values. This reflects the common physical assumption that switching activity correlates with power draw.

### 3) Mitigations Simulated

LeakSight models two defenses:

- Masking: removes direct secret correlation from intermediates.
- Noise injection: amplifies noise to bury leakage signal.

Both are labeled as secure in the dataset.

### 4) Multi-Model Detection

LeakSight trains and evaluates multiple detectors in [src/models/train.py](src/models/train.py):

- Random Forest (scikit-learn)
- Logistic Regression (scikit-learn)
- MLP (scikit-learn)
- 1D CNN (PyTorch)

This allows side-by-side model comparison in the dashboard.

### 5) Feature Importance for Localization

After training, feature importances indicate which time points are most useful for classification. The Streamlit app uses these to mark likely leakage regions.

## End-to-End Workflow

### Step 1: Generate Data

Script: [src/data/generate_traces.py](src/data/generate_traces.py)

- Creates vulnerable, masked, and noise-injected traces.
- Stores arrays like traces, labels, plaintexts, and key.

### Step 2: Train Models

Script: [src/models/train.py](src/models/train.py)

- Loads traces and labels.
- Performs stratified train/test split.
- Trains Random Forest, Logistic Regression, MLP, and 1D CNN.
- Saves:
	- Random Forest: `random_forest_leak_classifier.pkl`, `random_forest_feature_importances.npy`
	- Logistic Regression: `logistic_regression_leak_classifier.pkl`, `logistic_regression_feature_importances.npy`
	- MLP: `mlp_leak_classifier.pkl`, `mlp_feature_importances.npy`
	- 1D CNN: `cnn1d_leak_classifier.pt`, `cnn1d_feature_importances.npy`

Legacy compatibility outputs are also written for existing scripts:

- `leak_classifier.pkl`
- `feature_importances.npy`

### Step 3: Blind Verification

Script: [src/models/verify.py](src/models/verify.py)

- Loads saved model(s).
- Generates fresh unseen traces.
- Reports accuracy and confusion details.

Supported verification targets:

- Random Forest
- Logistic Regression
- MLP
- 1D CNN

Example output is available in [verify_output.txt](verify_output.txt).

### Step 4: Interactive Visualization

App: [src/app.py](src/app.py)

- Lets the user sample vulnerable vs secure traces.
- Lets the user choose a model from a dropdown (Random Forest, Logistic Regression, MLP, 1D CNN).
- Shows model prediction and confidence.
- Highlights high-importance leak regions on the plotted trace.

## Repository Status

Implemented and operational:

- Synthetic data generation pipeline.
- Multi-model training and comparison-ready artifacts.
- Verification script.
- Streamlit dashboard.

Scaffolded/placeholders (not fully implemented yet):

- [src/data/loader.py](src/data/loader.py)
- [src/data/preprocessing.py](src/data/preprocessing.py)
- [src/models/architectures.py](src/models/architectures.py)
- [src/training/train.py](src/training/train.py)
- [src/evaluation/metrics.py](src/evaluation/metrics.py)
- [src/visualization/plots.py](src/visualization/plots.py)

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate traces:

```bash
python src/data/generate_traces.py --num_traces 5000 --output_dir data/raw
```

Train models:

```bash
python src/models/train.py --data_dir data/raw --output_dir models
```

Run verification:

```bash
python src/models/verify.py
```

Verify a specific model:

```bash
python src/models/verify.py --model random_forest
python src/models/verify.py --model logistic_regression
python src/models/verify.py --model mlp
python src/models/verify.py --model cnn1d
```

Launch dashboard:

```bash
streamlit run src/app.py
```

## Educational Scope

LeakSight is intended for security education, experimentation, and mitigation awareness. It is not a replacement for certified hardware security validation processes.
