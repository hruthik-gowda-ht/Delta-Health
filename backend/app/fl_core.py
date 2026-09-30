"""
fl_core.py - the actual delta algorithm, now reading from the database
instead of CSV files.

Invariant to preserve: train_local_from_db() reads Patient rows for ONE
clinic and returns only a delta vector + summary stats. No patient row,
and no per-patient value, is ever returned from this module to a caller
outside the owning clinic's own request.
"""
import json
import numpy as np
from sqlalchemy.orm import Session

from . import models

FEATURES = ["age", "bmi", "resting_hr", "fever_days", "cough_severity",
            "fatigue_score", "existing_condition", "bp_systolic"]

# Reference normalization ranges -- these are fixed clinical-guideline-style
# constants, not derived from any single clinic's private data, so every
# clinic can normalize identically without sharing statistics.
REFERENCE_MEAN = [45, 26, 75, 1.5, 1.5, 4.5, 0.25, 122]
REFERENCE_STD = [18, 6, 12, 2, 1.2, 3, 0.4, 16]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def normalize(X, mean, std):
    return (X - np.array(mean)) / (np.array(std) + 1e-8)


def blank_weights():
    return [0.0] * (len(FEATURES) + 1)  # +1 bias


FEATURE_LABELS = {
    "age": "Age",
    "bmi": "BMI",
    "resting_hr": "Resting heart rate",
    "fever_days": "Fever duration",
    "cough_severity": "Cough severity",
    "fatigue_score": "Fatigue score",
    "existing_condition": "Existing condition",
    "bp_systolic": "Blood pressure (systolic)",
}


def predict_single(weights, mean, std, feature_values: dict):
    """Runs ONE patient's values through the current global model and
    decomposes the prediction into a per-feature contribution, so a doctor
    can see WHY the AI flagged (or didn't flag) the patient -- not just a
    black-box number. Because the model is logistic regression, this
    decomposition is exact, not an approximation like SHAP/LIME would need
    for a nonlinear model.

    Returns (probability, [(feature_name, value, contribution), ...] sorted
    by |contribution| descending).
    """
    x = np.array([feature_values[f] for f in FEATURES], dtype=float)
    xn = (x - np.array(mean)) / (np.array(std) + 1e-8)
    w = np.array(weights, dtype=float)
    bias = w[-1]
    per_feature_contribution = xn * w[:-1]  # each feature's push on the log-odds
    z = float(per_feature_contribution.sum() + bias)
    prob = float(sigmoid(z))

    ranked = sorted(
        zip(FEATURES, x, per_feature_contribution),
        key=lambda t: -abs(t[2]),
    )
    return prob, ranked



def patients_to_arrays(patients):
    X = np.array([[getattr(p, f) for f in FEATURES] for p in patients], dtype=float)
    y = np.array([p.at_risk for p in patients], dtype=float)
    return X, y


def train_local_from_db(db: Session, clinic_id: int, global_weights, mean, std,
                         epochs=25, lr=0.3, min_samples=5):
    """STEP 1+2 of the pipeline. Trains on this clinic's Patient rows only,
    returns (delta, local_accuracy, n_samples). Raises ValueError if the
    clinic doesn't have enough data yet to train responsibly."""
    patients = db.query(models.Patient).filter(models.Patient.clinic_id == clinic_id).all()
    if len(patients) < min_samples:
        raise ValueError(f"need at least {min_samples} patient records to train (have {len(patients)})")

    X, y = patients_to_arrays(patients)
    Xn = normalize(X, mean, std)
    Xb = np.hstack([Xn, np.ones((len(Xn), 1))])

    w = np.array(global_weights, dtype=float).copy()
    n = len(y)
    for _ in range(epochs):
        preds = sigmoid(Xb @ w)
        grad = Xb.T @ (preds - y) / n
        w -= lr * grad

    delta = w - np.array(global_weights, dtype=float)
    local_acc = float(((sigmoid(Xb @ w) > 0.5) == y).mean())
    return delta, local_acc, n


def add_dp_noise(delta: np.ndarray, epsilon: float = None) -> np.ndarray:
    """Optional differential-privacy style protection: adds Gaussian noise
    to the delta before transmission so the aggregator (or anyone who
    intercepts it) can't precisely reverse-engineer this clinic's local
    update. epsilon=None disables noise."""
    if epsilon is None:
        return delta
    scale = 1.0 / epsilon
    noise = np.random.normal(0, scale * 0.02, size=delta.shape)
    return delta + noise


def clip_outlier_deltas(deltas, max_norm_factor=3.0):
    """Safety check before aggregation: clip any delta whose magnitude is a
    wild outlier relative to the others, so one buggy or malicious clinic
    can't poison the global model."""
    norms = [float(np.linalg.norm(d)) for d in deltas]
    if not norms:
        return deltas
    median_norm = float(np.median(norms))
    cap = max(median_norm * max_norm_factor, 1e-6)
    return [d * (cap / n) if n > cap else d for d, n in zip(deltas, norms)]


def federated_average(deltas, sample_counts):
    total = sum(sample_counts)
    agg = np.zeros_like(deltas[0])
    for d, n in zip(deltas, sample_counts):
        agg += d * (n / total)
    return agg


def evaluate_weights(weights, mean, std, X_holdout, y_holdout):
    Xn = normalize(X_holdout, mean, std)
    Xb = np.hstack([Xn, np.ones((len(Xn), 1))])
    preds = sigmoid(Xb @ np.array(weights))
    return float(((preds > 0.5) == y_holdout).mean())


def get_or_create_model_state(db: Session) -> models.ModelState:
    state = db.query(models.ModelState).filter(models.ModelState.id == 1).first()
    if state is None:
        state = models.ModelState(
            id=1,
            weights_json=json.dumps(blank_weights()),
            mean_json=json.dumps(REFERENCE_MEAN),
            std_json=json.dumps(REFERENCE_STD),
            round=0,
        )
        db.add(state)
        db.commit()
        db.refresh(state)
    return state


def make_synthetic_holdout(n=400, seed=999):
    """A city-wide reference test set, standing in for a benchmark dataset a
    real deployment would maintain centrally (e.g. published by a health
    authority) purely to report global model accuracy -- never used for
    training, never tied to any one clinic's real patients."""
    rng = np.random.default_rng(seed)
    age = rng.normal(45, 16, n).clip(1, 95)
    bmi = rng.normal(26, 5.5, n).clip(14, 50)
    resting_hr = rng.normal(75, 11, n).clip(45, 130)
    fever_days = rng.poisson(1.5, n).clip(0, 14)
    cough_severity = rng.integers(0, 4, n)
    fatigue_score = rng.integers(0, 10, n)
    existing_condition = rng.binomial(1, 0.25, n)
    bp_systolic = rng.normal(122, 15, n).clip(80, 200)

    X = np.column_stack([age, bmi, resting_hr, fever_days, cough_severity,
                          fatigue_score, existing_condition, bp_systolic])
    logit = (0.04*(age-45) + 0.08*(bmi-25) + 0.03*(resting_hr-75) + 0.35*fever_days
             + 0.5*cough_severity + 0.15*fatigue_score + 1.1*existing_condition
             + 0.02*(bp_systolic-120) - 2.5)
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob)
    return X, y


HOLDOUT_X, HOLDOUT_Y = make_synthetic_holdout()
