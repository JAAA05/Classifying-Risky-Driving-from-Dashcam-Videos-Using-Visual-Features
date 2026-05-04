import os
import glob
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    balanced_accuracy_score,
    make_scorer,
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================

BASE_DIR = "data/processed/2026--04-20_split"
MODEL_DIR = "src/models"
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.joblib")
META_PATH = os.path.join(MODEL_DIR, "classifier_meta.joblib")

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Thresholds to test for Reckless probability
THRESHOLDS = np.arange(0.20, 0.71, 0.05)


# =========================
# FEATURE EXTRACTION
# =========================

def safe_numeric_col(df, col, default=0.0):
    """
    Safely read a numeric column.
    If the column does not exist, create a default column.
    """
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).fillna(default)
        return s

    return pd.Series(default, index=df.index, dtype=float)


def q(series, quantile):
    """
    Safe quantile helper.
    """
    if len(series) == 0:
        return 0.0
    return float(series.quantile(quantile))


def extract_video_features(df):
    """
    Aggregate frame-level features into one video-level feature vector.
    """

    ttc = safe_numeric_col(df, "min_ttc_smooth", default=10.0)
    rel_vel = safe_numeric_col(df, "max_rel_vel", default=0.0)
    lane_dep = safe_numeric_col(df, "lane_departure_smooth", default=0.0).abs()

    has_lane = safe_numeric_col(df, "has_lane", default=0.0)
    has_objects = safe_numeric_col(df, "has_objects", default=0.0)

    features = {}

    # TTC features
    features["ttc_mean"] = ttc.mean()
    features["ttc_min"] = ttc.min()
    features["ttc_std"] = ttc.std(ddof=0)
    features["ttc_p10"] = q(ttc, 0.10)
    features["ttc_p25"] = q(ttc, 0.25)
    features["low_ttc_1s_fraction"] = (ttc < 1.0).mean()
    features["low_ttc_1_5s_fraction"] = (ttc < 1.5).mean()
    features["low_ttc_2s_fraction"] = (ttc < 2.0).mean()

    # Relative velocity features
    features["rel_vel_mean"] = rel_vel.mean()
    features["rel_vel_max"] = rel_vel.max()
    features["rel_vel_std"] = rel_vel.std(ddof=0)
    features["rel_vel_p90"] = q(rel_vel, 0.90)
    features["rel_vel_p95"] = q(rel_vel, 0.95)

    # Lane departure features
    features["lane_dep_mean"] = lane_dep.mean()
    features["lane_dep_max"] = lane_dep.max()
    features["lane_dep_std"] = lane_dep.std(ddof=0)
    features["lane_dep_p90"] = q(lane_dep, 0.90)
    features["lane_dep_p95"] = q(lane_dep, 0.95)
    features["large_lane_dep_fraction"] = (lane_dep > 0.15).mean()
    features["very_large_lane_dep_fraction"] = (lane_dep > 0.25).mean()

    # Detection reliability
    features["has_lane_fraction"] = has_lane.mean()
    features["missing_lane_fraction"] = 1.0 - has_lane.mean()
    features["has_objects_fraction"] = has_objects.mean()
    features["missing_objects_fraction"] = 1.0 - has_objects.mean()

    # Heuristic "why" labels
    if "why" in df.columns:
        why_counts = df["why"].fillna("none").value_counts(normalize=True).to_dict()
    else:
        why_counts = {}

    features["why_tailgating_fraction"] = why_counts.get("tailgating", 0.0)
    features["why_lane_departure_fraction"] = why_counts.get("lane_departure", 0.0)
    features["why_aggressive_closing_fraction"] = why_counts.get("aggressive_closing", 0.0)

    return features


def load_dataset(base_dir):
    data = []
    labels = []
    file_paths = []

    categories = ["normal", "reckless"]

    for category in categories:
        pattern = os.path.join(base_dir, category, "*.csv")
        files = sorted(glob.glob(pattern))

        print(f"Loading {len(files)} files for category: {category}")

        for f in files:
            try:
                df = pd.read_csv(f)

                if df.empty:
                    continue

                features = extract_video_features(df)

                data.append(features)
                labels.append(1 if category == "reckless" else 0)
                file_paths.append(f)

            except Exception as e:
                print(f"Skipping {f}: {e}")

    X = pd.DataFrame(data)
    y = np.array(labels)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y, np.array(file_paths)


# =========================
# THRESHOLD TUNING
# =========================

def get_oof_probabilities(model, X_train, y_train):
    """
    Get out-of-fold probabilities on training data.
    This avoids choosing threshold directly on the test set.
    """
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    oof_probs = np.zeros(len(y_train))

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train[train_idx]

        m = clone(model)

        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_tr
        )

        m.fit(X_tr, y_tr, sample_weight=sample_weights)
        oof_probs[val_idx] = m.predict_proba(X_val)[:, 1]

    return oof_probs


def find_best_threshold(y_true, probs):
    """
    Choose threshold that maximizes F2-score for Reckless.
    F2 gives more importance to recall than precision.
    """
    rows = []

    for threshold in THRESHOLDS:
        y_pred = (probs >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        rows.append({
            "threshold": threshold,
            "precision_reckless": precision,
            "recall_reckless": recall,
            "f1_reckless": f1,
            "f2_reckless": f2,
            "balanced_accuracy": bal_acc
        })

    results = pd.DataFrame(rows)

    print("\nThreshold Search Results:")
    print(results.round(3).to_string(index=False))

    best_row = results.sort_values(
        by=["f2_reckless", "f1_reckless", "balanced_accuracy"],
        ascending=False
    ).iloc[0]

    best_threshold = float(best_row["threshold"])

    print(f"\nBest threshold selected: {best_threshold:.2f}")
    print(
        f"Recall Reckless: {best_row['recall_reckless']:.3f} | "
        f"F1 Reckless: {best_row['f1_reckless']:.3f} | "
        f"F2 Reckless: {best_row['f2_reckless']:.3f}"
    )

    return best_threshold, results


# =========================
# MAIN TRAINING
# =========================

def main():
    print("Extracting video-level features...")
    X, y, file_paths = load_dataset(BASE_DIR)

    if X.empty:
        print("No data found. Please check data/processed/normal and data/processed/reckless.")
        return

    print(f"\nDataset shape: {X.shape}")
    print(f"Class distribution [Normal, Reckless]: {np.bincount(y)}")

    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        X,
        y,
        file_paths,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nTrain size: {len(y_train)}")
    print(f"Test size: {len(y_test)}")
    print(f"Train distribution [Normal, Reckless]: {np.bincount(y_train)}")
    print(f"Test distribution [Normal, Reckless]: {np.bincount(y_test)}")

    # F2 scorer prioritizes Reckless recall
    reckless_f2 = make_scorer(
        fbeta_score,
        beta=2,
        pos_label=1,
        zero_division=0
    )

    model = GradientBoostingClassifier(random_state=RANDOM_STATE)

    param_distributions = {
        "n_estimators": [100, 150, 200, 300, 400, 500],
        "learning_rate": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.70, 0.85, 1.0],
        "max_features": [None, "sqrt", "log2"]
    }

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=RANDOM_STATE
    )

    sample_weights_train = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    print("\nTuning GradientBoosting with RandomizedSearchCV...")
    print("Optimizing for Reckless F2-score...")

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=80,
        scoring=reckless_f2,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train, sample_weight=sample_weights_train)

    print("\nBest CV score:", round(search.best_score_, 4))
    print("Best params:")
    print(search.best_params_)

    best_model = search.best_estimator_

    # Tune threshold using out-of-fold probabilities from train set
    print("\nFinding best classification threshold...")
    oof_probs = get_oof_probabilities(best_model, X_train, y_train)
    best_threshold, threshold_results = find_best_threshold(y_train, oof_probs)

    # Final training on all train data
    print("\nTraining final model...")
    final_model = clone(best_model)
    final_model.fit(X_train, y_train, sample_weight=sample_weights_train)

    test_probs = final_model.predict_proba(X_test)[:, 1]

    # Evaluate default threshold
    y_pred_default = (test_probs >= 0.50).astype(int)

    print("\n==============================")
    print("Evaluation with default threshold = 0.50")
    print("==============================")
    print(classification_report(
        y_test,
        y_pred_default,
        target_names=["Normal", "Reckless"],
        zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_default))

    # Evaluate tuned threshold
    y_pred_tuned = (test_probs >= best_threshold).astype(int)

    print("\n==============================")
    print(f"Evaluation with tuned threshold = {best_threshold:.2f}")
    print("==============================")
    print(classification_report(
        y_test,
        y_pred_tuned,
        target_names=["Normal", "Reckless"],
        zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_tuned))

    # Misclassified reckless videos
    print("\nMisclassified Reckless videos (False Normals):")
    false_normals = []

    for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred_tuned)):
        if true_label == 1 and pred_label == 0:
            false_normals.append(files_test[i])
            print("-", files_test[i])

    if len(false_normals) == 0:
        print("None. All reckless videos were detected.")

    # Misclassified normal videos
    print("\nMisclassified Normal videos (False Reckless):")
    false_reckless = []

    for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred_tuned)):
        if true_label == 0 and pred_label == 1:
            false_reckless.append(files_test[i])
            print("-", files_test[i])

    if len(false_reckless) == 0:
        print("None. No normal videos were incorrectly flagged.")

    # Feature importances
    importances = final_model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]

    print("\nTop Feature Importances:")
    for rank, idx in enumerate(indices[:15], start=1):
        print(f"{rank}. {feature_names[idx]} ({importances[idx]:.4f})")

    # Save model and metadata
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(final_model, MODEL_PATH)

    metadata = {
        "threshold": best_threshold,
        "feature_names": list(X.columns),
        "label_mapping": {
            "Normal": 0,
            "Reckless": 1
        },
        "best_params": search.best_params_,
        "best_cv_score_f2_reckless": search.best_score_,
        "threshold_results": threshold_results
    }

    joblib.dump(metadata, META_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {META_PATH}")


if __name__ == "__main__":
    main()