import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import joblib
import random

def extract_video_features(df):
    """
    Aggregate frame-level features into a single video-level vector.
    Must match the logic in train_classifier.py
    """
    # Use smoothed versions where available
    ttc = df['min_ttc_smooth']
    rel_vel = df['max_rel_vel']
    lane_dep = df['lane_departure_smooth'].abs()
    
    features = {
        'ttc_mean': ttc.mean(),
        'ttc_min': ttc.min(),
        'ttc_std': ttc.std(),
        'rel_vel_mean': rel_vel.mean(),
        'rel_vel_max': rel_vel.max(),
        'rel_vel_std': rel_vel.std(),
        'lane_dep_mean': lane_dep.mean(),
        'lane_dep_max': lane_dep.max(),
        'lane_dep_std': lane_dep.std(),
        'has_lane_fraction': df['has_lane'].mean(),
        'has_objects_fraction': df['has_objects'].mean(),
    }
    
    # Count occurrences of heuristic "why" labels
    why_counts = df['why'].value_counts(normalize=True).to_dict()
    features['why_tailgating_fraction'] = why_counts.get('tailgating', 0)
    features['why_lane_departure_fraction'] = why_counts.get('lane_departure', 0)
    features['why_aggressive_closing_fraction'] = why_counts.get('aggressive_closing', 0)
    
    return features

def load_dataset(base_dir):
    data = []
    labels = []
    
    categories = ['normal', 'reckless']
    
    for category in categories:
        path = os.path.join(base_dir, category, "*.csv")
        files = glob.glob(path)
        
        for f in files:
            df = pd.read_csv(f)
            if df.empty:
                continue
            
            features = extract_video_features(df)
            data.append(features)
            labels.append(1 if category == 'reckless' else 0)
            
    return pd.DataFrame(data), np.array(labels)

def run_ablation(X, y, ordered_features, params, random_order=False):
    features_to_use = list(ordered_features)
    if random_order:
        random.shuffle(features_to_use)
        print("\n--- Running Random Ablation ---")
    else:
        print("\n--- Running Ordered Ablation (by Importance) ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    current_features = []
    
    for i, feature in enumerate(features_to_use):
        current_features.append(feature)
        
        X_train_sub = X_train[current_features]
        X_test_sub = X_test[current_features]
        
        clf = GradientBoostingClassifier(**params, random_state=42)
        
        # Cross-validation
        cv_scores = cross_val_score(clf, X_train_sub, y_train, cv=skf, scoring='f1_macro')
        
        # Fit and test
        clf.fit(X_train_sub, y_train)
        y_pred = clf.predict(X_test_sub)
        test_f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"Step {i+1}: Added '{feature}'")
        print(f"  CV F1-Macro: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Test F1-Macro: {test_f1:.4f}")
        
        results.append({
            'num_features': i + 1,
            'last_feature': feature,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'test_f1': test_f1
        })
        
    return pd.DataFrame(results)

def main():
    base_dir = "data/processed/2026--04-20_split_random"
    X, y = load_dataset(base_dir)
    
    if X.empty:
        print("No data found.")
        return

    best_params = {'learning_rate': 0.2, 'max_depth': 4, 'n_estimators': 50}
    
    ordered_features = [
        'rel_vel_mean',
        'has_lane_fraction',
        'why_tailgating_fraction',
        'rel_vel_std',
        'why_lane_departure_fraction',
        'ttc_std',
        'has_objects_fraction',
        'lane_dep_std',
        'lane_dep_max',
        'rel_vel_max'
    ]
    
    # 1. Ordered Ablation
    ordered_results = run_ablation(X, y, ordered_features, best_params, random_order=False)
    
    # 2. Random Ablation (Optional, for comparison)
    random_results = run_ablation(X, y, ordered_features, best_params, random_order=True)
    
    print("\nSummary (Ordered):")
    print(ordered_results[['num_features', 'last_feature', 'cv_f1_mean', 'test_f1']])
    
    print("\nSummary (Random):")
    print(random_results[['num_features', 'last_feature', 'cv_f1_mean', 'test_f1']])

if __name__ == "__main__":
    main()
