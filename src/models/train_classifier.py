import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def extract_video_features(df):
    """
    Aggregate frame-level features into a single video-level vector.
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
    filenames = []
    
    categories = ['normal', 'reckless']
    
    for category in categories:
        path = os.path.join(base_dir, category, "*.csv")
        files = glob.glob(path)
        print(f"Loading {len(files)} files for category: {category}")
        
        for f in files:
            df = pd.read_csv(f)
            if df.empty:
                continue
            
            features = extract_video_features(df)
            data.append(features)
            labels.append(1 if category == 'reckless' else 0)
            filenames.append(f)
            
    return pd.DataFrame(data), np.array(labels), filenames

def main():
    base_dir = "data/processed/2026-04-20"
    # base_dir = "data/processed/2026-04-20_split"
    # base_dir = "data/processed/2026-04-20_split_random"
    print("Extracting video-level features...")
    X, y, filenames = load_dataset(base_dir)
    
    if X.empty:
        print("No data found. Please run batch_process.py first.")
        return
        
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        X, y, filenames, test_size=0.3, random_state=42, stratify=y
    )
    
    # Tuning Configuration
    search_type = 'grid'  # Set to 'grid' or 'random'
    
    # Define models to compare
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, random_state=42))
        ]),
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ])
    }
    
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100, 200, 300, 400],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.5],
            'max_depth': [3, 4, 5]
        },
        'SVM': {
            'svc__C': [0.1, 1, 10, 100],
            'svc__gamma': ['scale', 'auto'],
            'svc__kernel': ['rbf', 'poly', 'linear']
        },
        'LogisticRegression': {
            'lr__C': [0.01, 0.1, 1, 10, 100],
            'lr__solver': ['liblinear', 'lbfgs']
        }
    }
    
    print(f"\nPerforming Hyperparameter Tuning using {search_type} search...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_score = -1
    best_model_name = None
    best_clf = None
    results = {}

    for name, model in models.items():
        print(f"Tuning {name}...")
        
        if search_type == 'grid':
            search = GridSearchCV(
                model, 
                param_grids[name], 
                cv=skf, 
                scoring='f1_macro', 
                n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                model, 
                param_distributions=param_grids[name], 
                n_iter=10, 
                cv=skf, 
                scoring='f1_macro', 
                n_jobs=-1, 
                random_state=42
            )
            
        search.fit(X_train, y_train)
        
        mean_score = search.best_score_
        results[name] = mean_score
        print(f"{name}: Best F1-Macro = {mean_score:.4f}")
        print(f"Best params: {search.best_params_}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            best_clf = search.best_estimator_

    print(f"\nOverall Best Model: {best_model_name} with F1-Macro = {best_score:.4f}")
    
    # Final model for evaluation
    clf = best_clf
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    print(f"\nEvaluation of {best_model_name} on Test Set:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Reckless']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Identify misclassified reckless videos
    print("\nMisclassified Reckless videos (False Normals):")
    misclassified_count = 0
    for i in range(len(y_test)):
        if y_test[i] == 1 and y_pred[i] == 0:
            print(f"- {os.path.basename(files_test[i])}")
            misclassified_count += 1
    
    if misclassified_count == 0:
        print("None!")
    
    # Feature Importance (only for tree-based models or Logistic Regression)
    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
        feature_names = X.columns
        indices = np.argsort(importances)[::-1]
        
        print("\nTop Feature Importances:")
        for f in range(min(10, X.shape[1])):
            print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
    elif best_model_name == 'LogisticRegression':
        importances = np.abs(clf.named_steps['lr'].coef_[0])
        feature_names = X.columns
        indices = np.argsort(importances)[::-1]
        print("\nTop Coefficient Magnitudes (Logistic Regression):")
        for f in range(min(10, X.shape[1])):
            print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
    
    # Save model
    model_path = "src/models/classifier.joblib"
    joblib.dump(clf, model_path)
    print(f"\nBest model ({best_model_name}) saved to {model_path}")

if __name__ == "__main__":
    main()
