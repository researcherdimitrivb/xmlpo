"""
main_ML_V2.py — Clinical Diagnostic Pipeline (N≈40)
Implements:
1. Elastic Net Soft Selection (L1 coefficient path)
2. Consensus Feature Reporting (Stable Core vs Fold-Dependent)

PROBAST Documentation Notes:
- Predictor Exclusion (Pr2): MRI, BP, and tilt-table physiological measures are explicitly 
  excluded via `is_physio_feature()` because the core clinical objective of this specific 
  model is to evaluate standard autonomic and Heart Rate Variability (HRV) predictors.
- Outcome Ascertainment (O2): The model assumes `CAN_Diagnosis` was assigned blinded to 
  the HRV predictors, per standard clinical protocols, though this should be independently 
  verified by the study coordinators.
"""
import os
import sys
import time
import warnings
import json
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, LeaveOneOut
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                             precision_score, f1_score, confusion_matrix, roc_curve,
                             average_precision_score, matthews_corrcoef, brier_score_loss)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from collections import Counter
import sklearn

# ---------------------------------------------------------
# 1. Global Constants & Custom Transformers 
# ---------------------------------------------------------

FORCED_FEATURES = ['DB_RMSSD_ms']
EXCLUDED_FEATURES = ['DB_SDNN_ms', 'INT_Age_SDNN', 'DB_MaxMin_Diff_bpm']

class MissingnessFilter(BaseEstimator, TransformerMixin):
    """
    Dynamically filters columns based on a missingness threshold (e.g., >15%) 
    to prevent training data leakage (PROBAST P1).
    """
    def __init__(self, threshold=15.0):
        self.threshold = threshold
        self.selected_columns_ = None
        
    def fit(self, X, y=None):
        missing_pct = (X.isnull().sum() / len(X)) * 100
        self.selected_columns_ = list(X.columns[missing_pct <= self.threshold])
        dropped = list(X.columns[missing_pct > self.threshold])
        if dropped:
            # We log but do not spam the console; stored in state for provenance
            self.dropped_cols_ = dropped
        else:
            self.dropped_cols_ = []
        return self
        
    def transform(self, X):
        return X[self.selected_columns_]
        
    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_columns_)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Computes interaction terms post-imputation (PROBAST Pr4, Pr5).
    """
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self
        
    def transform(self, X):
        X_out = X.copy()
        
        def safe_mul(c1, c2):
            if c1 in X_out.columns and c2 in X_out.columns:
                s1 = pd.to_numeric(X_out[c1], errors='coerce').fillna(0)
                s2 = pd.to_numeric(X_out[c2], errors='coerce').fillna(0)
                return s1 * s2
            return None
            
        interactions = [
            ('INT_Vagal_Dizziness', 'RMSSD', 'Dizziness'),
            ('INT_Glycemic_Burden', 'HbA1c', 'Years of DM'),
            ('INT_Microvascular_Collapse', 'Neuropathy', 'RETINOPATHY'),
            ('INT_SH_OH_Clash', 'Hypertension', 'Dizziness'),
            ('INT_Neuro_Inflammation', 'RMSSD', 'IL-6'),
            ('INT_Vascular_Toxicity', 'Hypertension', 'Years of DM'),
            ('INT_Oxidative_Stress', 'RDW %', 'IL-6'),
            ('INT_Active_Nerve_Damage', 'Painful feet', 'HbA1c'),
            ('INT_Metabolic_Syndrome', 'BMI', 'Insulin'),
            ('INT_MCHC_RDW', 'MCHC %', 'RDW %'),
            ('INT_Age_SDNN', 'Age', 'DB_SDNN_ms'),
            ('INT_DM_RMSSD', 'Years of DM', 'RMSSD'),
            ('INT_Hypertension_Duration', 'Hypertension', 'Duration'),
            ('INT_Hct_Hgb', 'Hct %', 'Hgb g/dL'),
            ('INT_Glucose_Hct', 'Glucose mg/dL', 'Hct %')
        ]
        
        for name, c1, c2 in interactions:
            val = safe_mul(c1, c2)
            if val is not None:
                X_out[name] = val
                
        if 'Glucose mg/dL' in X_out.columns and 'Insulin' in X_out.columns:
            X_out['INT_Insulin_Resistance'] = X_out['Glucose mg/dL'].fillna(0) / (X_out['Insulin'].fillna(0) + 1e-5)
            
        return X_out

    def get_feature_names_out(self, input_features=None):
        out_features = list(self.feature_names_in_)
        interactions = [
            ('INT_Vagal_Dizziness', 'RMSSD', 'Dizziness'),
            ('INT_Glycemic_Burden', 'HbA1c', 'Years of DM'),
            ('INT_Microvascular_Collapse', 'Neuropathy', 'RETINOPATHY'),
            ('INT_SH_OH_Clash', 'Hypertension', 'Dizziness'),
            ('INT_Neuro_Inflammation', 'RMSSD', 'IL-6'),
            ('INT_Vascular_Toxicity', 'Hypertension', 'Years of DM'),
            ('INT_Oxidative_Stress', 'RDW %', 'IL-6'),
            ('INT_Active_Nerve_Damage', 'Painful feet', 'HbA1c'),
            ('INT_Metabolic_Syndrome', 'BMI', 'Insulin'),
            ('INT_MCHC_RDW', 'MCHC %', 'RDW %'),
            ('INT_Age_SDNN', 'Age', 'DB_SDNN_ms'),
            ('INT_DM_RMSSD', 'Years of DM', 'RMSSD'),
            ('INT_Hypertension_Duration', 'Hypertension', 'Duration'),
            ('INT_Hct_Hgb', 'Hct %', 'Hgb g/dL'),
            ('INT_Glucose_Hct', 'Glucose mg/dL', 'Hct %')
        ]
        for name, c1, c2 in interactions:
            if c1 in out_features and c2 in out_features:
                out_features.append(name)
        if 'Glucose mg/dL' in out_features and 'Insulin' in out_features:
            out_features.append('INT_Insulin_Resistance')
        return np.array(out_features)


class MutualHRVExclusionFilter(BaseEstimator, TransformerMixin):
    """
    Enforces the rule: if DB_RMSSD_ms is present, exclude DB_SDNN_ms and INT_Age_SDNN.
    """
    def fit(self, X, y=None):
        cols = list(X.columns)
        if 'DB_RMSSD_ms' in cols:
            cols = [c for c in cols if c not in ['DB_SDNN_ms', 'INT_Age_SDNN']]
        self.selected_columns_ = cols
        return self
        
    def transform(self, X):
        return X[self.selected_columns_]
        
    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_columns_)


class SensitivityThresholdCalibrator(BaseEstimator, ClassifierMixin):
    """
    Finds Youden's J optimal threshold using Out-Of-Fold predictions
    to prevent in-sample threshold leakage (PROBAST A1).
    Removed internal sigmoid calibration (PROBAST A3).
    """
    def __init__(self, estimator, cv=3):
        self.estimator = estimator
        self.cv = cv
        self.threshold_ = 0.5

    def fit(self, X, y):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        y_arr = y.values if hasattr(y, "values") else np.asarray(y)

        self.estimator.fit(X_arr, y_arr)

        try:
            inner_cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
            y_probs_oof = cross_val_predict(self.estimator, X_arr, y_arr, cv=inner_cv, method='predict_proba')[:, 1]
            
            fpr, tpr, thresholds = roc_curve(y_arr, y_probs_oof)
            youden_j = tpr - fpr
            best_thresh_idx = np.argmax(youden_j)
            self.threshold_ = float(np.clip(thresholds[best_thresh_idx], 0.10, 0.90))
        except Exception as e:
            warnings.warn(f"Threshold search failed: {str(e)}. Defaulting to 0.5")
            self.threshold_ = 0.5

        return self

    def predict(self, X):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        try:
            probs = self.predict_proba(X_arr)[:, 1]
            return (probs >= self.threshold_).astype(int)
        except Exception:
            return self.estimator.predict(X_arr)

    def predict_proba(self, X):
        X_arr = X.values if hasattr(X, "values") else np.asarray(X)
        return self.estimator.predict_proba(X_arr)


# ---------------------------------------------------------
# 2. Validation & Consensus Stability
# ---------------------------------------------------------

def evaluate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'Precision':   precision_score(y_true, y_pred, zero_division=0),
        'F1 Score':    f1_score(y_true, y_pred, zero_division=0),
        'MCC':         matthews_corrcoef(y_true, y_pred)
    }
    
    if y_prob is not None:
        try:
            metrics['AUROC'] = roc_auc_score(y_true, y_prob)
            metrics['PR-AUC'] = average_precision_score(y_true, y_prob)
            metrics['Brier Score'] = brier_score_loss(y_true, y_prob)
        except ValueError:
            metrics['AUROC'] = 0.5
            metrics['PR-AUC'] = 0.5
            metrics['Brier Score'] = 1.0
    else:
        metrics['AUROC'] = 0.5
        metrics['PR-AUC'] = 0.5
        metrics['Brier Score'] = 1.0

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except ValueError:
        metrics['Specificity'] = np.nan

    return metrics


def compute_jaccard_stability(feature_sets):
    n_folds = len(feature_sets)
    if n_folds < 2:
        return 1.0
    jaccards = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            set1, set2 = set(feature_sets[i]), set(feature_sets[j])
            union = len(set1.union(set2))
            if union == 0:
                jaccards.append(0.0)
            else:
                jaccards.append(len(set1.intersection(set2)) / union)
    return float(np.mean(jaccards))


def run_nested_cv(X, y, model_name, pipeline, param_grid):
    print(f"  [CV] Starting Repeated Nested Cross-Validation (5 seeds x 5 folds)...", flush=True)
    t_start = time.time()

    seeds = [42, 100, 200, 300, 400]
    all_fold_metrics = {
        'AUROC': [], 'PR-AUC': [], 'Sensitivity': [], 'Specificity': [],
        'Accuracy': [], 'F1 Score': [], 'Precision': [], 'MCC': [], 'Brier Score': []
    }
    
    all_outer_fold_features = []
    oof_y_true = []
    oof_y_prob = []
    
    provenance_drops = []

    for seed in seeds:
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        for fold, (train_ix, test_ix) in enumerate(outer_cv.split(X, y), 1):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]

            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed+fold)
            search = GridSearchCV(
                pipeline, param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=1
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            # Wrap the tuned classifier in the calibrator to find the optimal threshold ONCE
            tuned_lr = best_model.named_steps['clf']
            calibrator = SensitivityThresholdCalibrator(tuned_lr, cv=3)
            best_model.steps[-1] = ('clf', calibrator)
            best_model.fit(X_train, y_train)
            
            # Log missingness drops
            if 'missingness' in best_model.named_steps:
                drops = best_model.named_steps['missingness'].dropped_cols_
                if drops:
                    provenance_drops.extend(drops)

            # Feature extraction
            try:
                clf_step = best_model.named_steps['clf']
                lr_model = clf_step.estimator
                
                # To get feature names in, we must transform up to the classifier
                pre_pipe = clone(pipeline)
                pre_pipe.steps = pre_pipe.steps[:-1] # Drop clf
                X_trans = pre_pipe.fit_transform(X_train, y_train)
                feat_names_in = list(X_trans.columns)
                
                coefs = lr_model.coef_[0]
                fold_feats = np.array(feat_names_in)[coefs != 0]
                all_outer_fold_features.append(list(fold_feats))
            except Exception as e:
                warnings.warn(f"Feature extraction failed in fold: {str(e)}")
                all_outer_fold_features.append([])

            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            oof_y_true.extend(y_test)
            oof_y_prob.extend(y_prob)

            metrics = evaluate_metrics(y_test, y_pred, y_prob)
            for k, v in metrics.items():
                all_fold_metrics[k].append(v)

    elapsed_total = time.time() - t_start
    print(f"  [CV] 25 folds completed in {elapsed_total:.1f}s", flush=True)

    if provenance_drops:
        unique_drops = list(set(provenance_drops))
        print(f"  [INFO] Features dropped for >15% missingness across CV: {unique_drops}")

    final_results = {}
    for k, v in all_fold_metrics.items():
        arr = np.array(v, dtype=float)
        final_results[k] = (np.nanmean(arr), np.nanstd(arr))

    # Output Consensus Report and Stability Score
    stability_score = compute_jaccard_stability(all_outer_fold_features)
    final_results['Jaccard Stability'] = (stability_score, 0.0)

    print(f"\n  Consensus Feature Report (Across 25 Outer Folds):")
    feature_counts = Counter(f for fold in all_outer_fold_features for f in fold)
    stable_core = {f: c for f, c in feature_counts.items() if c >= 15}
    fold_dep = {f: c for f, c in feature_counts.items() if c < 15}
    
    print(f"    Stable Core Features (Selected in >= 15/25 folds):")
    if stable_core:
        for f, c in sorted(stable_core.items(), key=lambda x: x[1], reverse=True):
            print(f"      - [{c}/25] {f}")
    else:
        print("      - (None)")
        
    print(f"\n  Feature Selection Stability (Jaccard Index): {stability_score:.3f}")

    provenance_data = {
        'stability_score': stability_score,
        'stable_core': stable_core,
        'missingness_dropped': list(set(provenance_drops))
    }

    return final_results, np.array(oof_y_true), np.array(oof_y_prob), provenance_data


# ---------------------------------------------------------
# 3. Data Helpers
# ---------------------------------------------------------

def is_hard_leak_or_admin(c):
    c = str(c).upper().strip()
    leak_admin_terms = {
        'VM RATIO', 'TARGET', 'VAL1_RATIO', 'VAL2_RATIO', 'DB_MHRR_BPM',
        'OUR_GROUP', 'SUBJECT NUMBER', 'OUR_VAL1_RATIO', 'OUR_VAL2_RATIO',
        'OUR_DB_MHRR_BPM', 'PNN50', 'HF', 'LF', 'VLF', 'HRV',
        'DB INDEX', 'VALSALVA', 'RETINOPATHY GRADE', 'CAN_DIAGNOSIS', 'GROUP', 'CRITVALMHRR',
        'VAL_AVG', 'VAL_AVG_RATIO', 'EWING', 'EWING_30_15_RATIO', 'TILT_30_15_RATIO',
        'TOTAL_POWER', 'TOTAL POWER', 'INCLUSION', 'EYE EXAM', 'NOTES', 'MAHAL. DISTANCE', 'GROUP2', 'SUBJECT'
    }
    return c in leak_admin_terms

def is_physio_feature(c):
    c = str(c).upper()
    pattern = re.compile(
        r'(CSF|GM|WM|VENT|CROP|FLAIR|IR GLOBAL|IR CROP|IR VENT|GRADE|PUNCTUATE|SUM\(C\)|'
        r'MCAR|MCAL|RAAR|MCA|DIAMETER|CVR|VMR|L_IC|R_IC|'
        r'CO2|REACTIVITY|BF|HV|'
        r'MR BASELINE|HYPERMIN|HYPERMEAN|HYPOMAX|HYPO MEAN|TILT MN|HYPO TILT|\(BASELINE MR\)|\(HYPERMIN MN\)|\(HYPERMEAN\)|\(HYPOMAX\)|\(HYPO MEAN\)|\(TILT MN\)|\(HYPO TILT MEAN\)|MR HYPO|MR BAS|MR BASELINE 2|'
        r'BP|SBP|DBP|SYS_|DIA_|MN_|'
        r'TILT|1 MIN|3 MIN|5 MIN)'
    )
    return bool(pattern.search(c))

# ---------------------------------------------------------
# 4. Plotting & Summary
# ---------------------------------------------------------

def generate_performance_plot(model_results, out_dir, pathway_name):
    metrics_list = []
    excluded_metrics = ['Accuracy', 'Precision', 'MCC', 'Brier Score', 'Jaccard Stability']
    for algo_name, metrics in model_results.items():
        for metric_name, (mean, std) in metrics.items():
            if metric_name in excluded_metrics:
                continue
            metrics_list.append({
                'Model': algo_name,
                'Metric': metric_name,
                'Value (%)': mean * 100.0,
                'Std (%)': std * 100.0
            })
    perf_df = pd.DataFrame(metrics_list)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 8), gridspec_kw={'width_ratios': [1.8, 1]})
    sns.set_theme(style="whitegrid")

    ax_bar = sns.barplot(x='Metric', y='Value (%)', hue='Model', data=perf_df,
                         palette='viridis', edgecolor='black', ax=ax1)
    ax1.set_title(f"Algorithm Performance - {pathway_name}", fontsize=14, fontweight='bold', pad=15)
    max_val = perf_df['Value (%)'].max() if not perf_df.empty else 100
    ax1.set_ylim(0, min(115, max_val * 1.25))
    ax1.set_ylabel("Performance Value (%)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Metric Type", fontsize=11, fontweight='bold')

    for p in ax_bar.patches:
        height = p.get_height()
        if height > 0:
            ax1.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=8, fontweight='bold',
                         color='black', xytext=(0, 4), textcoords='offset points')

    ax1.legend(loc='upper left', frameon=True)
    ax2.axis('off')
    ax2.text(0.02, 0.98, "Nested CV Summary (Repeated 5x5)", fontsize=13, fontweight='bold', ha='left', va='top')

    summary_text = ""
    for algo_name, metrics in model_results.items():
        summary_text += f"■ {algo_name}\n"
        for metric_name, (mean, std) in metrics.items():
            summary_text += f"  • {metric_name:22s}: {mean*100:.1f}%\n"
        summary_text += "\n"

    ax2.text(0.02, 0.88, summary_text, fontsize=10, fontfamily='monospace', va='top', ha='left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"Performance_Graphs.svg"), dpi=300)
    plt.close()

def generate_calibration_plot(calibration_data, out_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.set_theme(style="whitegrid")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=1.5)
    colors = sns.color_palette("viridis", len(calibration_data))

    for i, (model_name, (y_true, y_prob)) in enumerate(calibration_data.items()):
        if len(y_prob) == 0: continue
        brier = brier_score_loss(y_true, y_prob)
        lowess_fit = sm.nonparametric.lowess(endog=y_true, exog=y_prob, frac=0.6, it=3)
        prob_pred_smooth = lowess_fit[:, 0]
        prob_true_smooth = lowess_fit[:, 1]
        ax.plot(prob_pred_smooth, prob_true_smooth, "-", color=colors[i], label=f"{model_name} (Brier={brier:.3f})", linewidth=2.5)
        ax.scatter(y_prob, y_true, color=colors[i], alpha=0.3, s=25, zorder=3)

    ax.set_title("Continuous Model Calibration Curves (LOWESS Smoothed)", fontsize=14, fontweight='bold')
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="upper left"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Models_Calibration_Plot.svg"), dpi=300)
    plt.close()

# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clinical ML Pipeline")
    parser.add_argument('--data-path', type=str, default="GE-71_Data_Summary_Table_final.csv", help="Path to input dataset")
    parser.add_argument('--output-dir', type=str, default="Results", help="Directory to save results")
    args = parser.parse_args()

    sklearn.set_config(transform_output="pandas")
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', module='sklearn')

    os.makedirs(args.output_dir, exist_ok=True)
    graphs_dir = os.path.join(args.output_dir, "Graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    print("=" * 70)
    print("  CLINICAL ML PIPELINE — MEDIAN IMPUTE & COST-SENSITIVE")
    print("=" * 70, flush=True)

    if not os.path.exists(args.data_path):
        search_paths = [
            r"C:\Unie\TG_master\STAGE2\Scripts_ZGT_CAN\Data\GE-71_Data_Summary_Table_final.csv",
            "../Data/GE-71_Data_Summary_Table_final.csv"
        ]
        for p in search_paths:
            if os.path.exists(p):
                args.data_path = p
                break

    if not os.path.exists(args.data_path):
        print(f"CRITICAL ERROR: Dataset '{args.data_path}' not found!")
        return

    df_master = pd.read_csv(args.data_path)
    if 'CAN_Diagnosis' in df_master.columns:
        pre_n = len(df_master)
        df_master = df_master.dropna(subset=['CAN_Diagnosis']).reset_index(drop=True)
        dropped_n = pre_n - len(df_master)
        if dropped_n > 0:
            print(f"[DATA] Dropped {dropped_n} rows missing CAN_Diagnosis. Final N = {len(df_master)}")
    
    print(f"\n[DATA] Final N = {len(df_master)}", flush=True)
    df_master = df_master.copy()
    df_master['Target'] = df_master['CAN_Diagnosis'].map(lambda x: 1.0 if str(x).strip().upper() == 'CAN' else 0.0)

    # Clean categorical YES/NO
    for col in df_master.columns:
        if (df_master[col].dtype == object or pd.api.types.is_string_dtype(df_master[col])) and col not in ['CAN_Diagnosis', 'Target']:
            df_master[col] = df_master[col].map(
                lambda x: 1.0 if str(x).strip().upper() == 'YES' else (0.0 if str(x).strip().upper() == 'NO' else x))
            df_master[col] = pd.to_numeric(df_master[col], errors='coerce')

    all_numeric_cols = df_master.select_dtypes(include=[np.number]).columns.tolist()
    if 'Target' in all_numeric_cols:
        all_numeric_cols.remove('Target')

    safe_numeric_cols = [c for c in all_numeric_cols if not is_hard_leak_or_admin(c)]
    dossier_cols = [c for c in safe_numeric_cols if not is_physio_feature(c)]
    
    for fc in FORCED_FEATURES:
        if fc not in dossier_cols and fc in df_master.columns:
            dossier_cols.append(fc)

    X = df_master[dossier_cols]
    y = df_master['Target'].values

    print(f"[DATA] Final Candidate Features: {X.shape[1]}", flush=True)

    models = {
        "Elastic Net Logistic Regression (EL-LR)": {
            "pipeline": Pipeline([
                ('missingness', MissingnessFilter(threshold=15.0)),
                ('imputer', KNNImputer(n_neighbors=3)),
                ('engineer', FeatureEngineer()),
                ('hrv_exclusion', MutualHRVExclusionFilter()),
                ('scaler', RobustScaler()),
                ('clf', LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', max_iter=3000, random_state=42))
            ]),
            "grid": {
                'clf__C': [0.05, 0.1, 1.0, 10.0],
                'clf__l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9]
            }
        },
        "Random Forest (RF)": {
            "pipeline": Pipeline([
                ('missingness', MissingnessFilter(threshold=15.0)),
                ('imputer', KNNImputer(n_neighbors=3)),
                ('engineer', FeatureEngineer()),
                ('hrv_exclusion', MutualHRVExclusionFilter()),
                ('scaler', RobustScaler()),
                ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
            ]),
            "grid": {
                'clf__n_estimators': [50, 100, 200],
                'clf__max_depth': [3, 5, None],
                'clf__min_samples_leaf': [1, 2, 5]
            }
        }
    }

    all_pathway_results = {}
    calibration_data = {}
    provenances = {}

    for model_name, config in models.items():
        print(f"\n{'-'*70}")
        print(f"  Running: {model_name}")
        print(f"{'-'*70}", flush=True)
        
        results_dict, y_true_oof, y_prob_oof, prov = run_nested_cv(
            X, y,
            model_name=model_name,
            pipeline=config['pipeline'],
            param_grid=config['grid']
        )
        all_pathway_results[model_name] = results_dict
        calibration_data[model_name] = (y_true_oof, y_prob_oof)
        provenances[model_name] = prov

    print(f"\n{'='*70}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    for model_name, model_res in all_pathway_results.items():
        print(f"\n  Model: {model_name}")
        for metric, (mean, std) in model_res.items():
            if metric == 'Jaccard Stability':
                print(f"    {metric:22s}: {mean:.4f}")
            else:
                print(f"    {metric:22s}: {mean*100:6.2f}% (+/- {std*100:.2f}%)")

    generate_performance_plot(all_pathway_results, graphs_dir, "Main_ML_V2_Models")
    generate_calibration_plot(calibration_data, graphs_dir)
    
    # Save provenance JSON
    prov_file = os.path.join(args.output_dir, f"run_provenance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(prov_file, 'w') as f:
        json.dump(provenances, f, indent=4)
    print(f"[FAIR] Saved run provenance to {prov_file}")

    print(f"\n[FINAL MODEL] Fitting final production models on FULL DATA...", flush=True)
    
    for model_name, config in models.items():
        print(f"  Fitting {model_name}...")
        final_search = GridSearchCV(
            config['pipeline'], config['grid'], 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
            scoring='roc_auc'
        )
        final_search.fit(X, y)
        best_model = final_search.best_estimator_
        
        # Wrap final model in calibrator
        final_calibrator = SensitivityThresholdCalibrator(best_model.named_steps['clf'], cv=3)
        best_model.steps[-1] = ('clf', final_calibrator)
        best_model.fit(X, y)
        
        # SHAP and Correlation Matrix
        try:
            pre_pipe = clone(config['pipeline'])
            pre_pipe.steps = pre_pipe.steps[:-1] # Drop clf
            X_transformed_df = pre_pipe.fit_transform(X, y)
            
            base_estimator = best_model.named_steps['clf'].estimator
            
            # Select features "in the model"
            if isinstance(base_estimator, LogisticRegression):
                coefs = base_estimator.coef_[0]
                selected_features = X_transformed_df.columns[coefs != 0]
                
                print(f"\n  [MODEL EQUATION] Final EL-LR Coefficients:")
                for f, c in zip(X_transformed_df.columns, coefs):
                    if c != 0:
                        print(f"    {f}: {c:.4f}")
                        
            elif isinstance(base_estimator, RandomForestClassifier):
                importances = base_estimator.feature_importances_
                sorted_idx = np.argsort(importances)[::-1]
                top_10 = sorted_idx[:10]
                selected_features = X_transformed_df.columns[top_10]
            else:
                selected_features = X_transformed_df.columns
                
            X_selected = X_transformed_df[selected_features]
            
            # SHAP
            explainer = shap.Explainer(base_estimator, X_transformed_df)
            shap_values_full = explainer(X_transformed_df)
            
            plt.figure(figsize=(10, 6))
            idx_list = [X_transformed_df.columns.get_loc(c) for c in selected_features]
            
            if hasattr(shap_values_full, "values"):
                if len(shap_values_full.values.shape) == 3:
                    shap_vals = shap_values_full.values[:, :, 1]
                else:
                    shap_vals = shap_values_full.values
            else:
                if isinstance(shap_values_full, list):
                    shap_vals = shap_values_full[1]
                else:
                    shap_vals = shap_values_full
                
            shap_plot_vals = shap_vals[:, idx_list]
            X_plot = X_selected.copy()

            # For EL-LR, exclude outlier datapoints with influence outside [-5, +5] exclusively from the SHAP plot
            if isinstance(base_estimator, LogisticRegression):
                valid_mask = np.all((shap_plot_vals >= -5.0) & (shap_plot_vals <= 5.0), axis=1)
                if not np.all(valid_mask):
                    n_excluded = int(np.sum(~valid_mask))
                    print(f"  [SHAP] Excluded {n_excluded} outlier sample(s) with |SHAP| > 5 from EL-LR SHAP plot.")
                    shap_plot_vals = shap_plot_vals[valid_mask]
                    X_plot = X_plot.iloc[valid_mask]

            shap.summary_plot(shap_plot_vals, X_plot, show=False)
                
            model_slug = "EL-LR" if "Elastic" in model_name else "RF"
            plt.title(f"SHAP Summary — {model_name}", fontsize=13, fontweight='bold', pad=15)
            plt.tight_layout()
            shap_out = os.path.join(graphs_dir, f"shap_summary_plot_{model_slug}.svg")
            plt.savefig(shap_out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [SHAP] Saved summary plot to: {shap_out}")
            
            # Correlation Matrix
            diag_cols = ['DB_MHRR_bpm', 'Val1_Ratio', 'Val2_Ratio', 'Ewing_30_15_Ratio', 'Tilt_30_15_Ratio', 'Classic_OH_SBP', 'Classic_OH_DBP']
            diag_data = df_master[diag_cols].copy()
            for c in diag_cols:
                diag_data[c] = pd.to_numeric(diag_data[c], errors='coerce')
                
            correlations = {}
            for feat in selected_features:
                correlations[feat] = {}
                for crit in diag_cols:
                    corr = X_selected[feat].corr(diag_data[crit], method='pearson')
                    correlations[feat][crit] = corr
            corr_df = pd.DataFrame(correlations).T
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_df, annot=True, cmap='RdBu', center=0, fmt='.2f', vmin=-1, vmax=1)
            plt.title(f'Correlation: Model Features vs CAN Criteria ({model_name})')
            plt.tight_layout()
            corr_out = os.path.join(graphs_dir, f"correlation_heatmap_{model_slug}.svg")
            plt.savefig(corr_out, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [CORR] Saved correlation matrix to: {corr_out}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to generate SHAP/Corr plots for {model_name}: {e}")

if __name__ == "__main__":
    main()