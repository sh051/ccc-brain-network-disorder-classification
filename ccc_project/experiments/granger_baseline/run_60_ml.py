import os
import glob
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# ==============================================================================
# CONFIGURATION & PATHS (Strictly aligned with ccc_parameter_search/grid_search.py)
# ==============================================================================
BASE_PATH = r"D:\capstone"
TIMESERIES_ASD_DIR = os.path.join(BASE_PATH, r"data_timeseries\ASD")
TIMESERIES_TDC_DIR = os.path.join(BASE_PATH, r"data_timeseries\TDC")

OUTPUT_DIR = os.path.join(BASE_PATH, r"ccc_project\artifacts\granger_baseline")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(OUTPUT_DIR, "granger_60_ml_results.csv")
DETAILED_FOLDS_CSV = os.path.join(OUTPUT_DIR, "granger_60_ml_fold_details.csv")

SUBSET_SIZE_PER_GROUP = 30  # 30 ASD + 30 TDC = 60 subjects total
SEED = 42
N_ROIS = 200

# ==============================================================================
# 1. VECTORIZED BIVARIATE GRANGER CAUSALITY (Frisch-Waugh-Lovell OLS)
# ==============================================================================
def compute_granger_matrix(file_path, lag=1):
    """
    Computes a 200x200 bivariate Granger Causality matrix for a subject time series.
    G[i, j] = directed causal strength from ROI j to ROI i (Y_j -> Y_i).
    Uses the Frisch-Waugh-Lovell projection theorem for fast OLS computation.
    """
    ts = np.loadtxt(file_path)
    T, N = ts.shape

    # Standardize each ROI time series
    ts = (ts - ts.mean(axis=0)) / (ts.std(axis=0) + 1e-9)

    T_eff = T - lag
    Y_target = ts[lag:, :]  # (T_eff, N)

    # Build lagged feature blocks for order p
    # For lag p, own lags = [X(t-1), ..., X(t-p)]
    lagged_blocks = [ts[lag - k : T - k, :] for k in range(1, lag + 1)]
    # lagged_blocks[k] is (T_eff, N)

    G = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        y_i = Y_target[:, i]

        # Restricted model features: intercept + own lags of ROI i
        own_lags = np.column_stack([lagged_blocks[k][:, i] for k in range(lag)])
        X_R = np.column_stack([np.ones(T_eff), own_lags])  # (T_eff, 1 + lag)

        # QR decomposition of restricted design matrix
        Q, _ = np.linalg.qr(X_R)

        # Restricted residuals
        e_R = y_i - Q @ (Q.T @ y_i)
        rss_r = np.sum(e_R**2)

        # For each source ROI j != i, unrestricted model adds lagged values of j
        # Orthogonalize source lags against X_R
        for j in range(N):
            if i == j:
                continue
            src_lags = np.column_stack([lagged_blocks[k][:, j] for k in range(lag)])
            Z_orth = src_lags - Q @ (Q.T @ src_lags)  # (T_eff, lag)

            # OLS residual reduction from adding Z_orth:
            # delta_rss = e_R^T Z_orth (Z_orth^T Z_orth)^(-1) Z_orth^T e_R
            try:
                Z_inv = np.linalg.pinv(Z_orth.T @ Z_orth)
                proj = Z_orth @ (Z_inv @ (Z_orth.T @ e_R))
                delta_rss = np.sum(proj**2)
            except np.linalg.LinAlgError:
                delta_rss = 0.0

            rss_u = max(rss_r - delta_rss, 1e-12)
            g_val = max(0.0, np.log((rss_r + 1e-12) / rss_u))
            G[i, j] = g_val

    return G

# ==============================================================================
# 2. FEATURE EXTRACTION & POST-PROCESSING (Matching CCC pipeline)
# ==============================================================================
def process_matrix_features(G, threshold=0.15):
    """
    Applies subject-wise Z-scoring, negative clipping, proportional thresholding,
    and returns off-diagonal edges as 1D features (size: 39,800).
    """
    N = G.shape[0]
    mask = ~np.eye(N, dtype=bool)

    # Subject-wise Z-scoring
    mean = G[mask].mean()
    std = G[mask].std() + 1e-9
    C = (G - mean) / std
    np.fill_diagonal(C, 0.0)

    # Clip negative values to 0 (keeps only stronger-than-average causal flow)
    C = np.maximum(C, 0.0)

    # Proportional thresholding
    if threshold is not None:
        flat_vals = C[mask]
        cutoff = np.percentile(flat_vals, 100 * (1 - threshold))
        C[C < cutoff] = 0.0

    return C[mask]

# ==============================================================================
# 3. 5-FOLD CV EVALUATION
# ==============================================================================
def evaluate_classifier(clf, X, y, skf):
    fold_aucs, fold_accs, fold_sens, fold_specs = [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf.fit(X_train, y_train)

        if hasattr(clf, "predict_proba"):
            val_probs = clf.predict_proba(X_val)[:, 1]
        elif hasattr(clf, "decision_function"):
            val_probs = clf.decision_function(X_val)
        else:
            val_probs = clf.predict(X_val)

        val_preds = (val_probs > (0.5 if hasattr(clf, "predict_proba") else 0.0)).astype(int)

        try:
            auc = roc_auc_score(y_val, val_probs)
        except Exception:
            auc = 0.5
        acc = accuracy_score(y_val, val_preds)

        cm = confusion_matrix(y_val, val_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        fold_aucs.append(auc)
        fold_accs.append(acc)
        fold_sens.append(sens)
        fold_specs.append(spec)

    return {
        "mean_auc": np.mean(fold_aucs),
        "std_auc": np.std(fold_aucs),
        "mean_acc": np.mean(fold_accs),
        "std_acc": np.std(fold_accs),
        "mean_sens": np.mean(fold_sens),
        "std_sens": np.std(fold_sens),
        "mean_spec": np.mean(fold_specs),
        "std_spec": np.std(fold_specs),
        "fold_aucs": fold_aucs,
        "fold_accs": fold_accs,
    }

# ==============================================================================
# 4. MAIN EXPERIMENT RUNNER
# ==============================================================================
def main():
    print("=" * 80)
    print("GRANGER CAUSALITY BASELINE - 60 SUBJECT ML EVALUATION")
    print("=" * 80)

    # 1. Select the exact same 60 subjects as ccc_parameter_search/grid_search.py
    asd_files = sorted(glob.glob(os.path.join(TIMESERIES_ASD_DIR, "*.1D")))[:SUBSET_SIZE_PER_GROUP]
    tdc_files = sorted(glob.glob(os.path.join(TIMESERIES_TDC_DIR, "*.1D")))[:SUBSET_SIZE_PER_GROUP]

    assert len(asd_files) == SUBSET_SIZE_PER_GROUP, f"Missing ASD files: {len(asd_files)}"
    assert len(tdc_files) == SUBSET_SIZE_PER_GROUP, f"Missing TDC files: {len(tdc_files)}"

    subset_paths = asd_files + tdc_files
    subset_labels = np.array([1] * len(asd_files) + [0] * len(tdc_files))
    print(f"Loaded {len(subset_paths)} subjects (30 ASD, 30 TDC).")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    n_cores = max(1, os.cpu_count() - 2)

    # Test lag orders p=1, p=2, and thresholding options
    configs = [
        {"lag": 1, "threshold": 0.15, "desc": "Lag-1 (p=1), Top 15% Edges (Exact CCC Match)"},
        {"lag": 1, "threshold": None, "desc": "Lag-1 (p=1), Unthresholded Full Edges"},
        {"lag": 2, "threshold": 0.15, "desc": "Lag-2 (p=2), Top 15% Edges"},
        {"lag": 2, "threshold": None, "desc": "Lag-2 (p=2), Unthresholded Full Edges"},
    ]

    all_summary_results = []
    all_fold_details = []

    for cfg in configs:
        lag = cfg["lag"]
        thresh = cfg["threshold"]
        desc = cfg["desc"]
        print(f"\n---> Computing Granger Causality: {desc}...")

        start_t = time.time()
        raw_matrices = Parallel(n_jobs=n_cores)(
            delayed(compute_granger_matrix)(path, lag=lag) for path in subset_paths
        )
        compute_time = time.time() - start_t
        print(f"     Computed 60 GC matrices in {compute_time:.2f}s ({compute_time/60:.2f}s/subj)")

        # Feature extraction
        X = np.array([process_matrix_features(g, threshold=thresh) for g in raw_matrices])
        y = subset_labels

        print(f"     Feature matrix shape: {X.shape}")

        # Classifiers to evaluate
        classifiers = {
            "Logistic Regression (C=1.0, L2) [GridSearch Match]": LogisticRegression(
                C=1.0, penalty="l2", solver="liblinear", random_state=SEED
            ),
            "Logistic Regression (C=0.1, L2)": LogisticRegression(
                C=0.1, penalty="l2", solver="liblinear", random_state=SEED
            ),
            "Logistic Regression (C=0.01, L2)": LogisticRegression(
                C=0.01, penalty="l2", solver="liblinear", random_state=SEED
            ),
            "Linear SVM (C=0.1)": SVC(
                kernel="linear", C=0.1, probability=True, random_state=SEED
            ),
            "Ridge Classifier (alpha=100.0)": RidgeClassifier(
                alpha=100.0, random_state=SEED
            ),
            "Random Forest (n=100)": RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=SEED
            ),
        }

        for clf_name, clf in classifiers.items():
            metrics = evaluate_classifier(clf, X, y, skf)
            print(f"       [{clf_name}]")
            print(f"         AUC : {metrics['mean_auc']:.3f} +/- {metrics['std_auc']:.3f}")
            print(f"         Acc : {metrics['mean_acc']:.3f} +/- {metrics['std_acc']:.3f}")
            print(f"         Sens: {metrics['mean_sens']:.3f} | Spec: {metrics['mean_spec']:.3f}")

            all_summary_results.append({
                "config_desc": desc,
                "lag": lag,
                "threshold": thresh,
                "classifier": clf_name,
                "mean_auc": metrics["mean_auc"],
                "std_auc": metrics["std_auc"],
                "mean_acc": metrics["mean_acc"],
                "std_acc": metrics["std_acc"],
                "mean_sens": metrics["mean_sens"],
                "std_sens": metrics["std_sens"],
                "mean_spec": metrics["mean_spec"],
                "std_spec": metrics["std_spec"],
                "compute_time_s": compute_time,
            })

            for f_idx, (f_auc, f_acc) in enumerate(zip(metrics["fold_aucs"], metrics["fold_accs"])):
                all_fold_details.append({
                    "config_desc": desc,
                    "lag": lag,
                    "threshold": thresh,
                    "classifier": clf_name,
                    "fold": f_idx + 1,
                    "auc": f_auc,
                    "acc": f_acc,
                })

    summary_df = pd.DataFrame(all_summary_results)
    summary_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved summary results to: {RESULTS_CSV}")

    detailed_df = pd.DataFrame(all_fold_details)
    detailed_df.to_csv(DETAILED_FOLDS_CSV, index=False)
    print(f"Saved detailed fold results to: {DETAILED_FOLDS_CSV}")

    print("\n" + "=" * 80)
    print("TOP 5 PERFORMING GRANGER CAUSALITY BASELINE CONFIGURATIONS:")
    print("=" * 80)
    top_5 = summary_df.sort_values(by="mean_auc", ascending=False).head(5)
    print(top_5[["config_desc", "classifier", "mean_auc", "std_auc", "mean_acc", "mean_sens", "mean_spec"]].to_string(index=False))

if __name__ == "__main__":
    main()
