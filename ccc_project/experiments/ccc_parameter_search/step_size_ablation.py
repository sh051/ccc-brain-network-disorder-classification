import os
import glob
import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

import ETC
from ETC import CCC

# Setup paths
BASE_PATH = r"D:\capstone"
TIMESERIES_ASD_DIR = os.path.join(BASE_PATH, r"data_timeseries\ASD")
TIMESERIES_TDC_DIR = os.path.join(BASE_PATH, r"data_timeseries\TDC")

# Select 60 subjects (30 ASD, 30 TDC)
asd_files = sorted(glob.glob(os.path.join(TIMESERIES_ASD_DIR, "*.1D")))[:30]
tdc_files = sorted(glob.glob(os.path.join(TIMESERIES_TDC_DIR, "*.1D")))[:30]
subset_paths = asd_files + tdc_files
subset_labels = [1.0] * 30 + [0.0] * 30

SEED = 42

# Helper extraction function
def extract_ccc_features(file_path, L, w, bins, k):
    data = np.loadtxt(file_path)
    n_rois = data.shape[1]
    C = np.zeros((n_rois, n_rois))
    
    # Compute directed CCC
    for i in range(n_rois):
        for j in range(n_rois):
            if i != j:
                C[i, j] = CCC.compute(
                    seq_x=list(data[:, i]),
                    seq_y=list(data[:, j]),
                    LEN_past=L,
                    ADD_meas=w,
                    STEP_size=k,
                    n_partitions=bins
                )
                
    # Subject-wise Z-scoring
    mask = ~np.eye(n_rois, dtype=bool)
    mean = C[mask].mean()
    std = C[mask].std() + 1e-9
    C = (C - mean) / std
    np.fill_diagonal(C, 0.0)
    
    # Clip negative values
    C = np.maximum(C, 0.0)
    
    # Proportional thresholding (keeps top 15% connections)
    flat_vals = C[mask]
    cutoff = np.percentile(flat_vals, 85)
    C[C < cutoff] = 0.0
    
    return C[mask]

# Evaluation function
def evaluate_config(L, w, bins, k, n_cores):
    start = time.time()
    print(f"\nExtracting: L={L}, w={w}, bins={bins}, k={k}...")
    
    X = Parallel(n_jobs=n_cores)(
        delayed(extract_ccc_features)(path, L, w, bins, k)
        for path in subset_paths
    )
    X = np.array(X)
    y = np.array(subset_labels)
    
    # Stratified 5-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    accs = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=SEED)
        clf.fit(X_train, y_train)
        
        preds_prob = clf.predict_proba(X_val)[:, 1]
        preds_class = clf.predict(X_val)
        
        aucs.append(roc_auc_score(y_val, preds_prob))
        accs.append(accuracy_score(y_val, preds_class))
        
    elapsed = time.time() - start
    print(f"--> Result: Mean AUC = {np.mean(aucs):.4f} | Mean Accuracy = {np.mean(accs):.4f} (took {elapsed:.1f}s)")
    return np.mean(aucs), np.mean(accs)

def run_test():
    n_cores = max(1, os.cpu_count() - 4)
    print("="*60)
    print("RUNNING k-VALUE COMPARISON TEST")
    print("="*60)
    
    results = {}
    
    # 1. Evaluate Winner (L=35, w=10, bins=6) with k=8 and k=5
    results["Winner_k8"] = evaluate_config(L=35, w=10, bins=6, k=8, n_cores=n_cores)
    results["Winner_k5"] = evaluate_config(L=35, w=10, bins=6, k=5, n_cores=n_cores)
    
    # 2. Evaluate Baseline (L=50, w=10, bins=8) with k=8 and k=5
    results["Baseline_k8"] = evaluate_config(L=50, w=10, bins=8, k=8, n_cores=n_cores)
    results["Baseline_k5"] = evaluate_config(L=50, w=10, bins=8, k=5, n_cores=n_cores)
    
    print("\n" + "="*60)
    print("TEST COMPLETED - RESULTS SUMMARY")
    print("="*60)
    print(f"Winner (L=35, w=10, bins=6) at k=8: AUC = {results['Winner_k8'][0]:.4f} | Acc = {results['Winner_k8'][1]:.4f}")
    print(f"Winner (L=35, w=10, bins=6) at k=5: AUC = {results['Winner_k5'][0]:.4f} | Acc = {results['Winner_k5'][1]:.4f}")
    print("-"*60)
    print(f"Baseline (L=50, w=10, bins=8) at k=8: AUC = {results['Baseline_k8'][0]:.4f} | Acc = {results['Baseline_k8'][1]:.4f}")
    print(f"Baseline (L=50, w=10, bins=8) at k=5: AUC = {results['Baseline_k5'][0]:.4f} | Acc = {results['Baseline_k5'][1]:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_test()
