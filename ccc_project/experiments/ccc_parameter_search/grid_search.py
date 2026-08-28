import os
import glob
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import ETC
from ETC import CCC

# Define paths (pointing directly to the new organized timeseries folders)
BASE_PATH = r"D:\capstone"
TIMESERIES_ASD_DIR = os.path.join(BASE_PATH, r"data_timeseries\ASD")
TIMESERIES_TDC_DIR = os.path.join(BASE_PATH, r"data_timeseries\TDC")
GRID_RESULTS_CSV = os.path.join(BASE_PATH, r"results\group_analysis\ccc_grid_search_results.csv")

# Search configurations
SUBSET_SIZE_PER_GROUP = 30  # 30 ASD + 30 TDC = 60 subjects total
STEP_SIZE_SEARCH = 8        # Larger step size (k=8) speeds up grid search extraction by ~60%
SEED = 42

# 1. SELECT BALANCED SUBSET (30 ASD and 30 TDC together)
print("Selecting balanced subset from organized timeseries folders...")
asd_files = sorted(glob.glob(os.path.join(TIMESERIES_ASD_DIR, "*.1D")))[:SUBSET_SIZE_PER_GROUP]
tdc_files = sorted(glob.glob(os.path.join(TIMESERIES_TDC_DIR, "*.1D")))[:SUBSET_SIZE_PER_GROUP]

if len(asd_files) < SUBSET_SIZE_PER_GROUP or len(tdc_files) < SUBSET_SIZE_PER_GROUP:
    print(f"Error: Not enough timeseries files. Found {len(asd_files)} ASD, {len(tdc_files)} TDC.")
    exit(1)

subset_paths = asd_files + tdc_files
subset_labels = [1.0] * len(asd_files) + [0.0] * len(tdc_files)
print(f"Selected {len(subset_paths)} subjects (30 ASD, 30 TDC) for grid search.")

# 2. DEFINE THE PARAMETER SEARCH SPACE
# We test different memory lengths (L), prediction windows (w), and discretization partitions (bins).
grid = [
    {"L": 35, "w": 5,  "bins": 4},
    {"L": 35, "w": 5,  "bins": 6},
    {"L": 35, "w": 10, "bins": 6},
    {"L": 50, "w": 10, "bins": 6},
    {"L": 50, "w": 10, "bins": 8},  # Baseline parameter set
    {"L": 65, "w": 15, "bins": 6},
    {"L": 65, "w": 15, "bins": 8},
]

# 3. MATRIX COMPUTATION FUNCTION FOR SEARCH
def compute_matrix_features(file_path, L, w, bins, threshold=0.15):
    """
    Computes a single subject's Z-scored, clipped, and thresholded CCC matrix,
    returning a flattened 1D feature vector of the off-diagonal edges.
    """
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
                    STEP_size=STEP_SIZE_SEARCH,
                    n_partitions=bins
                )
                
    # Subject-wise Z-scoring
    mask = ~np.eye(n_rois, dtype=bool)
    mean = C[mask].mean()
    std = C[mask].std() + 1e-9
    C = (C - mean) / std
    np.fill_diagonal(C, 0.0)
    
    # Clip negative causal flow values to 0 (removes reverse noise)
    C = np.maximum(C, 0.0)
    
    # Proportional thresholding (keeps top 15% connections)
    if threshold is not None:
        flat_vals = C[mask]
        cutoff = np.percentile(flat_vals, 100 * (1 - threshold))
        C[C < cutoff] = 0.0
        
    return C[mask] # Return off-diagonal edges as 1D features (size: 39,800)

# 4. GRID SEARCH EXECUTION
def run_grid_search():
    n_cores = max(1, os.cpu_count() - 4)  # Leave 4 cores free for system stability
    print(f"Starting Grid Search in parallel using {n_cores} CPU cores...")
    
    results = []
    
    for idx, config in enumerate(grid):
        start_time = time.time()
        print(f"\n[{idx+1}/{len(grid)}] Evaluating Config: L={config['L']}, w={config['w']}, bins={config['bins']}")
        
        # Parallel extract subset matrices using current parameters
        X = Parallel(n_jobs=n_cores)(
            delayed(compute_matrix_features)(path, config["L"], config["w"], config["bins"])
            for path in subset_paths
        )
        X = np.array(X)
        y = np.array(subset_labels)
        
        # Fast 5-fold CV evaluation using regularized Logistic Regression
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_aucs = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Simple L2-regularized logistic regression (trains in <0.05 seconds)
            clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=SEED)
            clf.fit(X_train, y_train)
            
            preds = clf.predict_proba(X_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_val, preds))
            
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        elapsed = time.time() - start_time
        
        print(f"  --> Mean 5-Fold AUC: {mean_auc:.4f} ± {std_auc:.4f} (took {elapsed:.1f}s)")
        results.append({**config, "mean_auc": mean_auc, "std_auc": std_auc, "time_seconds": elapsed})

    # Save results summary to CSV
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(GRID_RESULTS_CSV), exist_ok=True)
    df_results.to_csv(GRID_RESULTS_CSV, index=False)
    
    # Identify best configuration
    best_idx = df_res = df_results["mean_auc"].idxmax()
    best_config = df_results.iloc[best_idx]
    
    print("\n" + "="*60)
    print("GRID SEARCH COMPLETED SUCCESSFULLY")
    print("="*60)
    print(df_results.to_string(index=False))
    print(f"\nBest Config: L={int(best_config['L'])}, w={int(best_config['w'])}, bins={int(best_config['bins'])}")
    print(f"Best Mean AUC: {best_config['mean_auc']:.4f} ± {best_config['std_auc']:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_grid_search()
