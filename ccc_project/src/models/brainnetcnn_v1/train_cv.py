import os
import glob
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from brainnetcnn_model import BrainNetCNN

# Configuration
DATA_DIR = r"D:\capstone\data-1035\Capstone Dataset\ABIDE1"
ASD_DIR = os.path.join(DATA_DIR, "ASD")
TDC_DIR = os.path.join(DATA_DIR, "TDC")
RESULTS_CSV = r"D:\capstone\results\group_analysis\brainnetcnn_cv_results_1035.csv"

N_ROIS = 200
N_FOLDS = 5
EPOCHS = 100
LR = 5e-4
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 16
PATIENCE = 15
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for training: {DEVICE}")

class CCCDataset(Dataset):
    """
    Custom Dataset to load CCC CSV matrices on-the-fly and perform:
    1. Subject-wise Z-scoring (centers and scales each patient's brain network).
    2. Negative clipping (removes noise/distractors).
    3. Proportional thresholding (keeps top 15% connections).
    4. Two-Channel decomposition:
       - Channel 0: Symmetric part S = (C + C^T) / 2 (Overall coupling strength)
       - Channel 1: Asymmetric part A = (C - C^T) / 2 (Net causal flow direction)
    """
    def __init__(self, paths, labels, threshold=0.15):
        self.paths = paths
        self.labels = labels
        self.threshold = threshold

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 1. Load CSV file
        mat = np.loadtxt(self.paths[idx], delimiter=",").astype(np.float32)
        
        # 2. Subject-wise Z-scoring (excluding diagonal)
        mask = ~np.eye(mat.shape[0], dtype=bool)
        mean = mat[mask].mean()
        std = mat[mask].std() + 1e-9
        mat = (mat - mean) / std
        np.fill_diagonal(mat, 0.0)
        
        # 3. Clip negative values to 0 (keeps only positive directed causal flow)
        mat = np.maximum(mat, 0.0)
        
        # 4. Proportional Thresholding (keep top 15% connections)
        if self.threshold is not None:
            flat_vals = mat[mask]
            cutoff = np.percentile(flat_vals, 100 * (1 - self.threshold))
            mat[mat < cutoff] = 0.0
            
        # 5. Extract Symmetric and Asymmetric channels
        S = (mat + mat.T) / 2.0
        A = (mat - mat.T) / 2.0
        
        # Convert to torch tensors and stack into 2 channels -> (2, 200, 200)
        S_tensor = torch.from_numpy(S)
        A_tensor = torch.from_numpy(A)
        mat_tensor = torch.stack([S_tensor, A_tensor], dim=0)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return mat_tensor, label

def train_one_fold(train_ds, val_ds, fold):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = BrainNetCNN(n_rois=N_ROIS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()
                preds.extend(out)
                labels.extend(y.numpy())

        val_auc = roc_auc_score(labels, preds)
        epoch_time = time.time() - epoch_start
        print(f"  epoch {epoch + 1:3d}/{EPOCHS}  train_loss={train_loss:.4f}  val_auc={val_auc:.3f}  ({epoch_time:.1f}s)")

        # Save state based on best validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    model.load_state_dict(best_state)
    return model

def evaluate(model, val_ds):
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            out = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()
            preds.extend(out)
            labels.extend(y.numpy())

    preds_binary = (np.array(preds) > 0.5).astype(int)
    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, preds_binary)
    tn, fp, fn, tp = confusion_matrix(labels, preds_binary).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return auc, acc, sens, spec

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load paths dynamically from data directories
    print("Loading dataset matrix paths...")
    asd_paths = sorted(glob.glob(os.path.join(ASD_DIR, "*.csv")))
    tdc_paths = sorted(glob.glob(os.path.join(TDC_DIR, "*.csv")))

    print(f"Found {len(asd_paths)} ASD files and {len(tdc_paths)} TDC files.")
    paths = np.array(asd_paths + tdc_paths)
    labels = np.array([1.0] * len(asd_paths) + [0.0] * len(tdc_paths))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        print(f"\n=== Fold {fold + 1}/{N_FOLDS} ===")

        train_paths, val_paths = paths[train_idx], paths[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        # Datasets automatically handle Z-scoring, thresholding, and 2-channel stacking
        train_ds = CCCDataset(train_paths, train_labels, threshold=0.15)
        val_ds = CCCDataset(val_paths, val_labels, threshold=0.15)

        model = train_one_fold(train_ds, val_ds, fold + 1)
        auc, acc, sens, spec = evaluate(model, val_ds)

        print(f"Fold {fold + 1} Final Results: AUC={auc:.3f}  Acc={acc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")
        fold_metrics.append({"fold": fold + 1, "auc": auc, "acc": acc, "sens": sens, "spec": spec})

    results_df = pd.DataFrame(fold_metrics)
    print("\n=== 5-fold CV Summary on 1,035 subjects (mean +/- std) ===")
    print(results_df[["auc", "acc", "sens", "spec"]].agg(["mean", "std"]))

    # Save outputs
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved per-fold results to {RESULTS_CSV}")

if __name__ == "__main__":
    main()
