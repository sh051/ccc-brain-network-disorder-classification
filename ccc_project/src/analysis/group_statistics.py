import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Prevents GUI thread errors on headless Windows
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Create output directory
OUTPUT_DIR = r"D:\capstone\results\group_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data paths
DATA_DIR = r"D:\capstone\data-1035\Capstone Dataset\ABIDE1"
ASD_DIR = os.path.join(DATA_DIR, "ASD")
TDC_DIR = os.path.join(DATA_DIR, "TDC")

# =====================================================
# 1. LOAD MATRICES
# =====================================================

def load_group_matrices(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    matrices = []
    print(f"Loading {len(files)} CSV files from {os.path.basename(directory)}...")
    for file in files:
        try:
            matrix = np.loadtxt(file, delimiter=",").astype(np.float32)
            if matrix.shape == (200, 200):
                matrices.append(matrix)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return np.array(matrices)

asd_mats = load_group_matrices(ASD_DIR)
tdc_mats = load_group_matrices(TDC_DIR)
print(f"Successfully loaded {len(asd_mats)} ASD matrices and {len(tdc_mats)} TDC matrices.")

# =====================================================
# 2. EDGE-WISE T-TESTS
# =====================================================

print("\nComputing edge-wise t-tests between ASD and TDC groups...")
n_nodes = 200
p_values = np.ones((n_nodes, n_nodes))
t_stats = np.zeros((n_nodes, n_nodes))

for i in range(n_nodes):
    for j in range(n_nodes):
        if i != j:
            asd_vals = asd_mats[:, i, j]
            tdc_vals = tdc_mats[:, i, j]
            t_stat, p_val = ttest_ind(asd_vals, tdc_vals, equal_var=False)
            if np.isnan(t_stat):
                t_stat, p_val = 0.0, 1.0
            t_stats[i, j] = t_stat
            p_values[i, j] = p_val

# Apply FDR correction
print("Applying False Discovery Rate (FDR) correction...")
flat_p = p_values[~np.eye(n_nodes, dtype=bool)]
reject, q_values_flat, _, _ = multipletests(flat_p, alpha=0.05, method='fdr_bh')

q_values = np.ones((n_nodes, n_nodes))
idx = 0
for i in range(n_nodes):
    for j in range(n_nodes):
        if i != j:
            q_values[i, j] = q_values_flat[idx]
            idx += 1

print(f"Number of edges passing FDR correction (q < 0.05): {np.sum(q_values < 0.05)}")

# Save t-test results to CSV
results_list = []
for i in range(n_nodes):
    for j in range(n_nodes):
        if i != j:
            results_list.append({
                "ROI_i": i, "ROI_j": j,
                "t_stat": t_stats[i, j], "p_value": p_values[i, j], "q_value": q_values[i, j],
                "mean_asd": asd_mats[:, i, j].mean(), "mean_tdc": tdc_mats[:, i, j].mean()
            })
df_results = pd.DataFrame(results_list)
df_results.to_csv(os.path.join(OUTPUT_DIR, "t_test_results.csv"), index=False)
print(f"Saved complete t-test results to {os.path.join(OUTPUT_DIR, 't_test_results.csv')}")

# =====================================================
# 3. VISUALIZATIONS
# =====================================================

# A. Group Heatmaps
mean_asd = asd_mats.mean(axis=0)
mean_tdc = tdc_mats.mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(mean_asd, ax=axes[0], cmap="viridis", cbar=True, square=True)
axes[0].set_title("Mean ASD Connectivity Matrix")
sns.heatmap(mean_tdc, ax=axes[1], cmap="viridis", cbar=True, square=True)
axes[1].set_title("Mean TDC Connectivity Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "group_heatmaps.png"), dpi=300)
plt.close()
print(f"Saved group heatmaps to {os.path.join(OUTPUT_DIR, 'group_heatmaps.png')}")

# B. Altered Hub ROIs (Alteration Centrality)
# Compute out-degree and in-degree difference
diff_mat = mean_asd - mean_tdc
abs_diff_mat = np.abs(diff_mat)

out_degree_diff = abs_diff_mat.sum(axis=1)
in_degree_diff = abs_diff_mat.sum(axis=0)

# Get top 20 altered ROIs
top_out_rois = np.argsort(out_degree_diff)[-20:]
top_in_rois = np.argsort(in_degree_diff)[-20:]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(range(20), out_degree_diff[top_out_rois], align='center', color='coral')
plt.yticks(range(20), top_out_rois)
plt.xlabel("Cumulative Outgoing Alteration")
plt.ylabel("ROI Index")
plt.title("Top 20 Outgoing Altered Hubs")

plt.subplot(1, 2, 2)
plt.barh(range(20), in_degree_diff[top_in_rois], align='center', color='skyblue')
plt.yticks(range(20), top_in_rois)
plt.xlabel("Cumulative Incoming Alteration")
plt.ylabel("ROI Index")
plt.title("Top 20 Incoming Altered Hubs")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "altered_rois.png"), dpi=300)
plt.close()
print(f"Saved hub ROI alterations plot to {os.path.join(OUTPUT_DIR, 'altered_rois.png')}")

# C. Significant Directed Network Graph (Top 10% connection differences)
plt.figure(figsize=(10, 10))
G = nx.DiGraph()

# Threshold difference matrix to top 10% most different connections
flat_diff = abs_diff_mat[~np.eye(n_nodes, dtype=bool)]
cutoff = np.percentile(flat_diff, 99.8) # Keep top 0.2% for visualization to avoid cluttering

for i in range(n_nodes):
    for j in range(n_nodes):
        if i != j and abs_diff_mat[i, j] > cutoff:
            G.add_edge(i, j, weight=diff_mat[i, j])

pos = nx.circular_layout(G)
# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

edges = G.edges(data=True)
positive_edges = [(u, v) for u, v, d in edges if d['weight'] > 0]
negative_edges = [(u, v) for u, v, d in edges if d['weight'] < 0]

nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color='red', width=2, arrowstyle='->', arrowsize=15)
nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color='blue', width=2, arrowstyle='->', arrowsize=15)

plt.title("Significant Directed Causal Differences Network Graph\n(Red: ASD > TDC, Blue: TDC > ASD)", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "significant_network_graph.png"), dpi=300)
plt.close()
print(f"Saved significant directed network graph to {os.path.join(OUTPUT_DIR, 'significant_network_graph.png')}")
print("\nGroup statistical analysis complete!")
