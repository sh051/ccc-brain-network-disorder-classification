import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# =====================================================
# PATHS
# =====================================================

DATA_DIR = r"D:\capstone\results\v1\ccc_outputs\set1\ASD"

OUTPUT_DIR = os.path.join(DATA_DIR, "group_analysis_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD CSV MATRICES
# =====================================================

matrices = []
subject_names = []

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

print(f"Found {len(csv_files)} CSV files")

for file in csv_files:

    try:
        matrix = pd.read_csv(file, header=None).values

        if matrix.shape != (200, 200):
            print(f"Skipping {file} : shape = {matrix.shape}")
            continue

        matrices.append(matrix)

        subject_name = os.path.basename(file).replace(".csv", "")
        subject_names.append(subject_name)

    except Exception as e:
        print(f"Error loading {file}: {e}")

matrices = np.array(matrices)

print(f"\nLoaded matrices shape: {matrices.shape}")

# =====================================================
# LOAD JSON STATS
# =====================================================

stats_data = []

json_files = glob.glob(os.path.join(DATA_DIR, "*_stats.json"))

print(f"Found {len(json_files)} JSON stat files")

for file in json_files:

    try:
        with open(file, "r") as f:
            stats = json.load(f)

        stats_data.append(stats)

    except Exception as e:
        print(f"Error reading {file}: {e}")

stats_df = pd.DataFrame(stats_data)

# Save combined stats
stats_df.to_csv(
    os.path.join(OUTPUT_DIR, "combined_subject_statistics.csv"),
    index=False
)

# =====================================================
# MEAN MATRIX
# =====================================================

mean_matrix = np.mean(matrices, axis=0)

pd.DataFrame(mean_matrix).to_csv(
    os.path.join(OUTPUT_DIR, "mean_asd_matrix.csv"),
    header=False,
    index=False
)

# =====================================================
# MEAN HEATMAP
# =====================================================

plt.figure(figsize=(12, 10))

sns.heatmap(
    mean_matrix,
    cmap="coolwarm"
)

plt.title("Mean ASD CCC Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "mean_asd_heatmap.png"),
    dpi=300
)

plt.close()

# =====================================================
# VARIANCE MATRIX
# =====================================================

variance_matrix = np.var(matrices, axis=0)

plt.figure(figsize=(12, 10))

sns.heatmap(
    variance_matrix,
    cmap="viridis"
)

plt.title("Variance Across ASD Subjects")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "variance_heatmap.png"),
    dpi=300
)

plt.close()

# =====================================================
# ASYMMETRY DISTRIBUTION
# =====================================================

plt.figure(figsize=(8, 6))

plt.hist(
    stats_df["asymmetry"],
    bins=20
)

plt.xlabel("Asymmetry Index")
plt.ylabel("Number of Subjects")
plt.title("ASD Asymmetry Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "asymmetry_distribution.png"),
    dpi=300
)

plt.close()

# =====================================================
# GLOBAL CCC VALUE DISTRIBUTION
# =====================================================

all_values = matrices.flatten()

plt.figure(figsize=(8, 6))

plt.hist(
    all_values,
    bins=100,
    density=True
)

plt.xlabel("CCC Value")
plt.ylabel("Density")
plt.title("Global CCC Distribution (ASD)")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "ccc_distribution.png"),
    dpi=300
)

plt.close()

# =====================================================
# ROI IMPORTANCE ANALYSIS
# =====================================================

# Outgoing influence
outgoing_strength = np.mean(
    np.sum(mean_matrix, axis=0)
)

# Incoming influence
incoming_strength = np.mean(
    np.sum(mean_matrix, axis=1)
)

roi_outgoing = np.sum(mean_matrix, axis=0)
roi_incoming = np.sum(mean_matrix, axis=1)

roi_df = pd.DataFrame({
    "ROI": np.arange(200),
    "Outgoing_Strength": roi_outgoing,
    "Incoming_Strength": roi_incoming
})

roi_df.to_csv(
    os.path.join(OUTPUT_DIR, "roi_strengths.csv"),
    index=False
)

# =====================================================
# TOP 20 ROI PLOT
# =====================================================

top_rois = roi_df.sort_values(
    by="Outgoing_Strength",
    ascending=False
).head(20)

plt.figure(figsize=(12, 6))

plt.bar(
    top_rois["ROI"].astype(str),
    top_rois["Outgoing_Strength"]
)

plt.xlabel("ROI")
plt.ylabel("Outgoing Strength")
plt.title("Top 20 Most Influential ROIs")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "top_roi_strengths.png"),
    dpi=300
)

plt.close()

# =====================================================
# NETWORK GRAPH
# =====================================================

threshold = np.percentile(mean_matrix, 95)

G = nx.DiGraph()

for i in range(200):
    G.add_node(i)

for i in range(200):
    for j in range(200):

        if mean_matrix[i, j] > threshold:
            G.add_edge(j, i, weight=mean_matrix[i, j])

plt.figure(figsize=(14, 14))

pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=40
)

nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    alpha=0.4
)

plt.title("Top Directed Connections (ASD Mean Matrix)")

plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "asd_network_graph.png"),
    dpi=300
)

plt.close()

# =====================================================
# SUMMARY STATISTICS
# =====================================================

summary = {
    "num_subjects": len(matrices),
    "global_mean_ccc": float(np.mean(all_values)),
    "global_std_ccc": float(np.std(all_values)),
    "mean_asymmetry": float(stats_df["asymmetry"].mean()),
    "std_asymmetry": float(stats_df["asymmetry"].std()),
    "mean_positive_edges": float(stats_df["positive"].mean()),
    "mean_negative_edges": float(stats_df["negative"].mean())
}

summary_df = pd.DataFrame(
    summary.items(),
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "group_summary_statistics.csv"),
    index=False
)

# =====================================================
# DONE
# =====================================================

print("\n====================================")
print("ASD GROUP ANALYSIS COMPLETE")
print("====================================")
print(f"Subjects Processed : {len(matrices)}")
print(f"Outputs saved to   : {OUTPUT_DIR}")
print("====================================")


#The average CCC matrix exhibits distinct ROI-specific connectivity patterns, with several regions showing consistently elevated influence and variability across subjects. The variance analysis highlights a subset of ROIs that contribute disproportionately to inter-subject differences, suggesting potential hubs of altered information flow.