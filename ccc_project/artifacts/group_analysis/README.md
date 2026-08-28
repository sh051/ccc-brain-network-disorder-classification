# Curated experiment artifacts

This directory contains the small set of outputs included with the public proof-of-work repository.

- `ccc_grid_search_results.csv`: a 60-subject, five-fold parameter search over CCC settings.
- `brainnetcnn_cv_results_1035.csv`: five-fold development metrics from the 1,035-subject BrainNetCNN experiment.
- `group_heatmaps.png`: descriptive mean signed-CCC matrices by diagnostic group.
- `altered_rois.png`: ROIs with the largest aggregate raw directed-connectivity differences.

The group figures are descriptive. The current edge-wise Welch t-test analysis identified no edges surviving Benjamini-Hochberg FDR correction at q < 0.05.

Per-subject matrices, archives, source data, and complete statistical output tables are intentionally excluded because they are derived from the source dataset and make the repository unnecessarily large.
