# BrainNetCNN v1 - CCC Brain-Network ASD Classification

This module contains my first organized BrainNetCNN baseline for ASD versus TDC classification from directed Compression-Complexity Causality (CCC) brain-network matrices.

## Scope

- **Cohort:** 1,035 ABIDE I participants (505 ASD, 530 TDC).
- **Input:** precomputed signed 200 x 200 CCC matrices derived from CPAC `filt_global` CC200 ROI time series.
- **Model input:** each subject matrix is z-scored off diagonal, negative values are clipped, the top 15% of edges are retained, and symmetric/asymmetric channels are passed to BrainNetCNN.
- **Result:** 5-fold development evaluation gives mean AUC 0.629 and mean accuracy 0.596.

The complete full-cohort CCC matrix-construction script is not currently included. `notebooks/legacy/ccc_exploration.ipynb` records an earlier ASD-only CCC implementation and should be treated as a legacy reference, not as the end-to-end full-cohort pipeline.

## Layout

```text
src/data/                         ABIDE/CPAC time-series download helper
src/models/brainnetcnn_v1/        architecture and training script
src/analysis/                     descriptive group-level analysis
experiments/ccc_parameter_search/ CCC parameter search and step-size ablation
notebooks/legacy/                 earlier CCC exploration
tests/                            fast architecture-level tests
artifacts/                        curated metrics and descriptive figures
docs/                             module boundaries and hand-off notes
```

## Curated results

- `artifacts/brainnetcnn_v1/brainnetcnn_cv_results_1035.csv`: per-fold BrainNetCNN development metrics.
- `artifacts/ccc_parameter_search/ccc_grid_search_results.csv`: 60-subject exploratory CCC parameter search. The best of seven tested configurations was `L=35`, `w=10`, `bins=6` with mean AUC 0.883 (SD 0.077).
- `artifacts/group_analysis/`: descriptive raw-CCC group heatmaps and altered-ROI plot.

The parameter-search result is exploratory because selection and evaluation used the same 60 subjects. BrainNetCNN validation folds are also used for early stopping and final scoring, so the reported result is not an independent held-out clinical-performance estimate.

## Run notes

The scripts preserve original local Windows data paths and require locally obtained ABIDE derivatives plus ETCPy, PyTorch, NumPy, pandas, scikit-learn, SciPy, statsmodels, Matplotlib, Seaborn, NetworkX, and joblib. Configure paths locally before running.

The included group-level analysis found no edges surviving Benjamini-Hochberg FDR correction at q < 0.05. Its figures are descriptive, not claims of statistically significant connectivity differences.
