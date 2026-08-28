# AI-Based Diagnostic Classification of Mental Disorders using Causality of Brain Networks

> **Status: Active Research / Work in Progress**
> This repository is an ongoing project. Pipelines are functional end-to-end, but results are **preliminary and not yet clinically or statistically meaningful**. Model architectures, feature engineering, and normalization strategies are actively being revised. Expect frequent breaking changes.

**Project Mentor:** Dr. Chaitra N
**Team:** Shraddha Rao, Samyuktha S, Shashank R S, S Natisha

---

## Objective

Classify Autism Spectrum Disorder (ASD) vs. Typically Developing Controls (TDC) using **causal** (not just correlational) brain network connectivity, derived from resting-state fMRI in the 1,035-subject ABIDE I dataset.

We use **Compression-Complexity Causality (CCC)** — a measure of whether including a region's past activity improves prediction of another region's future activity — to build directed 200×200 brain connectivity matrices per subject, then feed these into two different classifiers.

---

## Current State: Experimental

Both downstream classifiers are still in early, unoptimized stages. Neither currently separates ASD from TDC reliably:

| Pipeline | Mean AUC | Mean Accuracy | Status |
|---|---|---|---|
| **BrainNetCNN** (2-channel) | 0.6295 ± 0.022 | 59.6% ± 2.1% | Better than chance, high fold variance, under active tuning |
| **ChaosNet** (ChaosFEX + centroid) | 0.5125 ± 0.035 | 50.8% ± 3.6% | Near chance-level, using un-tuned MNIST-default parameters |

**Key open problem:** CCC values across the dataset are heavily centered near zero and overlap substantially between ASD and TDC subjects (ASD mean 0.208 vs. TDC mean 0.201 on % positive edges — nearly identical). 1,011 of 1,035 subjects are "mostly negative" CCC, 0 are "mostly positive." We are actively investigating whether this reflects a real neurological signal, a preprocessing artifact, or a parameter/thresholding issue.

We are **not yet claiming a working diagnostic model** — this repo currently documents an evolving experimental pipeline.

---

## What We're Actively Experimenting With

- **Negative CCC interpretation:** unresolved whether negative values indicate reverse causality or simply no useful causal signal (candidate for clipping to zero as noise).
- **Diagonal handling:** currently zeroed out; considering whether to exclude self-connections from analysis entirely instead.
- **Phenotypic data fusion:** exploring injecting phenotypic covariates alongside CCC matrices into BrainNetCNN.
- **Multi-site normalization:** ABIDE spans 17 acquisition sites; subject-wise Z-scoring helped reduce validation variance, but we're evaluating site-aware normalization alternatives.
- **CCC hyperparameters:** grid search over history length (L), causal window (w), and bins on a 60-subject subset pointed to L=35, w=10, bins=6 as a promising configuration — this has **not yet been validated on the full 1,035-subject set**.
- **ChaosNet tuning:** currently using default skew-tent map parameters (q=0.336, b=0.331) inherited from MNIST benchmarks — these need to be re-tuned for ABIDE's connectivity feature space.
- **Filter sensitivity checks:** compared `filt_global`, `filt_noglobal`, `nofilt_global`, `nofilt_noglobal` CPAC variants to rule out band-filtering as the cause of near-zero CCC values — all variants produced similar near-zero distributions, so this is likely not the root cause.

---

> Raw ABIDE time-series and subject-level CCC matrices are **not pushed to GitHub** — only source code, documentation, curated summary metrics, and figures. ABIDE data is subject to its own data-use terms.

---

## Project Workflow

1. **Data collection & preparation**
   ABIDE I resting-state fMRI, CPAC `filt_global` preprocessing, CC200 atlas. Cohort: 1,035 subjects (505 ASD, 530 TDC).

2. **Data input**
   Each subject represented as a CC200 ROI time-series (200 brain regions).

3. **Feature construction**
   - Compute directed 200×200 CCC matrix per subject (via `ETCpy`).
   - Exclude self-connections (diagonal).
   - For BrainNetCNN: subject-wise Z-score off-diagonal values, clip negatives, retain top 15% of edges, split into symmetric/antisymmetric channels.
   - For ChaosNet: expand each connectivity value via ChaosFEX (TT, Energy, TTSS, Entropy) using a skew-tent chaotic map.

4. **Parameter tuning**
   Grid search over (L, w, bins) on a balanced 60-subject subset, using L2-regularized logistic regression + stratified 5-fold CV as an evaluation proxy. Current best: **L=35, w=10, bins=6** — pending full-dataset validation.

5. **Model training & classification**
   - **BrainNetCNN:** Edge-to-Edge → Edge-to-Node → Node-to-Graph → MLP classifier. Adam (lr=1e-4), BCE loss, stratified 5-fold CV, early stopping (patience 15, monitoring validation AUC).
   - **ChaosNet:** Builds ASD/TDC centroids from ChaosFEX features, classifies via cosine similarity to nearest centroid. Stratified 5-fold CV.

6. **Data visualization & interpretation**
   - Mean CCC heatmaps (ASD vs. TDC)
   - ROI-level connectivity-difference plots
   - Edge-wise Welch t-tests with Benjamini-Hochberg FDR correction
   - **Current finding:** no individual edges survive FDR correction — figures are descriptive, not confirmatory.

7. **Storage**
   - Raw time series → `data_timeseries/` (local only)
   - Precomputed CCC matrices → `data-1035/` (local only)
   - Curated metrics/figures → `results/group_analysis/`
   - GitHub → code, docs, selected metrics, and figures only

---

## Technologies Used

| Category | Technology | Status |
|---|---|---|
| Language | Python | Open-source |
| Dataset | ABIDE I | Public research dataset; subject to its data-use terms |
| Preprocessing | CPAC pipeline | Open-source |
| Brain atlas | CC200 atlas | Research resource |
| CCC computation | ETCPy / `ETC` | Open-source |
| Deep learning | PyTorch | Open-source |
| ML / validation | scikit-learn | Open-source |
| Data handling | NumPy, pandas | Open-source |
| Parallel processing | joblib | Open-source |
| Statistics | SciPy, statsmodels | Open-source |
| Visualization | Matplotlib, Seaborn, NetworkX | Open-source |
| Version control | Git / GitHub | Git open-source; GitHub is a hosted service |
| Hardware acceleration | CUDA-capable GPU (optional) | NVIDIA proprietary platform |

---

---

## Next Steps

- [ ] Resolve negative-CCC interpretation and decide on clipping strategy
- [ ] Re-run (L, w, bins) grid search validation on the full 1,035-subject dataset
- [ ] Tune ChaosNet skew-tent parameters (q, b) specifically for ABIDE connectivity features, rather than inherited MNIST defaults
- [ ] Evaluate diagonal-exclusion vs. zeroing
- [ ] Prototype phenotypic-data fusion into BrainNetCNN
- [ ] Investigate multi-site normalization alternatives to subject-wise Z-scoring
- [ ] Reduce BrainNetCNN cross-fold variance (Fold 5 sensitivity dropped to 26.7%)

---

## License / Data Usage

Code in this repository is shared for research and educational purposes. ABIDE I data is **not redistributed** here and must be obtained directly from the [Preprocessed Connectomes Project](http://preprocessed-connectomes-project.org/abide/) under its own terms of use.

---

*This README reflects the project's status as of the most recent update and will be revised as pipelines evolve.*
