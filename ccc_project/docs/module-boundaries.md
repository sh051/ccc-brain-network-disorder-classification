# Module boundaries

## Shared data contract

All model teams should use a subject identifier, diagnostic label, acquisition site, and a clearly versioned feature input. A CCC matrix must be 200 x 200, ordered by the same CC200 ROI convention, with a zero diagonal.

## BrainNetCNN v1

`src/models/brainnetcnn_v1/` owns the two-channel architecture and its development training loop. Its curated output belongs only in `artifacts/brainnetcnn_v1/`.

## BrainNetCNN v2

An alternative implementation should be added under `src/models/brainnetcnn_v2/` with a separate experiment folder and artifact directory. Do not overwrite v1 outputs.

## ChaosNet

ChaosNet code should be added under `src/models/chaosnet/`, with executable experiments under `experiments/chaosnet/` and outputs under `artifacts/chaosnet/`. Record the exact ChaosFEX feature settings, classifier, splits, and metrics.

## Evaluation rule

Every model should use the same subject split manifest and report AUC, accuracy, sensitivity, specificity, and site distribution. Parameter selection must be separated from final evaluation through nested validation or a held-out test set.
