import sys
from pathlib import Path

import torch


MODEL_DIR = Path(__file__).resolve().parents[1] / "src" / "models" / "brainnetcnn_v1"
sys.path.insert(0, str(MODEL_DIR))

from brainnetcnn_model import BrainNetCNN


def test_brainnetcnn_returns_one_logit_per_subject():
    model = BrainNetCNN(n_rois=200)
    inputs = torch.randn(2, 2, 200, 200)

    logits = model(inputs)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
