import torch
import torch.nn as nn

class E2EBlock(nn.Module):
    """
    Edge-to-edge layer: cross-shaped filter, implemented as a row conv
    (1 x n_rois) plus a column conv (n_rois x 1), broadcast-summed back to
    a full n_rois x n_rois map.
    """
    def __init__(self, in_channels, out_channels, n_rois, bias=True):
        super().__init__()
        self.n_rois = n_rois
        self.row_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, n_rois), bias=bias)
        self.col_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(n_rois, 1), bias=bias)

    def forward(self, x):
        row = self.row_conv(x)              # (B, C, n_rois, 1)
        col = self.col_conv(x)              # (B, C, 1, n_rois)
        row = row.repeat(1, 1, 1, self.n_rois)
        col = col.repeat(1, 1, self.n_rois, 1)
        return row + col

class BrainNetCNN(nn.Module):
    """
    BrainNetCNN architecture modified to accept 2-channel inputs:
    Channel 0: Symmetric coupling component S = (C + C^T) / 2
    Channel 1: Asymmetric causality component A = (C - C^T) / 2
    """
    def __init__(self, n_rois=200, e2e_channels=32, e2n_channels=64,
                 n2g_channels=96, dropout=0.4):
        super().__init__()
        self.n_rois = n_rois

        # e2e1 input channels = 2 (for Symmetric and Asymmetric channels)
        self.e2e1 = E2EBlock(2, e2e_channels, n_rois)
        self.e2e2 = E2EBlock(e2e_channels, e2e_channels, n_rois)
        self.act = nn.LeakyReLU(0.33)
        self.dropout1 = nn.Dropout(dropout)

        # Edge-to-Node layer
        self.e2n = nn.Conv2d(e2e_channels, e2n_channels, kernel_size=(1, n_rois))
        self.dropout2 = nn.Dropout(dropout)

        # Node-to-Graph layer
        self.n2g = nn.Conv2d(e2n_channels, n2g_channels, kernel_size=(n_rois, 1))
        self.dropout3 = nn.Dropout(dropout)

        # Fully Connected layers
        self.fc1 = nn.Linear(n2g_channels, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.act(self.e2e1(x))
        x = self.act(self.e2e2(x))
        x = self.dropout1(x)

        x = self.act(self.e2n(x))    # (B, e2n_channels, n_rois, 1)
        x = self.dropout2(x)

        x = self.act(self.n2g(x))    # (B, n2g_channels, 1, 1)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)         # raw logits -> use BCEWithLogitsLoss

if __name__ == "__main__":
    # Shape sanity check for 2-channel input
    model = BrainNetCNN(n_rois=200)
    dummy = torch.randn(4, 2, 200, 200) # Input shape is (B, 2, 200, 200)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (4,)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
