"""Tiling constants for denoise inference."""

# Per-tile activation memory scales with tile area; FBCNN is deeper than SCUNet.
FBCNN_TILE_SIZE = 256
FBCNN_TILE_OVERLAP = 32

# Multi-pass pipelines spill intermediate results to disk so only one full frame
# is held in RAM between passes (not two float32 copies).
MULTIPASS_SPILL_ENABLED = True
