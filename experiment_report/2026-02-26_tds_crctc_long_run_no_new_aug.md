# Experiment report: TDS + CR-CTC long run (80 epochs) — no new augmentation

## Summary

| Field | Value |
|-------|--------|
| **Date** | 2026-02-26 |
| **Model** | TDSConvCTCModule + CR-CTC (tds_conv_crctc) |
| **Best val CER** | **~20.6%** (late epochs 75–79) |
| **Best val loss** | ~0.95 |
| **Data augmentation** | **Did not use** AdditiveGaussianNoise or ChannelDropout (run predates those additions) |

---

## Important note: augmentation

This training run **did not use** the new data augmentation methods added later to the codebase:

- **AdditiveGaussianNoise** (noise std = fraction of signal std)
- **ChannelDropout** (per-channel dropout)

The train pipeline for this run was: ToTensor → band_rotation → temporal_jitter → logspec → specaug only. Future runs with `config/transforms/log_spectrogram.yaml` as of 2026-02-26 also include `noise` and `channel_dropout` in the train pipeline; this run does not.

---

## Training setup

| Setting | Value |
|---------|--------|
| Optimizer | AdamW (lr in log) |
| LR schedule | ReduceLROnPlateau (0.001 → 0.0001 at ~epoch 57, → 1e-05 at ~epoch 77) |
| Epochs | 0–79 (80 epochs total) |
| Metrics | val/CER, val/loss, train/CER, train/loss, IER, DER, SER |

---

## Metrics (from metrics CSV)

- **Validation CER:** Started at 100% (epoch 0), improved to ~22–24% by epoch 27–28, then to **~20.6–20.9%** from epoch 57 onward and **stable** through epoch 79.
- **Validation loss:** ~3.3 (epoch 0) down to **~0.95** in late epochs; stable in the 0.95–0.96 range for epochs 75–79.
- **Train CER:** Dropped to **~3.5–4%** by epochs 75–79.
- **Train–val gap:** Large (train CER ~3.5% vs val CER ~20.7%), consistent with overfitting; validation itself did not degrade in the final epochs.

---

## Learning-rate schedule (from log)

- **0.001:** epochs 0–56  
- **0.0001:** epochs 57–76  
- **1e-05:** epochs 77–79  

---

## Conclusion

- Best validation performance is **~20.6% CER**; use the checkpoint with best val CER or lowest val loss (likely around epochs 75–78), not the final epoch.
- This run is a **baseline without** the new noise and channel-dropout augmentations; comparison with a run that uses them can show whether those augmentations improve generalization or reduce overfitting.
