# Experiment report: 2026-02-25 — CNN+Transformer, AdamW, ReduceLROnPlateau

## Summary

| Field | Value |
|-------|--------|
| **Date** | 2026-02-25 |
| **Run** | Colab 07:57:43; log: `4078_0_log (1).err` |
| **Model** | CNNTransformerCTCModule (CNN + Transformer encoder) |
| **Best val CER** | **91.02792%** (at epoch 1, step 180) |
| **Epoch at best** | 1 (of 150) |
| **Final epoch** | 150 |
| **Parameters** | 83.7 M trainable (~334.8 MB) |

---

## Model and code (current version)

- **Architecture:** SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten → **CNNTransformerEncoder** (TemporalCNN → Transformer) → Linear → LogSoftmax (CTC).
- **Config:** `model=cnn_transformer_ctc`, `user=single_user`, `optimizer=adamw`, `lr_scheduler=reduce_on_plateau`, `decoder=ctc_greedy`.

---

## Training setup

| Setting | Value |
|---------|--------|
| max_epochs | 150 |
| batch_size | 32 |
| optimizer | AdamW (lr=1e-3, weight_decay=0.01) |
| lr_scheduler | ReduceLROnPlateau (adaptive; mode=min, factor=0.1, patience=10, min_lr=1e-6; monitor=val/CER) |
| monitor_metric | val/CER (min) |
| seed | 1501 |
| accelerator | GPU (CUDA) |

**CNN+Transformer (config/model/cnn_transformer_ctc.yaml):**

- in_features: 528, mlp_features: [384]
- cnn_layers: 3, cnn_kernel_size: 31
- transformer_layers: 4, n_heads: 8, ff_dim: 4×768, dropout: 0.1
- window_length: 8000, padding: [1800, 200]

---

## Metrics (from log)

- **Epoch 0:** val/CER 93.35401 (best), checkpoint saved at step 60.
- **Epoch 1:** val/CER 91.02792 (best), checkpoint saved at step 180.
- **Epochs 2–149:** No new best; best remained 91.02792.
- **Training end:** `Trainer.fit` stopped: `max_epochs=150` reached.

---

## Environment and notes

- **Platform:** Google Colab (Drive path: `/content/drive/MyDrive/Colab Notebooks/EMG_keyboard_NN/...`).
- **Logger:** TensorBoard logger folder reported missing for this run.
- **Warning (non-fatal):** cuDNN Conv descriptor warning during run; training completed.
- **Comparison:** TDS baseline (same user) reached ~19.07% val CER; this run (CNN+Transformer + AdamW + ReduceLROnPlateau) best 91.03% — much higher CER, likely needs more data, longer training, or different hyperparameters (e.g. revert to Adam + cosine schedule, or try TDS again for comparison).
