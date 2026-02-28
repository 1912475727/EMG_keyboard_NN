# Experiment report: 2026-02-25 run 13-01-52 — CNN+Transformer, AdamW, ReduceLROnPlateau (local)

## Summary

| Field | Value |
|-------|--------|
| **Date** | 2026-02-25 |
| **Run** | Local `13-01-52` — `job0_trainer.devices=1,user=single_user` |
| **Log dir** | `logs/2026-02-25/13-01-52/job0_trainer.devices=1,user=single_user/` |
| **Model** | CNNTransformerCTCModule (CNN + Transformer encoder, positional encoding, src_key_padding_mask) |
| **Best val CER** | **~18.1–18.3%** (over run) |
| **Final val CER** | **~18.3%** (epoch 149) |
| **Final val loss** | ~0.619–0.620 |
| **Epoch at end** | 149 (of 150) |
| **Parameters** | **83.7 M** trainable (~334.8 MB) |

---

## Model structure (from training summary)

| Name | Type | Params |
|------|------|--------|
| front_end | Sequential | 406 K |
| encoder | CNNTransformerEncoder | 83.2 M |
| head | Sequential | 76.1 K |
| ctc_loss | CTCLoss | 0 |
| metrics | ModuleDict | 0 |
| **Total** | | **83.7 M** trainable |

- **front_end:** SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten (spectrogram → per-frame features).
- **encoder:** Temporal CNN (local features) → sinusoidal positional encoding → Transformer encoder (global context); uses `src_key_padding_mask` for variable-length sequences.
- **head:** Linear → LogSoftmax (CTC log-probs).

---

## Model and data parameters (config)

**Module (`config/model/cnn_transformer_ctc.yaml`):**

| Parameter | Value |
|-----------|--------|
| in_features | 528 |
| mlp_features | [384] |
| cnn_layers | 3 |
| cnn_kernel_size | 31 |
| transformer_layers | 4 |
| n_heads | 8 |
| ff_dim | null (default 4×d_model = 3072 for d_model=768) |
| dropout | 0.1 |
| max_len | 5000 |

**Datamodule:**

| Parameter | Value |
|-----------|--------|
| window_length | 8000 |
| padding | [1800, 200] |

---

## Training setup

| Setting | Value |
|---------|--------|
| max_epochs | 150 |
| batch_size | 32 |
| optimizer | AdamW (lr=1e-3, weight_decay=0.01) |
| lr_scheduler | ReduceLROnPlateau (mode=min, factor=0.1, patience=10, min_lr=1e-6; monitor=val/CER) |
| monitor_metric | val/CER (min) |
| seed | 1501 |
| accelerator | GPU (CUDA), 1 device |
| num_workers | 0 (local Windows) |
| cluster | basic (in-process launcher) |

**LR schedule (from metrics):**

- 0.001 → 0.0001 at epoch 61
- 0.0001 → 1e-05 at epoch 114
- 1e-05 → ~1e-06 at epoch 132

---

## CER and related metrics (run 13-01-52)

- **Validation CER:** Start ~94% (epoch 0) → best **~18.1–18.3%** → final **~18.3%** (epoch 149).
- **Validation loss:** ~11.6 → **~0.619–0.620** at end.
- **Train CER (end):** ~13.7–13.8%.
- **Val IER/DER/SER (end):** ~4.5% / ~1.75% / ~12%.

---

## Environment and notes

- **Platform:** Local Windows; `cluster=basic`, `num_workers=0`.
- **Logs:** TensorBoard + CSV under `logs/2026-02-25/13-01-52/.../logs/` (metrics.csv includes CER, loss, LR).
- This run matches the “local training” setup (positional encoding, padding mask, CSV logging, basic launcher).
