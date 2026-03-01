# Experiment report: 2026-03-01 run 01-06-05 — CNN Transformer + CR-CTC (single-user)

## Summary

| Field | Value |
|-------|--------|
| **Date** | 2026-03-01 |
| **Run** | `01-06-05` — `job0_trainer.devices=1,user=single_user` |
| **Log dir** | `logs/2026-03-01/01-06-05/job0_trainer.devices=1,user=single_user/` |
| **Model** | CNNTransformerCTCModule + CR-CTC (`cnn_transformer_ctc`) |
| **Best val CER** | **~13.73%** (epoch 63) |
| **Final val CER** | **~13.73–13.87%** (epoch 63) |
| **Final val loss** | ~0.78 |
| **Epoch at end** | 64 (training in progress; last validation at epoch 63) |
| **Train CER (end)** | ~2.95% |
| **Checkpoint** | `last.ckpt` (in this folder) |

---

## Training setup

| Setting | Value |
|---------|--------|
| **User** | `single_user` |
| **Model** | `cnn_transformer_ctc` (CNNTransformerCTCModule + CR-CTC) |
| **CTC loss** | CR-CTC, `use_cr_ctc=true`, `cr_ctc_consistency_weight=0.0`, `cr_ctc_entropy_weight=0.01` |
| **Optimizer** | AdamW, lr = 0.001, weight_decay = 0.01 |
| **LR scheduler** | ReduceLROnPlateau (monitor `val/CER`, mode `min`, factor 0.1, **patience 8**, min_lr 1e-6) |
| **Decoder** | Greedy CTC (`ctc_greedy`) |
| **Front-end** | SpectrogramNorm → MultiBandRotationInvariantMLP (in_features=528, mlp_features=[384]) |
| **Encoder** | Temporal CNN (3 layers, kernel_size=31) + Transformer (4 layers, 8 heads, ff_dim=4×d_model, dropout=0.1) |

---

## Metrics (from metrics CSV, run `2026-03-01/01-06-05`)

- **Val CER:** Starts high (~17% epoch 0), improves to **~13.73%** by epoch 63. Best observed **~13.73%** at epoch 63.
- **Val loss:** ~0.78 at end of run.
- **Val IER/DER/SER (end):** ~3.19% / ~1.33–1.35% / ~9.17–9.37%.
- **Train CER (end):** ~2.95%; train–val gap ~10.8%.
- **LR at end:** 1e-6 (ReduceLROnPlateau had reduced from 0.001).

---

## Artifacts in this folder

| File | Description |
|------|-------------|
| `metrics.csv` | Lightning CSV metrics (lr, step, train/val loss, CER, IER, DER, SER). |
| `last.ckpt` | Last training checkpoint (PyTorch Lightning). |
| `2026-03-01_run_01-06-05_cnn_transformer_single_user.md` | This report. |

---

## Conclusion

This run improves on the previous single-user CR-CTC baseline (~14.7% best val CER) by using **ReduceLROnPlateau patience 8** (vs 10), reaching **~13.73% best val CER** by epoch 63. The checkpoint and metrics are stored in this c147model subfolder for reproducibility and comparison.
