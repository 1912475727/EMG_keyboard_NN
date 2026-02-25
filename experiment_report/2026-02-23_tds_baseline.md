# Experiment report: 2026-02-23 — TDS baseline (single user)

## Summary

| Field | Value |
|-------|--------|
| **Date** | 2026-02-23 |
| **Run ID / notes** | Colab run (19:53:21); TDS only, no CNN+Transformer |
| **Model** | TDSConvCTCModule (emg2qwerty baseline) |
| **Best val CER** | **19.14%** (19.14045) |
| **Epoch at best** | 149 (of 150) |
| **Final epoch** | 150 |

---

## Model and code

- **Architecture:** Original codebase only — **SpectrogramNorm → MultiBandRotationInvariantMLP → TDSConvEncoder → Linear → LogSoftmax** (CTC). No CNN+Transformer or Conformer.
- **Config:** `model=tds_conv_ctc`, `user=single_user`, `decoder=ctc_greedy`, `lr_scheduler=linear_warmup_cosine_annealing`.
- **Parameters:** ~5.3 M trainable; ~21.2 MB.

---

## Training setup

| Setting | Value |
|---------|--------|
| max_epochs | 150 |
| batch_size | 32 (base config) |
| optimizer | Adam |
| lr_scheduler | LinearWarmupCosineAnnealingLR (pl_bolts) |
| monitor_metric | val/CER (min) |
| seed | 1501 |
| accelerator | GPU (CUDA) |
| val_check_interval | 0.5 epoch |

**TDS config (from `config/model/tds_conv_ctc.yaml`):**

- in_features: 528  
- mlp_features: [384]  
- block_channels: [24, 24, 24, 24]  
- kernel_width: 32  
- window_length: 8000, padding: [1800, 200]  

---

## Metrics (from run log)

- **Best validation CER:** 19.14045 (saved at epoch 149, step 18000).
- **Validation CER over training:** Started very high (≈1358% epoch 0, 100% epoch 1), then decreased and stabilized; best at epoch 149.
- **Checkpoints:** Best and last saved under run `checkpoints/` (e.g. `epoch=149-step=18000.ckpt` as top 1).

---

## Environment and run location

- **Platform:** Google Colab (Drive path: `/content/drive/MyDrive/Colab Notebooks/EMG_keyboard_NN/...`).
- **Logger:** TensorBoard logger folder was reported missing for this run (`Missing logger folder: .../lightning_logs`); metrics were still logged to stderr/console.
- **Python:** 3.12 (from log paths).

---

## Reproducibility

- **Config:** Use `config/base.yaml` + `config/model/tds_conv_ctc.yaml` and `user=single_user` (no CNN+Transformer or Conformer).
- **Command (equivalent):**  
  `python -m emg2qwerty.train user=single_user`  
  (with defaults: max_epochs=150, batch_size=32, etc.)

---

## Notes

- Training completed normally; `Trainer.fit` stopped because `max_epochs=150` was reached.
- No test-set metrics were included in the provided log; final evaluation (val/test) would be in the run’s stdout or a results file if added later.
