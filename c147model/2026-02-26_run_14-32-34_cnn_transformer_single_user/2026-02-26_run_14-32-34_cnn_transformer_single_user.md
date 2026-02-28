# Experiment report: CNN Transformer + CR-CTC (single-user) — logspec pipeline

## Summary


| Field                 | Value                                                                                                      |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Date**              | 2026-02-26                                                                                                 |
| **Model**             | CNNTransformerCTCModule + CR-CTC (`cnn_transformer_ctc`)                                                   |
| **Best val CER**      | **≈14.7–14.8%** (best ~14.73% at epoch 109; stable ~14.75–14.9% in late epochs)                            |
| **Best val loss**     | ≈0.71–0.74 (around best val CER; val/loss in CSV)                                                           |
| **Data augmentation** | ToTensor → band_rotation → temporal_jitter → logspec → specaug (no noise / channel dropout / gain scaling) |


---

## Important note: augmentation

This experiment **does not use** the newer per-sample raw-EMG augmentations:

- **AdditiveGaussianNoise** (time-domain Gaussian noise)
- **ChannelDropout**
- **ChannelGainScaling**

Although these transforms are defined in `config/transforms/log_spectrogram.yaml`, the **train** pipeline for this run is:

- `ToTensor` (`fields`: `[emg_left, emg_right]`)
- `RandomBandRotation` (via `ForEach`)
- `TemporalAlignmentJitter`
- `LogSpectrogram`
- `SpecAugment`

Validation and test use:

- `ToTensor` → `LogSpectrogram` only.

So the augmentation stack matches the earlier TDS baseline (no new noise/dropout/gain), but with the CNN Transformer + CR-CTC model.

---

## Training setup


| Setting            | Value                                                                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| **User**           | `single_user` (`89335547`)                                                                                                              |
| **Train sessions** | 16 EMG sessions from 2021-06-02 to 2021-07-22 (including all three 2021-06-02 sessions 1622681518, 1622679967, and 1622682789 in train) |
| **Val session**    | `2021-06-04-1622862148-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f`                                                     |
| **Test session**   | `2021-06-02-1622682789-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f`                                                     |
| **Model**          | `cnn_transformer_ctc` (CNNTransformerCTCModule + CR-CTC)                                                                                |
| **Front-end**      | SpectrogramNorm → MultiBandRotationInvariantMLP (in_features=528, mlp_features=[384])                                                   |
| **Encoder**        | Temporal CNN (3 layers, kernel_size=31) + Transformer (4 layers, 8 heads, ff_dim=4×d_model, dropout=0.1)                                |
| **CTC loss**       | CR-CTC, `use_cr_ctc=true`, `cr_ctc_consistency_weight=0.0`, `cr_ctc_entropy_weight=0.01`                                                |
| **Decoder**        | Greedy CTC (`ctc_greedy`)                                                                                                               |
| **Optimizer**      | AdamW, lr = **0.001**, weight_decay = 0.01                                                                                              |
| **LR scheduler**   | ReduceLROnPlateau (monitor `val/CER`, mode `min`, factor 0.1, patience 10, min_lr 1e-6)                                                 |
| **Batch size**     | 32                                                                                                                                      |
| **Windowing**      | `window_length=8000`, `padding=[1800, 200]` (≈4 s windows, 900 ms past, 100 ms future)                                                  |
| **Seed**           | 1501                                                                                                                                    |
| **Trainer**        | GPU, 1 device, max_epochs = 150, `val_check_interval = 0.5`, `log_every_n_steps = 10`                                                   |


---

## Metrics (from metrics CSV, run `2026-02-26/14-32-34`)

High-level behavior:

- **Val CER**:
  - Starts at **100%** (epoch 0), then **≈70%** by end of epoch 0.
  - Falls to **≈34–37%** by epoch 1 and **≈26–28%** by epoch 10.
  - Reaches **≈18.5–19.5%** by epoch 74 (end of first LR plateau at 0.001).
  - Improves with LR reductions: **≈16.6–16.8%** at epoch 79, **≈14.9–15%** by epoch 106, **≈14.75–14.9%** through late epochs (144–149).
  - **Best val CER ≈14.73%** (epoch 109); stable **≈14.75–14.9%** in late training. Large improvement over TDS+CR-CTC baseline (~20.6% best val CER).
- **Train CER**:
  - Decreases over epochs and ends **≈4.0–4.2%** in late epochs (144–149), indicating the model fits the training data well.
  - **Train–val gap**: train CER ≈4%, val CER ≈14.7–14.9%.
- **Loss curves**:
  - **Train loss** drops from large initial values (due to checkpoint/optimizer mismatch at the very beginning) down to **≈0.3–0.5** in mid training and to **≈0.07–0.15** (per-step) in late epochs as LR decreases.
  - **Val loss** tracks CER, reaching **≈0.71–0.74** around the best CER region and then flattening as LR hits min (val/loss in the CSV is in the 0.71–0.83 range when val CER is in the mid–high teens %).

Overall, the run shows **steady and continued generalization improvements** as LR is stepped down, without obvious late-epoch overfitting on the validation set.

---

## Learning-rate schedule (from metrics/logs)

Using AdamW with ReduceLROnPlateau on `val/CER`:

- **0.001**: epochs 0–78  
- **0.0001**: epochs 79–~105 (first plateau triggers first LR drop)  
- **1e-5**: epochs ~106–~124  
- **1e-6 and below**: epochs ~125–149 (further LR reductions as `val/CER` plateaus at the ~14.75–14.9% level)

The LR reductions correlate with renewed improvements in both **val loss** and **val CER**, especially around the first two drops (→0.0001 and →1e-5).

---

## Conclusion

- This **CNN Transformer + CR-CTC** setup with the classic **log-spectrogram + SpecAug** pipeline achieves **≈14.7–14.8% best val CER** (best ~14.73% at epoch 109), a **substantial improvement** over the earlier TDS+CR-CTC baseline (~20.6%).
- The combination of:
  - stronger encoder capacity (CNN + Transformer),
  - CR-CTC entropy regularization (`0.01`),
  - and a conservative augmentation stack (no raw-noise / channel dropout / gain scaling)
  appears to give a good balance of fitting and generalization on this single-user dataset.
- For future comparisons, this run is a strong **reference configuration**:
  - `model=cnn_transformer_ctc`
  - `transforms=log_spectrogram` (train: ToTensor → band_rotation → temporal_jitter → logspec → specaug)
  - `optimizer=adamw` (lr=1e-3)
  - `lr_scheduler=reduce_on_plateau` (patience=10, factor=0.1)
  - `user=single_user` dataset as in `config/user/single_user.yaml`.

