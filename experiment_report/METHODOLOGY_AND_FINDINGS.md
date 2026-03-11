# Methodology and findings — report snippets

Use these sections in your report. They incorporate: (1) CR-CTC with minimal math, (2) why data augmentation and CR-CTC matter for the Transformer, (3) TDS + CR-CTC blank-rate result.

---

## 1. CR-CTC (Consistency-Regularized CTC)

We use **CR-CTC** (Consistency-Regularized CTC) as an extension of standard CTC. The loss combines the usual CTC cross-entropy with optional terms: a consistency term (e.g. between two augmented views of the same input) and an **entropy regularization** term that encourages the model’s output distribution to be less peaky (less overconfident on the blank token). In our experiments we set the consistency weight to zero and use a small entropy weight (e.g. 0.01–0.02) so that the main effect is to soften the posteriors and reduce over-reliance on the blank symbol, which helps avoid CTC “collapse” where the model predicts blank too often.

---

## 2. Transformer: why data augmentation and CR-CTC matter

For the **CNN + Transformer** encoder, we argue that **data augmentation** and **CR-CTC** are especially important for two reasons.

1. **Transformers are more data-hungry** than smaller models (e.g. TDS). With limited EMG data, augmentation (noise, channel gain scaling, temporal jitter, SpecAugment, etc.) effectively increases diversity and reduces overfitting, which is critical when using a high-capacity transformer encoder.

2. **Transformers are reported to produce peakier output distributions** than RNNs or small conv stacks. Peaky posteriors tend to over-predict the CTC blank token and can lead to collapsed or unstable decoding. CR-CTC’s entropy regularization directly counteracts this by encouraging less peaky, better-calibrated posteriors, which we use in combination with augmentation for the transformer model.

---

## 3. TDS + CR-CTC: blank-rate reduction

For the **TDS Conv + CR-CTC** baseline we observed that adding CR-CTC (with a small entropy weight) **reduces the CTC blank rate** compared to plain CTC. In our runs, the blank rate dropped from about **0.96** (plain CTC) to about **0.93** with CR-CTC, indicating that the model relies less on the blank symbol and allocates more probability to non-blank tokens, which is desirable for decoding quality.
