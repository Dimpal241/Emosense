# EmoSense: Facial Emotion Recognition with POSTER++ & Sharpness-Aware Minimization

**EmoSense** is a deep learning model for Facial Emotion Recognition (FER) trained on the **RAF-DB** dataset. Built upon the **POSTER++** (`pyramid_trans_expr2`) architecture, EmoSense introduces **Sharpness-Aware Minimization (SAM)**, **Stochastic Depth**, **Label Smoothing**, and **Class-Balanced Sampling** to achieve **91.23% test accuracy** and **87.13% Macro-F1** across 7 basic facial expressions.

---

## Key Enhancements Over Baseline (POSTER++)

* **Sharpness-Aware Minimization (SAM):** Paired Adam (`lr=1e-5`, `weight_decay=1e-4`) with SAM (`rho=0.05`) to converge toward flatter minima and improve generalization across unseen faces.
* **Stochastic Depth Regularization:** Rebuilt the Vision Transformer classification head (`in_c=147`, `depth=2`, `embed_dim=768`) with `drop_path_ratio=0.1`.
* **Class Imbalance Mitigation:** Applied `WeightedRandomSampler` during training to handle severe class skew in RAF-DB (e.g., *Happy* vs. *Fear*/*Disgust*), combined with **Label Smoothing (`0.1`)**.
* **Data Augmentation & Mixed Precision:** Trained at $224 \times 224$ resolution using random horizontal flips, $\pm 5^\circ$ rotation, color jitter, and `RandomErasing` (up to 25% scale) with PyTorch Automatic Mixed Precision (`FP16`).

---

## Performance & Results (RAF-DB Test Set)

Evaluated on **3,068 test images**, EmoSense outperforms the baseline POSTER++ model across all overall metrics and improves F1-score for every single emotion category:

| Metric | POSTER++ (Baseline) | EmoSense (Ours) | Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 89.15% | **91.23%** | **+2.09%** |
| **Mean Class Accuracy** | 85.59% | **86.71%** | **+1.12%** |
| **Macro-F1 Score** | 84.06% | **87.13%** | **+3.07%** |
| **Macro Precision** | 83.09% | **87.71%** | **+4.62%** |

### Per-Class F1-Score Breakdown
* **Disgust:** 71.62% $\rightarrow$ **79.75%** (+8.13%)
* **Fear:** 72.97% $\rightarrow$ **77.37%** (+4.40%)
* **Sad:** 84.85% $\rightarrow$ **88.96%** (+4.11%)
* **Angry:** 84.38% $\rightarrow$ **87.19%** (+2.81%)
* **Neutral:** 88.24% $\rightarrow$ **90.51%** (+2.27%)
* **Surprise:** 88.96% $\rightarrow$ **90.30%** (+1.34%)
* **Happy:** 95.38% $\rightarrow$ **95.88%** (+0.50%)

---

## Visual Results

### Training & Validation Curves
![Training Curves](assets/training_curves.png)

### Confusion Matrix (POSTER++ vs. EmoSense)
![Confusion Matrix](assets/confusion_matrix.png)

---

## Pretrained Weights

Download the trained EmoSense checkpoint (`best_emosense.pth`) and place it inside the `checkpoints/` directory:

* **EmoSense (`best_emosense.pth` — 91.23% Test Accuracy):** [Download from Google Drive](<YOUR_GOOGLE_DRIVE_LINK_HERE>)

---

## Repository Structure

```text
EmoSense/
├── models/
│   └── emosense.py          # EmoSenseNet & POSTER++ backbone architecture
├── utils/
│   └── sam.py               # Sharpness-Aware Minimization (SAM) optimizer
├── src/
│   ├── train.py             # Training & validation script
│   └── evaluate.py          # Test set evaluation & metric computation
├── notebooks/
│   └── EmoSense.ipynb       # Complete end-to-end experiment notebook
├── assets/
│   ├── training_curves.png  # Accuracy & loss progression plots
│   └── confusion_matrix.png # Test set confusion matrices
├── checkpoints/             # Directory for saved model weights (.pth)
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignores datasets, large checkpoints, and cache files
└── README.md                # Project documentation
