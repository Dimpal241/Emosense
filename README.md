# 🎭 EmoSense — Pyramid Cross-Attention for Facial Expression Recognition

A deep learning architecture for robust Facial Expression Recognition (FER) under real-world occlusion and pose variations. **EmoSense** fuses global semantic representations from **IR-50** with fine-grained local facial landmarks from **MobileFaceNet** via a multi-scale **Pyramid Cross-Attention** module.

---

## 🚀 Key Highlights & Architecture
- **Dual-Backbone Feature Extraction:** Combines deep global context (**IR-50**) with lightweight high-resolution local facial features (**MobileFaceNet**).
- **Pyramid Cross-Attention Fusion:** Multi-scale cross-attention mechanism that aligns local eye/mouth micro-expressions with global facial structure.
- **Statistical Rigor:** Evaluated across multiple runs with paired statistical significance testing against baseline CNN/ViT models ($p = 5.4 \times 10^{-7}$).

---

## 📊 Experimental Results
| Metric | Score |
| :--- | :---: |
| **Overall Accuracy** | **91.23%** |
| **Macro-F1 Score** | **87.13%** |
| **Statistical Significance** | **$p = 5.4 \times 10^{-7}$** |

---

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch, Torchvision, timm
- **Computer Vision & Data:** OpenCV, Albumentations, NumPy, Pandas
- **Evaluation & Visualization:** Scikit-learn, Matplotlib, Seaborn

---

## 📦 Quick Start
1. **Clone the repository**
   ```bash
   git clone [https://github.com/Dimpal241/Emosense.git](https://github.com/Dimpal241/Emosense.git)
   cd Emosense
