# Multi-Task Deep Learning Based on CLIP for Coffee Leaf Disease Identification and Severity Quantification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B%2F32-orange.svg)](https://github.com/mlfoundations/open_clip)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-green.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview

This project presents a **Multi-Task Deep Learning (MTL)** framework leveraging **CLIP (Contrastive Language-Image Pretraining)** to diagnose and quantify pathologies on coffee leaves (*Coffea arabica* / *Coffea canephora*).

The model simultaneously solves two critical agricultural vision tasks:
1. **Multi-label Disease Identification:** Accurately detects primary coffee foliar diseases, even when multiple infections co-occur on a single leaf.
2. **Disease Severity Quantification:** Estimates disease progression using an ordinal regression head guided by disease predictions and gating signals (Severity Levels 0–4).

An interactive **Gradio** web application is integrated, providing real-time diagnosis along with targeted **biological, cultural, and chemical treatment recommendations**.

---

## 🔬 Architecture Highlights

- **Backbone:** OpenCLIP `ViT-B/32` (pretrained on OpenAI data) extracting robust visual representations.
- **Disease Head:** Multi-label classifier with LayerNorm, GELU, feature normalization, and dynamic logit scaling.
- **Gating Mechanism:** Computes a gating signal from the healthy class probability to prevent spurious severity estimation on healthy leaves.
- **Severity Head:** Ordinal regression head using learned cumulative softplus thresholds (`cumsum(softplus(deltas)) - offset`) conditioned on concatenated visual features, disease probabilities, and gating signals.

---

## 🌿 Supported Diagnoses

| Disease | Scientific Name | Pathogen / Pest Type | Severity Scale |
| :--- | :--- | :--- | :--- |
| **Healthy** | — | Normal foliage | Level 0 |
| **Leaf Miner** | *Leucoptera coffeella* | Insect pest | Levels 1–4 |
| **Coffee Leaf Rust** | *Hemileia vastatrix* | Fungal pathogen | Levels 1–4 |
| **Phoma Leaf Spot** | *Phoma spp.* | Fungal pathogen | Levels 1–4 |
| **Cercospora (Brown Eye Spot)** | *Cercospora coffeicola* | Fungal pathogen | Levels 1–4 |

---

## 📂 Project Structure

```text
├── 11220189_Hồ Thị Kim Anh_Nghiên cứu và xây dựng mô hình học sâu đa nhiệm dựa trên CLIP trong.pdf  # Detailed thesis report
├── Source/
│   ├── app.py                      # Interactive Gradio web interface & treatment database
│   ├── model.py                    # PyTorch CLIPMultiTaskModel architecture definition
│   ├── Multi_task_learning.ipynb   # Model training, validation, and evaluation pipeline
│   ├── requirements.txt            # Python dependencies
│   └── models/
│       ├── .gitkeep
│       └── model.pt                # Trained model weights (~556 MB, download instructions below)
├── .gitignore                      # Git exclusion rules
└── README.md                       # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone Repository & Setup Environment

```bash
# Clone the repository
git clone https://github.com/HTKA0314/coffee-leaf-disease-clip-mtl.git
cd coffee-leaf-disease-clip-mtl

# (Optional) Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r Source/requirements.txt
```

### 2. Download Model Weights (`model.pt`)

Due to file size limits, the trained weights (`model.pt`, ~556 MB) are hosted on the repository's **Releases** page:

1. Download `model.pt` from **[GitHub Releases v1.0.0](https://github.com/HTKA0314/coffee-leaf-disease-clip-mtl/releases)**.
2. Place the downloaded file into the `Source/models/` directory:
   ```text
   Source/models/model.pt
   ```

*(If `model.pt` is not present, `app.py` will initialize the base architecture with fallback thresholds).*

### 3. Launch the Web Application

Navigate to the `Source` folder and start the Gradio app:

```bash
cd Source
python app.py
```

Once running, open your web browser and visit the local server:
👉 **`http://127.0.0.1:7860`**

---

## 📊 Training & Experiments

The complete training workflow is documented in `Source/Multi_task_learning.ipynb`:
- **Data Augmentation:** Random rotations, affine transformations, color jitters, and CLIP-standard normalization.
- **Loss Functions:** Multi-task loss combining Asymmetric/BCE Loss for disease recognition and Ordinal Cross-Entropy / Margin Loss for severity grading.
- **Threshold Optimization:** Class-specific decision thresholds tuned on validation data for optimal F1-score across imbalanced classes.

---

## 📖 Research Documentation

For detailed methodology, mathematical formulations, experimental setup, and comparative performance analyses, please refer to the thesis report:
* [`11220189_Hồ Thị Kim Anh_Nghiên cứu và xây dựng mô hình học sâu đa nhiệm dựa trên CLIP trong.pdf`](11220189_Hồ%20Thị%20Kim%20Anh_Nghiên%20cứu%20và%20xây%20dựng%20mô%20hình%20học%20sâu%20đa%20nhiệm%20dựa%20trên%20CLIP%20trong.pdf)

---

## 👤 Author

* **Hồ Thị Kim Anh** (Student ID: `11220189`)
* GitHub: [@HTKA0314](https://github.com/HTKA0314)

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
