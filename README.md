# Dynamic Swin-CRM

## Dynamic Swin-CRM for Contrastive Self-Supervised Train Driver Fatigue Detection

This repository provides the implementation of the paper:

> **A Dynamic Swin-CRM–Based Contrastive Self-Supervised Framework for Train Driver Fatigue Detection**  
> *IEEE Internet of Things Journal *

---

## 📌 Overview

Train driver fatigue poses a serious threat to railway transportation safety, particularly during long-duration and monotonous driving operations. Existing vision-based fatigue detection methods often rely heavily on manually annotated data and may have limited capability in learning robust fatigue-related representations across different drivers and operating conditions.

To address these challenges, this project proposes a **Dynamic Swin-CRM–based contrastive self-supervised learning framework** for train driver fatigue detection.

The proposed framework integrates:

- Swin Transformer–based hierarchical feature learning
- Deformable Window Modeling Module (DWMM)
- Masked image reconstruction learning
- Contrastive self-supervised representation learning
- Momentum encoder and negative feature queue
- Cross-subject fatigue classification using LOSO evaluation

During self-supervised pretraining, partially masked facial images are processed by the DWMM-enhanced Swin encoder. The model simultaneously learns to reconstruct masked image regions and optimize contrastive representations between different augmented views of the same sample.

The learned encoder is subsequently transferred to the downstream fatigue classification task.

---

## ✨ Key Contributions

### Dynamic Swin-CRM Backbone

A Dynamic Swin-CRM architecture incorporating a **Deformable Window Modeling Module (DWMM)** is introduced to improve adaptive local feature representation.

DWMM introduces learnable spatial offsets into window-based self-attention, allowing sampling locations within each local window to be dynamically adjusted according to the input features. Deformable feature sampling, deformation gating, local positional enhancement, and shifted-window attention are jointly employed to improve the modeling of subtle fatigue-related facial patterns.

---

### Reconstruction Learning

A masked image reconstruction objective is incorporated into the self-supervised pretraining framework.

During pretraining, a large proportion of image patches are masked, while the remaining visual information is encoded by the DWMM-enhanced Swin backbone. A lightweight reconstruction decoder predicts the missing image patches from the latent representations.

The reconstruction loss is calculated over the masked regions, encouraging the encoder to capture contextual dependencies, local structures, and fine-grained facial information.

---

### Contrastive Self-Supervised Learning

A contrastive learning branch is jointly optimized with the reconstruction objective.

The online branch extracts latent representations from masked images and maps them into the contrastive feature space through a feature predictor and projection head.

A momentum encoder processes an augmented view of the same image to generate the corresponding key representation. Query and key features form positive pairs, while historical representations stored in a feature queue are used as negative samples.

The overall self-supervised objective is formulated as:

\[
L_{\text{total}}
=
L_{\text{reconstruction}}
+
\lambda_{\text{CL}}L_{\text{contrastive}}
\]

This joint optimization enables the model to learn both reconstruction-aware local representations and discriminative semantic features.

---

### Cross-Subject Fatigue Detection

The learned self-supervised encoder is transferred to a binary fatigue classification task:

```text
0 = fatigue
1 = nofatigue
```

The downstream model is evaluated using **Leave-One-Subject-Out (LOSO)** validation to assess its generalization capability across unseen drivers.

---

### Train Driver Fatigue Dataset

A train driver fatigue dataset containing different drivers, driving postures, fatigue conditions, and long-duration driving recordings is constructed for model development and evaluation.

The dataset supports both self-supervised representation learning and cross-subject fatigue recognition experiments.

---

### Fatigue Mathematical Generation Model

A fatigue mathematical generation model is incorporated to characterize the temporal evolution of driver fatigue and provide continuous and physiologically plausible fatigue-related supervision for long-duration driving analysis.

---

## 🧠 Framework Overview

<p align="center">
  <img src="1.png" width="800">
</p>

<p align="center">
  <b>Figure 1. Overall framework of the proposed Dynamic Swin-CRM–based contrastive self-supervised fatigue detection method.</b>
</p>

The overall framework consists of three main learning components:

```text
Input facial image
        │
        ▼
Patch embedding and masking
        │
        ▼
DWMM-enhanced Swin Transformer
        │
        ├──────────────► Reconstruction branch
        │                    │
        │                    ▼
        │             Reconstruction loss
        │
        └──────────────► Contrastive branch
                             │
                   Online representation q
                             │
                   Momentum representation k
                             │
                       Feature queue
                             │
                             ▼
                      Contrastive loss
```

---

## 🔬 Main Components

### 1. Deformable Window Modeling Module

The DWMM dynamically adjusts feature sampling locations within local attention windows and improves the capability of the Swin Transformer to capture adaptive and fine-grained fatigue-related patterns.

The main mechanisms include:

- Learnable spatial offsets
- Bilinear feature resampling
- Deformation gating
- Local positional enhancement
- DW-MSA
- DSW-MSA

---

### 2. Reconstruction Learning

Masked facial image patches are reconstructed from latent features produced by the DWMM-enhanced encoder.

The reconstruction objective encourages the model to infer missing visual content using contextual information and learn robust local representations.

---

### 3. Contrastive Self-Supervised Learning

The contrastive learning module contains three main components.

#### Contrastive Learning Projection Head

High-dimensional encoder features are mapped into a lower-dimensional contrastive representation space through the projection head.

#### Online and Momentum Feature Mapping

The online branch produces query features, while the momentum branch generates the corresponding key features from another augmented view of the same sample.

#### Feature Normalization and Contrastive Loss Computation

Query and key representations are normalized before positive and negative similarities are computed. Negative samples are obtained from the feature queue and used to construct the contrastive objective.

---

## 📊 Experimental Evaluation

The proposed framework is evaluated using:

- Leave-One-Subject-Out (LOSO) cross-subject validation
- Binary fatigue classification
- Long-duration train driving analysis
- Accuracy
- Precision
- Recall
- F1-score
- Balanced Accuracy
- ROC-AUC
- Inference throughput
- Grad-CAM++ visualization

The class definition used throughout the experiments is:

```text
fatigue   = 0
nofatigue = 1
```

Experimental results demonstrate that the proposed framework provides reliable fatigue recognition performance and favorable cross-subject generalization under different operating conditions.

Detailed quantitative comparisons and ablation studies are provided in the paper.

---

## 📁 Dataset

The dataset contains:

- Multiple train drivers
- Multiple driving postures
- Fatigue and non-fatigue states
- Long-duration continuous driving recordings
- Subject-independent training and evaluation splits

For the LOSO experiments, one subject is held out for evaluation while the remaining subjects are used for training.

> ⚠️ Due to privacy, ethical, and institutional regulations, the dataset is currently not publicly downloadable.
>
> Researchers interested in academic collaboration may contact the authors for further information.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Dynamic-Swin-CRM
```

---

## 2. Create the Conda Environment

Python 3.10 is recommended.

```bash
conda create -n fatigue python=3.10 -y
conda activate fatigue
```

Upgrade the basic Python packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## 3. Install PyTorch

For an NVIDIA GPU environment using CUDA 12.8:

```bash
pip install \
torch==2.7.1 \
torchvision==0.22.1 \
torchaudio==2.7.1 \
--index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA availability:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## 4. Install Project Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1
timm==1.0.21
numpy==1.26.4
scipy
scikit-learn
pandas
Pillow
opencv-python
matplotlib
grad-cam
tqdm
tensorboard
einops
PyYAML
yacs
termcolor
```

---

## 5. Verify the Environment

```bash
python - <<'PY'
import torch
import torchvision
import timm
import cv2
import numpy

from pytorch_grad_cam import GradCAMPlusPlus

print("=" * 60)
print("Environment Check")
print("=" * 60)

print("PyTorch      :", torch.__version__)
print("TorchVision  :", torchvision.__version__)
print("timm         :", timm.__version__)
print("NumPy        :", numpy.__version__)
print("OpenCV       :", cv2.__version__)
print("CUDA runtime :", torch.version.cuda)
print("CUDA usable  :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU          :", torch.cuda.get_device_name(0))

print("Grad-CAM++   : OK")
print("=" * 60)
PY
```

---

# 📂 Project Structure

A typical project organization is:

```text
Dynamic-Swin-CRM/
│
├── configs/
│   └── swin_mae_dwmm_cl/
│
├── data/
│
├── models/
│   └── swin_mae_dwmm_cl_model.py
│
├── output/
│
├── requirements.txt
│
├── run_swin_mae_dwmm_cl_subject13_full_with_log.sh
│
├── run_swin_mae_dwmm_cl_subject13_smoke5_pretrain_with_log.sh
│
├── run_swin_mae_dwmm_cl_subject13_finetune_only_with_log.sh
│
└── README.md
```

---

# 🏋️ Self-Supervised Pretraining

The self-supervised model combines:

```text
Swin-B
  +
MAE-style reconstruction learning
  +
DWMM
  +
Contrastive Learning
```

The main pretraining configuration includes:

```text
Input size              : 224 × 224
Patch size              : 4 × 4
Mask ratio              : 75%
Mask window             : 4
Projection dimension    : 256
Projection hidden dim   : 2048
Contrastive temperature : 0.20
Contrastive weight      : 0.10
Feature queue size      : 4096
```

Run self-supervised pretraining using:

```bash
bash run_swin_mae_dwmm_cl_subject13_full_with_log.sh
```

For a short five-epoch test:

```bash
bash run_swin_mae_dwmm_cl_subject13_smoke5_pretrain_with_log.sh
```

---

# 🎯 Fine-Tuning

After self-supervised pretraining, the online DWMM-enhanced Swin encoder is transferred to the downstream fatigue classifier.

The reconstruction decoder, momentum encoder, projection head, and feature queue are not required during downstream inference.

Example:

```bash
bash run_swin_mae_dwmm_cl_subject13_finetune_only_with_log.sh \
./output/swin_mae_dwmm_cl_pretrain/subject13/swin_mae_dwmm_cl_last.pth
```

---

# 🔄 LOSO Evaluation

The framework uses Leave-One-Subject-Out evaluation.

For each experiment:

```text
N - 1 subjects → training
1 subject       → validation/testing
```

This protocol evaluates whether the learned fatigue representations can generalize to an unseen driver.

---

# 🔥 Grad-CAM++ Visualization

Grad-CAM++ is used to visualize spatial regions contributing to fatigue classification.

Install the required package using:

```bash
pip install grad-cam opencv-python
```

Generated visualizations can be used to analyze whether the model focuses on meaningful facial regions associated with fatigue.

---

# ⚡ Inference Performance

The evaluation scripts report batch-aware inference performance, including:

```text
Throughput        : images/s
Batch latency     : ms/batch
Effective latency : ms/image
```

These measurements are provided together with classification metrics for a comprehensive evaluation of both recognition performance and computational efficiency.

---

# 📈 Evaluation Metrics

The following metrics are reported:

```text
Accuracy
Precision
Recall
F1-score
Balanced Accuracy
ROC-AUC
```

For binary evaluation, **fatigue** is treated as the target class when calculating fatigue-specific Precision, Recall, F1-score, and ROC-AUC.

---

# 📦 Model Checkpoints

Self-supervised checkpoints are stored under:

```text
./output/swin_mae_dwmm_cl_pretrain/
```

A typical checkpoint is:

```text
swin_mae_dwmm_cl_last.pth
```

The best downstream classification model is selected according to **Balanced Accuracy** and saved as:

```text
best_ba_weights.pth
```

---

# 📝 Citation

If this work is useful for your research, please cite the corresponding paper after publication.

```bibtex
@article{dynamic_swin_crm,
  title   = {A Dynamic Swin-CRM-Based Contrastive Self-Supervised Framework for Train Driver Fatigue Detection},
  author  = {Author information will be updated},
  journal = {IEEE Internet of Things Journal},
  year    = {Under Review}
}
```

---

# 📧 Contact

For questions regarding the implementation, dataset, or academic collaboration, please contact the authors.

---

## Acknowledgements

This implementation builds upon open-source deep learning frameworks and related work in Swin Transformer, masked image modeling, contrastive self-supervised learning, and visual explainability.

We sincerely thank the corresponding authors and open-source communities for their valuable contributions.
