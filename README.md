# Dynamic Swin-CRM

## Dynamic Swin-CRM for Contrastive Self-Supervised Train Driver Fatigue Detection

This repository provides code examples and reference implementations related to the paper:

> **A Dynamic Swin-CRM-Based Contrastive Self-Supervised Framework for Train Driver Fatigue Detection**  

The current release provides representative implementations for train driver fatigue recognition, including an example based on **Swin-MAE with DWMM** for self-supervised visual representation learning.

---

## 📌 Overview

Train driver fatigue is an important safety concern in railway transportation, particularly during long-duration and monotonous driving operations.

Existing vision-based fatigue detection methods often rely heavily on manually annotated data and may have limited capability in learning robust representations across different drivers and operating conditions.

To address these challenges, this study investigates a **contrastive self-supervised learning framework** for train driver fatigue detection. The framework combines hierarchical visual representation learning, deformable window modeling, masked image reconstruction, and contrastive representation learning.

As a representative implementation example, this repository provides **Swin-MAE with DWMM**, illustrating how deformable window modeling can be integrated with masked image reconstruction for self-supervised fatigue-related visual representation learning.

Several commonly used visual backbones are also included for comparative experiments.

---

## 🧠 Framework Overview

<p align="center">
  <img src="1.png" width="800">
</p>

<p align="center">
  <b>Overall framework of the proposed train driver fatigue representation learning method.</b>
</p>

The overall framework mainly involves three learning components:

- **DWMM**
- **Reconstruction Learning**
- **Contrastive Self-Supervised Learning**

These components are designed to improve adaptive local feature modeling, contextual representation learning, and discriminative feature representation.

The implementation example highlighted in this repository focuses on **Swin-MAE + DWMM**.

---

## 🧩 DWMM

The **Deformable Window Modeling Module (DWMM)** is introduced into window-based self-attention to enhance adaptive local feature representation.

Conventional window attention performs feature interaction at fixed spatial sampling locations. In contrast, DWMM introduces learnable spatial offsets that allow feature sampling positions within each local window to be dynamically adjusted according to the input features.

The main mechanisms include:

- Learnable spatial offsets
- Bilinear feature resampling
- Deformation gating
- Local positional enhancement
- Deformable Window Multi-Head Self-Attention (DW-MSA)
- Deformable Shifted Window Multi-Head Self-Attention (DSW-MSA)

By combining deformable feature sampling with shifted-window attention, DWMM enables the Swin Transformer to capture more flexible local dependencies and fine-grained facial patterns associated with fatigue.

---

## 🔬 Reconstruction Learning

The reconstruction learning component follows a masked image modeling strategy and encourages the encoder to capture contextual dependencies and local structural information from partially observed facial images.

During self-supervised pretraining, a large proportion of the input image patches are masked, while the remaining visual information is processed by the DWMM-enhanced Swin Transformer encoder.

The resulting latent representations are subsequently passed through a lightweight decoder to reconstruct the missing image patches.

The reconstruction objective is calculated over the masked regions, encouraging the model to infer missing visual information from the surrounding context.

Through this process, the encoder learns meaningful representations of local facial structures, contextual information, and subtle appearance variations that can subsequently be transferred to downstream fatigue recognition tasks.

---

## 🔗 Contrastive Self-Supervised Learning

Contrastive self-supervised learning is incorporated into the overall framework to improve the consistency and discriminative capability of the learned representations.

Different augmented views of the same sample are mapped into a shared representation space. Representations corresponding to the same sample are encouraged to maintain semantic consistency, while features from different samples provide discriminative information.

The contrastive learning process mainly involves:

- Contrastive feature projection
- Online and momentum feature mapping
- Feature normalization
- Positive and negative similarity computation
- Contrastive loss optimization

Together with reconstruction-oriented representation learning and DWMM-based feature modeling, contrastive learning contributes to more robust fatigue-related visual representations and reduces dependence on manually annotated data.

---

## 📁 Dataset

The experiments are conducted using a train driver fatigue dataset containing facial images collected under different driving conditions.

The dataset includes:

- Multiple train drivers
- Different driving postures
- Fatigue and non-fatigue states
- Long-duration driving recordings
- Subject-independent experimental settings

Due to privacy, ethical, and institutional regulations, the dataset is currently not publicly downloadable.

Researchers interested in academic collaboration may contact the authors for further information.

---

## 📊 Experimental Setting

The downstream train driver fatigue recognition task is formulated as a binary classification problem:

```text
0 = fatigue
1 = nofatigue
```

Subject-independent evaluation can be conducted using the **Leave-One-Subject-Out (LOSO)** protocol.

For each experiment:

```text
Training:    N - 1 subjects
Evaluation:  1 unseen subject
```

This protocol is used to evaluate the generalization capability of the learned representations across different train drivers.

The main evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Balanced Accuracy
- ROC-AUC

Grad-CAM++ can additionally be used to visualize the spatial regions contributing to the classification decisions.

---

## 🚀 Getting Started

### 1. Create the Environment

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

### 2. Install PyTorch

For an NVIDIA GPU environment using CUDA 12.8:

```bash
pip install \
torch==2.7.1 \
torchvision==0.22.1 \
torchaudio==2.7.1 \
--index-url https://download.pytorch.org/whl/cu128
```

---

### 3. Install Dependencies

Install the remaining dependencies using:

```bash
pip install -r requirements.txt
```

The main dependencies include:

```text
PyTorch
torchvision
timm
NumPy
SciPy
scikit-learn
OpenCV
Pillow
Matplotlib
Grad-CAM
PyYAML
yacs
einops
```

---

## 📂 Project Structure

The repository contains the proposed implementation examples together with several comparison backbones and experimental utilities.

```text
Dynamic-Swin-CRM/
|
|-- configs/
|   |-- coatnet/
|   |-- convnext/
|   |-- deit/
|   |-- dinov2/
|   |-- edgenext/
|   |-- mixmae/
|   |-- mobilevitv2/
|   |-- resnet50/
|   |-- resnet50_mimcl/
|   |-- swin_base__100ep/
|   |-- swin_base__800ep/
|   |-- swin_cmae/
|   |-- swin_large__800ep/
|   |-- swin_mae_dwmm/
|   |-- vit_base__800ep/
|   `-- ...
|
|-- data/
|   `-- data loading and dataset utilities
|
|-- models/
|   |-- swin_transformer.py
|   |-- swin_mae_model.py
|   |-- swin_mae_dwmm_model.py
|   |-- simmim.py
|   |-- convnext_model.py
|   |-- deit_model.py
|   |-- deit3_model.py
|   |-- dinov2_model.py
|   |-- edgenext_model.py
|   |-- coatnet_model.py
|   |-- mobilevitv2_model.py
|   |-- resnet_model.py
|   |-- resnet50_mimcl_model.py
|   |-- swin_cmae_model.py
|   |-- vision_transformer.py
|   `-- ...
|
|-- config.py
|-- logger.py
|-- lr_scheduler.py
|
|-- main_simmim.py
|-- main_simmim_pretrain_reconstruction.py
|
|-- swin_mae_dwmm_train_gradcam_speed_single_gpu_loso_fixed_recon_v5.py
|
|-- comparison and fine-tuning scripts
|
|-- requirements.txt
`-- README.md
```

The main example implementation described in this README is:

```text
models/swin_mae_dwmm_model.py
```

Its corresponding configuration files are located under:

```text
configs/swin_mae_dwmm/
```

Other model files and training scripts are retained for backbone comparison, ablation experiments, and evaluation.

---

## 🏋️ Example Implementation

As a representative implementation example, this repository provides **Swin-MAE with DWMM** for self-supervised train driver fatigue representation learning.

The example combines:

```text
Swin Transformer
        +
       DWMM
        +
Masked Image Reconstruction
```

The main model is implemented in:

```text
models/swin_mae_dwmm_model.py
```

During self-supervised pretraining, partially masked facial images are processed by the DWMM-enhanced Swin Transformer encoder to obtain latent feature representations.

A lightweight decoder is subsequently used to reconstruct the masked image patches from the encoded features.

This example demonstrates how deformable window modeling can be integrated with masked image reconstruction to learn contextual and fine-grained visual representations from unlabeled facial images.

---

## 🔄 Pretraining Example

The provided **Swin-MAE + DWMM** implementation can be used as an example for self-supervised pretraining.

The pretraining process can be summarized as:

```text
Input Facial Image
        |
        v
Patch Embedding
        |
        v
Random Masking
        |
        v
Swin Transformer + DWMM
        |
        v
Latent Representation
        |
        v
Reconstruction Decoder
        |
        v
Masked Patch Prediction
```

A representative training script is provided in the project for the Swin-MAE + DWMM experiment.

The pretrained encoder obtained from this example can subsequently be transferred to downstream visual recognition tasks.

---

## 🎯 Downstream Example

As an example downstream application, the pretrained **Swin-MAE + DWMM** encoder can be transferred to binary train driver fatigue classification.

The classification task is defined as:

```text
0 = fatigue
1 = nofatigue
```

The pretrained encoder is combined with a classification head and fine-tuned using labeled fatigue data.

For subject-independent evaluation, the model can be evaluated under the LOSO protocol:

```text
Training:    N - 1 subjects
Evaluation:  1 unseen subject
```

This example illustrates the transfer of self-supervised representations to cross-subject train driver fatigue recognition.

---

## 🖼️ Reconstruction Visualization

The masked image reconstruction process can be visualized during self-supervised pretraining.

A typical visualization contains:

```text
Original Image
      |
      v
Masked Image
      |
      v
Reconstructed Image
```

The reconstruction results provide an intuitive view of the contextual and structural information learned by the model from partially observed facial images.

---

## 🔍 Comparison Backbones

Several commonly used visual architectures are included for comparative experiments.

Representative backbones available in the repository include:

- Swin Transformer
- Vision Transformer
- DeiT
- DeiT III
- ConvNeXt
- DINOv2
- EdgeNeXt
- CoAtNet
- MobileViTv2
- ResNet50
- MixMAE
- SimMIM

The associated model definitions, configuration files, and experimental scripts are organized under the `models/` and `configs/` directories.

These implementations are mainly used for comparative evaluation of fatigue recognition performance under consistent experimental settings.

---

## 🔥 Grad-CAM++ Visualization

Grad-CAM++ can be used to analyze the spatial regions contributing to train driver fatigue classification.

The required package can be installed with:

```bash
pip install grad-cam opencv-python
```

Generated attention maps can be used to examine whether the model focuses on meaningful facial regions associated with fatigue-related visual patterns.

---

## 📈 Evaluation

The downstream models can be evaluated using common binary classification metrics:

```text
Accuracy
Precision
Recall
F1-score
Balanced Accuracy
ROC-AUC
```

For cross-subject experiments, LOSO evaluation can be used to measure the generalization capability of the learned representations on unseen drivers.

The experimental scripts also support model efficiency analysis and visual interpretation where applicable.

---

## 📝 Citation

If this work is useful for your research, please cite the corresponding paper after publication.

```bibtex
@article{dynamic_swin_crm,
  title   = {A Dynamic Swin-CRM-Based Contrastive Self-Supervised Framework for Train Driver Fatigue Detection},
  author  = {To be updated},
  journal = {IEEE Internet of Things Journal},
  year    = {Under Review}
}
```

---

## 📧 Contact

For questions regarding the implementation, dataset, or academic collaboration, please contact the authors.

---

## 🙏 Acknowledgements

This repository builds upon research and open-source implementations related to Swin Transformer, masked image modeling, deformable attention, contrastive self-supervised learning, and visual representation learning.

We sincerely thank the corresponding authors and open-source communities for their valuable contributions.
