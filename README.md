# Dynamic Swin-CRM

## Dynamic Swin-CRM for Contrastive Self-Supervised Train Driver Fatigue Detection

This repository provides a reference implementation related to the paper:

> **A Dynamic Swin-CRM-Based Contrastive Self-Supervised Framework for Train Driver Fatigue Detection**  
> *IEEE Internet of Things Journal (under review)*

The current repository provides an implementation example based on **Swin-MAE with DWMM** for self-supervised representation learning and train driver fatigue recognition.

---

## 📌 Overview

Train driver fatigue is an important safety concern in railway transportation, particularly during long-duration and monotonous driving operations.

Existing vision-based fatigue detection methods often rely heavily on manually annotated data and may have limited capability in learning robust representations across different drivers and operating conditions.

To address these challenges, the study investigates a **contrastive self-supervised learning framework** for train driver fatigue detection. The framework combines hierarchical visual representation learning, masked image reconstruction, adaptive local feature modeling, and contrastive representation learning.

The implementation example provided in this repository focuses on **Swin-MAE with DWMM**, which learns fatigue-related visual representations from partially masked facial images.

---

## 🧠 Framework Overview

<p align="center">
  <img src="1.png" width="800">
</p>

<p align="center">
  <b>Overall framework for train driver fatigue representation learning.</b>
</p>

The framework is mainly based on three learning components:

- **DWMM**
- **Reconstruction Learning**
- **Contrastive Self-Supervised Learning**

These components are designed to improve local feature representation, contextual visual modeling, and discriminative representation learning.

---

## 🧩 DWMM

DWMM is introduced into the window-based self-attention mechanism to enhance adaptive local feature modeling.

Unlike conventional window attention with fixed sampling locations, DWMM introduces learnable spatial offsets that allow feature sampling positions within each local window to be dynamically adjusted according to the input content.

The module incorporates:

- Learnable spatial offsets
- Bilinear feature resampling
- Deformation gating
- Local positional enhancement
- DW-MSA
- DSW-MSA

By combining deformable feature sampling with shifted-window attention, DWMM improves the capability of the Swin Transformer to capture flexible local dependencies and fine-grained facial patterns associated with fatigue.

---

## 🔬 Reconstruction Learning

The reconstruction learning branch follows a masked image modeling strategy.

During self-supervised pretraining, a large proportion of input image patches are masked, while the remaining visual information is processed by the DWMM-enhanced Swin Transformer encoder.

The encoded latent features are subsequently passed through a lightweight decoder to reconstruct the missing image patches.

The reconstruction objective encourages the encoder to infer missing visual information from the surrounding context and learn meaningful representations of local facial structures and subtle appearance variations.

This process provides an effective self-supervised initialization for downstream fatigue recognition tasks.

---

## 🔗 Contrastive Self-Supervised Learning

Contrastive self-supervised learning is introduced to further improve the discriminative capability and consistency of the learned representations.

Different augmented views of the same sample are treated as semantically related representations. Their features are projected into a contrastive feature space and optimized to maintain representation consistency, while representations from different samples provide discriminative information.

The contrastive learning process mainly involves:

- Contrastive feature projection
- Representation mapping between different augmented views
- Feature normalization
- Positive and negative similarity computation
- Contrastive loss optimization

By combining reconstruction-oriented representation learning with contrastive learning, the framework can exploit unlabeled facial images more effectively and reduce dependence on manually annotated fatigue data.

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

Subject-independent experiments can be conducted using the **Leave-One-Subject-Out (LOSO)** protocol.

For each experiment:

```text
Training:    N - 1 subjects
Evaluation:  1 unseen subject
```

This protocol evaluates the generalization capability of the learned representations across different drivers.

The main evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Balanced Accuracy
- ROC-AUC

---

## 🚀 Getting Started

### Create the Environment

Python 3.10 is recommended.

```bash
conda create -n fatigue python=3.10 -y
conda activate fatigue
```

Upgrade the basic packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Install PyTorch

For an NVIDIA GPU environment using CUDA 12.8:

```bash
pip install \
torch==2.7.1 \
torchvision==0.22.1 \
torchaudio==2.7.1 \
--index-url https://download.pytorch.org/whl/cu128
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

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
```

---

## 📂 Project Structure

A typical project structure is:

```text
Dynamic-Swin-CRM/
|
|-- configs/
|
|-- data/
|
|-- models/
|   |-- swin_mae_dwmm_model.py
|
|-- output/
|
|-- requirements.txt
|
|-- README.md
```

The Swin-MAE + DWMM implementation is provided in:

```text
models/swin_mae_dwmm_model.py
```

---

## 🏋️ Pretraining Example

The provided self-supervised example combines:

```text
Swin Transformer
        +
      DWMM
        +
Masked Image Reconstruction
```

During pretraining, masked facial images are processed by the DWMM-enhanced Swin encoder. The latent representations are then decoded to reconstruct the missing image patches.

An example pretraining run can be performed using the provided **Swin-MAE + DWMM** training script:

```bash
bash run_swin_mae_dwmm_subject13_full_loso_fixed_recon_v4.sh
```

The resulting pretrained encoder can subsequently be used for downstream fatigue classification experiments.

---

## 🎯 Fine-Tuning Example

The pretrained Swin-MAE + DWMM encoder can be transferred to the downstream binary fatigue classification task.

An example fine-tuning run is:

```bash
bash run_swin_mae_dwmm_after_pretrain2_finetune100_bs16_acc2.sh
```

The classifier predicts:

```text
fatigue
nofatigue
```

This example demonstrates how the learned self-supervised representation can be transferred to train driver fatigue recognition.

---

## 🖼️ Reconstruction Visualization

The masked image reconstruction process can be visualized during self-supervised pretraining.

A typical reconstruction result contains:

```text
Original Image
      |
      v
Masked Image
      |
      v
Reconstructed Image
```

These visualization results provide an intuitive view of the contextual and structural information learned by the model.

---

## 📈 Evaluation

The downstream model can be evaluated using:

```text
Accuracy
Precision
Recall
F1-score
Balanced Accuracy
ROC-AUC
```

For cross-subject experiments, LOSO evaluation is used to assess the generalization capability of the learned representations on unseen drivers.

Grad-CAM++ can also be applied to visualize the spatial regions contributing to fatigue classification decisions.

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

## Acknowledgements

This implementation builds upon research on Swin Transformer, masked image modeling, contrastive self-supervised learning, and visual representation learning.

We thank the corresponding authors and open-source communities for their valuable contributions.
