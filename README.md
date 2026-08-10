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

By combining deformable feature sampling with shifted-window attention, DWMM enables the Swin Transformer to capture flexible local dependencies and fine-grained facial patterns associated with fatigue.

---

## 🔬 Reconstruction Learning

The reconstruction learning component follows a masked image modeling strategy and encourages the encoder to capture contextual dependencies and local structural information from partially observed facial images.

During self-supervised pretraining, a large proportion of the input image patches are masked, while the remaining visual information is processed by the DWMM-enhanced Swin Transformer encoder.

The resulting latent representations are subsequently passed through a lightweight decoder to reconstruct the missing image patches. The reconstruction objective is calculated over the masked regions, encouraging the model to infer missing visual information from the surrounding context.

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

Together with reconstruction-oriented representation learning and DWMM-based feature modeling, contrastive learning contributes to more robust fatigue-related visual representations.

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

Subject-independent evaluation can be conducted using the **Leave-One-Subject-Out (LOSO)** protocol:

```text
Training:    N - 1 subjects
Evaluation:  1 unseen subject
```

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
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If PyTorch needs to be installed separately:

```bash
pip install \
torch==2.7.1 \
torchvision==0.22.1 \
torchaudio==2.7.1 \
--index-url https://download.pytorch.org/whl/cu128
```

---

## 📂 Project Structure

The repository contains the example implementation together with comparison backbones and experimental utilities.

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
|
|-- models/
|   |-- coatnet_model.py
|   |-- convnext_model.py
|   |-- deit_model.py
|   |-- deit3_model.py
|   |-- dinov2_model.py
|   |-- edgenext_model.py
|   |-- mixmae_swinb_model.py
|   |-- mobilevitv2_model.py
|   |-- resnet_model.py
|   |-- resnet50_mimcl_model.py
|   |-- simmim.py
|   |-- swin_cmae_model.py
|   |-- swin_mae_dwmm_model.py
|   |-- swin_mae_model.py
|   |-- swin_transformer.py
|   |-- swinv2_model.py
|   |-- vision_transformer.py
|   `-- ...
|
|-- config.py
|-- logger.py
|-- lr_scheduler.py
|-- main_finetune.py
|-- main_simmim.py
|-- main_simmim_pretrain_reconstruction.py
|
|-- swin_mae_dwmm_train_gradcam_speed_single_gpu_loso_fixed_recon_v5.py
|-- run_swin_mae_dwmm_example.sh
|-- download_pretrained_weights.sh
|-- requirements.txt
`-- README.md
```

The main example model described in this README is:

```text
models/swin_mae_dwmm_model.py
```

and its configuration files are located under:

```text
configs/swin_mae_dwmm/
```

Other model definitions and training scripts are retained for comparison and experimental evaluation.

---

## 🏋️ Example Implementation

The provided example combines:

```text
Swin Transformer
        +
       DWMM
        +
Masked Image Reconstruction
```

During self-supervised pretraining, partially masked facial images are processed by the DWMM-enhanced Swin Transformer encoder. A lightweight decoder reconstructs the missing image patches from the learned latent representations.

The resulting pretrained encoder can subsequently be transferred to downstream fatigue classification tasks.

---

## ▶️ Run the Swin-MAE + DWMM Example

A single-GPU example launcher is provided:

```bash
chmod +x run_swin_mae_dwmm_example.sh
bash run_swin_mae_dwmm_example.sh
```

The launcher automatically looks for the newest available Swin-MAE + DWMM training driver in the project root.

The default dataset path is:

```text
./data/fatiguev2_105270
```

A different dataset path can be provided without editing the script:

```bash
DATA_PATH=/path/to/dataset \
bash run_swin_mae_dwmm_example.sh
```

Common options can also be overridden from the command line environment:

```bash
PRETRAIN_BATCH_SIZE=4 \
FINETUNE_BATCH_SIZE=16 \
ACCUMULATION_STEPS=2 \
bash run_swin_mae_dwmm_example.sh
```

To disable Grad-CAM++ during the example run:

```bash
SKIP_GRADCAM=1 \
bash run_swin_mae_dwmm_example.sh
```

---

## ⬇️ Pretrained Weights

A download script is provided for the comparison backbones:

```bash
chmod +x download_pretrained_weights.sh
```

Download the standard backbone weights:

```bash
bash download_pretrained_weights.sh ./pretrained standard
```

Download the standard weights together with the available SimMIM and MixMAE self-supervised checkpoints:

```bash
bash download_pretrained_weights.sh ./pretrained all
```

### Direct Download Sources

| Backbone | Pretrained weight |
|---|---|
| CoAtNet-0 | `https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k/resolve/main/pytorch_model.bin` |
| ConvNeXt-Base | `https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth` |
| DeiT-Base | `https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth` |
| DeiT III Base | `https://huggingface.co/timm/deit3_base_patch16_224.fb_in22k_ft_in1k/resolve/main/model.safetensors` |
| DINOv2 ViT-S/14 | `https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth` |
| EdgeNeXt-Small | `https://huggingface.co/timm/edgenext_small.usi_in1k/resolve/main/pytorch_model.bin` |
| MobileViTv2-1.0 | `https://huggingface.co/timm/mobilevitv2_100.cvnets_in1k/resolve/main/pytorch_model.bin` |
| ResNet50 | `https://download.pytorch.org/models/resnet50-11ad3fa6.pth` |
| Swin-Base | `https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth` |
| SwinV2-CR Small NS | `https://huggingface.co/timm/swinv2_cr_small_ns_224.sw_in1k/resolve/main/model.safetensors` |

Additional self-supervised checkpoints available from their corresponding public repositories include:

| Method | Checkpoint |
|---|---|
| MixMAE Swin-B/W14, 600 epochs | `https://drive.google.com/uc?id=1pZYmTv08xK_kOe2kk6ahuvgJVkHm-ZIa` |
| SimMIM Swin-Base, 100 epochs | `https://drive.google.com/uc?id=1Wcbr66JL26FF30Kip9fZa_0lXrDAKP-d` |
| SimMIM Swin-Base, 800 epochs | `https://drive.google.com/uc?id=15zENvGjHlM71uKQ3d2FbljWPubtrPtjl` |
| SimMIM Swin-Large, 800 epochs | `https://drive.google.com/uc?id=1qDxrTl2YUDB0505_4QrU5LU2R1kKmcBP` |
| SimMIM ViT-Base, 800 epochs | `https://drive.google.com/uc?id=1dJn6GYkwMIcoP3zqOEyW1_iQfpBi8UOw` |

The **Swin-MAE + DWMM** example does not require a downloadable task-specific checkpoint for self-supervised pretraining; its pretraining checkpoint is generated locally by the example training procedure.

Project-specific variants such as Swin-CMAE or other custom self-supervised combinations can likewise be trained locally when no external checkpoint is supplied.

---

## 🖼️ Reconstruction Visualization

The masked image reconstruction process can be visualized during self-supervised pretraining:

```text
Original Image
      |
      v
Masked Image
      |
      v
Reconstructed Image
```

These results provide an intuitive view of the contextual and structural information learned from partially observed facial images.

---

## 🔍 Comparison Backbones

The repository contains several visual architectures for comparison, including:

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

Their model definitions, configuration files, and experiment scripts are organized under `models/`, `configs/`, and the project root.

---

## 🔥 Grad-CAM++ Visualization

Grad-CAM++ can be used to analyze spatial regions contributing to fatigue classification.

```bash
pip install grad-cam opencv-python
```

---

## 📈 Evaluation

The downstream models can be evaluated using:

```text
Accuracy
Precision
Recall
F1-score
Balanced Accuracy
ROC-AUC
```

For cross-subject experiments, LOSO evaluation can be used to assess generalization to unseen drivers.

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

We thank the corresponding authors and open-source communities for their valuable contributions.
