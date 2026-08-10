# Dynamic Swin-CRM

## Dynamic Swin-CRM for Contrastive Self-Supervised Train Driver Fatigue Detection

This repository provides code examples and reference implementations related to the paper:

> **A Dynamic Swin-CRM-Based Contrastive Self-Supervised Framework for Train Driver Fatigue Detection**  

The current release provides representative implementations for train driver fatigue recognition. The runnable self-supervised example released in this repository is based on **Swin-MAE with DWMM**.

---

## 📌 Overview

Train driver fatigue is an important safety concern in railway transportation, particularly during long-duration and monotonous driving operations.

This study investigates a **contrastive self-supervised learning framework** that combines hierarchical visual representation learning, deformable window modeling, masked image reconstruction, and contrastive representation learning.

The repository provides **Swin-MAE + DWMM** as a representative runnable example for learning fatigue-related visual representations from unlabeled facial images. Several commonly used visual backbones are also included for comparison experiments.

---

## 🧠 Framework Overview

<p align="center">
  <img src="1.png" width="800">
</p>

<p align="center">
  <b>Figure 1. Overall framework of the proposed train driver fatigue representation learning method.</b>
</p>

The overall framework mainly involves three learning components:

- **DWMM**
- **Reconstruction Learning**
- **Contrastive Self-Supervised Learning**

The runnable example highlighted in this repository focuses on **Swin-MAE + DWMM**.

---

## 🧩 DWMM

The **Deformable Window Modeling Module (DWMM)** is introduced into window-based self-attention to enhance adaptive local feature representation.

Unlike conventional window attention with fixed spatial sampling locations, DWMM introduces learnable spatial offsets that allow local feature sampling positions to be dynamically adjusted according to the input features.

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

The reconstruction learning component follows a masked image modeling strategy.

During self-supervised pretraining, a large proportion of the input image patches are masked. The remaining visual information is encoded by the DWMM-enhanced Swin Transformer, and a lightweight decoder is used to reconstruct the missing image patches.

The reconstruction objective encourages the encoder to infer missing visual information from the surrounding context and learn meaningful local and contextual representations.

---

## 🔗 Contrastive Self-Supervised Learning

Contrastive self-supervised learning is incorporated into the overall framework to improve representation consistency and discriminative capability.

Different augmented views of the same sample are mapped into a shared representation space. Representations corresponding to the same sample are encouraged to maintain semantic consistency, while representations from different samples provide discriminative information.

The contrastive learning process mainly involves:

- Contrastive feature projection
- Online and momentum feature mapping
- Feature normalization
- Positive and negative similarity computation
- Contrastive loss optimization

Together with reconstruction learning and DWMM-based feature modeling, contrastive learning contributes to more robust fatigue-related visual representations.

---

## 📁 Dataset

The image dataset used in the experiments is available on Figshare.

**DOI and access link:**  
[https://doi.org/10.6084/m9.figshare.33135473](https://doi.org/10.6084/m9.figshare.33135473)

<p align="center">
  <img src="2.png" width="800">
</p>

<p align="center">
  <b>Figure 2. Example images from the train driver fatigue dataset.</b>
</p>

Please refer to the Figshare record for the dataset files and associated access information.

---

## 📊 Experimental Setting

The downstream fatigue recognition task is formulated as binary classification:

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

Grad-CAM++ can additionally be used for visual interpretation.

---

## 🚀 Installation

### 1. Recommended System

A CUDA-capable NVIDIA GPU is recommended for training and required by the provided GPU inference benchmark example.

Recommended software environment:

```text
Operating system : Linux / WSL2
Python           : 3.10
PyTorch          : 2.x
torchvision      : compatible with the installed PyTorch
NVIDIA driver    : compatible with the selected PyTorch CUDA build
```

If GPU monitoring is enabled during deployment benchmarking, the NVIDIA driver should provide the `nvidia-smi` command.

---

### 2. Create the Conda Environment

```bash
conda create -n fatigue python=3.10 -y
conda activate fatigue

python -m pip install --upgrade pip setuptools wheel
```

---

### 3. Install PyTorch

The following example uses PyTorch 2.7.1 with the CUDA 12.8 wheel:

```bash
pip install \
torch==2.7.1 \
torchvision==0.22.1 \
torchaudio==2.7.1 \
--index-url https://download.pytorch.org/whl/cu128
```

Check the installation:

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

### 4. Install Project Dependencies

```bash
pip install -r requirements.txt
```

The main Python dependencies include:

```text
timm
numpy
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
safetensors
gdown
```

### 5. Optional Apex AMP

Apex is **not required** when the training script is run with:

```text
--amp-opt-level O0
```

If a legacy experiment script explicitly uses Apex with `O1` or `O2`, install NVIDIA Apex separately and ensure that its CUDA build is compatible with the installed PyTorch environment.

---

## ⚡ CUDA and PyTorch Acceleration

For fixed-size CUDA inference, the deployment benchmark example enables the following performance options:

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

These settings are useful when the input size remains fixed.

On NVIDIA GPUs that support TF32, the input tensors and model parameters can remain in FP32 while supported matrix operations use Tensor Core acceleration.

If a strict FP32 benchmark without TF32 is required, disable TF32:

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

For training scripts that support activation checkpointing, enabling checkpointing can reduce GPU memory usage at the cost of additional computation.

---

## 📂 Project Structure

The repository contains the runnable example together with comparison backbones and experimental utilities.

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
|-- requirements.txt
`-- README.md
```

The main runnable example described in this README is:

```text
models/swin_mae_dwmm_model.py
```

with configuration files under:

```text
configs/swin_mae_dwmm/
```

Other model definitions and training scripts are retained for comparative experiments and evaluation.

---

## ▶️ Swin-MAE + DWMM Example

The provided runnable example combines:

```text
Swin Transformer
        +
       DWMM
        +
Masked Image Reconstruction
```

During self-supervised pretraining, partially masked facial images are processed by the DWMM-enhanced Swin Transformer encoder. A lightweight decoder reconstructs the missing image patches from the learned latent representations.

Run the example with:

```bash
chmod +x run_swin_mae_dwmm_example.sh
bash run_swin_mae_dwmm_example.sh
```

A different dataset location can be passed without modifying the script:

```bash
DATA_PATH=/path/to/dataset \
bash run_swin_mae_dwmm_example.sh
```

If GPU memory is limited:

```bash
PRETRAIN_BATCH_SIZE=4 \
FINETUNE_BATCH_SIZE=16 \
ACCUMULATION_STEPS=2 \
bash run_swin_mae_dwmm_example.sh
```

---

## ⬇️ Pretrained Weights

The comparison experiments use either public ImageNet pretrained weights or public self-supervised checkpoints.

### Standard Backbone Weights

| Model | Pretraining | Direct download |
|---|---|---|
| CoAtNet-0 | ImageNet-1K | [pytorch_model.bin](https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k/resolve/main/pytorch_model.bin) |
| ConvNeXt-Base | ImageNet-1K | [convnext_base_1k_224_ema.pth](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth) |
| DeiT-Base | ImageNet-1K | [deit_base_patch16_224-b5f2ef4d.pth](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) |
| DeiT III Base | ImageNet-22K -> ImageNet-1K | [model.safetensors](https://huggingface.co/timm/deit3_base_patch16_224.fb_in22k_ft_in1k/resolve/main/model.safetensors) |
| DINOv2 ViT-S/14 | DINOv2 self-supervised pretraining | [dinov2_vits14_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) |
| EdgeNeXt-Small | ImageNet-1K | [pytorch_model.bin](https://huggingface.co/timm/edgenext_small.usi_in1k/resolve/main/pytorch_model.bin) |
| MobileViTv2-1.0 | ImageNet-1K | [pytorch_model.bin](https://huggingface.co/timm/mobilevitv2_100.cvnets_in1k/resolve/main/pytorch_model.bin) |
| ResNet50 | ImageNet-1K V2 | [resnet50-11ad3fa6.pth](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) |
| Swin-Base | ImageNet-1K | [swin_base_patch4_window7_224.pth](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth) |
| SwinV2-CR Small NS | ImageNet-1K | [model.safetensors](https://huggingface.co/timm/swinv2_cr_small_ns_224.sw_in1k/resolve/main/model.safetensors) |

### Public Self-Supervised Weights

| Method | Backbone | Pretraining | Download |
|---|---|---|---|
| MixMAE | Swin-B / W14 | 600 epochs | [Google Drive](https://drive.google.com/file/d/1pZYmTv08xK_kOe2kk6ahuvgJVkHm-ZIa/view?usp=sharing) |
| SimMIM | Swin-Base | 100 epochs | [Google Drive](https://drive.google.com/file/d/1Wcbr66JL26FF30Kip9fZa_0lXrDAKP-d/view?usp=sharing) |
| SimMIM | Swin-Base | 800 epochs | [Google Drive](https://drive.google.com/file/d/15zENvGjHlM71uKQ3d2FbljWPubtrPtjl/view?usp=sharing) |
| SimMIM | Swin-Large | 800 epochs | [Google Drive](https://drive.google.com/file/d/1qDxrTl2YUDB0505_4QrU5LU2R1kKmcBP/view?usp=sharing) |
| SimMIM | ViT-Base | 800 epochs | [Google Drive](https://drive.google.com/file/d/1dJn6GYkwMIcoP3zqOEyW1_iQfpBi8UOw/view?usp=sharing) |

Google Drive checkpoints can also be downloaded with `gdown`:

```bash
pip install gdown

# MixMAE Swin-B/W14, 600 epochs
gdown "https://drive.google.com/uc?id=1pZYmTv08xK_kOe2kk6ahuvgJVkHm-ZIa" \
  -O mixmae_swin_base_600ep.pth

# SimMIM Swin-Base, 100 epochs
gdown "https://drive.google.com/uc?id=1Wcbr66JL26FF30Kip9fZa_0lXrDAKP-d" \
  -O simmim_pretrain_swin_base_100ep.pth

# SimMIM Swin-Base, 800 epochs
gdown "https://drive.google.com/uc?id=15zENvGjHlM71uKQ3d2FbljWPubtrPtjl" \
  -O simmim_pretrain_swin_base_800ep.pth

# SimMIM Swin-Large, 800 epochs
gdown "https://drive.google.com/uc?id=1qDxrTl2YUDB0505_4QrU5LU2R1kKmcBP" \
  -O simmim_pretrain_swin_large_800ep.pth

# SimMIM ViT-Base, 800 epochs
gdown "https://drive.google.com/uc?id=1dJn6GYkwMIcoP3zqOEyW1_iQfpBi8UOw" \
  -O simmim_pretrain_vit_base_800ep.pth
```

### Project-Specific Self-Supervised Variants

The following project-specific variants are trained locally and therefore do not require a separate public external checkpoint for their self-supervised pretraining stage:

```text
Swin-MAE + DWMM
Swin-CMAE
ResNet50 + MIM/CL
other custom experimental variants
```

Their pretrained checkpoints are generated by the corresponding training scripts.

---

## 🖥️ Deployment and FP32 Inference Benchmark

A frame-level CUDA inference benchmark can be used to evaluate deployment performance after fine-tuning.

The benchmark workflow includes:

```text
Fine-tuned checkpoint
        |
        v
Single-image preprocessing
        |
        v
FP32 CUDA inference
        |
        v
Warm-up
        |
        v
Frame-level CUDA Event timing
        |
        +--> Latency statistics
        +--> FPS / throughput
        +--> GPU memory
        +--> GPU temperature
        +--> SM clock
        +--> GPU utilization
        `--> GPU power
```

### Required Files

A typical deployment layout is:

```text
Dynamic-Swin-CRM/
|
|-- config.py
|-- models/
|-- configs/
|-- training_outputs/
|   `-- .../best_ba_weights.pth
|
|-- test.jpg
`-- swinb_fp32_inference_benchmark.py
```

Before running the benchmark, set the following paths in the inference script:

```python
CFG_PATH = ...
CHECKPOINT_PATH = ...
IMAGE_PATH = ...
```

### Runtime Requirements

The deployment benchmark requires:

```text
Python >= 3.10
CUDA-compatible PyTorch
torchvision
numpy
Pillow
PyYAML
yacs
```

For GPU telemetry, `nvidia-smi` should be available.

### Run

```bash
python swinb_fp32_inference_benchmark.py
```

The benchmark can report:

- Mean latency
- Minimum latency
- P50 / P90 / P95 / P99 latency
- Maximum latency
- FPS / continuous throughput
- Peak GPU memory
- GPU temperature
- SM clock
- GPU utilization
- GPU power

Frame-level measurements can also be exported to CSV together with a complete terminal log.

### Recommended Benchmark Conditions

For reproducible deployment measurements:

1. Close unrelated GPU workloads.
2. Use a fixed input resolution.
3. Use a fixed batch size.
4. Perform sufficient GPU warm-up.
5. Repeat the benchmark when reporting final results.
6. Keep the CUDA, PyTorch, NVIDIA driver, and hardware environment unchanged between comparisons.
7. Disable GPU telemetry if the goal is to minimize interference from repeated `nvidia-smi` queries.

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

The reconstruction results provide an intuitive view of the contextual and structural information learned from partially observed facial images.

---

## 🔍 Comparison Backbones

The repository contains several visual architectures for comparison, including:

- Swin Transformer
- Swin Transformer V2
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

Install it with:

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

Deployment experiments can additionally report frame-level latency, throughput, peak GPU memory, and GPU runtime statistics.

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
