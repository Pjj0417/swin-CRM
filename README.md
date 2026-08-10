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

### 2. Install PyTorch

For an NVIDIA GPU environment using CUDA 12.8:

```bash
pip install \
torch==2.7.1 \
torchvision==0.22.1 \
torchaudio==2.7.1 \
--index-url https://download.pytorch.org/whl/cu128
```

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

The main packages used by the training and deployment examples include:

```text
torch
torchvision
timm
numpy
scipy
scikit-learn
pandas
Pillow
opencv-python
matplotlib
grad-cam
PyYAML
yacs
einops
tensorboard
safetensors
gdown
```

### 4. Check the GPU Environment

For GPU training and inference, an NVIDIA GPU with a compatible driver is recommended.

Check that the driver and GPU are visible:

```bash
nvidia-smi
```

Then verify PyTorch CUDA support:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("cuDNN:", torch.backends.cudnn.version())
PY
```

A working CUDA-enabled PyTorch installation is required for the GPU training and frame-level inference benchmark examples.

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
|
|-- inference benchmark scripts
|
|-- requirements.txt
`-- README.md
```

The main example model described in this README is:

```text
models/swin_mae_dwmm_model.py
```

Its corresponding configuration files are located under:

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

The default dataset path is:

```text
./data/fatiguev2_105270
```

A different dataset path can be supplied without modifying the launcher:

```bash
DATA_PATH=/path/to/dataset \
bash run_swin_mae_dwmm_example.sh
```

The batch size can also be adjusted according to available GPU memory:

```bash
PRETRAIN_BATCH_SIZE=4 \
FINETUNE_BATCH_SIZE=16 \
ACCUMULATION_STEPS=2 \
bash run_swin_mae_dwmm_example.sh
```

Grad-CAM++ can be disabled during an example run when only training and evaluation are required:

```bash
SKIP_GRADCAM=1 \
bash run_swin_mae_dwmm_example.sh
```

---

## ⬇️ Pretrained Weights

A helper script is provided for downloading the public pretrained weights used by the comparison backbones.

```bash
chmod +x download_pretrained_weights.sh
```

Download the standard backbone weights:

```bash
bash download_pretrained_weights.sh ./pretrained standard
```

Download the standard weights together with available self-supervised checkpoints:

```bash
bash download_pretrained_weights.sh ./pretrained all
```

The downloaded weights are stored under:

```text
./pretrained/
```

Representative public checkpoints include:

| Backbone / Method | Checkpoint |
|---|---|
| CoAtNet-0 | timm ImageNet-1K |
| ConvNeXt-Base | ImageNet-1K |
| DeiT-Base | ImageNet-1K |
| DeiT III Base | ImageNet-22K -> ImageNet-1K |
| DINOv2 ViT-S/14 | DINOv2 pretrained |
| EdgeNeXt-Small | ImageNet-1K |
| MobileViTv2-1.0 | ImageNet-1K |
| ResNet50 | ImageNet-1K V2 |
| Swin-Base | ImageNet-1K |
| SwinV2 | ImageNet-1K |
| MixMAE | public self-supervised checkpoint |
| SimMIM | public self-supervised checkpoints |

The **Swin-MAE + DWMM** example can be pretrained locally and does not require a task-specific pretrained checkpoint.

---

## 🖥️ Deployment and Inference Benchmark Example

A frame-level GPU inference benchmark can be used to evaluate deployment performance after model fine-tuning.

The benchmark example performs:

- FP32 single-image inference
- CUDA warm-up
- Repeated frame-level latency measurement
- Mean / Min / P50 / P90 / P95 / P99 / Max latency statistics
- FPS calculation
- Peak GPU memory measurement
- GPU temperature, SM clock, utilization, and power monitoring
- Per-frame CSV export
- Complete terminal log export

The provided benchmark design uses **batch size 1**, a fixed **224 x 224** input, model warm-up, and CUDA Event timing for GPU latency measurement. fileciteturn8file0L6-L22

### Deployment Requirements

The deployment benchmark requires:

```text
Python >= 3.10
PyTorch >= 2.x
CUDA-compatible PyTorch
torchvision
numpy
Pillow
PyYAML
yacs
```

The benchmark also expects the required project files, model configuration, and fine-tuned checkpoint to be available in the project directory. fileciteturn8file0L25-L53

For GPU monitoring, the system should provide:

```text
NVIDIA GPU
NVIDIA driver
nvidia-smi
```

If `nvidia-smi` is unavailable, GPU temperature, clock, utilization, and power monitoring should be disabled.

### Recommended Project Files

A deployment benchmark typically requires:

```text
config.py
models/
configs/
training_outputs/
test.jpg
```

Before running the benchmark, update the configuration and checkpoint paths in the inference script to match the local project structure.

For example:

```python
CFG_PATH = PROJECT_ROOT / "configs" / "..." / "model_config.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "training_outputs" / "..." / "best_ba_weights.pth"
IMAGE_PATH = PROJECT_ROOT / "test.jpg"
```

The supplied benchmark example loads a fine-tuned checkpoint, performs single-image FP32 inference, runs warm-up iterations, and then measures a continuous sequence of frames. fileciteturn8file0L151-L178

### Run the Inference Benchmark

Place a test image in the project root:

```text
test.jpg
```

Then run the corresponding benchmark script:

```bash
python <inference_benchmark_script>.py
```

The benchmark can generate:

```text
*_Frame_Latency.csv

inference_logs/
    *_Inference_YYYYMMDD_HHMMSS.log
```

The benchmark implementation saves both frame-level CSV results and complete terminal logs. fileciteturn8file0L81-L102

---

## ⚡ CUDA / PyTorch Acceleration

For fixed input sizes, the deployment benchmark can enable several PyTorch CUDA optimizations:

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

These options are already supported by the benchmark implementation. fileciteturn8file0L55-L79

### cuDNN Benchmark

```python
torch.backends.cudnn.benchmark = True
```

This is useful when the input shape is fixed because cuDNN can select an efficient kernel for repeated inference.

For reproducible timing comparisons, keep the input size and batch size fixed across experiments.

### TF32 Acceleration

On supported NVIDIA GPUs, TF32 can accelerate selected float32 matrix and convolution operations using Tensor Cores.

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

The model parameters and input tensors remain `torch.float32`, while selected internal operations may use TF32 hardware acceleration. fileciteturn8file0L59-L73

If a benchmark requires **strict IEEE FP32 arithmetic**, disable TF32:

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

This distinction should be reported when comparing inference latency across hardware or software environments. fileciteturn8file0L75-L79

### Inference Mode

Deployment inference should use:

```python
model.eval()

with torch.inference_mode():
    output = model(image)
```

This disables autograd bookkeeping and is appropriate for inference-only evaluation.

---

## ⏱️ Recommended Benchmark Protocol

For reliable deployment performance measurements:

1. Close other GPU-intensive applications.
2. Keep the input resolution fixed.
3. Keep the batch size fixed.
4. Warm up the model before recording latency.
5. Use CUDA Event timing for GPU execution latency.
6. Run enough consecutive frames to observe stable behavior.
7. Keep the same GPU driver, CUDA runtime, PyTorch version, and power state when comparing models.
8. Record GPU temperature and SM clock when analyzing DVFS or thermal effects.

These conditions are consistent with the benchmark recommendations included in the deployment script. fileciteturn8file0L104-L128

For the example benchmark:

```text
Input size:       224 x 224
Batch size:       1
Warm-up:          50 iterations
Measured frames:  350
Timing method:    torch.cuda.Event
```

The benchmark uses raw CUDA Event milliseconds without artificial latency scaling. fileciteturn8file0L186-L201

### GPU Monitoring and Pure Latency Testing

GPU state monitoring is useful for studying:

```text
Temperature
SM clock
GPU utilization
Power
DVFS behavior
Latency spikes
```

However, `nvidia-smi` queries introduce additional system activity.

When the goal is the cleanest possible inference latency measurement, disable GPU monitoring in the benchmark script:

```python
ENABLE_GPU_MONITORING = False
```

This reduces interference from repeated external GPU-status queries. fileciteturn8file0L117-L128

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

For deployment-oriented experiments, additional metrics can include:

```text
Mean latency
P50 / P90 / P95 / P99 latency
Throughput / FPS
Peak GPU memory
GPU temperature
GPU utilization
GPU power
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
