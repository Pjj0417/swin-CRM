 Dynamic Swin-CRM
# Dynamic Swin-CRM for Contrastive Self-Supervised Train Driver Fatigue Detection

This repository provides the official implementation of the paper:

> **A Dynamic Swin-CRM–Based Contrastive Self-Supervised Framework for Train Driver Fatigue Detection**  
> *IEEE Internet of Things Journal (under review)*

---

## 📌 Overview

Driver fatigue poses a critical threat to railway transportation safety, especially in long-duration train driving scenarios.  
This project presents a **vision-based driver fatigue detection framework** built upon a **Dynamic Swin-CRM contrastive self-supervised learning strategy**, aiming to alleviate the limitations of feature representation capability and the strong reliance on annotated data in existing approaches.

The proposed framework leverages **self-supervised and contrastive learning** to exploit unlabeled data for learning discriminative fatigue-related visual representations.  
A **deformable window masking mechanism** is introduced to enhance hierarchical and multi-scale feature modeling.  
In addition, a **fatigue mathematical generation model** is incorporated to characterize the temporal evolution of fatigue and provide **continuous, physiologically plausible supervisory signals**.

---

## ✨ Key Contributions

- **Dynamic Swin-CRM Backbone**  
  A Dynamic Swin-CRM architecture with a *Deformable Window Masking Mechanism (DWMM)* is designed to improve hierarchical and multi-scale modeling of fatigue patterns.

- **Contrastive Self-Supervised Learning Framework**  
  A contrastive self-supervised learning strategy is developed to reduce dependence on labeled data and enhance generalization across different drivers and operating conditions.

- **Train Driver Fatigue Dataset**  
  A multi-posture, multi-level train driver fatigue dataset is constructed to support systematic evaluation and future research.

- **Fatigue Mathematical Generation Model**  
  A fatigue generation model is proposed to provide continuous and physiologically plausible fatigue supervision for training and evaluation.

---

## 🧠 Framework Overview

<p align="center">
  <img src="assets/framework.png" width="800">
</p>

*Figure: Overall framework of the proposed Dynamic Swin-CRM–based contrastive self-supervised fatigue detection method.*

---

## 📊 Experimental Evaluation

The proposed framework is evaluated using:
- **Leave-One-Subject-Out (LOSO) cross-subject validation**
- **Long-duration train driving analysis**

Experimental results demonstrate:
- Reliable and consistent detection performance  
- Favorable generalization across different drivers  
- Robustness under varying operating conditions  

Detailed quantitative results and ablation studies can be found in the paper.

---

## 📁 Dataset

The dataset contains:
- Multiple train drivers
- Multiple driving postures
- Multiple fatigue levels
- Long-duration continuous driving recordings

> ⚠️ Due to privacy and institutional regulations, the dataset is **not publicly downloadable** at this stage.  
> Please contact the authors for academic collaboration.

---

## 🚀 Getting Started

### Requirements
- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- torchvision  
- timm  
- numpy, scipy, opencv-python  

Install dependencies:
```bash
pip install -r requirements.txt
