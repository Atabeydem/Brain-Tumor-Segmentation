# Brain Tumor Segmentation and Autoencoder-Based Representation Learning

**Master's Research | Computer Engineering | Izmir University of Economics**

A collection of research experiments investigating brain tumor segmentation, medical image reconstruction, representation learning, and anomaly detection using deep learning.

The repository includes experiments conducted on the **BRISC, BraTS-PED, BraTS2023-GLI, and UPENN-GBM** datasets, with a particular focus on evaluating whether autoencoders trained on healthy brain MRI images can learn meaningful representations for tumor detection and localization.

## Research Objectives

The primary research objectives are:

- Investigating convolutional and ResNet-based autoencoders for brain MRI reconstruction.
- Analyzing the information learned within autoencoder bottleneck representations.
- Evaluating patch-level and image-level tumor detection using healthy-trained representations.
- Comparing unsupervised anomaly-scoring methods and supervised diagnostic linear probes.
- Developing and evaluating 2D and 3D U-Net-based segmentation approaches.
- Investigating the effects of preprocessing, patch geometry, image resolution, and representation normalization on model performance.

---

## 1. BRISC — Autoencoder and Anomaly Detection Research

The BRISC experiments constitute the main focus of the current autoencoder research.

### 1.1 Healthy MRI Reconstruction

Several autoencoder architectures were investigated, including conventional CNN autoencoders, U-Net-based autoencoders, U-Net++, and custom ResNet-style encoder-decoder models.

The main ResNet-based autoencoder was implemented in PyTorch and trained on healthy brain MRI images.

Its architecture includes:

- A convolutional encoder-decoder structure.
- Residual blocks with local skip connections.
- No U-Net-style long encoder-to-decoder skip connections.
- A base channel configuration of 64.
- A 2048-dimensional bottleneck representation.

The architecture was trained from scratch rather than initialized from pretrained torchvision ResNet weights.

**Selected reconstruction results:**

| Experiment | PSNR |
|---|---:|
| ResNet base64 — initial training | 64.12 dB |
| ResNet base64 — low-learning-rate continuation | **64.51 dB** |
| ResNet base64 — fixed 256×256 pipeline, whole-image evaluation | 63.49 dB |
| ResNet base64 — fixed 256×256 pipeline, patch evaluation | 64.40 dB |

The native-resolution ResNet model reached approximately 64.51 dB PSNR after low-learning-rate continuation.

These measurements represent reconstruction quality on healthy test images, not tumor detection or segmentation accuracy.

### 1.2 Fixed-Size Patch Processing

To investigate localized tumor information, the preprocessing pipeline was adapted to use fixed **256×256 patches without spatial resizing**.

The approach includes:

- Non-overlapping patch extraction.
- Replicate padding for incomplete image boundaries.
- Zero padding for corresponding segmentation masks.
- Exclusion of padded regions from reconstruction evaluation.
- Mask-based patch labeling.
- Independent feature extraction for each patch, without embedding averaging.

A patch is labeled tumor-positive when at least one valid tumor pixel is present in its corresponding segmentation mask.

**Dataset statistics:**

| Data split | Images | Patches | Tumor-positive patches |
|---|---:|---:|---:|
| Healthy training | 1,067 | 3,517 | 0 |
| Healthy testing | 140 | 406 | 0 |
| Segmentation training | 3,933 | 15,673 | 7,990 |
| Tumor testing | 860 | 3,341 | 1,684 |

### 1.3 Latent Space Analysis

A separate 2048-dimensional latent embedding was extracted for each patch.

The learned representations were investigated using:

- Principal Component Analysis (PCA)
- Locally Linear Embedding (LLE)
- t-distributed Stochastic Neighbor Embedding (t-SNE)
- Two-dimensional and interactive three-dimensional visualizations

Additional analyses examined the effects of padding, anatomical patch position, source image resolution, and patch geometry.

**Key observation:** The learned latent space was strongly influenced by anatomical and acquisition-related characteristics. Tumor-positive and tumor-negative patches were not consistently separable through raw cosine similarity alone.

### 1.4 Unsupervised Anomaly Detection

Several methods were evaluated using representations extracted from the healthy-trained autoencoder.

The experiments included raw cosine similarity, position- and geometry-matched reference comparisons, PCA whitening, nearest-neighbor scoring, and validation-based threshold selection.

**Selected independent test results:**

| Method | Patch sensitivity | Patch F1 | Image sensitivity | Image F1 |
|---|---:|---:|---:|---:|
| Raw cosine similarity | 13.48% | 0.208 | 38.37% | 0.553 |
| Matched cosine similarity | 10.99% | 0.181 | 24.19% | 0.382 |
| PCA128 whitening + top-10 | 25.24% | 0.359 | 53.72% | 0.694 |
| PCA120 whitening, balanced configuration | 37.35% | 0.457 | 72.21% | 0.829 |
| PCA112 whitening, high-sensitivity configuration | **43.47%** | **0.499** | **78.72%** | **0.867** |

The strongest selected high-sensitivity cosine configuration achieved 82.86% image-level specificity, although patch-level ROC-AUC remained approximately 0.62.

The improvement was obtained through latent-space processing and anomaly-scoring optimization without changing the underlying autoencoder architecture.

### 1.5 Frozen Encoder Linear Probing

To investigate whether the limited anomaly detection results originated from the learned representations or the similarity-based scoring mechanism, diagnostic supervised linear probing was performed.

The healthy-trained autoencoder encoder was **completely frozen**.

Its existing 2048-dimensional latent representations were evaluated using Logistic Regression and Linear Support Vector Machines (SVM).

Model selection and threshold tuning were performed using validation data, followed by evaluation on independent test data.

**Selected Logistic Regression results:**

| Metric | Patch-level | Image-level |
|---|---:|---:|
| Sensitivity | 76.54% | 94.88% |
| Specificity | 68.93% | 92.14% |
| Precision | 66.79% | 98.67% |
| F1-score | **0.713** | **0.967** |
| ROC-AUC | **0.818** | — |

Among 860 tumor-containing test images, 816 were classified as positive, and 752 contained at least one correctly detected tumor-positive patch.

**Methodological distinction:** Unlike the unsupervised anomaly-scoring experiments, the linear probe was trained using segmentation-mask-derived tumor labels. Therefore, these results must not be interpreted as healthy-only unsupervised anomaly detection performance.

The linear probe was introduced to assess the information available in the frozen autoencoder representations, not to replace the primary autoencoder architecture.

### 1.6 Current Research Direction

The current research focuses on improving the use and interpretation of latent representations while preserving the established ResNet-based autoencoder architecture.

The findings suggest that the healthy-trained encoder contains useful tumor-related information, but conventional similarity-based anomaly scores do not fully exploit it.

Further investigation is required into bottleneck behavior, feature normalization, anomaly scoring, and localized discrimination.

---

## 2. BRISC — Supervised Tumor Segmentation

Separate supervised segmentation experiments were conducted using U-Net architectures and corresponding MRI segmentation masks.

The experiments included baseline segmentation and autoencoder-pretrained initialization approaches.

The strongest previously recorded BRISC U-Net experiment achieved a Dice score of approximately **0.824** and IoU of approximately **0.728**.

These supervised segmentation experiments are separate from the healthy-only autoencoder anomaly detection pipeline.

---

## 3. BraTS-PED — Pediatric Brain Tumor Segmentation

Experiments on pediatric brain MRI data investigated volumetric image processing, autoencoder-based learning, and U-Net segmentation approaches.

The repository includes scripts for dataset analysis, split preparation, model training, sliding-window evaluation, and postprocessing.

A previously recorded 3D U-Net experiment achieved a Dice score of approximately **0.825** under its evaluation protocol.

---

## 4. BraTS2023-GLI — 3D Brain Tumor Segmentation

This research branch investigates three-dimensional glioma segmentation using volumetric MRI data.

The implementation includes dataset preparation, 3D U-Net training, input validation, and sliding-window inference.

The strongest recorded 3D U-Net experiment achieved a Dice score of approximately **0.904** under its evaluation protocol.

Results across different datasets and experimental protocols should not be interpreted as direct model-to-model comparisons.

---

## 5. UPENN-GBM — Dataset Exploration

The UPENN-GBM branch contains exploratory dataset analysis and visualization scripts for glioblastoma imaging data.

These experiments focus on understanding dataset characteristics, image formats, and preprocessing requirements.

---

## Technologies

**Programming:** Python

**Deep Learning:** PyTorch, CUDA

**Machine Learning:** scikit-learn

**Data Processing:** NumPy, Pandas

**Visualization:** Matplotlib, Plotly, PCA, LLE, t-SNE

**Research Methods:** Autoencoders, ResNet-style residual learning, U-Net, 3D U-Net, anomaly detection, nearest-neighbor analysis, linear probing

---

## Repository Structure

```text
Brain-Tumor-Segmentation/
├── BRISC/
│   ├── ae_anomaly/
│   │   ├── brisc_healthy_ae_pipeline/
│   │   │   ├── PATCH_LEVEL_ANALYSIS_20261013/
│   │   │   ├── brisc_ae_common.py
│   │   │   ├── train_eval_brisc_healthy_fixed256_resnet_20261011.py
│   │   │   └── README_BRISC_HEALTHY_AE.md
│   │   └── brisc_tumor_reconstruction_pipeline/
│   ├── autoencoder/
│   └── segmentation/
├── BraTS2023/
├── BraTS2023_GLI/
├── BraTS_ped/
├── UPENN_GBM/
├── configs/
├── requirements.txt
└── README.md
```

The repository preserves historical experimental scripts for research traceability. Some scripts contain experiment-specific paths and configurations and may require adaptation before execution in another environment.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Atabeydem/Brain-Tumor-Segmentation.git
cd Brain-Tumor-Segmentation
```

Create a Python virtual environment and install the listed dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

A compatible PyTorch/CUDA installation may be required for GPU execution.

Datasets must be obtained separately, and dataset paths in experiment scripts may need to be configured for the local environment.

The repository is a research code collection rather than a fully packaged, one-command reproducibility framework.

---

## Data and Model Availability

Raw medical imaging datasets, large model checkpoints, extracted embedding arrays, and generated experimental outputs are excluded from Git due to their size and/or dataset distribution considerations.

Selected evaluation summaries and visualizations are included where available.

The original datasets must be obtained from their respective authorized sources, subject to their licensing and access requirements.

---

## Research Status

**Ongoing master's-level research.**

The reported findings represent completed experiments and interim research results. Additional validation, methodological analysis, and further experiments are ongoing.

The principal focus remains understanding and improving autoencoder-based MRI representations while preserving the established autoencoder architecture.