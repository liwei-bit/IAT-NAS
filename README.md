<div align="center">

# 🧠 IAT-NAS

### Imbalance-Aware Training-Free Neural Architecture Search for Clinical Image Classification

<p>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-GPU%20Acceleration-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
</p>

<p>
  <img src="https://img.shields.io/badge/Neural%20Architecture%20Search-Training--Free-6C63FF?style=for-the-badge" alt="Training-Free NAS">
  <img src="https://img.shields.io/badge/Search%20Space-NAS--Bench--201-F39C12?style=for-the-badge" alt="NAS-Bench-201">
  <img src="https://img.shields.io/badge/Application-Medical%20AI-0D5BB9?style=for-the-badge" alt="Medical AI">
</p>

<p>
  <a href="https://github.com/liwei-bit/IAT-NAS/stargazers">
    <img src="https://img.shields.io/github/stars/liwei-bit/IAT-NAS?style=for-the-badge&logo=github&color=181717" alt="GitHub Stars">
  </a>
  <a href="https://github.com/liwei-bit/IAT-NAS/commits/main">
    <img src="https://img.shields.io/github/last-commit/liwei-bit/IAT-NAS?style=for-the-badge&logo=github&color=0D5BB9" alt="Last Commit">
  </a>
</p>

<br>

**IAT-NAS** is an imbalance-aware training-free neural architecture search framework for efficient clinical image classification. It combines an imbalance-aware zero-cost proxy with evolutionary computation to identify high-quality neural architectures without repeatedly training candidate networks.

</div>

---

## 📖 Abstract

Clinical image classification plays a critical role in early disease screening and clinical decision support. However, clinical datasets often exhibit severe class imbalance, with substantially fewer samples from lesion or disease-positive categories than from prevalent or negative categories. Although convolutional neural networks and neural architecture search have achieved promising performance in image classification, existing methods are often limited by high computational overhead, weak transferability, and susceptibility to local optima.

To address these limitations, we propose **IAT-NAS**, an imbalance-aware training-free neural architecture search framework integrated with evolutionary computation. First, IAT-NAS employs training-free architecture evaluation to avoid repeatedly training candidate networks during the search process, thereby substantially reducing the search cost. Second, because directly applying existing zero-cost proxies to medical datasets may result in weak ranking correlations under class-imbalanced conditions, we introduce an imbalance-aware zero-cost proxy specifically designed for clinical image classification. Third, a three-mechanism evolutionary search strategy is developed to enhance population diversity and improve global exploration. It consists of softmax-based parent selection, adaptive mutation, and population-variance-based mutation control.

Experimental results on multiple publicly available datasets demonstrate that IAT-NAS effectively alleviates the adverse effects of class imbalance and achieves competitive classification performance and stronger architecture-ranking consistency compared with recent state-of-the-art methods.

---

## ✨ Key Contributions

- 🚀 **Training-free architecture evaluation:** Candidate architectures are evaluated without full network training, substantially reducing the computational cost of neural architecture search.

- ⚖️ **Imbalance-aware zero-cost proxy:** A task-specific proxy is designed to provide more reliable architecture rankings under class-imbalanced clinical data.

- 🧬 **Diversity-aware evolutionary search:** Softmax-based parent selection, adaptive mutation, and variance-based mutation control are integrated into the evolutionary framework to enhance population diversity and reduce premature convergence.

- 📊 **Extensive experimental validation:** IAT-NAS is evaluated on multiple clinical image datasets and standard NAS benchmarks in terms of classification performance, ranking consistency, and search efficiency.

---

## 🏗️ Overall Framework

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/0ef623ff-5701-4662-b6ad-7b25c26eb1c8"
    width="95%"
    alt="Overall framework of IAT-NAS"
  />
</p>

<p align="center">
  <em>Overall framework of the proposed IAT-NAS method.</em>
</p>

---

## 🔬 Method Overview

IAT-NAS consists of an imbalance-aware training-free evaluation method and a diversity-aware evolutionary architecture search algorithm.

### ⚖️ Imbalance-Aware Training-Free Evaluation

The proposed zero-cost proxy evaluates candidate architectures using only a small number of forward and backward operations. Class-frequency-aware weighting is introduced to reduce the dominance of majority classes and improve the ranking consistency between proxy scores and the final performance of candidate architectures.

### 🧬 Evolutionary Architecture Search

The evolutionary search process contains three complementary mechanisms:

1. **Softmax-Based Parent Selection**

   Candidate parents are sampled according to a temperature-controlled softmax distribution over their normalized proxy scores.

2. **Adaptive Mutation**

   Architectures with relatively low proxy scores are assigned higher mutation rates to encourage exploration, whereas high-quality architectures undergo more conservative mutations.

3. **Variance-Based Mutation Control**

   The global mutation rate is dynamically adjusted according to the score variance of the current population. A low population variance increases exploration, while a high variance encourages stable exploitation.

These mechanisms jointly improve population diversity and help the search algorithm discover high-quality architectures without requiring candidate-network training.

---

## 🔎 Search Space

IAT-NAS adopts the **NAS-Bench-201** cell-based search space. Each candidate cell contains six directed edges, and each edge selects one operation from the following set:

| Operation | Description |
|---|---|
| `none` | No connection |
| `skip_connect` | Identity mapping |
| `conv1x1` | 1 × 1 convolution |
| `conv3x3` | 3 × 3 convolution |
| `avg_pool_3x3` | 3 × 3 average pooling |

This unified search space facilitates fair and reproducible comparisons between different training-free architecture evaluation methods.

---

## 🗃️ Datasets

The experiments involve clinical image datasets and standard NAS benchmark datasets.

| Category | Dataset | Task |
|---|---|---|
| 🏥 Clinical image | MedMNIST | Multi-domain medical image classification |
| 🔬 Clinical image | ISIC-2019 | Skin lesion classification |
| 🩻 Clinical image | BUSI | Breast ultrasound image classification |
| 🖼️ NAS benchmark | CIFAR-10 | Architecture-ranking evaluation |
| 🖼️ NAS benchmark | CIFAR-100 | Architecture-ranking evaluation |
| 🌐 NAS benchmark | ImageNet16-120 | Architecture-ranking evaluation |

> [!NOTE]
> The datasets should be downloaded from their official sources. Users must comply with the corresponding licenses and terms of use.

---

## 📁 Repository Structure

```text
IAT-NAS/
├── ieznas_switchable.py
├── lib/
│   ├── dataop/
│   │   └── ISIC_2019.py
│   ├── models/
│   │   └── nas201_model.py
│   ├── nas_201_api/
│   │   ├── __init__.py
│   │   └── api.py
│   └── procedures/
│       ├── fisher_proxy.py
│       ├── fisher_proxy_patch.py
│       ├── fisher_proxy_optimized.py
│       ├── otherproxies.py
│       └── proxies.py
└── README.md
```

---

## ⚙️ Environment

The code is implemented in Python using PyTorch.

### Main Dependencies

| Package | Requirement |
|---|---|
| Python | 3.10 |
| PyTorch | Required |
| torchvision | Required |
| NumPy | Required |
| pandas | Required |
| Pillow | Required |
| scikit-learn | Required |
| tqdm | Required |
| MedMNIST | Required for MedMNIST experiments |

A CUDA-enabled GPU is recommended for efficient proxy calculation and architecture search.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/liwei-bit/IAT-NAS.git
cd IAT-NAS
```

### 2. Create a Conda environment

```bash
conda create -n iatnas python=3.10 -y
conda activate iatnas
```

### 3. Install the dependencies

```bash
pip install torch torchvision
pip install numpy pandas pillow scikit-learn tqdm medmnist
```

---

## 🗂️ Dataset Preparation

### ISIC-2019

Download the ISIC-2019 training images and ground-truth labels, and organize them as follows:

```text
datasets/
└── ISIC_2019/
    ├── ISIC_2019_Training_Input/
    │   ├── ISIC_0000000.jpg
    │   ├── ISIC_0000001.jpg
    │   └── ...
    ├── ISIC_2019_Training_Input.zip
    └── ISIC_2019_Training_GroundTruth.csv
```

The image ZIP file is optional if the images have already been extracted. When the extracted image directory is unavailable, the data loader attempts to extract `ISIC_2019_Training_Input.zip` automatically.

By default, the dataset is divided into training, validation, and test sets using stratified sampling with a ratio of 70%, 15%, and 15%, respectively.

---

## 🚀 Architecture Search

### Full IAT-NAS Search

The following command enables all three evolutionary search mechanisms:

```bash
python ieznas_switchable.py \
  --dataset isic2019 \
  --isic_root ./datasets/ISIC_2019 \
  --isic_size 32 \
  --proxy tail_fisher \
  --class_weight_exp 0.3 \
  --batch_size 128 \
  --sample_budget 38400 \
  --mu 64 \
  --lmbda 64 \
  --generations 20 \
  --seeds 0,1,2 \
  --m1_softmax_parent 1 \
  --parent_sel softmax \
  --tau 3.0 \
  --m2_adaptive_mut 1 \
  --mutation_rate 0.3 \
  --m3_var_ctrl 1 \
  --var_low 1e-6 \
  --var_high 1e-4 \
  --out_dir results/iatnas
```

<details>
<summary><b>▶️ Baseline evolutionary search</b></summary>

<br>

The following command disables the three proposed mechanisms and uses uniform parent selection with a fixed mutation rate:

```bash
python ieznas_switchable.py \
  --dataset isic2019 \
  --isic_root ./datasets/ISIC_2019 \
  --isic_size 32 \
  --proxy tail_fisher \
  --class_weight_exp 0.3 \
  --batch_size 128 \
  --sample_budget 38400 \
  --mu 64 \
  --lmbda 64 \
  --generations 20 \
  --seeds 0,1,2 \
  --m1_softmax_parent 0 \
  --parent_sel uniform \
  --m2_adaptive_mut 0 \
  --mutation_rate 0.3 \
  --m3_var_ctrl 0 \
  --out_dir results/baseline
```

</details>

---

## 🧪 Ablation Settings

The three search mechanisms can be independently enabled or disabled:

| Mechanism | Argument | Description |
|---|---|---|
| 🎯 Softmax-based parent selection | `--m1_softmax_parent` | Enables score-guided parent selection |
| 🧬 Adaptive mutation | `--m2_adaptive_mut` | Dynamically adjusts the mutation rate for each parent |
| 📉 Variance-based mutation control | `--m3_var_ctrl` | Adjusts the global mutation rate according to population variance |

For fair comparisons, the population size, offspring size, number of generations, sample budget, random seeds, and dataset split should remain unchanged across different ablation settings.

---

## 🛠️ Main Arguments

<details>
<summary><b>▶️ View the complete argument list</b></summary>

<br>

| Argument | Default | Description |
|---|---:|---|
| `--dataset` | `organcmnist` | Target dataset or MedMNIST subset |
| `--proxy` | `tail_fisher` | Training-free architecture proxy |
| `--class_weight_exp` | `0.3` | Exponent used for class-frequency-aware weighting |
| `--batch_size` | `128` | Batch size used for proxy calculation |
| `--sample_budget` | `38400` | Number of samples used for each proxy score |
| `--device` | `cuda:0` | Computing device |
| `--mu` | `64` | Population size |
| `--lmbda` | `64` | Number of offspring per generation |
| `--generations` | `20` | Number of evolutionary generations |
| `--seeds` | `0` | Comma-separated random seeds |
| `--parent_sel` | `softmax` | Parent-selection strategy |
| `--tau` | `3.0` | Temperature for softmax parent selection |
| `--mutation_rate` | `0.3` | Initial mutation rate |
| `--mut_min` | `0.1` | Minimum mutation rate |
| `--mut_max` | `0.5` | Maximum mutation rate |
| `--out_dir` | `results/ieznas` | Output directory |
| `--report_topk` | `10` | Number of top-ranked architectures to report |

</details>

---

## 💾 Output Files

Search results are saved in the directory specified by `--out_dir`.

```text
results/
└── iatnas/
    ├── dataset_configuration_seed0.json
    ├── dataset_configuration_seed0.csv
    ├── dataset_configuration_seed1.json
    ├── dataset_configuration_seed1.csv
    └── ...
```

Each JSON file contains:

- ⚙️ Experimental configuration
- 🗃️ Dataset and proxy information
- 🧬 Evolutionary search parameters
- 📈 Generation-wise search history
- 🏆 Top-ranked architectures and proxy scores

Each CSV file records:

- Best proxy score
- Mean population score
- Standard deviation of population scores
- Number of unique architectures
- Population entropy
- Global mutation rate

---

## 📊 Evaluation Metrics

IAT-NAS is evaluated from both classification and architecture-ranking perspectives:

- 🎯 Classification accuracy
- 📈 Area under the ROC curve
- ⚖️ F1-score
- 🔢 Kendall's rank correlation coefficient
- 📉 Spearman's rank correlation coefficient
- ⏱️ Architecture search cost

---

## ♻️ Reproducibility

To ensure fair and reproducible comparisons, we recommend:

- Using the same dataset split across all methods
- Keeping the proxy sample budget unchanged
- Fixing the population and offspring sizes
- Using the same number of evolutionary generations
- Reporting results over multiple random seeds
- Using identical training settings to evaluate the searched architectures

> [!IMPORTANT]
> The sample budget, population size, number of generations, and random seeds should remain unchanged when comparing different ablation settings.

---

## 📝 Citation

If this work is useful for your research, please cite the corresponding paper. The complete citation information will be added after publication.

---

## 🙏 Acknowledgements

This project builds upon the NAS-Bench-201 search space and publicly available clinical image datasets. We thank the authors and maintainers of these resources for supporting reproducible neural architecture search research.

---

## 📬 Contact

If you have questions about the code or experimental settings, please open an issue in this repository.

<div align="center">

<br>

⭐ **If you find this project useful, please consider giving it a star.**

<br>

<img src="https://img.shields.io/badge/Research-Neural%20Architecture%20Search-6C63FF?style=flat-square" alt="NAS">
<img src="https://img.shields.io/badge/Application-Clinical%20Image%20Classification-0D5BB9?style=flat-square" alt="Clinical Image Classification">

</div>
