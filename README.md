# 📌 Stretcher — Deformation-Robust Keypoint Descriptors

[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

**Stretcher** is an open-source framework for improving keypoint descriptor robustness under large non-rigid deformations. It learns to apply latent space descriptor transformations conditioned on deformation parameters and enables improved matching in both synthetic and real image pairs.

This repository contains reproducibility notebooks for generating datasets, training models, and evaluating matching performance — aligned with the experiments in the associated paper.

---

## 🚀 Quick Start

### Reproducible Environment

To reproduce the exact environment used in the paper:

```bash
conda env create -f environment.yml
conda activate stretcher

---

## 🔧 Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/stretcher.git
cd stretcher
Create and activate environment

bash
Copy code
conda create -n stretcher python=3.10 -y
conda activate stretcher
Install dependencies

bash
Copy code
pip install -U pip
pip install torch torchvision numpy matplotlib pillow opencv-python lightglue
Optional (for synthetic FEM experiments):

bash
Copy code
conda install -c conda-forge fenics pyvista -y
📁 Repository Structure
css
Copy code
📦stretcher
├── 📂data
│   ├── SuperPoint_Descriptors_Dataset_Test.pth
│   └── medical_deformed/
├── 📂models
│   ├── stretcher.pth
│   └── spstretcher_new.pth
├── 📂notebooks
│   ├── dataset_creation.ipynb
│   ├── model_training.ipynb
│   ├── synthetic_matching.ipynb
│   └── real_matching.ipynb
├── 📂src
│   └── notebook_utils.py
├── 📂util
│   ├── Affine_Transformations.py
│   └── matching.py
├── 📂matchers
│   └── max_similarity.py
├── models.py
├── requirements.txt
└── README.md
🧠 Overview
The repository is organized to support the core experimental pipeline in the paper:

Dataset Generation
Extract descriptors and build deformation‐paired descriptor datasets.

Training
Train the Stretcher model to predict deformed descriptors conditioned on deformation parameters.

Synthetic Matching Evaluation
Apply synthetic FEM deformations and evaluate matching performance against baselines.

Real Matching Examples
Demonstrate the method on real surgical image pairs.

📓 Notebooks
Notebook	Purpose
dataset_creation.ipynb	Build descriptor deformation dataset
model_training.ipynb	Train the Stretcher model
synthetic_matching.ipynb	Evaluate on FEM deformation
real_matching.ipynb	Apply to real image pairs

Open each in Jupyter or VSCode and follow the linear workflow in the cells.

🧰 Usage
1) Descriptor Dataset Creation
Open the notebook:

bash
Copy code
jupyter notebook notebooks/dataset_creation.ipynb
This will:

Load source images

Extract keypoints & descriptors

Generate deformation modes

Save a dataset (data/*.pth)

2) Train the Stretcher Model
bash
Copy code
jupyter notebook notebooks/model_training.ipynb
This will:

Define the model

Load the descriptor dataset

Train and save weights to models/stretcher.pth

3) Synthetic Matching Evaluation
bash
Copy code
jupyter notebook notebooks/synthetic_matching.ipynb
Performs:

Synthetic FEM deformation generation

Matching with and without Stretcher

Quantitative analysis

4) Real Matching
bash
Copy code
jupyter notebook notebooks/real_matching.ipynb
Applies trained model to example real surgical pairs and visualizes descriptor matching.

📦 Data
The repository includes:

Descriptor dataset (.pth)

Example medical image pairs for evaluation

If using your own images, update paths inside the notebooks or define a configuration.

🧪 Evaluation
Matching performance is measured using:

Baseline descriptor matching

Stretcher-augmented descriptor matching

Synthetic ground truth from FEM deformation

📖 Citation
If you use this code in academic work, please cite:

bibtex
Copy code
@article{stretcher2025,
  title={Stretcher: A Learning-Based Framework for Deformation-Robust Keypoint Descriptors},
  author={Anonymized Authors},
  year={2025}
}
📄 License
This project is released under the MIT License. See the LICENSE file for details.

🧡 Acknowledgements
Built on feature extractors such as SuperPoint and matchers like LightGlue

Synthetic deformation using FEniCS (optional)

Contributions from 3rd-party modules in util/ and matchers/ folders