# R-GCN for High-Resolution Mass Spectrometry Imaging (MSI)

## 🧠 Overview

This repository contains the official code for the paper:

**"Relational Graph Convolutional Networks for Classifying and Interpreting High-Resolution Mass Spectrometry Imaging Data"**


In this work, we propose a deep learning architecture based on **Relational Graph Convolutional Networks (R-GCN)** to classify spectra from Mass Spectrometry Imaging (MSI) data while capturing the structural information specific to **high-resolution mass spectrometry (HRMS)**. Our model incorporates chemically meaningful mass relationships—such as **mass defect**, **mass**,**mass differences**, and **intensity patterns**—by modeling spectra as graphs.

In addition to improved robustness over conventional CNN or vector-based methods, the R-GCN model supports **ion-level interpretability** via **Class Activation Mapping (CAM)**, making it useful for molecular feature discovery in MSI datasets.

## 📁 Repository Description

This repository includes:
- Code for training and testing the R-GCN model (`src/`)
- Configuration utilities for preprocessing data
- Notebooks for running 1D-CNN and classical ML baselines
- A full Conda environment specification (`environment.yml`)

## 📦 Creating the Environment

This project uses a Conda environment to manage dependencies. To create and activate the environment:

```bash
conda env create -f environment.yml
conda activate Massconvnet
```

The file `environment.yml` includes all the required libraries and exact versions used in the paper.

## 🚀 Running the R-GCN Model

### ✅ Example command:

```bash
python src/main.py \
    --dataset_path="/path/to/data/" \
    --pre_process_param_name=param_pre1 \
    --network_param_name=param_net1 \
    --with_masses \
    --normalize \
    --with_intensity \
    --max_epochs 40 \
    --random_state 5 \
    --cam_only
```

### 🔍 Command-Line Argument Descriptions:

| Argument                  | Description |
|---------------------------|-------------|
| `--dataset_path`          | Path to the dataset folder (results also saved here) |
| `--pre_process_param_name` | Name of the preprocessing parameter `.json` file |
| `--network_param_name`     | Name of the network parameter `.json` file |
| `--with_masses`            | Include m/z and mass defect in node features |
| `--with_intensity`         | Include intensity as a node feature (if omitted, all intensities set to 1) |
| `--normalize`              | Normalize the intensity values |
| `--max_epochs`             | Number of training epochs |
| `--random_state`           | Random seed for reproducibility |
| `--cam_only`               | Skip training and run only CAM interpretation |

## 📊 CAM-Only Mode

To use `--cam_only`, a pre-trained model must exist at:

```
./models/GCN/model.pth.tar
```

This mode loads the model and performs CAM analysis across all spectra in the dataset. It outputs the following files:

```
dataset_path/models/CAM_output/
├── CAM_<spectrum_id>.pt              # CAM scores for each ion
├── OUT_<spectrum_id>.pt              # Classification result
└── Ion_embedding_<spectrum_id>.pt    # Learned ion embeddings
```

## 📓 Jupyter Notebooks

| Notebook                      | Purpose |
|------------------------------|---------|
| `how_to_run_RGCN.ipynb`      | Guide for dataset setup, JSON config creation, and model training |
| `how_to_run_1DCNN.ipynb`     | Train and evaluate the 1D-CNN baseline |
| `how_to_run_ML.ipynb`        | Run classical machine learning models (LDA, SVM, RF, XGBoost) |


## How to cite us?
Please cite our publication by using the following bibtex entry:


```bibtex

```

## License

Apache v2.0
See the [LICENSE](LICENSE) file for details.
