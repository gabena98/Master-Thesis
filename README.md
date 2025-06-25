# Repository Structure Overview

This repository is organized into several main folders, each with a specific purpose related to medical data processing, modeling, and explainability.

## Folders

### `BiLSTM`
Contains the core code for the BiLSTM model:
- `BiLSTM_model.py`: Implementation of the BiLSTM neural network for sequential medical data.
- `BiLSTM_trainer.py`: Training and evaluation pipeline for the BiLSTM model.
- `BiLSTM_config.toml`: Configuration file for model and training parameters.

### `DoctorXAI`
Contains scripts and notebooks for explainability, synthetic neighborhood generation and analysis:
- `convert.py`: Converts and processes raw medical data into a format suitable for modeling.
- `jaccard_neigh.py`: Generates patient neighborhoods and computes similarity metrics (e.g., Jaccard index).
- `neigh_distribution.ipynb`: Jupyter notebook for analyzing the distribution of codes in generated neighborhoods.
- `lib/`: Contains the zig library on which the data ontological generation process is based.

### `Generator`
Contains code and configuration for generative models:
- `generator.py`: Scripts for training the generator model.
- `llama2_model.py`: Defines the model architecture, a Llama2 based model, for generative perturbation.
- `generator_config.toml`: Configuration file for model and training parameters.

### `SETOR`
Contains code and data processing scripts for the SETOR model:
- `data_eicu_processing.py`, `data_graph_building.py`, etc.: Scripts for preparing and processing MIMIC-IV medical dataset, ICD-9 ontology, and other related data.
- `SETOR_model.py`: Implementation of the SETOR model for sequence modeling.
- `SETOR_config.toml`: Configuration file for SETOR model parameters and training settings.
- `SETOR_trainer.py`: Training and evaluation pipeline for the SETOR model.
- `__init__.py`: Marks the folder as a Python package.
- `datasets/`, `outputs/`, etc.: Subfolders for processed data, dictionaries, and results.

### `code`
Contains Jupyter notebooks for data preprocessing and exploratory analysis:
- `data_preprocessing.ipynb`: Notebook for initial data analysis and visualization.

### `data`
Contains raw and reference data files:
- `ICD9CM.csv`: ICD-9-CM ontology.
- `diagnosis_gems_2018/`, `Multi_Level_CCS_2015/`, `Single_Level_CCS_2015/`: Folders with mapping tables and reference files for code conversion and categorization.

---

Each folder is designed to be modular, supporting research in medical sequence modeling, synthetic data generation, and explainable AI.