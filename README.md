# Repository Structure Overview

This repository is organized into several main folders, each with a specific purpose related to medical data processing, modeling, and explainability.

## Folders

### `BiLSTM`
Contains the core code for the BiLSTM model:
- `BiLSTM_model.py`: Implementation of the BiLSTM neural network for sequential medical data.
- `BiLSTM_trainer.py`: Training and evaluation pipeline for the BiLSTM model.
- `BiLSTM_config.toml`: Configuration file for model and training parameters.

### `DoctorXAI`
Contains scripts and notebooks for explainability, neighborhood generation, and synthetic data analysis:
- `convert.py`: Converts and processes raw medical data into a format suitable for modeling.
- `jaccard_neigh.py`: Generates patient neighborhoods and computes similarity metrics (e.g., Jaccard index).
- `neigh_distribution.ipynb`: Jupyter notebook for analyzing the distribution of codes in generated neighborhoods.
- `old_generator.py`: Legacy or experimental code for synthetic data generation.
- `lib/`: May contain low-level or performance-critical code (e.g., Zig or C extensions).

### `Generator`
Contains code and configuration for generative models:
- `generator.py`: Scripts for generating synthetic patient data.
- `llama2_model.py`: Utilities for using LLMs (e.g., Llama 2) for generative perturbation.
- `generator_config.toml`: Configuration file for the generator.

### `SETOR`
Contains code and data processing scripts for the SETOR model:
- `data_eicu_processing.py`, `data_graph_building.py`, etc.: Scripts for preparing and processing various medical datasets (e.g., MIMIC, eICU).
- `__init__.py`: Marks the folder as a Python package.
- `datasets/`, `outputs/`, etc.: Subfolders for processed data, dictionaries, and results.
- `README.md`: Documentation specific to the SETOR model and its processing.

### `code`
Contains Jupyter notebooks for data preprocessing and exploratory analysis:
- `data_preprocessing.ipynb`: Notebook for initial data cleaning and preparation.

### `data`
Contains raw and reference data files:
- `ICD9CM.csv`: ICD-9-CM ontology.
- `diagnosis_gems_2018/`, `Multi_Level_CCS_2015/`, `Single_Level_CCS_2015/`: Folders with mapping tables and reference files for code conversion and categorization.

---

Each folder is designed to be modular, supporting research in medical sequence modeling, synthetic data generation, and explainable AI.