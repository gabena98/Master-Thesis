# DoctorXAI

This folder contains scripts, utilities, and libraries for explainability, neighborhood generation, and synthetic data analysis in the context of medical sequence modeling.

## Contents

- **convert.py**  
  Script for converting and preprocessing raw medical data (ICD, CCS, ontology) into efficient parquet files for downstream analysis and modeling.

  **How to run:**  
  ```sh
  python convert.py
  ```
  Make sure the input data paths in the script are correct before running.

- **jaccard_neigh.py**  
  Generates real and synthetic patient neighborhoods, computes similarity metrics (e.g., Jaccard index), and evaluates outlier scores (LOF) for explainability and robustness studies.

  **How to run:**  
  ```sh
  python jaccard_neigh.py --k_reals 50 --synt_neigh_size 100 --ontological_perturbation
  python jaccard_neigh.py --k_reals 50 --synt_neigh_size 100 --generative_perturbation
  ```
  Adjust the arguments as needed for your experiment.

- **neigh_distribution.ipynb**  
  Jupyter notebook for analyzing the distribution of codes in generated neighborhoods and visualizing neighborhood statistics.

  **How to run:**  
  Open with Jupyter:
  ```sh
  jupyter notebook neigh_distribution.ipynb
  ```


- **script_jaccard_lof.py**  
  Script to automate the computation of Jaccard similarity and Local Outlier Factor (LOF) metrics across different neighborhood types and sizes.

  **How to run:**  
  ```sh
  python script_jaccard_lof.py
  ```

- **script_test_bilstm.py**  
  Script for testing the BiLSTM model on different neighborhood types and sizes.

  **How to run:**  
  ```sh
  python script_test_bilstm.py
  ```

- **script_test_setor.py**  
  Script for testing the SETOR model on different neighborhood types and sizes.

  **How to run:**  
  ```sh
  python script_test_setor.py
  ```

- **single_explainer_BiLSTM.py**  
  Generates local explanations for BiLSTM model predictions using DoctorXAI pipeline.

  **How to run:**  
  ```sh
  python single_explainer_BiLSTM.py
  ```

- **single_explainer_SETOR.py**  
  Generates local explanations for SETOR model predictions using DoctorXAI pipeline.

  **How to run:**  
  ```sh
  python single_explainer_SETOR.py
  ```

- **statistics_utils.py**  
  Utility functions for statistical analysis, metric computation, and experiment configuration.  
  **How to use:**  
  Import as a module in other scripts.

- **lib/**  
  Contains low-level and performance-critical code, including:
  - `generator.so`: Compiled shared library for ontological synthetic data generation.
  - `main.zig`, `numpy_data.zig`, `out.zig`, `python.zig`, `tensor.zig`: Zig source files for custom data processing and integration with Python.
  - `Makefile`: Build instructions for the Zig code.

  **How to build:**  
    Instructions for compiling the library on Linux:

    1. Download version 0.11 of the Zig compiler [from the official website](https://ziglang.org/download/#release-0.11.0).
    2. Extract the downloaded package.
    3. Add the `zig` executable to your `PATH` environment variable.
    4. Change your current working directory to `lib/`:
        ```sh
        cd lib/
        ```
    5. Run `make` to build the library:
        ```sh
        make
        ```
    6. After compilation, the file `generator.so` should appear in the `lib/` directory.

## Typical Workflow

1. **Data Preparation**:  
   Use `convert.py` to preprocess and convert raw data into parquet files.

2. **Explainability**:  
   Use `single_explainer_BiLSTM.py` or `single_explainer_SETOR.py` to generate explanations for model predictions, with respective metrics.

3. **Statistical Analysis & Tests**:  
   Use `script_test_setor.py`, `script_test_bilstm.py`, and `neigh_distribution.ipynb` for further explainable metrics analysis and visualization.

4. **Low-level Optimization**:  
   The `lib/` folder provides optimized routines for heavy computations, compiled via the included Makefile.
