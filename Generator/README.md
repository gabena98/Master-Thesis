# Generator

This folder contains code and configuration for training and using a Large Language Model (based on Llama2 architectures) to create synthetic patient data for medical sequence modeling.

## Contents

- **generator.py**  
  Main script for training a generative model on patient diagnosis sequences.  
  Handles data loading, model training, evaluation, logging, and saving model checkpoints and configuration.

  **How to run:**  
  ```sh
  python generator.py
  ```
  Make sure to adjust the configuration file as needed before running.

- **llama2_model.py**  
  Implementation of the Llama2-based transformer model, including model classes, configuration dataclasses, and utility functions for batch preparation, inference, and generation.

  - Provides:  
    - `llama2_Filler` class  
    - Model loading utilities (`load_llama2_for_inference`, `load_llama2_for_generation`)  
    - Batch preparation functions (`prepare_batch_for_inference`, `prepare_batch_for_generation`)

- **generator_config.toml**  
  TOML configuration file specifying model hyperparameters, training parameters, and data paths.

  - Edit this file to set:
    - Path to the processed diagnoses data (`diagnoses_path`)
    - Model architecture parameters (hidden size, layers, heads, etc.)
    - Training parameters (batch size, learning rate, epochs, splits, etc.)
    - Device selection and save directory

- **old_generator.py**  
  Legacy code for generating synthetic patient data.

  **How to run:**  
  ```sh
  python old_generator.py
  ```


## Typical Workflow

1. **Configure the Model and Training**  
   Edit `generator_config.toml` to set the data path, model hyperparameters, and training options.

2. **Train the Generator**  
   Run the main script:
   ```sh
   python generator.py
   ```
   The script will train the model, evaluate it, and save logs, checkpoints, and updated configuration in the specified results directory.

3. **Use the Model for Generation**  
   Use the functions in `llama2_model.py` to load a trained model and generate synthetic sequences.

## Notes

- The generator is designed for research on synthetic data generation and perturbation in medical sequence datasets.
- Training and evaluation logs, as well as model checkpoints and configuration snapshots, are saved in the results directory specified in the config file.
- The code supports both CPU and GPU execution (set the `device` parameter in the config).
