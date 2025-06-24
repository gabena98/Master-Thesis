# BiLSTM

This folder contains the core code for training and evaluating a bidirectional long short-term memory (BiLSTM) model for predicting sequences of Electronic Health Records (EHRs).

## Contents

- **BiLSTM_model.py**  
  Implementation of the BiLSTM neural network.  
  Includes the model architecture, configuration dataclass, and utility functions for inference and batch preparation.

- **BiLSTM_trainer.py**  
  Training and evaluation pipeline for the BiLSTM model.  
  Handles data loading, training loop, evaluation, early stopping, logging of metrics, and saving model checkpoints.

- **BiLSTM_config.toml**  
  Configuration file for model and training parameters.  
  Specifies hyperparameters such as hidden size, number of layers, dropout, batch size, learning rate, and metrics to compute.

## How to Use

1. **Configure the Model**  
   Edit `BiLSTM_config.toml` to set the desired model and training hyperparameters, including the path to your processed dataset.

2. **Train the Model**  
   Run the training script:
   ```sh
   python BiLSTM_trainer.py
   ```
   The script will train the BiLSTM model, evaluate it, and save logs and model checkpoints in the specified results directory.

3. **Model Inference**  
   Use the functions in `BiLSTM_model.py` to load a trained model and perform inference on new data.

## Notes

- Make sure the `diagnoses_path` in `BiLSTM_config.toml` points to a valid path.
- Training and evaluation logs are saved in the results directory specified by `save_directory` in the config file.

