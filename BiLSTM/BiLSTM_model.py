from dataclasses import dataclass
import random
import pandas as pd
import numpy as np
import tomlkit
import torch
#from spacy import spacy
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')
from collections import Counter

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import os 

CFG_FILE_NAME = 'BiLSTM_config.toml' # this is the one that is reported in the save dir
MODEL_FILE_NAME = 'model.torch'
SEED = 42

def seed_everything(seed=10):
    """
    Sets the seed for various random number generators to ensure reproducibility.

    This function sets the seed for Python's built-in `random` module, NumPy, and PyTorch (both CPU and CUDA).
    It also sets the `PYTHONHASHSEED` environment variable and configures PyTorch to use deterministic algorithms.

    Args:
        seed (int, optional): The seed value to use. Defaults to 10.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)

@dataclass(kw_only=True)
class BiLSTM_Config:
    max_features: int = 9481
    embed_size: int = 256
    out_dim: int = 281
    num_layers: int = 2
    hidden_size: int = 256
    batch_size: int = 64
    epochs: int = 1000
    learning_rate: float = 0.001
    dropout: float = 0.2
    weight_decay: float = 1e-5
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class BiLSTM(nn.Module):
    """
    BiLSTM model for sequential data with hierarchical structure (e.g., sequences of visits, each with its own sequence).
    Args:
        config (BiLSTM_Config): Configuration object containing model hyperparameters:
            - max_features (int): Size of the vocabulary for the embedding layer.
            - embed_size (int): Dimension of the embedding vectors.
            - out_dim (int): Output dimension of the final linear layer.
            - num_layers (int): Number of LSTM layers in each BiLSTM block.
            - hidden_size (int): Number of features in the hidden state of the LSTM.
            - batch_size (int): Batch size for training.
            - epochs (int): Number of training epochs.
            - learning_rate (float): Learning rate for the optimizer.
            - dropout (float): Dropout rate for LSTM layers.
            - weight_decay (float): Weight decay (L2 regularization) for the optimizer.
            - device (torch.device): Device to run the model on ('cpu' or 'cuda').
    Attributes:
        embedding (nn.Embedding): Embedding layer for input tokens.
        lstm1 (nn.LSTM): First bidirectional LSTM for processing sequences within each visit.
        lstm2 (nn.LSTM): Second bidirectional LSTM for processing sequences of visit representations.
        fc (nn.Linear): Fully connected layer projecting LSTM outputs to the desired output dimension.
    Forward Input:
        batch (torch.Tensor): Input tensor of shape (batch_size, num_visit, seq_len), where
            - batch_size: Number of samples in the batch.
            - num_visit: Number of visits per sample.
            - seq_len: Length of each visit sequence.
    Forward Output:
        output (torch.Tensor): Output tensor of shape (batch_size, num_visit, out_dim), containing the model predictions for each visit.
    Workflow:
        1. Embeds input tokens.
        2. Processes each visit sequence independently with the first BiLSTM.
        3. Applies mean pooling over time steps to obtain visit representations.
        4. Processes the sequence of visit representations with the second BiLSTM.
        5. Projects the output to the desired output dimension.
    """
    
    def __init__(self, config: BiLSTM_Config):
        super(BiLSTM, self).__init__()
        self.config = config
        self.input_dim = config.max_features
        self.embedding_dim = config.embed_size
        self.output_dim = config.out_dim
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.learning_rate = config.learning_rate
        self.dropout = config.dropout
        self.weight_decay = config.weight_decay
        self.device = config.device
        
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim, padding_idx=0).to(self.device)
        self.lstm1 = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True, dropout=self.dropout).to(self.device)
        self.lstm2 = nn.LSTM(self.hidden_size * 2, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True, dropout=self.dropout).to(self.device)
        self.fc = nn.Linear(self.hidden_size * 2, self.output_dim).to(self.device)  # *2 because it's bidirectional
        
    def forward(self, batch: torch.Tensor):
        batch_size, num_visit, seq_len = batch.shape

        embedded = self.embedding(batch)  # embedding has size = (batch_size, num_visit, seq_len, embedding_dim)

        # Reshape for the first BiLSTM (independent processing of each visit)
        lstm_input = embedded.view(batch_size * num_visit, seq_len, self.embedding_dim)

        # BiLSTM on the sequences within each visit
        lstm_out1, _ = self.lstm1(lstm_input)  # Output: (batch_size * num_visit, seq_len, hidden_size*2)

        # Mean Pooling over the time-steps of each visit to obtain a compact representation
        visit_representation = torch.mean(lstm_out1, dim=1)  # (batch_size * num_visit, hidden_size*2)

        # Restore the original structure (batch_size, num_visit, hidden_size*2)
        visit_representation = visit_representation.view(batch_size, num_visit, self.hidden_size * 2)

        # Second BiLSTM to capture dependencies between visits
        lstm_out2, _ = self.lstm2(visit_representation)  # (batch_size, num_visit, hidden_size*2)

        # Projection into the out_dim space
        output = self.fc(lstm_out2)  # (batch_size, num_visit, out_dim)

        return output
    
def load_bilstm_for_inference(path: str, device: str) -> BiLSTM:
    """
    Loads a pre-trained BiLSTM model for inference from the specified directory.

    This function reads the model configuration and weights from the given path,
    initializes the BiLSTM model, loads its state dictionary, and moves it to the
    specified device.

    Args:
        path (str): The directory containing the model configuration and weights files.
        device (str): The device identifier (e.g., 'cpu' or 'cuda:0') to which the model should be moved.

    Returns:
        BiLSTM: The loaded BiLSTM model ready for inference on the specified device.
    """
    config_path = os.path.join(path, CFG_FILE_NAME)
    model_path = os.path.join(path, MODEL_FILE_NAME)

    with open(config_path, 'r') as f:
        txt = f.read()
    config = tomlkit.parse(txt)['model']
    config['device'] = str(device)  # Update the device in the config
    config = BiLSTM_Config(**config)

    model = BiLSTM(config)  # Do NOT .to(device) yet
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)  # Now move the model
    return model

@dataclass
class Inference_Batch:
    codes: torch.Tensor

    def unpack(self) -> torch.Tensor:
        self.codes = self.codes
        return self.codes

def prepare_batch_for_inference_bilstm (
    codes:     list[np.ndarray],
    counts:    list[np.ndarray],
    positions: list[np.ndarray],
    device:    torch.device,
) -> Inference_Batch:
    """
    Prepares a batch of input data for inference with a BiLSTM model by grouping, padding, and converting input arrays.
    Args:
        codes (list[np.ndarray]): List of arrays containing code indices for each visit.
        counts (list[np.ndarray]): List of arrays containing counts for each code (unused in this function).
        positions (list[np.ndarray]): List of arrays indicating the position of each code in the sequence.
        device (torch.device): The device (CPU or GPU) to which the resulting tensors will be moved.
    Returns:
        Inference_Batch: An object containing the padded and batched input codes tensor, ready for inference.
    Notes:
        - Each visit's codes are grouped by their position, and each group is padded to the maximum group length in the batch.
        - The resulting tensor is moved to the specified device and has dtype torch.int32.
        - The function assumes that the Inference_Batch class is defined elsewhere and expects a 'codes' attribute.
    """
    input_data = []
    for visit, position in zip(codes, positions):
        new_input = [visit[position == pos].tolist() for pos in np.unique(position)]
        input_data.append(new_input)
    
    # Function to apply padding
    def pad_input(data, max_length):
        return [sublist + [0] * (max_length - len(sublist)) for sublist in data]

    # Find the maximum length of each sublist
    max_length = max([max(len(sublist) for sublist in row) for row in input_data])

    # Apply padding to each sublist
    padded_input_data = []
    for row in input_data:
        padded_row = torch.tensor(pad_input(row, max_length), dtype=torch.int32, device=device)
        padded_input_data.append(padded_row)
    padded_input_data = pad_sequence(padded_input_data, batch_first=True, padding_value=0)

    return Inference_Batch (
        codes = padded_input_data.to(device=device, dtype=torch.int32)
    )
