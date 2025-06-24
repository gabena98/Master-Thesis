import math
import os
from dataclasses import dataclass

import polars as pl
import numpy as np
import random

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
import tomlkit
from datetime import datetime

from Generator.llama2_model import llama2_Config, llama2_Filler, CFG_FILE_NAME, MODEL_FILE_NAME

CONFIG_FILE_PATH = 'filler_config.toml'

LOG_FILE_NAME = 'log.txt'
CSV_FILE_NAME = 'log.csv'


@dataclass(kw_only=True)
class Trainer_Config:
    hole_token_id: int
    hole_prob: float
    masked_loss_gamma: float
    num_test_rounds: int

    batch_size:        int
    num_epochs:        int
    learning_rate:     float
    eval_split:        float
    test_split:        float
    weight_decay:      float
    eval_batch_size:   int
    save_directory:    str
    patience:          int
    max_patient_len:   int # @todo
    limit_num_batches: int | None = None


def nice_print(txt: str):
    """
    Prints the given text using tqdm's write method to ensure compatibility with tqdm progress bars.

    Args:
        txt (str): The text string to be printed.
    """
    tqdm.write(txt)

@dataclass
class Epoch_Metrics:
    epoch: int
    learn_rate: float
    train_loss: float
    eval_loss: float
    eval_masked_loss: float
    eval_mixed_loss: float
    eval_masked_accuracy: float
    eval_accuracy: float

def log_metrics(metrics: Epoch_Metrics, config: Trainer_Config):
    """
    Logs training and evaluation metrics to both a human-readable text file and a CSV file.

    Args:
        metrics (Epoch_Metrics): An object containing metrics for the current epoch, such as losses and accuracies.
        config (Trainer_Config): Configuration object containing training parameters, including the save directory.

    Side Effects:
        - Appends a formatted string of metrics to a log text file in the specified save directory.
        - Appends a row of metrics to a CSV file in the specified save directory. If it is the first epoch, writes the CSV header.
        - Raises an Error if the CSV file already exists at epoch zero.

    Notes:
        - The function assumes the existence of global constants LOG_FILE_NAME and CSV_FILE_NAME.
        - Uses a helper function `nice_print` to print the metrics to the console.
    """
    txt_file_path = os.path.join(trainer_config.save_directory, LOG_FILE_NAME)
    csv_file_path = os.path.join(trainer_config.save_directory, CSV_FILE_NAME)

    elems = [
        f'{metrics.epoch: >3}.',
        f'train_loss: {metrics.train_loss:.3f}',
        f'eval_loss: {metrics.eval_mixed_loss:.3f}',
        f'masked_loss: {metrics.eval_masked_loss:.3f}',
        f'accuracy: {metrics.eval_accuracy:.3f}',
        f'masked_accuracy: {metrics.eval_masked_accuracy:.3f}',
    ]
    txt = '   '.join(elems)

    nice_print(txt)
    with open(txt_file_path, 'a') as f:
        f.write(txt + '\n')

    if metrics.epoch == 0:
        if os.path.exists(csv_file_path):
            raise Error(f'We are in epoch zero, but csv file {csv_file_path} already exists')
        txt  = 'epoch,learn_rate,train_loss,eval_loss,eval_masked_loss,eval_mixed_loss,accuracy,masked_accuracy'
        txt += '\n'
    else:
        txt = ''

    elems = [
        f'{metrics.epoch}.',
        f'{metrics.learn_rate:.3e}',
        f'{metrics.train_loss:.6f}',
        f'{metrics.eval_loss:.6f}',
        f'{metrics.eval_masked_loss:.6f}',
        f'{metrics.eval_mixed_loss:.6f}',
        f'{metrics.eval_accuracy:.6f}',
        f'{metrics.eval_masked_accuracy:.6f}',
    ]
    txt += ','.join(elems)
    txt += '\n'

    with open(csv_file_path, 'a') as f:
        f.write(txt)
    pass

@dataclass(kw_only=True)
class Early_Stopper_Result:
    """
    Represents the result of an early stopping check during training.

    Attributes:
        is_best_round (bool): Indicates if the current round is the best so far.
        should_exit (bool): Indicates if the training process should be stopped early.
    """
    is_best_round:  bool
    should_exit: bool

class Early_Stopper:
    """
    Early_Stopper is a utility class for implementing early stopping in training loops.

    Attributes:
        best_result (float): The best metric value observed so far.
        best_epoch (int | None): The epoch at which the best metric was observed.
        patience (int): Number of epochs to wait for improvement before stopping.
        min_is_better (bool): If True, lower metric values are considered better (e.g., loss). If False, higher values are better (e.g., accuracy).

    Methods:
        __init__(patience: int, *, min_is_better: bool):
            Initializes the Early_Stopper with the specified patience and optimization direction.

        check(epoch: int, metric: float) -> Early_Stopper_Result:
            Checks if the current metric is an improvement. Returns an Early_Stopper_Result indicating if this is the best round and whether training should stop.
    """
    best_result: float
    best_epoch:  int | None
    patience:    int
    min_is_better: bool

    def __init__(self, patience: int, *, min_is_better: bool):
        self.best_result = 0.0
        self.best_epoch  = 0
        self.patience    = patience
        self.min_is_better = min_is_better

    def check(self, epoch: int, metric: float) -> Early_Stopper_Result:
        check = self.best_epoch is None or metric < self.best_result
        if check == self.min_is_better:
            # then we are doing good
            self.best_epoch  = epoch
            self.best_result = metric
            is_best_round  = True
            should_exit = False
        else:
            is_best_round  = False
            should_exit = epoch - self.best_epoch > self.patience
        return Early_Stopper_Result(is_best_round=is_best_round, should_exit=should_exit)

@dataclass(kw_only=True)
class Training_Results:
    """
    A class to store and represent the results of a training process.

    Attributes:
        num_epochs (int): The total number of epochs the model was trained for.
        best_epoch (int): The epoch number at which the best evaluation metric was achieved.
        train_loss (float): The training loss at the end of training.
        eval_loss (float): The evaluation loss at the end of training.
        eval_masked_loss (float): The evaluation loss computed on masked data (if applicable).
        eval_masked_accuracy (float): The evaluation accuracy computed on masked data (if applicable).
        eval_accuracy (float): The overall evaluation accuracy at the end of training.
    """
    num_epochs:           int
    best_epoch:           int
    train_loss:           float
    eval_loss:            float
    eval_masked_loss:     float
    eval_masked_accuracy: float
    eval_accuracy:        float
    

def train(model: nn.Module, diagnoses: pl.DataFrame, trainer_config: Trainer_Config) -> Training_Results:
    """
    Trains the given model on diagnosis data using the specified training configuration.
    Args:
        model (nn.Module): The PyTorch model to be trained.
        diagnoses (pl.DataFrame): A Polars DataFrame containing diagnosis data, with columns including 'role', 'out_id', 'icd9_id', 'position', and 'count'.
        trainer_config (Trainer_Config): Configuration object containing training hyperparameters such as batch size, learning rate, patience, save directory, and other options.
    Returns:
        Training_Results: An object containing training statistics and evaluation metrics, including the number of epochs, best epoch, training loss, evaluation loss, masked loss, masked accuracy, and overall accuracy.
    Notes:
        - The function splits the diagnosis data into training, evaluation, and test sets based on the 'role' column.
        - Implements early stopping based on evaluation metrics.
        - Saves the best model checkpoint during training.
        - Evaluates the model on the test set after training and returns the results.
    """
    def split(data):
        out_codes   = list(data['out_id']  .to_numpy())
        input_codes = list(data['icd9_id'] .to_numpy())
        positions   = list(data['position'].to_numpy())
        counts      = list(data['count']   .to_numpy())
        return out_codes, input_codes, counts, positions

    out_train, input_train, counts_train, positions_train = split(diagnoses.filter(pl.col('role') == 'train'))
    out_eval,  input_eval,  counts_eval,  positions_eval  = split(diagnoses.filter(pl.col('role') == 'eval'))
    out_test,  input_test,  counts_test,  positions_test  = split(diagnoses.filter(pl.col('role') == 'test'))

    # find the number of batches
    num_batches = len(out_train) // trainer_config.batch_size
    if trainer_config.limit_num_batches is not None:
        l_num_batches = min(num_batches, trainer_config.limit_num_batches)
        nice_print(
            f'dataset would contain {num_batches} batches of {trainer_config.batch_size}'
            f' but we are limiting it to {l_num_batches}'
        )
        num_batches = l_num_batches

    model_save_path = os.path.join(trainer_config.save_directory, MODEL_FILE_NAME)

    #configure the optimizer
    optimizer = torch.optim.AdamW (
        model.parameters(),
        trainer_config.learning_rate,
        weight_decay = trainer_config.weight_decay,
    )

    stopper = Early_Stopper(trainer_config.patience, min_is_better=False) 

    best_epoch = -1
    best_train_loss = -1

    # Train Loop

    try:
        for epoch in tqdm(range(trainer_config.num_epochs), 'epoch', leave=False):
            total_train_loss   = 0
            train_loss_divisor = 0

            for batch_id in tqdm(range(num_batches), 'train', leave=False):
                batch_start = batch_id * trainer_config.batch_size
                batch_end   = batch_start + trainer_config.batch_size

                # get the right data for the batch
                i_out       = out_train       [batch_start: batch_end]
                i_input     = input_train     [batch_start: batch_end]
                i_positions = positions_train [batch_start: batch_end]
                i_counts    = counts_train    [batch_start: batch_end]

                b_codes, b_positions, b_lengths, outputs = prepare_data (
                    i_input, i_input, i_positions, trainer_config, model.config.device
                )


                # feed-forward + backpropagation
                optimizer.zero_grad()
                model.train()
                # predictions shape is (batch_size (32), max_len (e.g. 92), vocab_size(500))
                predictions = model(b_codes, b_positions, b_lengths)
                mask = b_codes == trainer_config.hole_token_id
                total_loss, masked_loss = compute_loss(predictions, outputs, mask)

                loss = total_loss + trainer_config.masked_loss_gamma * masked_loss
                
                loss.backward()

                optimizer.step()

                total_train_loss   += float(loss)
                train_loss_divisor += 1

            train_loss = total_train_loss / train_loss_divisor
            metrics_results = evaluate (
                model,
                input_eval,
                input_eval,
                positions_eval,
                counts_eval,
                trainer_config,
            )
            
            metrics = Epoch_Metrics (
                epoch      = epoch,
                learn_rate = float(optimizer.param_groups[0]['lr']),
                train_loss = train_loss,
                eval_loss  = metrics_results.loss,
                eval_mixed_loss      = metrics_results.mixed_loss,
                eval_masked_loss     = metrics_results.masked_loss,
                eval_masked_accuracy = metrics_results.masked_accuracy,
                eval_accuracy        = metrics_results.accuracy
            )
            log_metrics(metrics, trainer_config)

            # The `min_is_better` field is relevant in constructor!!
            stopper_result = stopper.check(epoch, metrics.eval_masked_accuracy + metrics.eval_accuracy * 10)

            if stopper_result.is_best_round:
                torch.save(model.state_dict(), model_save_path)
                best_epoch = epoch
                best_train_loss = train_loss

            if stopper_result.should_exit:
                nice_print('It seems we are done here...')
                break
    except KeyboardInterrupt:
        nice_print('exiting loop...')


    model.load_state_dict(torch.load(model_save_path))
    metrics = evaluate (
        model,
        input_test,
        input_test,
        positions_test,
        counts_test,
        trainer_config,
        num_rounds = trainer_config.num_test_rounds,
    )
    training_results = Training_Results (
        num_epochs = epoch,
        best_epoch = best_epoch,
        train_loss = best_train_loss,
        eval_loss  = metrics.loss,
        eval_masked_loss = metrics.masked_loss,
        eval_masked_accuracy = metrics.masked_accuracy,
        eval_accuracy =  metrics.accuracy
    )
    return training_results

def evaluate (
    model: nn.Module,
    out_codes: list[np.ndarray],
    icd_codes: list[np.ndarray],
    positions: list[np.ndarray],
    counts:    list[np.ndarray],
    config:    Trainer_Config,
    *,
    num_rounds: int = 1,
):
    metrics_results_acc = Metrics_Results_Partial (0, 0, 0, 0, 0, 0)

    num_batches = len(out_codes) // config.eval_batch_size
    for _ in range(num_rounds):
        for batch_id in range(num_batches):
            batch_start = batch_id * config.eval_batch_size
            batch_end   = batch_start + config.eval_batch_size

            # get the right data for the batch
            i_icd       = icd_codes [batch_start: batch_end]
            i_out       = out_codes [batch_start: batch_end]
            i_positions = positions [batch_start: batch_end]
            i_counts    = counts    [batch_start: batch_end]

            b_codes, b_positions, b_lengths, outputs = prepare_data (
                i_icd, i_icd, i_positions, trainer_config, model.config.device
            )

            mask = b_codes == config.hole_token_id

            # computations
            model.eval()
            with torch.inference_mode():
                predictions = model(b_codes, b_positions, b_lengths)
                m = compute_metrics(predictions, outputs, mask)
            
            metrics_results_acc.total_loss            += m.total_loss
            metrics_results_acc.total_masked_loss     += m.total_masked_loss
            metrics_results_acc.total_accuracy        += m.total_accuracy
            metrics_results_acc.total_masked_accuracy += m.total_masked_accuracy
            metrics_results_acc.divisor               += m.divisor
            metrics_results_acc.masked_divisor        += m.masked_divisor

    mra = metrics_results_acc
    loss        = mra.total_loss        / mra.divisor
    masked_loss = mra.total_masked_loss / mra.masked_divisor if mra.masked_divisor > 0 else 0.0
    return Metrics_Results (
        loss            = loss,
        masked_loss     = masked_loss,
        mixed_loss      = loss + config.masked_loss_gamma * masked_loss,
        accuracy        = mra.total_accuracy        / mra.divisor,
        masked_accuracy = mra.total_masked_accuracy / mra.masked_divisor if mra.masked_divisor > 0 else 0.0,
    )

@dataclass
class Metrics_Results:
    loss:            float
    masked_loss:     float
    accuracy:        float
    masked_accuracy: float
    mixed_loss:      float

@dataclass
class Metrics_Results_Partial:
    total_loss:            float
    total_masked_loss:     float
    total_accuracy:        float
    total_masked_accuracy: float
    divisor:               int
    masked_divisor:        int

def compute_metrics (
    
    predictions: list[torch.Tensor],
    targets:     list[torch.Tensor],
    mask:        list[torch.Tensor],
) -> Metrics_Results_Partial:
    """
    Computes loss and accuracy metrics for model predictions with optional masking.
    Args:
        predictions (list[torch.Tensor]): List of model output tensors (logits) for each sample.
        targets (list[torch.Tensor]): List of ground truth target tensors for each sample.
        mask (list[torch.Tensor]): List of boolean tensors indicating which elements to include in masked metrics.
    Returns:
        Metrics_Results_Partial: An object containing:
            - total_loss (float): Sum of cross-entropy losses over all valid targets (targets != -1).
            - total_masked_loss (float): Sum of cross-entropy losses over masked elements.
            - total_accuracy (float): Sum of correct prediction probabilities over all valid targets.
            - total_masked_accuracy (float): Sum of correct prediction probabilities over masked elements.
            - divisor (int): Number of valid targets (targets != -1).
            - masked_divisor (int): Number of masked elements (mask.sum()).
    Notes:
        - The function ignores targets with value -1 when computing unmasked metrics.
        - Masked metrics are computed only over elements where mask is True.
        - Predictions are assumed to be raw logits and are softmaxed before accuracy computation.
    """
    predictions = predictions.reshape((-1, predictions.shape[-1]))
    targets     = targets.flatten()
    mask        = mask.flatten()

    cross_entropy = F.cross_entropy(predictions, targets, ignore_index=-1, reduction='none')
    total_loss  = cross_entropy[targets != -1].sum()
    masked_loss = cross_entropy[mask].sum()

    divisor = (targets != -1).sum()
    masked_divisor = mask.sum()

    predictions = F.softmax(predictions, dim=-1)
    correct_predictions = predictions.gather(1, targets.clamp(0, None).unsqueeze(1))

    total_accuracy  = correct_predictions[targets != -1].sum()
    masked_accuracy = correct_predictions[mask].sum()

    return Metrics_Results_Partial (
        total_loss            = float(total_loss),
        total_masked_loss     = float(masked_loss),
        total_accuracy        = float(total_accuracy),
        total_masked_accuracy = float(masked_accuracy),
        divisor               = int(divisor),
        masked_divisor        = int(masked_divisor),
    )

def compute_loss(predictions, targets, mask):
    """
    Computes the total and masked cross-entropy loss between predictions and targets.

    Args:
        predictions (torch.Tensor): The predicted logits of shape (batch_size, ..., num_classes).
        targets (torch.Tensor): The ground truth target indices of shape matching predictions (excluding num_classes).
        mask (torch.Tensor): A boolean or binary mask tensor of the same shape as targets, indicating which elements to include in the masked loss.

    Returns:
        tuple: A tuple (total_loss, masked_loss) where:
            - total_loss (torch.Tensor): The average cross-entropy loss over all valid (targets != -1) elements.
            - masked_loss (torch.Tensor): The average cross-entropy loss over elements where mask is True.

    Notes:
        - The function ignores targets with value -1 when computing the total loss.
        - The masked loss is computed only over elements where mask is True.
    """
    predictions = predictions.reshape((-1, predictions.shape[-1]))
    targets = targets.flatten()
    mask = mask.flatten()

    cross_entropy = F.cross_entropy(predictions, targets, ignore_index=-1, reduction='none')

    divisor = (targets != -1).sum()
    masked_divisor = mask.sum()

    total_loss  = cross_entropy[targets != -1].sum() / divisor
    masked_loss = cross_entropy[mask].sum() / masked_divisor

    return total_loss, masked_loss

def prepare_data(b_out, b_input, b_positions, trainer_config: Trainer_Config, device):
    """
    Prepares and pads input, output, and position batches for training, applies random masking, and moves tensors to the specified device.

    Args:
        b_out (list of np.ndarray): Batch of output sequences.
        b_input (list of np.ndarray): Batch of input sequences.
        b_positions (list of np.ndarray): Batch of position sequences.
        trainer_config (Trainer_Config): Configuration object containing training parameters such as hole probability and hole token ID.
        device (torch.device): Device to move the tensors to (e.g., 'cpu' or 'cuda').

    Returns:
        tuple:
            b_input (torch.LongTensor): Padded and masked input batch tensor on the specified device.
            b_positions (torch.LongTensor): Padded position batch tensor on the specified device.
            b_lengths (torch.LongTensor): Original lengths of each input sequence on the specified device.
            b_out (torch.LongTensor): Padded output batch tensor on the specified device.
    """
    lengths = [len(x) for x in b_input]
    b_n     = max(lengths)
    b_input     = np.array([np.pad(x, (0, b_n - len(x)), constant_values=0 ) for x in b_input])
    b_out       = np.array([np.pad(x.astype(int), (0, b_n - len(x)), constant_values=-1) for x in b_out])
    b_positions = np.array([np.pad(x.astype(int), (0, b_n - len(x)), constant_values=-1) for x in b_positions])


    # tweak input with holes
    mask = np.random.rand(*b_input.shape) < trainer_config.hole_prob
    b_input[mask] = trainer_config.hole_token_id
    b_out       = torch.from_numpy(b_out)
    b_positions = torch.from_numpy(b_positions)
    b_input     = torch.from_numpy(b_input.astype(np.int64))
    b_lengths   = torch.LongTensor(lengths)

    b_positions = b_positions.to(device)
    b_input     = b_input.to(device)
    b_lengths   = b_lengths.to(device)
    b_out       = b_out.to(device)

    return b_input, b_positions, b_lengths, b_out

DIR_ID_LENGTH = 5
ALL_LETTERS   = 'abcdefghijklmnopqrstuvxywz'
def format_path(path: str) -> str:
    """
    Formats the given file path by replacing placeholders with dynamic values.

    Replaces the '%(id)' placeholder with a randomly generated string of length DIR_ID_LENGTH,
    and the '%(date)' placeholder with the current date and time in the format 'YYYY-MM-DD_HH:MM:SS'.

    Args:
        path (str): The file path template containing '%(id)' and/or '%(date)' placeholders.

    Returns:
        str: The formatted file path with placeholders replaced by their respective values.
    """
    date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    id = ''.join([random.choice(ALL_LETTERS) for _ in range(DIR_ID_LENGTH)])
    path = path.replace('%(id)',   id)
    path = path.replace('%(date)', date)
    return path


if __name__ == '__main__':
    nice_print(f'Using config file: {CONFIG_FILE_PATH}')
    with open(CONFIG_FILE_PATH, 'r') as f:
        txt = f.read()
    config = tomlkit.parse(txt)

    diagnoses = pl.read_parquet(config['diagnoses_path'])

    config['model']['vocab_size']  = diagnoses['icd9_id'].list.max().max() + 2 # this is for the hole code
    config['model']['output_size'] = diagnoses['icd9_id'] .list.max().max() + 1
    config['trainer']['hole_token_id'] = config['model']['vocab_size'] - 1
    llama2_config = llama2_Config(**config['model'])
    llama2_config.device = torch.device("cuda:" + llama2_config.device if torch.cuda.is_available() else "mps")
    model = llama2_Filler(llama2_config)

    num_params  = sum([param.nelement()                      for param in model.parameters()])
    size_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    nice_print(f'model has {num_params/1e6:.2f}M params, occupying {size_params/1024/1024:.2f}M of memory')

    trainer_config = Trainer_Config(**config['trainer'])

    trainer_config.save_directory = format_path(trainer_config.save_directory) 
    nice_print(f'> Save directory is {trainer_config.save_directory}')
    os.makedirs(trainer_config.save_directory)

    num_patients_before = diagnoses.shape[0]
    diagnoses = diagnoses.filter(pl.col('count').list.sum() < trainer_config.max_patient_len)
    num_patients_after = diagnoses.shape[0]
    nice_print (
        f'Original dataset contains {num_patients_before} patients. We cut '
        f'those with more than {trainer_config.max_patient_len} codes, so we are '
        f'left with {num_patients_after} patients ' 
        f'({(1 - num_patients_after / num_patients_before)*100:.1f}% less).'
    )

    training_infos = tomlkit.table()
    training_infos['num_patients'] = num_patients_after
    training_infos['num_parameters'] = num_params
    config['training_infos'] = training_infos

    # update toml document
    new_config_text = tomlkit.dumps(config)
    new_config_path = os.path.join(trainer_config.save_directory, CFG_FILE_NAME)
    with open(new_config_path, 'w') as f:
        f.write(new_config_text)

    starting_time = datetime.now()

    results = train (
        model          = model,
        diagnoses      = diagnoses,
        trainer_config = trainer_config,
    )

    # compute training time
    end_time = datetime.now()
    time_delta = end_time - starting_time
    config['training_infos']['training_time'] = str(time_delta)

    # add test data to toml document
    test_results = tomlkit.table()
    test_results['training_epochs']      = results.num_epochs
    test_results['best_epoch']           = results.best_epoch
    test_results['train_loss']           = results.train_loss
    test_results['eval_loss']            = results.eval_loss
    test_results['eval_masked_loss']     = results.eval_masked_loss
    test_results['eval_masked_accuracy'] = results.eval_masked_accuracy
    test_results['eval_accuracy']        = results.eval_accuracy

    # save new results on config file
    new_config_text = tomlkit.dumps(config)
    with open(new_config_path, 'w') as f:
        f.write(new_config_text)

    # print the test results on screen
    elems = [
        f'best_epoch: {results.best_epoch}',
        f'train_loss: {results.train_loss:.3f}',
        f'eval_loss: {results.eval_loss:.3f}',
        f'masked_loss: {results.eval_masked_loss:.3f}',
        f'accuracy: {results.eval_accuracy:.3f}',
        f'masked_accuracy: {results.eval_masked_accuracy:.3f}',
    ]
    txt = '   '.join(elems)
    nice_print(txt)

