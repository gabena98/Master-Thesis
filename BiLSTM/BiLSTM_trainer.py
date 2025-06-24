import math
import os
from dataclasses import dataclass

import polars as pl
import numpy as np
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torcheval.metrics import TopKMultilabelAccuracy


from tqdm import tqdm
import tomlkit
from datetime import datetime

from BiLSTM.BiLSTM_model import BiLSTM, BiLSTM_Config, CFG_FILE_NAME, MODEL_FILE_NAME

LOG_FILE_NAME = 'log.txt'
CSV_FILE_NAME = 'log.csv'

def nice_print(txt: str):
    tqdm.write(txt)

@dataclass(kw_only=True)
class Metrics_Config:
    recalls: list[int]
    precisions: list[int]
    accuracies: list[int]


@dataclass(kw_only=True)
class Trainer_Config:
    batch_size:        int
    num_epochs:        int
    learning_rate:     float
    eval_split:        float
    test_split:        float
    weight_decay:      float
    eval_batch_size:   int
    ccs_as_inputs:     bool
    save_directory:    str
    patience:          int
    max_patient_len:   int # @todo
    limit_num_batches: int | None = None

    # auto-filled
    metrics_config: Metrics_Config | None = None

@dataclass(kw_only=True)
class Epoch_Metrics:
    epoch: int
    learn_rate: float
    train_loss: float
    eval_loss: float
    recalls: dict[int, float]
    precision: dict[int, float]
    accuracy: dict[int, float]
    

def log_metrics(metrics: Epoch_Metrics, trainer_config: Trainer_Config):
    """
    Logs training and evaluation metrics for each epoch to both a text file and a CSV file.

    The function prints a formatted summary of the current epoch's metrics, appends it to a log text file,
    and writes detailed metrics to a CSV file. On the first epoch, it creates the CSV file with a header.
    Raises an error if the CSV file already exists at epoch zero.

    Args:
        metrics (Epoch_Metrics): An object containing the metrics for the current epoch, including epoch number,
            training loss, evaluation loss, learning rate, and dictionaries for recall, precision, and accuracy.
        trainer_config (Trainer_Config): Configuration object containing the save directory and metrics configuration,
            including which recall metrics to log.

    Raises:
        Error: If the CSV file already exists when logging metrics for epoch zero.
    """
    txt_file_path = os.path.join(trainer_config.save_directory, LOG_FILE_NAME)
    csv_file_path = os.path.join(trainer_config.save_directory, CSV_FILE_NAME)

    txt  = f'{metrics.epoch: >3}. '
    txt += f'train_loss: {metrics.train_loss:.3f}'
    txt += f'   eval loss: {metrics.eval_loss:.3f}'
    for k in trainer_config.metrics_config.recalls:
        txt += f"    r{k}: {metrics.recalls[k]*100:.2f}%"
        txt += f"    p{k}: {metrics.precision[k]*100:.2f}%"
        txt += f"    a{k}: {metrics.accuracy[k]*100:.2f}%"
    txt += f"   lr:{metrics.learn_rate:.3e}"

    nice_print(txt)
    with open(txt_file_path, 'a') as f:
        f.write(txt + '\n')

    if metrics.epoch == 0:
        if os.path.exists(csv_file_path):
            raise Error(f'We are in epoch zero, but csv file {csv_file_path} already exists')
        txt  = 'epoch,learn_rate,train_loss,eval_loss,'
        txt += ','.join([f'recall_{k}' for k in sorted(metrics.recalls)])
        txt += ',' 
        txt += ','.join([f'precision_{k}' for k in sorted(metrics.precision)])
        txt += ','
        txt += ','.join([f'accuracy_{k}' for k in sorted(metrics.accuracy)])
        txt += '\n'
    else:
        txt = ''

    values = [
        metrics.epoch,
        metrics.learn_rate,
        metrics.train_loss,
        metrics.eval_loss,
    ]
    values += [metrics.recalls[k] for k in sorted(metrics.recalls)]
    values += [metrics.precision[k] for k in sorted(metrics.recalls)]
    values += [metrics.accuracy[k] for k in sorted(metrics.recalls)]
    # convert all values to string
    values  = [str(v) for v in values]
    txt += ','.join(values)
    txt += '\n'

    with open(csv_file_path, 'a') as f:
        f.write(txt)


@dataclass(kw_only=True)
class Early_Stopper_Result:
    is_best_round:  bool
    should_exit: bool

class Early_Stopper:
    best_result: float
    best_epoch:  int | None
    patience:    int
    min_is_better: bool

    def __init__(self, patience: int, *, min_is_better: bool):
        self.best_result = 0.0
        self.best_epoch  = None
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
    num_epochs: int
    loss: int
    recalls: dict[int, float]
    precision: dict[int, float]
    accuracy: dict[int, float]

def train(model: nn.Module, diagnoses: pl.DataFrame, trainer_config: Trainer_Config) -> Training_Results:
    def split(data):
        ccs_codes = list(data['ccs_id'].to_numpy())
        if trainer_config.ccs_as_inputs:
            input_codes = list(data['ccs_id'].to_numpy())
        else:
            input_codes = list(data['icd9_id'].to_numpy())
        positions = list(data['position'].to_numpy())
        counts    = list(data['count'].to_numpy())
        return ccs_codes, input_codes, counts, positions

    ccs_train, input_train, counts_train, positions_train = split(diagnoses.filter(pl.col('role') == 'train'))
    ccs_eval,  input_eval,  counts_eval,  positions_eval  = split(diagnoses.filter(pl.col('role') == 'eval'))
    ccs_test,  input_test,  counts_test,  positions_test  = split(diagnoses.filter(pl.col('role') == 'test'))

    # find the number of batches
    num_batches = len(ccs_train) // trainer_config.batch_size
    if trainer_config.limit_num_batches is not None:
        l_num_batches = min(num_batches, trainer_config.limit_num_batches)
        nice_print(
            f'dataset would contain {num_batches} batches of {trainer_config.batch_size}'
            f' but we are limiting it to {l_num_batches}'
        )
        num_batches = l_num_batches
    else:
        nice_print(f'dataset contains {num_batches} batches of {trainer_config.batch_size}')

    model_save_path = os.path.join(trainer_config.save_directory, MODEL_FILE_NAME)

    #configure the optimizer
    optimizer = torch.optim.AdamW (
        model.parameters(),
        trainer_config.learning_rate,
        weight_decay = trainer_config.weight_decay,
    )

    stopper = Early_Stopper(trainer_config.patience, min_is_better=True) 

    # Train Loop

    try:
        for epoch in tqdm(range(trainer_config.num_epochs), 'epoch', leave=False):
            total_train_loss   = 0
            train_loss_divisor = 0

            for batch_id in tqdm(range(num_batches), 'train', leave=False):
                batch_start = batch_id * trainer_config.batch_size
                batch_end   = batch_start + trainer_config.batch_size

                # get the right data for the batch
                i_ccs = ccs_train[batch_start: batch_end]
                i_input = input_train[batch_start: batch_end]
                i_positions = positions_train[batch_start: batch_end]
                i_counts = counts_train[batch_start: batch_end]

                b_codes, outputs = prepare_data (
                    i_ccs, i_input, i_positions, i_counts,
                )

                # feed-forward + backpropagation
                optimizer.zero_grad()
                model.train()
                predictions = model(b_codes)
                total_loss = compute_loss(predictions, outputs)
                divisor = sum([x.shape[0] for x in predictions])
                loss = total_loss / divisor
                loss.backward()

                optimizer.step()

                total_train_loss += float(total_loss)
                train_loss_divisor += divisor

            train_loss = total_train_loss / train_loss_divisor
            metrics_dict = evaluate (
                model,
                ccs_eval,
                input_eval,
                positions_eval,
                counts_eval,
                trainer_config,
            )
            
            metrics = Epoch_Metrics (
                epoch      = epoch,
                learn_rate = float(optimizer.param_groups[0]['lr']),
                train_loss = train_loss,
                eval_loss  = metrics_dict['loss'],
                recalls    = metrics_dict['recalls'],
                precision  = metrics_dict['precision'],
                accuracy   = metrics_dict['accuracy'],
            )
            log_metrics(metrics, trainer_config)

            # The `min_is_better` field is relevant in constructor!!
            stopper_result = stopper.check(epoch, metrics.eval_loss)

            if stopper_result.is_best_round:
                torch.save(model.state_dict(), model_save_path)

            if stopper_result.should_exit:
                nice_print('It seems we are done here...')
                break
    except KeyboardInterrupt:
        nice_print('exiting loop...')


    model.load_state_dict(torch.load(model_save_path))
    metrics_dict = evaluate (
        model,
        ccs_test,
        input_test,
        positions_test,
        counts_test,
        trainer_config,
    )
    training_results = Training_Results (
        num_epochs = epoch,
        loss       = metrics_dict['loss'],
        recalls    = metrics_dict['recalls'],
        precision  = metrics_dict['precision'],
        accuracy   = metrics_dict['accuracy'],
    )
    return training_results


def evaluate (
    model: nn.Module,
    ccs_codes: list[np.ndarray],
    icd_codes: list[np.ndarray],
    positions: list[np.ndarray],
    counts:    list[np.ndarray],
    config:    Trainer_Config,
):
    # prepare the metrics dict
    metrics = {}
    metrics['loss'] = 0
    metrics['recalls'] = {}
    metrics['precision'] = {}
    metrics['accuracy'] = {}
    for k in config.metrics_config.recalls:
        metrics['recalls'][k] = 0
        metrics['precision'][k] = 0
        metrics['accuracy'][k] = 0
    divisor = 0

    num_batches = len(ccs_codes) // config.eval_batch_size
    for batch_id in tqdm(range(num_batches), ' eval', leave=False):
        batch_start = batch_id * config.eval_batch_size
        batch_end   = batch_start + config.eval_batch_size

        # get the right data for the batch
        i_icd  = icd_codes[batch_start: batch_end]
        i_ccs = ccs_codes[batch_start: batch_end]
        i_positions = positions[batch_start: batch_end]
        i_counts = counts[batch_start: batch_end]

        b_codes, outputs = prepare_data(
            i_ccs, i_icd, i_positions, i_counts,
        )
       
        # computations
        model.eval()
        with torch.inference_mode():
            predictions = model(b_codes)
            m = compute_metrics(predictions, outputs, config.metrics_config)
        metrics['loss'] += m['loss']
        for k in config.metrics_config.recalls:
            metrics['recalls'][k] += m['recalls'][k]
            metrics['precision'][k] += m['precision'][k]
            metrics['accuracy'][k] += m['accuracy'][k]
        divisor += sum([x.shape[0] for x in predictions])

    metrics['loss'] /= divisor
    for k in config.metrics_config.recalls:
        metrics['recalls'][k] /= num_batches
        metrics['precision'][k] /= num_batches
        metrics['accuracy'][k] /= num_batches
    return metrics

def compute_loss(predictions, outputs) -> torch.Tensor:
    """
    Computes the total binary cross-entropy loss between predicted logits and target outputs for a batch of sequences.

    Args:
        predictions (Iterable[torch.Tensor]): An iterable of predicted logits tensors, each of shape (seq_len, num_classes).
        outputs (Iterable[torch.Tensor]): An iterable of target tensors, each of shape (seq_len, num_classes).

    Returns:
        torch.Tensor: The total loss computed as the sum of binary cross-entropy losses for each prediction-output pair.

    Notes:
        - Each prediction tensor is sliced to match the length of its corresponding output tensor.
        - The loss for each pair is computed with reduction='sum', and all losses are summed to obtain the total loss.
    """
    losses = []
    for pred, out in zip(predictions, outputs):
        pred = pred[:out.shape[0], :]
        # if reduce='none' the function returns the same shape as input
        loss = F.binary_cross_entropy_with_logits(pred, out, reduction='sum')
        losses.append(loss)
    total_loss = sum(losses)
    return total_loss


def compute_metrics (predictions: list[torch.Tensor], outputs: list[torch.Tensor], config: Metrics_Config):
    """
    Computes various evaluation metrics (loss, recall, precision, accuracy) for multi-label classification tasks.
    Args:
        predictions (list[torch.Tensor]): List of model prediction tensors, each of shape (batch_size, num_classes).
        outputs (list[torch.Tensor]): List of ground truth tensors, each of shape (batch_size, num_classes).
        config (Metrics_Config): Configuration object specifying which recall values (top-k) to compute.
    Returns:
        dict: Dictionary containing:
            - 'loss' (float): Computed loss value.
            - 'recalls' (dict): Recall values for each specified k.
            - 'precision' (dict): Precision values for each specified k.
            - 'accuracy' (dict): Accuracy values for each specified k.
    """
    metrics = {}
    metrics['loss'] = float(compute_loss(predictions, outputs))
    metrics['recalls'] = {}
    metrics['precision'] = {}
    metrics['accuracy'] = {}
    for k in config.recalls:
        rec, prec, acc = [], [], []
        accuracy = TopKMultilabelAccuracy(criteria="hamming", k=int(k))

        for pred, out in zip(predictions, outputs):
            pred = pred[:out.shape[0], :]
            # create shifter to index the flatten array
            t = torch.ones(out.shape[0], dtype=int, device=out.device) * out.shape[-1]
            t[0] = 0
            t = t.cumsum(0).unsqueeze(1)

            sel = pred.topk(k, dim=-1).indices
            tp = out.flatten()[sel+t].sum(-1).to(torch.float32) # true positives
            tt = out.sum(-1)
            fp = k - tp  # false positives
            fn = tt - tp # false negatives
            tn = (out.numel() - tt) - fp # true negatives

            recall = (tp / tt).mean()
            precision = (tp / (tp + fp)).mean()
            #accuracy = ((tp + tn) / (tp + tn + fp + fn)).mean()
            accuracy.update(pred, out)

            rec.append(float(recall))
            prec.append(float(precision))
            #acc.append(float(accuracy))
    
        metrics['recalls'][k] = sum(rec) / len(rec)
        metrics['precision'][k] = sum(prec) / len(prec)
        metrics['accuracy'][k] = accuracy.compute().item()
        #metrics['accuracy'][k] = sum(acc) / len(acc)
        accuracy.reset()
    return metrics


def prepare_data(i_ccs, i_input, i_positions, i_counts):
    """
    Prepares and processes input data for a BiLSTM model, including grouping, padding, and output tensor creation.

    Args:
        i_ccs (list of np.ndarray): List of arrays containing CCS codes for each sample in the batch.
        i_input (list of np.ndarray): List of arrays containing input features for each sample in the batch.
        i_positions (list of np.ndarray): List of arrays indicating the position (e.g., visit index) for each input in the batch.
        i_counts (list of np.ndarray): List of arrays indicating the count of items per group (e.g., per visit) for each sample.

    Returns:
        tuple:
            padded_input_data (torch.Tensor): Batched and padded input data tensor of shape (batch_size, max_num_groups, max_group_length).
            outputs (list of torch.Tensor): List of output tensors for each sample, each of shape (num_groups-1, out_dim), containing one-hot encoded targets for the loss computation.

    Notes:
        - The function groups input data by positions, applies zero-padding to ensure uniform group lengths, and stacks them into a batch tensor.
        - The output tensors are constructed as one-hot encodings for each group, suitable for use as targets in a multi-label classification loss.
        - The function assumes access to a global `model` object with `config.device` and `config.out_dim` attributes.
    """
    # Remove the last elements from input and positions based on the last count value
    b_input = [x[:-int(c[-1])] for x, c in zip(i_input, i_counts)]
    b_positions = [x[:-int(c[-1])].astype(np.int_) for x, c in zip(i_positions, i_counts)]

    input_data = []
    # Group input data by unique positions (e.g., visits)
    for visit, position in zip(b_input, b_positions):
        new_input = [visit[position == pos].tolist() for pos in np.unique(position)]
        input_data.append(new_input)
    
    # Function to pad each sublist to the maximum length
    def pad_input(data, max_length):
        return [sublist + [0] * (max_length - len(sublist)) for sublist in data]

    # Find the maximum length among all sublists in the batch
    max_length = max([max(len(sublist) for sublist in row) for row in input_data])

    # Pad each sublist in each sample to the maximum length and convert to tensor
    padded_input_data = []
    for row in input_data:
        padded_row = torch.tensor(pad_input(row, max_length), dtype=torch.int32, device=model.config.device)
        padded_input_data.append(padded_row)
    # Pad the batch to ensure uniform number of groups (visits) across samples
    padded_input_data = pad_sequence(padded_input_data, batch_first=True, padding_value=0)

    # Compute expected outputs for the loss (one-hot encoding for each group except the first)
    outputs = []
    for it in range(len(b_input)):  # Iterate over batch size
        sz = len(i_counts[it])
        out = np.zeros((sz-1, model.config.out_dim), dtype=float)
        cursor = i_counts[it][0]
        for jt in range(1, sz):
            cursor_end = cursor + i_counts[it][jt]
            # Set 1 for the corresponding CCS codes in the output
            out[jt-1, i_ccs[it][cursor:cursor_end]] = 1
            cursor = cursor_end
        out = torch.from_numpy(out).to(torch.float).to(model.config.device)
        outputs.append(out)
    return padded_input_data, outputs

DIR_ID_LENGTH = 5
ALL_LETTERS   = 'abcdefghijklmnopqrstuvxywz'
def format_path(path: str) -> str:
    date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    id = ''.join([random.choice(ALL_LETTERS) for _ in range(DIR_ID_LENGTH)])
    path = path.replace('%(id)',   id)
    path = path.replace('%(date)', date)
    return path


if __name__ == '__main__':
    nice_print(f'Using config file: {CFG_FILE_NAME}')
    with open(CFG_FILE_NAME, 'r') as f:
        txt = f.read()
    config = tomlkit.parse(txt)

    diagnoses = pl.read_parquet(config['diagnoses_path'])

    config['model']['max_features']  = diagnoses['icd9_id'].list.max().max() + 1
    config['model']['out_dim'] = diagnoses['ccs_id'] .list.max().max() + 1
    bilstm_config = BiLSTM_Config(**config['model'])
    # MODIFY GPU NUMBER TO USE IN CONFIG.TOML
    bilstm_config.device = torch.device("cuda:" + bilstm_config.device if torch.cuda.is_available() else "mps")
    model = BiLSTM(bilstm_config)

    num_params  = sum([param.nelement()                      for param in model.parameters()])
    size_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    nice_print(f'model has {num_params/1e6:.2f}M params, occupying {size_params/1024/1024:.2f}M of memory')

    trainer_config = Trainer_Config(**config['trainer'])
    metrics_config = Metrics_Config(**config['metrics'])
    trainer_config.metrics_config = metrics_config

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
    test_results['training_epochs'] = results.num_epochs
    test_results['loss'] = results.loss
    for k in sorted(metrics_config.recalls):
        test_results[f'recall_{k}'] = results.recalls[k]
        test_results[f'precision_{k}'] = results.precision[k]
        test_results[f'accuracy_{k}'] = results.accuracy[k]
    config['test_results'] = test_results

    # save new results on config file
    new_config_text = tomlkit.dumps(config)
    with open(new_config_path, 'w') as f:
        f.write(new_config_text)

    # print the test results on screen
    txt = [f'test loss: {results.loss:.3f}']
    for k in sorted(metrics_config.recalls):
        txt += [f'recall_{k: <2}: {results.recalls[k]*100:.2f}%']
        txt += [f'precision_{k: <2}: {results.precision[k]*100:.2f}%']
        txt += [f'accuracy_{k: <2}: {results.accuracy[k]*100:.2f}%']
    txt = '    '.join(txt)
    nice_print(txt)

