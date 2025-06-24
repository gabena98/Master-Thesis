import polars as pl
import numpy as np
import pickle
import lib.generator as gen
from sklearn.metrics import f1_score, make_scorer
from scipy.spatial import distance
from Generator.llama2_model import *
import tomlkit
import sys
import os

path_to_add = "/home/gbenanti/Tesi_Benanti/SETOR"
if path_to_add not in sys.path:
    sys.path.append(path_to_add)

sys.path.append(os.path.abspath("../.."))
from SETOR.datasets import FTDataset, llama2Dataset, collate_pt, load_data, collate_llama2
from SETOR.models.order import NextDxPrediction

import torch
from torch.utils.data import DataLoader
from sklearn.tree import DecisionTreeClassifier
# Enable experimental feature for halving search CV
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV, HalvingRandomSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics

from tqdm import tqdm

import random
import pandas as pd

ontology_path   = '../data/processed/ontology.parquet'
diagnoses_path  = '../data/processed/diagnoses.parquet'
generation_path = '../data/processed/generation.parquet'
ccs_path        = '../data/processed/ccs.parquet'
icd_path        = '../data/processed/icd.parquet'

config_path = 'explain_config.toml'
output_path = 'results/explainer.txt'

# Path to the generative model for filler
filler_path = 'results/filler-asqvj-2025-06-04_12:16:57'

#filler_path     = 'results/filler-mpgzq-2025-02-19_19:17:59'
k_reals          = 50
batch_size       = 64
keep_prob        = 0.8 # Probability for keeping codes during ontological perturbation
topk_predictions = 10
uniform_perturbation       = False
tree_train_fraction        = 0.8
synthetic_multiply_factor  = 4
generative_multiply_factor = 4

APPLE_SILICON = False
# Set the device depending on GPU availability
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

random.seed(42)
np.random.seed(seed=42)
torch.manual_seed(seed=42)

def device_info(path: str):
    """
    Determines the device to use for computation based on system configuration.

    If running on Apple Silicon, returns 'mps'. Otherwise, reads the device from a config file.

    Args:
        path (str): Directory containing 'config.toml'.

    Returns:
        str: Device type (e.g., 'mps', 'cpu', 'cuda').
    """
    if APPLE_SILICON:
        return 'mps'
    config_path = os.path.join(path, 'config.toml')
    with open(config_path, 'r') as f:
        config = tomlkit.loads(f.read())
    return config['model']['device']

def print_patient(ids: np.ndarray, cnt: np.ndarray, ontology: pl.DataFrame):
    """
    Print patient visit information using ICD-9 codes and their descriptions.

    Args:
        ids (np.ndarray): Array of ICD-9 code IDs.
        cnt (np.ndarray): Array with the number of codes per visit.
        ontology (pl.DataFrame): DataFrame mapping ICD-9 codes to descriptions, must have 'icd9_id' and 'label'.

    Prints each visit with its codes and corresponding descriptions.
    """
    codes = pl.DataFrame({'icd9_id':ids})
    codes = codes.join(ontology, how='left', on='icd9_id')

    cursor = 0
    for it in range(cnt.shape[0]):
        length = cnt[it]
        lines = []
        for jt in range(cursor, cursor+length):
            x = f'[{codes["icd_code"][jt]:}]' 
            txt  = f'    {x: <10}'
            txt += f'{codes["label"][jt]}'
            lines.append(txt)
        txt = '\n'.join(lines)
        print(f'visit {it+1}')
        print(txt)
        cursor += length

import numpy as np

def f1_micro(y_true, y_pred):
    """
    Compute the micro-averaged F1 score for multi-label classification.

    Args:
        y_true (np.ndarray): Ground truth binary array (n_samples, n_labels).
        y_pred (np.ndarray): Predicted binary array (n_samples, n_labels).

    Returns:
        float: Micro F1 score.
    """
    # Flatten arrays to treat as a single vector
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculate true positives, false positives, false negatives
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))

    # Compute precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Compute micro F1
    if precision + recall == 0:
        return 0.0
    f1_micro = 2 * precision * recall / (precision + recall)
    
    return f1_micro
         
def explain_label(neigh_icd, neigh_counts, labels, max_icd_id, tree_train_fraction, reference_patient, reference_label):
    """
    Fit a decision tree to explain a specific label using neighborhood data and a reference patient.

    Args:
        neigh_icd (list): ICD codes for neighboring patients.
        neigh_counts (list): Counts for ICD codes for neighboring patients.
        labels (list): Labels for neighboring patients.
        max_icd_id (int): Maximum ICD code ID for encoding.
        tree_train_fraction (float): Fraction of data for training.
        reference_patient (array-like): Encoded reference patient to include in training.
        reference_label (array-like): Label for the reference patient.

    Returns:
        tree_classifier (DecisionTreeClassifier): Trained decision tree.
        tree_inputs_eval (array-like): Encoded evaluation set inputs.
        labels_eval (array-like): Evaluation set labels.

    Notes:
        - Uses temporal encoding for input data.
        - Ensures the reference patient is included in the training set.
    """
    try:
        # Encode the neighborhood patients using temporal encoding
        tree_inputs = gen.ids_to_encoded(neigh_icd, neigh_counts, max_icd_id, 0.5)
    except Exception as e:
        print(f"Error during dataset encoding: {e}")

    tree_classifier = DecisionTreeClassifier(random_state = 42)

    train_split = int(tree_train_fraction * len(labels))

    tree_inputs_train = tree_inputs[:train_split]
    tree_inputs_eval = tree_inputs[train_split:]
    labels_train = labels[:train_split]
    labels_eval = labels[train_split:]
    tree_inputs_train = np.vstack([tree_inputs_train, reference_patient])
    labels_train = np.vstack([labels_train, reference_label])

    score = make_scorer(f1_micro, greater_is_better=True)

    param_distributions = {
        'max_depth': [2, 8, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    search = RandomizedSearchCV(
        estimator=tree_classifier,
        param_distributions=param_distributions,
        cv=5,
        scoring='f1_micro',
        random_state=42,
        n_iter=10
    )

    search.fit(tree_inputs_train, labels_train)
    tree_classifier = search.best_estimator_

    return tree_classifier, tree_inputs_eval, labels_eval

def setor_preprocessing(pids, inputs, labels, labels_visit, config):
    """
    Preprocess data for the SETOR model: split into train/val/test and create DataLoaders.

    Args:
        pids (list): Patient IDs.
        inputs (object): Input sequences (e.g., patient visits).
        labels (object): Labels for next visit prediction.
        labels_visit (object): Labels for visit-level classification.
        config (object): Model config with `num_ccs_classes` and `num_visit_classes`.

    Returns:
        tuple: (train_data_loader, val_data_loader, test_data_loader)
    """
    # Split data into train, validation, and test sets
    train_set, valid_set, test_set = load_data(pids, inputs, labels, labels_visit, 0.8)

    train_dataset = FTDataset(train_set[0], train_set[1], train_set[2], train_set[3])
    train_data_loader = DataLoader(train_dataset, batch_size=1,
                                    collate_fn=lambda batch: collate_pt(batch, config.num_ccs_classes,
                                                                        config.num_visit_classes),
                                    num_workers=0, shuffle=True)

    val_dataset = FTDataset(valid_set[0], valid_set[1], valid_set[2], valid_set[3])
    val_data_loader = DataLoader(val_dataset, batch_size=1,
                                    collate_fn=lambda batch: collate_pt(batch, config.num_ccs_classes,
                                                                        config.num_visit_classes),
                                    num_workers=0, shuffle=True)

    test_dataset = FTDataset(test_set[0], test_set[1], test_set[2], test_set[3])
    test_data_loader = DataLoader(test_dataset, batch_size=1,
                                    collate_fn=lambda batch: collate_pt(batch, config.num_ccs_classes,
                                                                        config.num_visit_classes),
                                    num_workers=0, shuffle=True)

    return train_data_loader, val_data_loader, test_data_loader

def setor_to_llama2_output(setor_output, setor_to_llama2_ccs, batch_size):
    """
    Convert SETOR model output to llama2-compatible format.

    Args:
        setor_output (list): Output from SETOR model (list of patients, each with visits and codes).
        setor_to_llama2_ccs (pandas.DataFrame): Mapping DataFrame from SETOR codes to llama2 CCS codes.
        batch_size (int): Batch size (currently unused).

    Returns:
        list: Nested list with SETOR codes replaced by llama2 CCS codes.

    Notes:
        - Assumes all codes in `setor_output` have a mapping in `setor_to_llama2_ccs`.
        - TODO: Handle missing mappings.
    """
    setor_ccs_codes = []
    for patient in setor_output:
        # TODO: Complete based on SETOR output structure
        setor_visits = []
        for visit in patient:
            setor_codes = []
            for codes in visit:
                setor_codes.append(setor_to_llama2_ccs.loc[setor_to_llama2_ccs['ccs_id_generator'] == codes, 'ccs_id_setor'].values)
            setor_visits.append(setor_codes)
        setor_ccs_codes.append(setor_visits) 

def llama2_to_setor_batch(icd_codes, positions, setor_to_llama2_icd, batch_size=1):
    """
    Convert ICD codes from llama2 format to SETOR format in batches for DataLoader.

    Args:
        icd_codes (list of lists): ICD codes per visit in llama2 format.
        positions (list of lists): Positions per visit.
        setor_to_llama2_icd (dict): Mapping dictionary with 'icd9_id_generator' and 'icd9_id_setor'.
        batch_size (int): Batch size for DataLoader.

    Returns:
        iterator: Iterator over the DataLoader yielding processed batches.

    Notes:
        - Maps llama2 ICD codes to SETOR ICD codes.
        - Groups codes by position, excluding the last unique position.
    """
    # Create mapping dictionary
    mapping_dict = dict(zip(setor_to_llama2_icd['icd9_id_generator'], setor_to_llama2_icd['icd9_id_setor']))
    filtered_groups = []

    for visit, position in zip(icd_codes, positions):
        # Replace llama2 ICD IDs with SETOR ICD IDs
        setor_ids = [mapping_dict[i] for i in visit]

        # Group by position (ensure lengths match)
        min_length = min(len(setor_ids), len(position))
        setor_ids = setor_ids[:min_length]
        position = position[:min_length]

        filtered_group = [np.array(setor_ids)[np.array(position) == pos].tolist() for pos in np.unique(np.array(position))[:]]
        filtered_groups.append(filtered_group)

    # Create dataset and DataLoader for batch processing
    batch_dataset = llama2Dataset(filtered_groups)
    batch_data_loader = DataLoader(batch_dataset, batch_size=batch_size,
                                    collate_fn=lambda batch: collate_llama2(batch),
                                    num_workers=0, shuffle=True)

    return iter(batch_data_loader)

def load_setor_predictor(device):
    """
    Load the SETOR predictor model and set it to evaluation mode.

    Args:
        device (torch.device): Device to load the model on.

    Returns:
        NextDxPrediction: Loaded SETOR model in eval mode.

    Notes:
        - Loads config and weights from specific relative paths.
    """
    model_config = torch.load("../../SETOR/outputs/outputs/model/dx_alpha_0.01_lr_0.1_bs_32_e_100.0_l_1.0_tr_0.8/SETOR_config.pth", map_location=device, weights_only=False)
    model_config.device = device

    setor = NextDxPrediction(model_config)
    setor.load_state_dict(torch.load("../../SETOR/outputs/outputs/model/dx_alpha_0.01_lr_0.1_bs_32_e_100.0_l_1.0_tr_0.8/pytorch_model_weights_only.bin", map_location=device, weights_only=False), strict=False)
    setor.to(device)
    setor.eval()
    return setor

def setor_to_llama2_output(labels, setor_ccs_data, device):
    """
    Map SETOR model output labels to llama2 format using a mapping.

    Args:
        labels (list of torch.Tensor): Predicted labels from SETOR model.
        setor_ccs_data (dict): Mapping with 'ccs_id_setor' and 'ccs_id_generator'.
        device (torch.device): Device to move the output tensor to.

    Returns:
        torch.Tensor: Transformed predictions in llama2 format on the specified device.
    """
    setor_to_llama2_ccs = dict(zip(setor_ccs_data['ccs_id_setor'], setor_ccs_data['ccs_id_generator']))
    predictions = [] 
    for pred in labels:
        pred = pred.cpu().numpy()
        pred = [setor_to_llama2_ccs[id] for id in pred]
        predictions.append(pred)
    return torch.tensor(predictions).to(device)

def compare_models(model1, model2):
    """
    Compare two PyTorch models for identical weights and parameter keys.

    Args:
        model1 (torch.nn.Module): First model.
        model2 (torch.nn.Module): Second model.

    Returns:
        bool: True if models have identical parameter keys and weights, False otherwise.

    Notes:
        - Checks parameter names and values for equality.
    """
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # Check if parameter keys match
    if state_dict1.keys() != state_dict2.keys():
        print("State_dict keys do not match.")
        return False

    # Check if weights are identical
    for key in state_dict1:
        if not torch.allclose(state_dict1[key], state_dict2[key]):
            print(f"Parameter {key} differs between models.")
            return False

    print("The two models have identical weights!")
    return True

def OneHotEncoding(data, max_ccs_id=281):
    """
    One-hot encode a list of integer indices.

    Args:
        data (list of int): Indices to set to 1 in the one-hot vector.
        max_ccs_id (int): Length of the one-hot vector.

    Returns:
        numpy.ndarray: Array of one-hot encoded vectors.
    """
    encoded_data = []
    for vector in data:
        one_hot_encoded = np.zeros(max_ccs_id, dtype=np.float32)
        one_hot_encoded[vector] = 1
        encoded_data.append(one_hot_encoded)
    return np.array(encoded_data)

def main(reference_index, ontological_perturbation, generative_perturbation):
    # Load SETOR data
    pids = pickle.load(open("../../SETOR/outputs/data/pids.seqs", 'rb'))
    inputs = pickle.load(open("../../SETOR/outputs/data/inputs.seqs", 'rb'))
    labels = pickle.load(open("../../SETOR/outputs/data/labels_next_visit.label", 'rb'))
    labels_visit = pickle.load(open("../../SETOR/outputs/data/labels_visit_cat1.label", 'rb'))

    # Load SETOR dictionaries
    ccs_cat1 = pickle.load(open("../../SETOR/outputs/data/ccs_cat1.dict", 'rb'))
    ccs_cat1 = pd.DataFrame.from_dict(ccs_cat1, orient='index', columns=['id'], dtype='uint16').reset_index().rename(columns={'index':'ccs_cat1'})
    ccs_to_id = pickle.load(open("../../SETOR/outputs/data/ccs_single_level.dict", 'rb'))
    ccs_to_id = pd.DataFrame.from_dict(ccs_to_id, orient='index', columns=['ccs_id'], dtype='uint16').reset_index().rename(columns={'index':'ccs'})
    ccs_to_id['ccs'] = ccs_to_id['ccs'].str.replace('D_', '', regex=True).astype('uint32') 
    icd9_to_id = pickle.load(open("../../SETOR/outputs/data/inputs.dict", 'rb'))
    icd9_to_id = pd.DataFrame.from_dict(icd9_to_id, orient='index', columns=['icd9_id'], dtype='uint32').reset_index().rename(columns={'index':'icd9'}) 
    icd9_to_id['icd9'] = icd9_to_id['icd9'].str.replace('D_', '', regex=True).astype('string')
    icd9_to_desc = pickle.load(open("../../SETOR/outputs/data/code2desc.dict", 'rb'))
    icd9_to_desc = pd.DataFrame.from_dict(icd9_to_desc, orient='index', columns=['description'], dtype='string').reset_index().rename(columns={'index':'icd9'}) 
    icd9_to_desc['icd9'] = icd9_to_desc['icd9'].str.replace('D_', '', regex=True).astype('string')

    # Load DoctorXAI data
    ontology = pl.read_parquet(ontology_path)
    diagnoses = pl.read_parquet(diagnoses_path)
    ccs_data = pl.read_parquet(ccs_path)
    icd9_data = pl.read_parquet(icd_path)
    icd9_data = icd9_data.join(ontology.select(['icd9_id', 'icd_code', 'label']), on=['icd9_id', 'icd_code'], how='left').fill_null('No description')

    # Merge ccs_to_id and ccs_data to get columns: ['ccs', 'description', 'ccs_id_generator', 'ccs_id_setor']
    setor_ccs_data = ccs_data.join(pl.DataFrame(ccs_to_id), on='ccs', how='left').fill_null(ccs_data.shape[0]).rename({'ccs_id_right': 'ccs_id_setor','ccs_id': 'ccs_id_generator'})

    # Merge icd9_to_id and icd9_to_desc to get columns: ['icd9', 'description', 'icd9_id_setor', 'icd9_id_generator']
    setor_icd9_data = pl.DataFrame(icd9_to_desc).join(pl.DataFrame(icd9_to_id), on='icd9', how='left').fill_null(0).rename({'icd9_id': 'icd9_id_setor'})
    setor_icd9_data = setor_icd9_data.join(icd9_data, left_on='icd9', right_on='icd_code', how='full').fill_null(0).rename({'icd9_id' : 'icd9_id_generator'}).drop('icd_code', 'label')

    # Load SETOR model
    setor = load_setor_predictor(device)

    # Load base data and model
    diagnoses_train = diagnoses.filter(pl.col('role') == 'train')
    diagnoses_eval  = diagnoses.filter(pl.col('role') == 'eval')
    diagnoses_test  = diagnoses.filter(pl.col('role') == 'test')

    unique_codes = diagnoses['icd9_id'].explode().unique().to_numpy()
    max_ccs_id = ccs_data['ccs_id'].max() + 1
    max_icd_id = unique_codes.max() + 1

    ontology_array = ontology[['icd9_id', 'parent_id']].to_numpy()
    # Used by compute_patient_distances
    gen.create_c2c_table(ontology_array, unique_codes)

    # Extract numpy arrays from data
    icd_codes_all = list(diagnoses['icd9_id'].to_numpy())
    ccs_codes_all = list(diagnoses['ccs_id'].to_numpy())
    positions_all = list(diagnoses['position'].to_numpy())
    counts_all = list(diagnoses['count'].to_numpy())

    ccs_codes_test = list(diagnoses_test['ccs_id'].to_numpy())
    icd_codes_test = list(diagnoses_test['icd9_id'].to_numpy())
    positions_test = list(diagnoses_test['position'].to_numpy())
    counts_test = list(diagnoses_test['count'].to_numpy())

    if generative_perturbation:
        filler, hole_prob, hole_token_id = load_llama2_for_generation(filler_path, device)
        conv_data = pl.read_parquet(generation_path).sort('out_id')
        zero_row  = pl.DataFrame({'icd9_id':0, 'out_id':0, 'ccs_id':0}, schema=conv_data.schema)
        conv_data = pl.concat([zero_row, conv_data])

        out_to_icd = conv_data['icd9_id'].to_numpy()
        out_to_ccs = conv_data['ccs_id' ].to_numpy()

    # Find the closest neighbors in the real data
    distance_list = gen.compute_patients_distances (
        icd_codes_test[reference_index],
        counts_test[reference_index],
        icd_codes_all,
        counts_all
    )
    
    topk = np.argpartition(distance_list, k_reals+1)[:k_reals+1]

    real_neigh_icd       = []
    real_neigh_ccs       = []
    real_neigh_counts    = []
    real_neigh_positions = []
    synt_neigh_icd       = []
    synt_neigh_ccs       = []
    synt_neigh_counts    = []
    synt_neigh_positions = []

    # Select the k most similar samples to the reference patient
    for it in range(k_reals+1):
        real_neigh_icd.append(icd_codes_all[topk[it]])
        real_neigh_ccs.append(ccs_codes_all[topk[it]])
        real_neigh_counts.append(counts_all[topk[it]])
        real_neigh_positions.append(positions_all[topk[it]])

    # Augment the neighbors with synthetic samples

    if ontological_perturbation:
        displacements, new_counts = gen.ontological_perturbation(real_neigh_icd, real_neigh_counts, synthetic_multiply_factor, keep_prob)

        new_neigh_icd = []
        new_neigh_ccs = []
        new_neigh_positions = []
        for it, (icd, ccs, pos) in enumerate(zip(real_neigh_icd, real_neigh_ccs, real_neigh_positions)):
            for jt in range(synthetic_multiply_factor):
                displ = displacements[synthetic_multiply_factor * it + jt]
                new_neigh_icd.append(icd[displ])
                new_neigh_ccs.append(ccs[displ])
                new_neigh_positions.append(pos[displ])

        synt_neigh_icd += new_neigh_icd
        synt_neigh_ccs += new_neigh_ccs
        synt_neigh_positions += new_neigh_positions
        synt_neigh_counts += new_counts

    if generative_perturbation:
        new_neigh_icd = []
        new_neigh_ccs = []
        new_neigh_counts = []
        new_neigh_positions = []
        cursor = 0
        while cursor < len(real_neigh_icd):
            new_cursor = min(cursor + batch_size, len(real_neigh_icd))

            for _ in range(generative_multiply_factor):
                batch = prepare_batch_for_generation (
                    real_neigh_icd[cursor:new_cursor],
                    real_neigh_counts[cursor:new_cursor],
                    real_neigh_positions[cursor:new_cursor],
                    hole_prob,
                    hole_token_id,
                    device
                )

                if uniform_perturbation:
                    bsz = batch.codes.shape[0]
                    b_n = batch.codes.shape[1]
                    n_out = filler.head.out_features
                    gen_output = torch.zeros((bsz, b_n, n_out))
                else:
                    gen_output = filler(**batch.unpack()) # (batch_size, seq_len, out_features)
                # batch_size = 64, seq_len = 62, out_features = 500 (top k most frequent icd9 codes)
                old_shape = gen_output.shape
                gen_output = gen_output.reshape((-1, gen_output.shape[-1]))
                gen_output = torch.softmax(gen_output, dim=-1)
                gen_output.shape # 3968x500
                new_codes = torch.multinomial(gen_output, 1)
                new_codes = new_codes.reshape(old_shape[:-1])
                new_codes.shape # 64x62
                new_codes = new_codes.cpu().numpy()
                # new generator
                new_icd = [np.array(codes, dtype=np.uint32) for codes in new_codes]
                
                # Add only generated codes, excluding padding codes
                for i, pos in enumerate(real_neigh_positions[cursor:new_cursor]):
                    new_neigh_icd.append(new_icd[i][:len(pos)])

                new_neigh_counts += real_neigh_counts[cursor:new_cursor]
                new_neigh_positions += real_neigh_positions[cursor:new_cursor]

            cursor = new_cursor
        synt_neigh_icd += new_neigh_icd
        synt_neigh_ccs += new_neigh_ccs
        synt_neigh_counts += new_neigh_counts
        synt_neigh_positions += new_neigh_positions
        
    # Convert llama2 data to SETOR format
    setor_batch = llama2_to_setor_batch([icd_codes_test[reference_index]], [positions_test[reference_index]], setor_icd9_data)

    for reference_batch in setor_batch:
        reference_batch = {k: t.to(device) for k, t in reference_batch.items()}
        # batch_size(1) x seq_len(e.g.11), num_labels(280)
        setor_output, _, _  = setor(reference_batch["input"], reference_batch["visit_mask"], reference_batch["code_mask"], output_attentions=True)
        #predicts = setor_output.cpu().detach().numpy()
        # seq_len(e.g. 3)x num_labels(280)
        predicts = setor_output.reshape(-1, setor_output.shape[-1])
        predicts = predicts[-1]
        labels = predicts.topk(topk_predictions, dim=-1).indices

    # Convert SETOR CCS id codes to llama2 CCS id codes
    reference_llama2_label = setor_to_llama2_output([labels], setor_ccs_data, device).reshape(-1)

    # Black-box predictions on neighbors
    neigh_labels = np.empty((len(synt_neigh_icd), len(reference_llama2_label), ), dtype=np.int32)

    setor_batch = llama2_to_setor_batch(synt_neigh_icd, synt_neigh_positions, setor_icd9_data, batch_size=batch_size)
    for iter, batch in enumerate(setor_batch):
        batch = {k: t.to(device) for k, t in batch.items()}
        setor_output, _, _  = setor(batch["input"], batch["visit_mask"], batch["code_mask"], output_attentions=True)
        
        # Take the last row (true) of each output matrix
        outputs = [x[-1] for x in setor_output]
        # batch_size 64 x ccs 281
        outputs = torch.stack(outputs)
        # Take the top-k indices of the ccs categories
        batch_labels = outputs.topk(topk_predictions, dim=-1).indices
        batch_labels = setor_to_llama2_output(batch_labels, setor_ccs_data, device)
        batch_labels = batch_labels.cpu().numpy()

        neigh_labels[iter*batch_size:(iter+1)*batch_size, :] = batch_labels

    # Convert neigh_labels to one-hot encoding
    neigh_labels_onehot = OneHotEncoding(neigh_labels, max_ccs_id)

    # Explanation
    reference_enc = gen.ids_to_encoded(
        [icd_codes_test[reference_index]],
        [counts_test[reference_index]],
        max_icd_id,
        0.5
    )[0]

    # Add one-hot encoding for tree labels
    reference_label_encoded = OneHotEncoding([reference_llama2_label.cpu().numpy()], max_ccs_id)

    # The decision tree is trained on the temporal encoding of patient visits, as described in the paper
    tree, tree_inputs_encoded, model_labels_encoded = explain_label(synt_neigh_icd, synt_neigh_counts, neigh_labels_onehot, max_icd_id, tree_train_fraction,
        reference_enc, reference_label_encoded)

    # Extract rules from the explanation

    tree_path = tree.tree_.decision_path(reference_enc.reshape((1,-1))).indices
    features = tree.tree_.feature

    expl_labels = [features[i] for i in tree_path]
    expl_labels = [x for x in expl_labels if x >= 0]

    thresholds = tree.tree_.threshold
    thresholds = [thresholds[i] for i in tree_path if features[i] >= 0]

    df = pl.DataFrame({'icd9_id': expl_labels, 'threasholds': thresholds}).with_columns(icd9_id=pl.col('icd9_id').cast(pl.UInt32))
    df = df.join(icd9_data, left_on='icd9_id', right_on='icd9_id', how='left')

    print_patient(icd_codes_test[reference_index], counts_test[reference_index], ontology)
    print('\n')

    print('ccs predicted')
    labels = reference_llama2_label.tolist()
    for id in labels:
        infos = ccs_data.filter(pl.col('ccs_id') == id)
        if len(infos) != 1:
            raise ValueError('should not happen')
        
        code = infos['ccs'][0]
        label = infos['description'][0]

        print(f'{"["+str(code)+"]": <7} {label}')

    print('\ndecision rules')
    for (id, thresh, icd9, desc) in df.iter_rows():
        txt = f'code {icd9: <4} - threashold: {thresh:.2f}, found {reference_enc[id]:.2f},  ({desc})'
        print(txt)

    print('\ncodes of the patient relevant for explanation')
    # For each ICD code in the decision tree path
    for id in expl_labels:
        for it in range(len(icd_codes_test[reference_index])):
            icd = icd_codes_test[reference_index][it]
            # If the ICD code matches the decision tree node
            if icd == id:
                visit = positions_test[reference_index][it] + 1

                infos = ontology.filter(pl.col('icd9_id') == icd)
                if len(infos) != 1:
                    raise ValueError('Should not happen')
                code = infos['icd_code'][0]
                label = infos['label'][0]

                print(f'At visit {(str(visit)+","): <3} code {code: <6} [{label}]')

    # Compute fidelity of the decision tree to the black box on the synthetic neighborhood
    fidelity_synth = f1_score(y_true=model_labels_encoded, y_pred=tree.predict(tree_inputs_encoded), average='micro')
    # Compute hit rate of the decision tree
    hit_synth = 1 - distance.hamming(reference_label_encoded.reshape(-1), tree.predict(reference_enc.reshape(1,-1)).flatten())
    return fidelity_synth, hit_synth

if __name__ == '__main__':
    reference_index = 0 # Index of the patient to explain
    ontological_perturbation   = True
    generative_perturbation    = False
    fidelity_synth, hit_synth = main(reference_index, ontological_perturbation, generative_perturbation)
    print('Evaluating explainability metrics...')
    print(f'Fidelity on synthetic neighborhood: {fidelity_synth}')
    print(f'Hit on synthetic neighborhood: {hit_synth}')
