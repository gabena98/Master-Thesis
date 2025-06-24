import polars as pl
import numpy as np

from sklearn.model_selection import RandomizedSearchCV
import lib.generator as gen
from sklearn.metrics import f1_score
from scipy.spatial import distance
from Generator.llama2_model import *
import tomlkit
import sys
import os

sys.path.append(os.path.abspath("../.."))
from BiLSTM.BiLSTM_model import *

import torch
from torch.utils.data import DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import random

# Paths to data and configuration files
ontology_path   = '../data/processed/ontology.parquet'
diagnoses_path  = '../data/processed/diagnoses.parquet'
generation_path = '../data/processed/generation.parquet'
ccs_path        = '../data/processed/ccs.parquet'
icd_path        = '../data/processed/icd.parquet'

config_path = 'BiLSTM_config.toml'
output_path = 'results/explainer.txt'

model_path      = 'results/BiLSTM-udayi-2025-04-05_20:12:13'
# Path to the generative model for synthetic data augmentation
#filler_path = 'results/filler-asqvj-2025-06-04_12:16:57' # new filler
filler_path     = 'results/filler-mpgzq-2025-02-19_19:17:59' # old filler

# Parameters for neighborhood and perturbation
k_reals          = 10  # Number of real neighbors to consider
synthetic_neighborhood_size = 200  # Number of synthetic neighbors to generate
batch_size       = 64   # Batch size for model inference/generation
keep_prob        = 0.85 # Probability to keep a code during ontological perturbation
topk_predictions = 10   # Number of top predictions to consider for each patient
uniform_perturbation       = False # If True, use uniform perturbation instead of generative model
tree_train_fraction        = 0.75  # Fraction of data to use for training the decision tree

APPLE_SILICON = False
# Set the computation device based on GPU availability
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(seed=42)
torch.manual_seed(seed=42)

def device_info(path: str):
    """
    Determines the device to be used for computation based on the system configuration.

    If the system is running on Apple Silicon, it returns 'mps' (Metal Performance Shaders).
    Otherwise, it reads the device information from a configuration file located at the given path.

    Args:
        path (str): The directory path where the 'config.toml' file is located.

    Returns:
        str: The device type to be used for computation (e.g., 'mps', 'cpu', 'cuda').
    """
    if APPLE_SILICON:
        return 'mps'
    config_path = os.path.join(path, 'BiLSTM_config.toml')
    with open(config_path, 'r') as f:
        config = tomlkit.loads(f.read())
    return config['model']['device']

def print_patient(ids: np.ndarray, cnt: np.ndarray, ontology: pl.DataFrame):
    """
    Prints the visit history of a patient, showing ICD-9 codes and their descriptions for each visit.

    Args:
        ids (np.ndarray): Array of ICD-9 code identifiers for all visits.
        cnt (np.ndarray): Array where each element is the number of codes in a specific visit.
        ontology (pl.DataFrame): DataFrame mapping ICD-9 codes to their descriptions. 
                                 Must contain 'icd9_id', 'icd_code', and 'label' columns.

    Behavior:
        - Joins the ICD-9 codes with the ontology to retrieve code labels.
        - Iterates through each visit, printing the codes and their descriptions.
        - Each visit is clearly separated in the output.

    Example Output:
        visit 1
        [12345]    Description of code 12345
        [67890]    Description of code 67890
        visit 2
        [54321]    Description of code 54321
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
            
def explain_label(neigh_icd, neigh_counts, labels, max_icd_id, tree_train_fraction, reference_patient, reference_label):
    """
    Trains a decision tree classifier to explain the model's prediction for a specific patient
    by fitting the tree on a synthetic neighborhood of similar patients.

    Parameters:
    -----------
    neigh_icd : list
        List of ICD codes for the neighboring (synthetic) patients.
    neigh_counts : list
        List of code counts for the neighboring patients.
    labels : list
        List of labels (predictions) for the neighboring patients.
    max_icd_id : int
        Maximum ICD code ID, used for encoding.
    tree_train_fraction : float
        Fraction of the neighborhood to use for training the decision tree.
    reference_patient : array-like
        Encoded representation of the reference patient (to ensure inclusion in training).
    reference_label : array-like
        Label of the reference patient (to ensure inclusion in training).

    Returns:
    --------
    tree_classifier : DecisionTreeClassifier
        The trained decision tree classifier.
    tree_inputs_eval : array-like
        Encoded inputs for the evaluation set.
    labels_eval : array-like
        Labels for the evaluation set.

    Notes:
    ------
    - Uses temporal encoding for patient data.
    - The reference patient and label are always included in the training set.
    - Hyperparameters are optimized using randomized search.
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
    # Ensure the reference patient is included in the training set
    tree_inputs_train = np.vstack([tree_inputs_train, reference_patient])
    labels_train = np.vstack([labels_train, reference_label])

    param_distributions = {
        'max_depth': [2, 8, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Hyperparameter search for the decision tree
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

def OneHotEncoding(data, max_ccs_id=281):
    """
    Converts a list of integer indices to one-hot encoded vectors.

    Args:
        data (list of int): List of integer indices to encode.
        max_ccs_id (int): Size of the one-hot vector (default: 281).

    Returns:
        numpy.ndarray: Array of one-hot encoded vectors.

    Example:
        >>> data = [0, 2, 5]
        >>> OneHotEncoding(data)
        array([1., 0., 1., ..., 0., 0., 0.])
    """
    encoded_data = []
    for vector in data:
        one_hot_encoded = np.zeros(max_ccs_id, dtype=np.float32)
        one_hot_encoded[vector] = 1
        encoded_data.append(one_hot_encoded)
    return np.array(encoded_data)

def main(reference_index, ontological_perturbation, generative_perturbation):
    """
    Main function to generate an explanation for a model prediction on a specific patient.

    Steps:
        1. Load all required data and models.
        2. Select a reference patient and find its closest real neighbors.
        3. Augment the neighborhood with synthetic patients using either ontological or generative perturbation.
        4. Predict the model's output for all neighbors.
        5. Train a decision tree to locally approximate the model's behavior.
        6. Extract and print the decision rules for the reference patient.
        7. Compute and return explainability metrics (fidelity and hit).

    Args:
        reference_index (int): Index of the patient to explain.
        ontological_perturbation (bool): Whether to use ontological perturbation for synthetic data.
        generative_perturbation (bool): Whether to use generative model for synthetic data.

    Returns:
        fidelity_synth (float): Fidelity of the decision tree to the black-box model on the synthetic neighborhood.
        hit_synth (float): Whether the decision tree matches the black-box prediction for the reference patient.
    """
    # Load DoctorXAI data
    ontology = pl.read_parquet(ontology_path)
    diagnoses = pl.read_parquet(diagnoses_path)
    ccs_data = pl.read_parquet(ccs_path)
    icd9_data = pl.read_parquet(icd_path)
    icd9_data = icd9_data.join(ontology.select(['icd9_id', 'icd_code', 'label']), on=['icd9_id', 'icd_code'], how='left').fill_null('No description')

    # Load diagnosis test sets
    diagnoses_test  = diagnoses.filter(pl.col('role') == 'test')

    unique_codes = diagnoses['icd9_id'].explode().unique().to_numpy()
    max_ccs_id = ccs_data['ccs_id'].max() + 1
    max_icd_id = unique_codes.max() + 1

    ontology_array = ontology[['icd9_id', 'parent_id']].to_numpy()
    # Used for patient distance computation
    gen.create_c2c_table(ontology_array, unique_codes)

    # Load the BiLSTM model for inference
    model = load_bilstm_for_inference(model_path, device)
    model.to(device)
    
    # Extract numpy arrays for all codes, counts, and positions
    icd_codes_all = list(diagnoses['icd9_id'].to_numpy())
    ccs_codes_all = list(diagnoses['ccs_id'].to_numpy())
    positions_all = list(diagnoses['position'].to_numpy())
    counts_all = list(diagnoses['count'].to_numpy())

    icd_codes_test = list(diagnoses_test['icd9_id'].to_numpy())
    positions_test = list(diagnoses_test['position'].to_numpy())
    counts_test = list(diagnoses_test['count'].to_numpy())

    if generative_perturbation:
        # Load the generative model for synthetic data creation
        filler, hole_prob, hole_token_id = load_llama2_for_generation(filler_path, device)
        conv_data = pl.read_parquet(generation_path).sort('out_id')
        zero_row  = pl.DataFrame({'icd9_id':0, 'out_id':0, 'ccs_id':0}, schema=conv_data.schema)
        conv_data = pl.concat([zero_row, conv_data])

        out_to_icd = conv_data['icd9_id'].to_numpy()

    # Find the k_reals+1 closest real neighbors to the reference patient
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

    # Collect the most similar real patients as neighbors
    for it in range(k_reals+1):
        real_neigh_icd.append(icd_codes_all[topk[it]])
        real_neigh_ccs.append(ccs_codes_all[topk[it]])
        real_neigh_counts.append(counts_all[topk[it]])
        real_neigh_positions.append(positions_all[topk[it]])

    # Compute how many synthetic samples to generate per real neighbor
    synthetic_multiply_factor = np.ceil(synthetic_neighborhood_size /k_reals).astype(int)
    
    if ontological_perturbation:
        # Generate synthetic neighbors by perturbing real neighbors using the ontology
        displacements, new_counts = gen.ontological_perturbation(real_neigh_icd, real_neigh_counts, synthetic_multiply_factor, keep_prob)
        while len(synt_neigh_icd) < synthetic_neighborhood_size:

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
        # Generate synthetic neighbors using the generative model
        new_neigh_icd = []
        new_neigh_ccs = []
        new_neigh_counts = []
        new_neigh_positions = []
        while len(new_neigh_icd) < synthetic_neighborhood_size:
            cursor = 0
            while cursor < len(real_neigh_icd):
                new_cursor = min(cursor + batch_size, len(real_neigh_icd))

                for _ in range(synthetic_multiply_factor):
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
                    # Reshape and sample new codes from the generative model's output
                    old_shape = gen_output.shape
                    gen_output = gen_output.reshape((-1, gen_output.shape[-1]))
                    gen_output = torch.softmax(gen_output, dim=-1)
                    new_codes = torch.multinomial(gen_output, 1)
                    new_codes = new_codes.reshape(old_shape[:-1])
                    new_codes = new_codes.cpu().numpy()
                    # Map generated output indices to ICD codes
                    new_icd = list(out_to_icd[new_codes])
                    
                    # Add only the generated codes, excluding padding codes
                    for i, pos in enumerate(real_neigh_positions[cursor:new_cursor]):
                        new_neigh_icd.append(new_icd[i][:len(pos)])

                    new_neigh_counts    += real_neigh_counts[cursor:new_cursor]
                    new_neigh_positions += real_neigh_positions[cursor:new_cursor]

                cursor = new_cursor
        synt_neigh_icd       += new_neigh_icd
        synt_neigh_ccs       += new_neigh_ccs
        synt_neigh_counts    += new_neigh_counts
        synt_neigh_positions += new_neigh_positions
        
    # Prepare the reference patient for inference
    batch = prepare_batch_for_inference_bilstm(
        [icd_codes_test[reference_index]],
        [counts_test[reference_index]],
        [positions_test[reference_index]],
        device
    )
    
    reference_output = model(batch.unpack())
    reference_output = reference_output[0][-1]
    reference_labels = reference_output.topk(topk_predictions, dim=-1).indices

    # Predict the model's output for all synthetic neighbors
    neigh_labels = np.empty((len(synt_neigh_icd), len(reference_labels), ), dtype=np.int32)
    cursor = 0
    while cursor < len(synt_neigh_icd):
        new_cursor = min(cursor+batch_size, len(synt_neigh_icd))
        batch = prepare_batch_for_inference_bilstm (
            synt_neigh_icd[cursor:new_cursor],
            synt_neigh_counts[cursor:new_cursor],
            synt_neigh_positions[cursor:new_cursor],
            device
        )
        outputs = model(batch.unpack())
        outputs = [x[-1] for x in outputs]
        outputs = torch.stack(outputs)
        batch_labels = outputs.topk(topk_predictions, dim=-1).indices
        batch_labels = batch_labels.cpu().numpy()

        neigh_labels[cursor:new_cursor, ] = batch_labels
        cursor = new_cursor

    # Convert neighborhood labels to one-hot encoding for tree training
    neigh_labels_onehot = OneHotEncoding(neigh_labels, max_ccs_id)

    # Prepare the reference patient encoding for the decision tree
    reference_enc = gen.ids_to_encoded(
        [icd_codes_test[reference_index]],
        [counts_test[reference_index]],
        max_icd_id,
        0.5
    )[0]

    # One-hot encode the reference label for the decision tree
    reference_label_encoded = OneHotEncoding([reference_labels.cpu().numpy()], max_ccs_id)

    # Train the decision tree to explain the model's prediction locally
    tree, tree_inputs_encoded, model_labels_encoded = explain_label(
        synt_neigh_icd, synt_neigh_counts, neigh_labels_onehot, max_icd_id, tree_train_fraction,
        reference_enc, reference_label_encoded
    )

    # Extract the decision path for the reference patient
    tree_path = tree.tree_.decision_path(reference_enc.reshape((1,-1))).indices
    features = tree.tree_.feature

    expl_labels = [features[i] for i in tree_path]
    expl_labels = [x for x in expl_labels if x >= 0]

    thresholds = tree.tree_.threshold
    thresholds = [thresholds[i] for i in tree_path if features[i] >= 0]

    # Build a DataFrame with the codes and thresholds used in the explanation
    df = pl.DataFrame({'icd9_id': expl_labels, 'threasholds': thresholds}).with_columns(icd9_id=pl.col('icd9_id').cast(pl.UInt32))
    df = df.join(icd9_data, left_on='icd9_id', right_on='icd9_id', how='left')

    print_patient(icd_codes_test[reference_index], counts_test[reference_index], ontology)
    print('\n')

    print('ccs predicted')
    labels = reference_labels.tolist()
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
    # For each ICD code in the decision path, print the corresponding visit and description
    for id in expl_labels:
        for it in range(len(icd_codes_test[reference_index])):
            icd = icd_codes_test[reference_index][it]
            if icd == id:
                visit = positions_test[reference_index][it] + 1

                infos = ontology.filter(pl.col('icd9_id') == icd)
                if len(infos) != 1:
                    raise ValueError('Should not happen')
                code = infos['icd_code'][0]
                label = infos['label'][0]

                print(f'At visit {(str(visit)+","): <3} code {code: <6} [{label}]')

    # Compute explainability metrics
    # Fidelity: F1 score between decision tree and black-box model on the synthetic neighborhood
    fidelity_synth = f1_score(y_true=model_labels_encoded, y_pred=tree.predict(tree_inputs_encoded), average='micro')
    # Hit: Whether the decision tree matches the black-box prediction for the reference patient
    hit_synth = 1 - distance.hamming(reference_label_encoded.reshape(-1), tree.predict(reference_enc.reshape(1,-1)).flatten())
    return fidelity_synth, hit_synth

if __name__ == '__main__':
    reference_index = 5 # Index of the patient to explain
    ontological_perturbation   = False
    generative_perturbation    = True
    fidelity_synth, hit_synth = main(reference_index, ontological_perturbation, generative_perturbation)
    print('Evaluating explainability metrics...')
    print(f'Fidelity on synthetic neighborhood: {fidelity_synth}')
    print(f'Hit on synthetic neighborhood: {hit_synth}')