import polars as pl
import numpy as np
import pickle
import lib.generator as gen
from sklearn.metrics import f1_score
from scipy.spatial import distance
# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV, RandomizedSearchCV
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
from BiLSTM.BiLSTM_model import *

import torch
from torch.utils.data import DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from tqdm import tqdm

import random
import pandas as pd
from datetime import datetime
import argparse

ontology_path = '../data/processed/ontology.parquet'
diagnoses_path = '../data/processed/diagnoses.parquet'
generation_path = '../data/processed/generation.parquet'
ccs_path = '../data/processed/ccs.parquet'
icd_path = '../data/processed/icd.parquet'

###### IMPOSTARE PRIMA DELL'ESECUZIONE ######

parser = argparse.ArgumentParser(description="Set parameters for the script.")
parser.add_argument("--k_reals", type=int, default=50, help="Number of real neighbors to consider.")
parser.add_argument("--synt_neigh_size", type=int, default=200, help="Size of the synthetic neighborhood.")
parser.add_argument("--ont_perturbation", action="store_true", help="Enable ontological perturbation.")
parser.add_argument("--gen_perturbation", action="store_true", help="Enable generative perturbation.")
parser.add_argument("--SETOR", action="store_true", help="Use SETOR model.")
parser.add_argument("--BILSTM", action="store_true", help="Use BiLSTM model.")
args = parser.parse_args()

ontological_perturbation = args.ont_perturbation
generative_perturbation = args.gen_perturbation
SETOR = args.SETOR
BILSTM = args.BILSTM

###### IMPOSTARE PRIMA DELL'ESECUZIONE ######

# Path to the configuration file
if SETOR:
    config_path = 'explain_config.toml'
    output_path = 'results/explainer.txt'

else:
    config_path = 'BiLSTM_config.toml'
    output_path = 'results/explainer.txt'
    model_path = 'results/BiLSTM-udayi-2025-04-05_20:12:13'

# Path to the filler model
filler_path = 'results/filler-mpgzq-2025-02-19_19:17:59'

# new filler
#filler_path = 'results/filler-asqvj-2025-06-04_12:16:57'

k_reals = args.k_reals
synthetic_neighborhood_size = args.synt_neigh_size
batch_size = 64
keep_prob = 0.80 # for ontological perturbation
llm_musk_prob = 0.20 # for generative perturbation
topk_predictions = 10
uniform_perturbation = False
tree_train_fraction = 0.75
#synthetic_multiply_factor = 4
generative_multiply_factor = 4

APPLE_SILICON = False
# modificare il device in base alla disponibiltà di una GPU
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

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
    config_path = os.path.join(path, 'config.toml')
    with open(config_path, 'r') as f:
        config = tomlkit.loads(f.read())
    return config['model']['device']

def print_patient(ids: np.ndarray, cnt: np.ndarray, ontology: pl.DataFrame):
    """
    Prints patient visit information based on ICD-9 codes and their descriptions.

    Args:
        ids (np.ndarray): An array of ICD-9 code identifiers.
        cnt (np.ndarray): An array where each element represents the number of codes in a specific visit.
        ontology (pl.DataFrame): A DataFrame containing the mapping of ICD-9 codes to their descriptions. 
                                 It should have at least two columns: 'icd9_id' and 'label'.

    Behavior:
        - The function joins the `ids` array with the `ontology` DataFrame to retrieve the corresponding labels.
        - It iterates through the visits, as defined by the `cnt` array, and prints the ICD-9 codes and their labels for each visit.
        - Each visit is printed with a header (e.g., "visit 1") followed by the list of codes and their descriptions.

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
    Fits a decision tree classifier to explain a specific label based on neighborhood data and reference inputs.

    Parameters:
    -----------
    neigh_icd : list
        A list of ICD codes for the neighboring patients.
    neigh_counts : list
        A list of counts corresponding to the ICD codes for the neighboring patients.
    labels : list
        A list of labels corresponding to the neighboring patients.
    max_icd_id : int
        The maximum ICD code ID used for encoding.
    tree_train_fraction : float
        The fraction of the data to use for training the decision tree.
    reference_patient : array-like
        Encoded representation of the reference patient to be included in the training set.
    reference_label : array-like
        Label of the reference patient to be included in the training set.

    Returns:
    --------
    tree_classifier : DecisionTreeClassifier
        The trained decision tree classifier.
    tree_inputs_eval : array-like
        Encoded inputs for the evaluation set.
    labels_eval : array-like
        Labels corresponding to the evaluation set.

    Raises:
    -------
    Exception
        If an error occurs during the encoding of the dataset, an exception is raised with an error message.

    Notes:
    ------
    - The function uses a temporal encoding method for the input data.
    - The reference patient and label are appended to the training set to ensure their inclusion in the model.
    """
    # Tree fitting
    try:
        """ new_neigh_icd = []
        new_neigh_counts = []
        for visit, count in zip(neigh_icd, neigh_counts):
            new_neigh_icd.append(visit[:-int(count[-1])])
            new_neigh_counts.append(count[:-1]) """
        # This takes the list of patients to encode (ids and counts) and the max id, then returns their encoded form with temporal encoding
        tree_inputs = gen.ids_to_encoded(neigh_icd, neigh_counts, max_icd_id, 0.5)
    except Exception as e:
        print(f"Errore durante la codifica del dataset: {e}")
        #print(f"Dati input - neigh_ccs: {neigh_ccs[0]}, neigh_counts: {neigh_counts[0]}, max_ccs_id: {max_ccs_id}")

    tree_classifier = DecisionTreeClassifier(random_state=42)

    train_split = int(tree_train_fraction * len(labels))

    tree_inputs_train = tree_inputs[:train_split]
    tree_inputs_eval  = tree_inputs[train_split:]
    labels_train      = labels[:train_split]
    labels_eval       = labels[train_split:]
    tree_inputs_train = np.vstack([tree_inputs_train, reference_patient])
    labels_train = np.vstack([labels_train, reference_label])

    #tree_classifier.fit(tree_inputs_train, labels_train)

    param_grid = {
        'max_depth': [ 2, 8, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    """ search = HalvingGridSearchCV(
        estimator = tree_classifier,
        param_grid = param_grid,
        factor = 2,  # ogni iterazione dimezza il numero di candidati
        resource = 'n_samples',
        max_resources = 'auto',
        scoring = 'f1_micro',
        random_state = 42
    ) """

    """ search = GridSearchCV(
        estimator = tree_classifier,
        param_grid = param_grid,
        cv = 5,
        scoring = 'f1_micro',
    ) """

    search = RandomizedSearchCV(
        estimator=tree_classifier,
        param_distributions=param_grid,
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
    Preprocesses data for the SETOR model by splitting it into training, validation, 
    and test sets, and creating corresponding data loaders.

    Args:
        pids (list): List of patient IDs.
        inputs (object): Input sequences, typically containing patient visit data.
        labels (object): Labels for the next visit predictions.
        labels_visit (object): Labels for visit-level classification.
        config (object): Configuration object containing model parameters such as 
                         `num_ccs_classes` and `num_visit_classes`.

    Returns:
        tuple: A tuple containing:
            - train_data_loader (DataLoader): DataLoader for the training set.
            - val_data_loader (DataLoader): DataLoader for the validation set.
            - test_data_loader (DataLoader): DataLoader for the test set.
    """
    # SETOR preprocessing
    # inputs = inputs.seqs, labels = labels_next_visit.label, labels_visit = labels_visit_cat1.label
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
    Converts SETOR output into a format compatible with llama2's input requirements.

    This function processes the output of the SETOR model and maps its codes to the corresponding
    llama2 CCS (Clinical Classifications Software) codes using a provided mapping DataFrame.

    Args:
        setor_output (list): A nested list representing the output of the SETOR model. 
            The structure is expected to be a list of patients, where each patient is a list of visits,
            and each visit is a list of codes.
        setor_to_llama2_ccs (pandas.DataFrame): A DataFrame containing the mapping between SETOR codes
            and llama2 CCS codes. It must have at least two columns:
            - 'ccs_id_generator': The SETOR code.
            - 'ccs_id_setor': The corresponding llama2 CCS code.
        batch_size (int): The batch size used for processing. This parameter is currently unused
            in the function but may be relevant for future extensions.

    Returns:
        list: A nested list structure similar to `setor_output`, but with SETOR codes replaced
        by their corresponding llama2 CCS codes. The structure is:
            - List of patients
                - List of visits
                    - List of llama2 CCS codes for each visit.

    Notes:
        - The function assumes that the mapping in `setor_to_llama2_ccs` is complete and that every
          code in `setor_output` has a corresponding entry in the DataFrame.
        - The `.values` attribute of the DataFrame is used to extract the mapped codes, which may
          result in NumPy arrays being included in the output.

    TODO:
        - Complete the implementation based on the exact structure of the SETOR output.
        - Handle cases where a code in `setor_output` does not have a corresponding mapping in
          `setor_to_llama2_ccs`.
    """

    setor_ccs_codes = []
    for patient in setor_output:
        # TODO: da completare in base alla struttura dell'output di SETOR
        setor_visits = []
        for visit in patient:
            setor_codes = []
            for codes in visit:
                setor_codes.append(setor_to_llama2_ccs.loc[setor_to_llama2_ccs['ccs_id_generator'] == codes, 'ccs_id_setor'].values)
            setor_visits.append(setor_codes)
        setor_ccs_codes.append(setor_visits) 

def llama2_to_setor_batch(icd_codes, positions, setor_to_llama2_icd, batch_size=1):
    """
    Converts ICD codes from the llama2 format to the SETOR format in batches and prepares
    the data for use in a PyTorch DataLoader.

    Args:
        icd_codes (list of lists): A list where each element is a list of ICD codes 
            for a specific visit in the llama2 format.
        positions (list of lists): A list where each element is a list of positions 
            corresponding to the ICD codes for a specific visit.
        setor_to_llama2_icd (dict): A dictionary containing the mapping between llama2 
            and SETOR ICD codes. It should have two keys:
            - 'icd9_id_generator': List of llama2 ICD codes.
            - 'icd9_id_setor': List of corresponding SETOR ICD codes.
        batch_size (int, optional): The size of each batch for the DataLoader. 
            Defaults to 1.

    Returns:
        iterator: An iterator over the PyTorch DataLoader that yields batches of 
        processed data.

    Notes:
        - The function maps llama2 ICD codes to SETOR ICD codes using the provided 
          mapping dictionary.
        - It groups the mapped ICD codes by their positions, excluding the last 
          unique position.
        - The resulting grouped data is used to create a custom dataset and 
          DataLoader for batch processing.
    """
    # Creazione di un dizionario per la mappatura
    mapping_dict = dict(zip(setor_to_llama2_icd['icd9_id_generator'], setor_to_llama2_icd['icd9_id_setor']))
    filtered_groups = []

    for visit, position in zip(icd_codes, positions):
        # Sostituzione dei icd_id llama2 con icd_id setor
        setor_ids = [mapping_dict[i] for i in visit]

        # Raggruppamento per posizione (considerando che visit e position potrebbero avere lunghezze diverse)
        min_length = min(len(setor_ids), len(position))
        setor_ids = setor_ids[:min_length]
        position = position[:min_length]

        filtered_group = [np.array(setor_ids)[np.array(position) == pos].tolist() for pos in np.unique(np.array(position))[:]]
        filtered_groups.append(filtered_group)

    # Creazione di un tensore per la maschera di visita
    batch_dataset = llama2Dataset(filtered_groups)
    batch_data_loader = DataLoader(batch_dataset, batch_size=batch_size,
                                    collate_fn=lambda batch: collate_llama2(batch),
                                    num_workers=0, shuffle=True)

    return iter(batch_data_loader)

def load_setor_predictor(device):
    """
    Loads the SETOR predictor model and prepares it for evaluation.

    Args:
        device (torch.device): The device (CPU or GPU) on which the model will be loaded.

    Returns:
        NextDxPrediction: An instance of the SETOR model loaded with the specified configuration 
        and weights, set to evaluation mode.

    Notes:
        - The function assumes the presence of specific configuration and weight files in the 
          relative paths provided.
        - The model configuration is loaded from 
          "../../SETOR/outputs/outputs/model/dx_alpha_0.01_lr_0.1_bs_32_e_100.0_l_1.0_tr_0.8/SETOR_config.pth".
        - The model weights are loaded from 
          "../../SETOR/outputs/outputs/model/dx_alpha_0.01_lr_0.1_bs_32_e_100.0_l_1.0_tr_0.8/pytorch_model_weights_only.bin".
        - The model is set to evaluation mode after loading.
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
    Transforms the output of a model to match the format of the DoctorXAI model.

    This function maps the predicted labels from the SETOR model to the corresponding
    labels in the llama2 format using a mapping provided in `setor_ccs_data`. The transformed
    predictions are then returned as a PyTorch tensor on the specified device.

    Args:
        labels (list of torch.Tensor): A list of tensors containing the predicted labels
            from the SETOR model. Each tensor is expected to be on the CPU.
        setor_ccs_data (dict): A dictionary containing the mapping between SETOR CCS IDs
            and llama2 CCS IDs. It must have the keys:
                - 'ccs_id_setor': A list of SETOR CCS IDs.
                - 'ccs_id_generator': A list of corresponding llama2 CCS IDs.
        device (torch.device): The device to which the output tensor should be moved.

    Returns:
        torch.Tensor: A tensor containing the transformed predictions in the llama2 format,
        moved to the specified device.
    """

    # Transforming the output of the model to the format of the DoctorXAI model
    setor_to_llama2_ccs = dict(zip(setor_ccs_data['ccs_id_setor'], setor_ccs_data['ccs_id_generator']))
    predictions = [] 
    for pred in labels:
        pred = pred.cpu().numpy() # topk_predictions
        pred = [setor_to_llama2_ccs[id] for id in pred]
        predictions.append(pred)
    #predictions = np.stack(predictions)
    return torch.tensor(predictions).to(device)

def compare_models(model1, model2):
    """
    Compares two PyTorch models to check if they have identical weights and parameter keys.

    Args:
        model1 (torch.nn.Module): The first PyTorch model to compare.
        model2 (torch.nn.Module): The second PyTorch model to compare.

    Returns:
        bool: True if the models have identical parameter keys and weights, False otherwise.

    Notes:
        - The function first checks if the keys (parameter names) in the state dictionaries
          of the two models are identical.
        - If the keys match, it then checks if the corresponding weights in the state
          dictionaries are identical using `torch.allclose`.
        - If any mismatch is found, the function prints a message indicating the issue
          and returns False.
        - If all checks pass, the function prints a success message and returns True.
    """
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # Controlla che abbiano le stesse chiavi (nomi dei parametri)
    if state_dict1.keys() != state_dict2.keys():
        print("Le chiavi degli state_dict non corrispondono.")
        return False

    # Controlla se i pesi sono identici
    for key in state_dict1:
        if not torch.allclose(state_dict1[key], state_dict2[key]):
            print(f"Il parametro {key} è diverso tra i modelli.")
            return False

    print("I due modelli hanno gli stessi pesi!")
    return True

def OneHotEncoding(data, max_ccs_id=281):
    """
    Performs one-hot encoding on a list of integer indices.

    Args:
        data (list of int): A list of integer indices where each index corresponds 
                            to the position to be set to 1 in the one-hot encoded vector.

    Returns:
        numpy.ndarray: A 1D array where each row is a one-hot encoded vector of size 281.
                       The length of the array matches the length of the input data.

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

def main(interval, ontological_perturbation, generative_perturbation):
    """
    Main function for processing medical data, generating synthetic data, and explaining predictions using decision trees.
    Args:
        interval (int): interval of the index of the reference patients in the test dataset.
        ontological_perturbation (bool): Whether to apply ontological perturbation to the neighborhood data.
        generative_perturbation (bool): Whether to apply generative perturbation to the neighborhood data.
    Returns:
        tuple: A tuple containing:
            - fidelity_synth (float): Fidelity of the decision tree to the black-box model on the synthetic neighborhood.
            - hit_synth (float): Hit rate of the decision tree on the synthetic neighborhood.
    Raises:
        ValueError: If unexpected conditions occur during processing, such as missing or inconsistent data.
    Notes:
        - This function loads and processes data from multiple sources, including SETOR and DoctorXAI datasets.
        - It computes distances between patients, generates synthetic neighbors, and explains predictions using a decision tree.
        - The function also prints detailed information about the reference patient, predicted labels, decision rules, and relevant codes.
    """
    # Load SETOR data
    labels = pickle.load(open("../../SETOR/outputs/data/labels_next_visit.label", 'rb'))

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

    # Merging ccs_to_id and ccs_data, columns = ['ccs', 'description', 'ccs_id_generator', 'ccs_id_setor']
    setor_ccs_data = ccs_data.join(pl.DataFrame(ccs_to_id), on='ccs', how='left').fill_null(ccs_data.shape[0]).rename({'ccs_id_right': 'ccs_id_setor','ccs_id': 'ccs_id_generator'})

    # Merging icd9_to_id and icd9_to_desc, columns = ['icd9', 'description', 'icd9_id_setor', 'icd9_id_generator']
    setor_icd9_data = pl.DataFrame(icd9_to_desc).join(pl.DataFrame(icd9_to_id), on='icd9', how='left').fill_null(0).rename({'icd9_id': 'icd9_id_setor'})
    setor_icd9_data = setor_icd9_data.join(icd9_data, left_on='icd9', right_on='icd_code', how='full').fill_null(0).rename({'icd9_id' : 'icd9_id_generator'}).drop('icd_code', 'label')
    
    # Load base data and model

    if SETOR:
        # Load SETOR model
        setor = load_setor_predictor(device)
    else:
        # Load the model
        model = load_bilstm_for_inference(model_path, device)

    diagnoses_test  = diagnoses.filter(pl.col('role') == 'test')

    unique_codes = diagnoses['icd9_id'].explode().unique().to_numpy()
    max_ccs_id = ccs_data['ccs_id'].max() + 1
    max_icd_id = unique_codes.max() + 1

    ontology_array = ontology[['icd9_id', 'parent_id']].to_numpy()
    # viene usato da compute_patient_distances
    gen.create_c2c_table(ontology_array, unique_codes)


    # Extract the numpy data

    icd_codes_all = list(diagnoses['icd9_id'].to_numpy())
    ccs_codes_all = list(diagnoses['ccs_id'].to_numpy())
    positions_all = list(diagnoses['position'].to_numpy())
    counts_all = list(diagnoses['count'].to_numpy())

    icd_codes_test = list(diagnoses_test['icd9_id'].to_numpy())
    positions_test = list(diagnoses_test['position'].to_numpy())
    counts_test = list(diagnoses_test['count'].to_numpy())

    if generative_perturbation:
        filler, hole_prob, hole_token_id = load_llama2_for_generation(filler_path, device)
        hole_prob = llm_musk_prob
        conv_data = pl.read_parquet(generation_path).sort('out_id')
        zero_row  = pl.DataFrame({'icd9_id':0, 'out_id':0, 'ccs_id':0}, schema=conv_data.schema)
        conv_data = pl.concat([zero_row, conv_data])

        out_to_icd = conv_data['icd9_id'].to_numpy()
    
    fidelity_synth_sum = 0
    hit_synth_sum = 0
    fidelity_synth_squared_sum = 0
    hit_synth_squared_sum = 0

    #print test parameters
    print(f'SETOR: {SETOR}')
    print(f'BILSTM: {BILSTM}')
    print(f'Ontological perturbation: {ontological_perturbation}')
    print(f'Generative perturbation: {generative_perturbation}')
    print(f'k_reals: {k_reals}')
    print(f'Synthetic neighborhood size: {synthetic_neighborhood_size}')
    
    for reference_index in range(interval[0], interval[1]):
        print(f'Processing patient {reference_index}/{interval[1]}...')

        # Find closest neighbours in the real data
        distance_list = gen.compute_patients_distances (
            icd_codes_test[reference_index],
            counts_test[reference_index],
            icd_codes_all,
            counts_all
        )

        # topk include the reference patient
        topk = np.argpartition(distance_list, k_reals+1)[:k_reals+1]

        real_neigh_icd       = []
        real_neigh_ccs       = []
        real_neigh_counts    = []
        real_neigh_positions = []
        synt_neigh_icd       = []
        synt_neigh_ccs       = []
        synt_neigh_counts    = []
        synt_neigh_positions = []

        # prendo i 50 sample più simili al paziente di riferimento
        for it in range(k_reals+1):
            real_neigh_icd.append(icd_codes_all[topk[it]])
            real_neigh_ccs.append(ccs_codes_all[topk[it]])
            real_neigh_counts.append(counts_all[topk[it]])
            real_neigh_positions.append(positions_all[topk[it]])

        # augment the neighbours with some synthetic points
        synthetic_multiply_factor = np.ceil(synthetic_neighborhood_size /k_reals).astype(int)

        if ontological_perturbation:
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
                        # batch_size = 64, seq_len = 62, out_features (top k codici icd9 più frequenti) = 500
                        old_shape = gen_output.shape
                        gen_output = gen_output.reshape((-1, gen_output.shape[-1]))
                        gen_output = torch.softmax(gen_output, dim=-1)
                        gen_output.shape # 3968x500
                        new_codes = torch.multinomial(gen_output, 1)
                        new_codes = new_codes.reshape(old_shape[:-1])
                        new_codes.shape # 64x62
                        new_codes = new_codes.cpu().numpy()
                        # old generator
                        new_icd = list(out_to_icd[new_codes])
                        #new_ccs = list(out_to_ccs[new_codes])
                        # new generator
                        #new_icd = [np.array(codes, dtype=np.uint32) for codes in new_codes]
                        # aggiungo solo i codici generati, escludendo i codici di padding
                        for i, pos in enumerate(real_neigh_positions[cursor:new_cursor]):
                            new_neigh_icd.append(new_icd[i][:len(pos)])
                            #new_neigh_ccs.append(new_ccs[i][:len(pos)])

                        new_neigh_counts += real_neigh_counts[cursor:new_cursor]
                        new_neigh_positions += real_neigh_positions[cursor:new_cursor]

                    cursor = new_cursor

            synt_neigh_icd       += new_neigh_icd
            synt_neigh_ccs       += new_neigh_ccs
            synt_neigh_counts    += new_neigh_counts
            synt_neigh_positions += new_neigh_positions

        if SETOR:    
            # converting llama2 data to setor data
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

            # converting setor ccs id codes to llama2 ccs id codes
            reference_labels = setor_to_llama2_output([labels], setor_ccs_data, device).reshape(-1)
        else:
            # converting llama2 data to setor data
            batch = prepare_batch_for_inference_bilstm(
                [icd_codes_test[reference_index]],
                [counts_test[reference_index]],
                [positions_test[reference_index]],
                device
            )
            
            reference_output = model(batch.unpack())
            reference_output = reference_output[0][-1]
            reference_labels = reference_output.topk(topk_predictions, dim=-1).indices

        # Black Box Predictions on neighbours
        neigh_labels = np.empty((len(synt_neigh_icd), len(reference_labels), ), dtype=np.int32)

        if SETOR:
            setor_batch = llama2_to_setor_batch(synt_neigh_icd, synt_neigh_positions, setor_icd9_data, batch_size=batch_size)
            for iter, batch in enumerate(setor_batch):
                batch = {k: t.to(device) for k, t in batch.items()}
                setor_output, _, _  = setor(batch["input"], batch["visit_mask"], batch["code_mask"], output_attentions=True)
                
                # prendo l'ultima riga(true) di ogni matrice di output
                outputs = [x[-1] for x in setor_output]
                # batch_size 64 x ccs 281
                outputs = torch.stack(outputs)
                # prendo i topk indici delle categorie ccs
                batch_labels = outputs.topk(topk_predictions, dim=-1).indices
                batch_labels = setor_to_llama2_output(batch_labels, setor_ccs_data, device)
                #batch_labels = (batch_labels == reference_llama2_label[:,None,None]).any(-1)
                batch_labels = batch_labels.cpu().numpy()

                neigh_labels[iter*batch_size:(iter+1)*batch_size, :] = batch_labels
        else:
            cursor = 0
            while cursor < len(synt_neigh_icd):
                new_cursor = min(cursor+batch_size, len(synt_neigh_icd))
                batch   = prepare_batch_for_inference_bilstm (
                    synt_neigh_icd[cursor:new_cursor],
                    synt_neigh_counts[cursor:new_cursor],
                    synt_neigh_positions[cursor:new_cursor],
                    device
                )
                # batch_size = 64, seq_len = 2, num_ccs = 281
                outputs = model(batch.unpack())
                # prendo l'ultima riga(true) di ogni matrice di output
                outputs = [x[-1] for x in outputs]
                # batch_size 64 x ccs 281
                outputs = torch.stack(outputs)
                # prendo i topk indici delle categorie ccs
                batch_labels = outputs.topk(topk_predictions, dim=-1).indices
                batch_labels = batch_labels.cpu().numpy()

                neigh_labels[cursor:new_cursor, ] = batch_labels
                cursor = new_cursor

        # converting neigh_labels to one-hot encoding
        neigh_labels_onehot = OneHotEncoding(neigh_labels, max_ccs_id)

        # explanation
        reference_enc = gen.ids_to_encoded(
            [icd_codes_test[reference_index]],
            [counts_test[reference_index]],
            max_icd_id,
            0.5
        )[0]

        # adding one-hot encoding for tree labels
        reference_label_encoded = OneHotEncoding([reference_labels.cpu().numpy()], max_ccs_id)

        # l'albero viene allenato sull'encoding temporale delle visite dei pazienti, come spiegato nel paper
        tree, tree_inputs_encoded, model_labels_encoded = explain_label(synt_neigh_icd, synt_neigh_counts, neigh_labels_onehot, max_icd_id, tree_train_fraction,
            reference_enc, reference_label_encoded)

        # fidelity of the DT to the black box on the synthetic neighborhood
        fidelity_synth = f1_score(y_true=model_labels_encoded, y_pred=tree.predict(tree_inputs_encoded), average='micro')
        #hit of the DT
        hit_synth = 1 - distance.hamming(reference_label_encoded.reshape(-1), tree.predict(reference_enc.reshape(1,-1)).flatten())
        fidelity_synth_sum += fidelity_synth
        hit_synth_sum += hit_synth

        # Track squared sums for variance calculation
        fidelity_synth_squared_sum += fidelity_synth ** 2
        hit_synth_squared_sum += hit_synth ** 2
        
    mean_fidelity_synth = fidelity_synth_sum / ((interval[1] - interval[0]))
    mean_hit_synth = hit_synth_sum / ((interval[1] - interval[0]))

    # Calculate variance
    fidelity_synth_variance = (fidelity_synth_squared_sum / (interval[1] - interval[0])) - (mean_fidelity_synth ** 2)
    hit_synth_variance = (hit_synth_squared_sum / (interval[1] - interval[0])) - (mean_hit_synth ** 2)

    return mean_fidelity_synth, mean_hit_synth, fidelity_synth_variance, hit_synth_variance

if __name__ == '__main__':
    
    interval = 0,1000  # which patient of the dataset to explain
    
    fidelity_synth, hit_synth, fidelity_synth_variance, hit_synth_variance = main(interval, ontological_perturbation, generative_perturbation)
    
    print('Evaluating explainability metrics...')
    print(f'Fidelity on synthetic neighborhood: {fidelity_synth} (Variance: {fidelity_synth_variance})')
    print(f'Hit on synthetic neighborhood: {hit_synth} (Variance: {hit_synth_variance})')
    
    # Determine the directory based on the type of perturbation and model used
    if SETOR:
        model_directory = 'setor'
    else:
        model_directory = 'bilstm'

    if ontological_perturbation:
        directory = f'stats/{model_directory}/ontological'
    elif generative_perturbation:
        directory = f'stats/{model_directory}/generative'
    else:
        directory = f'stats/{model_directory}/none'
    
    os.makedirs(directory, exist_ok=True)
    
    # Get the current date
    current_date = datetime.now().strftime('%Y-%m-%d')

    # save results to file
    filename = f'results_{interval}_{current_date}_k{k_reals}_synt{synthetic_neighborhood_size}_ont{int(ontological_perturbation)}_gen{int(generative_perturbation)}.txt'
    with open(f'{directory}/{filename}', 'w') as f:
        f.write(f'Fidelity on synthetic neighborhood: {fidelity_synth} (Variance: {fidelity_synth_variance})\n')
        f.write(f'Hit on synthetic neighborhood: {hit_synth} (Variance: {hit_synth_variance})\n')
    print(f'Results saved to {directory}/{filename}')
