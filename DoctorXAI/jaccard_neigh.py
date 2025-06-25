from itertools import combinations
import statistics
import polars as pl
import numpy as np
import pickle
import lib.generator as gen
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import LocalOutlierFactor
from Generator.llama2_model import *
import sys
import os

# update sys.path to include the path to the SETOR directory
path_to_add = "/home/gbenanti/Tesi_Benanti/SETOR"
if path_to_add not in sys.path:
    sys.path.append(path_to_add)

sys.path.append(os.path.abspath("../.."))

import torch

import random
import pandas as pd
from datetime import datetime
import argparse
import os

ontology_path = '../data/processed/ontology.parquet'
diagnoses_path = '../data/processed/diagnoses.parquet'
generation_path = '../data/processed/generation.parquet'
ccs_path = '../data/processed/ccs.parquet'
icd_path = '../data/processed/icd.parquet'

###### SET THESE BEFORE EXECUTION ######

parser = argparse.ArgumentParser(description="Set parameters for the script.")
parser.add_argument("--k_reals", type=int, default=50, help="Number of real neighbors to consider.")
parser.add_argument("--synt_neigh_size", type=int, default=200, help="Size of the synthetic neighborhood.")
parser.add_argument("--ont_perturbation", action="store_true", help="Enable ontological perturbation.")
parser.add_argument("--gen_perturbation", action="store_true", help="Enable generative perturbation.")
args = parser.parse_args()

ontological_perturbation = args.ont_perturbation
generative_perturbation = args.gen_perturbation

###### SET THESE BEFORE EXECUTION ######

# Path to the trained filler model
filler_path = 'results/filler-mpgzq-2025-02-19_19:17:59'

k_reals = args.k_reals
synthetic_neighborhood_size = args.synt_neigh_size
batch_size = 64
keep_prob = 0.75 # Probability to keep a code during ontological perturbation
llm_musk_prob = 0.25 # Probability to mask a code for generative perturbation
topk_predictions = 10
uniform_perturbation = False
tree_train_fraction = 0.75
synthetic_multiply_factor = 4
generative_multiply_factor = 4

APPLE_SILICON = False
# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
#############################################

random.seed(42)
np.random.seed(seed=42)
torch.manual_seed(seed=42)

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

def jaccard_similarity(set1, set2):
    """
    Calculate the Jaccard similarity between two sets.

    The Jaccard similarity is defined as the size of the intersection 
    divided by the size of the union of the two sets. It measures the 
    similarity between finite sets and is a value between 0 and 1.

    Args:
        set1 (set): The first set.
        set2 (set): The second set.

    Returns:
        float: The Jaccard similarity index, a value between 0.0 and 1.0. 
               Returns 0.0 if both sets are empty.
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def jaccard_stats(sets):
    """
    Calculate the mean and standard deviation of Jaccard similarities 
    for all unique pairs of sets.

    Args:
        sets (Iterable[set]): An iterable containing sets to compare.

    Returns:
        tuple: A tuple containing:
            - mean (float): The mean of the Jaccard similarities.
            - std_dev (float): The standard deviation of the Jaccard similarities.
              Returns 0 if there are fewer than two sets to compare.

    Notes:
        - The Jaccard similarity between two sets is a measure of their similarity,
          defined as the size of their intersection divided by the size of their union.
        - If no pairs of sets exist, the mean and standard deviation are both 0.
    """
    similarities = [
        jaccard_similarity(a, b)
        for a, b in combinations(sets, 2)
    ]
    mean = statistics.mean(similarities) if similarities else 0
    std_dev = statistics.stdev(similarities) if len(similarities) > 1 else 0
    return mean, std_dev

def jaccard_patient_stats(patient, sets):
    """
    Calculate the mean and standard deviation of Jaccard similarities 
    between a reference patient and a list of other patients.

    Args:
        patient (set): The set of codes for the reference patient.
        sets (Iterable[set]): An iterable containing sets to compare with the reference patient.

    Returns:
        tuple: A tuple containing:
            - mean (float): The mean of the Jaccard similarities.
            - std_dev (float): The standard deviation of the Jaccard similarities.
    """
    similarities = [jaccard_similarity(patient, s) for s in sets]
    mean = statistics.mean(similarities) if similarities else 0
    std_dev = statistics.stdev(similarities) if len(similarities) > 1 else 0
    return mean, std_dev

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

    # Merge icd9_to_id and icd9_to_desc, columns = ['icd9', 'description', 'icd9_id_setor', 'icd9_id_generator']
    setor_icd9_data = pl.DataFrame(icd9_to_desc).join(pl.DataFrame(icd9_to_id), on='icd9', how='left').fill_null(0).rename({'icd9_id': 'icd9_id_setor'})
    setor_icd9_data = setor_icd9_data.join(icd9_data, left_on='icd9', right_on='icd_code', how='full').fill_null(0).rename({'icd9_id' : 'icd9_id_generator'}).drop('icd_code', 'label')

    # Load base data and model
    diagnoses_test  = diagnoses.filter(pl.col('role') == 'test')

    unique_codes = diagnoses['icd9_id'].explode().unique().to_numpy()

    ontology_array = ontology[['icd9_id', 'parent_id']].to_numpy()
    # Used by compute_patient_distances
    gen.create_c2c_table(ontology_array, unique_codes)

    # Extract the numpy data

    icd_codes_all = list(diagnoses['icd9_id'].to_numpy())
    ccs_codes_all = list(diagnoses['ccs_id'].to_numpy())
    positions_all = list(diagnoses['position'].to_numpy())
    counts_all = list(diagnoses['count'].to_numpy())

    icd_codes_test = list(diagnoses_test['icd9_id'].to_numpy())
    counts_test = list(diagnoses_test['count'].to_numpy())

    if generative_perturbation:
        filler, hole_prob, hole_token_id = load_llama2_for_generation(filler_path, device)
        hole_prob = llm_musk_prob
        conv_data = pl.read_parquet(generation_path).sort('out_id')
        zero_row  = pl.DataFrame({'icd9_id':0, 'out_id':0, 'ccs_id':0}, schema=conv_data.schema)
        conv_data = pl.concat([zero_row, conv_data])

        out_to_icd = conv_data['icd9_id'].to_numpy()
        out_to_ccs = conv_data['ccs_id' ].to_numpy()
    
    total_jaccard_neigh = []
    total_jaccard_ref_neigh = []
    total_lof = []

    # Print test parameters
    if generative_perturbation:
        print(f'llm_musk_prob: {llm_musk_prob}')
        print(f'Generative perturbation: {generative_perturbation}')
    if ontological_perturbation:
        print(f'Ontological perturbation: {ontological_perturbation}')
        print(f'Keep prob: {keep_prob}')
    print(f'k_reals: {k_reals}')
    print(f'Synthetic neighborhood size: {synthetic_neighborhood_size}')
    
    for reference_index in range(interval[0], interval[1]):
        print(f'Processing patient {reference_index}/{interval[1]}...')

        # Find the closest neighbors in the real data
        distance_list = gen.compute_patients_distances (
            icd_codes_test[reference_index],
            counts_test[reference_index],
            icd_codes_all,
            counts_all
        )

        # topk includes the reference patient itself
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
            while len(synt_neigh_icd) < synthetic_neighborhood_size:
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

            while len(new_neigh_icd) < synthetic_neighborhood_size:
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
                        # batch_size = 64, seq_len = 62, out_features (top k most frequent icd9 codes) = 500
                        old_shape = gen_output.shape
                        gen_output = gen_output.reshape((-1, gen_output.shape[-1]))
                        gen_output = torch.softmax(gen_output, dim=-1)
                        gen_output.shape # 3968x500
                        new_codes = torch.multinomial(gen_output, 1)
                        new_codes = new_codes.reshape(old_shape[:-1])
                        new_codes.shape # 64x62
                        new_codes = new_codes.cpu().numpy()

                        new_icd = list(out_to_icd[new_codes])
                        new_ccs = list(out_to_ccs[new_codes])

                        # Add only generated codes, excluding padding codes
                        for i, pos in enumerate(real_neigh_positions[cursor:new_cursor]):
                            new_neigh_icd.append(new_icd[i][:len(pos)])
                            new_neigh_ccs.append(new_ccs[i][:len(pos)])

                        new_neigh_counts += real_neigh_counts[cursor:new_cursor]
                        new_neigh_positions += real_neigh_positions[cursor:new_cursor]

                    cursor = new_cursor

            synt_neigh_icd       += new_neigh_icd
            synt_neigh_ccs       += new_neigh_ccs
            synt_neigh_counts    += new_neigh_counts
            synt_neigh_positions += new_neigh_positions

        # Convert each visit to an array of codes (excluding zeros)
        visite_str = [np.array([code for code in visita if code != 0]) for visita in synt_neigh_icd]
        # Convert visits to sets
        sets = [set(visit) for visit in visite_str]

        # jaccard index computed with respect to the synthetic neighborhood
        mean, std_dev = jaccard_stats(sets)

        # jaccard index computed with respect to the reference patient
        #mean, std_dev = jaccard_patient_stats(set(icd_codes_test[reference_index]), sets)
        
        # Add the mean and standard deviation to the results list
        total_jaccard_neigh.append(mean)
        total_jaccard_ref_neigh.append(std_dev)

        # Compute LOF for the synthetic neighborhood
        
        # Convert each visit to a space-separated string of codes
        visite_str_for_vectorizer = [" ".join(str(code) for code in visit) for visit in visite_str]
        # CountVectorizer creates a frequency-based representation
        vectorizer = CountVectorizer()
        vectorized_visits = vectorizer.fit_transform(visite_str_for_vectorizer).toarray()
        lof = LocalOutlierFactor(n_neighbors=int(synthetic_neighborhood_size), metric='euclidean')
        lof_labels = lof.fit_predict(vectorized_visits)  # -1 = outlier, 1 = inlier
        #lof_scores = -lof.negative_outlier_factor_
        num_outliers = np.sum(lof_labels == -1)
        total_lof.append(num_outliers)

    jaccard_dist_mean = np.mean(total_jaccard_neigh)
    jaccard_dist_std = np.std(total_jaccard_neigh)
    lof_mean = np.mean(total_lof)
    lof_std = np.std(total_lof)
    total_lof = sum(total_lof)
    return jaccard_dist_mean, jaccard_dist_std, lof_mean, lof_std, total_lof

if __name__ == '__main__':
    
    interval = 0,1000  # Range of patients in the dataset to explain
    
    jaccard_dist_mean, jaccard_dist_std, lof_mean, lof_std, total_loaf = main(interval,
     ontological_perturbation, generative_perturbation)
    
    # Create directories for results
    generative_dir = "stats/jaccard_similarity/generative"
    ontological_dir = "stats/jaccard_similarity/ontological"
    os.makedirs(generative_dir, exist_ok=True)
    os.makedirs(ontological_dir, exist_ok=True)

    # Generate file name with date, k_reals, and synthetic_neighborhood_size
    current_date = datetime.now().strftime("%Y-%m-%d")
    file_name = f"results_{current_date}_k{k_reals}_synt{synthetic_neighborhood_size}.txt"

    # Determine the directory based on the type of test
    if generative_perturbation:
        file_path = os.path.join(generative_dir, file_name)
    elif ontological_perturbation:
        file_path = os.path.join(ontological_dir, file_name)
    else:
        file_path = os.path.join("results", file_name)

    # Print results to the console
    print(f"Jaccard index Mean: {jaccard_dist_mean}")
    print(f"Jaccard index Std Dev: {jaccard_dist_std}")
    print(f"LOF Mean: {lof_mean}")
    print(f"LOF Std Dev: {lof_std}")
    print(f"Total number of outliers: {total_loaf}")

    # Save results to a file
    with open(file_path, "w") as result_file:
        result_file.write(f"Jaccard index Mean: {jaccard_dist_mean}\n")
        result_file.write(f"Jaccard index Std Dev: {jaccard_dist_std}\n")
        result_file.write(f"LOF Mean: {lof_mean}\n")
        result_file.write(f"LOF Std Dev: {lof_std}\n")  
        result_file.write(f"Total number of outliers: {total_loaf}\n")

    print(f"Results saved to {file_path}")
