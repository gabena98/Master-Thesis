import pickle
import torch
# Caricamento
with open("outputs/data/ccs_cat1.dict", "rb") as f:
    ccs_cat1 = pickle.load(f)

ccs_cat1

with open("outputs/data/ccs_single_level.dict", "rb") as f:
    ccs_to_id = pickle.load(f)

ccs_to_id

with open("outputs/data/inputs.dict", "rb") as f:
    icd9_to_id = pickle.load(f)

icd9_to_id

with open("outputs/data/code2desc.dict", "rb") as f:
    icd9_to_desc = pickle.load(f)

icd9_to_desc

inputs = pickle.load(open("outputs/data/inputs.seqs", 'rb'))

inputs

# Caricamento
#loaded_config = torch.load("outputs/model/dx_alpha_0.01_lr_5e-05_bs_32_e_3.0_l_1.0_tr_0.2/SETOR_config.pth")

#print(loaded_config)

import copy
# Carica i pesi
model = torch.load("outputs/outputs/model/dx_alpha_0.01_lr_0.1_bs_32_e_100.0_l_1.0_tr_0.8/pytorch_model.bin", map_location=torch.device('cpu'))

""" # Stampa i layer e le forme dei pesi
for name, param in  model.state_dict().items():
    print(f"Layer: {name}, Shape: {param.shape}")
    print(param)  # Visualizza i valori """

print(model)