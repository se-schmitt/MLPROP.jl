from chemberta_utils import initialize_ChemBERTA, get_smiles_embedding
import torch
import numpy as np

smiles_list = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "C1=CC=C(C=C1)C2=C(C3=C(C(=NNC3=O)[O-])C(=N2)Cl)N.[Na+]",
    "CC[Hg]N1C(=O)C2C(C1=O)C3(C(=C(C2(C3(Cl)Cl)Cl)Cl)Cl)Cl",
]

# Loading the ChemBERTa model
bert, tokenizer = initialize_ChemBERTA()

# Calculating the embedding for a given SMILES
for smiles in smiles_list:
    embedding = get_smiles_embedding(smiles, tokenizer, bert, torch.device('cpu'))
    np.savetxt(f"{smiles}.csv", embedding.flatten(), delimiter=",")
