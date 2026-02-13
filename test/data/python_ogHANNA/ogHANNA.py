import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import types  
from Utils import predict, create_embedding_matrix
from chemberta_utils import initialize_ChemBERTA
import Own_Scaler

def register_pickle_compat_aliases():
    """Provide legacy module aliases required by the stored scaler pickle."""
    utils_module = types.ModuleType('utils')
    utils_module.Own_Scaler = types.ModuleType('Own_Scaler')
    utils_module.Own_Scaler.CustomScaler = Own_Scaler.CustomScaler
    sys.modules['utils'] = utils_module
    sys.modules['utils.Own_Scaler'] = utils_module.Own_Scaler

# HANNA Model
class HANNA(nn.Module):
    def __init__(self, Embedding_ChemBERT=384, nodes=96):
        super(HANNA, self).__init__()
        self.Embedding_ChemBERT = Embedding_ChemBERT 
        self.nodes = nodes 
        self.theta = nn.Sequential(nn.Linear(Embedding_ChemBERT, nodes), nn.SiLU())
        self.alpha = nn.Sequential(nn.Linear(nodes+2, nodes), nn.SiLU(), nn.Linear(nodes, nodes), nn.SiLU())
        self.phi = nn.Sequential(nn.Linear(nodes, nodes), nn.SiLU(), nn.Linear(nodes, 1))

    def forward(self, temperature, mole_fractions, E_i):
        # Determine batch_size (B) and number of components (N)
        batch_size, num_components, _ = E_i.shape # [B,N,E] E=384, ChemBERTa-2 embedding

        # Enable gradient tracking to use autograd
        E_i.requires_grad_(True)
        temperature.requires_grad_(True) # Standardized temperature
        mole_fractions.requires_grad_(True) # x_1

        # Calculate remaining mole fraction for the Nth component (here: N=2)
        mole_fraction_N = 1 - mole_fractions.sum(dim=1, keepdim=True) # x_2=1-x_1 [B,1]
        mole_fractions_complete = torch.cat([mole_fractions, mole_fraction_N], dim=1) # [x_1,1-x_1], [B,2]

        # Reshape mole fraction and temperature
        mole_fractions_complete_reshaped = mole_fractions_complete.unsqueeze(-1) # [B,N,1]
        T_reshaped = temperature.view(batch_size, 1, 1).expand(-1, num_components, 1) # [B,N,1]

        # Fine-tuning of the component embeddings
        theta_E_i = self.theta(E_i) # [B,N,nodes]

        # Calculate cosine similarity between the two components
        cosine_sim = F.cosine_similarity(theta_E_i[:, 0, :], theta_E_i[:, 1, :], dim=1) #[B]
        # Calculate cosine distance between the two components
        cosine_distance = 1 - cosine_sim # [B]

        # Concatenate embedding with T and x_i
        c_i = torch.cat([T_reshaped, mole_fractions_complete_reshaped, theta_E_i], dim=-1) #[B,N,nodes+2]
        alpha_c_i = self.alpha(c_i) # [B,N,nodes]
        c_mix = alpha_c_i.sum(dim=1) # [B,nodes]
        gE_NN = self.phi(c_mix).squeeze(-1) # [B]

        # Apply cosine similarity adjustment
        correction_factor_mole_fraction = torch.prod(mole_fractions_complete, dim=1) # [B] x1*(1-x1) term
        gE = gE_NN * correction_factor_mole_fraction * cosine_distance  # [B] Adjust gE_NN with the physical constraints and calculate gE/RT

        # Compute (dgE/dx1)/RT
        dgE_dx1 = torch.autograd.grad(gE.sum(), mole_fractions, create_graph=True)[0] # [B,1]

        # ln gamma_i equation (binary mixture). Unsqueeze to adjust dimension to [B,1] for gE/RT
        ln_gamma_1 = gE.unsqueeze(1) + (1 - mole_fractions) * dgE_dx1 # [B,1]
        ln_gamma_2 = gE.unsqueeze(1) - mole_fractions * dgE_dx1 # [B,1]
        # Concatenate the ln_gammas
        ln_gammas = torch.cat([ln_gamma_1, ln_gamma_2], dim=1) # [B,2]

        return ln_gammas, gE

def load_model_and_scaler(base_dir, device):
    model_path = os.path.join(base_dir, "ogHANNA_Val.pt")
    scaler_path = os.path.join(base_dir, "scaler_ogHANNA_Val.pkl")

    model = HANNA().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    register_pickle_compat_aliases()
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


def predict_binary_system(smiles_1, smiles_2, temperature, x1, model, scaler, chemberta_model, tokenizer, device):
    x1_values = np.array([x1], dtype=float)
    embedding_matrix = create_embedding_matrix(
        smiles_1,
        smiles_2,
        temperature,
        device,
        chemberta_model,
        tokenizer,
        x1_values,
    )
    _, ln_gammas_pred = predict(embedding_matrix, scaler, model, device)
    gamma_1 = float(np.exp(ln_gammas_pred[0, 0]))
    gamma_2 = float(np.exp(ln_gammas_pred[0, 1]))
    return gamma_1, gamma_2


def main():
    current_script_folder = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, scaler = load_model_and_scaler(current_script_folder, device)
    chemberta_model, tokenizer = initialize_ChemBERTA(device=device)

    # Define systems as [["SMILES1", "SMILES2"], ["S1", "S2"], ...]
    systems = [
        ["O", "CCO"],
        ["CC(=O)O", "O"],
        ["CC", "CCC"],
    ]

    temperature = 300.0
    x1 = 0.5

    print("REFERENCE (PYTHON ORIGINAL (ogHANNA))")
    print(f"T = {temperature} K, x1 = {x1}")

    for smiles_1, smiles_2 in systems:
        gamma_1, gamma_2 = predict_binary_system(
            smiles_1,
            smiles_2,
            temperature,
            x1,
            model,
            scaler,
            chemberta_model,
            tokenizer,
            device,
        )
        print(f"System: [{smiles_1}] + [{smiles_2}]")
        print(f"  Gamma First Comp.:  {gamma_1:.6f}")
        print(f"  Gamma Second Comp: {gamma_2:.6f}")


if __name__ == "__main__":
    main()
