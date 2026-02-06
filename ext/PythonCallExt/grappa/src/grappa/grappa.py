# taken from https://github.com/marco-hoffmann/GRAPPA

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter
import numpy as np
from pathlib import Path
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

# --- Constants and cache ---
DEFAULT_TEMPERATURE = 293.15
_MODEL_CACHE = None

# --- GNN building blocks ---
class GraphAttentionPooling(nn.Module):
    def __init__(self, n_features, key_dim):
        super(GraphAttentionPooling, self).__init__()

        self.n_features = n_features

        self.query_weight = nn.Parameter(torch.Tensor(n_features, key_dim))
        self.key_weight = nn.Parameter(torch.Tensor(n_features, key_dim))
        self.value_weight = nn.Parameter(torch.Tensor(n_features, n_features))

        nn.init.xavier_uniform_(self.query_weight)
        nn.init.xavier_uniform_(self.key_weight)
        nn.init.xavier_uniform_(self.value_weight)

    def get_attention_scores(self, node_out, batch):
        _ , n_features = node_out.size()
        device = node_out.device

        Q = torch.matmul(node_out, self.query_weight) 
        K = torch.matmul(node_out, self.key_weight)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (n_features ** 0.5)
        mask = (batch.unsqueeze(1) == batch.unsqueeze(0)).to(device)
        attention_scores[~mask] = float('-inf')
        attention_scores = torch.softmax(attention_scores, dim=-1)

        return attention_scores

    def forward(self, node_out, batch):
        _, n_features = node_out.size()
        n_graphs = batch.max().item() + 1
        device = node_out.device

        V = torch.matmul(node_out, self.value_weight)

        attention_scores = self.get_attention_scores(node_out, batch)

        context_matrix = torch.matmul(attention_scores, V)
        return scatter(context_matrix.view(-1, n_features), batch, dim=0, dim_size=n_graphs, reduce='sum')

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, n_features, key_dim, num_pooling_heads):
        super(MultiHeadAttentionPooling, self).__init__()

        self.n_features = n_features
        self.num_heads = num_pooling_heads

        self.heads = nn.ModuleList([GraphAttentionPooling(n_features, key_dim) for _ in range(self.num_heads)])

    def forward(self, node_out, batch):
        head_outputs = [head(node_out, batch) for head in self.heads]
        return torch.mean(torch.stack(head_outputs), dim=0)

class GNN_GAT(nn.Module):
    def __init__(self, node_dim, edge_dim, conv_dim, heads=5, dropout=0.1, num_layers=3):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(node_dim, conv_dim, heads, edge_dim=edge_dim, dropout=dropout, concat=False, share_weights=True))

        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(conv_dim, conv_dim, heads, edge_dim=edge_dim, dropout=dropout, concat=False, share_weights=True))

    def forward(self, x, edge_index, edge_attr):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)

        return x

class ScaleOutput(nn.Module):
    def __init__(self, ranges):
        super(ScaleOutput, self).__init__()
        self.ranges = torch.tensor(ranges)
    def forward(self, x):
        scaled_output = torch.sigmoid(x)
        scaled_output = scaled_output * (self.ranges[:, 1] - self.ranges[:, 0]) + self.ranges[:, 0]
        return scaled_output

class Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, out_dim):
        super().__init__()
        layers = []
        layers.append(nn.BatchNorm1d(input_dim))
        layers.append(nn.Linear(input_dim, hidden_dim, bias=True))
        layers.append(nn.ELU())
        for _ in range(num_hidden_layers):
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ELU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)

class GRAPPA(nn.Module):
    def __init__(self, node_dim, edge_dim, conv_dim, hidden_dim, num_hidden_layers, 
                 dropout, num_gnn_layers=3, num_antoine_params = 3, gnn_heads = 5, pooling_heads = 1):
        super().__init__()
        self.gnn = GNN_GAT(node_dim, edge_dim, conv_dim, num_layers=num_gnn_layers, dropout=dropout, heads=gnn_heads)
        self.out_dim = conv_dim
        self.pooling_function = MultiHeadAttentionPooling(self.out_dim, key_dim=32, num_pooling_heads=pooling_heads) 
        self.head = Head(self.out_dim +2, hidden_dim, num_hidden_layers, num_antoine_params)
        self.parameter_scaler = ScaleOutput([[5.0,20.0], [1500.0, 6000.0], [-300.0, 0.0]])

    def get_antoine_parameters(self, x, edge_index, edge_attr, numHDonors, numHAcceptors, batch):
        gnn_out = self.gnn(x, edge_index, edge_attr)
        graph_out = self.pooling_function(gnn_out, batch)
        graph_out = torch.cat((graph_out, numHDonors.unsqueeze(1), numHAcceptors.unsqueeze(1)), dim=1)
        antoine_parameters = self.head(graph_out)
        antoine_parameters = self.parameter_scaler(antoine_parameters)
        return antoine_parameters

# --- Feature encoding ---
possible_atom_list = ['C','N','O','Cl','S','F','Br','I','P']
possible_hybridization = [Chem.rdchem.HybridizationType.S,
                          Chem.rdchem.HybridizationType.SP, 
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3]
possible_num_bonds = [0,1,2,3,4]
possible_num_Hs  = [0,1,2,3] 
possible_stereo  = [Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE]

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def atom_feature(atom):
    symbol        = atom.GetSymbol()
    Type_atom     = one_of_k_encoding(symbol, possible_atom_list)
    if atom.GetFormalCharge() != 0:
        raise Exception("Atom has formal charge!")
    if atom.GetNumRadicalElectrons() != 0:
        raise Exception("Atom has radical electrons!")
    Ring_atom     = [atom.IsInRing()]
    Aromaticity   = [atom.GetIsAromatic()]
    Hybridization = one_of_k_encoding(atom.GetHybridization(), possible_hybridization)
    Bonds_atom    = one_of_k_encoding(len(atom.GetNeighbors()), possible_num_bonds)
    num_Hs        = one_of_k_encoding(atom.GetTotalNumHs(), possible_num_Hs)

    results = Type_atom + Ring_atom + Aromaticity + Hybridization + Bonds_atom + num_Hs

    return np.array(results).astype(np.float32)

def bond_feature(bond):
    bt = bond.GetBondType()
    type_stereo = one_of_k_encoding(bond.GetStereo(), possible_stereo)
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()] + \
        type_stereo

    return np.array(bond_feats).astype(np.float32)

def mol_to_pyg(mol, temperature):
    if mol is None:
      return None
    if not any([atom.GetSymbol() == 'C' for atom in mol.GetAtoms()]):
        raise Exception("Molecule does not contain at least one carbon atom.")

    numHDonors = AllChem.CalcNumHBD(mol)
    numHAcceptors = AllChem.CalcNumHBA(mol)

    id_pairs = ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    atom_pairs = [z for (i, j) in id_pairs for z in ((i, j), (j, i))]
    bonds = (mol.GetBondBetweenAtoms(i, j) for (i, j) in atom_pairs)
    atom_features = np.array([atom_feature(a) for a in mol.GetAtoms()])
    bond_features = np.array([bond_feature(b) for b in bonds])
    d = Data(edge_index=torch.tensor(np.array(list(zip(*atom_pairs))), dtype=torch.int64),
             x=torch.FloatTensor(atom_features), 
             edge_attr=torch.FloatTensor(bond_features),
             numHAcceptors=torch.tensor([numHAcceptors]),
             numHDonors=torch.tensor([numHDonors]),
             temperature=torch.tensor([temperature]))
    d.validate()
    return d

# --- Model loading ---
def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        model = GRAPPA(24, 9, 32, 16, 3, 0.0, 4, 3, 2)
        # Load model weights from package data
        data_path = files('grappa').joinpath('data/GRAPPA_state_dict.pt')
        model.load_state_dict(torch.load(data_path, map_location='cpu', weights_only=True))
        model.eval()
        _MODEL_CACHE = model
    return _MODEL_CACHE

# --- Inference ---
class GRAPPAantoine(torch.nn.Module):
    def __init__(self):
        super(GRAPPAantoine, self).__init__()
        self.model = load_model()

    def forward(self, smiles: str):
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("smiles must be a non-empty string")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("invalid SMILES string")

        data = mol_to_pyg(mol, DEFAULT_TEMPERATURE)
        batch = torch.zeros(data.x.size(0), dtype=torch.long)
        prediction = self.model.get_antoine_parameters(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.numHDonors,
            data.numHAcceptors,
            batch,
        ).detach().numpy().tolist()
        
        return prediction[0]

def predict(smiles: str):
    return GRAPPAantoine()(smiles)