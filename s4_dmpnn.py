# Molecule-to-graph featurization and the D-MPNN model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import Set2Set
from torch_geometric.utils import scatter
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, Lipinski

N_GLOBAL_FEATURES = 25


# Featurization: SMILES to graph

def build_base_graph_cache(smiles_list):
    cache = {}
    for s in set(smiles_list):
        g = smiles_to_graph(s, None)
        if g is not None:
            cache[s] = g
    return cache

# Builds the graph of one molecule; y (optional) is its 9 pIC50 values (used in training, caused data leakage).
def smiles_to_graph(smiles, y=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([get_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges, edge_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = get_bond_features(bond)
        edges += [[i, j], [j, i]]
        edge_features += [f, f]
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_feature_dim()), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    y_tensor = torch.tensor([y], dtype=torch.float) if y is not None else None
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor,
                global_features=get_global_features(mol).view(1, -1))


# One-hot encoding, with an extra trailing slot
def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


# The 29 features of one atom.
def get_atom_features(atom):
    features = []
    features += one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'])
    features.append(atom.GetDegree())
    features.append(atom.GetTotalNumHs())
    features.append(1 if atom.GetIsAromatic() else 0)
    features.append(1 if atom.IsInRing() else 0)
    features += one_hot_encoding(atom.GetHybridization(),
                                 [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3])
    features.append(atom.GetFormalCharge())
    features += one_hot_encoding(atom.GetChiralTag(),
                                 [Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW])
    for ring_size in (3, 4, 5, 6, 7):
        features.append(1 if atom.IsInRingSize(ring_size) else 0)
    features.append(atom.GetTotalValence())
    features.append(atom.GetNumRadicalElectrons())
    return features 


# The 10 features of one bond.
def get_bond_features(bond):
    features = one_hot_encoding(bond.GetBondType(), [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                                                     Chem.BondType.TRIPLE, Chem.BondType.AROMATIC])
    features.append(1 if bond.GetIsConjugated() else 0)
    features.append(1 if bond.IsInRing() else 0)
    features += one_hot_encoding(bond.GetStereo(), [Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE]) 
    return features 

def bond_feature_dim():
    return len(get_bond_features(Chem.MolFromSmiles('CC').GetBondWithIdx(0)))


# The 25 whole-molecule descriptors.
def get_global_features(mol):
    features = [Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.FractionCSP3(mol),
                Descriptors.MolMR(mol),
                GraphDescriptors.BertzCT(mol),
                GraphDescriptors.BalabanJ(mol),
                GraphDescriptors.Chi0v(mol),
                GraphDescriptors.Chi1v(mol),
                GraphDescriptors.Chi2v(mol),
                Lipinski.NumHeteroatoms(mol),
                Lipinski.HeavyAtomCount(mol),
                Descriptors.NumValenceElectrons(mol),
                Descriptors.NumRadicalElectrons(mol),
                Descriptors.LabuteASA(mol),
                GraphDescriptors.Kappa3(mol),
                Lipinski.RingCount(mol),
                Lipinski.NumSaturatedRings(mol),
                Lipinski.NumAromaticHeterocycles(mol),
                Lipinski.NumAliphaticHeterocycles(mol)]
    return torch.nan_to_num(torch.tensor(features, dtype=torch.float), nan=0.0, posinf=0.0, neginf=0.0)


# Builds the graphs for a list of molecules with their targets
def prepare_graph_data(smiles_list, y_list, base_cache=None):
    data_list = []
    for s, y in zip(smiles_list, y_list):
        if base_cache is not None and s in base_cache:
            graph = base_cache[s].clone()
            graph.y = torch.tensor([y], dtype=torch.float) if y is not None else None
        else:
            graph = smiles_to_graph(s, y)
        if graph is None:
            continue
        data_list.append(graph)
    return data_list


# Mean and std of each of the 25 global descriptors over the training graphs, used to standardize.
def fit_global_scaler(graphs):
    g = torch.stack([d.global_features.view(-1) for d in graphs], dim=0)
    return g.mean(dim=0), g.std(dim=0).clamp(min=1e-06)

def apply_global_scaler(graphs, scaler):
    mean, std = scaler
    dev = graphs[0].global_features.device
    mean, std = mean.to(dev), std.to(dev)
    for d in graphs:
        d.global_features = (d.global_features - mean) / std


# Training-time dataset with random augmentation, resampled on every access (so it differs each epoch).
class DynamicAugDataset(torch.utils.data.Dataset):

    def __init__(self, graphs, edge_drop_prob, node_mask_prob):
        self.graphs = graphs
        self.edge_drop_prob = edge_drop_prob
        self.node_mask_prob = node_mask_prob

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx].clone()
        if self.edge_drop_prob > 0 and g.edge_index.size(1) > 0:
            n_edges = g.edge_index.size(1) // 2  # edges are stored twice, once per direction
            keep = torch.rand(n_edges) >= self.edge_drop_prob
            full = keep.repeat_interleave(2)  # same decision for both directions of a bond
            if full.any():  # never drop every edge
                g.edge_index = g.edge_index[:, full]
                g.edge_attr = g.edge_attr[full]
        if self.node_mask_prob > 0 and g.x.shape[0] > 0:
            keep_nodes = (torch.rand(g.x.shape[0]) >= self.node_mask_prob).float().view(-1, 1)
            g.x = g.x * keep_nodes
        return g


# The D-MPNN model

class DMPNNModel(nn.Module):

    def __init__(self, node_in_features, edge_in_features, hidden_dim=128, out_dim=9,
                 num_global_features=N_GLOBAL_FEATURES, num_layers=3, conv_dropout=0.05,
                 fc_dropout=0.3, fc1_hidden=192, fc2_hidden=96, set2set_steps=3):
        super().__init__()
        self.depth = num_layers
        self.W_i = nn.Linear(node_in_features + edge_in_features, hidden_dim)  # initial edge state
        self.W_h = nn.Linear(hidden_dim, hidden_dim)  # message transform, per step
        self.W_o = nn.Linear(node_in_features + hidden_dim, hidden_dim)  # final atom state
        self.drop = nn.Dropout(conv_dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.readout = Set2Set(hidden_dim, processing_steps=set2set_steps)  # -> 2 * hidden_dim
        self.head = _Head(2 * hidden_dim, num_global_features, out_dim, fc1_hidden, fc2_hidden, fc_dropout)

    # Loads the self-supervised pretrained encoder weights (W_i, W_h, W_o, bn); the head is learned from scratch.
    def load_pretrained_encoder(self, path, device):
        state = torch.load(path, map_location=device)
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise ValueError(f'pretrained checkpoint has unknown tensors: {result.unexpected_keys}')

    def forward(self, x, edge_index, edge_attr, batch, global_features):
        N = x.size(0)
        s, t = edge_index[0].long(), edge_index[1].long()
        rev = self._reverse_index(edge_index, N)
        h0 = F.relu(self.W_i(torch.cat([x[s], edge_attr], dim=1)))
        h = h0
        for _ in range(self.depth):
            m_node = scatter(h, t, dim=0, dim_size=N, reduce='sum')  # messages arriving at each atom
            m_e = m_node[s] - h[rev]  # drop the reverse-edge contribution
            h = self.drop(F.relu(h0 + self.W_h(m_e)))  # residual connection
        m_node = scatter(h, t, dim=0, dim_size=N, reduce='sum')
        h_v = F.relu(self.bn(self.W_o(torch.cat([x, m_node], dim=1))))
        graph_emb = self.readout(h_v, batch)
        return self.head(graph_emb, global_features)
    
    @staticmethod
    def _reverse_index(edge_index, num_nodes):
        s, t = edge_index[0].long(), edge_index[1].long()
        key_f = s * num_nodes + t
        key_b = t * num_nodes + s
        order = torch.argsort(key_f)
        sk = key_f[order]
        pos = torch.searchsorted(sk, key_b).clamp(max=sk.numel() - 1)
        rev = order[pos]
        valid = sk[pos] == key_b
        idx = torch.arange(s.numel(), device=s.device)
        return torch.where(valid, rev, idx)


# Final head
class _Head(nn.Module):

    def __init__(self, graph_emb_dim, num_global_features, out_dim, fc1_hidden, fc2_hidden, fc_dropout):
        super().__init__()
        self.bn_global = nn.BatchNorm1d(num_global_features)
        self.fc1 = nn.Linear(graph_emb_dim + num_global_features, fc1_hidden)
        self.bn_fc1 = nn.BatchNorm1d(fc1_hidden)
        self.drop1 = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(fc1_hidden, fc2_hidden)
        self.bn_fc2 = nn.BatchNorm1d(fc2_hidden)
        self.drop2 = nn.Dropout(fc_dropout)
        self.fc3 = nn.Linear(fc2_hidden, out_dim)

    def forward(self, graph_emb, global_features):
        g = self.bn_global(global_features)
        x = torch.cat([graph_emb, g], dim=1)
        x = self.drop1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn_fc2(self.fc2(x))))
        return self.fc3(x)  # 9 pIC50 values, in standardized scale
