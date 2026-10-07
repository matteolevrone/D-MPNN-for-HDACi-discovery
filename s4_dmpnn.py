# =====================================================================================
# PASSO 4 di 7 — DALLA MOLECOLA AL GRAFO, E IL MODELLO D-MPNN
#
# Ruolo nella pipeline: definisce COSA vede la rete neurale e COME è fatta.
#  Parte A (featurizzazione): ogni molecola diventa un grafo.
#    - nodi = atomi, ognuno descritto da 29 numeri (elemento, grado, aromaticità, …);
#    - archi = legami, ognuno descritto da 10 numeri (tipo, coniugazione, anello, …);
#      ogni legame è salvato due volte, una per direzione (i→j e j→i);
#    - 25 descrittori "globali" della molecola intera (peso, logP, TPSA, …).
#  Parte B (modello): la rete D-MPNN (Directed Message Passing Neural Network, Yang et
#    al. 2019) che legge il grafo e predice i 9 pIC50.
#
# Viene usato dal passo 5 (training). Le funzioni sono nell'ordine in cui il training le
# chiama: prima la costruzione dei grafi, poi la normalizzazione, l'augmentation e il modello.
# =====================================================================================

# PyTorch: tensori e reti neurali.
import torch
# Moduli di base per costruire reti (strati lineari, BatchNorm, Dropout, …).
import torch.nn as nn
# Funzioni senza parametri (ReLU, …).
import torch.nn.functional as F
# Contenitore di PyTorch Geometric per un grafo (nodi, archi, attributi).
from torch_geometric.data import Data
# Set2Set: modo di riassumere tutti gli atomi di una molecola in un unico vettore.
from torch_geometric.nn import Set2Set
# scatter: somma i vettori di più archi sul loro nodo di destinazione.
from torch_geometric.utils import scatter
# RDKit: lettura delle molecole e descrittori chimici.
from rdkit import Chem
# Descrittori molecolari di RDKit (tre famiglie).
from rdkit.Chem import Descriptors, GraphDescriptors, Lipinski

# Numero di descrittori globali per molecola (vedi get_global_features).
N_GLOBAL_FEATURES = 25


# =====================================================================================
# PARTE A — FEATURIZZAZIONE: DA SMILES A GRAFO
# =====================================================================================

# Prepara i grafi di tutti gli SMILES distinti una sola volta (la parte lenta sono i 25
# descrittori RDKit). I grafi non hanno ancora il target: prepare_graph_data li copia e
# aggiunge i pIC50 al momento, così lo stesso lavoro è riusato da tutti i fold e i seed.
def build_base_graph_cache(smiles_list):
    # Dizionario SMILES -> grafo.
    cache = {}
    # Per ogni SMILES distinto (set elimina i duplicati)…
    for s in set(smiles_list):
        # …costruisce il grafo senza target…
        g = smiles_to_graph(s, None)
        # …e lo conserva se la molecola è valida.
        if g is not None:
            # Salva nel dizionario.
            cache[s] = g
    # Restituisce il dizionario.
    return cache


# Costruisce il grafo di una molecola; y (facoltativo) sono i suoi 9 pIC50.
def smiles_to_graph(smiles, y=None):
    # Interpreta lo SMILES.
    mol = Chem.MolFromSmiles(smiles)
    # SMILES non valido: nessun grafo.
    if mol is None:
        # Restituisce None (chi chiama scarta la molecola).
        return None
    # Matrice delle feature dei nodi: una riga di 29 numeri per atomo.
    x = torch.tensor([get_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    # Liste degli archi (coppie di indici di atomi) e delle loro feature.
    edges, edge_features = [], []
    # Per ogni legame della molecola…
    for bond in mol.GetBonds():
        # …indici dei due atomi collegati…
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # …feature del legame…
        f = get_bond_features(bond)
        # …l'arco viene salvato nelle due direzioni…
        edges += [[i, j], [j, i]]
        # …con le stesse feature per entrambe.
        edge_features += [f, f]
    # Molecola senza legami (es. un singolo ione): archi vuoti ma della forma giusta.
    if len(edges) == 0:
        # Nessun arco: tensore 2 × 0.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        # Nessuna feature di arco: 0 righe × 10 colonne.
        edge_attr = torch.empty((0, bond_feature_dim()), dtype=torch.float)
    # Caso normale.
    else:
        # Indici degli archi in formato 2 × n_archi (riga 0 = origine, riga 1 = destinazione).
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # Feature degli archi: n_archi × 10.
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    # Target come riga 1 × 9, se fornito.
    y_tensor = torch.tensor([y], dtype=torch.float) if y is not None else None
    # Impacchetta tutto in un oggetto grafo di PyTorch Geometric.
    return Data(x=x,  # feature dei nodi
                edge_index=edge_index,  # chi è collegato a chi
                edge_attr=edge_attr,  # feature dei legami
                y=y_tensor,  # pIC50 (o None)
                global_features=get_global_features(mol).view(1, -1))  # 25 descrittori, riga 1 × 25


# Codifica "one-hot": un vettore di zeri con un 1 nella posizione del valore. L'ultima
# posizione in più vale 1 quando il valore non è tra le scelte previste ("altro").
def one_hot_encoding(value, choices):
    # Vettore di zeri lungo quanto le scelte + 1.
    encoding = [0] * (len(choices) + 1)
    # Posizione del valore, oppure -1 (= ultima posizione) se non previsto.
    index = choices.index(value) if value in choices else -1
    # Accende la posizione trovata.
    encoding[index] = 1
    # Restituisce il vettore.
    return encoding


# Le 29 feature di un atomo.
def get_atom_features(atom):
    # Lista che verrà riempita.
    features = []
    # Elemento chimico: one-hot su 9 elementi comuni + "altro" (10 numeri).
    features += one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'])
    # Numero di atomi pesanti legati (grado).
    features.append(atom.GetDegree())
    # Numero di idrogeni legati.
    features.append(atom.GetTotalNumHs())
    # 1 se l'atomo è aromatico.
    features.append(1 if atom.GetIsAromatic() else 0)
    # 1 se l'atomo è in un anello.
    features.append(1 if atom.IsInRing() else 0)
    # Ibridazione: SP, SP2, SP3 o "altro" (4 numeri).
    features += one_hot_encoding(atom.GetHybridization(),
                                 [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3])
    # Carica formale.
    features.append(atom.GetFormalCharge())
    # Chiralità: orario, antiorario o nessuna/altro (3 numeri).
    features += one_hot_encoding(atom.GetChiralTag(),
                                 [Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW])
    # Per anelli da 3 a 7 atomi: 1 se l'atomo ne fa parte (5 numeri).
    for ring_size in (3, 4, 5, 6, 7):
        # Appartenenza a un anello di quella dimensione.
        features.append(1 if atom.IsInRingSize(ring_size) else 0)
    # Valenza totale.
    features.append(atom.GetTotalValence())
    # Elettroni spaiati (radicali).
    features.append(atom.GetNumRadicalElectrons())
    # Restituisce le 29 feature.
    return features


# Le 10 feature di un legame.
def get_bond_features(bond):
    # Tipo di legame: singolo, doppio, triplo, aromatico o "altro" (5 numeri).
    features = one_hot_encoding(bond.GetBondType(), [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                                                     Chem.BondType.TRIPLE, Chem.BondType.AROMATIC])
    # 1 se il legame è coniugato.
    features.append(1 if bond.GetIsConjugated() else 0)
    # 1 se il legame è in un anello.
    features.append(1 if bond.IsInRing() else 0)
    # Stereochimica del doppio legame: Z, E o nessuna/altro (3 numeri).
    features += one_hot_encoding(bond.GetStereo(), [Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE])
    # Restituisce le 10 feature.
    return features


# Lunghezza del vettore di feature di un legame (10), calcolata su un legame di prova.
def bond_feature_dim():
    # Etano (C-C): ha esattamente un legame di cui misurare le feature.
    return len(get_bond_features(Chem.MolFromSmiles('CC').GetBondWithIdx(0)))


# I 25 descrittori globali della molecola intera.
def get_global_features(mol):
    # Elenco dei descrittori calcolati da RDKit.
    features = [Descriptors.MolWt(mol),  # peso molecolare
                Descriptors.MolLogP(mol),  # lipofilia (logP)
                Descriptors.TPSA(mol),  # superficie polare topologica
                Descriptors.NumHDonors(mol),  # donatori di legame a idrogeno
                Descriptors.NumHAcceptors(mol),  # accettori di legame a idrogeno
                Descriptors.NumRotatableBonds(mol),  # legami ruotabili (flessibilità)
                Descriptors.NumAromaticRings(mol),  # anelli aromatici
                Descriptors.NumAliphaticRings(mol),  # anelli non aromatici
                Descriptors.FractionCSP3(mol),  # frazione di carboni sp3
                Descriptors.MolMR(mol),  # rifrattività molare
                GraphDescriptors.BertzCT(mol),  # complessità del grafo
                GraphDescriptors.BalabanJ(mol),  # indice topologico di Balaban
                GraphDescriptors.Chi0v(mol),  # indice di connettività di ordine 0
                GraphDescriptors.Chi1v(mol),  # indice di connettività di ordine 1
                GraphDescriptors.Chi2v(mol),  # indice di connettività di ordine 2
                Lipinski.NumHeteroatoms(mol),  # atomi diversi da C e H
                Lipinski.HeavyAtomCount(mol),  # atomi pesanti
                Descriptors.NumValenceElectrons(mol),  # elettroni di valenza
                Descriptors.NumRadicalElectrons(mol),  # elettroni radicalici
                Descriptors.LabuteASA(mol),  # area superficiale approssimata
                GraphDescriptors.Kappa3(mol),  # indice di forma kappa 3
                Lipinski.RingCount(mol),  # anelli totali
                Lipinski.NumSaturatedRings(mol),  # anelli saturi
                Lipinski.NumAromaticHeterocycles(mol),  # eterocicli aromatici
                Lipinski.NumAliphaticHeterocycles(mol)]  # eterocicli non aromatici
    # Converte in tensore; eventuali valori non numerici (NaN, infinito) diventano 0.
    return torch.nan_to_num(torch.tensor(features, dtype=torch.float), nan=0.0, posinf=0.0, neginf=0.0)


# Costruisce i grafi di una lista di molecole con i rispettivi target. Se è passata la
# cache, copia i grafi già pronti invece di ricalcolarli. Le molecole non valide sono saltate.
def prepare_graph_data(smiles_list, y_list, base_cache=None):
    # Lista dei grafi.
    data_list = []
    # Per ogni molecola e il suo vettore di 9 pIC50…
    for s, y in zip(smiles_list, y_list):
        # …se il grafo è in cache…
        if base_cache is not None and s in base_cache:
            # …ne fa una copia (così modificare la copia non tocca la cache)…
            graph = base_cache[s].clone()
            # …e le assegna il target.
            graph.y = torch.tensor([y], dtype=torch.float) if y is not None else None
        # Altrimenti lo costruisce da zero.
        else:
            # Grafo con target.
            graph = smiles_to_graph(s, y)
        # Scarta le molecole non valide.
        if graph is None:
            # Passa alla successiva.
            continue
        # Aggiunge il grafo alla lista.
        data_list.append(graph)
    # Restituisce i grafi.
    return data_list


# Calcola media e deviazione standard di ognuno dei 25 descrittori globali sui grafi di
# allenamento: servono a riportarli tutti sulla stessa scala (media 0, deviazione 1).
def fit_global_scaler(graphs):
    # Matrice molecole × 25 dei descrittori.
    g = torch.stack([d.global_features.view(-1) for d in graphs], dim=0)
    # Media per descrittore e deviazione standard (mai sotto 1e-6, per non dividere per 0).
    return g.mean(dim=0), g.std(dim=0).clamp(min=1e-06)


# Applica la standardizzazione (x - media) / deviazione ai descrittori dei grafi, sul posto.
def apply_global_scaler(graphs, scaler):
    # Media e deviazione calcolate da fit_global_scaler.
    mean, std = scaler
    # Le sposta sullo stesso dispositivo (CPU o GPU) dei grafi.
    dev = graphs[0].global_features.device
    # Spostamento della media.
    mean, std = mean.to(dev), std.to(dev)
    # Per ogni grafo…
    for d in graphs:
        # …sostituisce i descrittori con la loro versione standardizzata.
        d.global_features = (d.global_features - mean) / std


# Dataset di allenamento con "data augmentation" casuale, ricampionata a ogni accesso
# (quindi diversa a ogni epoca): toglie a caso alcuni legami (edge dropout) e azzera a
# caso alcuni atomi (node masking). Rende il modello robusto a piccole variazioni.
class DynamicAugDataset(torch.utils.data.Dataset):

    # Memorizza i grafi e le due probabilità.
    def __init__(self, graphs, edge_drop_prob, node_mask_prob):
        # Grafi già standardizzati.
        self.graphs = graphs
        # Probabilità di eliminare un legame.
        self.edge_drop_prob = edge_drop_prob
        # Probabilità di azzerare le feature di un atomo.
        self.node_mask_prob = node_mask_prob

    # Numero di molecole nel dataset (richiesto da PyTorch).
    def __len__(self):
        # Una molecola per grafo.
        return len(self.graphs)

    # Restituisce la molecola idx con un'augmentation nuova (chiamato dal DataLoader).
    def __getitem__(self, idx):
        # Copia del grafo, per non modificare l'originale.
        g = self.graphs[idx].clone()
        # Edge dropout, se attivo e se la molecola ha legami.
        if self.edge_drop_prob > 0 and g.edge_index.size(1) > 0:
            # Numero di legami (gli archi sono il doppio: una coppia per legame).
            n_edges = g.edge_index.size(1) // 2
            # Per ogni legame decide a caso se tenerlo.
            keep = torch.rand(n_edges) >= self.edge_drop_prob
            # Estende la decisione alle due direzioni di ogni legame.
            full = keep.repeat_interleave(2)
            # Se resta almeno un arco (altrimenti lascia il grafo intatto)…
            if full.any():
                # …tiene solo gli archi scelti…
                g.edge_index = g.edge_index[:, full]
                # …e le loro feature.
                g.edge_attr = g.edge_attr[full]
        # Node masking, se attivo.
        if self.node_mask_prob > 0 and g.x.shape[0] > 0:
            # Per ogni atomo: 1 = tenuto, 0 = azzerato.
            keep_nodes = (torch.rand(g.x.shape[0]) >= self.node_mask_prob).float().view(-1, 1)
            # Moltiplica le feature degli atomi per 0 o 1.
            g.x = g.x * keep_nodes
        # Restituisce il grafo modificato.
        return g


# =====================================================================================
# PARTE B — IL MODELLO D-MPNN
#
# Idea: i messaggi viaggiano sugli archi orientati. Ogni arco i→j ha uno stato nascosto
# h(i→j) che, a ogni passo, viene aggiornato sommando gli stati degli archi che entrano in
# i, ESCLUSO quello di ritorno j→i (così il messaggio non torna subito indietro). Dopo
# num_layers passi, ogni atomo raccoglie i messaggi in arrivo; Set2Set riassume gli atomi
# in un vettore per molecola; la "testa" lo unisce ai 25 descrittori e predice i 9 pIC50.
# =====================================================================================

# La rete completa.
class DMPNNModel(nn.Module):

    # Crea gli strati. node_in_features = 29, edge_in_features = 10 (vedi Parte A).
    def __init__(self, node_in_features, edge_in_features, hidden_dim=128, out_dim=9,
                 num_global_features=N_GLOBAL_FEATURES, num_layers=3, conv_dropout=0.05,
                 fc_dropout=0.3, fc1_hidden=192, fc2_hidden=96, set2set_steps=3):
        # Inizializzazione obbligatoria di ogni rete PyTorch.
        super().__init__()
        # Numero di passi di message passing.
        self.depth = num_layers
        # Stato iniziale di un arco: da [feature dell'atomo di partenza ; feature del legame].
        self.W_i = nn.Linear(node_in_features + edge_in_features, hidden_dim)
        # Trasformazione del messaggio a ogni passo.
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        # Stato finale di un atomo: da [feature dell'atomo ; somma dei messaggi in arrivo].
        self.W_o = nn.Linear(node_in_features + hidden_dim, hidden_dim)
        # Dropout sui messaggi (spegne a caso neuroni durante l'allenamento).
        self.drop = nn.Dropout(conv_dropout)
        # Normalizzazione degli stati degli atomi.
        self.bn = nn.BatchNorm1d(hidden_dim)
        # Lettura del grafo: Set2Set produce un vettore di dimensione 2 × hidden_dim.
        self.readout = Set2Set(hidden_dim, processing_steps=set2set_steps)
        # Testa finale che produce i 9 pIC50.
        self.head = _Head(2 * hidden_dim, num_global_features, out_dim, fc1_hidden, fc2_hidden, fc_dropout)

    # Carica i pesi dell'encoder (W_i, W_h, W_o, bn) pre-addestrati in modo auto-supervisionato
    # su molecole non etichettate; la testa resta casuale e verrà imparata.
    def load_pretrained_encoder(self, path, device):
        # Legge il file dei pesi.
        state = torch.load(path, map_location=device)
        # Li copia nella rete; strict=False perché la testa non è nel file.
        result = self.load_state_dict(state, strict=False)
        # Un tensore del file che non corrisponde a nessuno strato indica un file sbagliato.
        if result.unexpected_keys:
            # Errore esplicito con i nomi dei tensori estranei.
            raise ValueError(f'pretrained checkpoint has unknown tensors: {result.unexpected_keys}')

    # Calcolo in avanti su un "batch" di molecole unite in un unico grande grafo.
    # x: atomi × 29; edge_index: 2 × archi; edge_attr: archi × 10;
    # batch: per ogni atomo, l'indice della molecola a cui appartiene; global_features: molecole × 25.
    def forward(self, x, edge_index, edge_attr, batch, global_features):
        # Numero totale di atomi nel batch.
        N = x.size(0)
        # Atomo di origine (s) e di destinazione (t) di ogni arco.
        s, t = edge_index[0].long(), edge_index[1].long()
        # Per ogni arco i→j, l'indice del suo arco inverso j→i.
        rev = self._reverse_index(edge_index, N)
        # Stato iniziale degli archi da [atomo di partenza ; legame].
        h0 = F.relu(self.W_i(torch.cat([x[s], edge_attr], dim=1)))
        # Lo stato corrente parte da quello iniziale.
        h = h0
        # Passi di message passing.
        for _ in range(self.depth):
            # Somma, per ogni atomo, gli stati degli archi che entrano in lui.
            m_node = scatter(h, t, dim=0, dim_size=N, reduce='sum')
            # Messaggio per l'arco i→j: tutto ciò che entra in i meno l'arco di ritorno j→i.
            m_e = m_node[s] - h[rev]
            # Nuovo stato = stato iniziale + messaggio trasformato (connessione residua).
            h = self.drop(F.relu(h0 + self.W_h(m_e)))
        # Messaggi finali raccolti da ogni atomo.
        m_node = scatter(h, t, dim=0, dim_size=N, reduce='sum')
        # Stato finale di ogni atomo da [sue feature ; messaggi ricevuti].
        h_v = F.relu(self.bn(self.W_o(torch.cat([x, m_node], dim=1))))
        # Un vettore per molecola, riassunto dei suoi atomi.
        graph_emb = self.readout(h_v, batch)
        # La testa produce le 9 predizioni per molecola.
        return self.head(graph_emb, global_features)

    # Trova l'arco inverso di ogni arco. Codifica ogni arco (s, t) come numero unico
    # s * N + t, ordina, e cerca per ogni arco il codice del suo inverso t * N + s.
    @staticmethod
    def _reverse_index(edge_index, num_nodes):
        # Origine e destinazione degli archi.
        s, t = edge_index[0].long(), edge_index[1].long()
        # Codice di ogni arco s→t.
        key_f = s * num_nodes + t
        # Codice dell'inverso t→s.
        key_b = t * num_nodes + s
        # Ordine che mette i codici in fila crescente.
        order = torch.argsort(key_f)
        # Codici ordinati.
        sk = key_f[order]
        # Posizione in cui ogni codice inverso dovrebbe trovarsi nella fila ordinata.
        pos = torch.searchsorted(sk, key_b).clamp(max=sk.numel() - 1)
        # Indice originale dell'arco trovato in quella posizione.
        rev = order[pos]
        # Vero se in quella posizione c'è davvero l'inverso.
        valid = sk[pos] == key_b
        # Indici 0, 1, 2, … di tutti gli archi.
        idx = torch.arange(s.numel(), device=s.device)
        # Se l'inverso manca (es. tolto dall'augmentation) l'arco punta a se stesso.
        return torch.where(valid, rev, idx)


# Testa finale: unisce il vettore della molecola ai 25 descrittori globali e, con due
# strati nascosti, produce i 9 pIC50.
class _Head(nn.Module):

    # Crea gli strati della testa.
    def __init__(self, graph_emb_dim, num_global_features, out_dim, fc1_hidden, fc2_hidden, fc_dropout):
        # Inizializzazione obbligatoria.
        super().__init__()
        # Normalizzazione dei descrittori globali.
        self.bn_global = nn.BatchNorm1d(num_global_features)
        # Primo strato: [vettore molecola ; descrittori] -> fc1_hidden neuroni.
        self.fc1 = nn.Linear(graph_emb_dim + num_global_features, fc1_hidden)
        # Normalizzazione dopo il primo strato.
        self.bn_fc1 = nn.BatchNorm1d(fc1_hidden)
        # Dropout dopo il primo strato.
        self.drop1 = nn.Dropout(fc_dropout)
        # Secondo strato: fc1_hidden -> fc2_hidden.
        self.fc2 = nn.Linear(fc1_hidden, fc2_hidden)
        # Normalizzazione dopo il secondo strato.
        self.bn_fc2 = nn.BatchNorm1d(fc2_hidden)
        # Dropout dopo il secondo strato.
        self.drop2 = nn.Dropout(fc_dropout)
        # Strato di uscita: un numero per isoforma.
        self.fc3 = nn.Linear(fc2_hidden, out_dim)

    # Calcolo in avanti della testa.
    def forward(self, graph_emb, global_features):
        # Descrittori normalizzati.
        g = self.bn_global(global_features)
        # Concatena vettore della molecola e descrittori.
        x = torch.cat([graph_emb, g], dim=1)
        # Primo strato: lineare -> normalizzazione -> ReLU -> dropout.
        x = self.drop1(F.relu(self.bn_fc1(self.fc1(x))))
        # Secondo strato, stessa sequenza.
        x = self.drop2(F.relu(self.bn_fc2(self.fc2(x))))
        # Uscita: 9 pIC50 (in scala standardizzata durante il training, vedi passo 5).
        return self.fc3(x)
