# =====================================================================================
# PASSO 5 di 7 — TRAINING DEL D-MPNN IN CROSS-VALIDATION, SCELTA DEL SEED, PREDIZIONE
#
# Ruolo nella pipeline: allena il modello del passo 4 su train+val e produce:
#  - per ogni seed, 5 modelli (uno per fold della cross-validation per gruppi);
#  - le predizioni "out-of-fold" (OOF): ogni molecola predetta dal modello del fold in cui
#    era di validazione, cioè da un modello che non l'ha mai vista;
#  - la scelta del seed migliore guardando SOLO la CV (il test non viene mai usato per scegliere);
#  - le predizioni su molecole nuove, come media pesata dei 5 modelli del seed scelto.
#
# Ordine di esecuzione:
#   train_dmpnn (notebook)
#     -> best_of_n_group_kfold       (decide i 5 fold una volta per tutte)
#     -> per ogni seed, per ogni fold: _train_one_fold
#           -> UncertaintyWeightedLoss, ciclo delle epoche, _evaluate, _update_bn_gnn (SWA)
#   select_best_seed / seed_view     (notebook, dopo il training)
#   predict_dmpnn                    (notebook e passo 7)
# In questo file le funzioni di supporto vengono prima e train_dmpnn è dopo di esse;
# select_best_seed, seed_view e predict_dmpnn, chiamate dopo il training, sono in fondo.
# =====================================================================================

# Composizione dei percorsi dei file di output.
import os
# Calcolo numerico su array.
import numpy as np
# Tabelle (storia delle epoche, risultati per fold).
import pandas as pd
# PyTorch: tensori e reti neurali.
import torch
# Funzioni senza parametri (qui l'errore quadratico medio).
import torch.nn.functional as F
# Ottimizzatori e scheduler del learning rate.
import torch.optim as optim
# Stochastic Weight Averaging: media dei pesi delle ultime epoche e suo scheduler.
from torch.optim.swa_utils import AveragedModel, SWALR
# DataLoader di PyTorch Geometric: unisce più grafi in un unico batch.
from torch_geometric.loader import DataLoader
# Costanti condivise.
from s0_config import ISOFORMS, TARGET_COLS, SMILES_COL, GROUP_COL
# Grafici delle distribuzioni e test KS (passo 1).
from s1_split import ks_log_p
# Metriche (passo 2).
from s2_metrics import calculate_metrics
# Grafi, normalizzazione, augmentation e modello (passo 4).
from s4_dmpnn import (DMPNNModel, DynamicAugDataset, apply_global_scaler, bond_feature_dim,
                      build_base_graph_cache, fit_global_scaler, prepare_graph_data)


# -------------------------------------------------------------------------------------
# 1. I fold della cross-validation per gruppi
# -------------------------------------------------------------------------------------

# Assegna i gruppi ai k fold in modo che i fold abbiano dimensioni simili: dal gruppo più
# grande al più piccolo, ognuno va nel fold che in quel momento ha meno molecole.
# L'ordine casuale di partenza decide chi va dove a parità di dimensione.
def _greedy_balanced_assignment(groups, shuffled_groups, k_folds):
    # Numero di molecole di ogni gruppo.
    sizes = pd.Series(groups).value_counts().to_dict()
    # Molecole già assegnate a ogni fold.
    fold_counts = np.zeros(k_folds, dtype=int)
    # Dizionario gruppo -> fold.
    fold_assignment = {}
    # Gruppi in ordine di dimensione decrescente (a parità, conta l'ordine casuale).
    for g in sorted(shuffled_groups, key=lambda g: -sizes.get(g, 0)):
        # Il fold attualmente più piccolo.
        k = int(np.argmin(fold_counts))
        # Il gruppo va lì.
        fold_assignment[g] = k
        # Aggiorna il conteggio di quel fold.
        fold_counts[k] += sizes.get(g, 0)
    # Restituisce l'assegnazione.
    return fold_assignment


# Trasforma l'assegnazione gruppo -> fold in k coppie (indici di training, indici di validazione).
def _splits_from_assignment(groups, fold_assignment, k_folds):
    # Fold di ogni molecola.
    fold_of = np.array([fold_assignment[g] for g in groups])
    # Per ogni fold k: validazione = molecole del fold k, training = tutte le altre.
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in range(k_folds)]


# Punteggio di una suddivisione in fold: media dei log p-value KS tra training e validazione
# per ogni fold e ogni isoforma (più alto = distribuzioni di pIC50 più simili).
# Le isoforme con meno di min_n valori in uno dei due lati vengono ignorate.
def _score_fold_assignment(splits, Y, min_n=5):
    # Log p-value raccolti.
    log_scores = []
    # Per ogni fold…
    for tr_idx, va_idx in splits:
        # …e ogni isoforma…
        for t in range(Y.shape[1]):
            # …valori misurati nel training…
            tr = Y[tr_idx, t][~np.isnan(Y[tr_idx, t])]
            # …e nella validazione.
            va = Y[va_idx, t][~np.isnan(Y[va_idx, t])]
            # Troppo pochi valori: confronto ignorato.
            if len(tr) < min_n or len(va) < min_n:
                # Passa all'isoforma successiva.
                continue
            # Log p-value del test KS.
            log_scores.append(ks_log_p(tr, va))
    # Media dei punteggi (meno infinito se non c'era nulla da confrontare).
    return float(np.mean(log_scores)) if log_scores else float('-inf')


# Prova n_trials assegnazioni casuali dei gruppi ai fold. Scarta quelle con un fold vuoto o
# con fold troppo diversi in dimensione (oltre size_tol = ±10%); tra le rimaste tiene quella
# con il punteggio KS migliore. Se nessuna rispetta la tolleranza, ripiega sulla più bilanciata.
def best_of_n_group_kfold(groups, Y, k_folds, n_trials, seed, size_tol=0.1):
    # Generatore casuale con seme fisso: i fold sono sempre gli stessi.
    rng = np.random.default_rng(seed)
    # Elenco dei gruppi distinti.
    unique_groups = np.unique(groups)
    # Dimensione ideale di un fold di validazione.
    target_size = len(groups) / k_folds
    # Miglior punteggio e migliore suddivisione finora.
    best_score, best_splits = float('-inf'), None
    # Tentativi casuali.
    for _ in range(n_trials):
        # Ordine casuale dei gruppi.
        shuffled = unique_groups[rng.permutation(len(unique_groups))]
        # Fold candidati.
        splits = _splits_from_assignment(groups, _greedy_balanced_assignment(groups, shuffled, k_folds), k_folds)
        # Dimensione di ogni fold di validazione.
        val_sizes = np.array([len(va) for _, va in splits])
        # Scarta i candidati con un fold vuoto o troppo sbilanciati.
        if (val_sizes == 0).any() or np.max(np.abs(val_sizes - target_size)) > size_tol * target_size:
            # Passa al tentativo successivo.
            continue
        # Punteggio KS del candidato.
        sc = _score_fold_assignment(splits, Y)
        # Se è il migliore finora…
        if sc > best_score:
            # …lo conserva.
            best_score, best_splits = sc, splits
    # Ripiego (raro, solo su dataset piccoli): nessun candidato ha rispettato la tolleranza.
    if best_splits is None:
        # Sbilanciamento migliore visto.
        best_imbalance = float('inf')
        # Nuovi tentativi con un altro seme.
        rng2 = np.random.default_rng(seed + 1)
        # Stesso numero di tentativi.
        for _ in range(n_trials):
            # Ordine casuale dei gruppi.
            shuffled = unique_groups[rng2.permutation(len(unique_groups))]
            # Fold candidati.
            splits = _splits_from_assignment(groups, _greedy_balanced_assignment(groups, shuffled, k_folds), k_folds)
            # Dimensioni dei fold.
            val_sizes = np.array([len(va) for _, va in splits])
            # Un fold vuoto non è mai accettabile.
            if (val_sizes == 0).any():
                # Passa al tentativo successivo.
                continue
            # Massimo scostamento dalla dimensione ideale.
            imb = np.max(np.abs(val_sizes - target_size))
            # Tiene il candidato più bilanciato.
            if imb < best_imbalance:
                # Aggiorna il migliore.
                best_imbalance, best_splits = imb, splits
    # Se ancora nulla, il dataset non si può dividere in k fold.
    if best_splits is None:
        # Errore esplicito.
        raise RuntimeError('best_of_n_group_kfold: no valid split found.')
    # Restituisce le k coppie (training, validazione).
    return best_splits


# -------------------------------------------------------------------------------------
# 2. Loss e valutazione
# -------------------------------------------------------------------------------------

# Loss multi-task con pesi appresi (Kendall et al. 2018). Ogni isoforma ha un parametro
# log_var: la sua precisione exp(-log_var) pesa l'errore, e il termine + log_var impedisce di
# azzerare tutti i pesi. Le isoforme più rumorose ricevono così automaticamente meno peso.
# Le misure mancanti (NaN) vengono ignorate.
class UncertaintyWeightedLoss(torch.nn.Module):

    # Crea un parametro log_var per isoforma, inizialmente 0 (peso 1).
    def __init__(self, n_tasks):
        # Inizializzazione obbligatoria.
        super().__init__()
        # Numero di isoforme.
        self.n_tasks = n_tasks
        # Parametri appresi insieme alla rete.
        self.log_vars = torch.nn.Parameter(torch.zeros(n_tasks))

    # Calcola la loss su un batch.
    def forward(self, predictions, targets):
        # Somma dei contributi (tensore scalare a zero, sullo stesso dispositivo).
        total = predictions.new_zeros(())
        # Isoforme che hanno contribuito.
        active = 0
        # Per ogni isoforma…
        for i in range(self.n_tasks):
            # …molecole del batch misurate su di essa…
            mask = ~torch.isnan(targets[:, i])
            # …ne servono almeno 2, altrimenti l'isoforma è saltata in questo batch.
            if mask.sum() < 2:
                # Isoforma successiva.
                continue
            # Errore quadratico medio sull'isoforma.
            mse_i = F.mse_loss(predictions[mask, i], targets[mask, i])
            # Precisione = exp(-log_var).
            precision = torch.exp(-self.log_vars[i])
            # Contributo: errore pesato + regolarizzazione.
            total = total + 0.5 * precision * mse_i + 0.5 * self.log_vars[i]
            # Un'isoforma in più ha contribuito.
            active += 1
        # Nessuna isoforma utilizzabile nel batch: loss nulla (ma derivabile).
        if active == 0:
            # Restituisce zero.
            return predictions.new_zeros((), requires_grad=True)
        # Media dei contributi.
        return total / active


# Esegue il modello su tutte le molecole di un loader e restituisce predizioni e target.
@torch.no_grad()  # niente gradienti: solo calcolo in avanti, più veloce
def get_predictions(model, loader, device):
    # Modalità valutazione: dropout spento, BatchNorm con statistiche memorizzate.
    model.eval()
    # Liste dei risultati dei batch.
    preds, targets = [], []
    # Per ogni batch…
    for data in loader:
        # …sposta il batch sul dispositivo (GPU o CPU)…
        data = data.to(device)
        # …calcola le predizioni…
        preds.append(model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_features))
        # …e conserva i target veri.
        targets.append(data.y)
    # Concatena i batch in due matrici molecole × 9.
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


# Metriche del modello su un loader (in scala standardizzata, come vede la rete).
def _evaluate(model, loader, device):
    # Predizioni e target.
    preds, tgts = get_predictions(model, loader, device)
    # Metriche calcolate su array numpy.
    return calculate_metrics(preds.cpu().numpy(), tgts.cpu().numpy())


# Ricalcola le statistiche dei BatchNorm del modello SWA: la media dei pesi rende obsolete
# quelle accumulate durante il training. Fa un passaggio sui dati di training in modalità
# train (senza aggiornare i pesi) con momentum None, cioè una media esatta su tutti i batch.
@torch.no_grad()  # nessun gradiente: si aggiornano solo le statistiche
def _update_bn_gnn(loader, model, device):
    # Momentum originale di ogni BatchNorm, da ripristinare alla fine.
    momenta = {}
    # Per ogni strato del modello…
    for m in model.modules():
        # …se è un BatchNorm…
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            # …azzera le statistiche accumulate…
            m.reset_running_stats()
            # …e ricorda il suo momentum.
            momenta[m] = m.momentum
    # Modalità training: i BatchNorm aggiornano le statistiche.
    model.train()
    # Per ogni BatchNorm…
    for m in momenta:
        # …momentum None = media cumulativa esatta.
        m.momentum = None
    # Passaggio sui dati di training.
    for data in loader:
        # Sposta il batch sul dispositivo.
        data = data.to(device)
        # Calcolo in avanti (serve solo ad aggiornare le statistiche).
        model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_features)
    # Ripristina i momentum originali.
    for m, mom in momenta.items():
        # Momentum di questo strato.
        m.momentum = mom


# -------------------------------------------------------------------------------------
# 3. Training di un fold
# -------------------------------------------------------------------------------------

# Allena un modello sul training di un fold e lo valuta sulla sua validazione. Passi:
#  1) standardizza i pIC50 con media e deviazione del training (per isoforma);
#  2) costruisce i grafi e standardizza i 25 descrittori con le statistiche del training;
#  3) crea modello (con encoder pre-addestrato), loss, ottimizzatore AdamW, scheduler e SWA;
#  4) ciclo delle epoche con warmup del learning rate, early stopping e SWA nelle ultime;
#  5) tiene i pesi migliori (miglior MSE di validazione, oppure la media SWA se è migliore);
#  6) predice la validazione e riporta le predizioni in pIC50.
def _train_one_fold(fold_idx, tr_idx, va_idx, smiles, Y, cache, node_dim, edge_dim, model_kwargs,
                    pretrained, device, k_folds, epochs, patience, batch_size, lr, weight_decay,
                    grad_clip, edge_drop_prob, node_mask_prob, swa_start_frac, swa_lr, num_workers,
                    warmup_epochs, train_eval_every):
    # pIC50 di training e di validazione (matrici molecole × 9).
    tr_y, va_y = Y[tr_idx], Y[va_idx]
    # 1) Media di ogni isoforma sul training.
    mu = np.nanmean(tr_y, axis=0)
    # Deviazione standard di ogni isoforma sul training.
    sigma = np.nanstd(tr_y, axis=0)
    # Deviazioni nulle o non definite diventano 1 (nessuna divisione per zero).
    sigma = np.where(~np.isfinite(sigma) | (sigma < 1e-06), 1.0, sigma).astype(np.float32)
    # Medie non definite (isoforma senza misure) diventano 0.
    mu = np.nan_to_num(mu, nan=0.0).astype(np.float32)
    # 2) Grafi di training con target standardizzati (copiati dalla cache).
    tr_graphs = prepare_graph_data(smiles[tr_idx], (tr_y - mu) / sigma, base_cache=cache)
    # Grafi di validazione, stessa standardizzazione (statistiche del training!).
    va_graphs = prepare_graph_data(smiles[va_idx], (va_y - mu) / sigma, base_cache=cache)
    # Statistiche dei 25 descrittori calcolate sul solo training.
    scaler = fit_global_scaler(tr_graphs)
    # Standardizza i descrittori del training…
    apply_global_scaler(tr_graphs, scaler)
    # …e della validazione con le stesse statistiche.
    apply_global_scaler(va_graphs, scaler)
    # Su GPU, la memoria "pinned" velocizza il trasferimento dei batch.
    pin_memory = device == 'cuda'
    # Loader di training: augmentation nuova a ogni epoca, ordine mescolato.
    tr_loader = DataLoader(DynamicAugDataset(tr_graphs, edge_drop_prob, node_mask_prob),
                           batch_size=batch_size,  # molecole per batch
                           shuffle=True,  # ordine casuale a ogni epoca
                           num_workers=num_workers,  # processi paralleli che preparano i batch
                           pin_memory=pin_memory,  # trasferimento veloce verso la GPU
                           persistent_workers=num_workers > 0,  # processi tenuti vivi tra le epoche
                           drop_last=len(tr_graphs) % batch_size == 1)  # evita un ultimo batch di 1 molecola (rompe BatchNorm)
    # Loader di validazione: niente augmentation, ordine fisso.
    va_loader = DataLoader(va_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=pin_memory, persistent_workers=num_workers > 0)
    # 3) Modello nuovo per il fold, sul dispositivo scelto.
    model = DMPNNModel(node_dim, edge_dim, **model_kwargs).to(device)
    # Carica l'encoder pre-addestrato.
    model.load_pretrained_encoder(pretrained, device=device)
    # Loss con pesi per isoforma appresi.
    loss_fn = UncertaintyWeightedLoss(n_tasks=Y.shape[1]).to(device)
    # Parametri da ottimizzare: quelli della rete più i log_var della loss.
    params = list(model.parameters()) + [p for p in loss_fn.parameters() if p.requires_grad]
    # Ottimizzatore AdamW (Adam con weight decay separato).
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # Dimezza il learning rate se l'MSE di validazione non migliora per 4 epoche.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-06)
    # Storia delle epoche (una riga per epoca).
    history = []
    # Miglior MSE di validazione finora.
    best_val_mse = float('inf')
    # Pesi del modello migliore.
    best_state = None
    # Epoca del modello migliore.
    best_epoch = 0
    # Epoche consecutive senza miglioramento (per l'early stopping).
    patience_counter = 0
    # Metriche di training (calcolate solo ogni train_eval_every epoche).
    train_m = {'mse': float('nan'), 'r2': float('nan')}
    # R² di validazione del modello migliore (solo per la stampa).
    best_val_r2 = float('nan')
    # Nota da aggiungere alla riga finale del fold se viene usato lo SWA.
    swa_note = ''
    # Epoca da cui parte lo SWA (es. 75% di 120 = 90).
    swa_start_epoch = int(epochs * swa_start_frac)
    # Copia del modello che conterrà la media dei pesi.
    swa_model = AveragedModel(model)
    # Learning rate costante e basso durante la fase SWA.
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    # 4) Ciclo delle epoche.
    for epoch in range(1, epochs + 1):
        # Warmup: nelle prime epoche il learning rate sale linearmente fino a lr.
        if epoch <= warmup_epochs:
            # Per ogni gruppo di parametri dell'ottimizzatore…
            for pg in optimizer.param_groups:
                # …imposta il learning rate di warmup.
                pg['lr'] = lr * epoch / warmup_epochs
        # Modalità training: dropout e aggiornamento dei BatchNorm attivi.
        model.train()
        # Per ogni batch di training…
        for data in tr_loader:
            # …sposta il batch sul dispositivo…
            data = data.to(device)
            # …calcola le predizioni…
            preds = model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_features)
            # …e la loss rispetto ai target standardizzati.
            loss = loss_fn(preds, data.y)
            # Azzera i gradienti del passo precedente.
            optimizer.zero_grad()
            # Calcola i gradienti (backpropagation).
            loss.backward()
            # Limita la norma dei gradienti per evitare passi troppo grandi.
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            # Aggiorna i pesi.
            optimizer.step()
        # Siamo nella fase finale SWA?
        in_swa_phase = epoch >= swa_start_epoch
        # Se sì…
        if in_swa_phase:
            # …aggiunge i pesi attuali alla media…
            swa_model.update_parameters(model)
            # …e aggiorna il learning rate SWA.
            swa_scheduler.step()
        # Metriche di training, solo alla prima epoca e ogni train_eval_every (sono costose).
        if epoch == 1 or epoch % train_eval_every == 0:
            # Valuta il training (con l'augmentation del loader).
            train_m = _evaluate(model, tr_loader, device)
        # Metriche di validazione, a ogni epoca.
        val_m = _evaluate(model, va_loader, device)
        # Learning rate usato in questa epoca.
        current_lr = optimizer.param_groups[0]['lr']
        # Lo scheduler su plateau agisce solo dopo il warmup e prima della fase SWA.
        if epoch > warmup_epochs and not in_swa_phase:
            # Gli passa l'MSE di validazione.
            scheduler.step(val_m['mse'])
        # Registra l'epoca nella storia.
        history.append({'epoch': epoch,  # numero dell'epoca
                        'train_mse': train_m['mse'], 'val_mse': val_m['mse'],  # errori
                        'gap_mse': val_m['mse'] - train_m['mse'],  # >0 = la validazione va peggio (overfitting)
                        'train_r2': train_m['r2'], 'val_r2': val_m['r2'],  # R²
                        'gap_r2': train_m['r2'] - val_m['r2'],  # differenza di R²
                        'lr': current_lr})  # learning rate
        # Il modello è migliorato in validazione?
        if val_m['mse'] < best_val_mse:
            # Nuovo miglior MSE…
            best_val_mse = val_m['mse']
            # …a questa epoca…
            best_epoch = epoch
            # …con questo R²…
            best_val_r2 = val_m['r2']
            # …e salva una copia dei pesi su CPU.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # Riparte il conteggio della pazienza.
            patience_counter = 0
        # Nessun miglioramento fuori dalla fase SWA: un'epoca di pazienza consumata.
        elif not in_swa_phase:
            # Incrementa il contatore.
            patience_counter += 1
        # Riga di avanzamento che si sovrascrive a ogni epoca (\r torna a inizio riga).
        print(f"\r  fold {fold_idx}/{k_folds}  ep {epoch:>3}/{epochs}  val R² {val_m['r2']:.3f}  "
              f"(best {best_val_r2:.3f} @ ep {best_epoch})".ljust(72), end='', flush=True)
        # Early stopping: troppe epoche senza miglioramento (mai durante lo SWA).
        if patience_counter >= patience and not in_swa_phase:
            # Esce dal ciclo delle epoche.
            break
    # 5) Se il training è arrivato alla fase SWA…
    if epoch >= swa_start_epoch:
        # …ricalcola le statistiche BatchNorm del modello mediato…
        _update_bn_gnn(tr_loader, swa_model, device=device)
        # …ne copia i pesi su CPU…
        swa_state = {k: v.detach().cpu().clone() for k, v in swa_model.module.state_dict().items()}
        # …li carica nel modello…
        model.load_state_dict(swa_state)
        # …e misura l'MSE di validazione della media.
        swa_val_mse = _evaluate(model, va_loader, device)['mse']
        # Se la media SWA batte il miglior checkpoint…
        if swa_val_mse < best_val_mse:
            # …segnala l'uso dello SWA nella stampa…
            swa_note = '  · SWA'
            # …e tiene i pesi mediati.
            best_state, best_val_mse = swa_state, swa_val_mse
        # Altrimenti…
        else:
            # …torna ai pesi del miglior checkpoint.
            model.load_state_dict(best_state)
    # Fase SWA non raggiunta (early stopping prima).
    else:
        # Carica i pesi del miglior checkpoint.
        model.load_state_dict(best_state)
    # 6) Predizioni sulla validazione (in scala standardizzata).
    va_preds = get_predictions(model, va_loader, device)[0].cpu().numpy()
    # Pesi di ogni isoforma nella loss alla fine del training (precisione = exp(-log_var)).
    task_weights = torch.exp(-loss_fn.log_vars.detach()).cpu().numpy()
    # Restituisce tutto ciò che serve al chiamante.
    return {'state': best_state,  # pesi scelti
            'scaler': scaler,  # standardizzazione dei descrittori del fold
            'target_scaler': (mu, sigma),  # standardizzazione dei pIC50 del fold
            'val_preds': va_preds * sigma + mu,  # predizioni di validazione riportate in pIC50
            'best_epoch': best_epoch, 'best_val_mse': best_val_mse,  # epoca scelta e suo MSE
            'n_epochs': epoch,  # epoche effettivamente eseguite
            'history': pd.DataFrame(history),  # storia delle epoche
            'task_weights': task_weights,  # pesi appresi per isoforma
            'swa_note': swa_note}  # nota per la stampa


# -------------------------------------------------------------------------------------
# 4. Funzione principale: tutti i seed, tutti i fold
# -------------------------------------------------------------------------------------

# Allena il D-MPNN in k-fold CV per gruppi, ripetendo il tutto con n_seeds semi diversi
# (stessi fold, inizializzazioni e augmentation diverse). Per ogni seed salva in output_dir
# storie delle epoche, curve, metriche per fold, OOF, pesi e pesi della loss.
# Restituisce i modelli e le OOF di tutti i seed: il notebook poi sceglie il seed migliore.
def train_dmpnn(cv_df, n_seeds, model_kwargs, pretrained, output_dir, device, *,
                k_folds=5, epochs=120, patience=18, batch_size=32, lr=5e-4, weight_decay=0.01,
                grad_clip=1.0, edge_drop_prob=0.10, node_mask_prob=0.05, swa_start_frac=0.75,
                swa_lr=1e-4, fold_n_trials=30, fold_seed=42, num_workers=4, warmup_epochs=5,
                train_eval_every=5, seed_base=100):
    # Crea la cartella di output se manca.
    os.makedirs(output_dir, exist_ok=True)
    # Avvisa subito: la preparazione che segue richiede qualche minuto prima della prima epoca.
    print(f'preparing graphs, folds and fold plots for {len(cv_df)} molecules (a few minutes)...')
    # pIC50 di train+val (molecole × 9).
    Y = cv_df[TARGET_COLS].values.astype(np.float32)
    # SMILES di train+val.
    smiles = cv_df[SMILES_COL].values
    # Gruppo chimico di ogni molecola.
    groups = cv_df[GROUP_COL].values
    # Grafi di tutte le molecole, calcolati una volta sola.
    cache = build_base_graph_cache(smiles)
    # Numero di feature per atomo (29), letto da un grafo qualsiasi.
    node_dim = next(iter(cache.values())).x.shape[1]
    # Numero di feature per legame (10).
    edge_dim = bond_feature_dim()
    # I k fold, identici per tutti i seed.
    folds = best_of_n_group_kfold(groups, Y, k_folds, fold_n_trials, fold_seed)
    # Risultati di tutti i seed: liste piatte con k elementi per seed, uno dopo l'altro.
    result = {'fold_states': [], 'fold_scalers': [], 'fold_target_scalers': [], 'fold_metrics': [],
              'oof_per_seed': [],  # una matrice OOF per seed
              'n_seeds': n_seeds, 'model_kwargs': model_kwargs,  # configurazione
              'node_feat_dim': node_dim, 'edge_feat_dim': edge_dim}  # dimensioni dell'input
    # Per ogni seed…
    for s in range(n_seeds):
        # …fissa il seme di PyTorch (pesi iniziali, ordine dei batch, augmentation)…
        torch.manual_seed(seed_base + s)
        # …e quello di numpy.
        np.random.seed(seed_base + s)
        # Prefisso dei file di questo seed.
        run = f'seed{s}'
        # Intestazione nella stampa.
        print(f'DMPNN · seed {s}')
        # Matrice OOF del seed, inizialmente NaN.
        oof = np.full_like(Y, np.nan, dtype=np.float32)
        # Storie, metriche e pesi della loss dei fold di questo seed.
        histories, fold_metrics, task_weights = [], [], []
        # Per ogni fold…
        for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
            # …allena e valuta il modello del fold.
            f = _train_one_fold(fold_idx, tr_idx, va_idx, smiles, Y, cache, node_dim, edge_dim,
                                model_kwargs, pretrained, device, k_folds, epochs, patience,
                                batch_size, lr, weight_decay, grad_clip, edge_drop_prob,
                                node_mask_prob, swa_start_frac, swa_lr, num_workers,
                                warmup_epochs, train_eval_every)
            # Le predizioni di validazione riempiono le righe OOF del fold.
            oof[va_idx] = f['val_preds']
            # Metriche del fold in pIC50, più le informazioni sull'epoca scelta.
            fm = {**calculate_metrics(f['val_preds'], Y[va_idx]), 'fold': fold_idx,
                  'best_epoch': f['best_epoch'], 'best_val_mse': f['best_val_mse']}
            # Storia del fold con la colonna del numero di fold.
            h = f['history'].assign(fold=fold_idx)
            # Salva la storia (usata da figures.ipynb per le curve di loss).
            h.to_csv(os.path.join(output_dir, f'{run}_cv_history_fold{fold_idx}.csv'), index=False)
            # Accumula storia, metriche e pesi della loss del seed.
            histories.append(h)
            # Metriche del fold.
            fold_metrics.append(fm)
            # Pesi della loss del fold.
            task_weights.append(f['task_weights'])
            # Accumula modello e standardizzazioni nel risultato complessivo.
            result['fold_states'].append(f['state'])
            # Standardizzazione dei descrittori.
            result['fold_scalers'].append(f['scaler'])
            # Standardizzazione dei pIC50.
            result['fold_target_scalers'].append(f['target_scaler'])
            # Riga finale del fold (sovrascrive quella di avanzamento).
            print(f"\r  fold {fold_idx}/{k_folds}  R² {fm['r2']:.3f}  MSE {fm['mse']:.3f}  "
                  f"best ep {f['best_epoch']}/{f['n_epochs']}{f['swa_note']}".ljust(72))
        # R² e MSE dei fold del seed.
        r2s, mses = [m['r2'] for m in fold_metrics], [m['mse'] for m in fold_metrics]
        # Riassunto del seed.
        print(f'  fold mean R² {np.mean(r2s):.3f} ± {np.std(r2s):.3f}  MSE {np.mean(mses):.3f}')
        # Conserva metriche e OOF del seed.
        result['fold_metrics'] += fold_metrics
        # OOF del seed.
        result['oof_per_seed'].append(oof)
        # Salva tutti gli artefatti del seed.
        _save_seed_outputs(run, output_dir, histories, fold_metrics, oof, Y, smiles, groups,
                           result['fold_states'][-k_folds:], task_weights)
    # Restituisce il risultato di tutti i seed.
    return result


# Salva gli artefatti di un seed: curve di validazione, metriche per fold, OOF, pesi dei
# modelli e pesi della loss per isoforma (con media e deviazione sui fold).
def _save_seed_outputs(run, output_dir, histories, fold_metrics, oof, Y, smiles, groups, states, task_weights):
    # Metriche per fold (senza il dettaglio per isoforma) in CSV.
    pd.DataFrame(fold_metrics).drop(columns=['per_target']).to_csv(
        os.path.join(output_dir, f'{run}_cv_results.csv'), index=False)
    # OOF, valori veri, SMILES e gruppi in un archivio numpy.
    np.savez(os.path.join(output_dir, f'{run}_oof.npz'), oof=oof, y_true=Y, smiles=smiles, groups=groups)
    # Pesi dei k modelli del seed.
    torch.save(states, os.path.join(output_dir, f'{run}_fold_states.pt'))
    # Pesi della loss: matrice fold × 9.
    tw = np.stack(task_weights, axis=0)
    # Tabella con una riga per fold.
    tw_df = pd.DataFrame(tw, columns=ISOFORMS)
    # Colonna con il numero del fold all'inizio.
    tw_df.insert(0, 'fold', list(range(1, len(tw) + 1)))
    # Righe finali con media e deviazione standard sui fold.
    extra = pd.DataFrame([{'fold': 'mean', **dict(zip(ISOFORMS, tw.mean(axis=0)))},
                          {'fold': 'std', **dict(zip(ISOFORMS, tw.std(axis=0)))}])
    # Salva la tabella completa.
    pd.concat([tw_df, extra], ignore_index=True).to_csv(
        os.path.join(output_dir, f'{run}_task_weights.csv'), index=False)


# -------------------------------------------------------------------------------------
# 5. Dopo il training: scelta del seed e predizione
# -------------------------------------------------------------------------------------

# Estrae dal risultato di train_dmpnn il solo seed s: i suoi k modelli, le sue
# standardizzazioni, le sue metriche e le sue OOF. È il formato usato da predict_dmpnn,
# dallo stacking e dal bundle di inferenza.
def seed_view(result, s):
    # Numero di fold per seed.
    k = len(result['fold_states']) // result['n_seeds']
    # Posizioni del seed s nelle liste piatte.
    sl = slice(s * k, (s + 1) * k)
    # Metriche dei fold del seed.
    fold_metrics = result['fold_metrics'][sl]
    # Dizionario del solo seed.
    return {'seed': s,  # numero del seed
            'fold_states': result['fold_states'][sl],  # pesi dei k modelli
            'fold_scalers': result['fold_scalers'][sl],  # standardizzazione dei descrittori
            'fold_target_scalers': result['fold_target_scalers'][sl],  # standardizzazione dei pIC50
            'fold_metrics': fold_metrics,  # metriche di validazione per fold
            'fold_val_mses': [m['best_val_mse'] for m in fold_metrics],  # per pesare i modelli
            'oof': result['oof_per_seed'][s],  # predizioni out-of-fold
            'model_kwargs': result['model_kwargs'],  # iperparametri della rete
            'node_feat_dim': result['node_feat_dim'], 'edge_feat_dim': result['edge_feat_dim']}


# Sceglie il seed con l'R² più alto sulle predizioni OOF, cioè usando solo la
# cross-validation. Restituisce l'indice del seed e la lista degli R² di tutti i seed.
def select_best_seed(result, Y_cv):
    # R² OOF di ogni seed.
    cv_r2 = [calculate_metrics(oof, Y_cv)['r2'] for oof in result['oof_per_seed']]
    # Indice del massimo e lista completa.
    return int(np.argmax(cv_r2)), cv_r2


# Predice i 9 pIC50 di nuove molecole con i k modelli di un seed (output di seed_view o
# parte "dmpnn" del bundle). Ogni modello usa le standardizzazioni del suo fold; le
# predizioni, riportate in pIC50, sono mediate con peso 1 / MSE di validazione (i modelli
# migliori contano di più). Gli SMILES devono essere tutti validi.
def predict_dmpnn(model_res, smiles, device, batch_size=32):
    # Target fittizi (NaN): servono solo perché i grafi abbiano la forma attesa.
    dummy_y = [np.full(len(TARGET_COLS), np.nan, dtype=np.float32) for _ in smiles]
    # Grafi delle molecole.
    graphs = prepare_graph_data(list(smiles), dummy_y)
    # Predizioni di ogni modello.
    all_preds = []
    # Per ogni modello del seed, con le sue due standardizzazioni…
    for state, scaler, (mu, sigma) in zip(model_res['fold_states'], model_res['fold_scalers'],
                                          model_res['fold_target_scalers']):
        # …ricrea la rete…
        model = DMPNNModel(model_res['node_feat_dim'], model_res['edge_feat_dim'],
                           **model_res['model_kwargs']).to(device)
        # …ci carica i pesi del fold…
        model.load_state_dict(state)
        # …copia i grafi (ogni fold ha la sua standardizzazione)…
        graphs_fold = [d.clone() for d in graphs]
        # …standardizza i descrittori come nel training di quel fold…
        apply_global_scaler(graphs_fold, scaler)
        # …predice…
        preds = get_predictions(model, DataLoader(graphs_fold, batch_size=batch_size, shuffle=False),
                                device)[0].cpu().numpy()
        # …e riporta le predizioni da scala standardizzata a pIC50.
        all_preds.append(preds * sigma + mu)
    # Matrice modelli × molecole × 9.
    P = np.stack(all_preds, axis=0)
    # Peso di ogni modello: inverso del suo MSE di validazione (minimo 1e-6).
    w = 1.0 / np.clip(np.array(model_res['fold_val_mses'], dtype=np.float64), 1e-06, None)
    # Pesi normalizzati a somma 1.
    w = w / w.sum()
    # Media pesata dei modelli, in float32.
    return np.tensordot(w, P, axes=(0, 0)).astype(np.float32)
