# D-MPNN cross-validated training, seed selection, prediction

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from torch_geometric.loader import DataLoader
from s0_config import ISOFORMS, TARGET_COLS, SMILES_COL, GROUP_COL
from s1_split import ks_log_p
from s2_metrics import calculate_metrics
from s4_dmpnn import (DMPNNModel, DynamicAugDataset, apply_global_scaler, bond_feature_dim,
                      build_base_graph_cache, fit_global_scaler, prepare_graph_data)


# Grouped cross-validation folds

# Assigns groups to k folds of similar size
def _greedy_balanced_assignment(groups, shuffled_groups, k_folds):
    sizes = pd.Series(groups).value_counts().to_dict()
    fold_counts = np.zeros(k_folds, dtype=int)
    fold_assignment = {}
    for g in sorted(shuffled_groups, key=lambda g: -sizes.get(g, 0)):
        k = int(np.argmin(fold_counts))
        fold_assignment[g] = k
        fold_counts[k] += sizes.get(g, 0)
    return fold_assignment

def _splits_from_assignment(groups, fold_assignment, k_folds):
    fold_of = np.array([fold_assignment[g] for g in groups])
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in range(k_folds)]


# Score of a fold assignment: mean log(p) of the KS test
def _score_fold_assignment(splits, Y, min_n=5):
    log_scores = []
    for tr_idx, va_idx in splits:
        for t in range(Y.shape[1]):
            tr = Y[tr_idx, t][~np.isnan(Y[tr_idx, t])]
            va = Y[va_idx, t][~np.isnan(Y[va_idx, t])]
            if len(tr) < min_n or len(va) < min_n:
                continue
            log_scores.append(ks_log_p(tr, va))
    return float(np.mean(log_scores)) if log_scores else float('-inf')


# Tries n_trials random group-to-fold assignments.
def best_of_n_group_kfold(groups, Y, k_folds, n_trials, seed, size_tol=0.1):
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    target_size = len(groups) / k_folds
    best_score, best_splits = float('-inf'), None
    for _ in range(n_trials):
        shuffled = unique_groups[rng.permutation(len(unique_groups))]
        splits = _splits_from_assignment(groups, _greedy_balanced_assignment(groups, shuffled, k_folds), k_folds)
        val_sizes = np.array([len(va) for _, va in splits])
        if (val_sizes == 0).any() or np.max(np.abs(val_sizes - target_size)) > size_tol * target_size:
            continue
        sc = _score_fold_assignment(splits, Y)
        if sc > best_score:
            best_score, best_splits = sc, splits
    if best_splits is None: 
        best_imbalance = float('inf')
        rng2 = np.random.default_rng(seed + 1)
        for _ in range(n_trials):
            shuffled = unique_groups[rng2.permutation(len(unique_groups))]
            splits = _splits_from_assignment(groups, _greedy_balanced_assignment(groups, shuffled, k_folds), k_folds)
            val_sizes = np.array([len(va) for _, va in splits])
            if (val_sizes == 0).any():
                continue
            imb = np.max(np.abs(val_sizes - target_size))
            if imb < best_imbalance:
                best_imbalance, best_splits = imb, splits
    if best_splits is None:
        raise RuntimeError('best_of_n_group_kfold: no valid split found.')
    return best_splits


# Loss and evaluation: multi-task loss with learned weights (Kendall et al. 2018). 
class UncertaintyWeightedLoss(torch.nn.Module):

    def __init__(self, n_tasks):
        super().__init__()
        self.n_tasks = n_tasks
        self.log_vars = torch.nn.Parameter(torch.zeros(n_tasks))

    def forward(self, predictions, targets):
        total = predictions.new_zeros(())
        active = 0
        for i in range(self.n_tasks):
            mask = ~torch.isnan(targets[:, i])
            if mask.sum() < 2:
                continue
            mse_i = F.mse_loss(predictions[mask, i], targets[mask, i])
            precision = torch.exp(-self.log_vars[i])
            total = total + 0.5 * precision * mse_i + 0.5 * self.log_vars[i]
            active += 1
        if active == 0:
            return predictions.new_zeros((), requires_grad=True)
        return total / active


# Runs the model over a whole loader; returns predictions and targets.
@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    preds, targets = [], []
    for data in loader:
        data = data.to(device)
        preds.append(model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_features))
        targets.append(data.y)
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


# Metrics on a loader, in standardized scale.
def _evaluate(model, loader, device):
    preds, tgts = get_predictions(model, loader, device)
    return calculate_metrics(preds.cpu().numpy(), tgts.cpu().numpy())


# Recomputes the SWA model's BatchNorm statistics
@torch.no_grad()
def _update_bn_gnn(loader, model, device):
    momenta = {}
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            momenta[m] = m.momentum
    model.train()
    for m in momenta:
        m.momentum = None 
    for data in loader:
        data = data.to(device)
        model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_features)
    for m, mom in momenta.items():
        m.momentum = mom


# Training one fold

def _train_one_fold(fold_idx, tr_idx, va_idx, smiles, Y, cache, node_dim, edge_dim, model_kwargs,
                    pretrained, device, k_folds, epochs, patience, batch_size, lr, weight_decay,
                    grad_clip, edge_drop_prob, node_mask_prob, swa_start_frac, swa_lr, num_workers,
                    warmup_epochs, train_eval_every):
    tr_y, va_y = Y[tr_idx], Y[va_idx]
    mu = np.nanmean(tr_y, axis=0)
    sigma = np.nanstd(tr_y, axis=0)
    sigma = np.where(~np.isfinite(sigma) | (sigma < 1e-06), 1.0, sigma).astype(np.float32)
    mu = np.nan_to_num(mu, nan=0.0).astype(np.float32)
    tr_graphs = prepare_graph_data(smiles[tr_idx], (tr_y - mu) / sigma, base_cache=cache)
    va_graphs = prepare_graph_data(smiles[va_idx], (va_y - mu) / sigma, base_cache=cache) 
    scaler = fit_global_scaler(tr_graphs)
    apply_global_scaler(tr_graphs, scaler)
    apply_global_scaler(va_graphs, scaler)
    pin_memory = device == 'cuda'
    tr_loader = DataLoader(DynamicAugDataset(tr_graphs, edge_drop_prob, node_mask_prob),
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           persistent_workers=num_workers > 0,
                           drop_last=len(tr_graphs) % batch_size == 1)
    va_loader = DataLoader(va_graphs, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=pin_memory, persistent_workers=num_workers > 0)
    model = DMPNNModel(node_dim, edge_dim, **model_kwargs).to(device)
    model.load_pretrained_encoder(pretrained, device=device)
    loss_fn = UncertaintyWeightedLoss(n_tasks=Y.shape[1]).to(device)
    params = list(model.parameters()) + [p for p in loss_fn.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-06)
    history = []
    best_val_mse = float('inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0
    train_m = {'mse': float('nan'), 'r2': float('nan')}
    best_val_r2 = float('nan')
    swa_note = ''
    swa_start_epoch = int(epochs * swa_start_frac)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
    for epoch in range(1, epochs + 1):
        if epoch <= warmup_epochs:  # linear LR warmup
            for pg in optimizer.param_groups:
                pg['lr'] = lr * epoch / warmup_epochs
        model.train()
        for data in tr_loader:
            data = data.to(device)
            preds = model(data.x, data.edge_index, data.edge_attr, data.batch, data.global_features)
            loss = loss_fn(preds, data.y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
        in_swa_phase = epoch >= swa_start_epoch
        if in_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        if epoch == 1 or epoch % train_eval_every == 0:  # costly, so not every epoch
            train_m = _evaluate(model, tr_loader, device)
        val_m = _evaluate(model, va_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        if epoch > warmup_epochs and not in_swa_phase:
            scheduler.step(val_m['mse'])
        history.append({'epoch': epoch,
                        'train_mse': train_m['mse'], 'val_mse': val_m['mse'],
                        'gap_mse': val_m['mse'] - train_m['mse'],  # >0 = validation worse (overfitting)
                        'train_r2': train_m['r2'], 'val_r2': val_m['r2'],
                        'gap_r2': train_m['r2'] - val_m['r2'],
                        'lr': current_lr})
        if val_m['mse'] < best_val_mse:
            best_val_mse = val_m['mse']
            best_epoch = epoch
            best_val_r2 = val_m['r2']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        elif not in_swa_phase:
            patience_counter += 1
        print(f"\r  fold {fold_idx}/{k_folds}  ep {epoch:>3}/{epochs}  val R² {val_m['r2']:.3f}  "
              f"(best {best_val_r2:.3f} @ ep {best_epoch})".ljust(72), end='', flush=True)
        if patience_counter >= patience and not in_swa_phase:  # never stop mid-SWA
            break
    if epoch >= swa_start_epoch:  # training reached the SWA phase
        _update_bn_gnn(tr_loader, swa_model, device=device)
        swa_state = {k: v.detach().cpu().clone() for k, v in swa_model.module.state_dict().items()}
        model.load_state_dict(swa_state)
        swa_val_mse = _evaluate(model, va_loader, device)['mse']
        if swa_val_mse < best_val_mse:
            swa_note = '  · SWA'
            best_state, best_val_mse = swa_state, swa_val_mse
        else:
            model.load_state_dict(best_state)
    else:  # early-stopped before reaching SWA
        model.load_state_dict(best_state)
    va_preds = get_predictions(model, va_loader, device)[0].cpu().numpy()
    task_weights = torch.exp(-loss_fn.log_vars.detach()).cpu().numpy()
    return {'state': best_state,
            'scaler': scaler,  # fold's global-descriptor standardization
            'target_scaler': (mu, sigma),  # fold's pIC50 standardization
            'val_preds': va_preds * sigma + mu,  # back to pIC50 units
            'best_epoch': best_epoch, 'best_val_mse': best_val_mse,
            'n_epochs': epoch,
            'history': pd.DataFrame(history),
            'task_weights': task_weights,  # learned per-isoform loss weights
            'swa_note': swa_note}


# Final pipeline: every seed, every fold
def train_dmpnn(cv_df, n_seeds, model_kwargs, pretrained, output_dir, device, *,
                k_folds=5, epochs=120, patience=18, batch_size=32, lr=5e-4, weight_decay=0.01,
                grad_clip=1.0, edge_drop_prob=0.10, node_mask_prob=0.05, swa_start_frac=0.75,
                swa_lr=1e-4, fold_n_trials=30, fold_seed=42, num_workers=4, warmup_epochs=5,
                train_eval_every=5, seed_base=100):
    os.makedirs(output_dir, exist_ok=True)
    print(f'preparing graphs, folds and fold plots for {len(cv_df)} molecules (a few minutes)...')
    Y = cv_df[TARGET_COLS].values.astype(np.float32)
    smiles = cv_df[SMILES_COL].values
    groups = cv_df[GROUP_COL].values
    cache = build_base_graph_cache(smiles)
    node_dim = next(iter(cache.values())).x.shape[1]
    edge_dim = bond_feature_dim()
    folds = best_of_n_group_kfold(groups, Y, k_folds, fold_n_trials, fold_seed)
    result = {'fold_states': [], 'fold_scalers': [], 'fold_target_scalers': [], 'fold_metrics': [],
              'oof_per_seed': [],
              'n_seeds': n_seeds, 'model_kwargs': model_kwargs,
              'node_feat_dim': node_dim, 'edge_feat_dim': edge_dim}
    for s in range(n_seeds):
        torch.manual_seed(seed_base + s)
        np.random.seed(seed_base + s)
        run = f'seed{s}'
        print(f'DMPNN · seed {s}')
        oof = np.full_like(Y, np.nan, dtype=np.float32)
        histories, fold_metrics, task_weights = [], [], []
        for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
            f = _train_one_fold(fold_idx, tr_idx, va_idx, smiles, Y, cache, node_dim, edge_dim,
                                model_kwargs, pretrained, device, k_folds, epochs, patience,
                                batch_size, lr, weight_decay, grad_clip, edge_drop_prob,
                                node_mask_prob, swa_start_frac, swa_lr, num_workers,
                                warmup_epochs, train_eval_every)
            oof[va_idx] = f['val_preds']
            fm = {**calculate_metrics(f['val_preds'], Y[va_idx]), 'fold': fold_idx,
                  'best_epoch': f['best_epoch'], 'best_val_mse': f['best_val_mse']}
            h = f['history'].assign(fold=fold_idx)
            h.to_csv(os.path.join(output_dir, f'{run}_cv_history_fold{fold_idx}.csv'), index=False)
            histories.append(h)
            fold_metrics.append(fm)
            task_weights.append(f['task_weights'])
            result['fold_states'].append(f['state'])
            result['fold_scalers'].append(f['scaler'])
            result['fold_target_scalers'].append(f['target_scaler'])
            print(f"\r  fold {fold_idx}/{k_folds}  R² {fm['r2']:.3f}  MSE {fm['mse']:.3f}  "
                  f"best ep {f['best_epoch']}/{f['n_epochs']}{f['swa_note']}".ljust(72))
        r2s, mses = [m['r2'] for m in fold_metrics], [m['mse'] for m in fold_metrics]
        print(f'  fold mean R² {np.mean(r2s):.3f} ± {np.std(r2s):.3f}  MSE {np.mean(mses):.3f}')
        result['fold_metrics'] += fold_metrics
        result['oof_per_seed'].append(oof)
        _save_seed_outputs(run, output_dir, histories, fold_metrics, oof, Y, smiles, groups,
                           result['fold_states'][-k_folds:], task_weights)
    return result

def _save_seed_outputs(run, output_dir, histories, fold_metrics, oof, Y, smiles, groups, states, task_weights):
    pd.DataFrame(fold_metrics).drop(columns=['per_target']).to_csv(
        os.path.join(output_dir, f'{run}_cv_results.csv'), index=False)
    np.savez(os.path.join(output_dir, f'{run}_oof.npz'), oof=oof, y_true=Y, smiles=smiles, groups=groups)
    torch.save(states, os.path.join(output_dir, f'{run}_fold_states.pt'))
    tw = np.stack(task_weights, axis=0)
    tw_df = pd.DataFrame(tw, columns=ISOFORMS)
    tw_df.insert(0, 'fold', list(range(1, len(tw) + 1)))
    extra = pd.DataFrame([{'fold': 'mean', **dict(zip(ISOFORMS, tw.mean(axis=0)))},
                          {'fold': 'std', **dict(zip(ISOFORMS, tw.std(axis=0)))}])
    pd.concat([tw_df, extra], ignore_index=True).to_csv(
        os.path.join(output_dir, f'{run}_task_weights.csv'), index=False)


# After training: seed selection and prediction
def seed_view(result, s):
    k = len(result['fold_states']) // result['n_seeds']
    sl = slice(s * k, (s + 1) * k)
    fold_metrics = result['fold_metrics'][sl]
    return {'seed': s,
            'fold_states': result['fold_states'][sl],
            'fold_scalers': result['fold_scalers'][sl],
            'fold_target_scalers': result['fold_target_scalers'][sl],
            'fold_metrics': fold_metrics,
            'fold_val_mses': [m['best_val_mse'] for m in fold_metrics],
            'oof': result['oof_per_seed'][s],
            'model_kwargs': result['model_kwargs'],
            'node_feat_dim': result['node_feat_dim'], 'edge_feat_dim': result['edge_feat_dim']}


# Picks the seed with the highest R2 on its OOF predictions
def select_best_seed(result, Y_cv):
    cv_r2 = [calculate_metrics(oof, Y_cv)['r2'] for oof in result['oof_per_seed']]
    return int(np.argmax(cv_r2)), cv_r2


# Predicts the 9 pIC50 values of new molecules with one seed's k models
def predict_dmpnn(model_res, smiles, device, batch_size=32):
    dummy_y = [np.full(len(TARGET_COLS), np.nan, dtype=np.float32) for _ in smiles]  # only the shape matters
    graphs = prepare_graph_data(list(smiles), dummy_y)
    all_preds = []
    for state, scaler, (mu, sigma) in zip(model_res['fold_states'], model_res['fold_scalers'],
                                          model_res['fold_target_scalers']):
        model = DMPNNModel(model_res['node_feat_dim'], model_res['edge_feat_dim'],
                           **model_res['model_kwargs']).to(device)
        model.load_state_dict(state)
        graphs_fold = [d.clone() for d in graphs]  # each fold standardizes independently
        apply_global_scaler(graphs_fold, scaler)
        preds = get_predictions(model, DataLoader(graphs_fold, batch_size=batch_size, shuffle=False),
                                device)[0].cpu().numpy()
        all_preds.append(preds * sigma + mu)
    P = np.stack(all_preds, axis=0)
    w = 1.0 / np.clip(np.array(model_res['fold_val_mses'], dtype=np.float64), 1e-06, None)
    w = w / w.sum()
    return np.tensordot(w, P, axes=(0, 0)).astype(np.float32)
