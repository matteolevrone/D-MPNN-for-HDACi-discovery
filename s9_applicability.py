# Applicability domain: similarity to the training set, error calibration and prediction clipping

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import DataStructs
from s0_config import TARGET_COLS, SMILES_COL, GROUP_COL
from s1_split import generate_fingerprints
from s2_metrics import _style_axes, _spearman, C_BLUE, C_ORANGE, INK, INK_SOFT, GRID

# Similarity bin edges for the error calibration; the last bin collects everything above 0.8
SIM_BINS = (0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.80, 1.01)

# A compound is out of domain once the measured RMSE in its similarity bin exceeds this (pIC50 units)
RMSE_TOLERANCE = 1.0


# Nearest-neighbour similarity

# Maximum Tanimoto similarity of each query to the reference set.
# When both group arrays are given, reference compounds sharing the query's group are excluded: with
# GroupKFold the same cluster never straddles a fold, so this reproduces the similarity each
# out-of-fold prediction actually saw.
def nn_similarity(query_smiles, reference_smiles, query_groups=None, reference_groups=None):
    q_fps, q_valid = generate_fingerprints([str(s) for s in query_smiles])
    r_fps, r_valid = generate_fingerprints([str(s) for s in reference_smiles])
    if not q_fps or not r_fps:
        return np.zeros(0, dtype=np.float32), []
    exclude = query_groups is not None and reference_groups is not None
    r_groups = np.asarray(reference_groups)[r_valid] if exclude else None
    q_groups = np.asarray(query_groups)[q_valid] if exclude else None
    sim = np.zeros(len(q_fps), dtype=np.float32)
    for i, fp in enumerate(q_fps):
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, r_fps), dtype=np.float32)
        if exclude:
            s = s[r_groups != q_groups[i]]
        sim[i] = s.max() if s.size else 0.0
    return sim, q_valid


# Error calibration

# Measured error per similarity bin. One row per bin, over the non-missing entries of y_true.
def calibrate(similarity, y_true, y_pred, bins=SIM_BINS):
    sim, true, pred = np.asarray(similarity), np.asarray(y_true), np.asarray(y_pred)
    idx = np.digitize(sim, np.asarray(bins)[1:-1], right=False)
    rows = []
    for b in range(len(bins) - 1):
        m = idx == b
        t, p = true[m].ravel(), pred[m].ravel()
        ok = ~np.isnan(t)
        t, p = t[ok], p[ok]
        row = {'bin': f'{bins[b]:.2f}-{min(bins[b + 1], 1.0):.2f}', 'sim_low': bins[b],
               'sim_high': min(bins[b + 1], 1.0), 'n_mol': int(m.sum()), 'n_obs': int(len(t))}
        if len(t) >= 2:
            resid = t - p
            row.update({'mean_sim': float(sim[m].mean()), 'rmse': float(np.sqrt((resid ** 2).mean())),
                        'mae': float(np.abs(resid).mean()), 'bias': float(resid.mean()),
                        'spearman_r': _spearman(t, p),
                        'r2': float(1 - (resid ** 2).sum() / ((t - t.mean()) ** 2).sum())
                        if t.var() > 0 else np.nan})
        else:
            row.update({'mean_sim': np.nan, 'rmse': np.nan, 'mae': np.nan, 'bias': np.nan,
                        'spearman_r': np.nan, 'r2': np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


# Lowest similarity at which the measured RMSE is still within tolerance.
# Returns the upper edge of the highest-similarity bin that breaches it, so every bin below the
# threshold is one where the error was actually measured to be too large.
def choose_threshold(cal, tolerance=RMSE_TOLERANCE):
    bad = cal[(cal['n_obs'] >= 2) & (cal['rmse'] > tolerance)]
    return float(bad['sim_high'].max()) if len(bad) else 0.0


# Error-vs-similarity calibration plot
def plot_calibration(cal, threshold, output_path, title='Applicability domain calibration',
                     tolerance=RMSE_TOLERANCE):
    d = cal[cal['n_obs'] >= 2]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    _style_axes(ax)
    ax.plot(d['mean_sim'], d['rmse'], marker='o', color=C_BLUE, lw=2, zorder=3, label='measured RMSE')
    for _, r in d.iterrows():
        ax.annotate(f"n={r['n_obs']}", (r['mean_sim'], r['rmse']), xytext=(0, 7),
                    textcoords='offset points', ha='center', fontsize=8.5, color=INK_SOFT)
    ax.axhline(tolerance, color=INK_SOFT, lw=1, ls='--', zorder=2,
               label=f'tolerance = {tolerance:.2f} pIC50')
    if threshold > 0:
        ax.axvspan(0, threshold, color=C_ORANGE, alpha=0.12, zorder=1)
        ax.axvline(threshold, color=C_ORANGE, lw=1.6, zorder=4,
                   label=f'domain threshold = {threshold:.2f}')
    ax.set_xlabel('nearest-neighbour Tanimoto similarity to training set', fontsize=11, color=INK)
    ax.set_ylabel('RMSE (pIC50)', fontsize=11, color=INK)
    ax.set_title(title, fontsize=12.5, weight='bold', color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9.5, loc='upper right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.show()


# Prediction clipping

# pIC50 range spanned by the observed labels; predictions outside it are extrapolation artefacts.
def observed_bounds(y_true, margin=0.0):
    v = np.asarray(y_true, dtype=float).ravel()
    v = v[~np.isnan(v)]
    return float(v.min() - margin), float(v.max() + margin)


# Clips predictions into bounds and reports which rows were touched
def clip_predictions(preds, bounds):
    p = np.asarray(preds, dtype=np.float32)
    lo, hi = bounds
    clipped = np.clip(p, lo, hi)
    touched = np.any((p < lo) | (p > hi), axis=1) if p.ndim == 2 else (p < lo) | (p > hi)
    return clipped, touched


# Screening annotation

# Adds the applicability-domain columns to a screening table.
# preds is the (n, 9) matrix of per-isoform predictions aligned with df.
def annotate(df, preds, reference_smiles, threshold, bounds, smiles_col=SMILES_COL,
             reference_groups=None, query_groups=None):
    out = df.reset_index(drop=True).copy()
    sim, valid = nn_similarity(out[smiles_col], reference_smiles,
                               query_groups=query_groups, reference_groups=reference_groups)
    out['nn_tanimoto'] = np.nan
    out.loc[valid, 'nn_tanimoto'] = sim
    out['in_domain'] = out['nn_tanimoto'] >= threshold
    clipped, touched = clip_predictions(preds, bounds)
    out['pred_clipped'] = touched
    for j, col in enumerate(TARGET_COLS):
        out[f'pred_{col}'] = clipped[:, j]
    return out


# Reports how many compounds survive the domain filter
def domain_summary(annotated, label='library'):
    n = len(annotated)
    inside = int(annotated['in_domain'].sum())
    clipped = int(annotated['pred_clipped'].sum())
    print(f'[AD] {label}: N={n}  in domain={inside} ({100 * inside / n:.1f}%)  '
          f'out of domain={n - inside} ({100 * (n - inside) / n:.1f}%)  clipped={clipped}')
    print(f"[AD] {label}: NN-Tanimoto median={annotated['nn_tanimoto'].median():.3f}  "
          f"q90={annotated['nn_tanimoto'].quantile(0.90):.3f}  max={annotated['nn_tanimoto'].max():.3f}")
    return {'n': n, 'in_domain': inside, 'clipped': clipped}


# Entire applicability-domain analysis

# Calibrates the domain on out-of-fold and test predictions, then applies it to a screening library.
# The two calibration sources are complementary: the test set measures the error on a genuinely
# held-out split but its similarity never drops below ~0.27, so only the out-of-fold predictions
# (with same-cluster neighbours excluded) populate the low-similarity bins where the threshold falls.
def run_applicability(cv_df, test_df, oof_path, bundle, library_smiles, output_dir,
                      device='cpu', tolerance=RMSE_TOLERANCE, batch_size=32):
    from s5_training import predict_dmpnn  # imported here so the helpers above stay torch-free

    oof_npz = np.load(oof_path, allow_pickle=True)
    oof, y_oof, smiles_oof, groups_oof = (oof_npz['oof'], oof_npz['y_true'],
                                          oof_npz['smiles'], oof_npz['groups'])
    print('[AD] nearest-neighbour similarity on out-of-fold predictions (same cluster excluded)...')
    sim_oof, valid_oof = nn_similarity(smiles_oof, smiles_oof,
                                       query_groups=groups_oof, reference_groups=groups_oof)

    print('[AD] D-MPNN predictions on the test set...')
    preds_test = predict_dmpnn(bundle['dmpnn'], test_df[SMILES_COL].tolist(), device,
                               batch_size=batch_size)
    y_test = test_df[TARGET_COLS].values.astype(np.float32)
    sim_test, valid_test = nn_similarity(test_df[SMILES_COL], cv_df[SMILES_COL])

    cal = calibrate(np.concatenate([sim_oof, sim_test]),
                    np.vstack([y_oof[valid_oof], y_test[valid_test]]),
                    np.vstack([oof[valid_oof], preds_test[valid_test]]))
    threshold = choose_threshold(cal, tolerance)
    bounds = observed_bounds(np.vstack([y_oof, y_test]))
    print(f'[AD] threshold = {threshold:.2f} NN-Tanimoto (RMSE tolerance {tolerance:.2f} pIC50)')
    print(f'[AD] prediction bounds from observed labels = [{bounds[0]:.2f}, {bounds[1]:.2f}]')

    os.makedirs(output_dir, exist_ok=True)
    cal.to_csv(os.path.join(output_dir, 'calibration.csv'), index=False)
    plot_calibration(cal, threshold, os.path.join(output_dir, 'calibration.png'), tolerance=tolerance)

    print('[AD] D-MPNN predictions on the screening library...')
    lib = pd.DataFrame({SMILES_COL: [str(s) for s in library_smiles]})
    preds_lib = predict_dmpnn(bundle['dmpnn'], lib[SMILES_COL].tolist(), device, batch_size=batch_size)
    annotated = annotate(lib, preds_lib, cv_df[SMILES_COL], threshold, bounds)
    summary = domain_summary(annotated, 'screening library')
    annotated.to_csv(os.path.join(output_dir, 'library_annotated.csv'), index=False)
    return {'calibration': cal, 'threshold': threshold, 'bounds': bounds,
            'annotated': annotated, 'summary': summary,
            'similarity': {'oof': sim_oof, 'test': sim_test}}
