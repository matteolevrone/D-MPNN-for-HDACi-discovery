# Fingerprints and the group-based train/val/test split

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupShuffleSplit
from s0_config import TARGET_COLS, SMILES_COL, GROUP_COL

# Fingerprint parameters
FP_RADIUS = 3
FP_BITS = 2048

# Fingerprints

# Feature-based Morgan fingerprints
def generate_fingerprints(smiles_list):
    fmgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS, fpSize=FP_BITS,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
    fps, valid_idx = [], []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fps.append(fmgen.GetFingerprint(mol))
            valid_idx.append(i)
        else:
            print(f'[WARN] invalid SMILES at idx={i}: {smiles}')
    return fps, valid_idx


# RDKit fingerprints
def fps_to_numpy(fps):
    X = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.int8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, X[i])
    return X


# Clustering: UMAP + HDBSCAN

# Projects fingerprints with UMAP (Jaccard distance)
def cluster_hdbscan(fps, min_cluster_size, min_samples, n_components, n_neighbors, min_dist, seed):
    import umap 
    from sklearn.cluster import HDBSCAN
    reducer = umap.UMAP(n_components=n_components, metric='jaccard', n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=seed)
    emb = np.asarray(reducer.fit_transform(fps_to_numpy(fps)), dtype=np.float64)
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    raw_labels = clusterer.fit_predict(emb).astype(int)
    n_noise = int((raw_labels == -1).sum())
    n_dense = len(set(raw_labels[raw_labels >= 0]))
    labels = raw_labels.copy()
    if n_noise:
        start = (labels.max() + 1) if (labels >= 0).any() else 0
        labels[labels == -1] = np.arange(start, start + n_noise)
    _, labels = np.unique(labels, return_inverse=True)
    print(f'[cluster_hdbscan] min_cluster_size={min_cluster_size}, min_samples={min_samples}: '
          f'{n_dense} dense clusters, {n_noise} noise points ({100 * n_noise / len(emb):.1f}%) '
          f'-> {len(set(labels))} groups in total (noise as singletons)')
    report_cluster_quality(emb, raw_labels, fps)
    return labels.astype(int)


# Cluster quality
def report_cluster_quality(emb, raw_labels, fps):
    from sklearn.metrics import silhouette_score
    dense = raw_labels >= 0
    if dense.sum() < 2 or len(set(raw_labels[dense])) < 2:
        print('  [quality] too few dense clusters to validate.')
        return
    dbcv = float('nan')
    try:
        from hdbscan.validity import validity_index
        dbcv = float(validity_index(emb, raw_labels))
    except Exception:
        print('  [quality] DBCV unavailable; `pip install hdbscan`.')
    X = fps_to_numpy([fps[i] for i in np.where(dense)[0]])
    sil = float(silhouette_score(X, raw_labels[dense], metric='jaccard',
                                 sample_size=min(3000, len(X)), random_state=0))
    print(f'  [quality] DBCV (UMAP, density-based)={dbcv:+.3f}  |  '
          f'Tanimoto silhouette, dense-only (external check)={sil:+.3f}')


# Picking a split

# log(p-value) of a two-sample KS test
def ks_log_p(a, b, eps=1e-12):
    return np.log(max(float(ks_2samp(a, b).pvalue), eps))

# Score of a candidate split: mean log(p) of the KS test between train/val, train/test and val/test
def _split_score(train_df, val_df, test_df, min_n=5, eps=1e-12):
    log_scores = []
    for t in TARGET_COLS:
        tr, va, te = (d[t].dropna().values for d in (train_df, val_df, test_df))
        evaluable = len(tr) >= min_n and len(va) >= min_n and len(te) >= min_n
        for a, b in [(tr, va), (tr, te), (va, te)]:
            log_scores.append(ks_log_p(a, b, eps) if evaluable else np.log(eps))
    return float(np.mean(log_scores))

# Splits a DataFrame in two by GROUP_COL
def _group_split(df, test_size, random_state):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_a, idx_b = next(gss.split(df, df[TARGET_COLS], groups=df[GROUP_COL]))
    return df.iloc[idx_a].copy(), df.iloc[idx_b].copy()

# Tries n_trials random grouped splits (train / rest, then rest = val / test) and keeps the one with the best KS score
def best_of_n_train_val_test_split(df, test_size, val_size_within_temp, n_trials, seed):
    rng = np.random.default_rng(seed)
    best_score, best_split = float('-inf'), None
    for _ in range(n_trials):
        s = int(rng.integers(0, 10 ** 9))
        tr_df, temp_df = _group_split(df, test_size, s)
        va_df, te_df = _group_split(temp_df, val_size_within_temp, s + 1)
        sc = _split_score(tr_df, va_df, te_df)
        if sc > best_score:
            best_score, best_split = sc, (tr_df, va_df, te_df)
    return best_split


# Distribution plots

def plot_target_distributions(sets, kde_path, cdf_path, title=None):
    (fig_k, ax_k), (fig_c, ax_c) = [plt.subplots(3, 3, figsize=(18, 12)) for _ in range(2)]
    ax_k, ax_c = ax_k.flatten(), ax_c.flatten()
    for i, name in enumerate(TARGET_COLS):
        for c, (label, Y) in enumerate(sets.items()):
            if Y is None:
                continue
            v = np.sort(Y[:, i][~np.isnan(Y[:, i])])
            if len(v) > 1:
                sns.kdeplot(v, label=label, fill=True, alpha=0.3, ax=ax_k[i], warn_singular=False)
            if len(v) > 0:
                ax_c[i].plot(v, np.arange(1, len(v) + 1) / len(v), label=label, color=f'C{c}', lw=1.5)
        for ax in (ax_k[i], ax_c[i]):
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
        ax_c[i].set_ylabel('F(x)')
    for fig, axes, path in ((fig_k, ax_k, kde_path), (fig_c, ax_c, cdf_path)):
        axes[0].legend()  # only in the first panel, so it isn't repeated 9 times
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=600, bbox_inches='tight')
        plt.close(fig)


# Full split

# Pipeline
def splitter(df, output_dir, test_size=0.2, val_size_within_temp=0.5, n_trials=50, seed=42,
             umap_n_neighbors=30, umap_min_dist=0.25, umap_n_components=2,
             hdbscan_min_cluster_size=15, hdbscan_min_samples=None):
    df = df.copy()
    for c in TARGET_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')  # stray text/dirty cells -> NaN
    fps, valid_idx = generate_fingerprints(df[SMILES_COL].tolist())
    df = df.iloc[valid_idx].reset_index(drop=True)
    if len(df) <= 1:
        raise ValueError('Empty dataset after SMILES parsing.')
    df[GROUP_COL] = cluster_hdbscan(fps, hdbscan_min_cluster_size, hdbscan_min_samples,
                                    umap_n_components, umap_n_neighbors, umap_min_dist, seed)
    if df[GROUP_COL].nunique() <= 1:
        raise ValueError('Only one cluster: lower hdbscan_min_cluster_size.')
    train_df, val_df, test_df = best_of_n_train_val_test_split(df, test_size, val_size_within_temp, n_trials, seed)
    g_tr, g_va, g_te = (set(d[GROUP_COL]) for d in (train_df, val_df, test_df))
    if g_tr & g_va or g_tr & g_te or g_va & g_te:
        raise AssertionError('Group-level data leakage detected between train/val/test.')
    os.makedirs(output_dir, exist_ok=True)
    sets = {name: d[TARGET_COLS].values.astype(np.float32)
            for name, d in (('Train', train_df), ('Val', val_df), ('Test', test_df))}
    plot_target_distributions(sets, os.path.join(output_dir, 'split_distributions.png'),
                              os.path.join(output_dir, 'split_cdfs.png'))
    for name, d in (('train', train_df), ('val', val_df), ('test', test_df)):
        d.to_csv(os.path.join(output_dir, f'{name}_set_multitarget.csv'), index=False)
    return train_df, val_df, test_df


# Split-vs-random visualization 

SPLIT_STYLE = {
    'Train': dict(c='#9bc1bc', s=40, alpha=0.50, zorder=1, linewidths=0),
    'Val': dict(c='#f4f1bb', s=40, alpha=0.85, zorder=2, linewidths=0.5, edgecolors='white'),
    'Test': dict(c='#ed6a5a', s=40, alpha=0.90, zorder=3, linewidths=0.6, edgecolors='white'),
}

def nn_median(fps, sv):
    fit = [fps[i] for i in np.where((sv == 'Train') | (sv == 'Val'))[0]]
    te = np.where(sv == 'Test')[0]
    nn = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], fit)) for i in te])
    return float(np.median(nn))

def draw_split(ax, emb, sv, title):
    for sp in ['Train', 'Val', 'Test']:
        m = sv == sp
        ax.scatter(emb[m, 0], emb[m, 1], label=sp, rasterized=True, **SPLIT_STYLE[sp])
    ax.set_title(title, fontsize=17, fontweight='bold', pad=14, color='#1f1f1f')
    leg = ax.legend(loc='upper left', fontsize=17, frameon=True, borderpad=1.0,
                    handletextpad=0.8, labelspacing=0.85, markerscale=2.6, fancybox=True)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#dcdcdc')
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_alpha(0.92)
    ax.tick_params(colors='#888888', labelsize=11, length=4)
    ax.margins(0.03)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color('#bbbbbb')
    ax.label_outer()
