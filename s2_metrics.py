# Evaluation metrics

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from s0_config import ISOFORMS, CLASS_I_ISOFORMS, UNDERPOWERED_N


# Plot style

# Colour-blind-safe colours
C_BLUE, C_ORANGE, C_AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK_SOFT, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

def _style_axes(ax, grid_axis='y'):
    ax.set_facecolor(SURFACE)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=11)


# Bar panel

def _bar_panel(ax, df, col, title, ylabel, ref=None, ref_label=None, low_n=25):
    x = np.arange(len(df)); v = df[col].to_numpy(dtype=float)
    colors = [C_BLUE if n >= low_n else '#9ec5f4' for n in df['n']]
    ax.bar(x, v, width=0.68, color=colors, zorder=3)
    for xi, vi in zip(x, v):
        if not np.isnan(vi):
            ax.annotate(f'{vi:.2f}', (xi, vi), xytext=(0, 4 if vi >= 0 else -4), textcoords='offset points',
                        ha='center', va='bottom' if vi >= 0 else 'top', fontsize=9.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\nn={n}' for i, n in zip(df['isoform'], df['n'])], fontsize=9.5)
    if ref is not None:
        ax.axhline(ref, color=INK_SOFT, lw=1, ls='--' if ref_label else '-', zorder=4)
        if ref_label:
            ax.annotate(ref_label, (1, ref), xycoords=('axes fraction', 'data'), xytext=(5, 0),
                        textcoords='offset points', ha='left', va='center', fontsize=9.5, color=INK_SOFT,
                        annotation_clip=False)
    ax.set_title(title, loc='left', fontsize=13, weight='bold', color=INK)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=11)
    ax.margins(y=0.12)
    _style_axes(ax)


# Closes a bar figure
def _finish_bars(fig, output_path, low_n=25):
    fig.text(0.01, -0.02, f'Lighter bars: isoforms with n < {low_n} measurements.', fontsize=9.5, color=INK_SOFT)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


# Numeric metrics

# Spearman correlation (undefined (NaN) below 3 points)
def _spearman(t, p):
    return float(spearmanr(t, p).correlation) if len(t) >= 3 else float('nan')

# Regression metrics
def calculate_metrics(predictions, targets):
    pred, true = np.asarray(predictions), np.asarray(targets)
    p_all, t_all = pred.flatten(), true.flatten()
    ok = ~np.isnan(t_all)
    p_all, t_all = p_all[ok], t_all[ok]
    if len(p_all) < 2:
        return {'r2': 0.0, 'pearson_r': 0.0, 'pearson_p': 1.0, 'spearman_r': 0.0,
                'mae': 0.0, 'rmse': 0.0, 'mse': 0.0, 'per_target': []}
    mse = mean_squared_error(t_all, p_all)
    pr, pp = pearsonr(t_all, p_all)
    metrics = {'r2': r2_score(t_all, p_all),
               'pearson_r': pr,
               'pearson_p': pp,
               'spearman_r': _spearman(t_all, p_all),
               'mae': mean_absolute_error(t_all, p_all),
               'mse': mse,
               'rmse': float(np.sqrt(mse)),
               'per_target': []}
    for i in range(true.shape[1]):
        m = ~np.isnan(true[:, i])
        t, p = true[m, i], pred[m, i]
        if len(t) < 2:
            metrics['per_target'].append({'r2': np.nan, 'mae': np.nan, 'rmse': np.nan,
                                          'pearson_r': np.nan, 'spearman_r': np.nan, 'n': int(len(t))})
            continue
        metrics['per_target'].append({'r2': r2_score(t, p),
                                      'mae': mean_absolute_error(t, p),
                                      'rmse': float(np.sqrt(mean_squared_error(t, p))),
                                      'pearson_r': pearsonr(t, p)[0],
                                      'spearman_r': _spearman(t, p),
                                      'n': int(len(t))})
    return metrics


# One row per isoform. If output_path is given, saves the table as CSV and a bar-chart figure.
def per_isoform_report(y_true, y_pred, output_path=None):
    per_target = calculate_metrics(y_pred, y_true)['per_target']
    cols = ['n', 'r2', 'mae', 'rmse', 'pearson_r', 'spearman_r']
    df = pd.DataFrame([{'isoform': name, **{c: m[c] for c in cols}} for name, m in zip(ISOFORMS, per_target)])
    if not output_path:
        return df
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=SURFACE)
    _bar_panel(axes[0], df, 'r2', 'R² per isoform', 'R²', ref=0)
    _bar_panel(axes[1], df, 'mae', 'MAE per isoform (pIC50 units)', 'MAE')
    fig.tight_layout()
    _finish_bars(fig, output_path.replace('.csv', '.png'))
    return df


# Parity plots
def parity_plots(y_true, y_pred, output_path):
    fig, axes = plt.subplots(3, 3, figsize=(13, 12.5), facecolor=SURFACE)
    axes = axes.flatten()
    for i, name in enumerate(ISOFORMS):
        m = ~np.isnan(y_true[:, i]); ax = axes[i]
        if m.sum() < 2:
            ax.text(0.5, 0.5, f'{name}\nn<2', ha='center', va='center', transform=ax.transAxes, color=INK_SOFT)
            ax.axis('off')
            continue
        t, p = y_true[m, i], y_pred[m, i]
        lo, hi = min(t.min(), p.min()) - 0.2, max(t.max(), p.max()) + 0.2
        ax.plot([lo, hi], [lo, hi], color=INK_SOFT, lw=1.2, ls='--', zorder=2)
        ax.scatter(t, p, s=26, color=C_BLUE, alpha=0.6, edgecolor='white', linewidth=0.4, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
        ax.set_title(name, loc='left', fontsize=13, weight='bold', color=INK)
        ax.text(0.04, 0.95, f'R² = {r2_score(t, p):.2f}\nn = {m.sum()}', transform=ax.transAxes, va='top',
                fontsize=11, color=INK)
        ax.set_xlabel('True pIC50', color=INK_SOFT, fontsize=10.5)
        ax.set_ylabel('Predicted pIC50', color=INK_SOFT, fontsize=10.5)
        _style_axes(ax, grid_axis='both')
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)


# Top-k recall
def _topk_recall(t, p, k_frac):
    k = max(1, int(round(len(t) * k_frac)))
    top_true = set(np.argsort(-t)[:k])
    top_pred = set(np.argsort(-p)[:k])
    return len(top_true & top_pred) / k


# Per-isoform ranking metrics. If output_path is given, saves the table and a figure (Spearman and EF@10%).
def ranking_metrics_per_isoform(y_true, y_pred, output_path=None):
    k_fractions = (0.01, 0.05, 0.1)
    rows = []
    for i, name in enumerate(ISOFORMS):
        m = ~np.isnan(y_true[:, i])
        n = int(m.sum())
        row = {'isoform': name, 'n': n}
        if n < 5:  # ranking is meaningless below this
            row.update({'spearman_rho': np.nan, 'kendall_tau': np.nan})
            for k in k_fractions:
                row.update({f'recall@{int(k * 100)}pc': np.nan, f'EF@{int(k * 100)}pc': np.nan})
        else:
            t, p = y_true[m, i], y_pred[m, i]
            row.update({'spearman_rho': spearmanr(t, p).correlation, 'kendall_tau': kendalltau(t, p).correlation})
            for k in k_fractions:
                rec = _topk_recall(t, p, k)
                row.update({f'recall@{int(k * 100)}pc': rec, f'EF@{int(k * 100)}pc': rec / k})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not output_path:
        return df
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=SURFACE)
    _bar_panel(axes[0], df, 'spearman_rho', 'Spearman ρ per isoform', 'Spearman ρ', ref=0)
    _bar_panel(axes[1], df, 'EF@10pc', 'Enrichment factor @ 10%', 'EF@10%', ref=1.0, ref_label='random selection')
    fig.tight_layout()
    _finish_bars(fig, output_path.replace('.csv', '.png'))
    return df


# Regression + ranking table for one model. Used by the notebook to build the LightGBM/D-MPNN/stacking comparison.
def full_per_isoform_table(y_true, y_pred, ranking_df=None, underpowered_n=UNDERPOWERED_N):
    reg = per_isoform_report(y_true, y_pred)
    rnk = ranking_df if ranking_df is not None else ranking_metrics_per_isoform(y_true, y_pred)
    tab = reg[['isoform', 'n', 'r2', 'rmse', 'mae', 'pearson_r', 'spearman_r']].copy()
    tab['EF@10%'] = rnk['EF@10pc'].values
    tab.columns = ['Isoform', 'n', 'R2', 'RMSE', 'MAE', 'Pearson', 'Spearman', 'EF@10%']
    flag_col = f'n<{underpowered_n}'
    tab[flag_col] = tab['n'] < underpowered_n
    mean_row = {'Isoform': 'MEAN', 'n': int(tab['n'].sum()), flag_col: False}
    for c in ['R2', 'RMSE', 'MAE', 'Pearson', 'Spearman', 'EF@10%']:
        mean_row[c] = float(np.nanmean(tab[c].values))
    return pd.concat([tab, pd.DataFrame([mean_row])], ignore_index=True)


# Summary plots (figures.ipynb)

def r2_vs_ranking_plot(r2_df, ranking_df, output_path):
    merged = r2_df[['isoform', 'r2']].merge(ranking_df[['isoform', 'n', 'spearman_rho', 'EF@10pc']], on='isoform')
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), facecolor=SURFACE)
    _bar_panel(axes[0], merged, 'r2', 'R² (regression)', 'R²', ref=0)
    _bar_panel(axes[1], merged, 'spearman_rho', 'Spearman ρ (ranking)', 'Spearman ρ', ref=0)
    _bar_panel(axes[2], merged, 'EF@10pc', 'EF @ 10% (screening)', 'Enrichment factor', ref=1.0,
               ref_label='random selection')
    fig.tight_layout()
    _finish_bars(fig, output_path)


# Grouped bar chart of test R2 per isoform across models.
def r2_comparison_plot(df, output_path, models=('LGBM', 'DMPNN', 'Stacking'),
                       labels=('LightGBM (baseline)', 'D-MPNN', 'Stacking'), underpowered_n=25):
    iso = list(df['isoform']); n_test = df['n_test'].to_numpy()
    x = np.arange(len(iso) + 1); width = 0.26
    colors = [C_BLUE, C_ORANGE, C_AQUA]
    fig, ax = plt.subplots(figsize=(16, 6.2), facecolor=SURFACE)
    for i, name in enumerate(iso):  # light band behind Class I isoforms
        if name in CLASS_I_ISOFORMS:
            ax.axvspan(i - 0.5, i + 0.5, color=C_BLUE, alpha=0.06, lw=0, zorder=0)
    ax.axvline(len(iso) - 0.5, color=GRID, lw=1.2, zorder=1)  # separator before the mean group
    for k, (m, lab) in enumerate(zip(models, labels)):
        vals = df[f'R2_{m}'].to_numpy(dtype=float); vals = np.append(vals, np.nanmean(vals))
        off = (k - (len(models) - 1) / 2) * width
        ax.bar(x + off, vals, width * 0.92, color=colors[k], label=lab, zorder=3)
        for xi, vi in zip(x + off, vals):
            ax.annotate(f'{vi:.2f}', (xi, vi), xytext=(0, 3 if vi >= 0 else -3), textcoords='offset points',
                        ha='center', va='bottom' if vi >= 0 else 'top', fontsize=8.5, color=INK)
    ax.axhline(0, color=INK_SOFT, lw=1, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}{"*" if t < underpowered_n else ""}' for n, t in zip(iso, n_test)] + ['Mean'],
                       fontsize=12, color=INK)
    ax.set_ylabel('Test R²', color=INK_SOFT, fontsize=13)
    ax.set_title('Test R² per isoform: baseline vs D-MPNN vs stacking', loc='left', fontsize=16,
                 weight='bold', color=INK, pad=34)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(models), frameon=False, fontsize=12,
              labelcolor=INK, handlelength=1.2)
    ax.margins(y=0.1)
    _style_axes(ax)
    fig.text(0.07, -0.01, f'Shaded: Class I isoforms.   * n_test < {underpowered_n}.',
             fontsize=10.5, color=INK_SOFT)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
