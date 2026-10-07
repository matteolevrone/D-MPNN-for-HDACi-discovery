# =====================================================================================
# PASSO 2 di 7 — METRICHE DI VALUTAZIONE E GRAFICI DEI RISULTATI
#
# Ruolo nella pipeline: confronta i pIC50 predetti con quelli veri. Viene usato da tutti
# i passi successivi: dalla baseline LightGBM (passo 3), dal training del D-MPNN (passo 5,
# per scegliere l'epoca migliore) e dal notebook per le tabelle finali.
#
# Convenzione: y_true e y_pred sono matrici (molecole × 9 isoforme); i NaN di y_true sono
# le misure mancanti e vengono sempre esclusi dal calcolo.
# Le funzioni sono in ordine di primo utilizzo nel notebook.
# =====================================================================================

# Composizione dei percorsi dei file in cui salvare tabelle e grafici.
import os
# Calcolo numerico su array.
import numpy as np
# Tabelle dei risultati per isoforma.
import pandas as pd
# Grafici.
import matplotlib.pyplot as plt
# Correlazioni: Pearson (lineare), Spearman e Kendall (sull'ordine dei valori).
from scipy.stats import pearsonr, spearmanr, kendalltau
# Metriche di regressione: R², errore assoluto medio, errore quadratico medio.
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Nomi delle isoforme per etichettare righe e grafici.
from s0_config import ISOFORMS


# -------------------------------------------------------------------------------------
# Stile comune dei grafici
# -------------------------------------------------------------------------------------

# Colori delle serie (blu, arancione, acqua) e dell'inchiostro; validati per daltonismo.
C_BLUE, C_ORANGE, C_AQUA = '#2a78d6', '#eb6834', '#1baf7a'
# Testo principale, testo secondario, griglia e sfondo.
INK, INK_SOFT, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
# Isoforme di Classe I, evidenziate con una banda leggera nei grafici di confronto.
CLASS_I_ISOFORMS = ['HDAC1', 'HDAC2', 'HDAC3', 'HDAC8']


# Applica a un riquadro lo stile comune: niente bordi tranne l'asse x, griglia orizzontale leggera.
def _style_axes(ax, grid_axis='y'):
    # Sfondo chiaro.
    ax.set_facecolor(SURFACE)
    # Toglie i bordi superiore, destro e sinistro.
    ax.spines[['top', 'right', 'left']].set_visible(False)
    # Asse x tenue.
    ax.spines['bottom'].set_color(GRID)
    # Griglia leggera, dietro ai dati.
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    # Niente trattini, testo secondario.
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=11)


# Riquadro a barre con un valore per isoforma, scritto sopra la barra. Le isoforme con
# poche misure (n < low_n) sono più chiare e il loro n è scritto sotto il nome.
def _bar_panel(ax, df, col, title, ylabel, ref=None, ref_label=None, low_n=25):
    # Posizioni sull'asse x e valori.
    x = np.arange(len(df)); v = df[col].to_numpy(dtype=float)
    # Barre blu piene, più chiare quando n è basso.
    colors = [C_BLUE if n >= low_n else '#9ec5f4' for n in df['n']]
    ax.bar(x, v, width=0.68, color=colors, zorder=3)
    # Valore sopra (o sotto, se negativo) ogni barra.
    for xi, vi in zip(x, v):
        if not np.isnan(vi):
            ax.annotate(f'{vi:.2f}', (xi, vi), xytext=(0, 4 if vi >= 0 else -4), textcoords='offset points',
                        ha='center', va='bottom' if vi >= 0 else 'top', fontsize=9.5, color=INK)
    # Nomi delle isoforme con n sotto.
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}\nn={n}' for i, n in zip(df['isoform'], df['n'])], fontsize=9.5)
    # Linea di riferimento (zero o selezione casuale).
    if ref is not None:
        ax.axhline(ref, color=INK_SOFT, lw=1, ls='--' if ref_label else '-', zorder=4)
        if ref_label:
            ax.annotate(ref_label, (1, ref), xycoords=('axes fraction', 'data'), xytext=(5, 0),
                        textcoords='offset points', ha='left', va='center', fontsize=9.5, color=INK_SOFT,
                        annotation_clip=False)
    # Titolo a sinistra e etichetta dell'asse y.
    ax.set_title(title, loc='left', fontsize=13, weight='bold', color=INK)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=11)
    # Margine in alto per le etichette.
    ax.margins(y=0.12)
    _style_axes(ax)


# Chiude una figura di barre: nota sulle barre chiare e salvataggio.
def _finish_bars(fig, output_path, low_n=25):
    # Nota in basso a sinistra.
    fig.text(0.01, -0.02, f'Lighter bars: isoforms with n < {low_n} measurements.', fontsize=9.5, color=INK_SOFT)
    # Salva con sfondo chiaro.
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    # Chiude la figura.
    plt.close(fig)



# -------------------------------------------------------------------------------------
# 1. Metriche numeriche
# -------------------------------------------------------------------------------------

# Correlazione di Spearman (quanto l'ordine dei valori predetti segue quello dei veri);
# con meno di 3 valori non è definita e vale NaN.
def _spearman(t, p):
    # Calcola la correlazione solo se ci sono almeno 3 coppie di valori.
    return float(spearmanr(t, p).correlation) if len(t) >= 3 else float('nan')


# Metriche di regressione su tutte le misure insieme ("pooled") e, nella chiave
# 'per_target', separatamente per ognuna delle 9 isoforme.
def calculate_metrics(predictions, targets):
    # Converte gli input in array numpy (accetta anche liste o altri array).
    pred, true = np.asarray(predictions), np.asarray(targets)
    # Appiattisce le matrici in un unico vettore di misure.
    p_all, t_all = pred.flatten(), true.flatten()
    # Maschera delle misure presenti (pIC50 vero non mancante).
    ok = ~np.isnan(t_all)
    # Tiene solo le misure presenti.
    p_all, t_all = p_all[ok], t_all[ok]
    # Con meno di due misure le metriche non sono definite: restituisce zeri.
    if len(p_all) < 2:
        # Dizionario di metriche nulle con la stessa struttura di quello normale.
        return {'r2': 0.0, 'pearson_r': 0.0, 'pearson_p': 1.0, 'spearman_r': 0.0,
                'mae': 0.0, 'rmse': 0.0, 'mse': 0.0, 'per_target': []}
    # Errore quadratico medio su tutte le misure.
    mse = mean_squared_error(t_all, p_all)
    # Correlazione di Pearson e suo p-value.
    pr, pp = pearsonr(t_all, p_all)
    # Raccoglie le metriche pooled (l'ordine delle chiavi è quello delle colonne dei CSV).
    metrics = {'r2': r2_score(t_all, p_all),  # frazione di varianza spiegata (1 = perfetto)
               'pearson_r': pr,  # correlazione lineare
               'pearson_p': pp,  # significatività della correlazione
               'spearman_r': _spearman(t_all, p_all),  # correlazione sull'ordine
               'mae': mean_absolute_error(t_all, p_all),  # errore medio in unità di pIC50
               'mse': mse,  # errore quadratico medio
               'rmse': float(np.sqrt(mse)),  # radice dell'MSE, in unità di pIC50
               'per_target': []}  # verrà riempita qui sotto, un dizionario per isoforma
    # Per ogni isoforma (colonna i)…
    for i in range(true.shape[1]):
        # …maschera delle molecole misurate su quell'isoforma…
        m = ~np.isnan(true[:, i])
        # …e i loro valori veri e predetti.
        t, p = true[m, i], pred[m, i]
        # Con meno di due misure le metriche dell'isoforma sono NaN.
        if len(t) < 2:
            # Aggiunge la riga vuota con il numero di misure.
            metrics['per_target'].append({'r2': np.nan, 'mae': np.nan, 'rmse': np.nan,
                                          'pearson_r': np.nan, 'spearman_r': np.nan, 'n': int(len(t))})
            # Passa all'isoforma successiva.
            continue
        # Aggiunge le metriche di questa isoforma.
        metrics['per_target'].append({'r2': r2_score(t, p),
                                      'mae': mean_absolute_error(t, p),
                                      'rmse': float(np.sqrt(mean_squared_error(t, p))),
                                      'pearson_r': pearsonr(t, p)[0],
                                      'spearman_r': _spearman(t, p),
                                      'n': int(len(t))})  # quante molecole misurate
    # Restituisce tutte le metriche.
    return metrics


# Tabella con una riga per isoforma (n, R², MAE, RMSE, Pearson, Spearman). Se si passa
# output_path salva la tabella in CSV e, accanto, un grafico a barre di R² e MAE.
def per_isoform_report(y_true, y_pred, output_path=None):
    # Metriche separate per isoforma calcolate dalla funzione sopra.
    per_target = calculate_metrics(y_pred, y_true)['per_target']
    # Colonne da riportare, nell'ordine voluto.
    cols = ['n', 'r2', 'mae', 'rmse', 'pearson_r', 'spearman_r']
    # Una riga per isoforma: nome più le metriche scelte.
    df = pd.DataFrame([{'isoform': name, **{c: m[c] for c in cols}} for name, m in zip(ISOFORMS, per_target)])
    # Se non è richiesto il salvataggio la funzione finisce qui.
    if not output_path:
        # Restituisce solo la tabella.
        return df
    # Crea la cartella di destinazione se manca.
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    # Salva la tabella.
    df.to_csv(output_path, index=False)
    # Figura con due grafici affiancati: R² e MAE per isoforma.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=SURFACE)
    _bar_panel(axes[0], df, 'r2', 'R² per isoform', 'R²', ref=0)
    _bar_panel(axes[1], df, 'mae', 'MAE per isoform (pIC50 units)', 'MAE')
    # Salva accanto al CSV, con lo stesso nome ma estensione .png.
    fig.tight_layout()
    _finish_bars(fig, output_path.replace('.csv', '.png'))
    # Restituisce la tabella.
    return df


# Grafici di parità: per ogni isoforma, pIC50 vero (x) contro predetto (y). Un modello
# perfetto mette tutti i punti sulla diagonale rossa tratteggiata.
def parity_plots(y_true, y_pred, output_path):
    # Griglia 3×3, un riquadro per isoforma, con spazio per le etichette.
    fig, axes = plt.subplots(3, 3, figsize=(13, 12.5), facecolor=SURFACE)
    # Lista semplice di 9 riquadri.
    axes = axes.flatten()
    # Per ogni isoforma…
    for i, name in enumerate(ISOFORMS):
        # …maschera delle molecole misurate e riquadro corrispondente.
        m = ~np.isnan(y_true[:, i]); ax = axes[i]
        # Con meno di due punti non c'è niente da disegnare.
        if m.sum() < 2:
            ax.text(0.5, 0.5, f'{name}\nn<2', ha='center', va='center', transform=ax.transAxes, color=INK_SOFT)
            ax.axis('off')
            continue
        # Valori veri e predetti delle molecole misurate.
        t, p = y_true[m, i], y_pred[m, i]
        # Estremi comuni ai due assi (con un po' di margine).
        lo, hi = min(t.min(), p.min()) - 0.2, max(t.max(), p.max()) + 0.2
        # Diagonale y = x, sotto i punti.
        ax.plot([lo, hi], [lo, hi], color=INK_SOFT, lw=1.2, ls='--', zorder=2)
        # Punti con bordo bianco sottile, così restano distinguibili dove si sovrappongono.
        ax.scatter(t, p, s=26, color=C_BLUE, alpha=0.6, edgecolor='white', linewidth=0.4, zorder=3)
        # Stessi limiti e proporzioni sui due assi: la diagonale è a 45°.
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal', adjustable='box')
        # Nome dell'isoforma a sinistra; R² e n nell'angolo in alto a sinistra.
        ax.set_title(name, loc='left', fontsize=13, weight='bold', color=INK)
        ax.text(0.04, 0.95, f'R² = {r2_score(t, p):.2f}\nn = {m.sum()}', transform=ax.transAxes, va='top',
                fontsize=11, color=INK)
        ax.set_xlabel('True pIC50', color=INK_SOFT, fontsize=10.5)
        ax.set_ylabel('Predicted pIC50', color=INK_SOFT, fontsize=10.5)
        _style_axes(ax, grid_axis='both')
    # Sistema gli spazi.
    fig.tight_layout()
    # Salva l'immagine.
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    # Chiude la figura.
    plt.close(fig)


# Recall top-k: tra il k% di molecole davvero più potenti, quale frazione il modello mette
# anch'esso nel suo k% più alto. È la domanda pratica dello screening virtuale.
def _topk_recall(t, p, k_frac):
    # Numero di molecole nel top k% (almeno 1).
    k = max(1, int(round(len(t) * k_frac)))
    # Indici delle k molecole con pIC50 vero più alto.
    top_true = set(np.argsort(-t)[:k])
    # Indici delle k molecole con pIC50 predetto più alto.
    top_pred = set(np.argsort(-p)[:k])
    # Frazione di quelle vere ritrovate dal modello.
    return len(top_true & top_pred) / k


# Metriche di ranking per isoforma: Spearman, Kendall e, per k = 1%, 5%, 10%, recall top-k
# ed Enrichment Factor (EF = recall / k: quante volte meglio di una scelta a caso, EF = 1).
# Se si passa output_path salva tabella e grafico (Spearman ed EF@10%).
def ranking_metrics_per_isoform(y_true, y_pred, output_path=None):
    # Frazioni di molecole "top" considerate.
    k_fractions = (0.01, 0.05, 0.1)
    # Righe della tabella.
    rows = []
    # Per ogni isoforma…
    for i, name in enumerate(ISOFORMS):
        # …maschera delle molecole misurate e loro numero.
        m = ~np.isnan(y_true[:, i])
        # Numero di molecole misurate.
        n = int(m.sum())
        # Inizia la riga con nome e numerosità.
        row = {'isoform': name, 'n': n}
        # Con meno di 5 molecole il ranking non ha senso: tutto NaN.
        if n < 5:
            # Correlazioni mancanti.
            row.update({'spearman_rho': np.nan, 'kendall_tau': np.nan})
            # Recall ed EF mancanti per ogni k.
            for k in k_fractions:
                # Una colonna per metrica e per k (es. 'EF@10pc').
                row.update({f'recall@{int(k * 100)}pc': np.nan, f'EF@{int(k * 100)}pc': np.nan})
        # Altrimenti calcola le metriche.
        else:
            # Valori veri e predetti delle molecole misurate.
            t, p = y_true[m, i], y_pred[m, i]
            # Correlazioni di rango.
            row.update({'spearman_rho': spearmanr(t, p).correlation, 'kendall_tau': kendalltau(t, p).correlation})
            # Per ogni frazione top…
            for k in k_fractions:
                # …recall top-k…
                rec = _topk_recall(t, p, k)
                # …ed EF = recall diviso la frazione stessa.
                row.update({f'recall@{int(k * 100)}pc': rec, f'EF@{int(k * 100)}pc': rec / k})
        # Aggiunge la riga.
        rows.append(row)
    # Tabella finale.
    df = pd.DataFrame(rows)
    # Se non è richiesto il salvataggio la funzione finisce qui.
    if not output_path:
        # Restituisce la tabella.
        return df
    # Crea la cartella di destinazione se manca.
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    # Salva la tabella.
    df.to_csv(output_path, index=False)
    # Figura con due grafici affiancati: Spearman ed EF@10% per isoforma.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=SURFACE)
    _bar_panel(axes[0], df, 'spearman_rho', 'Spearman ρ per isoform', 'Spearman ρ', ref=0)
    _bar_panel(axes[1], df, 'EF@10pc', 'Enrichment factor @ 10%', 'EF@10%', ref=1.0, ref_label='random selection')
    # Salva accanto al CSV.
    fig.tight_layout()
    _finish_bars(fig, output_path.replace('.csv', '.png'))
    # Restituisce la tabella.
    return df


# -------------------------------------------------------------------------------------
# 2. Grafico riassuntivo (usato solo da figures.ipynb)
# -------------------------------------------------------------------------------------

# Per ogni isoforma affianca tre barre: R² (regressione), Spearman (ranking) ed EF@10%
# (screening, su un secondo asse y a destra), per mostrare che un R² basso non implica
# per forza un ranking inutile.
def r2_vs_ranking_plot(r2_df, ranking_df, output_path):
    # Unisce le due tabelle sulla colonna dell'isoforma.
    merged = r2_df[['isoform', 'r2']].merge(ranking_df[['isoform', 'n', 'spearman_rho', 'EF@10pc']], on='isoform')
    # Tre riquadri con scale proprie (mai due assi y sullo stesso grafico).
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), facecolor=SURFACE)
    _bar_panel(axes[0], merged, 'r2', 'R² (regression)', 'R²', ref=0)
    _bar_panel(axes[1], merged, 'spearman_rho', 'Spearman ρ (ranking)', 'Spearman ρ', ref=0)
    _bar_panel(axes[2], merged, 'EF@10pc', 'EF @ 10% (screening)', 'Enrichment factor', ref=1.0,
               ref_label='random selection')
    # Salva.
    fig.tight_layout()
    _finish_bars(fig, output_path)


# Istogramma raggruppato dell'R² sul test per isoforma, per più modelli affiancati, con in fondo
# il gruppo "Mean" (media sulle isoforme). `df` è la tabella riassuntiva con colonne
# 'isoform', 'n_test' e 'R2_<modello>'. Le isoforme con n_test < underpowered_n hanno un asterisco.
def r2_comparison_plot(df, output_path, models=('LGBM', 'DMPNN', 'Stacking'),
                       labels=('LightGBM (baseline)', 'D-MPNN', 'Stacking'), underpowered_n=25):
    # Un gruppo per isoforma più uno per la media.
    iso = list(df['isoform']); n_test = df['n_test'].to_numpy()
    x = np.arange(len(iso) + 1); width = 0.26
    colors = [C_BLUE, C_ORANGE, C_AQUA]
    # Figura larga.
    fig, ax = plt.subplots(figsize=(16, 6.2), facecolor=SURFACE)
    # Banda leggera dietro le isoforme di Classe I.
    for i, name in enumerate(iso):
        if name in CLASS_I_ISOFORMS:
            ax.axvspan(i - 0.5, i + 0.5, color=C_BLUE, alpha=0.06, lw=0, zorder=0)
    # Separatore tra le isoforme e la media.
    ax.axvline(len(iso) - 0.5, color=GRID, lw=1.2, zorder=1)
    # Barre di ogni modello.
    for k, (m, lab) in enumerate(zip(models, labels)):
        # Valori per isoforma e media finale.
        vals = df[f'R2_{m}'].to_numpy(dtype=float); vals = np.append(vals, np.nanmean(vals))
        # Spostamento della barra nel gruppo.
        off = (k - (len(models) - 1) / 2) * width
        ax.bar(x + off, vals, width * 0.92, color=colors[k], label=lab, zorder=3)
        # Valore sopra ogni barra (sotto se negativo).
        for xi, vi in zip(x + off, vals):
            ax.annotate(f'{vi:.2f}', (xi, vi), xytext=(0, 3 if vi >= 0 else -3), textcoords='offset points',
                        ha='center', va='bottom' if vi >= 0 else 'top', fontsize=8.5, color=INK)
    # Linea dello zero.
    ax.axhline(0, color=INK_SOFT, lw=1, zorder=4)
    # Nomi delle isoforme (asterisco se poche molecole di test) e "Mean".
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}{"*" if t < underpowered_n else ""}' for n, t in zip(iso, n_test)] + ['Mean'],
                       fontsize=12, color=INK)
    # Assi, titolo e legenda.
    ax.set_ylabel('Test R²', color=INK_SOFT, fontsize=13)
    ax.set_title('Test R² per isoform: baseline vs D-MPNN vs stacking', loc='left', fontsize=16,
                 weight='bold', color=INK, pad=34)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(models), frameon=False, fontsize=12,
              labelcolor=INK, handlelength=1.2)
    ax.margins(y=0.1)
    _style_axes(ax)
    # Nota sulle bande e sugli asterischi.
    fig.text(0.07, -0.01, f'Shaded: Class I isoforms.   * n_test < {underpowered_n}: R² not reliable.',
             fontsize=10.5, color=INK_SOFT)
    # Salva.
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    # Chiude la figura.
    plt.close(fig)
