# =====================================================================================
# PASSO 6 di 7 — STACKING: COMBINARE LIGHTGBM E D-MPNN
#
# Ruolo nella pipeline: invece di scegliere un solo modello, impara come combinarli.
# Per ogni isoforma allena una regressione Ridge (lineare, con pesi >= 0) che riceve le
# predizioni dei modelli di base e restituisce la predizione finale.
#
# Il meta-modello viene allenato sulle predizioni OUT-OF-FOLD (passi 3 e 5): ogni molecola
# è predetta da modelli che non l'hanno vista, quindi il meta-modello impara quanto fidarsi
# di ciascun modello su dati "nuovi", non su dati già memorizzati.
#
# Ordine nel notebook: fit_stacking_meta_learner (su OOF) e poi predict_stacking (sul test);
# il passo 7 riusa predict_stacking per le molecole nuove.
# =====================================================================================

# Calcolo numerico su array.
import numpy as np
# Regressione lineare con penalità L2 (Ridge).
from sklearn.linear_model import Ridge
# K-fold semplice per scegliere la forza della penalità.
from sklearn.model_selection import KFold
# Metriche per la stampa e la scelta della penalità.
from sklearn.metrics import mean_squared_error, r2_score


# Mette affiancate le predizioni dei modelli in un'unica matrice di input per il Ridge.
# I modelli sono presi in ordine alfabetico ('DMPNN', poi 'LGBM'), così l'ordine delle
# colonne è sempre lo stesso in training e in predizione: 9 colonne per modello.
def build_stacking_features(preds_dict):
    # Concatena le matrici molecole × 9 lungo le colonne.
    return np.concatenate([preds_dict[k] for k in sorted(preds_dict)], axis=1)


# Sceglie la forza della penalità (alpha) del Ridge con una 5-fold CV interna: per ogni
# alpha della griglia stima l'MSE su dati non visti e tiene il più basso.
def _select_alpha_cv(X, y, alpha_grid, cv_splits):
    # Fold casuali ma riproducibili.
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    # Miglior alpha e suo MSE (si parte dal primo della griglia).
    best_alpha, best_mse = alpha_grid[0], float('inf')
    # Per ogni alpha candidato…
    for a in alpha_grid:
        # …MSE di ogni fold…
        fold_mses = []
        # …per ogni divisione training/validazione…
        for tr_idx, va_idx in kf.split(X):
            # …Ridge con pesi non negativi…
            r = Ridge(alpha=a, positive=True)
            # …allenato sulla parte di training…
            r.fit(X[tr_idx], y[tr_idx])
            # …e valutato sulla parte di validazione.
            fold_mses.append(mean_squared_error(y[va_idx], r.predict(X[va_idx])))
        # MSE medio di questo alpha.
        cv_mse = float(np.mean(fold_mses))
        # Se è il migliore finora…
        if cv_mse < best_mse:
            # …lo conserva.
            best_alpha, best_mse = a, cv_mse
    # Restituisce l'alpha scelto e il suo MSE.
    return best_alpha, best_mse


# Allena un Ridge per isoforma sulle predizioni OOF. Restituisce un dizionario
# isoforma (0…8) -> modello Ridge (None se l'isoforma ha meno di 20 molecole utilizzabili).
def fit_stacking_meta_learner(oof_dict, y_true_oof, alpha_grid, cv_splits=5):
    # Matrice di input: predizioni OOF dei modelli affiancate (molecole × 18).
    X = build_stacking_features(oof_dict)
    # Dizionario dei meta-modelli.
    meta = {}
    # Stampa le dimensioni in gioco.
    print(f'[stacking +pos] X shape = {X.shape}, target shape = {y_true_oof.shape}, alpha_grid={alpha_grid}')
    # Per ogni isoforma…
    for t in range(y_true_oof.shape[1]):
        # …pIC50 veri…
        y_t = y_true_oof[:, t]
        # …righe utilizzabili: target misurato e nessuna predizione mancante.
        rows_ok = ~np.isnan(y_t) & ~np.isnan(X).any(axis=1)
        # Troppo poche molecole: nessun meta-modello per questa isoforma.
        if rows_ok.sum() < 20:
            # Avviso.
            print(f'  target {t}: skip (n={rows_ok.sum()})')
            # Segnaposto.
            meta[t] = None
            # Isoforma successiva.
            continue
        # Input e target delle righe utilizzabili.
        Xt, yt = X[rows_ok], y_t[rows_ok]
        # Con abbastanza righe sceglie alpha in CV…
        if rows_ok.sum() >= cv_splits * 4:
            # …tra i valori della griglia.
            alpha_t, cv_mse = _select_alpha_cv(Xt, yt, alpha_grid, cv_splits)
        # …altrimenti usa alpha = 1.
        else:
            # Nessuna stima di MSE disponibile.
            alpha_t, cv_mse = 1.0, float('nan')
        # Ridge finale con pesi non negativi (nessun modello può "contare al contrario").
        ridge = Ridge(alpha=alpha_t, positive=True)
        # Allenato su tutte le righe utilizzabili.
        ridge.fit(Xt, yt)
        # Conservato per l'isoforma t.
        meta[t] = ridge
        # Riga di riepilogo con l'R² sulle stesse righe di training (indicativo).
        print(f'  target {t}: n={rows_ok.sum()}  alpha={alpha_t:<5}  cv_mse={cv_mse:.4f}  '
              f'in-sample R²={r2_score(yt, ridge.predict(Xt)):.3f}')
    # Restituisce i meta-modelli.
    return meta


# Applica i meta-modelli alle predizioni dei modelli di base su nuove molecole. Per
# un'isoforma senza meta-modello usa la media semplice delle predizioni dei modelli.
def predict_stacking(meta, preds_dict):
    # Input nello stesso ordine di colonne usato in training.
    X = build_stacking_features(preds_dict)
    # Una matrice qualsiasi tra quelle in input, per leggerne le dimensioni.
    first = next(iter(preds_dict.values()))
    # Matrice dei risultati (molecole × 9).
    preds = np.zeros(first.shape, dtype=np.float32)
    # Per ogni isoforma…
    for t in range(first.shape[1]):
        # …se manca il meta-modello…
        if meta.get(t) is None:
            # …media delle predizioni dei modelli di base…
            preds[:, t] = np.mean([v[:, t] for v in preds_dict.values()], axis=0)
        # …altrimenti…
        else:
            # …predizione del Ridge.
            preds[:, t] = meta[t].predict(X)
    # Restituisce le predizioni combinate.
    return preds
