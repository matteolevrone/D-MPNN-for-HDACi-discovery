# =====================================================================================
# PASSO 3 di 7 — BASELINE LIGHTGBM SU FINGERPRINT MORGAN
#
# Ruolo nella pipeline: un modello "classico" di riferimento. Per ognuna delle 9 isoforme
# allena un regressore LightGBM (alberi di decisione in boosting) che riceve il fingerprint
# Morgan della molecola (2048 bit) e predice il pIC50. Il D-MPNN (passo 5) deve battere
# questo modello per giustificare la sua complessità.
#
# Produce anche le predizioni "out-of-fold" (OOF) sul train+val, che servono al passo 6
# (stacking) per imparare come combinare LightGBM e D-MPNN senza barare.
# Le funzioni sono in ordine di chiamata; quella principale è in fondo.
# =====================================================================================

# Composizione dei percorsi dei file di output.
import os
# Calcolo numerico su array.
import numpy as np
# Tabelle.
import pandas as pd
# K-fold per gruppi: ogni gruppo chimico finisce in un solo fold di validazione.
from sklearn.model_selection import GroupKFold
# Costanti condivise: colonne target, colonna SMILES, colonna dei gruppi.
from s0_config import TARGET_COLS, SMILES_COL, GROUP_COL
# Fingerprint Morgan e loro conversione in matrice (passo 1).
from s1_split import generate_fingerprints, fps_to_numpy
# Metriche di valutazione (passo 2).
from s2_metrics import calculate_metrics

# Iperparametri di LightGBM, uguali per tutte le isoforme:
LGBM_PARAMS = dict(n_estimators=500,  # numero di alberi in sequenza
                   min_child_samples=1,  # una foglia può contenere anche una sola molecola
                   random_state=42,  # seme fisso: risultati riproducibili
                   n_jobs=4,  # thread usati; -1 (tutti) sulla macchina condivisa si blocca
                   verbose=-1)  # nessun messaggio di LightGBM durante il fit


# Da una lista di SMILES alla matrice dei fingerprint (molecole valide × 2048, valori 0/1)
# e agli indici delle molecole valide nella lista di partenza.
def featurize(smiles):
    # Fingerprint RDKit e indici delle righe con SMILES valido.
    fps, valid = generate_fingerprints(list(smiles))
    # Matrice numpy; se nessuna molecola è valida una matrice vuota.
    X = fps_to_numpy(fps) if fps else np.zeros((0, 0), dtype=np.int8)
    # Restituisce matrice e indici validi.
    return X, valid


# Allena un regressore per isoforma usando solo le molecole misurate su quell'isoforma.
# Restituisce una lista di 9 modelli (None per un'isoforma con meno di 10 misure).
def fit_lgbm_per_target(X, Y):
    # Importa LightGBM solo quando serve davvero.
    from lightgbm import LGBMRegressor
    # Lista dei modelli, uno per isoforma.
    models = []
    # Per ogni isoforma (colonna t di Y)…
    for t in range(Y.shape[1]):
        # …maschera delle molecole con pIC50 misurato.
        mask = ~np.isnan(Y[:, t])
        # Con meno di 10 misure non vale la pena allenare un modello.
        if mask.sum() < 10:
            # Avviso a schermo.
            print(f'  [WARN] target {TARGET_COLS[t]}: only {mask.sum()} values, skipped')
            # Segnaposto: nessun modello per questa isoforma.
            models.append(None)
            # Passa all'isoforma successiva.
            continue
        # Crea un regressore con gli iperparametri comuni.
        m = LGBMRegressor(**LGBM_PARAMS)
        # Lo allena: input = fingerprint, output = pIC50, solo sulle molecole misurate.
        m.fit(X[mask], Y[mask, t])
        # Conserva il modello allenato.
        models.append(m)
    # Restituisce i 9 modelli.
    return models


# Usa i 9 modelli per predire tutte le isoforme di ogni molecola di X.
def predict_lgbm_per_target(models, X):
    # Matrice dei risultati, inizialmente NaN (resta NaN dove manca il modello).
    out = np.full((X.shape[0], len(models)), np.nan, dtype=np.float32)
    # Per ogni isoforma e il suo modello…
    for t, m in enumerate(models):
        # …se il modello esiste…
        if m is not None:
            # …riempie la colonna t con le sue predizioni.
            out[:, t] = m.predict(X)
    # Restituisce la matrice delle predizioni.
    return out


# Funzione principale chiamata dal notebook:
#  1) allena i 9 modelli su train+val e li valuta sul test;
#  2) rifà lo stesso in 5-fold CV per gruppi su train+val, per ottenere le predizioni OOF
#     (ogni molecola è predetta da modelli che non l'hanno mai vista) per lo stacking;
#  3) salva metriche e OOF in output_dir.
def run_lgbm_baseline(train_df, val_df, test_df, output_dir):
    # Train e validation uniti: qui non serve un validation separato.
    cv_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    # Fingerprint di train+val e indici delle molecole valide.
    X_train, valid_train = featurize(cv_df[SMILES_COL])
    # pIC50 delle molecole valide di train+val (matrice molecole × 9).
    Y_train = cv_df.iloc[valid_train][TARGET_COLS].values.astype(np.float32)
    # Fingerprint del test e indici delle molecole valide.
    X_test, valid_test_idx = featurize(test_df[SMILES_COL])
    # pIC50 veri delle molecole valide del test.
    Y_test = test_df.iloc[valid_test_idx][TARGET_COLS].values.astype(np.float32)

    # Messaggio di avanzamento.
    print(f'[LGBM] fit train+val (N={X_train.shape[0]}, T={Y_train.shape[1]})...')
    # 1) Allena i 9 modelli su tutto train+val.
    models = fit_lgbm_per_target(X_train, Y_train)
    # Predice il test.
    preds_test = predict_lgbm_per_target(models, X_test)
    # Confronta le predizioni con i valori veri del test.
    metrics_test = calculate_metrics(preds_test, Y_test)
    # Stampa R² e MSE del test.
    print(f"[LGBM] Test R²={metrics_test['r2']:.4f}  MSE={metrics_test['mse']:.4f}")

    # 2) Predizioni out-of-fold per lo stacking.
    print('[LGBM] OOF CV (5-fold grouped) for stacking...')
    # Gruppo chimico di ogni molecola valida.
    groups = cv_df[GROUP_COL].values[valid_train]
    # Matrice OOF inizialmente NaN, stessa forma di Y_train.
    oof = np.full_like(Y_train, np.nan, dtype=np.float32)
    # 5 fold per gruppi: in ogni giro un quinto dei gruppi fa da validazione.
    for fold_idx, (tr_i, va_i) in enumerate(GroupKFold(n_splits=5).split(X_train, Y_train, groups=groups), start=1):
        # Allena 9 modelli sui 4/5 di allenamento…
        fold_models = fit_lgbm_per_target(X_train[tr_i], Y_train[tr_i])
        # …e predice il quinto escluso, riempiendo quelle righe della matrice OOF.
        oof[va_i] = predict_lgbm_per_target(fold_models, X_train[va_i])
        # Messaggio di avanzamento.
        print(f'  fold {fold_idx}: done')

    # 3) Salvataggi. Crea la cartella se manca.
    os.makedirs(output_dir, exist_ok=True)
    # Metriche del test (senza il dettaglio per isoforma) in un CSV di una riga.
    pd.DataFrame([{k: v for k, v in metrics_test.items() if k != 'per_target'}]).to_csv(
        os.path.join(output_dir, 'lgbm_test_metrics.csv'), index=False)
    # Predizioni OOF, valori veri, SMILES e gruppi in un archivio numpy compresso.
    np.savez(os.path.join(output_dir, 'lgbm_oof.npz'), oof=oof, y_true=Y_train,
             smiles=cv_df[SMILES_COL].values[valid_train], groups=groups)
    # Restituisce tutto ciò che serve nelle celle successive del notebook.
    return {'test_metrics': metrics_test,  # metriche sul test
            'test_preds': preds_test,  # predizioni sul test (per lo stacking)
            'oof': oof,  # predizioni out-of-fold su train+val (per lo stacking)
            'models': models}  # i 9 modelli allenati (riusati per il bundle di inferenza)
