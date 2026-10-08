# LightGBM baseline on Morgan fingerprints

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from s0_config import TARGET_COLS, SMILES_COL, GROUP_COL
from s1_split import generate_fingerprints, fps_to_numpy
from s2_metrics import calculate_metrics

LGBM_PARAMS = dict(n_estimators=500,
                   min_child_samples=1,
                   random_state=42,
                   n_jobs=4,
                   verbose=-1)


# SMILES
def featurize(smiles):
    fps, valid = generate_fingerprints(list(smiles))
    X = fps_to_numpy(fps) if fps else np.zeros((0, 0), dtype=np.int8)
    return X, valid

# One regressor per isoform
def fit_lgbm_per_target(X, Y):
    from lightgbm import LGBMRegressor
    models = []
    for t in range(Y.shape[1]):
        mask = ~np.isnan(Y[:, t])
        if mask.sum() < 10:
            print(f'  [WARN] target {TARGET_COLS[t]}: only {mask.sum()} values, skipped')
            models.append(None)
            continue
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(X[mask], Y[mask, t])
        models.append(m)
    return models

def predict_lgbm_per_target(models, X):
    out = np.full((X.shape[0], len(models)), np.nan, dtype=np.float32)
    for t, m in enumerate(models):
        if m is not None:
            out[:, t] = m.predict(X)
    return out


# Entire baseline
def run_lgbm_baseline(train_df, val_df, test_df, output_dir):
    cv_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    X_train, valid_train = featurize(cv_df[SMILES_COL])
    Y_train = cv_df.iloc[valid_train][TARGET_COLS].values.astype(np.float32)
    X_test, valid_test_idx = featurize(test_df[SMILES_COL])
    Y_test = test_df.iloc[valid_test_idx][TARGET_COLS].values.astype(np.float32)

    print(f'[LGBM] fit train+val (N={X_train.shape[0]}, T={Y_train.shape[1]})...')
    models = fit_lgbm_per_target(X_train, Y_train)
    preds_test = predict_lgbm_per_target(models, X_test)
    metrics_test = calculate_metrics(preds_test, Y_test)
    print(f"[LGBM] Test R²={metrics_test['r2']:.4f}  MSE={metrics_test['mse']:.4f}")

    print('[LGBM] OOF CV (5-fold grouped) for stacking...')
    groups = cv_df[GROUP_COL].values[valid_train]
    oof = np.full_like(Y_train, np.nan, dtype=np.float32)
    for fold_idx, (tr_i, va_i) in enumerate(GroupKFold(n_splits=5).split(X_train, Y_train, groups=groups), start=1):
        fold_models = fit_lgbm_per_target(X_train[tr_i], Y_train[tr_i])
        oof[va_i] = predict_lgbm_per_target(fold_models, X_train[va_i])
        print(f'  fold {fold_idx}: done')

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([{k: v for k, v in metrics_test.items() if k != 'per_target'}]).to_csv(
        os.path.join(output_dir, 'lgbm_test_metrics.csv'), index=False)
    np.savez(os.path.join(output_dir, 'lgbm_oof.npz'), oof=oof, y_true=Y_train,
             smiles=cv_df[SMILES_COL].values[valid_train], groups=groups)
    return {'test_metrics': metrics_test,
            'test_preds': preds_test,
            'oof': oof,  # for stacking
            'models': models}
