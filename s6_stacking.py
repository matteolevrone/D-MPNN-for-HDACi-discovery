# Stacking

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def build_stacking_features(preds_dict):
    return np.concatenate([preds_dict[k] for k in sorted(preds_dict)], axis=1)


# Picks the Ridge penalty (alpha)
def _select_alpha_cv(X, y, alpha_grid, cv_splits):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    best_alpha, best_mse = alpha_grid[0], float('inf')
    for a in alpha_grid:
        fold_mses = []
        for tr_idx, va_idx in kf.split(X):
            r = Ridge(alpha=a, positive=True)
            r.fit(X[tr_idx], y[tr_idx])
            fold_mses.append(mean_squared_error(y[va_idx], r.predict(X[va_idx])))
        cv_mse = float(np.mean(fold_mses))
        if cv_mse < best_mse:
            best_alpha, best_mse = a, cv_mse
    return best_alpha, best_mse


# Fits one Ridge per isoform on the OOF predictions
def fit_stacking_meta_learner(oof_dict, y_true_oof, alpha_grid, cv_splits=5):
    X = build_stacking_features(oof_dict)
    meta = {}
    print(f'[stacking +pos] X shape = {X.shape}, target shape = {y_true_oof.shape}, alpha_grid={alpha_grid}')
    for t in range(y_true_oof.shape[1]):
        y_t = y_true_oof[:, t]
        rows_ok = ~np.isnan(y_t) & ~np.isnan(X).any(axis=1)
        if rows_ok.sum() < 20:
            print(f'  target {t}: skip (n={rows_ok.sum()})')
            meta[t] = None
            continue
        Xt, yt = X[rows_ok], y_t[rows_ok]
        if rows_ok.sum() >= cv_splits * 4:
            alpha_t, cv_mse = _select_alpha_cv(Xt, yt, alpha_grid, cv_splits)
        else:
            alpha_t, cv_mse = 1.0, float('nan')
        ridge = Ridge(alpha=alpha_t, positive=True)
        ridge.fit(Xt, yt)
        meta[t] = ridge
        print(f'  target {t}: n={rows_ok.sum()}  alpha={alpha_t:<5}  cv_mse={cv_mse:.4f}  '
              f'in-sample R²={r2_score(yt, ridge.predict(Xt)):.3f}')
    return meta


# Applies the meta-models
def predict_stacking(meta, preds_dict):
    X = build_stacking_features(preds_dict)
    first = next(iter(preds_dict.values()))
    preds = np.zeros(first.shape, dtype=np.float32)
    for t in range(first.shape[1]):
        if meta.get(t) is None:
            preds[:, t] = np.mean([v[:, t] for v in preds_dict.values()], axis=0)
        else:
            preds[:, t] = meta[t].predict(X)
    return preds
