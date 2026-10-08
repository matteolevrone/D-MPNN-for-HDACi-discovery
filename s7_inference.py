# Inference bundle

import argparse
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from s0_config import TARGET_COLS
from s3_baseline import featurize, predict_lgbm_per_target
from s5_training import predict_dmpnn
from s6_stacking import predict_stacking

BUNDLE_FORMAT = 'mithril-stack-v2'

def export_bundle(out_path, dmpnn_res, meta, lgbm_models):
    dmpnn = {k: dmpnn_res[k] for k in ('fold_states', 'fold_scalers', 'fold_target_scalers',
                                        'fold_val_mses', 'model_kwargs', 'node_feat_dim', 'edge_feat_dim')}
    bundle = {'format': BUNDLE_FORMAT,
              'target_cols': list(TARGET_COLS),
              'dmpnn': dmpnn,
              'lgbm_models': lgbm_models,
              'meta': meta}
    torch.save(bundle, out_path)
    print(f"[export] bundle saved to {out_path} (DMPNN seed {dmpnn_res['seed']}: "
          f"{len(dmpnn['fold_states'])} folds, LGBM={sum(m is not None for m in lgbm_models)}/{len(lgbm_models)}, "
          f"meta targets={sum(v is not None for v in meta.values())})")


# Loads the bundle
def load_bundle(path, map_location='cpu'):
    bundle = torch.load(path, map_location=map_location, weights_only=False)  # also holds LightGBM/Ridge objects
    assert bundle.get('format') == BUNDLE_FORMAT, 'unrecognised bundle format: re-export it with s7_inference.export_bundle'
    return bundle


# Predicts the 9 pIC50 values for a list of SMILES through the full chain.
def predict(smiles, bundle, device='cpu'):
    if isinstance(smiles, str):
        smiles = [smiles]
    valid = np.array([Chem.MolFromSmiles(s) is not None for s in smiles], dtype=bool)
    out = np.full((len(smiles), len(bundle['target_cols'])), np.nan, dtype=np.float32)
    if not valid.any():
        return out
    valid_smiles = [s for s, ok in zip(smiles, valid) if ok]
    lgbm_preds = predict_lgbm_per_target(bundle['lgbm_models'], featurize(valid_smiles)[0])
    dmpnn_preds = predict_dmpnn(bundle['dmpnn'], valid_smiles, device, batch_size=64)
    out[valid] = predict_stacking(bundle['meta'], {'DMPNN': dmpnn_preds, 'LGBM': lgbm_preds})
    return out

def predict_to_dataframe(smiles, bundle, device='cpu'):
    df = pd.DataFrame(predict(smiles, bundle, device=device), columns=bundle['target_cols'])
    df.insert(0, 'Smiles', smiles)
    return df


# Command-line entry point: reads a SMILES CSV, predicts, writes a results CSV.
def _main():
    p = argparse.ArgumentParser(description='Stacked D-MPNN + LightGBM inference on new SMILES.')
    p.add_argument('--bundle', required=True, help='inference_bundle.pt file')
    p.add_argument('--in', dest='in_csv', required=True, help='CSV with a SMILES column')
    p.add_argument('--smiles-col', default='Smiles')
    p.add_argument('--out', default='predictions.csv')
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = p.parse_args()
    bundle = load_bundle(args.bundle, map_location=args.device)
    smiles = pd.read_csv(args.in_csv)[args.smiles_col].astype(str).tolist()
    predict_to_dataframe(smiles, bundle, device=args.device).to_csv(args.out, index=False)
    print(f'[inference] {len(smiles)} molecules -> {args.out}')


if __name__ == '__main__':
    _main()
