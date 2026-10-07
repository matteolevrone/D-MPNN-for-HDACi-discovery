# =====================================================================================
# PASSO 7 di 7 — BUNDLE DI INFERENZA E PREDIZIONE SU MOLECOLE NUOVE
#
# Ruolo nella pipeline: salva in un unico file (il "bundle") tutto ciò che serve per
# predire molecole mai viste, senza riallenare nulla:
#  - i 5 modelli D-MPNN del seed scelto, con le loro standardizzazioni (passo 5);
#  - i 9 modelli LightGBM allenati su train+val (passo 3);
#  - i meta-modelli Ridge dello stacking (passo 6).
# Poi predict() applica la catena completa: SMILES -> D-MPNN e LightGBM -> stacking -> pIC50.
#
# Si usa dal notebook (celle finali) oppure da terminale:
#   python s7_inference.py --bundle outputs/inference_bundle.pt --in molecole.csv --out pred.csv
# Le funzioni sono in ordine di chiamata.
# =====================================================================================

# Lettura degli argomenti da riga di comando.
import argparse
# Calcolo numerico su array.
import numpy as np
# Tabelle.
import pandas as pd
# Salvataggio e caricamento del bundle (formato di PyTorch).
import torch
# RDKit, per verificare quali SMILES sono validi.
from rdkit import Chem
# Costanti condivise.
from s0_config import TARGET_COLS
# Fingerprint e predizione LightGBM (passo 3).
from s3_baseline import featurize, predict_lgbm_per_target
# Predizione con i modelli D-MPNN di un seed (passo 5).
from s5_training import predict_dmpnn
# Combinazione dei modelli (passo 6).
from s6_stacking import predict_stacking

# Etichetta del formato del file, controllata al caricamento.
BUNDLE_FORMAT = 'mithril-stack-v2'


# Salva il bundle. dmpnn_res è l'output di seed_view (passo 5) per il seed scelto,
# meta i meta-modelli dello stacking, lgbm_models i 9 modelli di run_lgbm_baseline.
def export_bundle(out_path, dmpnn_res, meta, lgbm_models):
    # Parte D-MPNN: solo le chiavi necessarie a predict_dmpnn (niente OOF né metriche).
    dmpnn = {k: dmpnn_res[k] for k in ('fold_states', 'fold_scalers', 'fold_target_scalers',
                                        'fold_val_mses', 'model_kwargs', 'node_feat_dim', 'edge_feat_dim')}
    # Il bundle completo.
    bundle = {'format': BUNDLE_FORMAT,  # etichetta di versione
              'target_cols': list(TARGET_COLS),  # nomi delle colonne prodotte
              'dmpnn': dmpnn,  # modelli D-MPNN del seed scelto
              'lgbm_models': lgbm_models,  # modelli LightGBM
              'meta': meta}  # meta-modelli Ridge
    # Scrive il file.
    torch.save(bundle, out_path)
    # Messaggio di conferma con il contenuto.
    print(f"[export] bundle saved to {out_path} (DMPNN seed {dmpnn_res['seed']}: "
          f"{len(dmpnn['fold_states'])} folds, LGBM={sum(m is not None for m in lgbm_models)}/{len(lgbm_models)}, "
          f"meta targets={sum(v is not None for v in meta.values())})")


# Legge il bundle; map_location decide se caricare i tensori su CPU o GPU.
def load_bundle(path, map_location='cpu'):
    # Carica il file (weights_only=False perché contiene anche oggetti LightGBM e Ridge).
    bundle = torch.load(path, map_location=map_location, weights_only=False)
    # Controlla che sia un bundle di questa versione.
    assert bundle.get('format') == BUNDLE_FORMAT, 'unrecognised bundle format: re-export it with s7_inference.export_bundle'
    # Restituisce il dizionario.
    return bundle


# Predice i 9 pIC50 di una lista di SMILES con la catena completa. Le righe con SMILES
# non valido restano NaN.
def predict(smiles, bundle, device='cpu'):
    # Accetta anche un singolo SMILES come stringa.
    if isinstance(smiles, str):
        # Lo trasforma in lista di un elemento.
        smiles = [smiles]
    # Maschera degli SMILES validi.
    valid = np.array([Chem.MolFromSmiles(s) is not None for s in smiles], dtype=bool)
    # Matrice dei risultati, inizialmente NaN.
    out = np.full((len(smiles), len(bundle['target_cols'])), np.nan, dtype=np.float32)
    # Se non c'è nessuna molecola valida non c'è niente da fare.
    if not valid.any():
        # Restituisce tutti NaN.
        return out
    # Solo gli SMILES validi.
    valid_smiles = [s for s, ok in zip(smiles, valid) if ok]
    # Predizioni LightGBM sui fingerprint.
    lgbm_preds = predict_lgbm_per_target(bundle['lgbm_models'], featurize(valid_smiles)[0])
    # Predizioni D-MPNN (batch da 64 molecole).
    dmpnn_preds = predict_dmpnn(bundle['dmpnn'], valid_smiles, device, batch_size=64)
    # Combinazione con lo stacking; le chiavi devono essere le stesse usate in training.
    out[valid] = predict_stacking(bundle['meta'], {'DMPNN': dmpnn_preds, 'LGBM': lgbm_preds})
    # Restituisce le predizioni.
    return out


# Come predict, ma restituisce una tabella con la colonna degli SMILES e una per isoforma.
def predict_to_dataframe(smiles, bundle, device='cpu'):
    # Tabella delle predizioni con i nomi delle isoforme come colonne.
    df = pd.DataFrame(predict(smiles, bundle, device=device), columns=bundle['target_cols'])
    # Aggiunge gli SMILES come prima colonna.
    df.insert(0, 'Smiles', smiles)
    # Restituisce la tabella.
    return df


# Uso da terminale: legge un CSV di SMILES, predice e scrive un CSV di risultati.
def _main():
    # Descrizione del comando.
    p = argparse.ArgumentParser(description='Stacked D-MPNN + LightGBM inference on new SMILES.')
    # File del bundle (obbligatorio).
    p.add_argument('--bundle', required=True, help='inference_bundle.pt file')
    # CSV di input (obbligatorio).
    p.add_argument('--in', dest='in_csv', required=True, help='CSV with a SMILES column')
    # Nome della colonna degli SMILES nel CSV.
    p.add_argument('--smiles-col', default='Smiles')
    # CSV di output.
    p.add_argument('--out', default='predictions.csv')
    # Dispositivo di calcolo.
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    # Legge gli argomenti.
    args = p.parse_args()
    # Carica il bundle.
    bundle = load_bundle(args.bundle, map_location=args.device)
    # Legge gli SMILES dal CSV come testo.
    smiles = pd.read_csv(args.in_csv)[args.smiles_col].astype(str).tolist()
    # Predice e salva.
    predict_to_dataframe(smiles, bundle, device=args.device).to_csv(args.out, index=False)
    # Messaggio di conferma.
    print(f'[inference] {len(smiles)} molecules -> {args.out}')


# Esegue _main solo se il file è lanciato da terminale (non quando viene importato).
if __name__ == '__main__':
    # Avvio.
    _main()
