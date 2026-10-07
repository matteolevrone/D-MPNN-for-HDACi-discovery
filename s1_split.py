# =====================================================================================
# PASSO 1 di 7 — FINGERPRINT E SPLIT TRAIN / VALIDATION / TEST PER GRUPPI CHIMICI
#
# Ruolo nella pipeline: dalle molecole di data.xlsx costruisce tre insiemi (train, val,
# test) in modo che molecole chimicamente simili finiscano sempre nello stesso insieme.
# Così il test misura la capacità di predire molecole "nuove" (fuori distribuzione).
#
# Flusso di splitter() (la funzione principale, in fondo al file):
#   SMILES -> fingerprint Morgan -> UMAP (2D) -> HDBSCAN (gruppi) -> tanti split casuali
#   per gruppi -> si tiene quello con le distribuzioni di pIC50 più simili (test KS).
#
# Il notebook usa anche generate_fingerprints e fps_to_numpy (cella della mappa UMAP) e
# il passo 3 (baseline LightGBM) riusa gli stessi fingerprint come input del modello.
# Le funzioni sono in ordine di chiamata; quella principale è in fondo.
# =====================================================================================

# Libreria standard per comporre i percorsi dei file in cui salvare CSV e grafici.
import os
# Calcolo numerico su array (matrici di fingerprint, punteggi, indici).
import numpy as np
# Tabelle (DataFrame) per le molecole e i loro valori di pIC50.
import pandas as pd
# Libreria per disegnare i grafici delle distribuzioni.
import matplotlib.pyplot as plt
# Estensione di matplotlib che sa disegnare le curve di densità (KDE).
import seaborn as sns
# RDKit: Chem legge gli SMILES, DataStructs converte i fingerprint in array.
from rdkit import Chem, DataStructs
# Generatore di fingerprint Morgan di RDKit.
from rdkit.Chem import rdFingerprintGenerator
# Test di Kolmogorov-Smirnov a due campioni: misura quanto due distribuzioni differiscono.
from scipy.stats import ks_2samp
# Divide un dataset in due parti rispettando i gruppi (un gruppo non viene mai spezzato).
from sklearn.model_selection import GroupShuffleSplit
# Costanti condivise: colonne target, colonna SMILES, colonna dei gruppi.
from s0_config import TARGET_COLS, SMILES_COL, GROUP_COL

# Raggio del fingerprint Morgan: considera intorni di ogni atomo fino a 3 legami di distanza.
FP_RADIUS = 3
# Lunghezza del fingerprint: 2048 bit (ogni bit acceso = presenza di un certo frammento).
FP_BITS = 2048


# -------------------------------------------------------------------------------------
# 1. Fingerprint: da SMILES a vettori di bit
# -------------------------------------------------------------------------------------

# Converte una lista di SMILES in fingerprint Morgan "feature-based" (atomi descritti per
# ruolo farmacoforico: donatore, accettore, aromatico, …) e restituisce anche gli indici
# delle molecole valide, così chi chiama può scartare le righe con SMILES illeggibili.
def generate_fingerprints(smiles_list):
    # Crea il generatore: raggio 3, 2048 bit, invarianti atomiche di tipo "feature".
    fmgen = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS,  # profondità dell'intorno di ogni atomo
        fpSize=FP_BITS,  # numero di bit del vettore finale
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())  # atomi per ruolo
    # Liste vuote: fingerprint calcolati e posizione (indice) delle molecole valide.
    fps, valid_idx = [], []
    # Scorre le molecole tenendo anche la loro posizione nella lista originale.
    for i, smiles in enumerate(smiles_list):
        # Interpreta lo SMILES; restituisce None se la stringa non è una molecola valida.
        mol = Chem.MolFromSmiles(smiles)
        # Se la molecola è valida…
        if mol is not None:
            # …ne calcola il fingerprint (un oggetto RDKit di 2048 bit) e lo conserva…
            fps.append(fmgen.GetFingerprint(mol))
            # …e ricorda che la riga i è valida.
            valid_idx.append(i)
        # Altrimenti segnala lo SMILES illeggibile (la riga verrà scartata).
        else:
            # Stampa l'avviso con posizione e testo dello SMILES.
            print(f'[WARN] invalid SMILES at idx={i}: {smiles}')
    # Restituisce i fingerprint e gli indici delle righe valide.
    return fps, valid_idx


# Trasforma la lista di fingerprint RDKit in una matrice numpy (molecole × 2048) di 0/1,
# il formato richiesto da UMAP, dalla silhouette e da LightGBM.
def fps_to_numpy(fps):
    # Matrice di zeri: una riga per molecola, una colonna per bit; int8 occupa 1 byte.
    X = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.int8)
    # Per ogni fingerprint…
    for i, fp in enumerate(fps):
        # …copia i suoi bit direttamente nella riga i della matrice.
        DataStructs.ConvertToNumpyArray(fp, X[i])
    # Restituisce la matrice completa.
    return X


# -------------------------------------------------------------------------------------
# 2. Clustering: UMAP + HDBSCAN per trovare i gruppi di molecole simili
# -------------------------------------------------------------------------------------

# Proietta i fingerprint in uno spazio a poche dimensioni con UMAP (distanza di Jaccard =
# 1 - similarità di Tanimoto) e cerca gruppi densi con HDBSCAN. Le molecole isolate, che
# HDBSCAN marca come rumore (-1), diventano gruppi da una sola molecola, così nessun dato
# viene scartato. min_cluster_size grande = pochi gruppi grandi = split più difficile.
def cluster_hdbscan(fps, min_cluster_size, min_samples, n_components, n_neighbors, min_dist, seed):
    # Importa UMAP qui: è una libreria lenta da caricare e serve solo se si ricalcola lo split.
    import umap
    # Importa HDBSCAN dalla versione inclusa in scikit-learn.
    from sklearn.cluster import HDBSCAN
    # Configura UMAP con la distanza di Jaccard, adatta a vettori di bit.
    reducer = umap.UMAP(
        n_components=n_components,  # dimensioni dello spazio di arrivo (2 nel notebook)
        metric='jaccard',  # distanza tra fingerprint = 1 - Tanimoto
        n_neighbors=n_neighbors,  # quanti vicini guardare: più alto = struttura più globale
        min_dist=min_dist,  # quanto possono avvicinarsi i punti nella proiezione
        random_state=seed)  # seme fisso per avere sempre la stessa proiezione
    # Calcola la proiezione e la converte in numeri a doppia precisione per HDBSCAN.
    emb = np.asarray(reducer.fit_transform(fps_to_numpy(fps)), dtype=np.float64)
    # Configura HDBSCAN: trova da solo il numero di gruppi in base alla densità dei punti.
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    # Assegna un'etichetta a ogni molecola (-1 = rumore, cioè molecola isolata).
    raw_labels = clusterer.fit_predict(emb).astype(int)
    # Conta le molecole isolate.
    n_noise = int((raw_labels == -1).sum())
    # Conta i gruppi densi veri e propri (etichette >= 0 distinte).
    n_dense = len(set(raw_labels[raw_labels >= 0]))
    # Copia le etichette per poterle modificare senza perdere quelle originali.
    labels = raw_labels.copy()
    # Se ci sono molecole isolate…
    if n_noise:
        # …la prima etichetta libera è il massimo esistente + 1 (o 0 se non ci sono gruppi).
        start = (labels.max() + 1) if (labels >= 0).any() else 0
        # …e ognuna riceve un'etichetta nuova e diversa: diventa un gruppo da sola.
        labels[labels == -1] = np.arange(start, start + n_noise)
    # Rinumera le etichette in modo compatto (0, 1, 2, …) senza buchi.
    _, labels = np.unique(labels, return_inverse=True)
    # Stampa un riassunto del clustering.
    print(f'[cluster_hdbscan] min_cluster_size={min_cluster_size}, min_samples={min_samples}: '
          f'{n_dense} dense clusters, {n_noise} noise points ({100 * n_noise / len(emb):.1f}%) '
          f'-> {len(set(labels))} groups in total (noise as singletons)')
    # Stampa due indici di qualità del clustering (vedi funzione successiva).
    report_cluster_quality(emb, raw_labels, fps)
    # Restituisce un'etichetta di gruppo per ogni molecola.
    return labels.astype(int)


# Valuta la qualità del clustering usando SOLO i gruppi densi (il rumore non è un gruppo):
#  - DBCV: indice pensato per HDBSCAN, calcolato nello spazio UMAP; > 0 = struttura valida.
#  - silhouette con distanza di Tanimoto sui fingerprint: controlla che i gruppi trovati in
#    UMAP siano compatti anche nello spazio chimico originale (va letta qualitativamente).
def report_cluster_quality(emb, raw_labels, fps):
    # Silhouette: misura quanto ogni punto è più vicino al suo gruppo che agli altri.
    from sklearn.metrics import silhouette_score
    # Maschera delle molecole che appartengono a un gruppo denso.
    dense = raw_labels >= 0
    # Servono almeno due gruppi densi per poter confrontare qualcosa.
    if dense.sum() < 2 or len(set(raw_labels[dense])) < 2:
        # Altrimenti lo segnala…
        print('  [quality] too few dense clusters to validate.')
        # …ed esce senza calcolare nulla.
        return
    # Valore di default nel caso il pacchetto opzionale hdbscan non sia installato.
    dbcv = float('nan')
    # Prova a calcolare il DBCV (esiste solo nel pacchetto esterno hdbscan, opzionale).
    try:
        # Importa la funzione che calcola il DBCV.
        from hdbscan.validity import validity_index
        # Calcola il DBCV sullo spazio UMAP con le etichette originali (rumore incluso come -1).
        dbcv = float(validity_index(emb, raw_labels))
    # Se il pacchetto manca (o il calcolo fallisce), avvisa come installarlo e continua.
    except Exception:
        # Messaggio di avviso.
        print('  [quality] DBCV unavailable; `pip install hdbscan`.')
    # Matrice di bit delle sole molecole nei gruppi densi.
    X = fps_to_numpy([fps[i] for i in np.where(dense)[0]])
    # Silhouette con distanza di Jaccard (= Tanimoto) su un campione di al massimo 3000 molecole.
    sil = float(silhouette_score(X, raw_labels[dense], metric='jaccard',
                                 sample_size=min(3000, len(X)),  # campione per velocità
                                 random_state=0))  # campione sempre uguale
    # Stampa i due indici.
    print(f'  [quality] DBCV (UMAP, density-based)={dbcv:+.3f}  |  '
          f'Tanimoto silhouette, dense-only (external check)={sil:+.3f}')


# -------------------------------------------------------------------------------------
# 3. Scelta dello split: tanti tentativi casuali per gruppi, si tiene il più bilanciato
# -------------------------------------------------------------------------------------

# Logaritmo del p-value del test di Kolmogorov-Smirnov tra due insiemi di valori.
# p alto (log vicino a 0) = distribuzioni simili; p basso (log molto negativo) = diverse.
# Il p-value è limitato a eps per evitare log(0). È usata anche dai fold della CV (passo 5).
def ks_log_p(a, b, eps=1e-12):
    # Esegue il test KS, prende il p-value, lo limita dal basso e ne fa il logaritmo.
    return np.log(max(float(ks_2samp(a, b).pvalue), eps))


# Punteggio di uno split candidato: media dei log p-value KS tra train/val, train/test e
# val/test per ogni isoforma. Più è alto, più i tre insiemi hanno pIC50 distribuiti allo
# stesso modo. Un'isoforma con meno di min_n valori in un insieme riceve la penalità log(eps).
def _split_score(train_df, val_df, test_df, min_n=5, eps=1e-12):
    # Lista dei log p-value raccolti.
    log_scores = []
    # Per ogni isoforma…
    for t in TARGET_COLS:
        # …i valori misurati (non NaN) nei tre insiemi.
        tr, va, te = (d[t].dropna().values for d in (train_df, val_df, test_df))
        # L'isoforma è valutabile solo se ha abbastanza valori in tutti e tre gli insiemi.
        evaluable = len(tr) >= min_n and len(va) >= min_n and len(te) >= min_n
        # Confronta le tre coppie di insiemi.
        for a, b in [(tr, va), (tr, te), (va, te)]:
            # Penalità massima se non valutabile, altrimenti il log p-value del test KS.
            log_scores.append(ks_log_p(a, b, eps) if evaluable else np.log(eps))
    # Il punteggio dello split è la media.
    return float(np.mean(log_scores))


# Divide un DataFrame in due parti per gruppi: nessun gruppo viene spezzato tra le parti.
def _group_split(df, test_size, random_state):
    # Splitter di scikit-learn: una sola divisione con la frazione e il seme richiesti.
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    # Ottiene gli indici delle due parti usando la colonna dei gruppi.
    idx_a, idx_b = next(gss.split(df, df[TARGET_COLS], groups=df[GROUP_COL]))
    # Restituisce le due parti come DataFrame indipendenti.
    return df.iloc[idx_a].copy(), df.iloc[idx_b].copy()


# Prova n_trials split casuali per gruppi (train / resto, poi resto = val / test) e tiene
# quello con il punteggio KS migliore, cioè con le distribuzioni di pIC50 più simili.
def best_of_n_train_val_test_split(df, test_size, val_size_within_temp, n_trials, seed):
    # Generatore di numeri casuali con seme fisso: i tentativi sono sempre gli stessi.
    rng = np.random.default_rng(seed)
    # Miglior punteggio visto finora (parte da meno infinito).
    best_score = float('-inf')
    # Miglior split visto finora.
    best_split = None
    # Ripete il tentativo n_trials volte.
    for _ in range(n_trials):
        # Estrae un seme casuale per questo tentativo.
        s = int(rng.integers(0, 10 ** 9))
        # Primo taglio: train da una parte, "resto" (test_size del totale) dall'altra.
        tr_df, temp_df = _group_split(df, test_size, s)
        # Secondo taglio: il resto diviso a metà tra validation e test.
        va_df, te_df = _group_split(temp_df, val_size_within_temp, s + 1)
        # Valuta quanto sono simili le distribuzioni dei tre insiemi.
        sc = _split_score(tr_df, va_df, te_df)
        # Se è il migliore finora…
        if sc > best_score:
            # …ne aggiorna il punteggio…
            best_score = sc
            # …e conserva i tre insiemi.
            best_split = (tr_df, va_df, te_df)
    # Restituisce train, validation e test dello split migliore.
    return best_split


# -------------------------------------------------------------------------------------
# 4. Grafici delle distribuzioni (usati qui per lo split e al passo 5 per i fold della CV)
# -------------------------------------------------------------------------------------

# Disegna, per ognuna delle 9 isoforme, la distribuzione dei pIC50 di ogni insieme in
# `sets` (es. {'Train': matrice, 'Val': matrice, 'Test': None}) in due figure 3×3:
# densità (KDE) e distribuzione cumulativa (CDF). Un insieme None viene saltato.
def plot_target_distributions(sets, kde_path, cdf_path, title=None):
    # Crea la figura delle densità e quella delle cumulative, ognuna con 3×3 riquadri.
    (fig_k, ax_k), (fig_c, ax_c) = [plt.subplots(3, 3, figsize=(18, 12)) for _ in range(2)]
    # Rende le due griglie di riquadri delle liste semplici da 9 elementi.
    ax_k, ax_c = ax_k.flatten(), ax_c.flatten()
    # Per ogni isoforma (i = posizione della colonna)…
    for i, name in enumerate(TARGET_COLS):
        # …e per ogni insieme, con il suo numero d'ordine c per il colore…
        for c, (label, Y) in enumerate(sets.items()):
            # …salta gli insiemi assenti…
            if Y is None:
                # …passando al successivo.
                continue
            # Valori misurati di questa isoforma nell'insieme, ordinati in modo crescente.
            v = np.sort(Y[:, i][~np.isnan(Y[:, i])])
            # La densità ha senso solo con almeno due valori.
            if len(v) > 1:
                # Curva di densità riempita e semitrasparente.
                sns.kdeplot(v, label=label, fill=True, alpha=0.3, ax=ax_k[i], warn_singular=False)
            # La cumulativa basta che abbia almeno un valore.
            if len(v) > 0:
                # A ogni valore ordinato corrisponde la frazione di valori minori o uguali.
                ax_c[i].plot(v, np.arange(1, len(v) + 1) / len(v), label=label, color=f'C{c}', lw=1.5)
        # Titolo, griglia ed etichetta dell'asse y dei due riquadri di questa isoforma.
        for ax in (ax_k[i], ax_c[i]):
            # Il titolo è il nome dell'isoforma.
            ax.set_title(name)
            # Griglia leggera di sfondo.
            ax.grid(True, alpha=0.3)
        # Asse y della cumulativa: F(x) = frazione di valori <= x.
        ax_c[i].set_ylabel('F(x)')
    # Salva entrambe le figure.
    for fig, axes, path in ((fig_k, ax_k, kde_path), (fig_c, ax_c, cdf_path)):
        # La legenda solo nel primo riquadro, per non ripeterla 9 volte.
        axes[0].legend()
        # Titolo generale della figura, se richiesto.
        if title:
            # Scrive il titolo sopra la griglia.
            fig.suptitle(title)
        # Sistema gli spazi tra i riquadri.
        fig.tight_layout()
        # Salva l'immagine ad alta risoluzione.
        fig.savefig(path, dpi=600, bbox_inches='tight')
        # Chiude la figura per liberare memoria (non la mostra nel notebook).
        plt.close(fig)


# -------------------------------------------------------------------------------------
# 5. Funzione principale chiamata dal notebook
# -------------------------------------------------------------------------------------

# Esegue tutto lo split: fingerprint, clustering, scelta dello split migliore, controllo
# che nessun gruppo sia diviso tra gli insiemi, grafici e salvataggio dei tre CSV.
# Nel notebook viene chiamata solo se la cartella split/ di riferimento non esiste.
def splitter(df, output_dir, test_size=0.2, val_size_within_temp=0.5, n_trials=50, seed=42,
             umap_n_neighbors=30, umap_min_dist=0.25, umap_n_components=2,
             hdbscan_min_cluster_size=15, hdbscan_min_samples=None):
    # Lavora su una copia, così il DataFrame del chiamante non viene modificato.
    df = df.copy()
    # Per ogni colonna target…
    for c in TARGET_COLS:
        # …forza i valori a numeri; testo o celle sporche diventano NaN.
        df[c] = pd.to_numeric(df[c], errors='coerce')
    # Calcola i fingerprint e scopre quali righe hanno SMILES validi.
    fps, valid_idx = generate_fingerprints(df[SMILES_COL].tolist())
    # Tiene solo le righe valide e rinumera l'indice da 0.
    df = df.iloc[valid_idx].reset_index(drop=True)
    # Controllo di sicurezza: servono almeno due molecole.
    if len(df) <= 1:
        # Errore esplicito se il dataset è vuoto.
        raise ValueError('Empty dataset after SMILES parsing.')
    # Calcola il gruppo chimico di ogni molecola e lo salva nella colonna dei gruppi.
    df[GROUP_COL] = cluster_hdbscan(fps, hdbscan_min_cluster_size, hdbscan_min_samples,
                                    umap_n_components, umap_n_neighbors, umap_min_dist, seed)
    # Controllo di sicurezza: con un solo gruppo non si può dividere nulla.
    if df[GROUP_COL].nunique() <= 1:
        # Errore esplicito con il suggerimento per risolverlo.
        raise ValueError('Only one cluster: lower hdbscan_min_cluster_size.')
    # Sceglie lo split migliore tra n_trials tentativi.
    train_df, val_df, test_df = best_of_n_train_val_test_split(df, test_size, val_size_within_temp, n_trials, seed)
    # Insiemi dei gruppi presenti in train, validation e test.
    g_tr, g_va, g_te = (set(d[GROUP_COL]) for d in (train_df, val_df, test_df))
    # Se un gruppo compare in due insiemi diversi c'è "leakage" (fuga di informazione).
    if g_tr & g_va or g_tr & g_te or g_va & g_te:
        # In quel caso si ferma con un errore.
        raise AssertionError('Group-level data leakage detected between train/val/test.')
    # Crea la cartella di output se non esiste.
    os.makedirs(output_dir, exist_ok=True)
    # Matrici dei pIC50 dei tre insiemi, nel formato richiesto dalla funzione dei grafici.
    sets = {name: d[TARGET_COLS].values.astype(np.float32)
            for name, d in (('Train', train_df), ('Val', val_df), ('Test', test_df))}
    # Disegna e salva le distribuzioni dei tre insiemi.
    plot_target_distributions(sets, os.path.join(output_dir, 'split_distributions.png'),
                              os.path.join(output_dir, 'split_cdfs.png'))
    # Salva i tre insiemi come CSV.
    for name, d in (('train', train_df), ('val', val_df), ('test', test_df)):
        # Un file per insieme, senza la colonna dell'indice.
        d.to_csv(os.path.join(output_dir, f'{name}_set_multitarget.csv'), index=False)
    # Restituisce i tre insiemi.
    return train_df, val_df, test_df
