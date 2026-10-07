# =====================================================================================
# PASSO 0 di 7 — COSTANTI CONDIVISE
# Questo file non esegue nulla: definisce solo i nomi delle colonne del dataset che tutti
# gli altri moduli (s1 … s7) e i notebook importano, così sono scritti in un solo posto.
# =====================================================================================

# Le nove isoforme umane di istone deacetilasi (HDAC) di cui il modello predice la potenza.
ISOFORMS = ['HDAC1', 'HDAC2', 'HDAC3', 'HDAC4', 'HDAC5', 'HDAC6', 'HDAC7', 'HDAC8', 'HDAC9']

# Nomi delle colonne di data.xlsx con il pIC50 di ogni isoforma ('HDAC1_HUMAN', …); una
# cella vuota (NaN) significa che quella molecola non è stata misurata su quell'isoforma.
TARGET_COLS = [f'{iso}_HUMAN' for iso in ISOFORMS]

# Colonna che contiene la struttura della molecola in formato SMILES (stringa di testo).
SMILES_COL = 'Smiles'

# Colonna con l'identificativo del gruppo (cluster chimico) di ogni molecola, usata per
# tenere molecole simili sempre dallo stesso lato degli split. Il nome è storico (una volta
# il clustering era Butina): oggi i gruppi vengono da HDBSCAN, ma la colonna si chiama
# ancora così perché è scritta nei CSV di riferimento in split/.
GROUP_COL = 'Cluster_Butina'
