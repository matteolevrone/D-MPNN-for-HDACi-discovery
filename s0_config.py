# data.xlsx pIC50 column per isoform
ISOFORMS = ['HDAC1', 'HDAC2', 'HDAC3', 'HDAC4', 'HDAC5', 'HDAC6', 'HDAC7', 'HDAC8', 'HDAC9']
CLASS_I_ISOFORMS = ['HDAC1', 'HDAC2', 'HDAC3', 'HDAC8']
TARGET_COLS = [f'{iso}_HUMAN' for iso in ISOFORMS]

# Notable columns
SMILES_COL = 'Smiles'
GROUP_COL = 'Cluster'

# Limit to determine underpowered isoforms
UNDERPOWERED_N = 25
