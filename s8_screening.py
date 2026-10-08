# Class I selectivity scoring and structural screening

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from s0_config import TARGET_COLS, CLASS_I_ISOFORMS

CLASS_I_COLS = [f'{iso}_HUMAN' for iso in CLASS_I_ISOFORMS]
OTHER_COLS = [c for c in TARGET_COLS if c not in CLASS_I_COLS]


# Selectivity score
def selectivity_score(df):
    mean_i = df[CLASS_I_COLS].mean(axis=1)
    max_o = df[OTHER_COLS].max(axis=1)
    return mean_i, max_o, mean_i - max_o


# Cleaning an external SMILES library (BitterDB)
def largest_fragment(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return Chem.MolToSmiles(max(frags, key=lambda m: m.GetNumHeavyAtoms())) if frags else None


# Score tables and distribution plots

# Saves a table image of the top-n compounds by score.
def save_top_table(df, smiles_col, output_path, title, n=15):
    d = df.sort_values('score', ascending=False).head(n).copy()
    d['mean_class_I'], d['max_others'], _ = (v.round(2) for v in selectivity_score(d))
    d['score'] = d['score'].round(2)
    d[smiles_col] = d[smiles_col].astype(str).str.slice(0, 30)
    cols = [smiles_col, 'mean_class_I', 'max_others', 'score']
    labels = ['SMILES', 'mean Class I', 'max other', 'score']
    fig, ax = plt.subplots(figsize=(9, 0.45 * (n + 1)))
    ax.axis('off')
    ax.set_title(title, fontsize=12, weight='bold', pad=12)
    tbl = ax.table(cellText=d[cols].values, colLabels=labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    for j in range(len(cols)):  # header row: blue background, bold white text
        cell = tbl[0, j]; cell.set_facecolor('#4472C4'); cell.set_text_props(color='white', weight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.show()

def ecdf(x):
    xs = np.sort(x)
    return xs, np.arange(1, len(xs) + 1) / len(xs)

def summ(name, s):
    print(f'{name:8s}: mean={s.mean():.2f}  median={np.median(s):.2f}  max={s.max():.2f}  min={s.min():.2f}  '
          f'| score>0: {(s > 0).sum()} ({100 * (s > 0).mean():.1f}%)')


# Zinc-binding groups and drug-likeness

# Known HDAC zinc-binding groups (ZBGs)
ZBG_SMARTS = {
    'hydroxamic acid (pan-HDAC)': '[CX3](=[OX1])[NX3][OX2H1]',
    'aminoanilide (Class I)': '[CX3](=[OX1])[NX3]c1ccccc1[NX3;H1,H2]',
    'mercaptoacetamide': '[SX2H][CH2][CX3](=[OX1])[NX3]',
    '2-mercaptobenzamide': '[SX2H]c1ccccc1[CX3](=[OX1])[NX3]',
    'alpha-ketoamide': '[CX3](=[OX1])[CX3](=[OX1])[NX3]',
    'trifluoromethyl ketone': '[CX3](=[OX1])C(F)(F)F',
    'difluoromethyl ketone': '[CX3](=[OX1])[CX4](F)F',
    'acylhydrazide': '[CX3](=[OX1])[NX3][NX3]',
    'oxadiazol-5-one': 'c1noc(=O)[nH]1',
    'hydroxypyridinethione (HOPTO)': '[OX2H]c1ccc[nX2]c1[SX2H]',
    'tropolone': 'Oc1cccccc1=O',
    'squaramide': '[#8]=c1c([#7])c([#7])c1=[#8]',
    '8-hydroxyquinoline': '[OX2H]c1cccc2cccnc12',
    'catechol': 'c1cc(O)c(O)cc1',
    'boronic acid': '[BX3]([OX2H])[OX2H]',
    'thiol': '[SX2H]',
    'carboxylic acid (weak)': '[CX3](=[OX1])[OX2H1]',
}
ZBG_PATT = {k: Chem.MolFromSmarts(v) for k, v in ZBG_SMARTS.items()}


# First known ZBG found in the molecule
def zbg_hit(mol):
    for name, patt in ZBG_PATT.items():
        match = mol.GetSubstructMatch(patt)
        if match:
            return name, list(match)
    return None, []


# Drug-likeness descriptors and the number of Lipinski
def judge(mol):
    mw, logp = Descriptors.MolWt(mol), Crippen.MolLogP(mol)
    hbd, hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
    viol = int((mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10))
    return {'MW': round(mw, 1), 'logP': round(logp, 2), 'HBD': hbd, 'HBA': hba,
            'TPSA': round(Descriptors.TPSA(mol), 1), 'RotB': Descriptors.NumRotatableBonds(mol),
            'Lipinski_viol': viol}


# Known ZBG name for a SMILES
def zbg_of_smiles(smi):
    m = Chem.MolFromSmiles(str(smi))
    return None if m is None else zbg_hit(m)[0]