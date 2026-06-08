"""
Merge SMILES from all BAF and BCF Excel files into a single unique SMILES file,
then count F-atom / CF2 / CF3 occurrences.
"""
import os
import sys
import pandas as pd
from collections import Counter
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ROOT = "/7t/lxkzero/claude/qd"

# All main Excel files across the two folders.
EXCELS = [
    # BAF
    "BAF/Arnot_Gobas_BAF/Arnot_Gobas_BAF.xlsx",
    "BAF/Arnot_Gobas_BAF/Arnot_Gobas_BAF_only.xlsx",
    "BAF/ECOTOX_BAF/ECOTOX_BAF.xlsx",
    "BAF/ECOTOX_BAF/ECOTOX_BAF_all.xlsx",
    "BAF/ECOTOX_BAF/ECOTOX_BAF_fish_only.xlsx",
    "BAF/EPA_BAF/EPA_RawBAF.xlsx",
    "BAF/EPA_BAF/EPA_NationalBAF.xlsx",
    # BCF
    "BCF/Arnot_Gobas_BCF/Arnot_Gobas_BCF.xlsx",
    "BCF/CEFIC_LRI_BCF/CEFIC_LRI_EURAS_BCF.xlsx",
    "BCF/CEFIC_LRI_BCF/CEFIC_LRI_EURAS_BCF_all.xlsx",
    "BCF/OECD_QSAR_Toolbox_BCF/OECD_QSARToolbox_CAESAR_BCF.xlsx",
    "BCF/UCI_QSAR_BCF/UCI_QSAR_fish_BCF.xlsx",
]


def canonicalize(smi: str):
    """Return canonical SMILES (largest fragment, neutralized as parsed)."""
    if smi is None:
        return None
    smi = str(smi).strip()
    if not smi or smi.lower() in ("nan", "none"):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Keep the largest fragment for salts/mixtures.
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Step 1: gather raw SMILES from all files
# --------------------------------------------------------------------------
raw_records = []
for rel in EXCELS:
    path = os.path.join(ROOT, rel)
    df = pd.read_excel(path)
    if "SMILES" not in df.columns:
        print(f"[WARN] {rel}: no SMILES column", file=sys.stderr)
        continue
    for s in df["SMILES"].dropna().astype(str):
        raw_records.append((rel, s))
    print(f"  {rel}: {df['SMILES'].notna().sum()} SMILES rows")

print(f"\nTotal raw SMILES rows: {len(raw_records)}")

# --------------------------------------------------------------------------
# Step 2: canonicalize, dedup, record source provenance
# --------------------------------------------------------------------------
canon_to_sources = {}
n_invalid = 0
for src, smi in raw_records:
    c = canonicalize(smi)
    if c is None:
        n_invalid += 1
        continue
    canon_to_sources.setdefault(c, set()).add(src)

unique_smiles = sorted(canon_to_sources.keys())
print(f"Unique canonical SMILES: {len(unique_smiles)}")
print(f"Failed to parse: {n_invalid}")

# --------------------------------------------------------------------------
# Step 3: write the final unique SMILES file
# --------------------------------------------------------------------------
out_smi = os.path.join(ROOT, "unique_smiles.smi")
out_csv = os.path.join(ROOT, "unique_smiles.csv")
with open(out_smi, "w") as f:
    for s in unique_smiles:
        f.write(s + "\n")

rows = []
for s in unique_smiles:
    srcs = sorted(canon_to_sources[s])
    in_baf = any(x.startswith("BAF/") for x in srcs)
    in_bcf = any(x.startswith("BCF/") for x in srcs)
    rows.append({
        "SMILES": s,
        "in_BAF": int(in_baf),
        "in_BCF": int(in_bcf),
        "n_sources": len(srcs),
        "sources": ";".join(srcs),
    })
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"Wrote: {out_smi}")
print(f"Wrote: {out_csv}")

# --------------------------------------------------------------------------
# Step 4: F / CF2 / CF3 statistics
# --------------------------------------------------------------------------
# SMARTS:
#   F atom:                            [F]
#   CF3 group (C bonded to exactly 3 F): [CX4H0](F)(F)F  -- but CF3 carbon
#                                       has 1 non-F neighbor so degree=4, H=0
#   CF2 group (C bonded to exactly 2 F): [CX4H0,CX4H1,CX4H2](F)(F) with not 3 F.
# We use the more general & correct approach: enumerate carbons; for each
# carbon, count F neighbors and bucket as CF1 / CF2 / CF3 / CF4.

patt_F = Chem.MolFromSmarts("[F]")

total_F_atoms = 0
n_mol_with_F = 0
n_mol_with_CF2 = 0
n_mol_with_CF3 = 0
total_CF2_groups = 0  # carbons with exactly 2 F neighbors
total_CF3_groups = 0  # carbons with exactly 3 F neighbors
total_CF1_groups = 0
total_CF4_groups = 0  # CF4-type (tetrafluoromethyl etc, rare)

f_count_hist = Counter()  # number-of-F per molecule -> count

per_mol = []
for s in unique_smiles:
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        continue
    n_F = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    total_F_atoms += n_F
    if n_F > 0:
        n_mol_with_F += 1
    f_count_hist[n_F] += 1

    n_cf2 = 0
    n_cf3 = 0
    n_cf1 = 0
    n_cf4 = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        nF = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 9)
        if nF == 1:
            n_cf1 += 1
        elif nF == 2:
            n_cf2 += 1
        elif nF == 3:
            n_cf3 += 1
        elif nF == 4:
            n_cf4 += 1
    total_CF1_groups += n_cf1
    total_CF2_groups += n_cf2
    total_CF3_groups += n_cf3
    total_CF4_groups += n_cf4
    if n_cf2 > 0:
        n_mol_with_CF2 += 1
    if n_cf3 > 0:
        n_mol_with_CF3 += 1

    per_mol.append({"SMILES": s, "n_F": n_F, "n_CF1": n_cf1,
                    "n_CF2": n_cf2, "n_CF3": n_cf3, "n_CF4": n_cf4})

pd.DataFrame(per_mol).to_csv(os.path.join(ROOT, "smiles_F_stats_per_mol.csv"),
                             index=False)

print("\n" + "=" * 60)
print("F / CF2 / CF3 statistics on the UNIQUE SMILES set")
print("=" * 60)
N = len(unique_smiles)
print(f"Unique molecules                : {N}")
print()
print(f"-- F atom --")
print(f"  Total F atoms (sum)           : {total_F_atoms}")
print(f"  Molecules containing >=1 F    : {n_mol_with_F}  ({n_mol_with_F/N*100:.2f}%)")
print()
print(f"-- CF2 group (carbon w/ exactly 2 F neighbors) --")
print(f"  Total CF2 groups (sum)        : {total_CF2_groups}")
print(f"  Molecules containing >=1 CF2  : {n_mol_with_CF2}  ({n_mol_with_CF2/N*100:.2f}%)")
print()
print(f"-- CF3 group (carbon w/ exactly 3 F neighbors) --")
print(f"  Total CF3 groups (sum)        : {total_CF3_groups}")
print(f"  Molecules containing >=1 CF3  : {n_mol_with_CF3}  ({n_mol_with_CF3/N*100:.2f}%)")
print()
print(f"-- For completeness --")
print(f"  Total CF1 groups (mono-fluoro C): {total_CF1_groups}")
print(f"  Total CF4 groups (per-fluoro C) : {total_CF4_groups}")
print()
print("Distribution of #F per molecule (top entries):")
for k in sorted(f_count_hist.keys()):
    print(f"   F={k:>3d}: {f_count_hist[k]} molecules")
