"""
PFAS 目标清单
=============
所有 collector 都用这个表作为输入。
CAS 是匹配主键，synonyms 用于在不同数据库里检索（不同库用不同名字）。

字段:
    cas        : CAS Registry Number (主键)
    name       : 推荐英文名 (paper 用)
    short      : 缩写 (PFOS / PFOA …)
    formula    : 分子式 (用于校对)
    chain_len  : 全氟碳链长度 (后续按链长分析很有用)
    family     : PFCA / PFSA / FTOH / sulfonamide / 替代品 / 其他
    synonyms   : 在不同数据库里出现过的别名

如果你之后要扩展 GenX, ADONA, F-53B 等替代品，直接在 PFAS_TARGETS 末尾加行即可。
"""
from __future__ import annotations

PFAS_TARGETS = [
    # ---- PFSAs (perfluoroalkyl sulfonic acids) ----
    {"cas": "1763-23-1",   "name": "Perfluorooctanesulfonic acid",  "short": "PFOS",  "formula": "C8HF17O3S",  "chain_len": 8, "family": "PFSA",
     "synonyms": ["PFOS", "Perfluorooctane sulfonate", "Heptadecafluorooctane-1-sulfonic acid"]},
    {"cas": "355-46-4",    "name": "Perfluorohexanesulfonic acid", "short": "PFHxS", "formula": "C6HF13O3S",  "chain_len": 6, "family": "PFSA",
     "synonyms": ["PFHxS", "Perfluorohexane sulfonate", "Tridecafluorohexane-1-sulfonic acid"]},
    {"cas": "375-73-5",    "name": "Perfluorobutanesulfonic acid", "short": "PFBS",  "formula": "C4HF9O3S",   "chain_len": 4, "family": "PFSA",
     "synonyms": ["PFBS", "Perfluorobutane sulfonate", "Nonafluorobutane-1-sulfonic acid"]},

    # ---- PFCAs (perfluoroalkyl carboxylic acids) ----
    {"cas": "335-67-1",    "name": "Perfluorooctanoic acid",   "short": "PFOA",  "formula": "C8HF15O2",   "chain_len": 8, "family": "PFCA",
     "synonyms": ["PFOA", "Pentadecafluoro-1-octanoic acid", "Pentadecafluorooctanoic acid"]},
    {"cas": "375-95-1",    "name": "Perfluorononanoic acid",   "short": "PFNA",  "formula": "C9HF17O2",   "chain_len": 9, "family": "PFCA",
     "synonyms": ["PFNA", "Heptadecafluorononanoic acid"]},
    {"cas": "335-76-2",    "name": "Perfluorodecanoic acid",   "short": "PFDA",  "formula": "C10HF19O2",  "chain_len": 10, "family": "PFCA",
     "synonyms": ["PFDA", "Nonadecafluorodecanoic acid"]},
    {"cas": "2058-94-8",   "name": "Perfluoroundecanoic acid", "short": "PFUnDA", "formula": "C11HF21O2", "chain_len": 11, "family": "PFCA",
     "synonyms": ["PFUnDA", "PFUdA", "Henicosafluoroundecanoic acid"]},
    {"cas": "307-55-1",    "name": "Perfluorododecanoic acid", "short": "PFDoDA", "formula": "C12HF23O2", "chain_len": 12, "family": "PFCA",
     "synonyms": ["PFDoDA", "PFDoA", "Tricosafluorododecanoic acid"]},
    {"cas": "375-22-4",    "name": "Perfluorobutanoic acid",   "short": "PFBA",  "formula": "C4HF7O2",    "chain_len": 4, "family": "PFCA",
     "synonyms": ["PFBA", "Heptafluorobutyric acid"]},
    {"cas": "2706-90-3",   "name": "Perfluoropentanoic acid",  "short": "PFPeA", "formula": "C5HF9O2",    "chain_len": 5, "family": "PFCA",
     "synonyms": ["PFPeA"]},
    {"cas": "307-24-4",    "name": "Perfluorohexanoic acid",   "short": "PFHxA", "formula": "C6HF11O2",   "chain_len": 6, "family": "PFCA",
     "synonyms": ["PFHxA", "Undecafluorohexanoic acid"]},
    {"cas": "375-85-9",    "name": "Perfluoroheptanoic acid",  "short": "PFHpA", "formula": "C7HF13O2",   "chain_len": 7, "family": "PFCA",
     "synonyms": ["PFHpA"]},

    # ---- Replacements / next-gen ----
    {"cas": "13252-13-6",  "name": "Hexafluoropropylene oxide dimer acid", "short": "HFPO-DA", "formula": "C6HF11O3", "chain_len": 6, "family": "PFCA_ether",
     "synonyms": ["GenX", "HFPO-DA", "Perfluoro-2-propoxypropanoic acid"]},
    {"cas": "919005-14-4", "name": "Ammonium 4,8-dioxa-3H-perfluorononanoate", "short": "ADONA", "formula": "C7H4F12NO4", "chain_len": 7, "family": "PFCA_ether",
     "synonyms": ["ADONA"]},
    {"cas": "73606-19-6",  "name": "Potassium 9-chlorohexadecafluoro-3-oxanonane-1-sulfonate", "short": "F-53B (6:2)", "formula": "C8ClF16KO4S", "chain_len": 8, "family": "PFSA_Cl_ether",
     "synonyms": ["F-53B", "6:2 Cl-PFESA", "Cl-PFESA"]},

    # ---- Precursors / fluorotelomers (often reported alongside) ----
    {"cas": "678-39-7",    "name": "8:2 Fluorotelomer alcohol", "short": "8:2 FTOH", "formula": "C10H5F17O", "chain_len": 8, "family": "FTOH",
     "synonyms": ["8:2 FTOH", "1H,1H,2H,2H-Perfluorodecan-1-ol"]},
    {"cas": "647-42-7",    "name": "6:2 Fluorotelomer alcohol", "short": "6:2 FTOH", "formula": "C8H5F13O",  "chain_len": 6, "family": "FTOH",
     "synonyms": ["6:2 FTOH", "1H,1H,2H,2H-Perfluorooctan-1-ol"]},
    {"cas": "754-91-6",    "name": "Perfluorooctanesulfonamide", "short": "FOSA",   "formula": "C8H2F17NO2S", "chain_len": 8, "family": "Sulfonamide",
     "synonyms": ["FOSA", "PFOSA", "Perfluorooctane sulfonamide"]},
]


def by_short(short: str) -> dict | None:
    """Lookup by abbreviation, e.g. 'PFOS'."""
    for r in PFAS_TARGETS:
        if r["short"].upper() == short.upper():
            return r
    return None


def by_cas(cas: str) -> dict | None:
    """Lookup by CAS."""
    cas_norm = cas.strip()
    for r in PFAS_TARGETS:
        if r["cas"] == cas_norm:
            return r
    return None


def all_cas() -> list[str]:
    return [r["cas"] for r in PFAS_TARGETS]


def cas_to_meta() -> dict[str, dict]:
    return {r["cas"]: r for r in PFAS_TARGETS}


if __name__ == "__main__":
    print(f"Total PFAS targets: {len(PFAS_TARGETS)}")
    for r in PFAS_TARGETS:
        print(f"  {r['short']:12s} CAS {r['cas']:14s} family={r['family']}")
