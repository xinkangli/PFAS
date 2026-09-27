"""Molecular descriptors, Morgan bits, SMARTS counts and graph-derived F counts.

Feature order: 14 descriptors, 1024 radius-2 Morgan bits, 14 SMARTS counts,
then n_F and n_CF1 through n_CF4. No MACCS keys or external F lookup.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")  # 关掉 RDKit 对脏 SMILES 的刷屏告警


# --------------------------------------------------------------------------- #
# 1. 理化描述符（与分配行为相关的精选子集，便于解释；想要全量可设 use_full=True）
# --------------------------------------------------------------------------- #
_CURATED_DESCRIPTORS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Crippen.MolLogP,                 # Crippen logP，作为 logKow 的结构先验
    "MolMR": Crippen.MolMR,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "LabuteASA": rdMolDescriptors.CalcLabuteASA,
    "NumHAcceptors": rdMolDescriptors.CalcNumHBA,
    "NumHDonors": rdMolDescriptors.CalcNumHBD,
    "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    "NumHeavyAtoms": lambda m: float(m.GetNumHeavyAtoms()),
    "NumHalogens": lambda m: float(sum(a.GetAtomicNum() in (9, 17, 35, 53)
                                       for a in m.GetAtoms())),
    "NumFluorine": lambda m: float(sum(a.GetAtomicNum() == 9 for a in m.GetAtoms())),
}


# --------------------------------------------------------------------------- #
# 2. PFAS 专属 SMARTS（head group / 支链 / 醚桥 / 间隔），统计匹配次数
# --------------------------------------------------------------------------- #
_PFAS_SMARTS = {
    # —— head group ——
    "carboxylic_acid":  "[CX3](=O)[OX2H1]",
    "carboxylate":      "[CX3](=O)[O-]",
    "sulfonic_acid":    "[SX4](=O)(=O)[OX2H1]",
    "sulfonate":        "[SX4](=O)(=O)[O-]",
    "sulfonamide":      "[SX4](=O)(=O)[NX3]",
    "phosphonic":       "[PX4](=O)([OX2H,O-])",
    "phosphate_ester":  "[PX4](=O)([OX2][#6])",
    # —— 全氟骨架 ——
    "cf3_terminal":     "[CX4](F)(F)F",
    "cf2_unit":         "[CX4](F)(F)",
    "perfluoro_chain":  "[CX4](F)(F)[CX4](F)(F)",   # 连续 -CF2-CF2-
    # —— 醚桥（如 GenX/HFPO-DA、ADONA 这类含氧 PFAS 的关键结构）——
    "ether_bridge":     "[#6][OX2][#6]",
    "perfluoro_ether":  "[CX4](F)(F)[OX2][CX4](F)",
    # —— 氟调聚物（FTOH/FTS 的 -CH2CH2- 间隔）——
    "fluorotelomer_spacer": "[CX4](F)(F)[CH2][CH2]",
    # —— 支链点（季/叔碳上挂氟，区分支链 vs 直链 PFAS，影响 logKoa/half-life）——
    "branch_cf":        "[CX4](F)([CX4])([CX4])",
}


def _build_smarts_patterns() -> dict[str, Chem.Mol]:
    pats = {}
    for name, smarts in _PFAS_SMARTS.items():
        p = Chem.MolFromSmarts(smarts)
        if p is None:
            warnings.warn(f"SMARTS 编译失败，已跳过：{name} = {smarts}")
            continue
        pats[name] = p
    return pats


# --------------------------------------------------------------------------- #
# 3. Featurizer
# --------------------------------------------------------------------------- #
@dataclass
class Featurizer:
    """SMILES → 特征矩阵。一次性 fit 出列名，之后 transform 任意 SMILES 列表。"""

    morgan_radius: int = 2
    morgan_nbits: int = 1024
    use_full_descriptors: bool = False
    fstats_cols: Sequence[str] = ("n_F", "n_CF1", "n_CF2", "n_CF3", "n_CF4")

    feature_names_: list[str] = field(default_factory=list, init=False)
    _smarts: dict = field(default_factory=dict, init=False)
    _desc_fns: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._smarts = _build_smarts_patterns()
        if self.use_full_descriptors:
            self._desc_fns = {n: f for n, f in Descriptors._descList}
        else:
            self._desc_fns = dict(_CURATED_DESCRIPTORS)

        # 锁定列顺序：描述符 → Morgan → SMARTS → F-stats
        self.feature_names_ = (
            [f"desc__{n}" for n in self._desc_fns]
            + [f"morgan__{i}" for i in range(self.morgan_nbits)]
            + [f"smarts__{n}" for n in self._smarts]
            + [f"fstat__{c}" for c in self.fstats_cols]
        )

    # --- 单分子 -> 1D 向量（F-stats 在 transform 中由分子图生成）---
    def _featurize_mol(self, mol: Chem.Mol) -> np.ndarray:
        # 描述符
        desc = np.empty(len(self._desc_fns), dtype=np.float32)
        for i, fn in enumerate(self._desc_fns.values()):
            try:
                desc[i] = float(fn(mol))
            except Exception:
                desc[i] = np.nan

        # Morgan 指纹（经典 API；新版 RDKit 可能提示 deprecation，功能不受影响）
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.morgan_radius, nBits=self.morgan_nbits
        )
        morgan = np.zeros(self.morgan_nbits, dtype=np.float32)
        for b in fp.GetOnBits():
            morgan[b] = 1.0

        # SMARTS 计数
        smarts = np.array(
            [len(mol.GetSubstructMatches(p)) for p in self._smarts.values()],
            dtype=np.float32,
        )
        return np.concatenate([desc, morgan, smarts])

    def transform(
        self,
        smiles: Sequence[str],
        fstats_df: pd.DataFrame | None = None,
        fstats_key: str = "SMILES",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        参数
        ----
        smiles      : SMILES 列表
        fstats_df   : 旧接口保留参数；不使用查找表，始终从分子图计算
        fstats_key  : F_stats_df 里用于 join 的列名（默认 'SMILES'）

        返回
        ----
        X          : float32 [n, n_features]，解析失败行为 NaN
        valid_mask : bool [n]，True=SMILES 解析成功
        """
        n = len(smiles)
        n_struct = len(self._desc_fns) + self.morgan_nbits + len(self._smarts)
        X_struct = np.full((n, n_struct), np.nan, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)

        for i, smi in enumerate(smiles):
            if not isinstance(smi, str) or not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                X_struct[i] = self._featurize_mol(mol)
                valid[i] = True
            except Exception:
                valid[i] = False

        # Count fluorine directly from each molecular graph.
        X_fs = np.zeros((n, len(self.fstats_cols)), dtype=np.float32)
        for i, smi in enumerate(smiles):
            if not valid[i]:
                continue
            mol = Chem.MolFromSmiles(smi)
            X_fs[i] = [sum(a.GetAtomicNum() == 9 for a in mol.GetAtoms())] + [
                sum(a.GetAtomicNum() == 6 and
                    sum(v.GetAtomicNum() == 9 for v in a.GetNeighbors()) == k
                    for a in mol.GetAtoms()) for k in range(1, 5)]

        X = np.concatenate([X_struct, X_fs], axis=1)
        return X, valid


def impute_and_record(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把残留 NaN（个别描述符算不出来的）用列中位数补上，返回补后矩阵 + 列中位数。"""
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    inds = np.where(np.isnan(X))
    X = X.copy()
    X[inds] = np.take(med, inds[1])
    return X, med
