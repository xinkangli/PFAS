"""Training-only context vocabulary and imputation, preserved from the executed model."""
from __future__ import annotations
import numpy as np
import pandas as pd

BIO_CAT_COLS = ["organism_group", "tissue", "exposure_route", "water_type", "study_type"]

BIO_NUM_COLS = ["exposure_days"]

BIO_TOPK = {"tissue": 12, "exposure_route": 8, "organism_group": 8,
            "water_type": 4, "study_type": 4}

class BioEncoder:
    """学训练集类别表，对训练/验证/池一致地 one-hot。

    设计：每个 cat 列保留 top-K 高频值，其余归 OTHER；NaN 归 NA；
         数值列用训练集中位数填补。最终拼成定长向量。
    """

    def __init__(self, cat_cols: list[str], num_cols: list[str],
                 topk: dict[str, int]):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.topk = topk
        self.vocab: dict[str, list[str]] = {}        # {col: [val1, val2, ..., 'OTHER', 'NA']}
        self.num_median: dict[str, float] = {}
        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame):
        for c in self.cat_cols:
            vc = df[c].dropna().astype(str).value_counts()
            keep = list(vc.head(self.topk.get(c, 8)).index)
            self.vocab[c] = keep + ["OTHER", "NA"]
        for c in self.num_cols:
            v = pd.to_numeric(df[c], errors="coerce")
            self.num_median[c] = float(np.nanmedian(v)) if v.notna().any() else 0.0

        names = []
        for c in self.cat_cols:
            for v in self.vocab[c]:
                names.append(f"bio__{c}={v}")
        for c in self.num_cols:
            names.append(f"bio__{c}")
        self.feature_names = names
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        cols = []
        for c in self.cat_cols:
            vocab = self.vocab[c]
            idx = {v: i for i, v in enumerate(vocab)}
            mat = np.zeros((n, len(vocab)), dtype=np.float32)
            ser = df[c].astype("object")
            for i, val in enumerate(ser.values):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    mat[i, idx["NA"]] = 1.0
                else:
                    s = str(val)
                    mat[i, idx.get(s, idx["OTHER"])] = 1.0
            cols.append(mat)
        for c in self.num_cols:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy()
            v = np.where(np.isnan(v), self.num_median[c], v).astype(np.float32)
            cols.append(v.reshape(-1, 1))
        return np.concatenate(cols, axis=1)
