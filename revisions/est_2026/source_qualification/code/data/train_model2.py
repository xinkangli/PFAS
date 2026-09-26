"""
train_model2.py — 模型2（Bioaccumulation Foundation Model · 论文核心）

数据流（对应建模流程.txt 阶段2）：
    Model2_Bioaccumulation.xlsx
        └─ X_Y_joined  2,404 × 48
              ├─ 结构：     SMILES, n_F/n_CF*  → Featurizer（复用模型1的特征工程）
              ├─ Transport：用模型1 predictions.csv 按 SMILES join，得 7 endpoint × (pred, unc)
              ├─ 生物上下文：organism_group/tissue/exposure_route/water_type/study_type/exposure_days
              ├─ Y：       log_value（按 endpoint_type 拆 head）
              └─ 权重：    reliability → sample_weight

    输出：
        model_data/model2_out/model2_predictions.csv   对 X_pool_full（127k）的 logBAF/logBCF/logBCFD 预测
        model_data/model2_out/model2_state.pt          模型权重
        model_data/model2_out/model2_meta.json         特征/标准化/类别表/权重映射

依赖：torch, rdkit, pandas, numpy
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from featurize import Featurizer, impute_and_record
from model import MultiTaskTransportNet, masked_multitask_loss, mc_predict


# =========================================================================== #
# 配置
# =========================================================================== #
DATA_ROOT = Path("/7t/lxkzero/claude/qd/data/model_data")
M1_OUT = DATA_ROOT / "model1_out"
M2_OUT = DATA_ROOT / "model2_out"
XLSX_PATH = DATA_ROOT / "Model2_Bioaccumulation.xlsx"
M1_PRED_CSV = M1_OUT / "model1_predictions.csv"        # 阶段1产物
M1_PRED_PARQUET = M1_OUT / "model1_predictions.parquet"
POOL_XLSX = DATA_ROOT / "Model1_Transport.xlsx"        # X_pool_full 在这

# 模型1的 7 个 endpoint，会按 pred_/unc_ 前缀拼成 14 维
M1_ENDPOINTS = ["logKow", "logKoc", "water_solubility",
                "Henry", "vapor_pressure", "pKa", "logD"]

# 模型2的 head：按 endpoint_type 拆
ENDPOINTS = ["logBAF", "logBCF", "logBCFD"]
ENDPOINT_FROM_TYPE = {"BAF": "logBAF", "BCF": "logBCF", "BCFD": "logBCFD"}

# 生物上下文：哪些列做 one-hot、哪些做数值
BIO_CAT_COLS = ["organism_group", "tissue", "exposure_route", "water_type", "study_type"]
BIO_NUM_COLS = ["exposure_days"]                       # test_concentration/temperature 全空，不用
BIO_TOPK = {"tissue": 12, "exposure_route": 8, "organism_group": 8,
            "water_type": 4, "study_type": 4}          # 长尾类合并为 OTHER

# reliability → 样本权重
RELIABILITY_W = {
    "ITRC-compiled":              1.0,
    "ECOTOX-curated; analysis=M": 1.5,   # measured 最高
    "ECOTOX-curated; analysis=Z": 0.7,
    "ECOTOX-curated; analysis=U": 0.5,
    "ECOTOX-curated; analysis=NR": 0.3,
}
DEFAULT_W = 0.8

# 超参
SEED = 42
EPOCHS = 600
LR = 5e-4
WD = 1e-4
DROPOUT = 0.25
HIDDEN = (512, 256, 128)
BATCH = 128
VAL_FRAC = 0.15
PATIENCE = 80
MC_SAMPLES = 30
POOL_PRED_BATCH = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DRY_RUN = False


# =========================================================================== #
# 工具
# =========================================================================== #
def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def load_m1_predictions() -> pd.DataFrame:
    """读模型1预测：parquet 优先，缺失退回 CSV。"""
    if M1_PRED_PARQUET.exists():
        df = pd.read_parquet(M1_PRED_PARQUET)
    elif M1_PRED_CSV.exists():
        df = pd.read_csv(M1_PRED_CSV)
    else:
        raise FileNotFoundError(
            f"找不到模型1预测：{M1_PRED_PARQUET} 或 {M1_PRED_CSV}。先跑 train_model1.py。"
        )
    return df


def merge_transport(joined: pd.DataFrame, m1_pred: pd.DataFrame) -> pd.DataFrame:
    """X_Y_joined.SMILES <- m1_pred.SMILES，左连接。重名 SMILES 取第一行。"""
    smi_col_y = pick_col(joined, ["SMILES", "smiles", "canonical_smiles"])
    if smi_col_y is None:
        raise KeyError("X_Y_joined 没有 SMILES 列")
    keep = ["SMILES"] + [f"pred_{e}" for e in M1_ENDPOINTS] + [f"unc_{e}" for e in M1_ENDPOINTS]
    keep = [c for c in keep if c in m1_pred.columns]
    m1 = m1_pred.drop_duplicates(subset=["SMILES"])[keep]
    out = joined.merge(m1, left_on=smi_col_y, right_on="SMILES", how="left",
                       suffixes=("", "__m1"))
    return out


# --------------------------------------------------------------------------- #
# 生物上下文编码：长尾合并 + one-hot
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 多任务网络（复用 model.py 的 MultiTaskTransportNet）
# --------------------------------------------------------------------------- #
def build_net(in_dim: int) -> MultiTaskTransportNet:
    return MultiTaskTransportNet(
        in_dim=in_dim, endpoints=ENDPOINTS, hidden=HIDDEN, dropout=DROPOUT
    ).to(DEVICE)


# --------------------------------------------------------------------------- #
# 加权 masked MSE：在 model.py 的 masked_multitask_loss 基础上加 sample_weight
# --------------------------------------------------------------------------- #
def weighted_masked_loss(pred, target, mask, sw, ep_w):
    """
    pred/target/mask : [B, K]   sw : [B]   ep_w : [K]
    每个观测点的 loss = sw_i * (pred-target)^2，按 endpoint 求样本和 / 样本数加权和。
    """
    se = (pred - target) ** 2 * mask                   # [B, K]
    se = se * sw.unsqueeze(1)                          # [B, K]，按样本权重
    denom = (mask * sw.unsqueeze(1)).sum(dim=0).clamp(min=1e-6)  # [K]
    per_ep = se.sum(dim=0) / denom                     # [K]
    w = ep_w / ep_w.sum().clamp(min=1e-8)
    return (per_ep * w).sum()


def derive_endpoint_weights(y_long: pd.DataFrame) -> np.ndarray:
    """按 endpoint 样本数 1/sqrt(n) 给权重，少的更重要。"""
    w = np.ones(len(ENDPOINTS), dtype=np.float64)
    for i, ep in enumerate(ENDPOINTS):
        n = (y_long["__ep__"] == ep).sum()
        w[i] = 1.0 / np.sqrt(max(n, 1))
    return np.clip(w, 1e-3, None)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    M2_OUT.mkdir(parents=True, exist_ok=True)

    # ---------- 0. 读数据 ----------
    print("[load] X_Y_joined ...")
    joined = pd.read_excel(XLSX_PATH, sheet_name="X_Y_joined")
    print(f"        shape={joined.shape}")

    # endpoint 标准化
    joined["__ep__"] = joined["endpoint_type"].map(ENDPOINT_FROM_TYPE)
    joined = joined.dropna(subset=["__ep__"])
    # log_value 必须有
    y = pd.to_numeric(joined["log_value"], errors="coerce")
    joined = joined.loc[y.notna()].copy()
    joined["log_value"] = pd.to_numeric(joined["log_value"], errors="coerce")
    print(f"[load] 有效 log_value 样本：{len(joined)}")
    print("       endpoint 分布：", dict(joined["__ep__"].value_counts()))

    # SMILES 必须有（结构特征 + 与模型1 join 都要它）
    smi_col_y = pick_col(joined, ["SMILES", "smiles"])
    joined = joined.dropna(subset=[smi_col_y]).reset_index(drop=True)
    print(f"[load] 有 SMILES 的样本：{len(joined)}")

    # 模型1预测 join
    print("[load] 模型1 predictions ...")
    m1 = load_m1_predictions()
    joined = merge_transport(joined, m1)
    transport_cols = [f"pred_{e}" for e in M1_ENDPOINTS] + [f"unc_{e}" for e in M1_ENDPOINTS]
    transport_cols = [c for c in transport_cols if c in joined.columns]
    cov = joined[transport_cols[0]].notna().sum() if transport_cols else 0
    print(f"[load] transport join 命中 {cov}/{len(joined)}（缺失会用训练集列均值补）")

    # F-stats（保持与模型1一致：从 X_Y_joined 自带的 n_F/n_CF* 喂给 Featurizer）
    fstats_cols = ["SMILES", "n_F", "n_CF1", "n_CF2", "n_CF3", "n_CF4"]
    fstats_df = joined[[c for c in fstats_cols if c in joined.columns]].drop_duplicates(
        subset=[smi_col_y]
    ).rename(columns={smi_col_y: "SMILES"})

    # ---------- 1. 三块特征 ----------
    print("[feat] 结构特征（复用模型1 Featurizer）...")
    feat = Featurizer()
    X_struct, valid = feat.transform(joined[smi_col_y].astype(str).tolist(), fstats_df)
    print(f"       结构维度 = {X_struct.shape[1]}")

    # 解析失败的 SMILES：在训练里丢掉
    if (~valid).any():
        joined = joined.loc[valid].reset_index(drop=True)
        X_struct = X_struct[valid]

    # transport 特征
    if transport_cols:
        X_trans = joined[transport_cols].to_numpy(dtype=np.float32)
    else:
        X_trans = np.zeros((len(joined), 0), dtype=np.float32)

    # 生物上下文
    print("[feat] 生物上下文 one-hot ...")
    bio = BioEncoder(BIO_CAT_COLS, BIO_NUM_COLS, BIO_TOPK).fit(joined)
    X_bio = bio.transform(joined)
    print(f"       生物上下文维度 = {X_bio.shape[1]}")

    # 拼接 + impute（结构里残留 NaN 用列中位数；transport 缺失也用列均值）
    X_all = np.concatenate([X_struct, X_trans, X_bio], axis=1)
    X_all, fill_med = impute_and_record(X_all)

    # 标准化（结构 + transport + 生物数值列；one-hot 列也标准化无害）
    in_mean = X_all.mean(axis=0)
    in_std = X_all.std(axis=0)
    in_std[in_std < 1e-8] = 1.0
    Xn = ((X_all - in_mean) / in_std).astype(np.float32)

    # 拼接列名
    feature_names = (
        feat.feature_names_
        + transport_cols
        + bio.feature_names
    )
    assert len(feature_names) == Xn.shape[1], (len(feature_names), Xn.shape[1])

    # ---------- 2. Y / mask / sample_weight ----------
    K = len(ENDPOINTS)
    N = len(joined)
    y_wide = np.zeros((N, K), dtype=np.float32)
    mask = np.zeros((N, K), dtype=np.float32)
    for j, ep in enumerate(ENDPOINTS):
        sel = (joined["__ep__"] == ep).to_numpy()
        y_wide[sel, j] = joined.loc[sel, "log_value"].to_numpy(dtype=np.float32)
        mask[sel, j] = 1.0

    # 每 endpoint 独立 z-score（更稳）
    y_mean = np.zeros(K, dtype=np.float64)
    y_std = np.ones(K, dtype=np.float64)
    for j in range(K):
        v = y_wide[mask[:, j] > 0, j]
        if v.size:
            y_mean[j] = float(v.mean())
            s = float(v.std()); y_std[j] = s if s > 1e-8 else 1.0
    y_norm = y_wide.copy()
    for j in range(K):
        y_norm[mask[:, j] > 0, j] = (
            (y_wide[mask[:, j] > 0, j] - y_mean[j]) / y_std[j]
        ).astype(np.float32)

    # sample_weight 来自 reliability
    rel = joined["reliability"].astype(str).fillna("UNK")
    sw = rel.map(RELIABILITY_W).fillna(DEFAULT_W).to_numpy(dtype=np.float32)
    print("[weight] reliability → sample_weight 分布：")
    print(rel.map(RELIABILITY_W).fillna(DEFAULT_W).value_counts().to_dict())

    # endpoint 权重（少的更重要）
    ep_w = derive_endpoint_weights(joined)
    print(f"[weight] endpoint loss weight: {dict(zip(ENDPOINTS, np.round(ep_w, 3)))}")

    if DRY_RUN:
        print("[dry-run] 仅检查形状，结束。")
        print(f"  X={Xn.shape}  y={y_norm.shape}  mask sum per ep={mask.sum(0)}")
        return

    # ---------- 3. 训练集/验证集（按 SMILES 分组，避免泄漏） ----------
    rng = np.random.default_rng(SEED)
    smis = joined[smi_col_y].astype(str).to_numpy()
    uniq = np.array(sorted(set(smis.tolist())))
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * VAL_FRAC))
    val_smis = set(uniq[:n_val])
    val_idx = np.array([i for i, s in enumerate(smis) if s in val_smis])
    tr_idx = np.array([i for i in range(N) if i not in set(val_idx.tolist())])
    print(f"[split] 按 SMILES 切分：train={len(tr_idx)}  val={len(val_idx)}（unique SMILES val={n_val}）")

    def make_loader(idx, shuffle):
        ds = TensorDataset(
            torch.tensor(Xn[idx], dtype=torch.float32),
            torch.tensor(y_norm[idx], dtype=torch.float32),
            torch.tensor(mask[idx], dtype=torch.float32),
            torch.tensor(sw[idx], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, drop_last=False)

    tr_loader = make_loader(tr_idx, True)
    va_loader = make_loader(val_idx, False)

    model = build_net(in_dim=Xn.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    ep_w_t = torch.tensor(ep_w, dtype=torch.float32, device=DEVICE)

    def run(loader, train):
        model.train(train)
        tot, nb = 0.0, 0
        for xb, yb, mb, swb in loader:
            xb, yb, mb, swb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE), swb.to(DEVICE)
            pred = model(xb)
            loss = weighted_masked_loss(pred, yb, mb, swb, ep_w_t)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item(); nb += 1
        return tot / max(nb, 1)

    best_val, best_state, bad = float("inf"), None, 0
    for e in range(1, EPOCHS + 1):
        tr = run(tr_loader, True)
        va = run(va_loader, False)
        if va < best_val - 1e-4:
            best_val, best_state, bad = va, {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if e % 20 == 0 or e == 1:
            print(f"  epoch {e:3d}  train={tr:.4f}  val={va:.4f}  best={best_val:.4f}")
        if bad >= PATIENCE:
            print(f"  早停于 epoch {e}（val 连续 {PATIENCE} 轮无改善）")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # 验证集分 endpoint 评估（反标准化空间）
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xn[val_idx], dtype=torch.float32, device=DEVICE)).cpu().numpy()
    print("\n[val] per-endpoint MAE / RMSE / n（反标准化空间）：")
    for j, ep in enumerate(ENDPOINTS):
        sel = mask[val_idx, j] > 0
        if sel.sum() == 0:
            print(f"  {ep:8s} 无验证样本")
            continue
        p = pred[sel, j] * y_std[j] + y_mean[j]
        t = y_wide[val_idx][sel, j]
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        print(f"  {ep:8s} MAE={mae:.3f}  RMSE={rmse:.3f}  n={int(sel.sum())}")

    # ---------- 4. 落盘模型 + 元信息（先存，预测段失败也保住权重） ----------
    torch.save(model.state_dict(), M2_OUT / "model2_state.pt")
    meta = {
        "endpoints": ENDPOINTS,
        "feature_names": feature_names,
        "in_mean": in_mean.tolist(),
        "in_std": in_std.tolist(),
        "fill_median": fill_med.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "bio_vocab": bio.vocab,
        "bio_num_median": bio.num_median,
        "reliability_weights": RELIABILITY_W,
        "endpoint_loss_weights": dict(zip(ENDPOINTS, ep_w.tolist())),
        "m1_endpoints": M1_ENDPOINTS,
        "transport_cols": transport_cols,
        "hyperparams": {"hidden": HIDDEN, "dropout": DROPOUT, "lr": LR, "wd": WD,
                        "epochs": EPOCHS, "batch": BATCH, "mc_samples": MC_SAMPLES},
    }
    with open(M2_OUT / "model2_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[save] 模型+元信息 -> {M2_OUT}")

    # ---------- 5. 对 X_pool_full 全量预测 ----------
    print("\n===== 全量预测 X_pool_full =====")
    pool = pd.read_excel(POOL_XLSX, sheet_name="X_pool_full")
    print(f"[pool] {len(pool)} rows")

    pool_smiles = pool["SMILES"].astype(str).tolist()
    fstats_pool = pool[[c for c in fstats_cols if c in pool.columns]].drop_duplicates(
        subset=["SMILES"]
    )

    Xp_struct, p_valid = feat.transform(pool_smiles, fstats_pool)

    # transport：直接用 m1_predictions
    m1_lut = load_m1_predictions().drop_duplicates(subset=["SMILES"]).set_index("SMILES")
    Xp_trans = np.full((len(pool), len(transport_cols)), np.nan, dtype=np.float32)
    for i, s in enumerate(pool_smiles):
        if s in m1_lut.index:
            for j, c in enumerate(transport_cols):
                v = m1_lut.at[s, c]
                if pd.notna(v):
                    Xp_trans[i, j] = float(v)

    # 生物上下文：池没有这些字段 → 全部填 'NA'/中位数（即"标准条件"）
    pool_bio_df = pd.DataFrame({c: [None] * len(pool) for c in BIO_CAT_COLS + BIO_NUM_COLS})
    Xp_bio = bio.transform(pool_bio_df)

    Xp_all = np.concatenate([Xp_struct, Xp_trans, Xp_bio], axis=1)
    # 用训练集 fill_median 补 NaN，再用训练集 mean/std 标准化
    nan_idx = np.where(np.isnan(Xp_all))
    Xp_all[nan_idx] = np.take(fill_med, nan_idx[1])
    Xp = ((Xp_all - in_mean) / in_std).astype(np.float32)

    Xp_t = torch.tensor(Xp, dtype=torch.float32)
    mean_z, std_z = mc_predict(model, Xp_t,
                               n_samples=MC_SAMPLES, batch_size=POOL_PRED_BATCH, device=DEVICE)

    # 反标准化
    mean_real = mean_z * y_std + y_mean
    std_real = std_z * y_std

    out = pd.DataFrame({"SMILES": pool_smiles, "smiles_valid": p_valid})
    for j, ep in enumerate(ENDPOINTS):
        out[f"pred_{ep}"] = mean_real[:, j]
        out[f"unc_{ep}"] = std_real[:, j]
    bad_rows = ~p_valid
    pred_cols = [c for c in out.columns if c.startswith(("pred_", "unc_"))]
    out.loc[bad_rows, pred_cols] = np.nan

    out_path = M2_OUT / "model2_predictions.parquet"
    try:
        out.to_parquet(out_path, index=False)
        print(f"[save] 预测落盘 -> {out_path}  shape={out.shape}")
    except Exception as e:
        csv_path = M2_OUT / "model2_predictions.csv"
        out.to_csv(csv_path, index=False)
        print(f"[save] parquet 失败（{e!r}），改写 CSV -> {csv_path}  shape={out.shape}")

    print("\n完成。model2_predictions 是模型3的 BCF/BAF 输入。")


if __name__ == "__main__":
    main()
