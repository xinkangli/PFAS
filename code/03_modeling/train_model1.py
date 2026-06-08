"""
train_model1.py — 模型1（Transport Foundation Model）主流程

数据流（对应文档阶段1的 Step 1-4）：
    Model1_Transport.xlsx
        ├─ X_pool_full      127,993 SMILES 池        → 全量预测目标
        ├─ F_stats          1,270 SMILES n_F/n_CF*    → join 进特征
        ├─ Y_silver_logKow  68k 弱标签                → Step1 预训练 logKow head
        ├─ Y_gold_long      25 PFAS × 8 endpoint=200  → Step2 fine-tune 全部 head
        └─ missing_priority 各 endpoint 优先级         → Step3 多任务损失权重
    →（Step4）对 X_pool_full 全量预测 8 个 endpoint + MC-dropout 不确定性
    → 落盘 model1_predictions.parquet，即模型2的 transport 输入（按 SMILES join）

用法：
    python train_model1.py
只想看数据形状不训练：把 DRY_RUN=True

依赖：torch, rdkit, pandas, numpy, pyarrow(写 parquet)
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
# 配置 —— Sheet 名按你 README 实际命名核对后修改
# =========================================================================== #
DATA_ROOT = Path("/7t/lxkzero/claude/qd/data/model_data")
XLSX_PATH = DATA_ROOT / "Model1_Transport.xlsx"
OUT_DIR = DATA_ROOT / "model1_out"

SHEETS = {
    "pool":     "X_pool_full",
    "fstats":   "F_stats_dict",
    "silver":   "Y_silver_logKow",
    "gold":     "Y_gold_long",
    "priority": "missing_priority",
    "target25": "target_25_X",    # 提供 CAS→SMILES，给 Y_gold_long 补 SMILES
}

# 与 Y_gold_long 实际可用 endpoint 对齐：logKd 在 long 表里 value 全空，
# logKoa/half_life 在 Model1 没金标签 —— 因此 7 个 endpoint
ENDPOINTS = [
    "logKow", "logKoc", "water_solubility",
    "Henry", "vapor_pressure", "pKa", "logD",
]

# 金标签 sheet 里 endpoint 写法可能不统一，归一化到标准名
ENDPOINT_ALIASES = {
    "logkow": "logKow", "log_kow": "logKow", "logp": "logKow", "kow": "logKow",
    "logkoc": "logKoc", "log_koc": "logKoc", "koc": "logKoc",
    "logkd": "logKd", "log_kd": "logKd", "kd": "logKd",
    "pka": "pKa",
    "water_solubility": "water_solubility", "solubility": "water_solubility",
    "logs": "water_solubility", "ws": "water_solubility",
    "henry": "Henry", "logh": "Henry", "kh": "Henry",
    "henry_constant": "Henry", "henry's_law_constant": "Henry",
    "vapor_pressure": "vapor_pressure", "vp": "vapor_pressure", "logvp": "vapor_pressure",
    "logd": "logD", "log_d": "logD",
}

# 超参
SEED = 42
PRETRAIN_EPOCHS = 40
PRETRAIN_LR = 1e-3
PRETRAIN_BATCH = 512
FINETUNE_EPOCHS = 400
FINETUNE_LR = 3e-4          # 低 LR 全网微调（25 个金样本，防过拟合）
FINETUNE_WD = 1e-3          # 较强 weight decay
FREEZE_BACKBONE = False     # True=只微调 head；与低 LR 可二选一/叠加
DROPOUT = 0.3
HIDDEN = (512, 256, 128)
MC_SAMPLES = 30             # 预测时 MC-dropout 前向次数
VAL_FRAC = 0.2              # 金标签上的小验证集（25 样本本就脆弱，仅供早停）
PATIENCE = 60
POOL_PRED_BATCH = 4096
DRY_RUN = False            # True 时只打印数据形状、不训练

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================================== #
# 工具
# =========================================================================== #
def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None


def norm_endpoint(s) -> str | None:
    if not isinstance(s, str):
        return None
    return ENDPOINT_ALIASES.get(s.strip().lower())


class TargetScaler:
    """各 endpoint 独立标准化，忽略 NaN。"""

    def __init__(self, endpoints: list[str]):
        self.endpoints = endpoints
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}

    def fit(self, observed: dict[str, np.ndarray]):
        for ep in self.endpoints:
            v = observed.get(ep, np.array([]))
            v = v[~np.isnan(v)] if v.size else v
            if v.size == 0:
                self.mean[ep], self.std[ep] = 0.0, 1.0
            else:
                self.mean[ep] = float(np.mean(v))
                s = float(np.std(v))
                self.std[ep] = s if s > 1e-8 else 1.0
        return self

    def transform_wide(self, wide: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """wide[N,K] -> (标准化 target, mask)，NaN 处 mask=0、target=0。"""
        N, K = len(wide), len(self.endpoints)
        tgt = np.zeros((N, K), dtype=np.float32)
        mask = np.zeros((N, K), dtype=np.float32)
        for j, ep in enumerate(self.endpoints):
            col = wide[ep].to_numpy(dtype=np.float64)
            obs = ~np.isnan(col)
            mask[:, j] = obs.astype(np.float32)
            tgt[obs, j] = ((col[obs] - self.mean[ep]) / self.std[ep]).astype(np.float32)
        return tgt, mask

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        out = arr.astype(np.float64).copy()
        for j, ep in enumerate(self.endpoints):
            out[:, j] = out[:, j] * self.std[ep] + self.mean[ep]
        return out

    def inverse_std(self, std_arr: np.ndarray) -> np.ndarray:
        """不确定性 std 反标准化：只乘尺度，不加均值。"""
        out = std_arr.astype(np.float64).copy()
        for j, ep in enumerate(self.endpoints):
            out[:, j] = out[:, j] * self.std[ep]
        return out


# =========================================================================== #
# 数据加载
# =========================================================================== #
def load_sheets() -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(XLSX_PATH)
    print(f"[load] sheets in {XLSX_PATH.name}: {xl.sheet_names}")
    out = {}
    for key, name in SHEETS.items():
        if name in xl.sheet_names:
            out[key] = xl.parse(name)
            print(f"[load] {key:9s} <- '{name}'  shape={out[key].shape}")
        else:
            print(f"[load] !! sheet '{name}' 不存在，{key} 置空（请核对 SHEETS 配置）")
            out[key] = pd.DataFrame()
    return out


def get_smiles_series(df: pd.DataFrame) -> pd.Series:
    col = pick_col(df, ["SMILES", "smiles", "canonical_smiles", "smi"])
    if col is None:
        raise KeyError(f"找不到 SMILES 列，现有列：{list(df.columns)}")
    return df[col].astype(str)


def load_gold_wide(
    df_long: pd.DataFrame,
    endpoints: list[str],
    smiles_lookup: dict | None = None,
):
    """金标签 long → wide。返回 (smiles_list, wide_df[endpoints])。

    smiles_lookup: 当 long 表自己没有 SMILES 列时（例如 Y_gold_long 只有 CAS/InChIKey），
                   传一个 {CAS: SMILES} 或 {InChIKey: SMILES} 字典，按 id_col join 进来。
    """
    id_col = pick_col(df_long, ["cas", "casrn", "cas_rn", "cas_number"])
    smi_col = pick_col(df_long, ["SMILES", "smiles", "canonical_smiles"])
    ep_col = pick_col(df_long, ["endpoint", "endpoint_type", "property", "prop", "param"])
    val_col = pick_col(df_long, ["value", "val", "y", "measurement", "result"])
    if None in (ep_col, val_col):
        raise KeyError(f"金标签缺列。需要 endpoint/value，现有：{list(df_long.columns)}")
    if smi_col is None and smiles_lookup is None:
        raise KeyError(
            f"金标签没有 SMILES 列也没有 smiles_lookup。现有：{list(df_long.columns)}"
        )
    if id_col is None:
        # 没 CAS 也没 SMILES 列：兜底用 InChIKey
        id_col = pick_col(df_long, ["inchikey", "inchi_key"]) or smi_col

    df = df_long.copy()
    df["__ep__"] = df[ep_col].map(norm_endpoint)
    df = df.dropna(subset=["__ep__", val_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])

    wide = df.pivot_table(index=id_col, columns="__ep__", values=val_col, aggfunc="mean")

    if smi_col is not None:
        smi = df.dropna(subset=[smi_col]).groupby(id_col)[smi_col].first()
    else:
        # 用外部 lookup（CAS/InChIKey → SMILES）
        smi = pd.Series({k: smiles_lookup.get(k) for k in wide.index})
    wide = wide.join(smi.rename("__smiles__"))
    wide = wide.dropna(subset=["__smiles__"])

    for ep in endpoints:
        if ep not in wide.columns:
            wide[ep] = np.nan
    return wide["__smiles__"].astype(str).tolist(), wide[endpoints].reset_index(drop=True)


def derive_loss_weights(priority_df: pd.DataFrame, endpoints: list[str]) -> np.ndarray:
    """
    从 missing_priority 推多任务损失权重（Step3）。优先级别：
      1) 有 weight / loss_weight 列  -> 直接用
      2) 有 priority 列（数字，越大越重要）-> 归一化用
      3) 有 n_available / n_obs 列   -> 用 1/sqrt(n) 提升稀缺 endpoint 权重
      4) 有 missing_endpoints 文本列 -> 统计每个 endpoint 出现次数，缺失多则权重高
      5) 都没有                       -> 全 1
    """
    w = np.ones(len(endpoints), dtype=np.float64)
    if priority_df is None or priority_df.empty:
        return w
    ep_col = pick_col(priority_df, ["endpoint", "endpoint_type", "property", "name"])
    miss_col = pick_col(priority_df, ["missing_endpoints", "missing"])
    if ep_col is None and miss_col is None:
        return w

    if ep_col is not None:
        pdf = priority_df.copy()
        pdf["__ep__"] = pdf[ep_col].map(norm_endpoint)
        pdf = pdf.dropna(subset=["__ep__"]).set_index("__ep__")

        wcol = pick_col(pdf.reset_index(), ["weight", "loss_weight"])
        pcol = pick_col(pdf.reset_index(), ["priority", "importance"])
        ncol = pick_col(pdf.reset_index(), ["n_available", "n_obs", "n", "count"])

        for i, ep in enumerate(endpoints):
            if ep not in pdf.index:
                continue
            row = pdf.loc[ep]
            if wcol and pd.notna(row.get(wcol)):
                w[i] = float(row[wcol])
            elif pcol and pd.notna(row.get(pcol)):
                w[i] = float(row[pcol])
            elif ncol and pd.notna(row.get(ncol)) and float(row[ncol]) > 0:
                w[i] = 1.0 / np.sqrt(float(row[ncol]))
        return np.clip(w, 1e-3, None)

    # per-PFAS missing_endpoints（'logKd|logD' 这种 |分隔文本）→ 缺失越多权重越高
    miss_counts = {ep: 0 for ep in endpoints}
    for s in priority_df[miss_col].dropna().astype(str):
        for tok in s.replace(",", "|").split("|"):
            ep = norm_endpoint(tok)
            if ep in miss_counts:
                miss_counts[ep] += 1
    n_pfas = max(len(priority_df), 1)
    for i, ep in enumerate(endpoints):
        # 1 + 缺失率：全填 -> 权重 1；全缺 -> 权重 2
        w[i] = 1.0 + miss_counts[ep] / n_pfas
    return np.clip(w, 1e-3, None)


# =========================================================================== #
# 训练循环
# =========================================================================== #
def run_epoch(model, loader, weights_t, optim=None):
    train = optim is not None
    model.train(train)
    total, nb = 0.0, 0
    for xb, tb, mb in loader:
        xb, tb, mb = xb.to(DEVICE), tb.to(DEVICE), mb.to(DEVICE)
        pred = model(xb)
        loss = masked_multitask_loss(pred, tb, mb, weights_t)
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        total += loss.item()
        nb += 1
    return total / max(nb, 1)


def make_loader(X, tgt, mask, batch, shuffle):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(tgt, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=False)


def build_model(in_dim: int) -> MultiTaskTransportNet:
    return MultiTaskTransportNet(
        in_dim=in_dim, endpoints=ENDPOINTS, hidden=HIDDEN, dropout=DROPOUT
    ).to(DEVICE)


# =========================================================================== #
# 主流程
# =========================================================================== #
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 0. 读数据 ----------
    sh = load_sheets()
    fstats_df = sh["fstats"] if not sh["fstats"].empty else None
    if fstats_df is not None and "SMILES" not in fstats_df.columns:
        # 把 F_stats 的 smiles 列统一成 'SMILES'
        c = pick_col(fstats_df, ["SMILES", "smiles", "canonical_smiles"])
        if c:
            fstats_df = fstats_df.rename(columns={c: "SMILES"})

    # silver: SMILES + logKow
    sv_smiles = get_smiles_series(sh["silver"]).tolist()
    sv_col = pick_col(sh["silver"], ["logKow_Norman", "logKow", "log_kow", "logp", "xlogp", "value"])
    sv_y = pd.to_numeric(sh["silver"][sv_col], errors="coerce").to_numpy(dtype=np.float64)

    # gold: long -> wide。Y_gold_long 没 SMILES 列，需用 target_25_X 的 CAS→SMILES 补
    smi_lookup = None
    if not sh.get("target25", pd.DataFrame()).empty:
        t25 = sh["target25"]
        smi_c = pick_col(t25, ["SMILES", "smiles", "canonical_smiles"])
        cas_c = pick_col(t25, ["CAS", "cas", "casrn"])
        if smi_c and cas_c:
            smi_lookup = dict(
                t25.dropna(subset=[cas_c, smi_c])[[cas_c, smi_c]].astype(str).values
            )
    gold_smiles, gold_wide = load_gold_wide(sh["gold"], ENDPOINTS, smiles_lookup=smi_lookup)

    # pool: 全量预测目标
    pool_smiles = get_smiles_series(sh["pool"]).tolist()

    print(f"\n[data] silver={len(sv_smiles)}  gold={len(gold_smiles)}  pool={len(pool_smiles)}")
    for ep in ENDPOINTS:
        print(f"       gold {ep:16s} obs={int(gold_wide[ep].notna().sum())}")

    # ---------- 1. 特征工程 ----------
    feat = Featurizer()
    print(f"[feat] 特征维度 = {len(feat.feature_names_)}")

    Xs_raw, sv_valid = feat.transform(sv_smiles, fstats_df)
    Xg_raw, g_valid = feat.transform(gold_smiles, fstats_df)

    # 过滤无效 SMILES
    Xs_raw, sv_y, sv_smiles = Xs_raw[sv_valid], sv_y[sv_valid], list(np.array(sv_smiles)[sv_valid])
    Xg_raw = Xg_raw[g_valid]
    gold_wide = gold_wide.loc[g_valid].reset_index(drop=True)
    gold_smiles = list(np.array(gold_smiles)[g_valid])
    # silver 还要去掉 logKow 本身为 NaN 的
    sv_ok = ~np.isnan(sv_y)
    Xs_raw, sv_y, sv_smiles = Xs_raw[sv_ok], sv_y[sv_ok], list(np.array(sv_smiles)[sv_ok])

    print(f"[feat] 有效 silver={len(sv_smiles)}  有效 gold={len(gold_smiles)}")

    if DRY_RUN:
        print("[dry-run] 仅检查数据形状，结束。")
        return

    # ---------- 输入标准化（用 silver+gold 拟合，再套用到 pool）----------
    train_struct = np.vstack([Xs_raw, Xg_raw])
    train_struct, fill_med = impute_and_record(train_struct)   # 残留 NaN -> 列中位数
    in_mean = train_struct.mean(axis=0)
    in_std = train_struct.std(axis=0)
    in_std[in_std < 1e-8] = 1.0

    def scale_X(X_raw):
        idx = np.where(np.isnan(X_raw))           # 用训练集中位数补 pool 的 NaN
        X = X_raw.copy()
        X[idx] = np.take(fill_med, idx[1])
        return ((X - in_mean) / in_std).astype(np.float32)

    Xs = scale_X(Xs_raw)
    Xg = scale_X(Xg_raw)

    # ---------- 目标标准化 ----------
    observed = {ep: gold_wide[ep].to_numpy(dtype=np.float64) for ep in ENDPOINTS}
    observed["logKow"] = np.concatenate([observed["logKow"], sv_y])  # logKow 把 silver 也算进尺度
    tscaler = TargetScaler(ENDPOINTS).fit(observed)

    # silver 的 wide：只有 logKow 一列有值
    sv_wide = pd.DataFrame({ep: np.full(len(sv_y), np.nan) for ep in ENDPOINTS})
    sv_wide["logKow"] = sv_y
    sv_tgt, sv_mask = tscaler.transform_wide(sv_wide)
    g_tgt, g_mask = tscaler.transform_wide(gold_wide)

    # ---------- 损失权重（Step3）----------
    base_w = derive_loss_weights(sh["priority"], ENDPOINTS)
    print(f"[weight] missing_priority -> {dict(zip(ENDPOINTS, np.round(base_w, 3)))}")

    model = build_model(in_dim=Xs.shape[1])

    # ========== Step1：弱标签预训练 logKow head ==========
    print("\n===== Step1 预训练（68k 弱标签，只学 logKow）=====")
    pre_w = torch.zeros(len(ENDPOINTS), device=DEVICE)
    pre_w[ENDPOINTS.index("logKow")] = 1.0
    pre_loader = make_loader(Xs, sv_tgt, sv_mask, PRETRAIN_BATCH, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)
    for ep in range(1, PRETRAIN_EPOCHS + 1):
        loss = run_epoch(model, pre_loader, pre_w, opt)
        if ep % 5 == 0 or ep == 1:
            print(f"  pretrain epoch {ep:3d}  loss={loss:.4f}")

    # ========== Step2：金标签 fine-tune 全部 head ==========
    print("\n===== Step2 fine-tune（25 PFAS × 8 endpoint）=====")
    if FREEZE_BACKBONE:
        model.set_backbone_trainable(False)
        print("  backbone 已冻结，只微调 head")

    # 金标签上切小验证集（按分子，避免泄漏）
    n_g = len(Xg)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_g)
    n_val = max(1, int(n_g * VAL_FRAC))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    ft_w = torch.tensor(base_w, dtype=torch.float32, device=DEVICE)

    tr_loader = make_loader(Xg[tr_idx], g_tgt[tr_idx], g_mask[tr_idx],
                            batch=max(4, len(tr_idx)), shuffle=True)
    va_loader = make_loader(Xg[val_idx], g_tgt[val_idx], g_mask[val_idx],
                            batch=len(val_idx), shuffle=False)

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR, weight_decay=FINETUNE_WD,
    )
    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(1, FINETUNE_EPOCHS + 1):
        tr = run_epoch(model, tr_loader, ft_w, opt)
        va = run_epoch(model, va_loader, ft_w, None)
        if va < best_val - 1e-4:
            best_val, best_state, bad = va, {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 20 == 0 or ep == 1:
            print(f"  finetune epoch {ep:3d}  train={tr:.4f}  val={va:.4f}  best={best_val:.4f}")
        if bad >= PATIENCE:
            print(f"  早停于 epoch {ep}（val 连续 {PATIENCE} 轮无改善）")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- 先把模型 + 元信息存盘（万一预测/落盘失败，模型权重仍在）----------
    torch.save(model.state_dict(), OUT_DIR / "model1_state.pt")
    meta = {
        "endpoints": ENDPOINTS,
        "feature_names": feat.feature_names_,
        "in_mean": in_mean.tolist(),
        "in_std": in_std.tolist(),
        "fill_median": fill_med.tolist(),
        "target_mean": tscaler.mean,
        "target_std": tscaler.std,
        "loss_weights": dict(zip(ENDPOINTS, base_w.tolist())),
        "hyperparams": {
            "hidden": HIDDEN, "dropout": DROPOUT, "mc_samples": MC_SAMPLES,
            "pretrain_epochs": PRETRAIN_EPOCHS, "finetune_epochs": FINETUNE_EPOCHS,
        },
    }
    with open(OUT_DIR / "model1_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[save] 模型+元信息 -> {OUT_DIR}")

    # ========== Step4：X_pool_full 全量预测 + 不确定性 ==========
    print("\n===== Step4 全量预测 X_pool_full =====")
    Xp_raw, p_valid = feat.transform(pool_smiles, fstats_df)
    Xp = scale_X(Xp_raw)
    Xp_t = torch.tensor(Xp, dtype=torch.float32)

    mean_std_space, std_std_space = mc_predict(
        model, Xp_t, n_samples=MC_SAMPLES, batch_size=POOL_PRED_BATCH, device=DEVICE
    )
    mean = tscaler.inverse(mean_std_space)        # 反标准化回真实单位
    unc = tscaler.inverse_std(std_std_space)      # 不确定性（真实单位 std）

    out = pd.DataFrame({"SMILES": pool_smiles, "smiles_valid": p_valid})
    for j, ep in enumerate(ENDPOINTS):
        out[f"pred_{ep}"] = mean[:, j]
        out[f"unc_{ep}"] = unc[:, j]
    # SMILES 解析失败的行预测无意义，置 NaN（保留行以便上游对齐）
    bad_rows = ~p_valid
    pred_cols = [c for c in out.columns if c.startswith(("pred_", "unc_"))]
    out.loc[bad_rows, pred_cols] = np.nan

    pred_path = OUT_DIR / "model1_predictions.parquet"
    try:
        out.to_parquet(pred_path, index=False)
        print(f"[save] 预测落盘 -> {pred_path}  shape={out.shape}")
    except Exception as e:
        csv_path = OUT_DIR / "model1_predictions.csv"
        out.to_csv(csv_path, index=False)
        print(f"[save] parquet 失败（{e!r}），改写 CSV -> {csv_path}  shape={out.shape}")

    print("\n完成。model1_predictions 即模型2的 transport 输入（按 SMILES join）。")


if __name__ == "__main__":
    main()