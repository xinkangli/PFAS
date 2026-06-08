"""
analyze_model3.py — 阶段3（Persistence + Transformation 机制发现）

不是监督学习，是 clustering / graph / SHAP / 风险排序。
4 个机制问题 + TMF zero-shot 验证，每个问题都落一份图 + 一份 CSV。

输入
----
  model_data/Model3_Persistence.xlsx
      ├─ risk_persistence_x_BCF   持久 × BCF 双高散点
      ├─ transformation_edges_90  转化网络
      ├─ TMF_axis_l2_133          实测 TMF
      ├─ compounds_104            化合物属性
      └─ persistence_129          持久性原表
  model_data/model1_out/model1_predictions.csv  阶段1 transport
  model_data/model2_out/model2_predictions.csv  阶段2 BCF/BAF
  model_data/Model1_Transport.xlsx::X_pool_full  127k SMILES（用于 CAS↔SMILES 映射）

输出 → model_data/model3_out/
  Q1_risk_scatter.png + Q1_risk_table.csv
  Q2_transformation_graph.png + Q2_high_risk_edges.csv
  Q3_shap_summary.png + Q3_shap_values.csv   （目标 = 模型1 logKow，作"远距离迁移"代理）
  Q4_priority_list.csv                       综合 risk_score top-N
  TMF_zero_shot_vs_observed.png + TMF_compare.csv

依赖：pandas, numpy, matplotlib, networkx, shap, torch, rdkit
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================================== #
# 路径
# =========================================================================== #
DATA_ROOT = Path("/7t/lxkzero/claude/qd/data/model_data")
M1_OUT = DATA_ROOT / "model1_out"
M2_OUT = DATA_ROOT / "model2_out"
M3_OUT = DATA_ROOT / "model3_out"
XLSX_M1 = DATA_ROOT / "Model1_Transport.xlsx"
XLSX_M3 = DATA_ROOT / "Model3_Persistence.xlsx"

M3_OUT.mkdir(parents=True, exist_ok=True)


# =========================================================================== #
# 通用工具
# =========================================================================== #
def read_pred_csv(p_parquet: Path, p_csv: Path) -> pd.DataFrame:
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    return pd.read_csv(p_csv)


def cas_to_smiles_map(pool: pd.DataFrame) -> dict:
    """X_pool_full 的 CAS → SMILES 字典；CAS 一对多取首个。"""
    sub = pool.dropna(subset=["CAS", "SMILES"]).drop_duplicates(subset=["CAS"])
    return dict(zip(sub["CAS"].astype(str), sub["SMILES"].astype(str)))


# persistence_label → 数值评分（vP 最高）
PERSIST_SCORE = {"vP": 3.0, "P": 2.0, "B": 1.0, "NP": 0.0}
def persist_score(label) -> float:
    if not isinstance(label, str):
        return np.nan
    return PERSIST_SCORE.get(label.strip(), np.nan)


# 解析 "5-30" "2-10" ">90" 这种 molar_yield_pct 字段 → 中位数
def parse_yield(s):
    if not isinstance(s, str):
        try:
            return float(s)
        except Exception:
            return np.nan
    t = s.strip().replace("%", "").replace(">", "").replace("<", "").replace("~", "")
    if "-" in t:
        a, _, b = t.partition("-")
        try:
            return (float(a) + float(b)) / 2.0
        except Exception:
            return np.nan
    try:
        return float(t)
    except Exception:
        return np.nan


# =========================================================================== #
# Q1：persistence × BCF 双高散点
# =========================================================================== #
def q1_persistence_vs_bcf():
    print("\n===== Q1 持久 × 富集 双高风险榜 =====")
    df = pd.read_excel(XLSX_M3, sheet_name="risk_persistence_x_BCF")

    # 持久性数值化（多条记录取最高）
    df["p_score"] = df["persistence_label"].map(persist_score)
    g = df.groupby(["compound_id", "abbreviation", "name"], dropna=False).agg(
        p_score=("p_score", "max"),
        median_log_BCF=("median_log_BCF", "max"),
        median_log_BAF=("median_log_BAF", "max"),
        n_BCF=("n_BCF", "max"),
    ).reset_index()

    # 拿不到 median_log_BCF 的，用 logBAF 兜底；都没有的丢
    g["BCF_or_BAF"] = g["median_log_BCF"].fillna(g["median_log_BAF"])
    sub = g.dropna(subset=["p_score", "BCF_or_BAF"]).copy()
    if sub.empty:
        print("  [warn] 没有同时有 persistence 和 BCF/BAF 的样本，跳过")
        return
    sub["risk"] = sub["p_score"] * 0.5 + sub["BCF_or_BAF"]
    sub = sub.sort_values("risk", ascending=False).reset_index(drop=True)

    out_csv = M3_OUT / "Q1_risk_table.csv"
    sub.to_csv(out_csv, index=False)
    print(f"  [save] {out_csv}  shape={sub.shape}")

    # 散点
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(sub["p_score"], sub["BCF_or_BAF"],
                    c=sub["risk"], cmap="Reds", s=80, edgecolors="k", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="risk = 0.5·persistence + logBCF")
    ax.set_xlabel("persistence score (vP=3, P=2, B=1, NP=0)")
    ax.set_ylabel("median log(BCF or BAF)")
    ax.set_title(f"Q1  Persistence × Bioaccumulation  (n={len(sub)})")
    for _, r in sub.head(8).iterrows():
        ax.annotate(r["abbreviation"], (r["p_score"], r["BCF_or_BAF"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(M3_OUT / "Q1_risk_scatter.png", dpi=140)
    plt.close(fig)
    print(f"  [save] {M3_OUT / 'Q1_risk_scatter.png'}")


# =========================================================================== #
# Q2：转化网络，按终端富集染色
# =========================================================================== #
def q2_transformation_graph(m2_pred: pd.DataFrame, cas2smi: dict, compounds: pd.DataFrame):
    print("\n===== Q2 转化网络：前体→高富集终端 =====")
    edges = pd.read_excel(XLSX_M3, sheet_name="transformation_edges_90")
    edges["yield"] = edges["molar_yield_pct"].apply(parse_yield)

    # 给每个化合物算一个 pred_logBCF：用 compound -> cas -> SMILES -> m2 lookup
    smi_lut = m2_pred.drop_duplicates(subset=["SMILES"]).set_index("SMILES")
    def lookup_bcf(cas):
        smi = cas2smi.get(str(cas))
        if smi and smi in smi_lut.index:
            v = smi_lut.at[smi, "pred_logBCF"]
            return float(v) if pd.notna(v) else np.nan
        return np.nan

    comp = compounds.copy()
    comp["pred_logBCF"] = comp["cas"].map(lookup_bcf)
    comp_idx = comp.set_index("compound_id")

    # 构图：节点= compound_id，边= precursor->terminal，宽度=yield
    G = nx.DiGraph()
    for _, r in edges.iterrows():
        p, t = r["precursor_id"], r["terminal_id"]
        if pd.isna(p) or pd.isna(t):
            continue
        G.add_edge(str(p), str(t), yield_pct=r["yield"], pathway=r["pathway_id"])
    for n in list(G.nodes):
        if n in comp_idx.index:
            G.nodes[n]["abbr"] = comp_idx.loc[n, "abbreviation"]
            G.nodes[n]["pred_logBCF"] = comp_idx.loc[n, "pred_logBCF"]
        else:
            G.nodes[n]["abbr"] = n
            G.nodes[n]["pred_logBCF"] = np.nan

    # 高风险边 = 终端 logBCF × yield，从高到低
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "precursor": G.nodes[u]["abbr"],
            "terminal": G.nodes[v]["abbr"],
            "terminal_pred_logBCF": G.nodes[v]["pred_logBCF"],
            "yield_pct_mid": d["yield_pct"],
            "risk": (G.nodes[v]["pred_logBCF"] or 0) * np.log1p(d["yield_pct"] or 0),
            "pathway_id": d["pathway"],
        })
    he = pd.DataFrame(rows).sort_values("risk", ascending=False).reset_index(drop=True)
    he_csv = M3_OUT / "Q2_high_risk_edges.csv"
    he.to_csv(he_csv, index=False)
    print(f"  [save] {he_csv}  shape={he.shape}")

    # 画图：节点颜色 = pred_logBCF
    fig, ax = plt.subplots(figsize=(12, 9))
    try:
        pos = nx.spring_layout(G, seed=42, k=0.8)
    except Exception:
        pos = nx.shell_layout(G)
    node_colors = [G.nodes[n].get("pred_logBCF", np.nan) for n in G.nodes]
    node_colors = [c if pd.notna(c) else 0.0 for c in node_colors]
    edge_w = [max(0.4, (d.get("yield_pct") or 0) / 20) for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, width=edge_w, alpha=0.5, edge_color="gray",
                           arrows=True, arrowsize=10, ax=ax)
    nc = nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap="YlOrRd",
                                node_size=420, edgecolors="k", linewidths=0.5, ax=ax)
    labels = {n: G.nodes[n]["abbr"] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    plt.colorbar(nc, ax=ax, label="pred_logBCF (model 2)")
    ax.set_title(f"Q2  Transformation pathways colored by terminal logBCF  "
                 f"(nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(M3_OUT / "Q2_transformation_graph.png", dpi=140)
    plt.close(fig)
    print(f"  [save] {M3_OUT / 'Q2_transformation_graph.png'}")
    return G


# =========================================================================== #
# Q3：SHAP 解释结构 → transport（用模型 1）
# =========================================================================== #
def q3_shap_transport():
    """用 Integrated Gradients 做结构 → transport 的归因（替代 SHAP，避免 numba 依赖冲突）。

    IG: 在一条基线 x' → 真实输入 x 的线性插值路径上，把模型输出 f(x) - f(x') 的
        变化按梯度积分分摊到每个特征。结果是每个特征对预测的贡献量，
        平均 |IG| 就是该特征的全局重要性，等价于 SHAP 在线性近似下的解释力。
    """
    print("\n===== Q3 Integrated Gradients：结构 → transport（模型1） =====")
    import torch

    from featurize import Featurizer
    from model import MultiTaskTransportNet

    meta = json.loads((M1_OUT / "model1_meta.json").read_text())
    eps = meta["endpoints"]
    in_mean = np.array(meta["in_mean"], dtype=np.float32)
    in_std = np.array(meta["in_std"], dtype=np.float32)
    fill_med = np.array(meta["fill_median"], dtype=np.float32)
    hidden = tuple(meta["hyperparams"]["hidden"])
    dropout = float(meta["hyperparams"]["dropout"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = MultiTaskTransportNet(in_dim=len(in_mean), endpoints=eps,
                                hidden=hidden, dropout=dropout).to(device)
    net.load_state_dict(torch.load(M1_OUT / "model1_state.pt", map_location=device))
    net.eval()

    # 从池里采样
    pool = pd.read_excel(XLSX_M1, sheet_name="X_pool_full")
    fstats_cols = ["SMILES", "n_F", "n_CF1", "n_CF2", "n_CF3", "n_CF4"]
    fstats = pool[[c for c in fstats_cols if c in pool.columns]].drop_duplicates(subset=["SMILES"])

    take = pool.dropna(subset=["SMILES"]).sample(min(1000, len(pool)), random_state=42)
    smis = take["SMILES"].astype(str).tolist()

    feat = Featurizer()
    X_raw, valid = feat.transform(smis, fstats)
    X_raw = X_raw[valid]
    idx = np.where(np.isnan(X_raw))
    X_raw[idx] = np.take(fill_med, idx[1])
    X = ((X_raw - in_mean) / in_std).astype(np.float32)

    ep_target = "logKow" if "logKow" in eps else eps[0]
    k = eps.index(ep_target)

    # 用 200 个分子做归因，基线 = 训练集均值（标准化后即 0 向量）
    n_ex = min(200, X.shape[0])
    x = torch.tensor(X[:n_ex], dtype=torch.float32, device=device)
    baseline = torch.zeros_like(x)              # 标准化空间的 0 = 训练集均值
    steps = 32                                  # IG 积分步数

    # 在 [baseline, x] 上等步长插值，求每步对目标 head 输出的梯度，最后平均 × (x-baseline)
    grads_sum = torch.zeros_like(x)
    for s in range(1, steps + 1):
        alpha = s / steps
        xi = baseline + alpha * (x - baseline)
        xi.requires_grad_(True)
        out = net(xi)[:, k].sum()              # 标量
        g, = torch.autograd.grad(out, xi, retain_graph=False)
        grads_sum = grads_sum + g.detach()
    ig = (grads_sum / steps) * (x - baseline)  # [n_ex, n_feat]，IG 归因
    ig_np = ig.cpu().numpy()

    feat_names = meta["feature_names"]
    mean_abs = np.mean(np.abs(ig_np), axis=0)
    top = np.argsort(mean_abs)[::-1][:25]
    df_attr = pd.DataFrame({
        "feature": [feat_names[i] for i in top],
        "mean_abs_IG": mean_abs[top],
        "mean_signed_IG": ig_np.mean(axis=0)[top],
    })
    df_attr.to_csv(M3_OUT / "Q3_shap_values.csv", index=False)
    print(f"  [save] {M3_OUT / 'Q3_shap_values.csv'}")

    # 画 top-25 条形（颜色编码符号方向）
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["tab:red" if v > 0 else "tab:blue" for v in df_attr["mean_signed_IG"].values[::-1]]
    ax.barh(range(len(top)), df_attr["mean_abs_IG"].values[::-1], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(df_attr["feature"].values[::-1], fontsize=8)
    ax.set_xlabel(f"mean |IG|  (target: {ep_target}, model 1, n={n_ex})")
    ax.set_title(f"Q3  Top-25 structural drivers of {ep_target}\n红=提升 logKow，蓝=降低")
    fig.tight_layout()
    fig.savefig(M3_OUT / "Q3_shap_summary.png", dpi=140)
    plt.close(fig)
    print(f"  [save] {M3_OUT / 'Q3_shap_summary.png'}")


# =========================================================================== #
# Q4：综合 risk_score → top-N 优先管控名单
# =========================================================================== #
def q4_priority_list(m1_pred, m2_pred, cas2smi, top_n=50):
    print("\n===== Q4 综合 risk_score 优先管控名单 =====")
    # 1) persistence 表（按 compound 取最高 p_score）
    pers = pd.read_excel(XLSX_M3, sheet_name="risk_persistence_x_BCF")
    pers["p_score"] = pers["persistence_label"].map(persist_score)
    pers_g = pers.groupby(["compound_id", "abbreviation", "name", "cas"], dropna=False).agg(
        p_score=("p_score", "max")
    ).reset_index()

    # 2) 是否有 precursor 路径（出现在 transformation_edges_90 的 terminal_id）
    edges = pd.read_excel(XLSX_M3, sheet_name="transformation_edges_90")
    terminals = set(edges["terminal_id"].dropna().astype(str))
    pers_g["has_precursor_path"] = pers_g["compound_id"].astype(str).isin(terminals).astype(int)

    # 3) 模型2 logBCF（按 SMILES）
    smi_lut2 = m2_pred.drop_duplicates(subset=["SMILES"]).set_index("SMILES")
    def lk_bcf(cas):
        s = cas2smi.get(str(cas))
        return float(smi_lut2.at[s, "pred_logBCF"]) if s in smi_lut2.index and pd.notna(smi_lut2.at[s, "pred_logBCF"]) else np.nan
    pers_g["pred_logBCF"] = pers_g["cas"].map(lk_bcf)

    # 4) n_F：从模型1 predictions 找不到，去 X_pool_full
    pool = pd.read_excel(XLSX_M1, sheet_name="X_pool_full")
    if "n_F" in pool.columns:
        cas_to_nf = pool.dropna(subset=["CAS", "n_F"]).drop_duplicates("CAS").set_index("CAS")["n_F"]
        pers_g["n_F"] = pers_g["cas"].map(cas_to_nf)
    else:
        pers_g["n_F"] = np.nan

    # 5) risk_score：归一化加权
    def norm(s):
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() == 0:
            return s.fillna(0.0)
        lo, hi = s.min(skipna=True), s.max(skipna=True)
        if pd.isna(lo) or pd.isna(hi) or hi - lo < 1e-9:
            return s.fillna(0.0) * 0.0
        return ((s - lo) / (hi - lo)).fillna(0.0)

    pers_g["risk_score"] = (
        0.35 * norm(pers_g["p_score"])
        + 0.35 * norm(pers_g["pred_logBCF"])
        + 0.15 * pers_g["has_precursor_path"].astype(float)
        + 0.15 * norm(pers_g["n_F"])
    )

    top = pers_g.sort_values("risk_score", ascending=False).head(top_n).reset_index(drop=True)
    out_csv = M3_OUT / "Q4_priority_list.csv"
    top.to_csv(out_csv, index=False)
    print(f"  [save] {out_csv}  shape={top.shape}")
    return top


# =========================================================================== #
# TMF：模型2 zero-shot vs 实测
# =========================================================================== #
def tmf_zero_shot(m2_pred, cas2smi):
    print("\n===== TMF: 模型2 zero-shot vs 实测 =====")
    tmf = pd.read_excel(XLSX_M3, sheet_name="TMF_axis_l2_133")
    # PFAS 缩写 → CAS：从 compounds_104 拿
    comp = pd.read_excel(XLSX_M3, sheet_name="compounds_104")
    abbr2cas = dict(zip(comp["abbreviation"].astype(str), comp["cas"].astype(str)))
    tmf["cas"] = tmf["PFAS"].astype(str).map(abbr2cas)

    smi_lut = m2_pred.drop_duplicates(subset=["SMILES"]).set_index("SMILES")
    def lookup_bcf_baf(cas):
        smi = cas2smi.get(str(cas))
        if smi and smi in smi_lut.index:
            row = smi_lut.loc[smi]
            return float(row.get("pred_logBCF", np.nan)), float(row.get("pred_logBAF", np.nan))
        return np.nan, np.nan
    tmf[["pred_logBCF", "pred_logBAF"]] = tmf["cas"].apply(
        lambda c: pd.Series(lookup_bcf_baf(c))
    )

    cmp_df = tmf.dropna(subset=["log_TMF", "pred_logBAF"]).copy()
    out_csv = M3_OUT / "TMF_compare.csv"
    cmp_df.to_csv(out_csv, index=False)
    print(f"  [save] {out_csv}  n_compare={len(cmp_df)}")

    if cmp_df.empty:
        print("  [warn] 没有可比对样本，跳过画图")
        return

    fig, ax = plt.subplots(figsize=(6.5, 6))
    if "data_provenance" in cmp_df.columns:
        for grade, color in [("A_SI", "tab:red"), ("B_PDF", "tab:orange"), ("C_REVIEW", "tab:gray")]:
            sub = cmp_df[cmp_df["data_provenance"].astype(str) == grade]
            if len(sub):
                ax.scatter(sub["pred_logBAF"], sub["log_TMF"], label=grade, s=50,
                           edgecolors="k", linewidths=0.4, color=color, alpha=0.85)
        ax.legend(title="provenance")
    else:
        ax.scatter(cmp_df["pred_logBAF"], cmp_df["log_TMF"], s=50, edgecolors="k", linewidths=0.4)
    # 1:1 参考线
    lo = float(min(cmp_df["pred_logBAF"].min(), cmp_df["log_TMF"].min()))
    hi = float(max(cmp_df["pred_logBAF"].max(), cmp_df["log_TMF"].max()))
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    # 简单线性相关
    if len(cmp_df) > 2:
        r = float(np.corrcoef(cmp_df["pred_logBAF"], cmp_df["log_TMF"])[0, 1])
        ax.set_title(f"TMF  pred_logBAF vs measured log_TMF  (Pearson r={r:.2f}, n={len(cmp_df)})")
    else:
        ax.set_title(f"TMF  pred_logBAF vs measured log_TMF  (n={len(cmp_df)})")
    ax.set_xlabel("model 2 pred_logBAF (zero-shot)")
    ax.set_ylabel("measured log_TMF (TMF_axis_l2_133)")
    fig.tight_layout()
    fig.savefig(M3_OUT / "TMF_zero_shot_vs_observed.png", dpi=140)
    plt.close(fig)
    print(f"  [save] {M3_OUT / 'TMF_zero_shot_vs_observed.png'}")


# =========================================================================== #
# main
# =========================================================================== #
def main():
    pool = pd.read_excel(XLSX_M1, sheet_name="X_pool_full")
    cas2smi = cas_to_smiles_map(pool)
    print(f"[load] CAS↔SMILES 字典  size={len(cas2smi)}")

    m1_pred = read_pred_csv(M1_OUT / "model1_predictions.parquet",
                            M1_OUT / "model1_predictions.csv")
    m2_pred = read_pred_csv(M2_OUT / "model2_predictions.parquet",
                            M2_OUT / "model2_predictions.csv")
    print(f"[load] m1_pred {m1_pred.shape}  m2_pred {m2_pred.shape}")

    compounds = pd.read_excel(XLSX_M3, sheet_name="compounds_104")

    q1_persistence_vs_bcf()
    q2_transformation_graph(m2_pred, cas2smi, compounds)
    q3_shap_transport()
    q4_priority_list(m1_pred, m2_pred, cas2smi, top_n=50)
    tmf_zero_shot(m2_pred, cas2smi)

    print("\n全部完成。输出位于：", M3_OUT)


if __name__ == "__main__":
    main()
