#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TMF Literature Priority Ranker  v2
====================================
v2 新增:
  1) DOI 去重  — 同一篇只保留 priority_score 最高的一条
  2) 资格审查  — 4 个字段自动判定文章是否可用于 TMF 数据提取:
       is_pfas_specific         是否 PFAS 专门文章
       has_trophic_level        是否含营养级信息
       has_concentration_matrix 是否有物种-浓度矩阵
       keep_for_extraction      三项全 yes 才为 yes
  3) 最终 Top N = N 篇 unique papers

用法:
    python3 tmf_literature_ranker.py \
        --input pfas_data/raw_refs.csv \
        --meta  pfas_data/meta_dois.txt \
        --top   100 \
        --out   pfas_ranked/
"""

from __future__ import annotations

import re
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ranker")


# ─── 评分规则 ──────────────────────────────────────────────────────────────

KEYWORD_RULES = [
    (re.compile(r"\b(trophic magnif|trophic level|trophic transfer|tmf\b|bmf\b)", re.I), 4, "trophic_magnif"),
    (re.compile(r"\b(biomagni|bioaccumul|biomagnif)", re.I),                              4, "biomagnif"),
    (re.compile(r"\b(food web|food chain|food-web|foodweb)", re.I),                       3, "food_web"),
    (re.compile(r"\bmultiple species\b|\bseveral species\b|\bmulti-species\b", re.I),     3, "multi_species"),
    (re.compile(r"\b(pfos|pfoa|pfhxs|pfna|pfda|pfas)\b", re.I),                          3, "pfas_compound"),
    (re.compile(r"\b(precursor|biotransformation|transformation product)", re.I),         5, "precursor"),
    (re.compile(r"\b(arctic|antarctic|polar|svalbard|barents|weddell)", re.I),            3, "polar"),
    (re.compile(r"\b(seal|polar bear|orca|killer whale|beluga|penguin|walrus|narwhal)", re.I), 2, "apex_species"),
    (re.compile(r"trophic level[s]?\s*[=:]\s*\d+\.?\d*", re.I),                          2, "tl_numeric"),
]

JOURNAL_RULES = [
    (re.compile(r"environ(mental)?\s+sci(ence)?\s*(technol|&\s*tech)", re.I), 5, "journal_est"),
    (re.compile(r"es&t|es\&t letters",                                  re.I), 5, "journal_est_letters"),
    (re.compile(r"water research",                                       re.I), 4, "journal_wr"),
    (re.compile(r"environment international",                            re.I), 4, "journal_ei"),
    (re.compile(r"science of the total environ",                        re.I), 3, "journal_stoten"),
    (re.compile(r"environmental pollution",                              re.I), 3, "journal_envpol"),
    (re.compile(r"chemosphere",                                          re.I), 3, "journal_chemosphere"),
    (re.compile(r"environmental science.*letters",                       re.I), 4, "journal_es_letters"),
]

# ─── 资格审查规则 ──────────────────────────────────────────────────────────

PFAS_POSITIVE = re.compile(
    r"\b(pfas|pfos|pfoa|pfhxs|pfna|pfda|pfca|pfba|pfbs|pfsa|pfua|"
    r"perfluoro|polyfluoro|fluorotelomer|ftoh|fts|genx|hfpo|adona|"
    r"per.?and.?poly.?fluoroalkyl)\b",
    re.I,
)
PFAS_NEGATIVE = re.compile(
    r"\b(pharmaceut|antibiotic|pesticide|herbicide|heavy metal|"
    r"microplastic|polycyclic aromatic|pah\b|pcb\b|dioxin|"
    r"organochlorine|organophosphate|phthalate|bisphenol|"
    r"stimulant|illicit drug|cocaine|caffeine|ibuprofen|naproxen|"
    r"triclosan|carbamazepine)\b",
    re.I,
)
TROPHIC_POSITIVE = re.compile(
    r"\b(trophic level|trophic magnif|tmf\b|bmf\b|"
    r"delta.?15n|stable isotope|nitrogen isotope|"
    r"food web|food chain|trophic transfer|trophic position)\b",
    re.I,
)
CONC_MATRIX_POSITIVE = re.compile(
    r"\b(concentration|ng.?g|ng.?l|ng.?kg|ug.?g|"
    r"tissue concentration|blood|liver|muscle|serum|"
    r"whole body|multiple species|several species|"
    r"fish|seal|bird|mammal|invertebrate|plankton|zooplankton)\b",
    re.I,
)


def qualify_row(text: str) -> dict:
    pfas_hits   = len(PFAS_POSITIVE.findall(text))
    has_negword = bool(PFAS_NEGATIVE.search(text))

    if pfas_hits == 0:
        is_pfas  = "no"
        pfas_rsn = "no PFAS keyword"
    elif has_negword and pfas_hits < 2:
        is_pfas  = "no"
        pfas_rsn = "dominated by non-PFAS pollutant"
    else:
        is_pfas  = "yes"
        pfas_rsn = ""

    if TROPHIC_POSITIVE.search(text):
        has_tl   = "yes"; tl_rsn = ""
    else:
        has_tl   = "no";  tl_rsn = "no trophic level / food web keyword"

    if CONC_MATRIX_POSITIVE.search(text):
        has_cm   = "yes"; cm_rsn = ""
    else:
        has_cm   = "no";  cm_rsn = "no species-concentration keyword"

    keep    = "yes" if (is_pfas == "yes" and has_tl == "yes" and has_cm == "yes") else "no"
    reasons = [r for r in [pfas_rsn, tl_rsn, cm_rsn] if r]

    return {
        "is_pfas_specific":         is_pfas,
        "has_trophic_level":        has_tl,
        "has_concentration_matrix": has_cm,
        "keep_for_extraction":      keep,
        "exclude_reason":           "; ".join(reasons),
    }


# ─── 评分 ──────────────────────────────────────────────────────────────────

def score_row(row: pd.Series, meta_dois: set) -> tuple:
    tags  = []
    total = 0
    title    = str(row.get("title",    "") or "").lower()
    abstract = str(row.get("abstract", "") or "").lower()
    journal  = str(row.get("journal",  "") or "").lower()
    doi      = str(row.get("doi",      "") or "").strip().lower()
    text     = title + " " + abstract

    for pat, pts, label in KEYWORD_RULES:
        if pat.search(text):
            total += pts; tags.append(label)

    for pat, pts, label in JOURNAL_RULES:
        if pat.search(journal):
            total += pts; tags.append(label); break

    suppl = str(row.get("supplement_link", "") or "").lower()
    has_xlsx = bool(suppl) and any(
        x in suppl for x in [".xlsx", ".xls", ".csv", ".zip", "supplementary"])
    if not has_xlsx and re.search(r"supplement(ary|al)?\s+(table|data|excel|file)", text):
        has_xlsx = True
    if has_xlsx:
        total += 10; tags.append("has_supplement_xlsx")

    oa = str(row.get("open_access", "") or "").lower()
    if oa in ("true", "yes", "1", "oa", "open"):
        total += 3; tags.append("open_access")

    if doi and doi != "nan" and doi.startswith("10."):
        total += 2; tags.append("has_doi")

    try:
        cit = int(float(row.get("citations", 0) or 0))
    except (ValueError, TypeError):
        cit = 0
    if cit >= 100:
        total += 3; tags.append("high_cited_100+")
    elif cit >= 50:
        total += 2; tags.append("high_cited_50+")
    elif cit >= 20:
        total += 1; tags.append("high_cited_20+")

    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if doi_clean and doi_clean in meta_dois:
        total += 8; tags.append("FROM_META_ANALYSIS")

    return total, tags


# ─── 列名规范化 ────────────────────────────────────────────────────────────

COLUMN_MAP = {
    "title":           ["title", "paper_title", "article_title", "name"],
    "abstract":        ["abstract", "summary", "abstr"],
    "journal":         ["journal", "journal_name", "source", "publication"],
    "doi":             ["doi", "doi_url"],
    "citations":       ["citations", "cited_by", "citation_count", "times_cited", "tc"],
    "supplement_link": ["supplement_link", "supplementary_link", "suppl", "si_link"],
    "open_access":     ["open_access", "oa", "is_oa", "access_type"],
    "year":            ["year", "pub_year", "publication_year", "py"],
    "authors":         ["authors", "author"],
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for std, aliases in COLUMN_MAP.items():
        if std in df.columns:
            continue
        for alias in aliases:
            if alias.lower() in cols_lower:
                rename[cols_lower[alias.lower()]] = std; break
    if rename:
        log.info(f"  列名映射: {rename}")
        df = df.rename(columns=rename)
    for col in COLUMN_MAP:
        if col not in df.columns:
            df[col] = ""
    return df


# ─── 读取 ──────────────────────────────────────────────────────────────────

def load_refs(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    log.info(f"  读取: {path.name}")
    if suf in (".xlsx", ".xls"):
        xf = pd.ExcelFile(path)
        dfs = []
        for sh in xf.sheet_names:
            try:
                d = pd.read_excel(path, sheet_name=sh)
                if len(d.columns) >= 2 and len(d) > 0:
                    d["_sheet"] = sh; dfs.append(d)
            except Exception:
                pass
        if not dfs:
            raise ValueError("Excel 无有效 sheet")
        return pd.concat(dfs, ignore_index=True)
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码: {path}")


# ─── DEMO ──────────────────────────────────────────────────────────────────

DEMO_DATA = [
    {"title":"Trophic magnification of PFAS in Arctic polar food web",
     "abstract":"PFOS/PFOA concentrations in plankton, fish, seals, polar bear Svalbard. TMF=4.5 via δ15N trophic level. Supplementary Excel with all food web concentrations.",
     "journal":"Environmental Science & Technology","doi":"10.1021/es1234","citations":210,"open_access":"yes","year":2019},
    {"title":"PFAS trophic transfer Great Lakes food web multiple species",
     "abstract":"Mysids, alewife, chinook, lake trout. PFOS PFOA PFHxS tissue concentrations ng/g. TMF by log concentration vs trophic level regression.",
     "journal":"Environmental Science & Technology","doi":"10.1021/es5678","citations":145,"open_access":"no","year":2018},
    {"title":"Precursor PFAS biotransformation sewage-impacted river food chain",
     "abstract":"6:2 FTOH biotransformation to PFOA in perch liver ng/g. Multiple species. Food web trophic magnification factor.",
     "journal":"Water Research","doi":"10.1016/wr9012","citations":88,"open_access":"yes","year":2021},
    {"title":"Global meta-analysis PFAS biomagnification factors ecosystems",
     "abstract":"230 food web studies. TMF for 18 PFAS. Arctic freshwater terrestrial. Species concentrations supplementary file. Trophic level nitrogen isotope.",
     "journal":"Environment International","doi":"10.1016/ei3456","citations":312,"open_access":"yes","year":2022},
    {"title":"PFOS Antarctic penguin Weddell seal food web",
     "abstract":"Krill silverfish seal penguin PFOS TMF=3.2. Trophic level δ15N 1.0-4.2. Tissue concentrations ng/g.",
     "journal":"Science of the Total Environment","doi":"10.1016/scitot7890","citations":67,"open_access":"yes","year":2020},
    # 应被排除: 药物主体
    {"title":"Bioaccumulation of pharmaceuticals and stimulants in urban river food web",
     "abstract":"Cocaine caffeine ibuprofen in fish liver invertebrates. Trophic magnification. Multiple species concentrations.",
     "journal":"Environmental Science & Technology","doi":"10.1021/pharma001","citations":55,"open_access":"yes","year":2021},
    {"title":"PFAS beluga whale Arctic Canada food web trophic level",
     "abstract":"Beluga blood PFOS 450 ng/g. TMF 3.8 from prey. Supplementary Excel food web concentrations per species.",
     "journal":"Environmental Science & Technology","doi":"10.1021/es7777","citations":122,"open_access":"yes","year":2018},
    # 重复 DOI
    {"title":"PFAS trophic transfer Great Lakes [duplicate entry]",
     "abstract":"Duplicate.",
     "journal":"Environmental Science & Technology","doi":"10.1021/es5678","citations":145,"open_access":"no","year":2018},
    {"title":"Fluorotelomer PFCA biotransformation fish liver trophic level",
     "abstract":"6:2 FTOH precursor PFOA PFNA in rainbow trout liver. Food web trophic level 2.9. Concentrations ng/g.",
     "journal":"Environmental Science & Technology","doi":"10.1021/es4444","citations":176,"open_access":"yes","year":2017},
    {"title":"Arctic seabird PFAS food web Svalbard TMF multiple species",
     "abstract":"Glaucous gull little auk prey. TMF PFHxS PFOS PFNA. Species concentrations ng/g supplementary S1. Trophic level isotope.",
     "journal":"ES&T Letters","doi":"10.1021/estlett5555","citations":91,"open_access":"yes","year":2020},
]


# ─── 主函数 ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="TMF Literature Priority Ranker v2")
    ap.add_argument("--input", "-i", type=Path, default=None)
    ap.add_argument("--meta",  "-m", type=Path, default=None)
    ap.add_argument("--top",   "-n", type=int,  default=100)
    ap.add_argument("--out",   "-o", type=Path, default=Path("./pfas_ranked"))
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log.addHandler(logging.FileHandler(
        out_dir / f"ranking_{TIMESTAMP}.log", encoding="utf-8"))

    # MetaAnalysis DOI
    meta_dois: set = set()
    if args.meta and args.meta.exists():
        raw = args.meta.read_text(encoding="utf-8").splitlines()
        meta_dois = {
            line.strip().lower().replace("https://doi.org/","").replace("http://doi.org/","")
            for line in raw if line.strip()
        }
        log.info(f"MetaAnalysis DOI: {len(meta_dois)} 条")
    else:
        log.info("未提供 --meta，跳过 +8 加分项")

    # 读取
    if args.input and args.input.exists():
        df = load_refs(args.input)
    else:
        if args.input:
            log.warning(f"找不到 {args.input}，改用 DEMO 数据")
        else:
            log.info("使用内置 DEMO 数据")
        df = pd.DataFrame(DEMO_DATA)
        df["supplement_link"] = df["abstract"].apply(
            lambda x: "si.xlsx" if "supplementary" in x.lower() else "")

    df = normalize_columns(df)
    total_raw = len(df)
    log.info(f"原始总量: {total_raw} 行")

    # ── Step 1: 评分
    log.info("Step 1: 评分...")
    scores, tag_list = [], []
    itr = tqdm(df.iterrows(), total=len(df)) if HAS_TQDM else df.iterrows()
    for _, row in itr:
        s, tags = score_row(row, meta_dois)
        scores.append(s)
        tag_list.append("|".join(sorted(set(tags))))
    df["priority_score"] = scores
    df["score_tags"]     = tag_list
    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)

    # ── Step 2: DOI 去重
    log.info("Step 2: DOI 去重...")

    def clean_doi(d):
        d = str(d or "").strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/"):
            d = d.replace(prefix, "")
        return d if d.startswith("10.") else ""

    df["doi_clean"] = df["doi"].apply(clean_doi)
    has_doi = df[df["doi_clean"] != ""].sort_values(
        "priority_score", ascending=False).drop_duplicates("doi_clean", keep="first")
    no_doi  = df[df["doi_clean"] == ""]
    df_dedup = pd.concat([has_doi, no_doi], ignore_index=True)
    df_dedup = df_dedup.sort_values("priority_score", ascending=False).reset_index(drop=True)
    df_dedup["rank"] = df_dedup.index + 1
    removed_dups = total_raw - len(df_dedup)
    log.info(f"  去重前 {total_raw} 行 → 去重后 {len(df_dedup)} 篇（去掉 {removed_dups} 条重复）")

    # ── Step 3: 资格审查
    log.info("Step 3: 资格审查...")
    quals = []
    for _, row in df_dedup.iterrows():
        text = (str(row.get("title","") or "") + " " + str(row.get("abstract","") or ""))
        quals.append(qualify_row(text))
    df_dedup = pd.concat([df_dedup.reset_index(drop=True), pd.DataFrame(quals)], axis=1)

    # ── Step 4: Top N unique papers
    top_n    = min(args.top, len(df_dedup))
    df_top   = df_dedup.head(top_n).copy()
    df_keep  = df_top[df_top["keep_for_extraction"] == "yes"].copy()
    df_excl  = df_top[df_top["keep_for_extraction"] == "no"].copy()

    # ── 输出
    df_dedup.to_csv(out_dir / f"all_dedup_ranked_{TIMESTAMP}.csv",
                    index=False, encoding="utf-8-sig")
    df_top.to_csv(out_dir / f"top{top_n}_unique_{TIMESTAMP}.csv",
                  index=False, encoding="utf-8-sig")
    df_keep.to_csv(out_dir / f"keep_for_extraction_{TIMESTAMP}.csv",
                   index=False, encoding="utf-8-sig")
    df_excl.to_csv(out_dir / f"excluded_top{top_n}_{TIMESTAMP}.csv",
                   index=False, encoding="utf-8-sig")
    try:
        df_top.to_excel(out_dir / f"top{top_n}_unique_{TIMESTAMP}.xlsx",    index=False)
        df_keep.to_excel(out_dir / f"keep_for_extraction_{TIMESTAMP}.xlsx", index=False)
    except Exception:
        pass

    log.info(f"  → all_dedup_ranked.csv       ({len(df_dedup)} 篇)")
    log.info(f"  → top{top_n}_unique.xlsx      ({top_n} 篇)")
    log.info(f"  → keep_for_extraction.xlsx   ({len(df_keep)} 篇)  ← 直接用这个")
    log.info(f"  → excluded_top{top_n}.csv     ({len(df_excl)} 篇)")

    # ── 统计
    log.info("")
    log.info("=" * 65)
    log.info("  汇总")
    log.info("=" * 65)
    log.info(f"  原始行数:                    {total_raw}")
    log.info(f"  DOI 去重后唯一文献:          {len(df_dedup)}")
    log.info(f"  Top {top_n} 篇:               {top_n}")
    log.info(f"  ✅ keep_for_extraction=yes:  {len(df_keep)}")
    log.info(f"  ❌ keep_for_extraction=no:   {len(df_excl)}")
    log.info("")
    log.info("  资格审查统计 (Top N 内):")
    for field in ["is_pfas_specific","has_trophic_level",
                  "has_concentration_matrix","keep_for_extraction"]:
        vc = df_top[field].value_counts().to_dict() if field in df_top.columns else {}
        log.info(f"    {field:<32} yes={vc.get('yes',0):>4}  no={vc.get('no',0):>4}")
    log.info("")
    log.info("  Top 10 预览:")
    for _, row in df_top.head(10).iterrows():
        flag = "✅" if row.get("keep_for_extraction") == "yes" else "❌"
        log.info(f"  {flag} #{int(row['rank']):>3} [{int(row['priority_score']):>2}分]"
                 f"  {str(row.get('title',''))[:55]}")
        log.info(f"       pfas={row.get('is_pfas_specific','?')}  "
                 f"tl={row.get('has_trophic_level','?')}  "
                 f"conc={row.get('has_concentration_matrix','?')}"
                 + (f"\n       ⚠ {row['exclude_reason']}" if row.get('exclude_reason') else ""))

    (out_dir / "README.txt").write_text(
        f"TMF Literature Ranker v2\n时间: {TIMESTAMP}\n\n"
        f"原始行数: {total_raw}\n去重后: {len(df_dedup)} 篇\n"
        f"Top {top_n}: {top_n} 篇\nkeep_for_extraction: {len(df_keep)} 篇\n\n"
        "字段说明:\n"
        "  is_pfas_specific         是否PFAS专门文章\n"
        "  has_trophic_level        是否含营养级\n"
        "  has_concentration_matrix 是否有物种-浓度矩阵\n"
        "  keep_for_extraction      三项全yes才为yes\n"
        "  exclude_reason           排除原因\n\n"
        "下一步:\n"
        "  1) 打开 keep_for_extraction.xlsx\n"
        "  2) 优先: FROM_META_ANALYSIS + has_supplement_xlsx\n"
        "  3) 提取 Layer B (species trophic) + Layer C (PFAS concentration)\n",
        encoding="utf-8",
    )
    log.info(f"\n✅ 完成. 输出: {out_dir.resolve()}")


if __name__ == "__main__":
    main()