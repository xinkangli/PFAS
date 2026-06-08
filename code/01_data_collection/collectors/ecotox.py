"""
EPA ECOTOX collector
====================
EPA 把整个 ECOTOX 数据库 (>1M test records) 公开成 pipe-分隔的 ASCII 文件。

流程:
  1. 找/复用本地 ecotox_ascii_*.zip
  2. 读 results / tests / species / chemicals / references / validation/media_type_codes
  3. 按 PFAS CAS 过滤
  4. 过滤 endpoint = BCF/BAF/BMF/BSAF/BCFD/BAFD/BMFD ...
  5. 真实 BCF/BAF 值在 bcf1_mean 列, 不是 conc1_mean (后者是 test 浓度)
  6. 把 metadata 全部带上: tissue / exposure / temp / media / lipid / dry-or-wet / guideline
"""
from __future__ import annotations

import io
import os
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pfas_targets import PFAS_TARGETS, cas_to_meta
from schema import insert_record


ECOTOX_INDEX_URL = "https://cfpub.epa.gov/ecotox/index.cfm"

BIOACC_ENDPOINT_PATTERNS = [
    r"^BCF",
    r"^BAF",
    r"^BMF",
    r"^BSAF",
    r"BIOCONC",
    r"BIOACCUM",
    r"BIOMAGN",
]


def cas_no_dashes(cas: str) -> str:
    return cas.replace("-", "").strip()


def find_latest_zip_url(session: requests.Session) -> str:
    if (env_url := os.environ.get("ECOTOX_ZIP_URL")):
        return env_url
    r = session.get(ECOTOX_INDEX_URL, timeout=60)
    r.raise_for_status()
    candidates = re.findall(
        r'href="(https?://[^"]*ecotox_ascii_\d{2}_\d{2}_\d{4}\.zip)"', r.text)
    if not candidates:
        candidates = re.findall(
            r'href="([^"]*ecotox_ascii_\d{2}_\d{2}_\d{4}\.zip)"', r.text)
        candidates = [urljoin(ECOTOX_INDEX_URL, c) for c in candidates]
    if not candidates:
        raise RuntimeError("No ecotox_ascii zip link found on EPA page.")
    return candidates[0]


def download_zip(dest_dir: Path, session: requests.Session | None = None) -> Path:
    session = session or requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    })
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Prefer any local zip
    for search_dir in [dest_dir, Path.cwd()]:
        local_existing = sorted(search_dir.glob("ecotox_ascii_*.zip"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
        if local_existing and local_existing[0].stat().st_size > 1_000_000:
            print(f"[ECOTOX] using existing local file: {local_existing[0]}")
            return local_existing[0]

    url = find_latest_zip_url(session)
    local = dest_dir / Path(url).name
    print(f"[ECOTOX] downloading {url}")
    with session.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(local, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return local


def _read_pipe(z: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    with z.open(name) as fh:
        data = fh.read()
    return pd.read_csv(io.BytesIO(data), sep="|", encoding="latin-1",
                       low_memory=False, on_bad_lines="skip", dtype=str, **kwargs)


def _find(z: zipfile.ZipFile, stem: str) -> str | None:
    for n in z.namelist():
        if n.endswith(f"/{stem}.txt") or n == f"{stem}.txt":
            return n
    return None


def parse_zip_for_pfas(zip_path: Path) -> pd.DataFrame:
    target_cas_normed = {cas_no_dashes(c["cas"]): c for c in PFAS_TARGETS}

    with zipfile.ZipFile(zip_path) as z:
        chemicals  = _read_pipe(z, _find(z, "chemicals"))
        results    = _read_pipe(z, _find(z, "results"))
        tests      = _read_pipe(z, _find(z, "tests"))
        species    = _read_pipe(z, _find(z, "species"))
        references = None
        ref_path   = _find(z, "references")
        if ref_path:
            references = _read_pipe(z, ref_path)
        # validation lookup tables for human-readable code expansion
        media_codes = exposure_codes = test_method_codes = endpoint_codes = response_site_codes = None
        for stem, var in [
            ("media_type_codes",       "media_codes"),
            ("exposure_type_codes",    "exposure_codes"),
            ("test_method_codes",      "test_method_codes"),
            ("endpoint_codes",         "endpoint_codes"),
            ("response_site_codes",    "response_site_codes"),
        ]:
            p = _find(z, stem)
            if p:
                df = _read_pipe(z, p)
                if var == "media_codes":     media_codes = df
                if var == "exposure_codes":  exposure_codes = df
                if var == "test_method_codes": test_method_codes = df
                if var == "endpoint_codes":  endpoint_codes = df
                if var == "response_site_codes": response_site_codes = df

    chemicals["cas_number"] = chemicals["cas_number"].astype(str).str.replace("-", "").str.strip()
    chemicals = chemicals[chemicals["cas_number"].isin(target_cas_normed.keys())].copy()
    if chemicals.empty:
        return pd.DataFrame()

    tests = tests.copy()
    tests["test_cas"] = tests["test_cas"].astype(str).str.replace("-", "").str.strip()
    tests_pf = tests.merge(chemicals, left_on="test_cas", right_on="cas_number",
                           how="inner", suffixes=("", "_chem"))
    rows = results.merge(tests_pf, on="test_id", how="inner", suffixes=("", "_t"))
    rows = rows.merge(species, on="species_number", how="left", suffixes=("", "_sp"))
    if references is not None and "reference_number" in rows.columns:
        rows = rows.merge(references, on="reference_number", how="left",
                          suffixes=("", "_ref"))

    # Expand validation codes if available
    def _expand(df_codes, col_in_data, code_col_in_table, label_col):
        if df_codes is None or col_in_data not in rows.columns: return
        if code_col_in_table not in df_codes.columns: return
        m = dict(zip(df_codes[code_col_in_table].astype(str).str.strip(),
                     df_codes[label_col].astype(str)))
        rows[col_in_data + "_label"] = rows[col_in_data].astype(str).str.strip().map(m)

    _expand(media_codes,       "media_type",      "code", "description")
    _expand(exposure_codes,    "exposure_type",   "code", "description")
    _expand(test_method_codes, "test_method",     "code", "description")
    _expand(endpoint_codes,    "endpoint",        "code", "description")
    _expand(response_site_codes, "response_site", "code", "description")

    ep_col = "endpoint" if "endpoint" in rows.columns else "endpt"
    pat = "|".join(BIOACC_ENDPOINT_PATTERNS)
    rows = rows[rows[ep_col].astype(str).str.contains(pat, case=False, na=False, regex=True)].copy()
    return rows


# ---------------------------------------------------------------------------
# Convert + insert
# ---------------------------------------------------------------------------

def _f(x):
    if x is None: return None
    s = str(x).strip()
    if not s or s.lower() in ("nan", "nr", "na", "n/a", "ne"): return None
    s = s.replace(",", "").lstrip("~<>=").strip()
    try: return float(s)
    except Exception: return None


# Convert duration to days based on unit string
DUR_TO_DAYS = {
    "s": 1/86400, "sec": 1/86400, "second": 1/86400, "seconds": 1/86400,
    "min": 1/1440, "minute": 1/1440, "minutes": 1/1440,
    "h": 1/24, "hr": 1/24, "hour": 1/24, "hours": 1/24,
    "d": 1, "day": 1, "days": 1, "dat": 1, "dap": 1, "dpf": 1, "dph": 1,
    "wk": 7, "week": 7, "weeks": 7,
    "mo": 30.44, "month": 30.44, "months": 30.44,
    "yr": 365.25, "year": 365.25, "years": 365.25,
}


def _to_days(v, unit):
    f = _f(v)
    if f is None or unit is None: return None
    u = str(unit).strip().lower()
    mult = DUR_TO_DAYS.get(u)
    if mult is None:
        # try first token
        u0 = u.split()[0] if u else ""
        mult = DUR_TO_DAYS.get(u0)
    if mult is None: return None
    return f * mult


def _safe_str(x) -> str:
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception: pass
    return str(x).strip()


def _media_to_water_type(label, media_code) -> str | None:
    raw = _safe_str(media_code).upper().rstrip("/").strip()
    code_map = {
        "FW":  "freshwater", "SW":  "marine", "BR": "brackish", "EST": "brackish",
        "AQU": "freshwater", "NAT": "freshwater", "CUL": "freshwater",
        "ART": "freshwater", "HYP": "freshwater", "LIT": "freshwater",
        "SED": "sediment", "SOI": "soil", "SOL": "soil", "MAN": "soil",
    }
    if raw in code_map:
        return code_map[raw]
    s = (_safe_str(label) + " " + raw).lower()
    if not s.strip(): return None
    if any(k in s for k in ["fresh", "lake", "stream", "river", "pond", "tap", "well", "ground"]):
        return "freshwater"
    if any(k in s for k in ["sea", "salt", "marine", "ocean", "estuar"]):
        return "marine"
    if "brack" in s: return "brackish"
    if "sediment" in s: return "sediment"
    if "soil" in s: return "soil"
    return None


def _tissue_from_response(rs, comments, label) -> str | None:
    """ECOTOX uses 2-3 letter codes (WO=whole organism, LI=liver, MU=muscle...).
    The validation/response_site_codes.txt expansion populates response_site_label.
    Fall back to scanning comments if label is missing or vague."""
    raw = _safe_str(rs).upper()
    lab = _safe_str(label)
    com = _safe_str(comments)

    # Direct code mappings (most common in PFAS BCF data)
    code_map = {
        "WO":   "whole body",        # Whole organism
        "WB":   "whole body",
        "TB":   "whole body",        # Total body
        "LI":   "liver",
        "MU":   "muscle",
        "MUL":  "muscle",
        "MT":   "muscle",
        "GI":   "gill",
        "KI":   "kidney",
        "BL":   "blood",
        "PL":   "plasma",
        "SE":   "serum",
        "SK":   "skin",
        "EG":   "egg",
        "GO":   "gonad",
        "BR":   "brain",
        "ST":   "stomach",
        "TI":   "testis",
        "RO":   "root",
        "SO":   "shoot",
        "LE":   "leaf",
        "LD":   "leaf",
        "SLL":  "leaf",
        "RR":   "root",
        "VI":   "viscera",
        "HP":   "hepatopancreas",
        "AB":   "abdomen",
        "ABD":  "abdomen",
        "AD":   "adipose",
        "FA":   "fat",
        "ET":   "entire tissue",
        "PT":   "soft parts",
        "MA":   "mantle",
        "SS":   "soft parts",
    }
    if raw in code_map:
        return code_map[raw]

    # If label looks like an unhelpful "Unspecified", drop it
    s = (lab + " " + com).lower()
    if not s.strip(): return None
    for k in ["whole body", "whole organism", "whole animal", "carcass",
              "liver", "muscle", "kidney", "blood", "plasma", "serum",
              "egg", "gonad", "brain", "gill", "skin", "fillet", "fat",
              "yolk", "viscera"]:
        if k in s: return k.replace("whole organism", "whole body").replace("whole animal", "whole body")
    return None


def _basis_from_dry_wet(dry_wet) -> str | None:
    s = _safe_str(dry_wet).upper()
    if not s: return None
    if s in ("W", "WW"): return "wet weight"
    if s in ("D", "DW"): return "dry weight"
    if s in ("L", "LW", "LIPID"): return "lipid"
    return None


def _guideline_from_method(method, label) -> str | None:
    """
    ECOTOX test_method is a code like OECD / USEPA / ASTM / EPAOECD / OPPTS.
    These map (loosely) to common BCF guidelines:
        OECD     -> OECD 305 (default for BCF studies)
        OPPTS    -> EPA OPPTS 850
        EPAOECD  -> OECD 305
        USEPA    -> EPA OPPTS
    Combined with comments, sometimes a specific OECD number is mentioned.
    """
    code = _safe_str(method).upper()
    lab = _safe_str(label)
    full = (code + " " + lab)
    if not full.strip(): return None

    m = re.search(r"OECD\s*(?:TG\s*)?(\d+[A-Za-z\-]*)", full, re.I)
    if m: return f"OECD {m.group(1)}"
    m = re.search(r"OPPTS\s*(\d+[\d\-]*)", full, re.I)
    if m: return f"EPA OPPTS {m.group(1)}"
    m = re.search(r"\bEU\s*(?:Method\s*)?(C\.?\d+[A-Z\-]*)", full, re.I)
    if m: return f"EU Method {m.group(1)}"

    code_short = {
        "OECD":      "OECD (likely 305)",
        "OPPTS":     "EPA OPPTS",
        "EPAOECD":   "EPA + OECD",
        "USEPA":     "USEPA",
        "ASTM":      "ASTM",
        "ASTMOECD":  "ASTM + OECD",
        "EPAASTM":   "EPA + ASTM",
        "ISO":       "ISO",
        "ISOOECD":   "ISO + OECD",
        "EPAOM":     "EPA Other Method",
        "GLP":       "GLP only",
        "STDMETH":   "Standard Methods",
        "EEC":       "EEC method",
    }
    if code in code_short:
        return code_short[code]
    if code in ("NC", "NR", "--", ""):
        return None
    return code or None


def to_master_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty: return []
    meta = cas_to_meta()
    no_dash_meta = {c.replace("-",""): m for c, m in meta.items()}

    out = []
    ep_col = "endpoint" if "endpoint" in df.columns else "endpt"

    for _, r in df.iterrows():
        cas_raw = str(r.get("cas_number", "")).replace("-", "").strip()
        m = no_dash_meta.get(cas_raw)
        if not m: continue

        endpoint_raw = str(r.get(ep_col, "")).strip().upper()
        # Pick the BCF/BAF value from bcf1_mean (true BCF value column),
        # not conc1_mean (which is test concentration).
        bcf_val   = _f(r.get("bcf1_mean"))
        bcf_unit  = (_safe_str(r.get("bcf1_unit")) or None)
        bcf_min   = _f(r.get("bcf1_min"))
        bcf_max   = _f(r.get("bcf1_max"))
        if bcf_val is None: bcf_val = _f(r.get("bcf2_mean"))
        if bcf_val is None: bcf_val = _f(r.get("bcf3_mean"))
        if bcf_unit is None: bcf_unit = (_safe_str(r.get("bcf2_unit")) or _safe_str(r.get("bcf3_unit")) or None)

        # If still no BCF value available, skip. The conc1_mean stuff was
        # being inserted incorrectly before.
        if bcf_val is None:
            continue

        exposure_days = _to_days(r.get("exposure_duration_mean"),
                                 r.get("exposure_duration_unit"))
        if exposure_days is None:
            exposure_days = _to_days(r.get("study_duration_mean"),
                                     r.get("study_duration_unit"))
        if exposure_days is None:
            exposure_days = _to_days(r.get("obs_duration_mean"),
                                     r.get("obs_duration_unit"))

        water_type = _media_to_water_type(
            r.get("media_type_label"), r.get("media_type"))
        tissue = _tissue_from_response(
            r.get("response_site"), r.get("response_site_comments"),
            r.get("response_site_label"))

        steady_state = _safe_str(r.get("steady_state")).upper() or None
        kinetic_or_ss = None
        if endpoint_raw.endswith("D") or "KINETIC" in endpoint_raw:
            kinetic_or_ss = "kinetic"
        elif steady_state in ("Y", "YES", "S", "SS"):
            kinetic_or_ss = "steady-state"

        guideline = _guideline_from_method(
            r.get("test_method"), r.get("test_method_label"))

        test_conc = None
        cm = _safe_str(r.get("conc1_mean"))
        if cm:
            unit_c = _safe_str(r.get("conc1_unit"))
            test_conc = f"{cm} {unit_c}".strip()

        # Reliability: ECOTOX is curated; flag QA-relevant signals if present.
        reliability_bits = ["ECOTOX-curated"]
        if r.get("chem_analysis_method"):
            reliability_bits.append(f"analysis={r['chem_analysis_method']}")

        rec = {
            "pfas_short":    m["short"],
            "pfas_name":     m["name"],
            "cas":           m["cas"],
            "family":        m["family"],
            "chain_len":     m["chain_len"],

            "endpoint_type": endpoint_raw.split("/")[0].split(" ")[0] or "BCF",
            "value":         bcf_val,
            "value_units":   bcf_unit or "L/kg",
            "raw_value":     str(r.get("bcf1_mean") or ""),
            "raw_units":     bcf_unit,
            "basis":         _basis_from_dry_wet(r.get("dry_wet")),
            "kinetic_or_ss": kinetic_or_ss,

            "species_sci":   (_safe_str(r.get("latin_name")) or None),
            "species_common":(_safe_str(r.get("common_name")) or None),
            "tissue":        tissue,

            "exposure_days":     exposure_days,
            "exposure_route":    (_safe_str(r.get("exposure_type_label")) or
                                  _safe_str(r.get("exposure_type")) or None),
            "test_concentration": test_conc,
            "temperature_c":     None,
            "water_type":        water_type,
            "guideline":         guideline,
            "study_type":        "laboratory" if _safe_str(r.get("test_location")).upper() == "LAB" else (
                                  "field" if _safe_str(r.get("test_location")).upper() == "FIELD" else
                                  "laboratory"),

            "source":        "ECOTOX",
            "source_url":    f"https://cfpub.epa.gov/ecotox/search.cfm?sub=A&q=true&CAS={m['cas']}",
            "citation":      " ".join(_safe_str(r.get(k)) for k in ("author","publication_year","title","source"))[:500],
            "reliability":   "; ".join(reliability_bits),
            "notes":         f"test_id={r.get('test_id')} ref={r.get('reference_number')} lipid_pct={r.get('lipid_pct_mean')} resp_site={r.get('response_site')}",
        }
        out.append(rec)
    return out


def run(conn, data_dir: Path):
    data_dir = Path(data_dir) / "ecotox"
    zip_path = download_zip(data_dir)
    df = parse_zip_for_pfas(zip_path)
    print(f"[ECOTOX] PFAS bioaccumulation rows (raw): {len(df)}")
    rows = to_master_rows(df)
    print(f"[ECOTOX] rows with valid BCF/BAF value: {len(rows)}")
    # Wipe old (broken) ECOTOX rows so we don't keep the conc-as-BCF mess.
    n_old = conn.execute("DELETE FROM pfas_bcf_master WHERE source='ECOTOX'").rowcount
    conn.commit()
    print(f"[ECOTOX] removed {n_old} old (broken) ECOTOX rows before reload")
    n_new = 0
    for rec in rows:
        try:
            if insert_record(conn, rec):
                n_new += 1
        except Exception as e:
            print(f"[ECOTOX] insert err: {e}")
    print(f"[ECOTOX] inserted {n_new} new rows ({len(rows) - n_new} duplicates skipped)")
    return n_new


if __name__ == "__main__":
    from schema import init_db, stats
    conn = init_db("data/pfas_master.sqlite")
    run(conn, Path("data"))
    print(stats(conn))
