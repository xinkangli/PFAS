"""
ITRC PFAS BCF-BAF compilation Table 5-1 (aquatic) + Table 5-2 (plants)
======================================================================
Interstate Technology & Regulatory Council 把多年文献里 PFAS BCF/BAF
整理成两个 Excel 数据库，免费公开:

  Table 5-1 (aquatic):   1460 rows of BCF or BAF for fish / invertebrate / etc.
  Table 5-2 (plants):    BCF/BAF/BMF for plants

每行结构非常干净: PFAS Name | Acronym | CAS | BCF/BAF | Reference |
                    Lab/Field/Model | Location | Freshwater/Marine |
                    Organism | Latin name | Tissue | Wet/Dry/Lipid | Notes

Source: https://pfas-1.itrcweb.org/external-data-tables/
Files are direct downloads (PDF/Wayback-archived links discovered via search).
"""
from __future__ import annotations

import re
from pathlib import Path

import openpyxl
import requests

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pfas_targets import PFAS_TARGETS, cas_to_meta, by_short
from schema import insert_record


# These URLs work directly (verified via referer probe). Cloudflare 403s the
# external-data-tables page in headless requests, but the wp-content/ assets
# themselves are reachable as long as we send a real-browser UA + referer.
ITRC_FILES = [
    {
        "label": "aquatic",
        "url": ("https://pfas-1.itrcweb.org/wp-content/uploads/2023/10/"
                "ITRC_PFAS_-BCF-BAF_compilation_Table5-1_Oct2021.xlsx"),
        "sheet": "BCF-BAF Database",
        "header_row": 7,   # 1-indexed
        "default_organism_group": None,   # let species heuristic decide
    },
    {
        "label": "plants",
        "url": ("https://pfas-1.itrcweb.org/wp-content/uploads/2022/01/"
                "ITRC_PFASTable5-2_BCFBAFBMF_Plants_Apr2020-1.xlsx"),
        "sheet": None,   # find a sheet name with 'BCF' or 'Plant'
        "header_row": None,
        "default_organism_group": "plant",
    },
]


HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux) Chrome/120 Safari/537.36",
    "Referer": "https://pfas-1.itrcweb.org/external-data-tables/",
}


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 5000:
        return dest
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def _norm_cas(c) -> str | None:
    if c is None: return None
    s = str(c).strip()
    if not s or s == "0" or s.lower() in ("none", "nan", "n/a"): return None
    # ITRC sometimes stores CAS as int (drops dashes). Re-insert.
    s_digits = s.replace("-", "")
    if not s_digits.isdigit(): return s if "-" in s else None
    if "-" in s: return s
    # reformat NNNNNNN-NN-N
    if len(s_digits) < 5: return None
    return f"{s_digits[:-3]}-{s_digits[-3:-1]}-{s_digits[-1]}"


def _f(x):
    if x is None: return None
    s = str(x).strip()
    if not s or s.lower() in ("nan","n/a","nd","na","-"): return None
    s = s.lstrip("~<>=").replace(",", "")
    # strip trailing units / chars like "BCF range"
    m = re.match(r"^([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)", s)
    if not m: return None
    try: return float(m.group(1))
    except Exception: return None


def _norm_water(s) -> str | None:
    if not s: return None
    t = str(s).lower()
    if "fresh" in t: return "freshwater"
    if "marine" in t or "salt" in t or "ocean" in t: return "marine"
    if "estuar" in t or "brack" in t: return "brackish"
    return None


def _norm_tissue(t) -> str | None:
    if not t: return None
    s = str(t).strip().lower()
    if "whole" in s or s == "wb": return "whole body"
    for k in ["liver","muscle","kidney","blood","plasma","egg","gonad","brain",
              "gill","skin","fillet","yolk","fat","viscera","root","leaf",
              "shoot","grain","seed","fruit"]:
        if k in s: return k
    return s


def _norm_basis(b) -> str | None:
    if not b: return None
    s = str(b).strip().lower()
    if "lipid" in s: return "lipid"
    if "wet" in s or s in {"w","ww"}: return "wet weight"
    if "dry" in s or s in {"d","dw"}: return "dry weight"
    return None


def _study_type(s) -> str:
    if not s: return "laboratory"
    t = str(s).lower()
    if "field" in t: return "field"
    if "model" in t: return "modelled"
    return "laboratory"


def parse_plants_xlsx(path: Path, cfg: dict) -> list[dict]:
    """ITRC Table 5-2 has a 2-row header and PFAS Name/CAS only filled on first row
    of each chemical block; species rows below get None in those columns."""
    wb = openpyxl.load_workbook(path, data_only=True)
    sheets = wb.sheetnames
    sheet_name = next((s for s in sheets if "bcf" in s.lower() or "baf" in s.lower()), sheets[0])
    ws = wb[sheet_name]

    # find header rows: row containing "PFAS Name" plus the row below with "Min", "Max", "Mean"
    pfas_row = sub_row = None
    for r in range(1, 12):
        vals = [str(c.value or "").strip().lower() for c in ws[r]]
        if "pfas name" in vals: pfas_row = r
        if pfas_row and "min" in vals and "max" in vals and "mean" in vals:
            sub_row = r; break
    if not pfas_row or not sub_row:
        return []

    header_top = [str(c.value or "").strip() for c in ws[pfas_row]]
    header_sub = [str(c.value or "").strip() for c in ws[sub_row]]

    # Identify column indices
    name_i    = header_top.index("PFAS Name") if "PFAS Name" in header_top else 0
    acro_i    = header_top.index("Acronym") if "Acronym" in header_top else 1
    cas_i     = next((i for i,h in enumerate(header_top) if h.lower().startswith("cas")), None)
    species_i = next((i for i,h in enumerate(header_top) if "species" in h.lower()), None)
    mat_i     = next((i for i,h in enumerate(header_top) if "material" in h.lower() or "part" in h.lower()), None)
    ref_i     = next((i for i,h in enumerate(header_top) if "reference" in h.lower()), None)
    min_i     = next((i for i,h in enumerate(header_sub) if h.lower() == "min"), None)
    max_i     = next((i for i,h in enumerate(header_sub) if h.lower() == "max"), None)
    mean_i    = next((i for i,h in enumerate(header_sub) if h.lower() == "mean"), None)

    short_to_meta = {m["short"].upper(): m for m in PFAS_TARGETS}
    syn_to_meta = {syn.upper(): m for m in PFAS_TARGETS for syn in m.get("synonyms", [])}
    cas_to_m = cas_to_meta()

    out = []
    cur_pfas = None  # carries forward across rows in a block
    for row in ws.iter_rows(min_row=sub_row+1, values_only=True):
        if all(v is None for v in row): continue
        # Refresh PFAS context if name/acronym present
        name = row[name_i] if name_i is not None and name_i < len(row) else None
        acro = row[acro_i] if acro_i is not None and acro_i < len(row) else None
        cas_v = row[cas_i] if cas_i is not None and cas_i < len(row) else None
        if name or acro or cas_v:
            cas = _norm_cas(cas_v)
            m = None
            if cas: m = cas_to_m.get(cas)
            if not m and acro: m = short_to_meta.get(str(acro).strip().upper())
            if not m and name: m = syn_to_meta.get(str(name).strip().upper())
            if not m and acro:
                m = {"short": str(acro).strip(), "name": str(name or ""),
                     "cas": cas or "0-00-0", "family": "unclassified",
                     "chain_len": None}
            if m: cur_pfas = m

        if cur_pfas is None: continue

        species = row[species_i] if species_i is not None and species_i < len(row) else None
        material = row[mat_i] if mat_i is not None and mat_i < len(row) else None
        ref = row[ref_i] if ref_i is not None and ref_i < len(row) else None
        v_mean = _f(row[mean_i]) if mean_i is not None and mean_i < len(row) else None
        v_min  = _f(row[min_i])  if min_i  is not None and min_i  < len(row) else None
        v_max  = _f(row[max_i])  if max_i  is not None and max_i  < len(row) else None

        value = v_mean
        if value is None and v_min is not None and v_max is not None:
            value = (v_min + v_max) / 2
        elif value is None:
            value = v_min if v_min is not None else v_max
        if value is None or not species: continue

        # Plants: Material Sampled holds tissue (Leaves, Roots, Fruit, Grain, etc.)
        rec = {
            "pfas_short":   cur_pfas["short"],
            "pfas_name":    cur_pfas.get("name"),
            "cas":          cur_pfas["cas"],
            "family":       cur_pfas.get("family") or "unclassified",
            "chain_len":    cur_pfas.get("chain_len"),
            "endpoint_type":"BCF",   # plant root concentration factors are typically called BCF
            "value":        value,
            "value_units":  "L/kg",
            "raw_value":    str(v_mean if v_mean is not None else f"{v_min}-{v_max}"),
            "species_sci":  None,
            "species_common": str(species).strip() if species else None,
            "organism_group": "plant",
            "tissue":       _norm_tissue(material) if material else None,
            "exposure_route":"root uptake",
            "study_type":   "field",
            "source":       "ITRC",
            "source_url":   cfg["url"],
            "citation":     str(ref).strip() if ref else "ITRC PFAS Table 5-2",
            "reliability":  "ITRC-compiled",
            "notes":        f"plant: material={material}",
        }
        out.append(rec)
    return out


def parse_xlsx(path: Path, cfg: dict, debug: bool = False) -> list[dict]:
    wb = openpyxl.load_workbook(path, data_only=True)
    sheets = wb.sheetnames
    if debug: print(f'[ITRC] {path.name} sheets: {sheets}')

    # Pick sheet
    sheet_name = cfg["sheet"]
    if sheet_name is None:
        # auto-pick: first sheet whose name mentions BCF/Plant/Database
        for cand in sheets:
            low = cand.lower()
            if any(k in low for k in ("bcf","baf","database","plant","data")):
                if "readme" in low: continue
                if "log" in low: continue
                if "ref" in low: continue
                if "review" in low: continue
                sheet_name = cand
                break
    if sheet_name is None: sheet_name = sheets[0]
    ws = wb[sheet_name]
    if debug: print(f'[ITRC] using sheet: {sheet_name}')

    # Pick header row
    header_row = cfg["header_row"]
    if header_row is None:
        # scan first 12 rows for one containing both "PFAS" and "CAS"
        for r in range(1, 13):
            vals = [str(c.value or "") for c in ws[r]]
            joined = " | ".join(vals).lower()
            if "pfas" in joined and "cas" in joined:
                header_row = r
                break
    if header_row is None:
        raise RuntimeError(f"No header row found in {path.name}")

    header = [str(c.value).strip() if c.value else "" for c in ws[header_row]]
    if debug: print(f'[ITRC] header row {header_row}: {header}')

    # Map header text → column index. Match exactly first, then prefix.
    norm_h = [h.lower().strip().replace("  "," ") for h in header]
    def col_exact(*candidates):
        for cand in candidates:
            cl = cand.lower().strip()
            for i, h in enumerate(norm_h):
                if h == cl:
                    return i
        return None
    def col_prefix(*candidates):
        for cand in candidates:
            cl = cand.lower().strip()
            for i, h in enumerate(norm_h):
                if h.startswith(cl):
                    return i
        return None
    def col_contains(*candidates):
        for cand in candidates:
            cl = cand.lower().strip()
            for i, h in enumerate(norm_h):
                if cl in h:
                    return i
        return None

    idx = {
        "name":    col_exact("PFAS Name") or col_contains("PFAS Name", "PFAS"),
        "acronym": col_exact("Acronym"),
        "carbons": col_contains("Number of Carbons", "Carbons"),
        "cas":     col_exact("CAS No.") or col_contains("CAS"),
        "bcf_min": col_exact("BCF Min"),
        "bcf_max": col_exact("BCF Max"),
        "bcf":     col_exact("BCF"),
        "baf_min": col_exact("BAF Min"),
        "baf_max": col_exact("BAF Max"),
        "baf":     col_exact("BAF"),
        "bmf":     col_exact("BMF") or col_contains("BMF"),
        "ref":     col_exact("Reference") or col_contains("Reference"),
        "study_t": col_contains("Lab or Field", "Study Type"),
        "location":col_contains("Location"),
        "water":   col_contains("Freshwater", "Water Type"),
        "waterbody":col_contains("Waterbody"),
        "org_common":col_contains("Common Name", "Organism Common"),
        "org_sci": col_contains("Scientific Name", "Latin Name"),
        "tissue":  col_contains("Tissue", "Plant Part", "Material Sampled"),
        "basis":   col_contains("Wet/Dry", "Wet Dry", "Basis"),
        "notes":   col_contains("Reviewer", "Notes"),
        "uid":     col_contains("Unique Identifier", "UID"),
    }
    if debug: print('[ITRC] column mapping:', {k:v for k,v in idx.items() if v is not None})

    # Build CAS lookup from PFAS_TARGETS (we want to keep all rows; CAS only used for tagging)
    meta = cas_to_meta()
    short_to_meta = {m["short"].upper(): m for m in PFAS_TARGETS}
    syn_to_meta = {}
    for m in PFAS_TARGETS:
        for syn in m.get("synonyms", []):
            syn_to_meta[syn.upper()] = m

    out = []
    for row in ws.iter_rows(min_row=header_row+1, values_only=True):
        if all(v is None for v in row): continue
        name = row[idx["name"]] if idx["name"] is not None else None
        acronym = row[idx["acronym"]] if idx["acronym"] is not None else None
        cas_raw = row[idx["cas"]] if idx["cas"] is not None else None
        cas = _norm_cas(cas_raw)

        m = None
        if cas: m = meta.get(cas)
        if not m and acronym: m = short_to_meta.get(str(acronym).strip().upper())
        if not m and name: m = syn_to_meta.get(str(name).strip().upper())
        if not m:
            # Unknown PFAS — keep but tag with raw name/CAS, attach to closest short
            # Most ITRC rows ARE for PFAS in our target list; for those that aren't,
            # we use a "ITRC-other" pseudo-short so we don't lose them.
            family = None
            short = (str(acronym).strip() if acronym else None) or (str(name).strip()[:20] if name else None) or "ITRC-other"
            m = {
                "short": short or "ITRC-other",
                "name": str(name or short),
                "cas": cas or "0-00-0",
                "family": "unclassified",
                "chain_len": row[idx["carbons"]] if idx["carbons"] is not None else None,
            }

        bcf_val = _f(row[idx["bcf"]]) if idx["bcf"] is not None else None
        baf_val = _f(row[idx["baf"]]) if idx["baf"] is not None else None
        bmf_val = _f(row[idx["bmf"]]) if idx["bmf"] is not None else None

        # Sometimes only Min/Max present; use mean
        if bcf_val is None and idx["bcf_min"] is not None and idx["bcf_max"] is not None:
            lo, hi = _f(row[idx["bcf_min"]]), _f(row[idx["bcf_max"]])
            if lo is not None and hi is not None: bcf_val = (lo + hi) / 2
            elif lo is not None: bcf_val = lo
            elif hi is not None: bcf_val = hi
        if baf_val is None and idx["baf_min"] is not None and idx["baf_max"] is not None:
            lo, hi = _f(row[idx["baf_min"]]), _f(row[idx["baf_max"]])
            if lo is not None and hi is not None: baf_val = (lo + hi) / 2
            elif lo is not None: baf_val = lo
            elif hi is not None: baf_val = hi

        # Each row can produce 1-3 master records depending on which endpoints exist.
        common_fields = {
            "pfas_short":    m["short"],
            "pfas_name":     m.get("name") or name,
            "cas":           m["cas"],
            "family":        m.get("family") or "unclassified",
            "chain_len":     m.get("chain_len"),
            "value_units":   "L/kg",
            "basis":         _norm_basis(row[idx["basis"]]) if idx["basis"] is not None else None,
            "species_sci":   (str(row[idx["org_sci"]]).strip() if idx["org_sci"] is not None and row[idx["org_sci"]] else None),
            "species_common":(str(row[idx["org_common"]]).strip() if idx["org_common"] is not None and row[idx["org_common"]] else None),
            "tissue":        _norm_tissue(row[idx["tissue"]]) if idx["tissue"] is not None else None,
            "water_type":    _norm_water(row[idx["water"]]) if idx["water"] is not None else None,
            "study_type":    _study_type(row[idx["study_t"]]) if idx["study_t"] is not None else "laboratory",
            "source":        "ITRC",
            "source_url":    cfg["url"],
            "citation":      str(row[idx["ref"]]).strip() if idx["ref"] is not None and row[idx["ref"]] else "ITRC PFAS Table 5-1/5-2",
            "reliability":   "ITRC-compiled",
            "notes":         "; ".join(filter(None, [
                f'location={row[idx["location"]]}' if idx["location"] is not None and row[idx["location"]] else "",
                f'waterbody={row[idx["waterbody"]]}' if idx["waterbody"] is not None and row[idx["waterbody"]] else "",
                f'note={row[idx["notes"]]}' if idx["notes"] is not None and row[idx["notes"]] else "",
                f'uid={row[idx["uid"]]}' if idx["uid"] is not None and row[idx["uid"]] else "",
            ]))[:1000],
        }
        if cfg["default_organism_group"]:
            common_fields["organism_group"] = cfg["default_organism_group"]

        if bcf_val is not None:
            rec = dict(common_fields,
                       endpoint_type="BCF",
                       value=bcf_val,
                       raw_value=str(row[idx["bcf"]] if idx["bcf"] is not None else bcf_val))
            out.append(rec)
        if baf_val is not None:
            rec = dict(common_fields,
                       endpoint_type="BAF",
                       value=baf_val,
                       raw_value=str(row[idx["baf"]] if idx["baf"] is not None else baf_val))
            out.append(rec)
        if bmf_val is not None:
            rec = dict(common_fields,
                       endpoint_type="BMF",
                       value=bmf_val,
                       raw_value=str(row[idx["bmf"]] if idx["bmf"] is not None else bmf_val))
            out.append(rec)
    return out


def run(conn, data_dir: Path | None = None, cas_subset=None):
    data_dir = (Path(data_dir) if data_dir else Path("data")) / "itrc"
    data_dir.mkdir(parents=True, exist_ok=True)
    total_new = 0
    for cfg in ITRC_FILES:
        local = data_dir / f"ITRC_Table_{cfg['label']}.xlsx"
        try:
            download(cfg["url"], local)
            print(f"[ITRC] {cfg['label']}: file {local.name} ({local.stat().st_size//1024} KB)")
        except Exception as e:
            print(f"[ITRC] {cfg['label']}: download failed: {e}")
            if not local.exists() or local.stat().st_size < 5000:
                continue
        try:
            if cfg["label"] == "plants":
                recs = parse_plants_xlsx(local, cfg)
            else:
                recs = parse_xlsx(local, cfg, debug=True)
            print(f"[ITRC] {cfg['label']}: parsed {len(recs)} records")
            for rec in recs:
                try:
                    if insert_record(conn, rec):
                        total_new += 1
                except Exception as e:
                    pass
        except Exception as e:
            print(f"[ITRC] {cfg['label']}: parse failed: {e}")
    print(f"[ITRC] total new rows inserted: {total_new}")
    return total_new


if __name__ == "__main__":
    from schema import init_db, stats
    conn = init_db("data/pfas_master.sqlite")
    run(conn)
    print(stats(conn))
