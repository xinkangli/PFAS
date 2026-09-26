"""
统一主表 schema (master table)
==============================
不管哪个数据源 (ECHA / ECOTOX / NITE / paper)，
最终都要写到同一个 SQLite 表里。

字段设计原则:
1. 保留原始值 (raw_*) — 永远不要丢
2. 同时存归一化值 (例如 logBCF) — 方便分析
3. metadata 完整保留 (species, tissue, exposure, temp, water type)
   PFAS 富集严重依赖这些 — 不保留就是灾难
4. reliability / source 必填 — 后面做权重必须用

使用方式:
    from schema import init_db, insert_record
    conn = init_db("data/pfas_master.sqlite")
    insert_record(conn, {...})
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pfas_bcf_master (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- chemical identity
    pfas_short      TEXT NOT NULL,              -- e.g. PFOS
    pfas_name       TEXT,
    cas             TEXT NOT NULL,
    family          TEXT,                       -- PFCA / PFSA / FTOH ...
    chain_len       INTEGER,

    -- endpoint
    endpoint_type   TEXT NOT NULL,              -- 'BCF' | 'BAF' | 'BMF' | 'BSAF' | 'TMF'
    value           REAL,                       -- normalized numeric value (L/kg for BCF/BAF)
    log_value       REAL,                       -- log10 of value
    value_units     TEXT,                       -- normalized unit, e.g. 'L/kg ww'
    raw_value       TEXT,                       -- string as reported (keep verbatim)
    raw_units       TEXT,
    basis           TEXT,                       -- 'wet weight' | 'dry weight' | 'lipid' | NULL
    kinetic_or_ss   TEXT,                       -- 'kinetic' | 'steady-state' | NULL

    -- biology
    species_sci     TEXT,                       -- Latin name preferred
    species_common  TEXT,
    organism_group  TEXT,                       -- 'fish' | 'invertebrate' | 'mammal' | 'bird' | 'algae'
    tissue          TEXT,                       -- 'whole body' | 'liver' | 'muscle' | 'blood' | ...

    -- exposure / conditions
    exposure_days   REAL,
    exposure_route  TEXT,                       -- 'aqueous' | 'dietary' | 'field'
    test_concentration TEXT,
    temperature_c   REAL,
    water_type      TEXT,                       -- 'freshwater' | 'marine' | 'brackish'
    guideline       TEXT,                       -- 'OECD 305' | 'OECD 305-I' | 'OECD 315' ...
    study_type      TEXT,                       -- 'laboratory' | 'field' | 'modelled'

    -- provenance
    source          TEXT NOT NULL,              -- 'ECHA' | 'ECOTOX' | 'NITE-CHRIP' | 'OECD' | 'literature'
    source_url      TEXT,
    citation        TEXT,                       -- DOI or full ref
    reliability     TEXT,                       -- Klimisch 1-4 if known, or 'reported' / 'modelled'
    notes           TEXT,
    fetched_at      TEXT NOT NULL               -- ISO datetime
);

CREATE INDEX IF NOT EXISTS idx_pfas    ON pfas_bcf_master(pfas_short);
CREATE INDEX IF NOT EXISTS idx_cas     ON pfas_bcf_master(cas);
CREATE INDEX IF NOT EXISTS idx_ep      ON pfas_bcf_master(endpoint_type);
CREATE INDEX IF NOT EXISTS idx_source  ON pfas_bcf_master(source);
CREATE INDEX IF NOT EXISTS idx_species ON pfas_bcf_master(species_sci);

-- de-dup signature: same chemical + same endpoint + same species + same value + same source = same record
CREATE UNIQUE INDEX IF NOT EXISTS uq_record ON pfas_bcf_master(
    cas, endpoint_type, COALESCE(species_sci, ''), COALESCE(tissue, ''),
    COALESCE(raw_value, ''), COALESCE(source_url, ''), source
);
"""


def init_db(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def to_log10(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return round(math.log10(value), 4)


def organism_group_from_species(species: str | None) -> str | None:
    """Heuristic — works for most PFAS BCF studies."""
    if not species:
        return None
    s = species.lower()
    fish_keys = ["carp", "trout", "salmon", "fish", "tilapia", "rerio", "danio",
                 "oncorhynchus", "cyprinus", "lepomis", "pimephales", "minnow",
                 "perch", "bass", "catfish", "eel", "anguilla", "rutilus",
                 "ictalurus", "gobio", "stickleback", "smelt", "char"]
    invert_keys = ["mussel", "clam", "oyster", "shrimp", "daphnia", "earthworm",
                   "lumbricus", "eisenia", "hyalella", "chironomus", "amphipod",
                   "bivalve", "gammarus", "snail", "polychaete", "crab", "lobster",
                   "krill", "copepod", "scallop", "annelid"]
    mammal_keys = ["dolphin", "seal", "whale", "porpoise", "otter", "polar bear",
                   "rat", "mouse", "mink", "vole", "bear", "cetacean",
                   "tursiops", "orcinus", "phoca", "halichoerus", "pagophilus",
                   "ursus", "lutra", "delphinapterus", "monodon", "neogale"]
    bird_keys = ["bird", "gull", "egg", "tern", "duck", "cormorant", "guillemot",
                 "eider", "puffin", "eagle", "owl"]
    algae_keys = ["algae", "diatom", "chlorella", "scenedesmus", "selenastrum",
                  "raphidocelis"]
    for k in fish_keys:
        if k in s: return "fish"
    for k in invert_keys:
        if k in s: return "invertebrate"
    for k in mammal_keys:
        if k in s: return "mammal"
    for k in bird_keys:
        if k in s: return "bird"
    for k in algae_keys:
        if k in s: return "algae"
    return None


def normalize_endpoint(ep: str | None) -> str | None:
    if not ep:
        return None
    e = ep.upper().strip()
    # collapse common phrasings
    aliases = {
        "BIOCONCENTRATION FACTOR": "BCF",
        "BCF":                    "BCF",
        "BIOACCUMULATION FACTOR": "BAF",
        "BAF":                    "BAF",
        "BIOMAGNIFICATION FACTOR":"BMF",
        "BMF":                    "BMF",
        "BIOTA-SEDIMENT ACCUMULATION FACTOR": "BSAF",
        "BSAF":                   "BSAF",
        "TROPHIC MAGNIFICATION FACTOR": "TMF",
        "TMF":                    "TMF",
    }
    return aliases.get(e, e)


def normalize_tissue(t: str | None) -> str | None:
    if not t:
        return None
    s = t.lower().strip()
    if "whole" in s or s in {"wb", "body"}:
        return "whole body"
    for k in ["liver", "muscle", "kidney", "blood", "plasma", "serum",
             "egg", "gonad", "brain", "gill", "skin", "fillet"]:
        if k in s:
            return k
    return s


def normalize_basis(b: str | None) -> str | None:
    if not b:
        return None
    s = b.lower().replace(".", "").replace("/", "").replace(" ", "")
    # check lipid FIRST: "lipidweight" contains "dw" as a substring, which would
    # otherwise be falsely matched as dry weight.
    if "lipid" in s or s == "lw":
        return "lipid"
    if "wet" in s or s == "ww":
        return "wet weight"
    if "dry" in s or s == "dw":
        return "dry weight"
    return b.lower().strip()


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = ("pfas_short", "cas", "endpoint_type", "source")


def insert_record(conn: sqlite3.Connection, rec: dict) -> bool:
    """
    Insert one record, returning True if a new row was inserted.
    Duplicate (same CAS+endpoint+species+tissue+value+url+source) is silently skipped.
    """
    for f in REQUIRED_FIELDS:
        if not rec.get(f):
            raise ValueError(f"missing required field: {f}; got {rec}")

    rec = dict(rec)  # defensive copy
    rec["endpoint_type"] = normalize_endpoint(rec.get("endpoint_type"))
    rec["tissue"] = normalize_tissue(rec.get("tissue"))
    rec["basis"] = normalize_basis(rec.get("basis"))
    rec.setdefault("fetched_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))

    # derived
    if rec.get("value") is not None and rec.get("log_value") is None:
        rec["log_value"] = to_log10(rec["value"])
    if rec.get("species_sci") and not rec.get("organism_group"):
        rec["organism_group"] = organism_group_from_species(rec["species_sci"])

    cols = [
        "pfas_short","pfas_name","cas","family","chain_len",
        "endpoint_type","value","log_value","value_units","raw_value","raw_units",
        "basis","kinetic_or_ss",
        "species_sci","species_common","organism_group","tissue",
        "exposure_days","exposure_route","test_concentration","temperature_c",
        "water_type","guideline","study_type",
        "source","source_url","citation","reliability","notes","fetched_at",
    ]
    placeholders = ",".join("?" for _ in cols)
    sql = f"INSERT OR IGNORE INTO pfas_bcf_master ({','.join(cols)}) VALUES ({placeholders})"
    cur = conn.execute(sql, [rec.get(c) for c in cols])
    conn.commit()
    return cur.rowcount > 0


def stats(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    out = {}
    out["total_rows"] = cur.execute("SELECT COUNT(*) FROM pfas_bcf_master").fetchone()[0]
    out["by_source"] = dict(cur.execute(
        "SELECT source, COUNT(*) FROM pfas_bcf_master GROUP BY source").fetchall())
    out["by_endpoint"] = dict(cur.execute(
        "SELECT endpoint_type, COUNT(*) FROM pfas_bcf_master GROUP BY endpoint_type").fetchall())
    out["by_pfas"] = dict(cur.execute(
        "SELECT pfas_short, COUNT(*) FROM pfas_bcf_master GROUP BY pfas_short ORDER BY 2 DESC").fetchall())
    return out
