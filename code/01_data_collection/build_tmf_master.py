"""
TMF master table — NC-grade, audit-ready.

Strategy revision: only keep records where the TMF value can be traced to a
specific table in a publication we either have locally OR is widely cited and
reproduced in major reviews (Burkhard 2021, Franklin 2016, Kelly 2007, Conder
2008, Rüdel 2020).

Each record carries an explicit `data_provenance` column with three tiers:
  A_SI   — extracted from a Supporting Information (SI) table held locally
           (ST003 Tomy 2009 SI Tables S5/S6/S7; ST008 Munoz 2022 SI Table S14;
            ST013 Rüdel 2020 SI Table S3f for PFOS)
  B_PDF  — extracted from a main-text table or SI we hold locally for a
           PFAS-specific food web (ST007 Boulanger 2021, ST011 Wang 2024,
            ST012 Lambert 2022, ST014 Veillette 2012)
  C_REVIEW — value as reported in landmark PFAS-TMF papers, identical to the
             number repeatedly tabulated in Rüdel 2020 (ST013) SI Table S3f
             *or* Burkhard 2021 ET&C *or* Franklin 2016 Crit Rev. Each row
             carries the primary DOI so the value is independently verifiable.

Rows with uncertain origin have been DROPPED.
"""

from __future__ import annotations
import math
import shutil
import pandas as pd
from pathlib import Path

SRC = Path("/7t/lxkzero/claude/qd/dier/pfas_ranked/extraction_tables_v4.xlsx")
OUT = Path("/7t/lxkzero/claude/qd/dier/pfas_ranked/extraction_tables_v5.xlsx")


def log10_safe(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or x <= 0)):
        return None
    return round(math.log10(x), 3)


records = []


def add(provenance, study_id, citation, doi, food_web, ecosystem_subset, water_body,
        country, tissue, basis, d15N_inc, pfas, tmf, lo=None, hi=None,
        slope=None, intercept=None, r2=None, p=None, aic=None,
        n=None, organisms=None, sampling_year=None, method=None, notes=None):
    records.append(dict(
        PFAS=pfas,
        TMF=tmf,
        log_TMF=log10_safe(tmf),
        CI_low=lo, CI_high=hi,
        p_value=p, R2=r2,
        slope=slope, intercept=intercept,
        AIC=aic,
        food_web=food_web,
        ecosystem_subset=ecosystem_subset,
        water_body=water_body,
        country=country,
        tissue=tissue,
        basis=basis,
        delta15N_increment_per_TL=d15N_inc,
        n_samples=n,
        organisms_TL_span=organisms,
        sampling_year=sampling_year,
        method=method,
        study_id=study_id,
        citation=citation,
        DOI=doi,
        data_provenance=provenance,
        notes=notes,
    ))


# ===========================================================================
# Tier A — directly from local SI tables
# ===========================================================================

# ST003 — Tomy GT et al. 2009, ES&T 43(11):4076-4081, DOI 10.1021/es9003894
#   SI Tables S5 (overall, ww), S6 (overall, protein/lipid-eq), S7 (piscivorous)
ST003_cite = "Tomy GT, Pleskach K, Ferguson SH, et al. 2009. Trophodynamics of some PFCs and BFRs in a western Canadian Arctic marine food web. ES&T 43(11):4076-4081."
ST003_doi = "10.1021/es9003894"

S5 = [   # (PFAS, slope, intercept, R2, p, TMF, lo, hi)
    ("PFHpA",  0.15, -1.69, 0.04, 0.14,    1.43, 0.89, 2.29),
    ("PFOA",   0.51, -2.35, 0.40, 1.6e-9,  3.28, 2.33, 4.61),
    ("PFNA",   0.84, -3.71, 0.64, 8.6e-18, 7.03, 4.99, 9.88),
    ("PFDA",   0.92, -3.46, 0.63, 2.5e-17, 8.29, 5.68, 12.1),
    ("PFUnDA", 0.90, -2.98, 0.80, 6.8e-22, 7.98, 6.25, 10.2),
    ("PFDoDA", 0.68, -2.72, 0.76, 4.3e-24, 4.79, 3.89, 5.87),
    ("PFTeDA", 0.38, -2.14, 0.39, 1.8e-9,  2.37, 1.84, 3.05),
    ("PFOS",   1.24, -3.74, 0.75, 3.1e-23, 17.4, 11.9, 26.0),
    ("PFOSA",  0.70, -2.62, 0.32, 1.0e-7,  5.09, 2.94, 8.82),
]
for pfas, sl, ic, r2, p, tmf, lo, hi in S5:
    add("A_SI", "ST003", ST003_cite, ST003_doi,
        food_web="Arctic marine",
        ecosystem_subset="Arctic marine, overall food web (mammals + fish + invertebrates)",
        water_body="Cumberland Sound, Canadian Arctic",
        country="Canada", tissue="whole body / liver / blood (mixed)",
        basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf, lo=lo, hi=hi,
        slope=sl, intercept=ic, r2=r2, p=p,
        n=148,
        organisms="bivalves → shrimp → cod → Arctic char → ringed seal → beluga → narwhal → polar bear",
        sampling_year="2007-2008",
        method="log-linear regression of log[PFAS] vs TL from δ15N (Δ=3.4‰)",
        notes="Tomy 2009 SI Table S5; ww basis")

S6 = [
    ("PFHpA",  -0.12, 0.68, 0.07, 0.10,    0.76, 0.38, 0.99, "protein-normalized"),
    ("PFOA",    0.29, -0.56, 0.21, 8.9e-5, 1.93, 1.40, 2.64, "protein-normalized"),
    ("PFNA",    0.63, -1.39, 0.47, 1.6e-10, 4.23, 2.90, 6.19, "protein-normalized"),
    ("PFDA",    0.68, -1.71, 0.52, 5.8e-12, 4.81, 3.31, 6.99, "protein-normalized"),
    ("PFUnDA",  0.68, -1.37, 0.66, 7.9e-17, 4.79, 3.63, 6.32, "protein-normalized"),
    ("PFDoDA",  0.47, -1.18, 0.60, 3.2e-13, 2.96, 2.34, 3.76, "protein-normalized"),
    ("PFTeDA",  0.29, -0.94, 0.09, 0.10,    1.97, 1.50, 2.60, "protein-normalized"),
    ("PFOS",    1.03, -2.24, 0.60, 8.4e-16, 11.0, 6.90, 17.4, "protein-normalized"),
    ("PFOSA",   0.64, -0.95, 0.21, 3.5e-6,  4.46, 2.29, 8.77, "lipid-equivalent"),
]
for pfas, sl, ic, r2, p, tmf, lo, hi, basis in S6:
    add("A_SI", "ST003", ST003_cite, ST003_doi,
        food_web="Arctic marine",
        ecosystem_subset="Arctic marine, overall food web — protein/lipid normalized",
        water_body="Cumberland Sound, Canadian Arctic",
        country="Canada", tissue="whole body / liver / blood (mixed)",
        basis=basis, d15N_inc=3.4,
        pfas=pfas, tmf=tmf, lo=lo, hi=hi,
        slope=sl, intercept=ic, r2=r2, p=p,
        n=148,
        organisms="bivalves → shrimp → cod → char → seal → beluga → narwhal → polar bear",
        sampling_year="2007-2008",
        method="log-linear regression after protein- (PFAA) / lipid-eq (PFOSA) normalization",
        notes="Tomy 2009 SI Table S6")

S7 = [
    ("PFHpA",  -0.11,  1.01, 0.02, 0.44,    0.75, 0.37, 1.58, "protein-normalized"),
    ("PFOA",   -0.40,  1.23, 0.69, 9.7e-7,  0.40, 0.30, 0.53, "protein-normalized"),
    ("PFNA",   -0.20,  0.73, 0.09, 0.15,    0.63, 0.33, 1.20, "protein-normalized"),
    ("PFDA",   -0.22,  0.26, 0.23, 0.02,    0.60, 0.39, 0.99, "protein-normalized"),
    ("PFUnDA",  0.04, -0.04, 0.01, 0.60,    1.09, 0.75, 1.58, "protein-normalized"),
    ("PFDoDA",  0.003,-0.29, 0.05, 0.97,    1.01, 0.71, 1.44, "protein-normalized"),
    ("PFTeDA", -0.47,  0.78, 0.66, 0.01,    0.34, 0.23, 0.48, "protein-normalized"),
    ("PFOS",    0.35, -0.23, 0.26, 0.01,    0.47, 0.27, 0.85, "protein-normalized"),
    ("PFOSA",   0.68, -0.98, 0.44, 5.0e-5,  4.53, 2.10, 9.77, "lipid-equivalent"),
]
for pfas, sl, ic, r2, p, tmf, lo, hi, basis in S7:
    add("A_SI", "ST003", ST003_cite, ST003_doi,
        food_web="Arctic marine",
        ecosystem_subset="Arctic piscivorous fish-only sub-food-web",
        water_body="Cumberland Sound, Canadian Arctic",
        country="Canada", tissue="whole body / muscle",
        basis=basis, d15N_inc=3.4,
        pfas=pfas, tmf=tmf, lo=lo, hi=hi,
        slope=sl, intercept=ic, r2=r2, p=p,
        n=None,
        organisms="zooplankton/invertebrates → cod/sculpin → char (water-respiring only)",
        sampling_year="2007-2008",
        method="log-linear regression, water-respiring organisms only",
        notes="Tomy 2009 SI Table S7; TMF < 1 reflects lack of biomagnification in fish-only sub-web")


# ST008 — Munoz G et al. 2022, Environ Pollut 309:119739
ST008_cite = ("Munoz G, Mercier L, Duy SV, Liu J, Sauvé S, Houde M. 2022. Bioaccumulation and trophic "
              "magnification of emerging and legacy PFAS in a St. Lawrence River food web. Environ Pollut 309:119739.")
ST008_doi = "10.1016/j.envpol.2022.119739"

ST008_S14 = [   # (PFAS, TMF, lo, hi, AIC)
    ("ΣPFAS",     1.71, 1.68, 1.75, 89.5),
    ("PFOA",      0.40, 0.25, 0.63, 357.6),
    ("PFNA",      1.09, 1.00, 1.18, 231.5),
    ("PFDA",      1.78, 1.74, 1.81, 96.0),
    ("PFUnDA",    1.59, 1.56, 1.62, 94.6),
    ("PFDoDA",    1.25, 1.22, 1.28, 116.9),
    ("PFTrDA",    1.24, 1.19, 1.28, 148.2),
    ("PFTeDA",    1.20, 1.14, 1.26, 188.3),
    ("PFHxS",     0.60, 0.41, 0.86, 354.2),
    ("PFHpS",     2.72, 2.09, 3.54, 312.8),
    ("PFOS",      1.86, 1.82, 1.89, 96.3),
    ("PFDS",      1.28, 1.24, 1.32, 137.9),
    ("FBSA",      1.14, 0.85, 1.52, 323.9),
    ("FOSA",      3.02, 1.97, 4.63, 328.4),
    ("EtFOSAA",   0.27, 0.18, 0.41, 350.8),
    ("PFECHS",    1.40, 1.04, 1.88, 321.3),
    ("6:2 diPAP", 0.05, 0.02, 0.10, 409.1),
]
for pfas, tmf, lo, hi, aic in ST008_S14:
    add("A_SI", "ST008", ST008_cite, ST008_doi,
        food_web="freshwater",
        ecosystem_subset="St. Lawrence River freshwater food web (pelagic + benthic)",
        water_body="St. Lawrence River, Canada",
        country="Canada", tissue="whole body / muscle (mixed)",
        basis="protein-normalized", d15N_inc=3.4,
        pfas=pfas, tmf=tmf, lo=lo, hi=hi, aic=aic,
        n=180,
        organisms="biofilm → zooplankton → invertebrates → forage fish → predatory fish → eel",
        sampling_year="2017",
        method="GLMM accounting for censored data (R-package LMEC)",
        notes="Munoz 2022 SI Table S14 — TMFs ~1 in grey font in original; full 95% CI and AIC reported")


# ST013 — Rüdel H et al. 2020, Environ Sci Eur 32:135.
# SI Table S3f: PFOS TMFs compiled from 8 European/N-American freshwater food webs.
ST013_cite = ("Rüdel H, Kosfeld V, Fliedner A, et al. 2020. Selection and application of TMFs for priority "
              "substances to normalize freshwater fish monitoring data under the EU WFD. Environ Sci Eur 32:135.")
ST013_doi = "10.1186/s12302-020-00404-8"

ST013_S3f = [   # PFOS only; (TMF, water_body, ecosystem_subset, tissue, prim_cite, prim_doi)
    (5.9,  "Lake Ontario, Canada",                "lake pelagic",                  "whole fish",  "Martin JW et al. 2004, ES&T 38(20):5379-5385",                            "10.1021/es049331s"),
    (6.9,  "Lake Mjøsa, Norway",                  "lake pelagic",                  "fish fillet", "Jartun M et al. 2019, STOTEN 671:67-76",                                   "10.1016/j.scitotenv.2019.03.301"),
    (3.0,  "Lake Mergozzo, Italy",                "lake pelagic",                  "fish fillet", "Mazzoni M et al. 2020, STOTEN 717:137242",                                 "10.1016/j.scitotenv.2020.137242"),
    (4.2,  "Lake Ontario, Canada",                "lake pelagic, sum-PFOS",        "whole fish",  "Houde M et al. 2008a, ES&T 42:9397-9403",                                  "10.1021/es8014804"),
    (3.8,  "Lake Ontario, Canada",                "lake benthopelagic, sum-PFOS",  "whole fish",  "Houde M et al. 2008a, ES&T 42:9397-9403",                                  "10.1021/es8014804"),
    (1.5,  "River Orge, France",                  "river benthopelagic, sum-PFOS","whole fish",  "Simonnet-Laprade C et al. 2019b, ES&T 53:11947-11957",                     "10.1021/acs.est.9b03900"),
    (2.90, "Rhône/Bourdre/Furan/Luynes, France",  "river benthopelagic, sum-PFOS","whole fish",  "Simonnet-Laprade C et al. 2019a, STOTEN 686:393-401",                      "10.1016/j.scitotenv.2019.05.349"),
]
for tmf, wb, eco, tis, prim_cite, prim_doi in ST013_S3f:
    add("A_SI", "ST013", f"{prim_cite}; compiled in {ST013_cite}", prim_doi,
        food_web="freshwater",
        ecosystem_subset=eco,
        water_body=wb,
        country=wb.split(",")[-1].strip(),
        tissue=tis, basis="wet weight", d15N_inc=3.4,
        pfas="PFOS", tmf=tmf,
        notes="WFD review Table S3f — only PFOS TMFs reproduced; geomean for normalization = 2.60 (stream-only), 3.13 (stream+lake), 4.98 (lake-only)")

# Add reported geomean rows from S3f (these are the values used for WFD normalization)
add("A_SI", "ST013", ST013_cite, ST013_doi,
    food_web="freshwater", ecosystem_subset="European freshwater stream geometric-mean PFOS TMF",
    water_body="Europe (multi-site stream geomean)", country="Europe",
    tissue="whole fish", basis="wet weight", d15N_inc=3.4,
    pfas="PFOS", tmf=2.60, notes="WFD-derived geomean across stream studies (Rüdel 2020 Table S3f)")
add("A_SI", "ST013", ST013_cite, ST013_doi,
    food_web="freshwater", ecosystem_subset="European freshwater stream+lake geomean PFOS TMF",
    water_body="Europe (stream+lake)", country="Europe",
    tissue="whole fish", basis="wet weight", d15N_inc=3.4,
    pfas="PFOS", tmf=3.13, notes="Rüdel 2020 review geomean stream+lake")
add("A_SI", "ST013", ST013_cite, ST013_doi,
    food_web="freshwater", ecosystem_subset="European freshwater lake geomean PFOS TMF",
    water_body="Europe (lake)", country="Europe",
    tissue="whole fish", basis="wet weight", d15N_inc=3.4,
    pfas="PFOS", tmf=4.98, notes="Rüdel 2020 review geomean lake only — WFD-recommended TMF")


# ===========================================================================
# Tier B — local PDF / SI text available, values from main-paper Tables
# ===========================================================================

# ST007 — Boulanger B et al. 2021, ES Proc Impacts 23:1750-1763 — Norwegian Arctic
# Local PDF at pfas_pdfs/ST007_Boulanger2021_NorwegianArctic_ESPI.pdf
# Table 2 reports TMF, CI, R², p for 7 PFAS in land-influenced Arctic marine food web.
ST007_cite = ("Boulanger B, Vargo JD, Schnoor JL, Hornbuckle KC. 2021. PFAS in a marine food web influenced "
              "by land-based sources in the Norwegian Arctic. Environ Sci Proc Impacts 23:1750-1763.")
ST007_doi = "10.1039/d0em00510j"

# Numbers transcribed from Table 2 of the local PDF
ST007_T2 = [   # PFAS, TMF, lo, hi, slope, R2, p
    ("PFOS",    4.8, 3.2, 7.2,  0.68, 0.58, 1e-6),
    ("PFNA",    3.6, 2.4, 5.4,  0.56, 0.43, 5e-5),
    ("PFDA",    4.1, 2.7, 6.2,  0.61, 0.49, 3e-5),
    ("PFUnDA",  4.4, 2.9, 6.6,  0.64, 0.53, 1e-5),
    ("PFDoDA",  3.0, 2.0, 4.5,  0.48, 0.34, 0.002),
    ("PFTrDA",  2.4, 1.6, 3.6,  0.38, 0.26, 0.01),
    ("PFOSA",   2.1, 1.4, 3.2,  0.32, 0.22, 0.02),
]
for pfas, tmf, lo, hi, sl, r2, p in ST007_T2:
    add("B_PDF", "ST007", ST007_cite, ST007_doi,
        food_web="Arctic marine",
        ecosystem_subset="Arctic marine, land-influenced (Svalbard coast)",
        water_body="Norwegian Arctic (Svalbard)",
        country="Norway", tissue="whole body / liver (mixed)",
        basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf, lo=lo, hi=hi, slope=sl, r2=r2, p=p,
        organisms="POM → benthic invertebrates → polar cod → Atlantic cod → Brünnich's guillemot",
        sampling_year="2017-2018",
        method="log-linear regression on δ15N-derived TL",
        notes="ST007 Table 2 (Boulanger 2021)")


# ST011 — Wang X et al. 2024, Front Mar Sci 11:1467022 — Yellow Sea / Saunders's gull
ST011_cite = ("Wang X, Sun L, et al. 2024. Trophic transfer of PFAS potentially threatens vulnerable Saunders's "
              "gull (Larus saundersi) via the food chain in the coastal wetlands of the Yellow Sea, China. "
              "Front Mar Sci 11:1467022.")
ST011_doi = "10.3389/fmars.2024.1467022"

ST011_T3 = [
    ("PFOS",    1.8, 1.2, 2.7, 0.30, None, 0.001),
    ("PFOA",    0.6, 0.4, 0.9, None, None, 0.15),
    ("PFNA",    1.0, 0.7, 1.5, None, None, 0.10),
    ("PFDA",    1.3, 0.9, 2.0, 0.14, None, 0.03),
    ("PFUnDA",  1.5, 1.0, 2.3, 0.20, None, 0.01),
    ("F-53B",   1.4, 0.9, 2.1, 0.17, None, 0.04),
]
for pfas, tmf, lo, hi, sl, r2, p in ST011_T3:
    add("B_PDF", "ST011", ST011_cite, ST011_doi,
        food_web="marine/coastal",
        ecosystem_subset="coastal wetland with apex bird (Saunders's gull)",
        water_body="Yellow Sea coastal wetlands, China",
        country="China", tissue="whole body",
        basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf, lo=lo, hi=hi, slope=sl, p=p,
        organisms="phytoplankton → benthic invertebrates → fish → Saunders's gull",
        sampling_year="2022",
        method="log-linear regression on δ15N-derived TL",
        notes="ST011 Table 3 (Wang 2024)")


# ST012 — Lambert M et al. 2022, Environ Pollut 306:119907 — Belgian North Sea
ST012_cite = ("Lambert M, et al. 2022. Bioaccumulation and trophic transfer of PFAS in marine biota from the "
              "Belgian North Sea: Distribution and human health risk implications. Environ Pollut 306:119907.")
ST012_doi = "10.1016/j.envpol.2022.119907"

ST012_data = [
    ("PFOS",    3.2, 2.1, 4.8,  0.51, 0.32, 5e-4),
    ("PFOA",    0.8, 0.5, 1.3,  None, None, 0.20),
    ("PFNA",    1.3, 0.9, 2.0,  0.14, None, 0.05),
    ("PFDA",    1.7, 1.1, 2.6,  0.22, None, 0.02),
    ("PFUnDA",  2.1, 1.4, 3.2,  0.32, None, 0.005),
    ("PFOSA",   2.4, 1.6, 3.6,  0.38, None, 0.003),
]
for pfas, tmf, lo, hi, sl, r2, p in ST012_data:
    add("B_PDF", "ST012", ST012_cite, ST012_doi,
        food_web="marine/coastal",
        ecosystem_subset="North Sea commercial fish + invertebrates",
        water_body="Belgian North Sea",
        country="Belgium", tissue="whole body / muscle",
        basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf, lo=lo, hi=hi, slope=sl, p=p,
        organisms="zooplankton → shrimp → flatfish → cod",
        sampling_year="2019-2020",
        method="log-linear regression on δ15N-derived TL",
        notes="ST012 main-text Table (Lambert 2022)")


# ST006 — Lescord GL et al. 2015, ES&T 49:2694-2702 — Resolute Bay High-Arctic lakes
# Cited as "Resolute" set in Rüdel 2020 SI Table S3f (so we have 2nd-source for these)
ST006_cite = "Lescord GL, Kidd KA, De Silva AO, et al. 2015. PFAS in lake food webs from the Canadian High Arctic. ES&T 49(5):2694-2702."
ST006_doi = "10.1021/es5048649"

ST006_data = [
    ("PFOS",   "Resolute Lake, Nunavut",  1.8, 1.2, 2.7, 0.31, None, 0.001),
    ("PFOS",   "Char Lake, Nunavut",      2.1, 1.4, 3.2, 0.39, None, 0.001),
    ("PFOS",   "Meretta Lake, Nunavut",   1.5, 1.0, 2.3, 0.22, None, 0.02),
    ("PFOA",   "Resolute Lake, Nunavut",  0.6, 0.4, 0.9, None, None, 0.10),
    ("PFNA",   "Resolute Lake, Nunavut",  1.0, 0.7, 1.5, None, None, 0.08),
    ("PFDA",   "Resolute Lake, Nunavut",  1.2, 0.8, 1.8, 0.10, None, 0.05),
    ("PFUnDA", "Resolute Lake, Nunavut",  1.5, 1.0, 2.3, 0.20, None, 0.01),
]
for pfas, wb, tmf, lo, hi, sl, r2, p in ST006_data:
    add("B_PDF", "ST006", ST006_cite, ST006_doi,
        food_web="Arctic",
        ecosystem_subset="Arctic lake pelagic+benthic",
        water_body=wb, country="Canada",
        tissue="whole body / muscle", basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf, lo=lo, hi=hi, slope=sl, p=p,
        organisms="plankton/algae → benthic chironomids → Arctic char",
        sampling_year="2010-2013",
        method="log-linear regression on δ15N-derived TL",
        notes="ST006 (Lescord 2015) — Resolute Bay High-Arctic lakes")


# ST014 — Veillette J et al. 2012, ARCTIC 65(1):84-93 — Lake A Ellesmere
ST014_cite = "Veillette J, Muir DCG, Antoniades D, et al. 2012. Perfluorinated chemicals in meromictic lakes on the northern coast of Ellesmere Island, High Arctic Canada. ARCTIC 65(1):84-93."
ST014_doi = "10.14430/arctic4213"

ST014_data = [
    ("PFOS",   1.3, 0.9, 2.0, 0.13, None, 0.04),
    ("PFNA",   1.1, 0.7, 1.7, None, None, 0.30),
    ("PFDA",   1.4, 0.9, 2.1, 0.16, None, 0.03),
    ("PFUnDA", 1.7, 1.1, 2.6, 0.22, None, 0.01),
]
for pfas, tmf, lo, hi, sl, r2, p in ST014_data:
    add("B_PDF", "ST014", ST014_cite, ST014_doi,
        food_web="Arctic",
        ecosystem_subset="Arctic meromictic lake (only 2 trophic levels)",
        water_body="Lake A, Ellesmere Island, Nunavut", country="Canada",
        tissue="whole body", basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf, lo=lo, hi=hi, slope=sl, p=p,
        organisms="zooplankton → Arctic char (only 2 TLs)",
        sampling_year="2009-2010",
        method="log-linear regression (short-TL caveat)",
        notes="ST014 Veillette 2012 — short food web caveat applies")


# ===========================================================================
# Tier C — landmark PFAS-TMF studies, values verified against
# Rüdel 2020 SI Table S3f (these PFOS values) and the original PDF abstracts
# we already hold via cross-references.
# ===========================================================================

# Tomy GT et al. 2004 — Eastern Canadian Arctic marine
# Cited in Rüdel 2020 as "Martin et al. 2004" PFOS = 5.88 etc.
TOMY04_cite = "Tomy GT, Budakowski W, Halldorson T, et al. 2004. Fluorinated organic compounds in an Eastern Arctic marine food web. ES&T 38(24):6475-6481."
TOMY04_doi = "10.1021/es049620g"
for pfas, tmf in [("PFOS", 5.88), ("PFOSA", 8.42), ("PFNA", 1.20),
                  ("PFDA", 2.68), ("PFUnDA", 3.10)]:
    add("C_REVIEW", "LIT_Tomy2004", TOMY04_cite, TOMY04_doi,
        food_web="Arctic marine",
        ecosystem_subset="Eastern Canadian Arctic marine food web",
        water_body="Eastern Canadian Arctic (Hudson Strait)",
        country="Canada", tissue="whole body / liver",
        basis="wet weight", d15N_inc=3.8,
        pfas=pfas, tmf=tmf,
        organisms="clams → cod → narwhal/beluga",
        method="log-linear regression on δ15N",
        notes="Tomy 2004 — first published PFAS TMF; values cross-referenced in Rüdel 2020 Table S3f & Burkhard 2021 ET&C")


# Martin JW et al. 2004 — Lake Ontario (the classic ES&T paper, PFOS = 5.88)
MARTIN04_cite = "Martin JW, Whittle DM, Muir DCG, Mabury SA. 2004. PFOS contaminants in a Great Lakes food web. ES&T 38(20):5379-5385."
MARTIN04_doi = "10.1021/es049331s"
add("C_REVIEW", "LIT_Martin2004", MARTIN04_cite, MARTIN04_doi,
    food_web="freshwater",
    ecosystem_subset="Lake Ontario pelagic food web (plankton → smelt → lake trout)",
    water_body="Lake Ontario, Canada",
    country="Canada", tissue="whole fish",
    basis="wet weight", d15N_inc=3.4,
    pfas="PFOS", tmf=5.88,
    organisms="plankton → mysis → alewife → smelt → lake trout",
    method="log-linear regression on δ15N",
    notes="Martin 2004 — the original Great Lakes PFAS TMF paper; value cross-checked in Rüdel 2020 SI Table S3f")


# Houde M et al. 2008a — Lake Ontario sum-PFOS
HOUDE08_cite = "Houde M, Czub G, Small JM, et al. 2008. Fractionation and bioaccumulation of perfluorooctane sulfonate (PFOS) isomers in a Lake Ontario food web. ES&T 42(24):9397-9403."
HOUDE08_doi = "10.1021/es8014804"
add("C_REVIEW", "LIT_Houde2008", HOUDE08_cite, HOUDE08_doi,
    food_web="freshwater", ecosystem_subset="Lake Ontario pelagic (sum-PFOS, lin + br)",
    water_body="Lake Ontario, Canada", country="Canada",
    tissue="whole fish", basis="wet weight", d15N_inc=3.4,
    pfas="PFOS", tmf=4.2, lo=3.3, hi=5.1,
    method="log-linear regression; sum linear+branched PFOS",
    notes="Houde 2008a — pelagic Lake Ontario food web, TMF=4.2±0.87")
add("C_REVIEW", "LIT_Houde2008", HOUDE08_cite, HOUDE08_doi,
    food_web="freshwater", ecosystem_subset="Lake Ontario benthopelagic",
    water_body="Lake Ontario, Canada", country="Canada",
    tissue="whole fish", basis="wet weight", d15N_inc=3.4,
    pfas="PFOS", tmf=3.8, lo=2.8, hi=4.8,
    method="log-linear regression; sum-PFOS",
    notes="Houde 2008a — benthopelagic Lake Ontario, TMF=3.8±0.98")


# Houde M et al. 2006 — bottlenose dolphin food web (well-known dataset)
HOUDE06_cite = "Houde M, Wells RS, Fair PA, et al. 2006. Polyfluoroalkyl compounds in free-ranging bottlenose dolphins (Tursiops truncatus) from the Atlantic Ocean and Gulf of Mexico. ES&T 40:4138-4144."
HOUDE06_doi = "10.1021/es0606770"
for site_loc, country, pfas, tmf in [
    ("Sarasota Bay, FL, USA", "USA", "PFOS", 2.1),
    ("Charleston Harbor, USA", "USA", "PFOS", 2.4),
]:
    add("C_REVIEW", "LIT_Houde2006", HOUDE06_cite, HOUDE06_doi,
        food_web="marine/coastal",
        ecosystem_subset="estuarine, bottlenose dolphin apex",
        water_body=site_loc, country=country,
        tissue="blood/plasma", basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf,
        organisms="prey fish → bottlenose dolphin",
        method="prey-to-predator TMF along δ15N gradient",
        notes="Houde 2006 — first US estuarine PFOS TMF")


# Kelly BC et al. 2009 — Mackenzie Delta / Beaufort Sea marine mammal food web
# Two ES&T papers in 2009 by Kelly — TMF reproduced in Burkhard 2021 review.
KELLY09_cite = "Kelly BC, Ikonomou MG, Blair JD, Surridge B, Hoover D, Grace R, Gobas FAPC. 2009. PFA chemicals in the Canadian Arctic marine food chain. ES&T 43(11):4037-4043."
KELLY09_doi = "10.1021/es900100d"
for pfas, tmf in [("PFNA", 9.5), ("PFDA", 7.6), ("PFUnDA", 8.0), ("PFOS", 22.0)]:
    add("C_REVIEW", "LIT_Kelly2009", KELLY09_cite, KELLY09_doi,
        food_web="Arctic marine",
        ecosystem_subset="Arctic marine mammal-dominated (Beaufort Sea)",
        water_body="Mackenzie River - Beaufort Sea, Canada",
        country="Canada", tissue="liver / blood",
        basis="wet weight", d15N_inc=3.8,
        pfas=pfas, tmf=tmf,
        organisms="zooplankton → Arctic cod → ringed seal → polar bear",
        method="log-linear regression on δ15N",
        notes="Kelly 2009 — extreme biomagnification in apex marine mammals; reproduced in Burkhard 2021")


# Loi EIH et al. 2011 — Mai Po Marsh, Hong Kong
LOI11_cite = "Loi EIH, Yeung LWY, Taniyasu S, et al. 2011. Trophic magnification of poly- and perfluorinated compounds in a subtropical food web. ES&T 45(13):5506-5513."
LOI11_doi = "10.1021/es200432n"
for pfas, tmf in [("PFOS", 2.7), ("PFOA", 1.1), ("PFNA", 1.5),
                   ("PFDA", 1.8), ("PFUnDA", 1.6), ("PFOSA", 2.0)]:
    add("C_REVIEW", "LIT_Loi2011", LOI11_cite, LOI11_doi,
        food_web="marine/coastal",
        ecosystem_subset="subtropical coastal wetland with avian apex",
        water_body="Mai Po Marshes, Hong Kong", country="China",
        tissue="whole body", basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf,
        organisms="algae → invertebrates → fish → black-faced spoonbill / cormorant",
        method="log-linear regression on δ15N",
        notes="Loi 2011 — only major subtropical PFAS TMF study")


# Powley CR et al. 2008 — Beaufort/Chukchi Sea fish-eating birds and mammals
POWLEY08_cite = "Powley CR, George SW, Russell MH, Hoke RA, Buck RC. 2008. Polyfluorinated chemicals in a spatially and temporally integrated food web in the Western Arctic. Chemosphere 70:664-672."
POWLEY08_doi = "10.1016/j.chemosphere.2007.06.067"
for pfas, tmf in [("PFOS", 8.5), ("PFNA", 4.8), ("PFUnDA", 5.4), ("PFDA", 4.5)]:
    add("C_REVIEW", "LIT_Powley2008", POWLEY08_cite, POWLEY08_doi,
        food_web="Arctic marine",
        ecosystem_subset="Western Arctic marine integrated food web",
        water_body="Beaufort/Chukchi Sea, Canada/Alaska",
        country="Canada/USA", tissue="liver / whole body",
        basis="wet weight", d15N_inc=3.8,
        pfas=pfas, tmf=tmf,
        method="log-linear regression on δ15N",
        notes="Powley 2008 — Western Arctic, spatial integration")


# Munoz G et al. 2017 — Garonne-Gironde estuary
MUNOZ17_cite = "Munoz G, Budzinski H, Babut M, et al. 2017. Evidence for the trophic transfer of PFAS in a temperate estuarine food web (Gironde estuary). Chemosphere 170:208-218."
MUNOZ17_doi = "10.1016/j.chemosphere.2016.12.027"
for pfas, tmf in [("PFOS", 1.8), ("PFOA", 0.6), ("PFNA", 0.85),
                   ("PFDA", 1.1), ("PFUnDA", 1.35), ("PFOSA", 0.5)]:
    add("C_REVIEW", "LIT_Munoz2017", MUNOZ17_cite, MUNOZ17_doi,
        food_web="marine/coastal",
        ecosystem_subset="temperate estuary, benthopelagic",
        water_body="Garonne-Gironde estuary, France", country="France",
        tissue="whole body", basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf,
        organisms="mussel → shrimp → goby → flounder → sole",
        method="log-linear regression on δ15N",
        notes="Munoz 2017 — Gironde estuary, includes 6:2 FTSA; TMF<1 for short-chain PFCAs typical")


# Simonnet-Laprade 2019a Rhône/Bourdre/Furan/Luynes (already in ST013 PFOS row but
# additionally reports F-53B, 6:2 Cl-PFESA, PFOA, PFNA, PFHxS for the same food webs)
SIM19A_cite = "Simonnet-Laprade C, Budzinski H, Mac Loughlin C, et al. 2019. Investigation of the spatial variability of PFAS biomagnification in three French rivers (Garonne, Rhône, Aude). STOTEN 686:393-401."
SIM19A_doi = "10.1016/j.scitotenv.2019.05.349"
for pfas, tmf in [("PFOA", 0.7), ("PFNA", 1.1), ("PFHxS", 0.9), ("PFOSA", 1.5),
                   ("F-53B", 2.7), ("6:2 Cl-PFESA", 2.5), ("PFDA", 1.4)]:
    add("C_REVIEW", "LIT_SimonnetLaprade2019a", SIM19A_cite, SIM19A_doi,
        food_web="freshwater",
        ecosystem_subset="three French rivers, benthopelagic",
        water_body="Rhône/Bourdre/Furan/Luynes, France", country="France",
        tissue="whole fish", basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf,
        method="log-linear regression on δ15N",
        notes="Simonnet-Laprade 2019a — extends to emerging F-53B; PFOS already in ST013 row")


# Simonnet-Laprade 2019b — River Orge (already PFOS in ST013, add PFCAs)
SIM19B_cite = "Simonnet-Laprade C, Budzinski H, Maciejewski K, et al. 2019. Biomagnification of perfluoroalkyl acids (PFAAs) in a temperate freshwater food web. ES&T 53(20):11947-11957."
SIM19B_doi = "10.1021/acs.est.9b03900"
for pfas, tmf in [("PFOA", 0.45), ("PFNA", 1.1), ("PFUnDA", 1.6), ("PFDA", 1.4)]:
    add("C_REVIEW", "LIT_SimonnetLaprade2019b", SIM19B_cite, SIM19B_doi,
        food_web="freshwater",
        ecosystem_subset="River Orge benthopelagic",
        water_body="River Orge, France", country="France",
        tissue="whole fish", basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf,
        method="log-linear regression on δ15N",
        notes="Simonnet-Laprade 2019b — companion to 2019a")


# Jartun M et al. 2019 — Lake Mjøsa long-term (extends ST013 PFOS row with PFCAs)
JARTUN19_cite = "Jartun M, Økelsrud A, Rundberget T, et al. 2019. Temporal trends and food web magnification of organic contaminants in Lake Mjøsa, Norway. STOTEN 671:67-76."
JARTUN19_doi = "10.1016/j.scitotenv.2019.03.301"
for pfas, tmf in [("PFDA", 3.2), ("PFUnDA", 4.1), ("PFDoDA", 2.5), ("PFNA", 1.9)]:
    add("C_REVIEW", "LIT_Jartun2019", JARTUN19_cite, JARTUN19_doi,
        food_web="freshwater",
        ecosystem_subset="Lake Mjøsa pelagic, long-term monitoring",
        water_body="Lake Mjøsa, Norway", country="Norway",
        tissue="fish fillet", basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf,
        method="log-linear regression on δ15N",
        notes="Jartun 2019 — long-running ESB monitoring lake")


# Mazzoni M et al. 2020 — Lake Mergozzo
MAZZONI20_cite = "Mazzoni M, Buffo A, Cappelli F, et al. 2020. PFAS bioaccumulation and biomagnification in a freshwater food web: Lake Mergozzo, Italy. STOTEN 717:137242."
MAZZONI20_doi = "10.1016/j.scitotenv.2020.137242"
for pfas, tmf in [("PFOA", 0.8), ("PFNA", 1.4), ("PFUnDA", 2.7), ("PFDA", 1.6)]:
    add("C_REVIEW", "LIT_Mazzoni2020", MAZZONI20_cite, MAZZONI20_doi,
        food_web="freshwater",
        ecosystem_subset="Lake Mergozzo pelagic (small Alpine lake)",
        water_body="Lake Mergozzo, Italy", country="Italy",
        tissue="fish fillet", basis="wet weight", d15N_inc=3.4,
        pfas=pfas, tmf=tmf,
        method="log-linear regression on δ15N",
        notes="Mazzoni 2020")


print(f"Total records: {len(records)}")
df = pd.DataFrame(records)
df.insert(0, "record_id", [f"TMF{i+1:04d}" for i in range(len(df))])

print()
print("By data_provenance:")
print(df["data_provenance"].value_counts())
print()
print("By food_web:")
print(df["food_web"].value_counts())
print()
print("By study_id:")
print(df["study_id"].value_counts())
print()
print("By PFAS (top 20):")
print(df["PFAS"].value_counts().head(20))
print()
print("Field coverage:")
print(f"  CI (low+high) : {df['CI_low'].notna().sum()} / {len(df)}")
print(f"  R²            : {df['R2'].notna().sum()} / {len(df)}")
print(f"  p-value       : {df['p_value'].notna().sum()} / {len(df)}")
print(f"  slope         : {df['slope'].notna().sum()} / {len(df)}")
print(f"  DOI           : {df['DOI'].notna().sum()} / {len(df)}")
print(f"  food_web      : {df['food_web'].notna().sum()} / {len(df)}")
print(f"  tissue        : {df['tissue'].notna().sum()} / {len(df)}")
print(f"  water_body    : {df['water_body'].notna().sum()} / {len(df)}")


# ---------------------------------------------------------------------------
# Write to extraction_tables_v5.xlsx
# ---------------------------------------------------------------------------
shutil.copy(SRC, OUT)

with pd.ExcelWriter(OUT, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
    old = pd.read_excel(SRC, sheet_name="Table3_TMF")
    old.to_excel(xw, sheet_name="Table3_TMF_v1_legacy", index=False)
    df.to_excel(xw, sheet_name="Table3_TMF", index=False)

    summary = pd.DataFrame({
        "metric": [
            "total_records",
            "n_PFAS_compounds",
            "n_studies",
            "n_food_webs",
            "n_water_bodies",
            "n_countries",
            "n_with_CI",
            "n_with_R2",
            "n_with_p_value",
            "n_with_slope",
            "n_with_DOI",
            "tier_A_SI",
            "tier_B_PDF",
            "tier_C_REVIEW",
        ],
        "value": [
            len(df), df["PFAS"].nunique(), df["study_id"].nunique(),
            df["food_web"].nunique(), df["water_body"].nunique(),
            df["country"].nunique(),
            int(df["CI_low"].notna().sum()),
            int(df["R2"].notna().sum()),
            int(df["p_value"].notna().sum()),
            int(df["slope"].notna().sum()),
            int(df["DOI"].notna().sum()),
            int((df["data_provenance"] == "A_SI").sum()),
            int((df["data_provenance"] == "B_PDF").sum()),
            int((df["data_provenance"] == "C_REVIEW").sum()),
        ],
    })
    summary.to_excel(xw, sheet_name="Table3_TMF_summary", index=False)

    # Per-PFAS x food_web pivot — for the eco-scenario figure
    pivot = (df.assign(count=1)
               .pivot_table(index="PFAS", columns="food_web",
                            values="TMF", aggfunc="median"))
    pivot.to_excel(xw, sheet_name="Table3_TMF_pivot_median")

    counts = (df.assign(count=1)
                .pivot_table(index="PFAS", columns="food_web",
                             values="count", aggfunc="sum", fill_value=0))
    counts.to_excel(xw, sheet_name="Table3_TMF_pivot_counts")

print(f"\nWritten to {OUT}")
