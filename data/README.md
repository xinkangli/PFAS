# Data sources and schema

## Biological observations

`raw/bioaccumulation_workbook.xlsx` is the original 2,404-row biological compilation: 1,381 BAF, 880 BCF and 143 BCFD records. Compilation does not mean every record is an independently qualified original measurement. Database extraction used scripts and was checked by Xinkang Li, not two independent extractors.

The sequential screens retain 2,343 parsable structures, 1,925 positive values with confirmed L/kg conversion, 1,209 wet-weight observations, and 1,078 after the initial censoring-note screen. `sources/historical_training_filter_counts.csv` records these counts. All 1,078 screened records trace to the ITRC aquatic compilation; ECOTOX contributed to the broader historical collection.

`final_observations.csv` retains those 1,078 rows in the order used by the feature arrays. Only `included == True` enters primary fitting: **216 observations, 19 parent compounds, six studies**. This comprises 147 BAF observations (17 compounds, five studies) and 69 BCF observations (12 compounds, one study). Compound counts overlap across endpoints. BCF values retain their original tissue-specific log10 targets.

| Field | Meaning |
|---|---|
| `record_id` | Stable observation key used by fixed partitions |
| `compound_id`, `group_smiles` | Standardized parent identity used for grouping |
| `SMILES`, `canonical_smiles` | Supplied molecular representation and exact canonical identity |
| `study_id`, `doi_normalized` | Study grouping and primary-source DOI |
| `endpoint_type`, `log10_value_L_kg` | Modeling endpoint and log10 target in L/kg |
| `included`, `qualification_status`, `source_locator` | Final admission decision and original source locator |
| `original_source_value`, `original_source_scale`, `original_source_url` | Recovered primary numerical evidence where available |
| `provider_sheet`, `provider_row`, `provider_cells`, `provider_raw_cells` | ITRC workbook-cell reconstruction |
| `source_excel_row` | Row in the original modeling workbook, not necessarily the ITRC row |
| `organism_group`, `tissue`, `exposure_route`, `water_type`, `study_type`, `exposure_days` | Training-only encoded biological context |

`sources/all_1078_disposition.csv` contains mutually exclusive decisions: 216 admitted; 696 database-traceable records without qualified original values; 166 source-located but ineligible records. The latter comprise 68 censored summaries, 46 reconstructed whole-body values, 24 plasma-unit mismatches, 13 value/basis conflicts, seven LOQ substitutions, six isomer sums and two alternative estimates. Exclusion is not a declaration that a source experiment is erroneous. No response averaging is used to collapse the 216 admitted observations.

The six admitted source DOIs and counts are:

| DOI | Endpoint | Observations |
|---|---|---:|
| 10.1021/acs.est.5b04299 | BAF | 40 |
| 10.1021/acs.est.6b01197 | BAF | 27 |
| 10.1021/acs.est.7b02259 | BAF | 36 |
| 10.1021/acs.est.7b02399 | BAF | 42 |
| 10.1021/acs.est.9b04906 | BAF | 2 |
| 10.1016/j.scitotenv.2016.05.215 | BCF | 69 |

## Transport inputs

`raw/transport_workbook.xlsx` preserves the original upstream and auxiliary compilation. `silver_training.csv` supplies 62,714 rows representing 62,624 structures with prediction-derived logKow labels attributed to PubChem XLogP3. This is a mixed chemical population, not a PFAS-only experimental dataset. Structure-provider strings are distinct from label provenance.

`auxiliary_wide.csv` has 24 usable compounds and seven properties. `auxiliary_original_long.csv` retains all 200 original endpoint slots for 25 named substances: 168 filled, including 144 prediction-labelled and 24 measurement-flagged values. A measurement flag is not independent verification of an experiment. Provider labels include PubChem XLogP3, NORMAN SusDat/EPI Suite Koc, HENRYWIN, MPBPVP, WSKOW, literature-labelled and homolog-derived values. Empty or abbreviated source-reference fields remain explicit.

Twenty-four auxiliary logKoc cells summarize predicted ranges by the log10 of their geometric mean; only PFOS, PFOA and PFBS have unequal bounds. This does not average biological responses. Original flags and bounds are retained in the source workbook/metadata.

## Features and partitions

`features_graph_corrected.npz` contains `bio` (1,078 × 1,057), `silver` (62,714 × 1,057), and `gold` (24 × 1,057) in the corresponding CSV row order. `gold` is an array key, not a claim of experimental gold-standard quality. `python run.py features` recomputes all arrays from SMILES, including the five graph-derived F counts.

`final_jobs.json` fixes 15 chemical jobs (five folds for seeds 42–44) and six study holdouts. `similarity_jobs.json` fixes seven similarity-component holdouts at threshold 0.70. Jobs list train/test observation IDs and excluded upstream identities. Preserve record keys and array row order when adapting the data.

## External inputs

`external_panel.csv` contains 283 unique observation IDs across four endpoint-specific retrospective panels: Pickard field BAF (181 observations, 19 compounds; DOI 10.1021/acs.est.4c07016), Sims seven-day aqueous-exposure ratios (57, 19; DOI 10.1093/etojnl/vgaf098), Yao liver BCF (18, 18; DOI 10.1021/acs.est.4c13813), and Hayman whole-body BCF (27, 9; DOI 10.1093/etojnl/vgag168). Compound identities overlap across panels. Source independence and chemical independence are evaluated separately; these are not prospective tests. Short-duration ratios, BAF and tissue-specific/whole-body BCF are not interchangeable.

`opera_source_matched_records.csv` is an archived external-tool comparison input, not a fitted output of this package. The full summary uses its 27 whole-body BCF records for nine compounds. The workflow does not run OPERA; it compares the supplied OPERA 2.9.5 outputs and preserves their applicability flags.

## Provider links and optional literature retrieval

- EPA ECOTOX: https://cfpub.epa.gov/ecotox/
- ITRC PFAS: https://pfas-1.itrcweb.org/
- ITRC aquatic workbook: https://pfas-1.itrcweb.org/wp-content/uploads/2023/10/ITRC_PFAS_-BCF-BAF_compilation_Table5-1_Oct2021.xlsx
- PubChem: https://pubchem.ncbi.nlm.nih.gov/
- NORMAN SusDat: https://www.norman-network.com/nds/susdat/
- EPA CompTox: https://comptox.epa.gov/dashboard/

Resource links identify providers; DOI and workbook-cell locators identify available record-level evidence. `sources/literature_queries.json` contains the six exact PubMed query expressions. Run a dated candidate retrieval independently of numerical extraction:

```bash
python code/retrieve_literature.py --output outputs/literature_search
```

It saves query expressions, counts, IDs and raw ESearch responses. A new retrieval can differ as PubMed changes; it does not add qualified observations automatically. Full-text publications are not included. `SHA256.json` verifies input files from the repository root.
