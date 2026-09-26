# Observation fields and precedence

The tables preserve historical fields for lineage; old status columns are not silently overwritten.

- `record_id`: stable observation key used by partitions and predictions.
- `compound_id`: archived standardized parent-group identity used for grouping.
- `endpoint_type`, `log10_value_L_kg`: the current endpoint and numerical modeling target.
- `v15_primary`: final admission mask for this corrected run; exactly 216 true rows, 19 compounds, six studies.
- `v15_status`, `v15_locator`: final source qualification decision and source locator.
- `v8_source_value`, `v8_source_scale`, `v8_locator`, `v8_source_url`, `v8_source_sha256`: recovered original-source evidence when supplied.
- `v9_raw_workbook`, `v9_sheet`, `v9_row`, `v9_cells`, `v9_raw_cells`: database-workbook reconstruction fields, distinct from original-paper locations.
- `study_id`, `doi_normalized`: current source grouping and DOI.
- `historical_log10`, `original_log_discrepancy`, `paper_verification_status`, `provenance_level`, `provenance_gap`, and earlier eligibility flags document earlier stages. In particular, an early "not verified" field must not be mistaken for the later admission decision. Consult `v15_primary` and `v15_locator` and the final disposition ledger together.
- `source_excel_row`: historical modeling-workbook position, not necessarily the original ITRC sheet row.

`primary_216_only.csv` is the final admitted export. `final_observations.csv` retains the full 1,078-row positional order used by the feature arrays. The final mutually exclusive dispositions and the raw-cell checks are also recorded in `../source_qualification/audit/all_1078_disposition.csv`.
