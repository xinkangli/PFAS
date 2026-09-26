# EST revision reproducibility package

This directory is the corrected analysis accompanying es-2026-11771j. It preserves 216 source-qualified biological observations (19 parent compounds, six studies), the exclusion ledger, fixed partitions, property inputs, molecular arrays, predictions, and model code. Legacy code elsewhere in this repository is retained for historical traceability; use this directory for the revised results.

## Reproduce reported metrics without a GPU

From this directory, with Python 3.10 or 3.11:

```sh
python -m pip install -r requirements_metrics.txt
python reproduce_metrics.py
```

Expected: 132 metric groups verified at absolute tolerance 1e-10. The calculation reads supplied predictions and writes `reproduced/verification.json`. It does not fit a model.

## Refit models from supplied data

Use Linux, Python 3.8.20, and `requirements_training.txt`. The executed reference environment used PyTorch 1.13.1+cu117; install the CUDA 11.7 wheel of that version for the matching GPU environment. Training automatically selects CUDA when available, otherwise CPU; `PFAS_DEVICE=cpu` forces CPU. File locking uses Linux fcntl. CPU execution may be much slower. No server credentials or private source directories are needed by these training entry points.

Create a fresh destination so archived predictions cannot cause fitting to be skipped:

```sh
python new_training_run.py ../fresh_training
cd ../fresh_training
python run_final.py --workers 1 --limit 1
```

That working example runs one complete frozen chemical holdout, including upstream pretraining, auxiliary fitting, neural ablations, and conventional controls. It keeps the same epoch counts as the reported analysis. For all results, continue in that new directory:

```sh
python run_final.py --workers 1
python run_similarity.py --workers 1
python external.py
python external_cv.py
python decision.py
python summarize.py
python paired_joint.py
python validate_checkpoints.py
```

The fixed schedules are 40 pretraining, 400 auxiliary, and 120 downstream epochs, with seeds 42–44; there is no grid-search or test-set early stopping. Training reruns generate their own checkpoints, audit JSON, predictions and summaries. Small floating-point differences can occur across devices. The archived OPERA output is supplied for comparison, rather than pretending that OPERA is run by these commands.

## Provenance and search

`data/README.md` defines field precedence. `data/primary_216_only.csv` is the admitted export; `data/final_observations.csv` preserves the 1,078-row order required by the molecular arrays. `source_qualification/audit/all_1078_disposition.csv` records inclusion/exclusion reasons and locators. The 696 unverified source values and 166 source-located but ineligible observations are excluded from primary training.

`literature_search/` documents the six PubMed query expressions recovered from the original collector, and provides a runnable, dated retrieval script. PubMed candidate retrieval is distinct from extracting experimental values. The historical biological database was extracted by scripts and checked by Xinkang Li. A fresh search is not represented as the original historical search, nor as new admitted training data.

`summary/` contains numerical figure inputs. `audit/` retains normalization and checkpoint checks from the complete corrected reference fit. `publication_checks/` records independent tests of this portable release. `historical_audit/` and `source_qualification/code/` preserve historical audits; those historical scripts may refer to their original input directories and are not the portable training entry points.

## Interpretation and access

Data and code are openly downloadable from this repository. The commit identifier fixes the exact released version. Third-party publications and complete provider workbooks are linked by DOI/URL and source locator; they are not redistributed here as full-text articles. No Zenodo DOI is asserted. Nineteen development chemicals and one development BCF study limit inference. External panels are retrospective; no validated regulatory P/B/T classification, prospective experiment, mechanistic proof, or active-learning efficiency is claimed.
