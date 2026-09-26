# Evidence-based manuscript correction draft

These passages describe the current audited files and completed diagnostic analyses. They are working replacement text, not a submission-ready manuscript. They must not be inserted alongside unchanged conflicting tables or claims. The original paper.docx has not been overwritten.

## Dataset size and provenance

The archived bioaccumulation input contains 2,404 records, comprising 1,381 BAF, 880 BCF and 143 BCFD records. Of these, 61 records have no assigned SMILES. The remaining records represent 28 distinct non-empty SMILES and 28 parent-structure groups under the identity procedure used in the present audit. The previously stated total of 847 compounds is not supported by these input files and should be removed unless a separate, traceable dataset can be identified.

The transport reference table comprises 200 endpoint slots for 25 named PFAS, of which 168 contain values. Its provenance fields identify 144 values as predicted and 24 as measured; 32 slots are missing. We therefore distinguish software-estimated auxiliary labels from entries reported as measurements, rather than referring to the whole table as an experimental gold standard. The measurement designations still require verification against their cited sources.

## Identity and unit handling for the revised diagnostic analysis

Original structures and measurement fields were retained. For grouping, molecular structures were parsed with RDKit 2023.09.6; exact isomeric canonical SMILES were retained, while a separate grouping identifier was generated from the largest carbon-containing fragment after uncharging and non-isomeric canonicalization. This conservative grouping rule is intended to prevent related identities from appearing on both sides of a split. It does not establish equivalence of salt forms, stereoisomers or measured properties under different conditions.

Explicit L/kg, ml/g, L/g and L/mg values were converted to L/kg before log10 transformation. Unresolved ratio units, percentage units and ambiguous inverse units were flagged for source review. Dry-weight and wet-weight measurements were not assumed interchangeable. A provisional wet-weight cohort further excluded records with missing wet-weight designation or notes indicating detection-limit treatment. These exclusions constitute an auditable conservative diagnostic policy and require source-level review before a final dataset is frozen.

## Revised internal evaluation

Four structure-based reference models were evaluated: a training-median predictor, ridge regression, random forest and extra trees. Compound groups were assigned to five fixed outer folds for each of five random seeds. All records of a compound were retained on one side of each split. Imputation and scaling, where applicable, were fitted using only the corresponding training data. Model settings were fixed before evaluating these splits; the outer test data were not used for hyperparameter selection or early stopping.

Performance summaries use the mean out-of-fold prediction for each record across seeds. Compound-cluster bootstrap intervals and paired differences in absolute error are reported. These analyses evaluate a provisional dataset and do not remove study-level dependence. They also do not establish the incremental value of the original transport-informed multitask model, which requires a separate evaluation under the corrected protocol.

## Reinterpretation of the historical OOD results

Reconstruction of the historical random split showed that all 351 test records belonged to compounds represented in training. The three historical phylogeny-out tests contained 50 records from compounds also represented in training. The logs show actual species holdouts rather than the fallback split, but the tests did not simultaneously hold out compounds. Their high correlation therefore does not establish generalization to new PFAS or demonstrate evolutionarily conserved mechanisms.

The different holdout designs represent distinct changes in data composition and should not be interpreted as a universal ordering of increasing OOD difficulty. Study composition, response range, sample size and compound overlap must be reported alongside each comparison.

## Reinterpretation of external evidence

The historical direct BAF comparison contains 20 observations from five compounds, all of which overlap the bioaccumulation training set at the compound-and-endpoint level. An inconsistency between training and external endpoint keys prevented the intended overlap filtering. After correcting the matching logic, no records in the current assembled external data meet the strict requirement of compound exclusion from both bioaccumulation training and transport pretraining.

The historical classification achieved 60% accuracy, equal to the majority-class rule, with sensitivity 0.25, specificity 0.833, balanced accuracy 0.542 and MCC 0.102. These values describe the archived comparison at its exploratory cutoff; they should not be presented as independent regulatory validation. The interpretation of the original BAF values and their units also requires confirmation.

TMF, BMF and BSAF analyses are retained as associations involving repeated observations. Compound aggregation and separate compound- and study-cluster sensitivity analyses are reported where the number of clusters permits. These associations do not replace independent BAF or BCF prediction tests.

## Uncertainty and active learning

The historical cross-endpoint discrepancy plot is not an error-calibration analysis because it compares two predictions rather than predictions with observations. The historical acquisition curves were generated from an assumed exponential gain function; they do not document an executed acquisition-and-retraining experiment. Claims of a 50-versus-150 measurement advantage or a fixed efficiency multiplier should therefore be removed.

A new exploratory diagnostic used observed compound-level median logBAF values for 23 PFAS and a structure-only extra-trees baseline. Actual acquisition runs compared random, structural-diversity and tree-uncertainty selection with six initial compounds, twelve available pool compounds and five held-out compounds per seed. Five seeds and acquisition budgets of 0, 2, 4, 8 and 12 compounds were used. These results are limited to this small retrospective setting and do not validate the original transport network or its MC-dropout estimates.

A separate split-conformal diagnostic used nine training, nine calibration and five test compounds per seed. Although empirical coverage was often high, interval widths were large. This illustrates that nominal coverage alone does not establish useful predictive precision. The analysis uses tree-to-tree variability and should not be relabelled as MC-dropout calibration.

## Screening and interpretation

The archived screening file contains 127,993 rows, including 117,018 rows marked as structurally valid. These valid rows contain 114,896 distinct raw SMILES, so valid-row counts should not be described as counts of unique compounds. Candidate totals must be recalculated after documented identity handling, endpoint selection and threshold definition.

Absence from the selected OECD inventory is described as non-membership in that inventory, not absence of regulation. A screening output is a candidate list for further assessment and measurement, not a formal regulatory classification.

The B/P/T composite score in the Methods is not equivalent to the hydrophobicity, bioaccumulation and vP conditions used in the inspected plotting code. Toxicity and persistence claims require traceable inputs and definitions. Until those are provided, the unsupported composite-risk interpretation should be removed. Model-derived associations with chain length, headgroup and fluorination should be presented as predictive associations or hypotheses, rather than demonstrated mechanisms.

## Material that cannot yet be completed truthfully

The actual literature search dates and queries, number of human screeners and reviewers, degree of automated or AI-assisted extraction, handling of reviewer disagreement, original training environment and any separate 847-compound dataset remain to be confirmed. A larger independent external dataset and a corrected comparison of the original complete model against matched baselines are also required before a final manuscript can be assembled.

