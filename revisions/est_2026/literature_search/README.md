# Recoverable PubMed search expressions

Original source: `code/01_data_collection/collectors/pubmed_lit.py` at repository commit a7b97828d2cd1da8779c251e1271e8728bcad64c. These query constants establish the exact implemented searches; they do not establish a complete historical retrieval date/count log. The original collector uses NCBI ESearch (retmax 500), EFetch batches of 100, and PMID deduplication. It stores literature candidates, not experimental effect values.

```sh
python literature_search/retrieve.py --output literature_search/new_search
```

This independent script records UTC time, exact query, returned count, IDs, and raw ESearch JSON. It retrieves up to the PubMed 10,000-ID ceiling and fails explicitly if the ceiling would omit records. It does not change the admitted biological table.

## TMF_PFAS

`("trophic magnification" OR TMF) AND (PFAS OR PFOS OR PFOA OR perfluor*)`

## BAF_PFAS_field

`(bioaccumulation OR BAF OR "biota-sediment") AND (PFAS OR PFOS OR PFOA OR perfluor*) AND ("food web" OR field OR wild)`

## BCF_PFAS_fish

`(BCF OR bioconcentration) AND (PFAS OR PFOS OR PFOA OR perfluor*) AND (fish OR carp OR trout OR zebrafish)`

## PFAS_Arctic_foodweb

`(PFAS OR PFOS OR perfluor*) AND Arctic AND (food web OR trophic)`

## PFAS_Baltic

`(PFAS OR PFOS OR perfluor*) AND Baltic AND (food web OR trophic OR biomagnification)`

## PFAS_marine_mammal

`(PFAS OR PFOS OR perfluor*) AND (marine mammal OR cetacean OR seal OR dolphin OR whale)`
