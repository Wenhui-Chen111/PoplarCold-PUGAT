## Core reproduction pipeline

The following workflow prepares the input datasets, generates random and homology-aware grouped split manifests, trains the PoplarCold-PUGAT model and evaluates baseline methods.

---

## 1. Prepare data

```bash
python Prepare_data.py \
  --data_dir <directory_containing_input_datasets> \
  --raw_counts_tsv <raw_expression_count_matrix> \
  --de_table_tsv <differential_expression_table> \
  --positives_xlsx <known_cold_associated_positive_genes> \
  --ppi_csv <protein_interaction_network> \
  --ppi_score_threshold 800 \
  --coexp_corr_threshold 0.75 \
  --coexp_top_mr 15 \
  --output_dir <preprocessed_dataset_directory>
```


---

## 2. Generate random split manifest

Create random train/validation/test split information:

```bash
python Train_ColdPU.py \
  --data_dir <preprocessed_dataset_directory> \
  --split_mode random \
  --num_repeats 10 \
  --seed 42 \
  --manifest_only \
  --save_root <random_split_manifest_directory>
```

---

## 3. Generate homology-aware grouped split manifest

Generate grouped splits based on gene homology/grouping information:

```bash
python Train_ColdPU.py \
  --data_dir <preprocessed_dataset_directory> \
  --seed_metadata <gene_grouping_metadata_file> \
  --split_mode grouped \
  --num_repeats 10 \
  --seed 42 \
  --manifest_only \
  --save_root <grouped_split_manifest_directory>
```

The grouping metadata should contain the gene identifiers and grouping information required for homology-aware splitting.

---

## 4. Train PoplarCold-PUGAT

### Random split evaluation

```bash
python Train_ColdPU.py \
  --data_dir <preprocessed_dataset_directory> \
  --split_mode random \
  --split_manifest_in <random_split_manifest_file> \
  --save_root <random_split_model_output_directory>
```

### Homology-aware grouped split evaluation

```bash
python code/Train_ColdPU.py \
  --data_dir <preprocessed_dataset_directory> \
  --seed_metadata <gene_grouping_metadata_file> \
  --split_mode grouped \
  --split_manifest_in <grouped_split_manifest_file> \
  --save_root <grouped_split_model_output_directory>
```

---

## 5. Run baseline methods

The framework evaluates five baseline approaches for comparison.

### Random split baseline evaluation

```bash
python run_all_baselines.py \
  --data_dir <preprocessed_dataset_directory> \
  --split_manifest <random_split_manifest_file> \
  --split_mode random \
  --out_root <random_split_baseline_output_directory>
```

### Homology-aware grouped split baseline evaluation

```bash
python run_all_baselines.py \
  --data_dir <preprocessed_dataset_directory> \
  --split_manifest <grouped_split_manifest_file> \
  --split_mode grouped \
  --out_root <grouped_split_baseline_output_directory>
```

---
