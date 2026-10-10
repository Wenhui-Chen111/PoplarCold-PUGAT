# PoplarCold-PUGAT

Official implementation and reproducibility materials for **PoplarCold-PUGAT**, a positive-unlabeled heterogeneous graph attention framework for prioritizing cold-associated genes in *Populus trichocarpa*.


## Installation

```bash
git clone https://github.com/Wenhui-Chen111/PoplarCold-PUGAT.git
cd PoplarCold-PUGAT
pip install -r requirements.txt
```

## Core reproduction pipeline

The following commands generate the random and homology-aware grouped split manifests, run the main PoplarCold-PUGAT model, and evaluate the five baseline methods.

### 1. Generate the random-split manifest

```bash
python code/Train_ColdPU.py \
  --data_dir ${DATA_DIR} \
  --split_mode random \
  --num_repeats 10 \
  --seed 42 \
  --manifest_only \
  --save_root results/random_manifest
```

### 2. Generate the homology-aware grouped-split manifest

```bash
python code/Train_ColdPU.py \
  --data_dir ${DATA_DIR} \
  --seed_metadata ${SEED_META} \
  --split_mode grouped \
  --num_repeats 10 \
  --seed 42 \
  --manifest_only \
  --save_root results/grouped_manifest
```

### 3. Run PoplarCold-PUGAT

Random split:

```bash
python code/Train_ColdPU.py \
  --data_dir ${DATA_DIR} \
  --split_mode random \
  --split_manifest_in results/random_manifest/split_manifest.csv \
  --save_root results/main_random
```

Grouped split:

```bash
python code/Train_ColdPU.py \
  --data_dir ${DATA_DIR} \
  --seed_metadata ${SEED_META} \
  --split_mode grouped \
  --split_manifest_in results/grouped_manifest/split_manifest.csv \
  --save_root results/main_grouped
```

### 4. Run the five baseline methods

Random split:

```bash
python code/run_all_baselines.py \
  --data_dir ${DATA_DIR} \
  --split_manifest results/random_manifest/split_manifest.csv \
  --split_mode random \
  --out_root results/baselines_random
```

Grouped split:

```bash
python code/run_all_baselines.py \
  --data_dir ${DATA_DIR} \
  --split_manifest results/grouped_manifest/split_manifest.csv \
  --split_mode grouped \
  --out_root results/baselines_grouped
```

## Reviewer-requested standalone analyses

Additional analyses requested during peer review are provided as standalone scripts:

- `code/Evidence_Separation_Experiment.py` — evidence-separation analysis.
- `code/PU_Objective_Controls.py` — matched PU-objective and unlabeled-sampling controls.
- `code/Run_enrichment.py` — GO, KEGG, and transcription-factor-family enrichment.
- `code/Run_independent_cold_validation.py` — independent cold-treatment transcriptome validation.
- `code/Run_external_validation_controls.py` — expression-only and matched-random external-validation controls.

Run any script with `--help` to display its required arguments, for example:

```bash
python code/Run_enrichment.py --help
```
