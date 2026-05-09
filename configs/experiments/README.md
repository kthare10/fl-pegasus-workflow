# Experiment Config Pack

This directory contains configuration files for the requested experiment
matrix. They are split into two groups:

- runnable single-dataset configs that work with the current workflow
- planned specs for experiments that need code changes before they are valid

## Current Workflow Limits

The current workflow can run one dataset family per workflow:

- TCIA-style runs use `preprocess_tcia`
- NIH ChestX-ray14 runs use `preprocess_nih_cxr14`

It does not yet support:

- one workflow containing both TCIA and NIH clients at the same time
- true FedProx local training
- explicit centralized-training baselines
- cross-dataset transfer with checkpoint warm-start as a first-class config

Because of that:

- E1, E3, and E4 are provided as runnable per-dataset templates
- E2 is provided as a planned FedProx spec, but the current code still behaves
  as FedAvg unless `train_client.py` is extended
- E5 is provided as a planned cross-dataset spec, but the current code does not
  yet orchestrate TCIA-to-NIH transfer in one config

## Runnable Configs

These can be used now after filling in real paths and container image values:

- `e1_baseline_tcia.json`
- `e1_baseline_nih.json`
- `e3_scalability_tcia.json`
- `e3_scalability_nih.json`
- `e4_communication_tcia.json`
- `e4_communication_nih.json`

## Planned / Not Yet Runnable As Intended

- `e2_algorithm_fedprox_tcia.json`
- `e2_algorithm_fedprox_nih.json`
- `e5_cross_dataset_transfer.json`

## Mapping To The Requested Matrix

- E1 Baseline
  Current approximation: run `e1_baseline_tcia.json` and `e1_baseline_nih.json`
  separately. A true `10 TCIA + 10 NIH` joint FL run needs mixed-dataset
  manifest support.
- E2 Algorithm
  Current files capture the intended hyperparameters and `prox_mu`, but true
  FedProx needs code support in `bin/train_client.py`.
- E3 Scalability
  Current approximation: run `e3_scalability_tcia.json` and
  `e3_scalability_nih.json` separately with `5` clients each.
- E4 Communication
  Current approximation: run `e4_communication_tcia.json` and
  `e4_communication_nih.json` separately with fewer rounds.
- E5 Cross-Dataset
  Captured as a design config only. This needs explicit cross-dataset
  initialization / checkpoint handoff support.
