# Federated Learning Pegasus Workflow Specification

## 1. Purpose

This specification defines a Pegasus WMS workflow for reproducible federated learning (FL) over public medical imaging datasets. The workflow uses TCIA as the primary naturally heterogeneous imaging source and NIH ChestX-ray14 as a complementary 2D radiograph benchmark. The goal is to show how Pegasus can make FL experiments reproducible, portable, auditable, and scalable without requiring raw data to move between clients.

## 2. Research Questions

1. How can Pegasus represent a federated learning experiment as a repeatable scientific workflow?
2. How does FL performance compare with local-only and centralized baselines under realistic medical imaging heterogeneity?
3. How do data modality, client partitioning, and compute placement affect runtime, data movement, and model quality?
4. What workflow provenance is needed to make medical imaging FL experiments reproducible?

## 3. Dataset Scope

### 3.1 TCIA Primary Dataset

Use The Cancer Imaging Archive (TCIA) as the primary source of naturally heterogeneous cancer imaging data. TCIA hosts de-identified cancer imaging collections, primarily DICOM radiology, with supporting data such as clinical outcomes, treatment details, genomics, pathology, and expert analyses when available.

Recommended initial TCIA collection: **LIDC-IDRI**.

Rationale:

- Public lung CT collection with thoracic CT scans and nodule annotations.
- TCIA reports 1,010 subjects, CT/DX/CR images, measurement data, radiomic features, diagnosis data, and a size of about 133 GB.
- Useful for a 3D medical imaging FL case study with realistic preprocessing and substantial compute.

Primary TCIA task:

- Lung nodule classification or detection proxy using LIDC-IDRI CT volumes and nodule annotations.

Alternative TCIA task:

- NSCLC-Radiomics survival or outcome modeling using pretreatment CT scans, RTSTRUCT/SEG objects, and clinical outcome data.

### 3.2 NIH ChestX-ray14 Secondary Dataset

Use NIH ChestX-ray14 as the 2D radiograph benchmark. NIH released over 100,000 anonymized chest x-ray images from more than 30,000 patients. Public mirrors describe 112,120 frontal chest x-ray images from 30,805 patients with labels for 14 thoracic findings.

Primary NIH task:

- Multi-label thoracic disease classification.

Important design constraint:

- NIH ChestX-ray14 is not naturally multi-institutional in the same way TCIA is. Use it as a reproducible 2D benchmark with simulated FL clients partitioned by patient ID, view position, label prevalence, age group, or controlled non-IID shards. Avoid presenting NIH partitions as real hospital sites.

## 4. Experimental Design

### 4.1 Client Definitions

For TCIA:

- Prefer natural client boundaries where metadata supports them: collection, trial, site, scanner manufacturer, modality, or acquisition protocol.
- If site labels are unavailable or too sparse, define clients by scanner manufacturer/protocol and document the proxy.
- Keep patient IDs exclusive to one client.

For NIH ChestX-ray14:

- Partition by patient ID only.
- Define experimental partitions:
  - IID: balanced patient-level random split.
  - Label-skew non-IID: clients enriched for different pathology groups.
  - Demographic/view-skew non-IID: clients grouped by age range, sex, or AP/PA view.

### 4.2 Baselines

Run the following for each dataset:

- **Local-only:** train one model per client and evaluate on all held-out clients.
- **Centralized:** train on pooled data as an upper-bound reference.
- **Federated averaging:** train client models locally and aggregate model weights per round.
- **Optional variants:** FedProx, weighted FedAvg, or secure aggregation simulation if time permits.

### 4.3 Evaluation Metrics

Model quality:

- NIH ChestX-ray14: AUROC per label, macro AUROC, micro AUROC, F1, calibration error.
- TCIA LIDC-IDRI: AUROC/F1 for nodule classification or sensitivity/FROC if detection is implemented.
- Domain generalization: client-wise metrics and worst-client performance.

Workflow and systems metrics:

- Total makespan.
- Per-round runtime.
- GPU/CPU utilization where available.
- Data staged per job.
- Number of retries/failures.
- Model artifact sizes.
- Pegasus provenance records for every generated artifact.

## 5. Pegasus Workflow Architecture

Represent the FL experiment as a hierarchical Pegasus workflow. The root
workflow handles setup and finalization, and each FL round is a Pegasus
`SubWorkflow` containing a fan-out/fan-in training and aggregation pattern.

```text
prepare_manifest
    -> preprocess_client_001
    -> preprocess_client_002
    -> ...
preprocess_* -> initialize_model
initialize_model -> subwf_round_001
subwf_round_001 -> subwf_round_002
...
subwf_round_N -> final_evaluation -> package_results

subwf_round_r:
    train_round_r_client_* -> aggregate_round_r -> bundle_round_outputs_r

parent workflow after each round:
    unpack_round_outputs_r -> next dependent jobs
```

Recommended Pegasus patterns:

- Use `Workflow(..., infer_dependencies=True)`.
- Use `SubWorkflow(...)` to encapsulate each FL round.
- Use one `File` object per manifest, preprocessed shard, model checkpoint, metrics file, and aggregate.
- Use fan-out for client training jobs.
- Use fan-in for model aggregation jobs.
- Prefer exporting one bundled child artifact per FL round when a parent workflow
  must consume multiple round outputs from a child `SubWorkflow`.
- Unpack child round bundles in an explicit parent-side job before final
  evaluation or packaging.
- Use `stage_out=True` for final reports, metrics, plots, selected checkpoints,
  and bundled round exports that must cross subworkflow boundaries.
- Use `stage_out=False` for intermediate per-round files that remain internal to
  a given round workflow.
- Do not scan directories inside jobs; pass explicit input file paths to wrappers.

## 6. Proposed Repository Layout

```text
fl-pegasus-workflow/
├── workflow_generator.py
├── bin/
│   ├── prepare_manifest.py
│   ├── preprocess_tcia.py
│   ├── preprocess_nih_cxr14.py
│   ├── initialize_model.py
│   ├── train_client.py
│   ├── aggregate.py
│   ├── bundle_round_outputs.py
│   ├── unpack_round_outputs.py
│   ├── evaluate.py
│   └── package_results.py
├── Docker/
│   └── FLMedicalImaging_Dockerfile
├── configs/
│   ├── tcia_lidc.yaml
│   ├── nih_cxr14.yaml
│   └── experiment.yaml
├── data/
│   ├── manifests/
│   └── test/
├── output/
├── scratch/
└── README.md
```

## 7. Workflow Jobs

### 7.1 `prepare_manifest`

Inputs:

- Dataset configuration file.
- Optional user-provided TCIA collection list or NIH metadata files.

Outputs:

- `manifests/dataset_manifest.csv`
- `manifests/client_manifest.csv`
- `manifests/splits.json`

Responsibilities:

- Record dataset source URLs, access date, checksums where feasible, and citation metadata.
- Build patient-level train/validation/test splits.
- Assign patients to clients.
- Validate that no patient crosses split or client boundaries.

### 7.2 `preprocess_tcia`

Inputs:

- TCIA DICOM series list or downloaded collection path.
- TCIA client manifest.

Outputs:

- Client-specific JSONL shards containing normalized image arrays, labels, and
  explicit split membership.

Responsibilities:

- Load real DICOM/image/volume inputs where available.
- Emit one JSONL shard per client with explicit train/val/test membership.
- Avoid directory scanning inside downstream jobs.
- Preprocessing metadata.

Responsibilities:

- Convert DICOM to analysis-ready volumes.
- Normalize voxel spacing and intensity windows.
- Extract or align annotations.
- Write deterministic outputs with checksums.

### 7.3 `preprocess_nih_cxr14`

Inputs:

- NIH image archive path.
- `Data_Entry_2017*.csv`
- Train/test lists if used.

Outputs:

- Client-specific image shards.
- Multi-label target files.

Responsibilities:

- Resize or center-crop images to the selected model resolution.
- Preserve patient-level split integrity.
- Encode 14 labels plus `No Finding` handling policy.

### 7.4 `initialize_model`

Outputs:

- `models/round_000_global.pt`
- `models/model_config.json`

Responsibilities:

- Create the initial model checkpoint.
- Record architecture, random seed, framework versions, and hyperparameters.
- Support pretrained image backbones such as `resnet18`.

### 7.5 `bundle_round_outputs`

Inputs:

- Round-global checkpoint generated by `aggregate`.
- Round aggregation metric file.

Outputs:

- One bundled round artifact, for example
  `artifacts/round_{r}_outputs.tar`.

Responsibilities:

- Package the child round checkpoint and round metric into one exported
  subworkflow output.
- Reduce planner/runtime fragility when a parent workflow needs multiple files
  from one child round.

### 7.6 `train_client`

Inputs:

- Current global model checkpoint.
- One client shard.
- Client-specific configuration.

Outputs:

- `models/round_{r}/client_{id}_weights.pt`
- `metrics/round_{r}/client_{id}_train.json`

Responsibilities:

- Train locally for configured epochs.
- Never require raw data from other clients.
- Emit model delta or full weights plus training metrics.
- Support end-to-end image training with configurable augmentation,
  optimizer/scheduler choice, gradient clipping, class-weighted loss, optional
  backbone freezing, and periodic CPU/GPU/RAM monitoring.

### 7.7 `aggregate`

Inputs:

- Explicit list of client model outputs for the round.
- Client sample counts.

Outputs:

- `models/round_{r}_global.pt`
- `metrics/round_{r}/round_{r}_aggregation.json`

Responsibilities:

- Implement weighted FedAvg by default.
- Support FedProx-compatible aggregation metadata.
- Validate that all required client updates are present.
- Record excluded clients and failure policy.

### 7.7 `evaluate`

Inputs:

- Round or final global model.
- Held-out client test shards.

Outputs:

- `metrics/final_evaluation.json`
- optional future figures such as client heatmaps or ROC curves

Responsibilities:

- Evaluate globally and per client.
- Compute quality and fairness/domain-shift summaries.
- Support cross-dataset generalization checks when multiple dataset branches are
  present.

### 7.8 `train_centralized`

Inputs:

- Pooled client training shards for one dataset branch.
- Dataset-specific configuration.

Outputs:

- `metrics/{branch}/{branch}_baseline.json`

Responsibilities:

- Train a centralized supervised baseline on pooled branch data.
- Use the same image-model stack and optimization settings as FL training where
  applicable.
- Provide a realistic upper-bound reference for FL comparisons.

### 7.9 `package_results`

Inputs:

- Final metrics, plots, workflow metadata, and selected checkpoints.

Outputs:

- `results.tar.gz`
- `paper_tables.csv`
- `provenance_summary.json`

Responsibilities:

- Collect final workflow outputs.
- Export enough metadata to reproduce the experiment.

## 8. Containers and Execution Sites

Use a container image that includes the full training stack:

- Python 3.10 or 3.11.
- Pegasus Python API (`pegasus-wms`) for local workflow generation.
- Pillow for NIH image loading.
- pydicom, SimpleITK, nibabel.
- pandas, numpy, scikit-learn.
- matplotlib, seaborn.
- PyTorch.
- torchvision.
- psutil and pynvml for resource monitoring.

Pegasus execution:

- CPU preprocessing jobs can run on general execute nodes.
- Training jobs may run on CPU or GPU depending on `request_gpus` and the
  selected container/configuration.
- GPU training jobs should request GPU resources through Condor profiles.
- Aggregation jobs usually require less memory and may run on CPU.

External data directories should be transferred with CondorIO `transfer_input_files` when needed. Do not rely on container bind mounts for dataset directories in portable experiments.

## 9. Configuration Parameters

Current generator invocation:

```bash
python3 workflow_generator.py \
  --config configs/tcia_lidc.yaml \
  --output workflow.yml
```

Configuration fields:

- `dataset_name`
- `task`
- `data_root`
- `metadata_files`
- `client_partition_strategy`
- `num_clients`
- `split_seed`
- `model_name`
- `image_size`
- `batch_size`
- `learning_rate`
- `rounds`
- `local_epochs`
- `aggregation`
- `optimizer`
- `scheduler`
- `gradient_clip_norm`
- `pretrained`
- `freeze_backbone`
- `unfreeze_backbone_epoch`
- `class_weighted_loss`
- `request_gpus`
- `monitor_interval_seconds`
- `evaluation_metrics`
- `sample_id_column`
- `patient_id_column`
- `label_column`
- `image_path_column`
- `client_id_column`
- `allow_synthetic_fallback`

## 10. Reproducibility Requirements

Each run must record:

- Git commit of the workflow repository.
- Container image digest.
- Dataset source, access date, and citation.
- Manifest checksums.
- Patient/client/split manifests.
- Random seeds.
- Hyperparameters.
- Pegasus run directory.
- Pegasus statistics and analyzer output if failures occur.

All final reported results should be generated from `package_results` outputs, not manually copied logs.

## 11. Data Governance and Ethics

The experiment uses public, de-identified datasets, but it still models sensitive medical workflows. The workflow documentation should state:

- Public data are used to simulate institutional FL.
- No claim is made that NIH ChestX-ray14 clients correspond to real hospitals.
- TCIA and NIH dataset citation and usage policies must be followed.
- No attempt should be made to re-identify subjects.
- Generated models are research artifacts, not clinical decision tools.

## 12. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| TCIA collection metadata may not expose clean site labels | Use scanner/protocol/client proxies and document them clearly. |
| NIH labels are noisy because they were mined from reports | Report label-noise limitations and use robust metrics. |
| 3D CT training may exceed available GPU resources | Start with resized 2D slices, cropped patches, or smaller backbones before full-volume models. |
| Dataset downloads are large | Support manifest-only dry runs and small smoke-test subsets. |
| FL rounds create many intermediate checkpoints | Stage out only selected checkpoints and final artifacts. |
| Directory scanning breaks Pegasus staging | Pass every input file explicitly through manifests and `File` objects. |

## 13. Implementation Milestones

1. Create manifest generator and small smoke-test manifests.
2. Implement NIH ChestX-ray14 preprocessing and centralized baseline.
3. Implement generic FedAvg loop with simulated clients.
4. Add Pegasus fan-out/fan-in jobs for client training and aggregation.
5. Refactor FL rounds into Pegasus `SubWorkflow`s.
6. Add real NIH/TCIA file ingestion with image-based preprocessing.
7. Add final evaluation, plots, and `package_results`.
8. Add centralized baseline, GPU runtime support, and image-model training.
9. Add cross-dataset evaluation and resource monitoring.
10. Run scalability experiments with increasing clients and rounds.

## 14. Source Links

- TCIA access and usage policy: https://www.cancerimagingarchive.net/access-data/
- TCIA API guides: https://wiki.cancerimagingarchive.net/x/NIIiAQ
- TCIA LIDC-IDRI collection: https://www.cancerimagingarchive.net/collection/lidc-idri/
- TCIA NSCLC-Radiomics collection: https://www.cancerimagingarchive.net/collection/nsclc-radiomics/
- NIH public release article: https://irp.nih.gov/news-and-events/in-the-news/nih-clinical-center-provides-one-of-the-largest-publicly-available-chest-x
- NIH ChestX-ray14 public mirror metadata: https://www.kaggle.com/datasets/nih-chest-xrays/data
- Pegasus federated learning example: https://github.com/pegasus-isi/federated-learning-example/tree/0b134a60899483305cb386cbd705cfc25e662379
- Pegasus service documentation: https://pegasus.isi.edu/documentation/reference-guide/pegasus-service.html
- Pegasus `SubWorkflow` API: https://pegasus.isi.edu/documentation/python/Pegasus.api.html#Pegasus.api.workflow.SubWorkflow
