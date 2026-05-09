#!/usr/bin/env python3

"""Generate a Pegasus workflow for federated medical imaging experiments."""

import argparse
import csv
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - optional local convenience
    yaml = None

from Pegasus.api import *


TOOL_CONFIGS = {
    "validate_dataset": {"memory": "512 MB", "cores": 1},
    "prepare_manifest": {"memory": "1 GB", "cores": 1},
    "preprocess_tcia": {"memory": "4 GB", "cores": 2},
    "preprocess_nih_cxr14": {"memory": "2 GB", "cores": 2},
    "initialize_model": {"memory": "1 GB", "cores": 1},
    "train_client": {"memory": "8 GB", "cores": 2},
    "train_centralized": {"memory": "8 GB", "cores": 2},
    "aggregate": {"memory": "2 GB", "cores": 1},
    "evaluate": {"memory": "4 GB", "cores": 2},
    "baseline_evaluate": {"memory": "2 GB", "cores": 1},
    "compute_branch_stats": {"memory": "1 GB", "cores": 1},
    "cross_eval": {"memory": "1 GB", "cores": 1},
    "plot_results": {"memory": "1 GB", "cores": 1},
    "generate_report": {"memory": "1 GB", "cores": 1},
    "package_results": {"memory": "1 GB", "cores": 1},
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)


def load_prepare_manifest_module(base_dir):
    module_path = os.path.join(base_dir, "bin", "prepare_manifest.py")
    spec = importlib.util.spec_from_file_location("fl_prepare_manifest", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FederatedLearningWorkflow:
    wf_name = "fl_medical_imaging"

    def __init__(self, config, config_path, dagfile="workflow.yml"):
        self.config = config
        self.config_path = os.path.abspath(config_path)
        self.dagfile = dagfile
        self.wf_dir = str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")
        self.container_image = config.get("container_image")
        self.dataset = config["dataset_name"]
        self.branch_pipelines = config.get("branch_pipelines", [])
        self.dataset_pipelines = config.get("dataset_pipelines", [])
        self.client_specs = self._build_client_specs()
        self.num_clients = len(self.client_specs)
        self.rounds = int(config.get("rounds", 2))
        self.preprocess_tool = (
            "preprocess_tcia" if self.dataset.startswith("tcia") else "preprocess_nih_cxr14"
        )
        default_stage = self.preprocess_tool == "preprocess_tcia" or bool(self.dataset_pipelines)
        self.stage_input_data = bool(config.get("stage_input_data", default_stage))
        self.prepare_manifest_module = load_prepare_manifest_module(self.wf_dir)
        self.runtime_support_files = {
            "flwr_torch_utils": File("flwr_torch_utils.py"),
        }
        self.staged_input_rows = []
        self.client_input_files = {}
        self.input_replicas = {}
        self.static_manifest_paths = {}
        self.branch_states = []
        self.cross_eval_spec_path = None
        if self.branch_pipelines:
            self._prepare_branch_states()
            self._write_cross_eval_spec()
        elif self.stage_input_data:
            self._prepare_condorio_inputs()

    def _pipeline_tool(self, dataset_name):
        return "preprocess_tcia" if str(dataset_name).startswith("tcia") else "preprocess_nih_cxr14"

    def _build_client_specs(self):
        pipelines = self.config.get("dataset_pipelines", [])
        if not pipelines:
            count = int(self.config.get("num_clients", 3))
            return [
                {"client_id": client_id, "dataset_name": self.config["dataset_name"], "tool": self._pipeline_tool(self.config["dataset_name"])}
                for client_id in range(count)
            ]

        specs = []
        client_id = 0
        for index, pipeline in enumerate(pipelines):
            pipeline_name = pipeline.get("dataset_name", pipeline.get("name", f"pipeline_{index:02d}"))
            tool = self._pipeline_tool(pipeline_name)
            for _ in range(int(pipeline["num_clients"])):
                specs.append({"client_id": client_id, "dataset_name": pipeline_name, "tool": tool})
                client_id += 1
        return specs

    def _prepare_condorio_inputs(self):
        rows = self.prepare_manifest_module.load_metadata_rows(self.config)
        self.prepare_manifest_module.apply_staged_image_paths(rows, self.config)
        self.staged_input_rows = rows
        assignments = self.prepare_manifest_module.assign_splits(
            rows,
            int(self.config.get("split_seed", 13)),
            float(self.config.get("train_fraction", 0.7)),
            float(self.config.get("val_fraction", 0.1)),
        )
        self._write_static_manifests(rows, assignments)

        for row in rows:
            source_path = os.path.abspath(row.get("source_path", row["image_path"]))
            lfn = row["image_path"]
            client_id = int(row["client_id"])
            if lfn not in self.input_replicas:
                self.input_replicas[lfn] = source_path
            self.client_input_files.setdefault(client_id, [])
            self.client_input_files[client_id].append(File(lfn))

    def _write_static_manifests(self, rows, assignments):
        manifest_dir = os.path.join(self.wf_dir, "manifests")
        os.makedirs(manifest_dir, exist_ok=True)
        self.static_manifest_paths = {
            "dataset_manifest": os.path.join(manifest_dir, "planned_dataset_manifest.csv"),
            "client_manifest": os.path.join(manifest_dir, "planned_client_manifest.csv"),
            "splits": os.path.join(manifest_dir, "planned_splits.json"),
        }

        fieldnames = [
            "sample_id",
            "patient_id",
            "client_id",
            "split",
            "image_path",
            "labels",
            "dataset",
        ]
        with open(self.static_manifest_paths["client_manifest"], "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.prepare_manifest_module.output_rows(rows))

        with open(self.static_manifest_paths["dataset_manifest"], "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["key", "value"])
            writer.writeheader()
            writer.writerow({"key": "dataset_name", "value": self.config["dataset_name"]})
            writer.writerow({"key": "generated_at", "value": datetime.now(timezone.utc).isoformat()})
            writer.writerow({"key": "num_samples", "value": len(rows)})
            writer.writerow({"key": "num_patients", "value": len(assignments)})
            writer.writerow({"key": "num_clients", "value": self.num_clients})

        with open(self.static_manifest_paths["splits"], "w", encoding="utf-8") as handle:
            json.dump({"patient_splits": assignments}, handle, indent=2, sort_keys=True)

    def _prefixed_path(self, prefix, relative_path):
        return f"{prefix}/{relative_path}" if prefix else relative_path

    def _branch_merged_config(self, branch):
        merged = dict(self.config)
        for key in ("branch_pipelines", "dataset_pipelines", "num_clients"):
            merged.pop(key, None)
        merged.update(branch)
        if merged.get("dataset_archive"):
            merged["stage_input_data"] = False
        return merged

    def _write_branch_config(self, branch_id, branch_config):
        config_dir = os.path.join(self.wf_dir, "manifests", "branch_configs")
        os.makedirs(config_dir, exist_ok=True)
        path = os.path.join(config_dir, f"{branch_id}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(branch_config, handle, indent=2, sort_keys=True)
        return path

    def _prepare_branch_states(self):
        self.num_clients = 0
        for index, branch in enumerate(self.branch_pipelines):
            branch_id = branch.get("name", f"branch_{index:02d}")
            branch_config = self._branch_merged_config(branch)
            branch_config_path = self._write_branch_config(branch_id, branch_config)
            rows = []
            assignments = {}
            client_input_files = {}
            input_replicas = {}
            can_precompute_inputs = bool(branch_config.get("stage_input_data")) and os.path.isdir(
                branch_config.get("data_root", "")
            )
            if can_precompute_inputs:
                rows = self.prepare_manifest_module.load_metadata_rows_single(branch_config)
                self.prepare_manifest_module.apply_staged_image_paths(rows, branch_config)
                assignments = self.prepare_manifest_module.assign_splits(
                    rows,
                    int(branch_config.get("split_seed", self.config.get("split_seed", 13))),
                    float(branch_config.get("train_fraction", self.config.get("train_fraction", 0.7))),
                    float(branch_config.get("val_fraction", self.config.get("val_fraction", 0.1))),
                )
                for row in rows:
                    source_path = os.path.abspath(row.get("source_path", row["image_path"]))
                    lfn = row["image_path"]
                    client_id = int(row["client_id"])
                    if lfn not in input_replicas:
                        input_replicas[lfn] = source_path
                    client_input_files.setdefault(client_id, [])
                    client_input_files[client_id].append(File(lfn))
            self.branch_states.append(
                {
                    "id": branch_id,
                    "config": branch_config,
                    "config_path": branch_config_path,
                    "config_lfn": f"configs_branch_{branch_id}.json",
                    "archive_path": branch_config.get("dataset_archive"),
                    "archive_lfn": f"archives/{branch_id}/{os.path.basename(branch_config['dataset_archive'])}"
                    if branch_config.get("dataset_archive")
                    else None,
                    "tool": self._pipeline_tool(branch_config["dataset_name"]),
                    "num_clients": int(branch_config["num_clients"]),
                    "rows": rows,
                    "assignments": assignments,
                    "client_input_files": client_input_files,
                    "input_replicas": input_replicas,
                }
            )
            self.num_clients += int(branch_config["num_clients"])

    def _write_cross_eval_spec(self):
        spec_dir = os.path.join(self.wf_dir, "manifests")
        os.makedirs(spec_dir, exist_ok=True)
        self.cross_eval_spec_path = os.path.join(spec_dir, "branch_matrix_spec.json")
        payload = {"branches": {}}
        for branch in self.branch_states:
            branch_id = branch["id"]
            payload["branches"][branch_id] = {
                "config": branch["config_lfn"],
                "model": f"models/{branch_id}/round_{self.rounds:03d}_global.pt",
                "client_data": [
                    f"preprocessed/{branch_id}/client_{client_id:03d}.jsonl"
                    for client_id in range(branch["num_clients"])
                ],
            }
        with open(self.cross_eval_spec_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def create_pegasus_properties(self):
        self.props = Properties()
        self.props["pegasus.transfer.threads"] = "16"
        self.props["pegasus.data.configuration"] = "condorio"

    def create_sites_catalog(self, exec_site_name="condorpool"):
        self.sc = SiteCatalog()
        local = Site("local").add_directories(
            Directory(Directory.SHARED_SCRATCH, self.shared_scratch_dir).add_file_servers(
                FileServer("file://" + self.shared_scratch_dir, Operation.ALL)
            ),
            Directory(Directory.LOCAL_STORAGE, self.local_storage_dir).add_file_servers(
                FileServer("file://" + self.local_storage_dir, Operation.ALL)
            ),
        )
        condorio = (
            Site("condorio")
            .add_condor_profile(universe="local")
            .add_pegasus_profile(style="condor")
        )
        if exec_site_name == "submithost":
            exec_site = (
                Site(exec_site_name)
                .add_condor_profile(universe="local")
                .add_pegasus_profile(style="condor")
            )
        else:
            exec_site = (
                Site(exec_site_name)
                .add_condor_profile(universe="vanilla")
                .add_pegasus_profile(style="condor")
            )
        self.sc.add_sites(local, condorio, exec_site)

    def create_transformation_catalog(self, exec_site_name="condorpool"):
        self.tc = TransformationCatalog()
        container = None
        if self.container_image:
            container = Container(
                "fl_medical_imaging_container",
                container_type=Container.SINGULARITY,
                image=self.container_image,
                image_site="docker_hub",
            )
            self.tc.add_containers(container)

        transformations = []
        branch_archive_mode = self.branch_pipelines and any(
            branch["archive_path"] for branch in self.branch_states
        )
        for tool_name, resources in TOOL_CONFIGS.items():
            tx_site = exec_site_name
            if tool_name == "validate_dataset" and not branch_archive_mode:
                tx_site = "local"
            if tool_name == "prepare_manifest" and self.stage_input_data and not self.branch_pipelines:
                tx_site = "local"
            tx = Transformation(
                tool_name,
                site=tx_site,
                pfn=os.path.join(self.wf_dir, "bin", f"{tool_name}.py"),
                is_stageable=True,
                container=container,
            ).add_pegasus_profile(memory=resources["memory"], cores=resources.get("cores", 1))
            if tool_name in {"train_client", "train_centralized"} and self.config.get("request_gpus", 0):
                tx.add_condor_profile(request_gpus=str(self.config["request_gpus"]))
            transformations.append(tx)

        self.tc.add_transformations(*transformations)

    def create_replica_catalog(self):
        self.rc = ReplicaCatalog()
        self.rc.add_replica("local", "configs/experiment.yaml", "file://" + self.config_path)
        self.rc.add_replica(
            "local",
            self.runtime_support_files["flwr_torch_utils"].lfn,
            "file://" + os.path.join(self.wf_dir, "bin", "flwr_torch_utils.py"),
        )

        if self.branch_pipelines:
            if self.cross_eval_spec_path:
                self.rc.add_replica(
                    "local",
                    "manifests/branch_matrix_spec.json",
                    "file://" + os.path.abspath(self.cross_eval_spec_path),
                )
            for branch in self.branch_states:
                self.rc.add_replica(
                    "local",
                    branch["config_lfn"],
                    "file://" + os.path.abspath(branch["config_path"]),
                )
                for metadata_path in branch["config"].get("metadata_files", []):
                    if not os.path.exists(metadata_path):
                        continue
                    abs_path = os.path.abspath(metadata_path)
                    lfn = f"metadata/{branch['id']}/{os.path.basename(metadata_path)}"
                    self.rc.add_replica("local", lfn, "file://" + abs_path)
                if branch["archive_path"]:
                    self.rc.add_replica(
                        "local",
                        branch["archive_lfn"],
                        "file://" + os.path.abspath(branch["archive_path"]),
                    )
                for lfn, abs_path in sorted(branch["input_replicas"].items()):
                    self.rc.add_replica("local", lfn, "file://" + abs_path)
            return

        if self.static_manifest_paths:
            self.rc.add_replica(
                "local",
                "manifests/dataset_manifest.csv",
                "file://" + self.static_manifest_paths["dataset_manifest"],
            )
            self.rc.add_replica(
                "local",
                "manifests/client_manifest.csv",
                "file://" + self.static_manifest_paths["client_manifest"],
            )
            self.rc.add_replica(
                "local",
                "manifests/splits.json",
                "file://" + self.static_manifest_paths["splits"],
            )

        for metadata_path in self.config.get("metadata_files", []):
            if not os.path.exists(metadata_path):
                continue
            abs_path = os.path.abspath(metadata_path)
            lfn = "metadata/" + os.path.basename(metadata_path)
            self.rc.add_replica("local", lfn, "file://" + abs_path)

        for lfn, abs_path in sorted(self.input_replicas.items()):
            self.rc.add_replica("local", lfn, "file://" + abs_path)

    def create_round_subworkflow(
        self,
        round_idx,
        config_file,
        config_path,
        current_global,
        client_shards,
        workflow_name=None,
        path_prefix="",
    ):
        round_wf = Workflow(workflow_name or f"fl_round_{round_idx:03d}", infer_dependencies=True)
        child_rc = ReplicaCatalog()
        child_rc.add_replica("local", config_file.lfn, "file://" + os.path.abspath(config_path))
        child_rc.add_replica(
            "local",
            self.runtime_support_files["flwr_torch_utils"].lfn,
            "file://" + os.path.join(self.wf_dir, "bin", "flwr_torch_utils.py"),
        )
        round_wf.add_replica_catalog(child_rc)

        client_updates = []
        client_counts = []
        for client_id, shard in client_shards:
            update = File(
                self._prefixed_path(
                    path_prefix,
                    f"models/round_{round_idx:03d}/client_{client_id:03d}_weights.pt",
                )
            )
            train_metrics = File(
                self._prefixed_path(
                    path_prefix,
                    f"metrics/round_{round_idx:03d}/{path_prefix or 'global'}_client_{client_id:03d}_train.json",
                )
            )
            count_file = File(
                self._prefixed_path(
                    path_prefix,
                    f"metrics/round_{round_idx:03d}/{path_prefix or 'global'}_client_{client_id:03d}_count.json",
                )
            )
            client_updates.append(update)
            client_counts.append(count_file)

            train_job = (
                Job(
                    "train_client",
                    _id=f"train_r{round_idx:03d}_client_{client_id:03d}",
                    node_label=f"train_r{round_idx:03d}_client_{client_id:03d}",
                )
                .add_args(
                    "--config",
                    config_file,
                    "--client-id",
                    str(client_id),
                    "--round",
                    str(round_idx),
                    "--global-model",
                    current_global,
                    "--client-data",
                    shard,
                    "--output-model",
                    update,
                    "--metrics",
                    train_metrics,
                    "--count-output",
                    count_file,
                )
                .add_inputs(
                    config_file,
                    current_global,
                    shard,
                    self.runtime_support_files["flwr_torch_utils"],
                )
                .add_outputs(update, stage_out=False, register_replica=False)
                .add_outputs(train_metrics, stage_out=True, register_replica=False)
                .add_outputs(count_file, stage_out=False, register_replica=False)
                .add_pegasus_profiles(label=f"client_{client_id:03d}")
            )
            round_wf.add_jobs(train_job)

        next_global = File(self._prefixed_path(path_prefix, f"models/round_{round_idx:03d}_global.pt"))
        aggregation_metrics = File(
            self._prefixed_path(
                path_prefix,
                f"metrics/round_{round_idx:03d}/{path_prefix or 'global'}_round_{round_idx:03d}_aggregation.json",
            )
        )
        aggregate_job = (
            Job(
                "aggregate",
                _id=f"aggregate_round_{round_idx:03d}",
                node_label=f"aggregate_round_{round_idx:03d}",
            )
            .add_args(
                "--config",
                config_file,
                "--round",
                str(round_idx),
                "--output-model",
                next_global,
                "--metrics",
                aggregation_metrics,
            )
            .add_inputs(
                config_file,
                self.runtime_support_files["flwr_torch_utils"],
                *client_updates,
                *client_counts,
            )
            .add_outputs(next_global, stage_out=False, register_replica=False)
            .add_outputs(aggregation_metrics, stage_out=False, register_replica=False)
        )
        for update in client_updates:
            aggregate_job.add_args("--client-update", update)
        for count_file in client_counts:
            aggregate_job.add_args("--client-count", count_file)
        round_wf.add_jobs(aggregate_job)

        return round_wf, next_global, aggregation_metrics

    def create_dual_branch_workflow(self):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)
        branch_outputs = []

        for branch in self.branch_states:
            branch_id = branch["id"]
            config_file = File(branch["config_lfn"])
            dataset_manifest = File(f"manifests/{branch_id}/dataset_manifest.csv")
            client_manifest = File(f"manifests/{branch_id}/client_manifest.csv")
            splits = File(f"manifests/{branch_id}/splits.json")
            validation = File(f"metrics/{branch_id}/{branch_id}_dataset_validation.json")
            metadata_inputs = [
                File(f"metadata/{branch_id}/{os.path.basename(path)}")
                for path in branch["config"].get("metadata_files", [])
                if os.path.exists(path)
            ]
            archive_input = File(branch["archive_lfn"]) if branch["archive_lfn"] else None

            validate_job = (
                Job("validate_dataset", _id=f"download_{branch_id}", node_label=f"download_{branch_id}")
                .add_args("--config", config_file)
                .add_outputs(validation, stage_out=True, register_replica=False)
            )
            validate_job.add_inputs(config_file, *metadata_inputs)
            for metadata_file in metadata_inputs:
                validate_job.add_args("--metadata", metadata_file)
            if archive_input:
                validate_job.add_args("--archive", archive_input)
                validate_job.add_inputs(archive_input)
            validate_job.add_args("--output", validation)
            self.wf.add_jobs(validate_job)

            prepare_job = (
                Job("prepare_manifest", _id=f"partition_{branch_id}", node_label=f"partition_{branch_id}")
                .add_args(
                    "--config",
                    config_file,
                    "--dataset-manifest",
                    dataset_manifest,
                    "--client-manifest",
                    client_manifest,
                    "--splits",
                    splits,
                )
                .add_inputs(config_file, *metadata_inputs)
                .add_outputs(dataset_manifest, stage_out=True, register_replica=False)
                .add_outputs(client_manifest, stage_out=True, register_replica=False)
                .add_outputs(splits, stage_out=True, register_replica=False)
            )
            for metadata_file in metadata_inputs:
                prepare_job.add_args("--metadata", metadata_file)
            if archive_input:
                prepare_job.add_args("--archive", archive_input)
                prepare_job.add_inputs(archive_input)
            self.wf.add_jobs(prepare_job)

            client_shards = []
            for client_id in range(branch["num_clients"]):
                shard = File(f"preprocessed/{branch_id}/client_{client_id:03d}.jsonl")
                client_shards.append((client_id, shard))
                image_inputs = branch["client_input_files"].get(client_id, [])
                preprocess_job = (
                    Job(
                        branch["tool"],
                        _id=f"preprocess_{branch_id}_client_{client_id:03d}",
                        node_label=f"preprocess_{branch_id}_client_{client_id:03d}",
                    )
                    .add_args(
                        "--config",
                        config_file,
                        "--client-manifest",
                        client_manifest,
                        "--client-id",
                        str(client_id),
                        "--output",
                        shard,
                    )
                    .add_inputs(
                        config_file,
                        client_manifest,
                        self.runtime_support_files["flwr_torch_utils"],
                        *image_inputs,
                    )
                    .add_outputs(shard, stage_out=False, register_replica=False)
                    .add_pegasus_profiles(label=f"{branch_id}_client_{client_id:03d}")
                )
                if archive_input:
                    preprocess_job.add_args("--archive", archive_input)
                    preprocess_job.add_inputs(archive_input)
                self.wf.add_jobs(preprocess_job)

            model_config = File(f"models/{branch_id}/model_config.json")
            current_global = File(f"models/{branch_id}/round_000_global.pt")
            init_job = (
                Job(
                    "initialize_model",
                    _id=f"initialize_model_{branch_id}",
                    node_label=f"initialize_model_{branch_id}",
                )
                .add_args(
                    "--config",
                    config_file,
                    "--model",
                    current_global,
                    "--model-config",
                    model_config,
                )
                .add_inputs(config_file, self.runtime_support_files["flwr_torch_utils"])
                .add_outputs(current_global, stage_out=False, register_replica=False)
                .add_outputs(model_config, stage_out=True, register_replica=False)
            )
            self.wf.add_jobs(init_job)

            for round_idx in range(1, self.rounds + 1):
                round_wf, next_global, aggregation_metrics = self.create_round_subworkflow(
                    round_idx,
                    config_file,
                    branch["config_path"],
                    current_global,
                    client_shards,
                    workflow_name=f"fl_round_{branch_id}_{round_idx:03d}",
                    path_prefix=branch_id,
                )
                round_job = (
                    SubWorkflow(
                        round_wf,
                        _id=f"fl_round_{branch_id}_r{round_idx - 1}",
                        node_label=f"fl_round_{branch_id}_r{round_idx - 1}",
                    )
                    .add_inputs(config_file, current_global, *(shard for _, shard in client_shards))
                    .add_outputs(next_global, stage_out=False, register_replica=False)
                )
                self.wf.add_jobs(round_job)
                current_global = next_global

            evaluation_metrics = File(f"metrics/{branch_id}/{branch_id}_final_evaluation.json")
            evaluate_job = (
                Job(
                    "evaluate",
                    _id=f"evaluate_final_{branch_id}",
                    node_label=f"evaluate_final_{branch_id}",
                )
                .add_args(
                    "--config",
                    config_file,
                    "--model",
                    current_global,
                    "--output",
                    evaluation_metrics,
                )
                .add_inputs(
                    config_file,
                    current_global,
                    self.runtime_support_files["flwr_torch_utils"],
                    *(shard for _, shard in client_shards),
                )
                .add_outputs(evaluation_metrics, stage_out=True, register_replica=False)
            )
            for _, shard in client_shards:
                evaluate_job.add_args("--client-data", shard)
            self.wf.add_jobs(evaluate_job)

            baseline_metrics = File(f"metrics/{branch_id}/{branch_id}_baseline.json")
            baseline_job = (
                Job(
                    "train_centralized",
                    _id=f"baseline_{branch_id}",
                    node_label=f"baseline_{branch_id}",
                )
                .add_args("--config", config_file, "--output", baseline_metrics)
                .add_inputs(config_file, self.runtime_support_files["flwr_torch_utils"], *(shard for _, shard in client_shards))
                .add_outputs(baseline_metrics, stage_out=True, register_replica=False)
            )
            for _, shard in client_shards:
                baseline_job.add_args("--client-data", shard)
            self.wf.add_jobs(baseline_job)

            stats_metrics = File(f"metrics/{branch_id}/{branch_id}_stats.json")
            stats_job = (
                Job(
                    "compute_branch_stats",
                    _id=f"stats_{branch_id}",
                    node_label=f"stats_{branch_id}",
                )
                .add_args(
                    "--config",
                    config_file,
                    "--client-manifest",
                    client_manifest,
                    "--evaluation",
                    evaluation_metrics,
                    "--output",
                    stats_metrics,
                )
                .add_inputs(config_file, client_manifest, evaluation_metrics)
                .add_outputs(stats_metrics, stage_out=True, register_replica=False)
            )
            self.wf.add_jobs(stats_job)

            branch_outputs.append(
                {
                    "id": branch_id,
                    "config_file": config_file,
                    "model": current_global,
                    "client_shards": [shard for _, shard in client_shards],
                    "evaluation": evaluation_metrics,
                    "baseline": baseline_metrics,
                    "stats": stats_metrics,
                    "validation": validation,
                }
            )

        cross_eval_spec = File("manifests/branch_matrix_spec.json")
        cross_eval = File("metrics/cross_eval.json")
        cross_job = (
            Job("cross_eval", _id="cross_eval", node_label="cross_eval")
            .add_args("--matrix-spec", cross_eval_spec, "--output", cross_eval)
            .add_inputs(cross_eval_spec)
            .add_outputs(cross_eval, stage_out=True, register_replica=False)
        )
        for branch in branch_outputs:
            cross_job.add_args(
                "--branch",
                branch["id"],
                "--evaluation",
                branch["evaluation"],
                "--baseline",
                branch["baseline"],
                "--stats",
                branch["stats"],
            )
            cross_job.add_inputs(
                branch["evaluation"],
                branch["baseline"],
                branch["stats"],
                branch["config_file"],
                branch["model"],
                *branch["client_shards"],
                self.runtime_support_files["flwr_torch_utils"],
            )
        self.wf.add_jobs(cross_job)

        plot_summary = File("results/plot_summary.json")
        plot_job = (
            Job("plot_results", _id="plot_results", node_label="plot_results")
            .add_args("--cross-eval", cross_eval, "--output", plot_summary)
            .add_inputs(cross_eval)
            .add_outputs(plot_summary, stage_out=True, register_replica=False)
        )
        for branch in branch_outputs:
            plot_job.add_args("--branch-stats", branch["stats"])
            plot_job.add_inputs(branch["stats"])
        self.wf.add_jobs(plot_job)

        report = File("results/report.md")
        results = File("results/results.tar.gz")
        paper_tables = File("results/paper_tables.csv")
        provenance = File("results/provenance_summary.json")
        final_evaluation = File("metrics/final_evaluation.json")
        report_job = (
            Job("generate_report", _id="generate_report", node_label="generate_report")
            .add_args(
                "--config",
                File("configs/experiment.yaml"),
                "--cross-eval",
                cross_eval,
                "--plot-summary",
                plot_summary,
                "--report",
                report,
                "--results",
                results,
                "--paper-tables",
                paper_tables,
                "--provenance",
                provenance,
                "--final-evaluation",
                final_evaluation,
            )
            .add_inputs(File("configs/experiment.yaml"), cross_eval, plot_summary)
            .add_outputs(report, stage_out=True, register_replica=False)
            .add_outputs(results, stage_out=True, register_replica=False)
            .add_outputs(paper_tables, stage_out=True, register_replica=False)
            .add_outputs(provenance, stage_out=True, register_replica=False)
            .add_outputs(final_evaluation, stage_out=True, register_replica=False)
        )
        for branch in branch_outputs:
            report_job.add_args(
                "--branch",
                branch["id"],
                "--evaluation",
                branch["evaluation"],
                "--baseline",
                branch["baseline"],
                "--stats",
                branch["stats"],
                "--validation",
                branch["validation"],
            )
            report_job.add_inputs(
                branch["evaluation"],
                branch["baseline"],
                branch["stats"],
                branch["validation"],
            )
        self.wf.add_jobs(report_job)

    def create_workflow(self):
        if self.branch_pipelines:
            self.create_dual_branch_workflow()
            return

        self.wf = Workflow(self.wf_name, infer_dependencies=True)

        config_file = File("configs/experiment.yaml")
        dataset_manifest = File("manifests/dataset_manifest.csv")
        client_manifest = File("manifests/client_manifest.csv")
        splits = File("manifests/splits.json")

        if not self.static_manifest_paths:
            prepare_job = (
                Job("prepare_manifest", _id="prepare_manifest", node_label="prepare_manifest")
                .add_args(
                    "--config",
                    config_file,
                    "--dataset-manifest",
                    dataset_manifest,
                    "--client-manifest",
                    client_manifest,
                    "--splits",
                    splits,
                )
                .add_inputs(config_file)
                .add_outputs(dataset_manifest, stage_out=True, register_replica=False)
                .add_outputs(client_manifest, stage_out=True, register_replica=False)
                .add_outputs(splits, stage_out=True, register_replica=False)
            )
            self.wf.add_jobs(prepare_job)

        client_shards = []
        for client_spec in self.client_specs:
            client_id = client_spec["client_id"]
            shard = File(f"preprocessed/client_{client_id:03d}.jsonl")
            client_shards.append(shard)
            image_inputs = self.client_input_files.get(client_id, [])
            preprocess_job = (
                Job(
                    client_spec["tool"],
                    _id=f"preprocess_client_{client_id:03d}",
                    node_label=f"preprocess_client_{client_id:03d}",
                )
                .add_args(
                    "--config",
                    config_file,
                    "--client-manifest",
                    client_manifest,
                    "--client-id",
                    str(client_id),
                    "--output",
                    shard,
                )
                .add_inputs(
                    config_file,
                    client_manifest,
                    self.runtime_support_files["flwr_torch_utils"],
                    *image_inputs,
                )
                .add_outputs(shard, stage_out=False, register_replica=False)
                .add_pegasus_profiles(label=f"client_{client_id:03d}")
            )
            self.wf.add_jobs(preprocess_job)

        model_config = File("models/model_config.json")
        current_global = File("models/round_000_global.pt")
        init_job = (
            Job("initialize_model", _id="initialize_model", node_label="initialize_model")
            .add_args(
                "--config",
                config_file,
                "--model",
                current_global,
                "--model-config",
                model_config,
            )
            .add_inputs(config_file, self.runtime_support_files["flwr_torch_utils"])
            .add_outputs(current_global, stage_out=False, register_replica=False)
            .add_outputs(model_config, stage_out=True, register_replica=False)
        )
        self.wf.add_jobs(init_job)

        round_metrics = []
        for round_idx in range(1, self.rounds + 1):
            round_wf, next_global, aggregation_metrics = self.create_round_subworkflow(
                round_idx,
                config_file,
                self.config_path,
                current_global,
                list(enumerate(client_shards)),
            )
            round_job = (
                SubWorkflow(
                    round_wf,
                    _id=f"subwf_round_{round_idx:03d}",
                    node_label=f"subwf_round_{round_idx:03d}",
                )
                .add_inputs(config_file, current_global, *client_shards)
                .add_outputs(next_global, stage_out=False, register_replica=False)
                .add_outputs(aggregation_metrics, stage_out=True, register_replica=False)
            )
            self.wf.add_jobs(round_job)
            current_global = next_global
            round_metrics.append(aggregation_metrics)

        evaluation_metrics = File("metrics/final_evaluation.json")
        evaluate_job = (
            Job("evaluate", _id="evaluate_final", node_label="evaluate_final")
            .add_args(
                "--config",
                config_file,
                "--model",
                current_global,
                "--output",
                evaluation_metrics,
            )
            .add_inputs(
                config_file,
                current_global,
                self.runtime_support_files["flwr_torch_utils"],
                *client_shards,
            )
            .add_outputs(evaluation_metrics, stage_out=True, register_replica=False)
        )
        for shard in client_shards:
            evaluate_job.add_args("--client-data", shard)
        self.wf.add_jobs(evaluate_job)

        results = File("results/results.tar.gz")
        paper_tables = File("results/paper_tables.csv")
        provenance = File("results/provenance_summary.json")
        package_job = (
            Job("package_results", _id="package_results", node_label="package_results")
            .add_args(
                "--config",
                config_file,
                "--evaluation",
                evaluation_metrics,
                "--results",
                results,
                "--paper-tables",
                paper_tables,
                "--provenance",
                provenance,
            )
            .add_inputs(config_file, evaluation_metrics, *round_metrics)
            .add_outputs(results, stage_out=True, register_replica=False)
            .add_outputs(paper_tables, stage_out=True, register_replica=False)
            .add_outputs(provenance, stage_out=True, register_replica=False)
        )
        for metric in round_metrics:
            package_job.add_args("--round-metric", metric)
        self.wf.add_jobs(package_job)

    def write(self, skip_sites=False):
        if not skip_sites:
            self.sc.write()
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write(file=self.dagfile)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default="configs/experiment.yaml",
        help="Experiment configuration file.",
    )
    parser.add_argument(
        "-o", "--output", default="workflow.yml", help="Output workflow YAML file."
    )
    parser.add_argument(
        "-e",
        "--execution-site",
        default="condorpool",
        help="Pegasus execution site name.",
    )
    parser.add_argument(
        "-s",
        "--skip-sites",
        action="store_true",
        help="Do not write sites.yml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    for required in ("dataset_name", "rounds"):
        if required not in config:
            print(f"Missing required config key: {required}", file=sys.stderr)
            return 2
    if (
        "num_clients" not in config
        and not config.get("dataset_pipelines")
        and not config.get("branch_pipelines")
    ):
        print("Missing required config key: num_clients", file=sys.stderr)
        return 2

    workflow = FederatedLearningWorkflow(config, args.config, args.output)
    workflow.create_pegasus_properties()
    if not args.skip_sites:
        workflow.create_sites_catalog(args.execution_site)
    workflow.create_transformation_catalog(args.execution_site)
    workflow.create_replica_catalog()
    workflow.create_workflow()
    workflow.write(skip_sites=args.skip_sites)
    print(f"Wrote Pegasus workflow to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
