"""
Data loaders for experiment analysis.

Provides utilities to load step metrics, run summaries, and
precision comparisons into pandas DataFrames for analysis.
"""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


def load_step_metrics(run_dir: str) -> pd.DataFrame:
    """
    Load step metrics from JSONL into DataFrame.

    Args:
        run_dir: Path to run directory containing step_metrics.jsonl

    Returns:
        DataFrame with one row per step
    """
    records = []
    metrics_path = Path(run_dir) / "step_metrics.jsonl"

    if not metrics_path.exists():
        return pd.DataFrame()

    with open(metrics_path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Convert special string values back
                for key, value in list(record.items()):
                    if value == "inf":
                        record[key] = float('inf')
                    elif value == "-inf":
                        record[key] = float('-inf')
                    elif value == "nan":
                        record[key] = float('nan')
                records.append(record)

    return pd.DataFrame(records)


def load_all_runs(
    output_dir: str,
    config_filter: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load step metrics from all runs into single DataFrame.

    Adds 'run_id', 'config_name', and 'seed' columns for grouping.

    Args:
        output_dir: Base output directory containing experiment_manifest.json
        config_filter: Optional list of config names to include

    Returns:
        Combined DataFrame with all step metrics
    """
    manifest_path = Path(output_dir) / "experiment_manifest.json"

    if not manifest_path.exists():
        print(f"Warning: No manifest found at {manifest_path}")
        return pd.DataFrame()

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_dfs = []
    for run in manifest["runs"]:
        if config_filter and run["config_name"] not in config_filter:
            continue

        run_dir = Path(output_dir) / run["path"]
        df = load_step_metrics(run_dir)

        if len(df) > 0:
            df["run_id"] = run["run_id"]
            df["config_name"] = run["config_name"]
            df["seed"] = run["seed"]
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def load_summaries(output_dir: str) -> pd.DataFrame:
    """
    Load all run summaries into DataFrame for comparison.

    Args:
        output_dir: Base output directory

    Returns:
        DataFrame with one row per run, columns flattened from nested dicts
    """
    manifest_path = Path(output_dir) / "experiment_manifest.json"

    if not manifest_path.exists():
        return pd.DataFrame()

    with open(manifest_path) as f:
        manifest = json.load(f)

    summaries = []
    for run in manifest["runs"]:
        summary_path = Path(output_dir) / run["path"] / "run_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
                # Flatten nested dicts for DataFrame
                flat = {"run_id": run["run_id"], "config_name": run["config_name"]}
                for key, value in summary.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            flat[f"{key}_{k}"] = v
                    else:
                        flat[key] = value
                summaries.append(flat)

    return pd.DataFrame(summaries)


def load_precision_errors(output_dir: str) -> pd.DataFrame:
    """
    Load precision error comparisons from JSONL.

    Args:
        output_dir: Base output directory

    Returns:
        DataFrame with precision error records
    """
    errors_path = Path(output_dir) / "comparisons" / "precision_errors.jsonl"

    if not errors_path.exists():
        return pd.DataFrame()

    records = []
    with open(errors_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return pd.DataFrame(records)


def load_run_config(run_dir: str) -> dict:
    """
    Load frozen config for a run.

    Args:
        run_dir: Path to run directory

    Returns:
        Config dictionary
    """
    config_path = Path(run_dir) / "config.json"

    if not config_path.exists():
        return {}

    with open(config_path) as f:
        return json.load(f)


def get_completed_runs(output_dir: str) -> List[str]:
    """
    Get list of completed run IDs.

    Args:
        output_dir: Base output directory

    Returns:
        List of run_id strings for completed runs
    """
    manifest_path = Path(output_dir) / "experiment_manifest.json"

    if not manifest_path.exists():
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    return [
        run["run_id"]
        for run in manifest["runs"]
        if run["status"] == "completed"
    ]


def get_failed_runs(output_dir: str) -> List[dict]:
    """
    Get list of failed runs with failure details.

    Args:
        output_dir: Base output directory

    Returns:
        List of dicts with run_id, status, and failure_step
    """
    manifest_path = Path(output_dir) / "experiment_manifest.json"

    if not manifest_path.exists():
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    return [
        {
            "run_id": run["run_id"],
            "config_name": run["config_name"],
            "status": run["status"],
            "failure_step": run.get("failure_step"),
        }
        for run in manifest["runs"]
        if run["status"] != "completed" and run["status"] != "running"
    ]
