"""
Structured JSON logging for precision experiments.

NO WANDB - we use a custom JSON reporting system for full control over
data format and plotting.

Output structure:
    results/
    ├── experiment_manifest.json      # Master index of all runs
    ├── runs/
    │   ├── bf16_pure_seed0/
    │   │   ├── config.json           # Frozen config for this run
    │   │   ├── step_metrics.jsonl    # Per-step metrics (one JSON per line)
    │   │   └── run_summary.json      # Final summary (written even on failure)
    │   └── ...
    ├── comparisons/
    │   └── precision_errors.jsonl    # BF16 vs FP32 comparisons
    └── plots/
        └── ...
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch

from ..metrics.diagnostics import RunSummary


class ExperimentLogger:
    """
    Structured JSON logger for precision experiments.

    Usage:
        logger = ExperimentLogger("results/", "bf16_pure_seed0")
        logger.log_config(config)

        for step in range(max_steps):
            metrics = train_step()
            logger.log_step(step, metrics)

        logger.finalize(summary)
    """

    def __init__(self, output_dir: str, run_id: str):
        """
        Initialize logger.

        Args:
            output_dir: Base output directory
            run_id: Unique run identifier
        """
        self.run_dir = Path(output_dir) / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id
        self.step_file = open(self.run_dir / "step_metrics.jsonl", "w")
        self.start_time = datetime.now()

    def log_config(self, config: Dict[str, Any]):
        """
        Write frozen config at start of run.

        Args:
            config: Configuration dictionary
        """
        config = config.copy()
        config["run_id"] = self.run_id
        config["started_at"] = self.start_time.isoformat()

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    def log_step(self, step: int, metrics: Dict[str, Any]):
        """
        Append step metrics to JSONL file.

        Args:
            step: Training step number
            metrics: Metrics dictionary
        """
        metrics = metrics.copy()
        metrics["step"] = step
        metrics["timestamp"] = datetime.now().isoformat()

        # Handle non-JSON-serializable values
        for key, value in list(metrics.items()):
            if isinstance(value, float):
                if value == float('inf'):
                    metrics[key] = "inf"
                elif value == float('-inf'):
                    metrics[key] = "-inf"
                elif value != value:  # NaN check
                    metrics[key] = "nan"

        self.step_file.write(json.dumps(metrics) + "\n")
        self.step_file.flush()  # Ensure data is written even on crash

    def log_precision_comparison(self, step: int, component: str, errors: Dict[str, float]):
        """
        Log BF16 vs FP32 precision errors.

        Args:
            step: Training step
            component: Component name (e.g., "kl_penalty", "log_ratio")
            errors: Error metrics dictionary
        """
        comparison_dir = self.run_dir.parent.parent / "comparisons"
        comparison_dir.mkdir(exist_ok=True)

        comparison_file = comparison_dir / "precision_errors.jsonl"

        record = {
            "step": step,
            "component": component,
            "run_id": self.run_id,
            **errors
        }
        with open(comparison_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def finalize(self, summary: RunSummary):
        """
        Write final summary and close files.

        Args:
            summary: RunSummary dataclass
        """
        self.step_file.close()

        summary_dict = summary.to_structured_dict()
        summary_dict["timestamps"] = {
            "started_at": self.start_time.isoformat(),
            "completed_at": datetime.now().isoformat()
        }

        with open(self.run_dir / "run_summary.json", "w") as f:
            json.dump(summary_dict, f, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure files are closed even on exception."""
        if not self.step_file.closed:
            self.step_file.close()


class ExperimentManifest:
    """
    Manages the master experiment index.

    Tracks all runs, their status, and provides easy lookup.
    """

    def __init__(self, output_dir: str, experiment_name: str):
        """
        Initialize or load manifest.

        Args:
            output_dir: Base output directory
            experiment_name: Name for this experiment
        """
        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "experiment_manifest.json"
        self.experiment_name = experiment_name

        self._init_manifest()

    def _init_manifest(self):
        """Load existing manifest or create new one."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {
                "experiment_name": self.experiment_name,
                "created_at": datetime.now().isoformat(),
                "hardware": self._get_hardware_info(),
                "runs": [],
                "configs_tested": [],
                "seeds": []
            }
            self._save()

    def _get_hardware_info(self) -> Dict[str, str]:
        """Get hardware information for manifest."""
        return {
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
            "cuda_version": str(torch.version.cuda) if torch.cuda.is_available() else "N/A",
            "pytorch_version": torch.__version__
        }

    def register_run(self, run_id: str, config_name: str, seed: int):
        """
        Register a new run at start.

        Args:
            run_id: Unique run identifier
            config_name: Precision config name
            seed: Random seed
        """
        self.manifest["runs"].append({
            "run_id": run_id,
            "config_name": config_name,
            "seed": seed,
            "status": "running",
            "path": f"runs/{run_id}/",
            "started_at": datetime.now().isoformat()
        })

        if config_name not in self.manifest["configs_tested"]:
            self.manifest["configs_tested"].append(config_name)
        if seed not in self.manifest["seeds"]:
            self.manifest["seeds"].append(seed)

        self._save()

    def complete_run(
        self,
        run_id: str,
        status: str,
        failure_step: Optional[int] = None
    ):
        """
        Update run status on completion.

        Args:
            run_id: Run identifier
            status: Final status (e.g., "completed", "failed:nan_loss")
            failure_step: Step number where failure occurred (if applicable)
        """
        for run in self.manifest["runs"]:
            if run["run_id"] == run_id:
                run["status"] = status
                run["completed_at"] = datetime.now().isoformat()
                if failure_step is not None:
                    run["failure_step"] = failure_step
                break
        self._save()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run info by ID."""
        for run in self.manifest["runs"]:
            if run["run_id"] == run_id:
                return run
        return None

    def get_runs_by_config(self, config_name: str) -> list:
        """Get all runs for a given config."""
        return [r for r in self.manifest["runs"] if r["config_name"] == config_name]

    def _save(self):
        """Save manifest to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)
