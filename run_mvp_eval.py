#!/usr/bin/env python3
"""
Run or orchestrate the stage-1 MVP robustness evaluation.

This script is intentionally conservative:
- It supports a mock mode for local pipeline validation.
- It provides an adapter mode scaffold for future integration with the local
  AdaReasoner evaluation stack under external/AdaReasoner/tool_server/tf_eval/.
- It does not modify the external/AdaReasoner repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAREASONER_ROOT = PROJECT_ROOT / "external" / "AdaReasoner"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "external" / "models"
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "external" / "datasets"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "external" / "logs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "mvp_vsp"
PERTURBATION_ENGINE = PROJECT_ROOT / "scripts" / "perturbation_engine.py"
TF_EVAL_DIR = DEFAULT_ADAREASONER_ROOT / "tool_server" / "tf_eval"
VSP_CONFIG_PATH = TF_EVAL_DIR / "tasks" / "vsp" / "config.yaml"

SEVERITIES = ("light", "medium", "heavy")
PERTURBATION_TYPES = (
    "tool_name_rename",
    "tool_name_alias",
    "parameter_name_rename",
    "parameter_order_shuffle",
    "task_description_paraphrase_light",
    "task_description_compress_light",
    "tool_description_rewrite_light",
)


class MVPEvalError(RuntimeError):
    """Raised when experiment setup or execution fails."""


@dataclass
class Setting:
    name: str
    perturbation_type: str
    severity: str
    is_clean: bool
    sample_path: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or orchestrate the stage-1 MVP robustness evaluation."
    )
    parser.add_argument("--input", required=True, help="Path to clean JSON or JSONL samples.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--perturbation-type",
        default="tool_name_rename",
        choices=PERTURBATION_TYPES,
        help="Perturbation type to evaluate.",
    )
    parser.add_argument(
        "--severity",
        choices=SEVERITIES,
        help="Single perturbation severity to evaluate.",
    )
    parser.add_argument(
        "--all-severities",
        action="store_true",
        help="Evaluate clean plus light/medium/heavy for the given perturbation type.",
    )
    parser.add_argument(
        "--use-existing-perturbed",
        action="store_true",
        help="Reuse existing perturbed files under the output directory if present.",
    )
    parser.add_argument(
        "--model-path",
        help="Optional model directory for later adapter-mode integration.",
    )
    parser.add_argument(
        "--dataset-path",
        help="Optional dataset directory override. Defaults to --input parent.",
    )
    parser.add_argument(
        "--task-type",
        default="vsp",
        help="Task type label. Stage-1 defaults to VSP.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible orchestration and mock mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned experiment settings without executing evaluation.",
    )
    parser.add_argument(
        "--runner-mode",
        default="mock",
        choices=("mock", "adapter"),
        help="mock validates the pipeline locally; adapter is reserved for a real tf_eval bridge.",
    )
    parser.add_argument(
        "--adapter-command",
        help=(
            "Optional command template for adapter mode. Placeholders supported: "
            "{sample_file}, {task_type}, {model_path}, {dataset_path}, {setting}."
        ),
    )
    return parser.parse_args()


def load_samples(path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        raise MVPEvalError(f"Input path does not exist: {path}")

    raw_text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(raw_text)
        if isinstance(data, dict):
            samples = [data]
        elif isinstance(data, list):
            samples = data
        else:
            raise MVPEvalError("JSON input must be an object or a list of objects.")
    elif suffix == ".jsonl":
        samples = []
        for line_no, line in enumerate(raw_text.splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise MVPEvalError(f"JSONL line {line_no} is not an object.")
            samples.append(item)
    else:
        raise MVPEvalError("Input must be .json or .jsonl")

    normalized = [sample for sample in samples if isinstance(sample, dict)]
    if max_samples is not None:
        normalized = normalized[:max_samples]
    return normalized


def build_experiment_settings(
    perturbation_type: str,
    severity: Optional[str],
    all_severities: bool,
) -> List[Setting]:
    settings = [Setting(name="clean", perturbation_type="clean", severity="clean", is_clean=True)]

    if all_severities:
        levels = list(SEVERITIES)
    elif severity:
        levels = [severity]
    else:
        levels = ["light"]

    for level in levels:
        settings.append(
            Setting(
                name=f"{perturbation_type}:{level}",
                perturbation_type=perturbation_type,
                severity=level,
                is_clean=False,
            )
        )
    return settings


def infer_task_id(sample: Dict[str, Any], index: int) -> str:
    for key in ("task_id", "id", "sample_id", "uid"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return f"sample_{index}"


def get_tool_items(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("tool_registry", "tools"):
        container = sample.get(key)
        if isinstance(container, list):
            return [item for item in container if isinstance(item, dict)]
        if isinstance(container, dict):
            normalized: List[Dict[str, Any]] = []
            for name, item in container.items():
                if isinstance(item, dict):
                    clone = dict(item)
                    clone.setdefault("name", name)
                    normalized.append(clone)
            return normalized
    return []


def get_gold_tool(sample: Dict[str, Any]) -> str:
    for key in ("gold_tool", "expected_tool", "target_tool", "tool"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value

    tool_items = get_tool_items(sample)
    if tool_items:
        return str(tool_items[0].get("name") or tool_items[0].get("tool_name") or "")
    return ""


def get_gold_params(sample: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("gold_params", "expected_params", "target_params", "params"):
        value = sample.get(key)
        if isinstance(value, dict):
            return value

    tool_items = get_tool_items(sample)
    if tool_items:
        params = tool_items[0].get("parameters") or tool_items[0].get("args") or tool_items[0].get("schema")
        if isinstance(params, dict):
            return {str(k): v for k, v in params.items()}
    return {}


def stable_random(task_id: str, setting: Setting, seed: int) -> random.Random:
    payload = f"{task_id}|{setting.name}|{seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return random.Random(int(digest[:16], 16))


def mock_predict(
    sample: Dict[str, Any],
    setting: Setting,
    seed: int,
) -> Tuple[str, Dict[str, Any], bool, bool, str]:
    gold_tool = get_gold_tool(sample)
    gold_params = get_gold_params(sample)

    if setting.is_clean:
        return gold_tool, gold_params, True, True, "mock_clean"

    rng = stable_random(gold_tool or "unknown", setting, seed)
    degrade_prob = {
        "light": 0.15,
        "medium": 0.35,
        "heavy": 0.60,
    }[setting.severity]

    tool_ok = rng.random() > degrade_prob
    params_ok = rng.random() > (degrade_prob + 0.05)
    invocation_ok = rng.random() > (degrade_prob - 0.05 if degrade_prob > 0.05 else 0.02)
    success_ok = tool_ok and params_ok and invocation_ok

    pred_tool = gold_tool if tool_ok else f"pred_{setting.perturbation_type}_{setting.severity}"
    pred_params = gold_params if params_ok else {"mismatch": True, "severity": setting.severity}

    return pred_tool, pred_params, invocation_ok, success_ok, "mock_perturbed"


def adapter_predict(
    sample: Dict[str, Any],
    setting: Setting,
    args: argparse.Namespace,
    sample_file: Path,
) -> Tuple[str, Dict[str, Any], bool, bool, str, str]:
    if not args.adapter_command:
        return "", {}, False, False, "", (
            "Adapter mode scaffold is enabled, but no --adapter-command was provided. "
            "Manual integration with external/AdaReasoner/tool_server/tf_eval is still needed."
        )

    command = args.adapter_command.format(
        sample_file=str(sample_file),
        task_type=args.task_type,
        model_path=args.model_path or "",
        dataset_path=args.dataset_path or "",
        setting=setting.name,
    )
    try:
        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return "", {}, False, False, "", f"Adapter command failed to start: {exc}"

    if completed.returncode != 0:
        return "", {}, False, False, completed.stdout, (
            f"Adapter command exited with code {completed.returncode}: {completed.stderr.strip()}"
        )

    # Adapter parsing is intentionally conservative. A real implementation should
    # convert model/tf_eval outputs into the normalized sample-level fields below.
    return "", {}, False, False, completed.stdout, (
        "Adapter command ran, but output parsing is not implemented yet. "
        "Please map AdaReasoner/tf_eval output into pred_tool / pred_params / validity / success."
    )


def run_single_sample(
    sample: Dict[str, Any],
    setting: Setting,
    args: argparse.Namespace,
    sample_file: Optional[Path] = None,
    index: int = 0,
) -> Dict[str, Any]:
    task_id = infer_task_id(sample, index)
    gold_tool = get_gold_tool(sample)
    gold_params = get_gold_params(sample)
    error_message = ""

    if args.runner_mode == "mock":
        pred_tool, pred_params, valid_invocation, task_success, raw_output = mock_predict(
            sample, setting, args.seed
        )
    else:
        pred_tool, pred_params, valid_invocation, task_success, raw_output, error_message = (
            adapter_predict(sample, setting, args, sample_file or Path(args.input))
        )

    tool_correct = bool(gold_tool) and pred_tool == gold_tool
    params_correct = bool(gold_params) and pred_params == gold_params

    return {
        "task_id": task_id,
        "setting": setting.name,
        "perturbation_type": setting.perturbation_type,
        "severity": setting.severity,
        "gold_tool": gold_tool,
        "pred_tool": pred_tool,
        "tool_correct": tool_correct,
        "gold_params": gold_params,
        "pred_params": pred_params,
        "params_correct": params_correct,
        "valid_invocation": bool(valid_invocation),
        "task_success": bool(task_success),
        "raw_output": raw_output,
        "error_message": error_message,
    }


def aggregate_metrics(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        grouped.setdefault(item["setting"], []).append(item)

    aggregated: List[Dict[str, Any]] = []
    clean_metrics: Optional[Dict[str, float]] = None

    for setting_name, rows in grouped.items():
        total = len(rows) or 1
        metrics = {
            "setting": setting_name,
            "perturbation_type": rows[0]["perturbation_type"],
            "severity": rows[0]["severity"],
            "sample_count": len(rows),
            "tool_selection_accuracy": sum(1 for r in rows if r["tool_correct"]) / total,
            "parameter_match_rate": sum(1 for r in rows if r["params_correct"]) / total,
            "invocation_validity_rate": sum(1 for r in rows if r["valid_invocation"]) / total,
            "task_success_rate": sum(1 for r in rows if r["task_success"]) / total,
        }
        aggregated.append(metrics)
        if setting_name == "clean":
            clean_metrics = metrics

    aggregated.sort(key=lambda item: _severity_rank(item["severity"]))

    if clean_metrics:
        for item in aggregated:
            item["tool_selection_drop"] = clean_metrics["tool_selection_accuracy"] - item["tool_selection_accuracy"]
            item["parameter_match_drop"] = clean_metrics["parameter_match_rate"] - item["parameter_match_rate"]
            item["invocation_validity_drop"] = (
                clean_metrics["invocation_validity_rate"] - item["invocation_validity_rate"]
            )
            item["task_success_drop"] = clean_metrics["task_success_rate"] - item["task_success_rate"]
    else:
        for item in aggregated:
            item["tool_selection_drop"] = None
            item["parameter_match_drop"] = None
            item["invocation_validity_drop"] = None
            item["task_success_drop"] = None

    return aggregated


def _severity_rank(value: str) -> int:
    order = {"clean": 0, "light": 1, "medium": 2, "heavy": 3}
    return order.get(value, 99)


def write_reports(
    output_dir: Path,
    sample_results: Sequence[Dict[str, Any]],
    aggregated_metrics: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_path = output_dir / "sample_level_results.jsonl"
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in sample_results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_path = output_dir / "aggregated_metrics.csv"
    fieldnames = [
        "setting",
        "perturbation_type",
        "severity",
        "sample_count",
        "tool_selection_accuracy",
        "parameter_match_rate",
        "invocation_validity_rate",
        "task_success_rate",
        "tool_selection_drop",
        "parameter_match_drop",
        "invocation_validity_drop",
        "task_success_drop",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_metrics:
            writer.writerow(row)

    summary_path = output_dir / "experiment_summary.md"
    summary_path.write_text(
        build_experiment_summary(aggregated_metrics, args),
        encoding="utf-8",
    )


def build_experiment_summary(metrics: Sequence[Dict[str, Any]], args: argparse.Namespace) -> str:
    clean = next((item for item in metrics if item["setting"] == "clean"), None)
    lines = [
        "# MVP Evaluation Summary",
        "",
        f"- Runner mode: `{args.runner_mode}`",
        f"- Task type: `{args.task_type}`",
        f"- Perturbation type: `{args.perturbation_type}`",
        f"- Input: `{args.input}`",
        f"- Model path: `{args.model_path or 'not provided'}`",
        f"- Dataset path: `{args.dataset_path or 'not provided'}`",
        f"- Local tf_eval path: `{TF_EVAL_DIR}`",
        f"- Local VSP config path: `{VSP_CONFIG_PATH}`",
        "",
        "## Clean Metrics",
        "",
    ]

    if clean:
        lines.extend(_metric_block(clean))
    else:
        lines.append("- Clean metrics were not produced.")

    lines.extend(["", "## Perturbation Metrics", ""])
    for item in metrics:
        if item["setting"] == "clean":
            continue
        lines.append(f"### {item['setting']}")
        lines.extend(_metric_block(item))
        lines.append("")

    trend = infer_trend(metrics)
    lines.extend(
        [
            "## Trend Assessment",
            "",
            trend,
            "",
            "## Adapter Notes",
            "",
            "- Mock mode is intended only for local pipeline validation.",
            "- Adapter mode still needs a concrete bridge to the local AdaReasoner evaluation path under `external/AdaReasoner/tool_server/tf_eval/`.",
            "- A future adapter should translate real model/tf_eval outputs into the normalized fields used here: `pred_tool`, `pred_params`, `valid_invocation`, and `task_success`.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _metric_block(item: Dict[str, Any]) -> List[str]:
    return [
        f"- `tool_selection_accuracy`: `{item['tool_selection_accuracy']:.4f}`",
        f"- `parameter_match_rate`: `{item['parameter_match_rate']:.4f}`",
        f"- `invocation_validity_rate`: `{item['invocation_validity_rate']:.4f}`",
        f"- `task_success_rate`: `{item['task_success_rate']:.4f}`",
        f"- `tool_selection_drop`: `{_fmt_optional(item['tool_selection_drop'])}`",
        f"- `parameter_match_drop`: `{_fmt_optional(item['parameter_match_drop'])}`",
        f"- `invocation_validity_drop`: `{_fmt_optional(item['invocation_validity_drop'])}`",
        f"- `task_success_drop`: `{_fmt_optional(item['task_success_drop'])}`",
    ]


def _fmt_optional(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def infer_trend(metrics: Sequence[Dict[str, Any]]) -> str:
    ordered = [item for item in metrics if item["severity"] in SEVERITIES]
    if len(ordered) < 2:
        return "- Not enough perturbed severities to judge a trend."

    drops = [item["task_success_drop"] for item in ordered if item["task_success_drop"] is not None]
    if len(drops) < 2:
        return "- Task success drops are unavailable, so severity trend cannot be judged."

    monotonic = all(left <= right for left, right in zip(drops, drops[1:]))
    if monotonic:
        return "- A stronger perturbation is associated with greater or equal task-success degradation in these results."
    return "- No clear monotonic 'stronger perturbation -> larger degradation' trend was observed in these results."


def maybe_plot_results(output_dir: Path, metrics: Sequence[Dict[str, Any]]) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    ordered = sorted(metrics, key=lambda item: _severity_rank(item["severity"]))
    labels = [item["severity"] for item in ordered]
    y_values = [item["task_success_rate"] for item in ordered]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(labels, y_values, marker="o", label="task_success_rate")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Performance")
    ax.set_title("Performance Drop by Severity")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plot_path = output_dir / "performance_drop_by_severity.png"
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_perturbed_samples(
    input_path: Path,
    output_path: Path,
    perturbation_type: str,
    severity: str,
    seed: int,
) -> Path:
    cmd = [
        sys.executable,
        str(PERTURBATION_ENGINE),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--perturbation-type",
        perturbation_type,
        "--severity",
        severity,
        "--seed",
        str(seed),
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise MVPEvalError(
            f"Perturbation generation failed for {perturbation_type}/{severity}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return output_path


def load_perturbed_records(path: Path, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    records = load_samples(path, max_samples=max_samples)
    extracted: List[Dict[str, Any]] = []
    for record in records:
        if "perturbed_sample" in record and isinstance(record["perturbed_sample"], dict):
            extracted.append(record["perturbed_sample"])
        else:
            extracted.append(record)
    return extracted


def materialize_setting_samples(
    settings: Sequence[Setting],
    args: argparse.Namespace,
    clean_input: Path,
    output_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    sample_map: Dict[str, List[Dict[str, Any]]] = {}
    clean_samples = load_samples(clean_input, max_samples=args.max_samples)
    sample_map["clean"] = clean_samples

    for setting in settings:
        if setting.is_clean:
            continue

        perturbed_path = output_dir / f"perturbed_{setting.perturbation_type}_{setting.severity}.json"
        setting.sample_path = perturbed_path

        if args.use_existing_perturbed and perturbed_path.exists():
            sample_map[setting.name] = load_perturbed_records(
                perturbed_path, max_samples=args.max_samples
            )
            continue

        generate_perturbed_samples(
            input_path=clean_input,
            output_path=perturbed_path,
            perturbation_type=setting.perturbation_type,
            severity=setting.severity,
            seed=args.seed,
        )
        sample_map[setting.name] = load_perturbed_records(
            perturbed_path, max_samples=args.max_samples
        )

    return sample_map


def run_experiment(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task_type.lower() != "vsp":
        raise MVPEvalError("Stage-1 MVP currently targets VSP. Use --task-type vsp.")

    dataset_path = args.dataset_path or str(input_path.parent)
    args.dataset_path = dataset_path

    settings = build_experiment_settings(
        perturbation_type=args.perturbation_type,
        severity=args.severity,
        all_severities=args.all_severities,
    )

    if args.dry_run:
        plan = {
            "input": str(input_path),
            "output_dir": str(output_dir),
            "runner_mode": args.runner_mode,
            "task_type": args.task_type,
            "settings": [setting.__dict__ for setting in settings],
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "tf_eval_path": str(TF_EVAL_DIR),
            "vsp_config_path": str(VSP_CONFIG_PATH),
        }
        save_json(output_dir / "dry_run_plan.json", plan)
        (output_dir / "experiment_summary.md").write_text(
            "\n".join(
                [
                    "# MVP Evaluation Dry Run",
                    "",
                    f"- Runner mode: `{args.runner_mode}`",
                    f"- Task type: `{args.task_type}`",
                    f"- Perturbation type: `{args.perturbation_type}`",
                    f"- Planned settings: `{', '.join(setting.name for setting in settings)}`",
                    f"- Input: `{input_path}`",
                    f"- Output dir: `{output_dir}`",
                    "",
                    "## Note",
                    "",
                    "- This was a dry run.",
                    "- No sample-level evaluation was executed.",
                    "- `sample_level_results.jsonl` and `aggregated_metrics.csv` are not expected in dry-run mode.",
                    "- Use mock mode without `--dry-run` for a local end-to-end pipeline check.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(f"Dry-run plan written to: {output_dir / 'dry_run_plan.json'}")
        print(f"Dry-run summary written to: {output_dir / 'experiment_summary.md'}")
        return

    sample_map = materialize_setting_samples(
        settings=settings,
        args=args,
        clean_input=input_path,
        output_dir=output_dir,
    )

    sample_results: List[Dict[str, Any]] = []
    for setting in settings:
        current_samples = sample_map[setting.name]
        for index, sample in enumerate(current_samples):
            sample_results.append(
                run_single_sample(
                    sample=sample,
                    setting=setting,
                    args=args,
                    sample_file=setting.sample_path or input_path,
                    index=index,
                )
            )

    aggregated = aggregate_metrics(sample_results)
    write_reports(output_dir, sample_results, aggregated, args)
    plot_path = maybe_plot_results(output_dir, aggregated)

    print(f"Sample-level results: {output_dir / 'sample_level_results.jsonl'}")
    print(f"Aggregated metrics: {output_dir / 'aggregated_metrics.csv'}")
    print(f"Experiment summary: {output_dir / 'experiment_summary.md'}")
    if plot_path:
        print(f"Plot: {plot_path}")
    else:
        print("Plot: skipped (matplotlib not available)")


def main() -> int:
    args = parse_args()
    try:
        run_experiment(args)
        return 0
    except MVPEvalError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive boundary
        print(f"Unhandled error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
