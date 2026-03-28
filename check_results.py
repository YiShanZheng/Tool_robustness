#!/usr/bin/env python3
"""
Validate the outputs produced by scripts/run_mvp_eval.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiments" / "mvp_vsp"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "external" / "logs" / "result_check_report.md"

REQUIRED_SAMPLE_FIELDS = {
    "task_id",
    "setting",
    "perturbation_type",
    "severity",
    "tool_correct",
    "params_correct",
    "valid_invocation",
    "task_success",
}

REQUIRED_METRIC_FIELDS = {
    "setting",
    "perturbation_type",
    "severity",
    "tool_selection_accuracy",
    "parameter_match_rate",
    "invocation_validity_rate",
    "task_success_rate",
}

DROP_FIELDS = {
    "tool_selection_drop",
    "parameter_match_drop",
    "invocation_validity_drop",
    "task_success_drop",
}


class ResultCheckError(RuntimeError):
    """Raised when result validation cannot proceed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the completeness and consistency of MVP evaluation outputs."
    )
    parser.add_argument(
        "--results-dir",
        "--input-dir",
        dest="results_dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing run_mvp_eval.py outputs.",
    )
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Where to write the markdown validation report.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResultCheckError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
        if not isinstance(item, dict):
            raise ResultCheckError(f"JSONL record at line {line_no} is not an object.")
        records.append(item)
    return records


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def check_sample_records(records: Sequence[Dict]) -> List[str]:
    issues: List[str] = []
    if not records:
        issues.append("sample_level_results.jsonl is empty.")
        return issues

    present_fields = set(records[0].keys())
    missing = sorted(REQUIRED_SAMPLE_FIELDS - present_fields)
    if missing:
        issues.append(f"sample_level_results.jsonl is missing required fields: {missing}")
    return issues


def check_metric_rows(rows: Sequence[Dict[str, str]]) -> List[str]:
    issues: List[str] = []
    if not rows:
        issues.append("aggregated_metrics.csv is empty.")
        return issues

    present_fields = set(rows[0].keys())
    missing = sorted(REQUIRED_METRIC_FIELDS - present_fields)
    if missing:
        issues.append(f"aggregated_metrics.csv is missing required fields: {missing}")

    clean_row = next((row for row in rows if row.get("setting") == "clean"), None)
    if clean_row is None:
        issues.append("aggregated_metrics.csv does not contain a clean row.")

    severities_present = {row.get("severity") for row in rows}
    if {"light", "medium", "heavy"} & severities_present:
        missing_drop_fields = sorted(DROP_FIELDS - present_fields)
        if missing_drop_fields:
            issues.append(
                "aggregated_metrics.csv contains perturbed severities but is missing drop fields: "
                f"{missing_drop_fields}"
            )
    return issues


def build_report(
    results_dir: Path,
    file_status: Dict[str, bool],
    sample_issues: Sequence[str],
    metric_issues: Sequence[str],
) -> str:
    all_issues = list(sample_issues) + list(metric_issues)
    lines = [
        "# Result Check Report",
        "",
        f"- Results directory: `{results_dir}`",
        "",
        "## File Presence",
        "",
    ]

    for name, exists in file_status.items():
        lines.append(f"- `{name}`: `{'OK' if exists else 'MISSING'}`")

    lines.extend(["", "## Validation Findings", ""])
    if all_issues:
        for issue in all_issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- All required files and fields passed the basic checks.")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    report_path = Path(args.report_path)

    sample_path = results_dir / "sample_level_results.jsonl"
    metrics_path = results_dir / "aggregated_metrics.csv"
    summary_path = results_dir / "experiment_summary.md"

    file_status = {
        "sample_level_results.jsonl": sample_path.exists(),
        "aggregated_metrics.csv": metrics_path.exists(),
        "experiment_summary.md": summary_path.exists(),
    }

    sample_issues: List[str] = []
    metric_issues: List[str] = []

    try:
        if sample_path.exists():
            sample_records = load_jsonl(sample_path)
            sample_issues.extend(check_sample_records(sample_records))
        else:
            sample_issues.append("sample_level_results.jsonl does not exist.")

        if metrics_path.exists():
            metric_rows = load_csv(metrics_path)
            metric_issues.extend(check_metric_rows(metric_rows))
        else:
            metric_issues.append("aggregated_metrics.csv does not exist.")

        if not summary_path.exists():
            metric_issues.append("experiment_summary.md does not exist.")

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            build_report(results_dir, file_status, sample_issues, metric_issues),
            encoding="utf-8",
        )

        print(f"Report written to: {report_path}")
        if sample_issues or metric_issues:
            print("Result checks completed with issues.")
            return 1
        print("Result checks passed.")
        return 0
    except ResultCheckError as exc:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            build_report(results_dir, file_status, [str(exc)], metric_issues),
            encoding="utf-8",
        )
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Report written to: {report_path}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            build_report(results_dir, file_status, [f"Unhandled error: {exc}"], metric_issues),
            encoding="utf-8",
        )
        print(f"Unhandled error: {exc}", file=sys.stderr)
        print(f"Report written to: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
