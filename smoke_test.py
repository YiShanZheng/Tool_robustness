#!/usr/bin/env python3
"""
Minimal smoke test for the AdaReasoner MVP setup.

This script checks local paths, lightly inspects dataset/model directories,
optionally attempts a lightweight model metadata load, and writes a markdown
report. It does not run formal training or evaluation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "external"
ADAREASONER_DIR = EXTERNAL_DIR / "AdaReasoner"
TOOL_SERVER_DIR = ADAREASONER_DIR / "tool_server"
TF_EVAL_DIR = TOOL_SERVER_DIR / "tf_eval"
VSP_CONFIG = TF_EVAL_DIR / "tasks" / "vsp" / "config.yaml"
EXAMPLE_CONFIGS_DIR = TF_EVAL_DIR / "examples" / "configs"
REQUIREMENTS_DIR = ADAREASONER_DIR / "requirements"
DATASETS_DIR = EXTERNAL_DIR / "datasets"
MODELS_DIR = EXTERNAL_DIR / "models"
LOGS_DIR = EXTERNAL_DIR / "logs"
REPORT_PATH = LOGS_DIR / "smoke_test_report.md"

VSP_DATASET_NAME_HINTS = ("vsp", "visual-spatial-planning", "adareasoner-tc-vsp")
MODEL_NAME_HINTS = ("adareasoner",)
MODEL_KEY_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "model.safetensors",
    "pytorch_model.bin",
)
IMPORT_CANDIDATES = (
    "yaml",
    "huggingface_hub",
    "datasets",
    "transformers",
    "torch",
)


class SmokeTestError(RuntimeError):
    """Raised when the smoke test hits a fatal setup issue."""


@dataclass
class CheckResult:
    label: str
    path: Path
    exists: bool
    note: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal local smoke test for the AdaReasoner MVP setup."
    )
    parser.add_argument(
        "--try-load-model",
        action="store_true",
        help="Attempt lightweight model metadata loading if a candidate model directory is found.",
    )
    return parser.parse_args()


def find_candidate_dirs(base_dir: Path, hints: Sequence[str]) -> List[Path]:
    if not base_dir.exists():
        return []
    candidates: List[Path] = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        normalized = child.name.lower()
        if any(hint in normalized for hint in hints):
            candidates.append(child)
    return candidates


def list_dir_tree(root: Path, max_depth: int = 2, max_entries_per_level: int = 12) -> List[str]:
    lines: List[str] = []
    if not root.exists():
        return lines

    def walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception as exc:
            lines.append(f"- {current.name}/ <unreadable: {exc}>")
            return

        shown = 0
        for child in children:
            if shown >= max_entries_per_level:
                lines.append(f"{'  ' * depth}- ...")
                break
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{'  ' * depth}- {child.name}{suffix}")
            shown += 1
            if child.is_dir():
                walk(child, depth + 1)

    lines.append(f"- {root.name}/")
    walk(root, 1)
    return lines


def try_imports() -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []
    for module_name in IMPORT_CANDIDATES:
        try:
            importlib.import_module(module_name)
            results.append((module_name, True, "ok"))
        except Exception as exc:
            results.append((module_name, False, str(exc)))
    return results


def sniff_json_like_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return summarize_sample_data(data)
    if suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                data = json.loads(line)
                return summarize_sample_data(data)
        return {"note": "jsonl file is empty"}
    raise SmokeTestError(f"Unsupported sample sniff file: {path}")


def summarize_sample_data(data: Any) -> Dict[str, Any]:
    if isinstance(data, list):
        if not data:
            return {"container_type": "list", "note": "empty list"}
        first = data[0]
        return {
            "container_type": "list",
            "first_item_type": type(first).__name__,
            "first_item_keys": sorted(first.keys())[:20] if isinstance(first, dict) else None,
        }
    if isinstance(data, dict):
        return {
            "container_type": "dict",
            "keys": sorted(data.keys())[:30],
        }
    return {"container_type": type(data).__name__}


def find_sample_files(root: Path) -> List[Path]:
    matches: List[Path] = []
    for pattern in ("*.json", "*.jsonl"):
        matches.extend(sorted(root.rglob(pattern)))
    return matches[:8]


def inspect_dataset_dir(dataset_dir: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(dataset_dir),
        "tree": list_dir_tree(dataset_dir),
        "sample_files": [],
    }

    for sample_file in find_sample_files(dataset_dir)[:3]:
        try:
            summary = sniff_json_like_file(sample_file)
            info["sample_files"].append(
                {
                    "path": str(sample_file),
                    "summary": summary,
                }
            )
        except Exception as exc:
            info["sample_files"].append(
                {
                    "path": str(sample_file),
                    "summary": {"error": str(exc)},
                }
            )
    return info


def inspect_model_dir(model_dir: Path, try_load_model: bool) -> Dict[str, Any]:
    key_files = [str(model_dir / name) for name in MODEL_KEY_FILES if (model_dir / name).exists()]
    result: Dict[str, Any] = {
        "path": str(model_dir),
        "top_level": [p.name for p in sorted(model_dir.iterdir())[:20]],
        "key_files_found": key_files,
    }

    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            result["config_keys"] = sorted(config_data.keys())[:30]
            result["architectures"] = config_data.get("architectures")
            result["model_type"] = config_data.get("model_type")
        except Exception as exc:
            result["config_error"] = str(exc)

    if try_load_model:
        try:
            from transformers import AutoConfig

            loaded_config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
            result["try_load_model"] = {
                "status": "ok",
                "config_class": loaded_config.__class__.__name__,
                "model_type": getattr(loaded_config, "model_type", None),
            }
        except Exception as exc:
            result["try_load_model"] = {
                "status": "failed",
                "error": str(exc),
            }
    else:
        result["try_load_model"] = {
            "status": "skipped",
            "reason": "--try-load-model not provided",
        }

    return result


def build_path_checks() -> List[CheckResult]:
    return [
        CheckResult("external/AdaReasoner", ADAREASONER_DIR, ADAREASONER_DIR.exists()),
        CheckResult("external/AdaReasoner/tool_server", TOOL_SERVER_DIR, TOOL_SERVER_DIR.exists()),
        CheckResult("external/AdaReasoner/tool_server/tf_eval", TF_EVAL_DIR, TF_EVAL_DIR.exists()),
        CheckResult(
            "external/AdaReasoner/tool_server/tf_eval/tasks/vsp/config.yaml",
            VSP_CONFIG,
            VSP_CONFIG.exists(),
        ),
        CheckResult(
            "external/AdaReasoner/tool_server/tf_eval/examples/configs",
            EXAMPLE_CONFIGS_DIR,
            EXAMPLE_CONFIGS_DIR.exists(),
        ),
        CheckResult("external/AdaReasoner/requirements", REQUIREMENTS_DIR, REQUIREMENTS_DIR.exists()),
    ]


def build_report(
    path_checks: Sequence[CheckResult],
    import_results: Sequence[Tuple[str, bool, str]],
    dataset_candidates: Sequence[Path],
    dataset_inspection: Optional[Dict[str, Any]],
    model_candidates: Sequence[Path],
    model_inspection: Optional[Dict[str, Any]],
) -> str:
    lines: List[str] = [
        "# AdaReasoner Smoke Test Report",
        "",
        f"- Project root: `{PROJECT_ROOT}`",
        f"- Repo root checked: `{ADAREASONER_DIR}`",
        "",
        "## Repository Drift Notes",
        "",
        "- The top-level `AdaEval` directory mentioned in the README does not exist in this local repository.",
        "- The actual evaluation implementation path is `external/AdaReasoner/tool_server/tf_eval/`.",
        "- The README and the local repository layout have drifted; follow the local repository structure.",
        "",
        "## Path Checks",
        "",
    ]

    for item in path_checks:
        status = "OK" if item.exists else "MISSING"
        lines.append(f"- `{item.label}`: `{status}`")

    lines.extend(["", "## Import Checks", ""])
    for module_name, ok, note in import_results:
        status = "OK" if ok else "FAILED"
        lines.append(f"- `{module_name}`: `{status}`")
        if not ok:
            lines.append(f"  - Note: `{note}`")

    lines.extend(["", "## Dataset Checks", ""])
    if dataset_candidates:
        lines.append(f"- Candidate VSP dataset directories found: `{len(dataset_candidates)}`")
        for candidate in dataset_candidates:
            lines.append(f"- `{candidate}`")
    else:
        lines.append("- No VSP-like dataset directory found under `external/datasets`.")

    if dataset_inspection:
        lines.extend(["", "### Dataset Structure", ""])
        lines.extend(dataset_inspection["tree"])
        lines.extend(["", "### Sample File Field Checks", ""])
        if dataset_inspection["sample_files"]:
            for item in dataset_inspection["sample_files"]:
                lines.append(f"- `{item['path']}`")
                lines.append(f"  - Summary: `{json.dumps(item['summary'], ensure_ascii=False)}`")
        else:
            lines.append("- No JSON/JSONL sample files found for lightweight field inspection.")

    lines.extend(["", "## Model Checks", ""])
    if model_candidates:
        lines.append(f"- Candidate AdaReasoner model directories found: `{len(model_candidates)}`")
        for candidate in model_candidates:
            lines.append(f"- `{candidate}`")
    else:
        lines.append("- No AdaReasoner-like model directory found under `external/models`.")

    if model_inspection:
        lines.extend(["", "### Model Metadata Check", ""])
        lines.append(f"- Path: `{model_inspection['path']}`")
        lines.append(f"- Top-level entries: `{model_inspection['top_level']}`")
        lines.append(f"- Key files found: `{model_inspection['key_files_found']}`")
        if "config_keys" in model_inspection:
            lines.append(f"- Config keys: `{model_inspection['config_keys']}`")
        if "architectures" in model_inspection:
            lines.append(f"- Architectures: `{model_inspection['architectures']}`")
        if "model_type" in model_inspection:
            lines.append(f"- Model type: `{model_inspection['model_type']}`")
        lines.append(f"- Try load result: `{model_inspection['try_load_model']}`")

    return "\n".join(lines).rstrip() + "\n"


def write_report(content: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()

    try:
        path_checks = build_path_checks()
        import_results = try_imports()

        dataset_candidates = find_candidate_dirs(DATASETS_DIR, VSP_DATASET_NAME_HINTS)
        model_candidates = find_candidate_dirs(MODELS_DIR, MODEL_NAME_HINTS)

        dataset_inspection = inspect_dataset_dir(dataset_candidates[0]) if dataset_candidates else None
        model_inspection = (
            inspect_model_dir(model_candidates[0], try_load_model=args.try_load_model)
            if model_candidates
            else None
        )

        report = build_report(
            path_checks=path_checks,
            import_results=import_results,
            dataset_candidates=dataset_candidates,
            dataset_inspection=dataset_inspection,
            model_candidates=model_candidates,
            model_inspection=model_inspection,
        )
        write_report(report)

        print(f"Report written to: {REPORT_PATH}")
        for item in path_checks:
            status = "OK" if item.exists else "MISSING"
            print(f"{item.label}: {status}")
        print(f"Dataset candidates found: {len(dataset_candidates)}")
        print(f"Model candidates found: {len(model_candidates)}")
        return 0

    except SmokeTestError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive boundary
        print(f"Unhandled error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
