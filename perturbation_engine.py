#!/usr/bin/env python3
"""
Generate controlled perturbations for agent tool-use robustness experiments.

This script only rewrites structured inputs. It does not call any model or run
inference. It supports JSON and JSONL input, keeps the original sample, and
emits a perturbed sample with perturbation metadata.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "external" / "logs"
REPORT_PATH = LOGS_DIR / "perturbation_report.md"

SUPPORTED_PERTURBATIONS = (
    "tool_name_rename",
    "tool_name_alias",
    "parameter_name_rename",
    "parameter_order_shuffle",
    "task_description_paraphrase_light",
    "task_description_compress_light",
    "tool_description_rewrite_light",
)

SUPPORTED_SEVERITIES = ("light", "medium", "heavy")

TOOL_CONTAINER_KEYS = ("tool_registry", "tools")
PARAMETER_KEYS = ("parameters", "args", "schema", "properties")
TASK_DESCRIPTION_KEYS = (
    "task_description",
    "description",
    "task",
    "instruction",
    "prompt",
)
TOOL_NAME_KEYS = ("name", "tool_name", "id")
TOOL_DESCRIPTION_KEYS = ("description", "tool_description", "summary")


class PerturbationError(RuntimeError):
    """Raised when perturbation generation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate controlled perturbations for tool-use robustness samples."
    )
    parser.add_argument("--input", required=True, help="Input JSON or JSONL file path.")
    parser.add_argument("--output", required=True, help="Output JSON or JSONL file path.")
    parser.add_argument(
        "--perturbation-type",
        required=True,
        choices=SUPPORTED_PERTURBATIONS,
        help="Perturbation type to apply.",
    )
    parser.add_argument(
        "--severity",
        default="light",
        choices=SUPPORTED_SEVERITIES,
        help="Perturbation severity. Framework supports light/medium/heavy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic perturbation generation.",
    )
    return parser.parse_args()


def load_samples(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    if not path.exists():
        raise PerturbationError(f"Input file does not exist: {path}")

    raw_text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise PerturbationError(f"Failed to parse JSON input: {exc}") from exc

        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict):
            samples = [data]
        else:
            raise PerturbationError("JSON input must be an object or a list of objects.")
        return _validate_samples(samples), "json"

    if suffix == ".jsonl":
        samples: List[Dict[str, Any]] = []
        for line_no, line in enumerate(raw_text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PerturbationError(
                    f"Failed to parse JSONL at line {line_no}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise PerturbationError(f"JSONL line {line_no} is not an object.")
            samples.append(item)
        return _validate_samples(samples), "jsonl"

    raise PerturbationError("Input must be .json or .jsonl")


def _validate_samples(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    validated: List[Dict[str, Any]] = []
    for index, item in enumerate(samples):
        if not isinstance(item, dict):
            raise PerturbationError(f"Sample at index {index} is not a JSON object.")
        validated.append(item)
    return validated


def save_outputs(path: Path, records: List[Dict[str, Any]], output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    if output_format == "jsonl":
        lines = [json.dumps(record, ensure_ascii=False) for record in records]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return

    raise PerturbationError(f"Unsupported output format: {output_format}")


def write_report(
    *,
    input_path: Path,
    output_path: Path,
    perturbation_type: str,
    severity: str,
    seed: int,
    record_count: int,
    status: str,
    note: str,
) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "# Perturbation Engine Report",
            "",
            f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
            f"- Input: `{input_path}`",
            f"- Output: `{output_path}`",
            f"- Perturbation type: `{perturbation_type}`",
            f"- Severity: `{severity}`",
            f"- Seed: `{seed}`",
            f"- Record count: `{record_count}`",
            f"- Status: `{status}`",
            f"- Note: {note}",
            "",
            "## Input Format Assumptions",
            "",
            "- Input must be `.json` or `.jsonl`.",
            "- Each sample should be a JSON object.",
            "- Compatible fields are inferred from common names like `task_id`, `task_description`, `tool_registry`, `tools`, `parameters`, `args`, and `schema`.",
        ]
    ) + "\n"
    REPORT_PATH.write_text(content, encoding="utf-8")


def get_task_id(sample: Dict[str, Any], index: int) -> str:
    value = sample.get("task_id") or sample.get("id") or sample.get("sample_id")
    return str(value) if value is not None else f"sample_{index}"


def find_first_key(mapping: MutableMapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in mapping:
            return key
    return None


def get_nested_tool_containers(sample: Dict[str, Any]) -> List[Tuple[MutableMapping[str, Any], str, Any]]:
    containers: List[Tuple[MutableMapping[str, Any], str, Any]] = []
    for key in TOOL_CONTAINER_KEYS:
        if key in sample:
            containers.append((sample, key, sample[key]))
    return containers


def normalize_tool_items(container_value: Any) -> List[MutableMapping[str, Any]]:
    if isinstance(container_value, list):
        return [item for item in container_value if isinstance(item, MutableMapping)]
    if isinstance(container_value, dict):
        items: List[MutableMapping[str, Any]] = []
        for key, value in container_value.items():
            if isinstance(value, MutableMapping):
                item = dict(value)
                item.setdefault("name", key)
                items.append(item)
        return items
    return []


def collect_name_mapping(sample: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for _, _, container_value in get_nested_tool_containers(sample):
        for tool in normalize_tool_items(container_value):
            name_key = find_first_key(tool, TOOL_NAME_KEYS)
            if not name_key:
                continue
            original_name = str(tool[name_key])
            mapping.setdefault(original_name, original_name)
    return mapping


def severity_variant(base: str, severity: str) -> str:
    suffix_map = {
        "light": "v2",
        "medium": "plus",
        "heavy": "alt",
    }
    return f"{base}_{suffix_map[severity]}"


def alias_variant(base: str, severity: str) -> str:
    alias_map = {
        "light": {
            "crop": "image_crop",
            "ocr": "text_reader",
            "point": "point_locator",
            "draw2dpath": "path_drawer",
        },
        "medium": {
            "crop": "region_cropper",
            "ocr": "optical_text_reader",
            "point": "target_pointer",
            "draw2dpath": "route_renderer",
        },
        "heavy": {
            "crop": "visual_region_selector",
            "ocr": "scene_text_extractor",
            "point": "coordinate_grounder",
            "draw2dpath": "trajectory_annotator",
        },
    }
    lookup_key = re.sub(r"[^a-z0-9]", "", base.lower())
    return alias_map.get(severity, {}).get(lookup_key, severity_variant(base, severity))


def parameter_name_variant(base: str, severity: str) -> str:
    replacements = {
        "light": {
            "image": "input_image",
            "text": "input_text",
            "x": "coord_x",
            "y": "coord_y",
            "path": "path_value",
        },
        "medium": {
            "image": "source_image",
            "text": "query_text",
            "x": "target_x",
            "y": "target_y",
            "path": "planned_path",
        },
        "heavy": {
            "image": "visual_input",
            "text": "instruction_text",
            "x": "horizontal_coord",
            "y": "vertical_coord",
            "path": "candidate_trajectory",
        },
    }
    return replacements.get(severity, {}).get(base.lower(), severity_variant(base, severity))


def rename_text_occurrences(text: str, mapping: Dict[str, str]) -> str:
    updated = text
    for old_name, new_name in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        updated = re.sub(rf"\b{re.escape(old_name)}\b", new_name, updated)
    return updated


def paraphrase_light(text: str, severity: str) -> str:
    replacements = [
        ("please", "please"),
        ("use", "make use of"),
        ("find", "identify"),
        ("return", "provide"),
        ("locate", "identify the location of"),
        ("determine", "work out"),
    ]
    updated = text
    for source, target in replacements:
        if severity == "light" and source == target:
            continue
        updated = re.sub(rf"\b{re.escape(source)}\b", target, updated, flags=re.IGNORECASE)
    return updated


def compress_light(text: str, severity: str) -> str:
    updated = re.sub(r"\s+", " ", text).strip()
    updated = re.sub(r"\b[Pp]lease\b", "", updated)
    updated = re.sub(r"\b[Yy]ou should\b", "", updated)
    updated = re.sub(r"\b[Yy]ou need to\b", "", updated)
    updated = re.sub(r"\s{2,}", " ", updated).strip(" ,")
    if severity == "medium":
        updated = updated.replace(" in order to ", " to ")
    if severity == "heavy":
        updated = updated.replace(" and ", "; ")
    return updated


def rewrite_tool_description(text: str, severity: str) -> str:
    if not text:
        return text
    if severity == "light":
        return f"Used to {text[0].lower() + text[1:]}" if len(text) > 1 else text
    if severity == "medium":
        return f"This tool helps with: {text}"
    return f"Primary function: {text}"


def rewrite_task_descriptions(sample: Dict[str, Any], text_rewriter) -> int:
    changed = 0
    for key in TASK_DESCRIPTION_KEYS:
        if key in sample and isinstance(sample[key], str):
            sample[key] = text_rewriter(sample[key])
            changed += 1
    return changed


def rebuild_container_from_items(original_value: Any, items: List[MutableMapping[str, Any]]) -> Any:
    if isinstance(original_value, list):
        return items
    if isinstance(original_value, dict):
        rebuilt: Dict[str, Any] = {}
        for item in items:
            name_key = find_first_key(item, TOOL_NAME_KEYS) or "name"
            rebuilt[str(item[name_key])] = item
        return rebuilt
    return original_value


def apply_tool_name_transform(sample: Dict[str, Any], severity: str, alias_mode: bool) -> Tuple[Dict[str, str], int]:
    name_mapping: Dict[str, str] = {}
    changed = 0

    for parent, key, container_value in get_nested_tool_containers(sample):
        tool_items = normalize_tool_items(container_value)
        new_items: List[MutableMapping[str, Any]] = []

        for tool in tool_items:
            name_key = find_first_key(tool, TOOL_NAME_KEYS)
            if not name_key:
                new_items.append(tool)
                continue

            original_name = str(tool[name_key])
            new_name = alias_variant(original_name, severity) if alias_mode else severity_variant(original_name, severity)
            tool[name_key] = new_name
            name_mapping[original_name] = new_name
            changed += 1
            new_items.append(tool)

        parent[key] = rebuild_container_from_items(container_value, new_items)

    if name_mapping:
        _rewrite_tool_name_references(sample, name_mapping)
    return name_mapping, changed


def _rewrite_tool_name_references(sample: Dict[str, Any], mapping: Dict[str, str]) -> None:
    for key in TASK_DESCRIPTION_KEYS:
        if key in sample and isinstance(sample[key], str):
            sample[key] = rename_text_occurrences(sample[key], mapping)

    for field in ("tool_choice", "selected_tools", "allowed_tools", "tool_selection"):
        value = sample.get(field)
        if isinstance(value, str):
            sample[field] = rename_text_occurrences(value, mapping)
        elif isinstance(value, list):
            sample[field] = [mapping.get(str(item), item) for item in value]


def iter_parameter_containers(tool: MutableMapping[str, Any]) -> List[Tuple[MutableMapping[str, Any], str, Any]]:
    containers: List[Tuple[MutableMapping[str, Any], str, Any]] = []
    for key in PARAMETER_KEYS:
        if key in tool:
            containers.append((tool, key, tool[key]))
    return containers


def rename_parameter_references(payload: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(payload, dict):
        renamed: Dict[str, Any] = {}
        for key, value in payload.items():
            new_key = mapping.get(key, key)
            renamed[new_key] = rename_parameter_references(value, mapping)
        return renamed
    if isinstance(payload, list):
        return [rename_parameter_references(item, mapping) for item in payload]
    if isinstance(payload, str):
        return rename_text_occurrences(payload, mapping)
    return payload


def apply_parameter_name_rename(sample: Dict[str, Any], severity: str) -> Tuple[Dict[str, str], int]:
    renamed_parameters: Dict[str, str] = {}
    changed = 0

    for parent, key, container_value in get_nested_tool_containers(sample):
        tool_items = normalize_tool_items(container_value)
        new_items: List[MutableMapping[str, Any]] = []
        for tool in tool_items:
            for tool_parent, tool_key, parameter_value in iter_parameter_containers(tool):
                if not isinstance(parameter_value, dict):
                    continue
                new_parameters: Dict[str, Any] = {}
                local_mapping: Dict[str, str] = {}
                for param_name, param_spec in parameter_value.items():
                    new_name = parameter_name_variant(str(param_name), severity)
                    local_mapping[str(param_name)] = new_name
                    renamed_parameters[str(param_name)] = new_name
                    new_parameters[new_name] = rename_parameter_references(param_spec, local_mapping)
                    changed += 1
                tool_parent[tool_key] = new_parameters
                tool = rename_parameter_references(tool, local_mapping)
            new_items.append(tool)
        parent[key] = rebuild_container_from_items(container_value, new_items)

    if renamed_parameters:
        sample.update(rename_parameter_references(sample, renamed_parameters))
    return renamed_parameters, changed


def apply_parameter_order_shuffle(sample: Dict[str, Any], rng: random.Random) -> int:
    changed = 0
    for parent, key, container_value in get_nested_tool_containers(sample):
        tool_items = normalize_tool_items(container_value)
        new_items: List[MutableMapping[str, Any]] = []
        for tool in tool_items:
            for tool_parent, tool_key, parameter_value in iter_parameter_containers(tool):
                if isinstance(parameter_value, dict) and len(parameter_value) > 1:
                    items = list(parameter_value.items())
                    rng.shuffle(items)
                    tool_parent[tool_key] = dict(items)
                    changed += 1
            new_items.append(tool)
        parent[key] = rebuild_container_from_items(container_value, new_items)
    return changed


def apply_tool_description_rewrite(sample: Dict[str, Any], severity: str) -> int:
    changed = 0
    for parent, key, container_value in get_nested_tool_containers(sample):
        tool_items = normalize_tool_items(container_value)
        new_items: List[MutableMapping[str, Any]] = []
        for tool in tool_items:
            description_key = find_first_key(tool, TOOL_DESCRIPTION_KEYS)
            if description_key and isinstance(tool[description_key], str):
                tool[description_key] = rewrite_tool_description(tool[description_key], severity)
                changed += 1
            new_items.append(tool)
        parent[key] = rebuild_container_from_items(container_value, new_items)
    return changed


def apply_perturbation(
    sample: Dict[str, Any],
    perturbation_type: str,
    severity: str,
    rng: random.Random,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    perturbed = copy.deepcopy(sample)
    details: Dict[str, Any] = {
        "perturbation_type": perturbation_type,
        "severity": severity,
        "changes": {},
    }

    if perturbation_type == "tool_name_rename":
        mapping, count = apply_tool_name_transform(perturbed, severity, alias_mode=False)
        details["changes"] = {"tool_name_mapping": mapping, "change_count": count}
    elif perturbation_type == "tool_name_alias":
        mapping, count = apply_tool_name_transform(perturbed, severity, alias_mode=True)
        details["changes"] = {"tool_name_mapping": mapping, "change_count": count}
    elif perturbation_type == "parameter_name_rename":
        mapping, count = apply_parameter_name_rename(perturbed, severity)
        details["changes"] = {"parameter_name_mapping": mapping, "change_count": count}
    elif perturbation_type == "parameter_order_shuffle":
        count = apply_parameter_order_shuffle(perturbed, rng)
        details["changes"] = {"change_count": count}
    elif perturbation_type == "task_description_paraphrase_light":
        count = rewrite_task_descriptions(
            perturbed, lambda text: paraphrase_light(text, severity)
        )
        details["changes"] = {"change_count": count}
    elif perturbation_type == "task_description_compress_light":
        count = rewrite_task_descriptions(
            perturbed, lambda text: compress_light(text, severity)
        )
        details["changes"] = {"change_count": count}
    elif perturbation_type == "tool_description_rewrite_light":
        count = apply_tool_description_rewrite(perturbed, severity)
        details["changes"] = {"change_count": count}
    else:
        raise PerturbationError(f"Unsupported perturbation type: {perturbation_type}")

    return perturbed, details


def build_output_record(
    original: Dict[str, Any],
    perturbed: Dict[str, Any],
    metadata: Dict[str, Any],
    task_id: str,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "perturbation_type": metadata["perturbation_type"],
        "severity": metadata["severity"],
        "perturbation_details": metadata["changes"],
        "original_sample": original,
        "perturbed_sample": perturbed,
    }


def process_samples(
    samples: Sequence[Dict[str, Any]],
    perturbation_type: str,
    severity: str,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    outputs: List[Dict[str, Any]] = []

    for index, sample in enumerate(samples):
        task_id = get_task_id(sample, index)
        perturbed, metadata = apply_perturbation(sample, perturbation_type, severity, rng)
        outputs.append(build_output_record(sample, perturbed, metadata, task_id))

    return outputs


def main() -> int:
    args = parse_args()

    try:
        input_path = Path(args.input)
        output_path = Path(args.output)
        samples, input_format = load_samples(input_path)
        outputs = process_samples(
            samples=samples,
            perturbation_type=args.perturbation_type,
            severity=args.severity,
            seed=args.seed,
        )
        output_format = output_path.suffix.lower().lstrip(".") or input_format
        if output_format not in {"json", "jsonl"}:
            raise PerturbationError("Output must end with .json or .jsonl")
        save_outputs(output_path, outputs, output_format)
        write_report(
            input_path=input_path,
            output_path=output_path,
            perturbation_type=args.perturbation_type,
            severity=args.severity,
            seed=args.seed,
            record_count=len(outputs),
            status="ok",
            note="Perturbation generation completed successfully.",
        )

        print(f"Wrote {len(outputs)} perturbed records to: {output_path}")
        print(f"Perturbation type: {args.perturbation_type}")
        print(f"Severity: {args.severity}")
        print(f"Report written to: {REPORT_PATH}")
        return 0
    except PerturbationError as exc:
        write_report(
            input_path=Path(args.input),
            output_path=Path(args.output),
            perturbation_type=args.perturbation_type,
            severity=args.severity,
            seed=args.seed,
            record_count=0,
            status="failed",
            note=str(exc),
        )
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive boundary
        write_report(
            input_path=Path(args.input),
            output_path=Path(args.output),
            perturbation_type=args.perturbation_type,
            severity=args.severity,
            seed=args.seed,
            record_count=0,
            status="failed",
            note=f"Unhandled error: {exc}",
        )
        print(f"Unhandled error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
