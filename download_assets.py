#!/usr/bin/env python3
"""
Download official AdaReasoner MVP assets from Hugging Face.

This script does not run automatically. It only downloads resources when the
user invokes it explicitly from the command line.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "external"
MODELS_DIR = EXTERNAL_DIR / "models"
DATASETS_DIR = EXTERNAL_DIR / "datasets"
LOGS_DIR = EXTERNAL_DIR / "logs"
REPORT_PATH = LOGS_DIR / "download_report.md"


@dataclass(frozen=True)
class Asset:
    flag: str
    repo_id: str
    asset_type: str
    target_dir: Path
    description: str


ASSETS = (
    Asset(
        flag="download_vsp_data",
        repo_id="AdaReasoner/AdaReasoner-TC-VSP",
        asset_type="dataset",
        target_dir=DATASETS_DIR / "AdaReasoner-TC-VSP",
        description="AdaReasoner official VSP dataset asset",
    ),
    Asset(
        flag="download_vsp_model",
        repo_id="AdaReasoner/AdaReasoner-VSP-7B",
        asset_type="model",
        target_dir=MODELS_DIR / "AdaReasoner-VSP-7B",
        description="AdaReasoner official VSP model asset",
    ),
    Asset(
        flag="download_randomized",
        repo_id="AdaReasoner/AdaReasoner-7B-Randomized",
        asset_type="model",
        target_dir=MODELS_DIR / "AdaReasoner-7B-Randomized",
        description="AdaReasoner official randomized 7B model asset",
    ),
    Asset(
        flag="download_nonrandomized",
        repo_id="AdaReasoner/AdaReasoner-7B-Non-Randomized",
        asset_type="model",
        target_dir=MODELS_DIR / "AdaReasoner-7B-Non-Randomized",
        description="AdaReasoner official non-randomized 7B model asset",
    ),
)


class DownloadError(RuntimeError):
    """Raised when an asset download or validation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official AdaReasoner MVP assets from Hugging Face."
    )
    parser.add_argument(
        "--download-vsp-data",
        action="store_true",
        help="Download AdaReasoner/AdaReasoner-TC-VSP into external/datasets.",
    )
    parser.add_argument(
        "--download-vsp-model",
        action="store_true",
        help="Download AdaReasoner/AdaReasoner-VSP-7B into external/models.",
    )
    parser.add_argument(
        "--download-randomized",
        action="store_true",
        help="Download AdaReasoner/AdaReasoner-7B-Randomized into external/models.",
    )
    parser.add_argument(
        "--download-nonrandomized",
        action="store_true",
        help="Download AdaReasoner/AdaReasoner-7B-Non-Randomized into external/models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if the target directory already exists and is non-empty.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned actions without downloading anything.",
    )
    return parser.parse_args()


def ensure_base_dirs() -> None:
    for path in (MODELS_DIR, DATASETS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def select_assets(args: argparse.Namespace) -> List[Asset]:
    selected = [asset for asset in ASSETS if getattr(args, asset.flag)]
    if not selected:
        raise DownloadError(
            "No assets selected. Use one or more of: "
            "--download-vsp-data, --download-vsp-model, "
            "--download-randomized, --download-nonrandomized."
        )
    return selected


def is_non_empty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def import_huggingface_hub():
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError
    except ImportError as exc:
        raise DownloadError(
            "Missing dependency: huggingface_hub. "
            "Install it first, for example: pip install huggingface_hub"
        ) from exc

    return snapshot_download, HfHubHTTPError, LocalEntryNotFoundError


def download_asset(asset: Asset, force: bool, dry_run: bool) -> str:
    target = asset.target_dir

    if is_non_empty_dir(target) and not force:
        return f"Skipped: {asset.repo_id} -> {target} (target already exists and is non-empty)"

    if dry_run:
        action = "Would re-download" if is_non_empty_dir(target) and force else "Would download"
        return f"{action}: {asset.repo_id} -> {target}"

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download, HfHubHTTPError, LocalEntryNotFoundError = import_huggingface_hub()

    try:
        snapshot_download(
            repo_id=asset.repo_id,
            repo_type=asset.asset_type,
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except HfHubHTTPError as exc:
        message = str(exc)
        if "401" in message or "403" in message:
            raise DownloadError(
                f"Failed to access {asset.repo_id}. "
                "If the repo requires authentication, log in first with "
                "`huggingface-cli login` or set `HF_TOKEN` in your environment."
            ) from exc
        raise DownloadError(f"Failed to download {asset.repo_id}: {exc}") from exc
    except LocalEntryNotFoundError as exc:
        raise DownloadError(
            f"Unable to resolve files for {asset.repo_id}. Please verify your network "
            "connection and Hugging Face authentication state."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive error boundary
        raise DownloadError(f"Unexpected error while downloading {asset.repo_id}: {exc}") from exc

    if not target.exists() or not any(target.iterdir()):
        raise DownloadError(
            f"Download finished but target directory is missing or empty: {target}"
        )

    return f"Downloaded: {asset.repo_id} -> {target}"


def format_summary_lines(asset: Asset) -> List[str]:
    target = asset.target_dir
    exists = target.exists()
    non_empty = is_non_empty_dir(target)
    item_count = sum(1 for _ in target.iterdir()) if exists and target.is_dir() else 0
    return [
        f"- Repo: `{asset.repo_id}`",
        f"  - Type: `{asset.asset_type}`",
        f"  - Target: `{target}`",
        f"  - Exists: `{exists}`",
        f"  - Non-empty: `{non_empty}`",
        f"  - Top-level items: `{item_count}`",
    ]


def build_report(
    selected_assets: Iterable[Asset],
    action_lines: Iterable[str],
    dry_run: bool,
) -> str:
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# AdaReasoner Download Report",
        "",
        f"- Generated at: `{timestamp}`",
        f"- Project root: `{PROJECT_ROOT}`",
        f"- Dry run: `{dry_run}`",
        "",
        "## Official Sources",
        "",
        "- GitHub repository: `https://github.com/ssmisya/AdaReasoner`",
        "- Hugging Face namespace: `AdaReasoner/*`",
        "",
        "## Actions",
        "",
    ]
    lines.extend(f"- {line}" for line in action_lines)
    lines.extend(["", "## Post-download Summary", ""])

    for asset in selected_assets:
        lines.extend(format_summary_lines(asset))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(content: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()

    try:
        ensure_base_dirs()
        selected_assets = select_assets(args)

        action_lines: List[str] = []
        for asset in selected_assets:
            action_lines.append(
                download_asset(asset=asset, force=args.force, dry_run=args.dry_run)
            )

        report = build_report(
            selected_assets=selected_assets,
            action_lines=action_lines,
            dry_run=args.dry_run,
        )
        write_report(report)

        print(f"Report written to: {REPORT_PATH}")
        for line in action_lines:
            print(line)
        return 0

    except DownloadError as exc:
        error_report = build_report(
            selected_assets=[],
            action_lines=[f"Error: {exc}"],
            dry_run=args.dry_run,
        )
        write_report(error_report)
        print(f"Error: {exc}", file=sys.stderr)
        print(f"Report written to: {REPORT_PATH}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - final safety net
        error_message = f"Unhandled error: {exc}"
        write_report(
            build_report(selected_assets=[], action_lines=[error_message], dry_run=args.dry_run)
        )
        print(error_message, file=sys.stderr)
        print(f"Report written to: {REPORT_PATH}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
