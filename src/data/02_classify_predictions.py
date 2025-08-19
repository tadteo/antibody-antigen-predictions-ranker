#!/usr/bin/env python3
"""
This script is designed to classify AlphaFold3 predictions by success/failed and copy files to the appropriate subdirectories.

to run:
python 02_classify_predictions.py --input-dir input_path_with_original_h5 --output-dir output_path_with_classified_h5
"""

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


LOG_FILENAME_PATTERN = re.compile(r"^(?P<pdb_id>[0-9a-zA-Z]{4})_log\.json$")


@dataclass
class ComponentCounts:
    dockq_failed_samples: int = 0
    af3_failed_samples: int = 0
    tm_failed_samples: int = 0
    metadata_failed_samples: int = 0


@dataclass
class ComplexSummary:
    pdb_id: str
    total_samples: int
    success_samples: int
    failed_samples: int
    complex_success: bool
    component_failures: ComponentCounts
    successful_sample_names: List[str]
    failed_sample_names: List[str]
    failed_components_by_sample: Dict[str, List[str]]


@dataclass
class Report:
    input_dir: str
    output_dir: str
    total_complexes: int
    success_complexes: int
    failed_complexes: int
    total_samples: int
    success_samples: int
    failed_samples: int
    success_pdb_ids: List[str]
    failed_pdb_ids: List[str]
    per_complex: Dict[str, ComplexSummary]


def discover_log_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return [p for p in input_dir.iterdir() if p.is_file() and LOG_FILENAME_PATTERN.match(p.name)]


def read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def evaluate_complex(log_json: dict) -> Tuple[int, int, ComponentCounts, List[str], List[str], Dict[str, List[str]]]:
    samples = log_json.get("samples", [])
    success_samples = 0
    component_counts = ComponentCounts()
    successful_sample_names: List[str] = []
    failed_sample_names: List[str] = []
    failed_components_by_sample: Dict[str, List[str]] = {}

    for sample in samples:
        components = sample.get("components", {}) or {}
        dockq = bool(components.get("dockq_success", False))
        af3 = bool(components.get("af3_success", False))
        tm = bool(components.get("tm_success", False))
        meta = bool(components.get("metadata_success", False))
        status_success = str(sample.get("status", "")).upper() == "SUCCESS"
        sample_name = str(sample.get("sample_name") or sample.get("seed_name") or sample.get("sample_index"))

        all_ok = dockq and af3 and tm and meta and status_success
        if all_ok:
            success_samples += 1
            successful_sample_names.append(sample_name)
        else:
            failed_list: List[str] = []
            if not dockq:
                component_counts.dockq_failed_samples += 1
                failed_list.append("dockq")
            if not af3:
                component_counts.af3_failed_samples += 1
                failed_list.append("af3")
            if not tm:
                component_counts.tm_failed_samples += 1
                failed_list.append("tm")
            if not meta:
                component_counts.metadata_failed_samples += 1
                failed_list.append("metadata")
            if not status_success:
                failed_list.append("status")
            failed_sample_names.append(sample_name)
            failed_components_by_sample[sample_name] = failed_list

    total_samples = len(samples)
    return (
        total_samples,
        success_samples,
        component_counts,
        successful_sample_names,
        failed_sample_names,
        failed_components_by_sample,
    )


def ensure_dirs(base_out: Path) -> Dict[str, Path]:
    success_h5 = base_out / "success" / "h5"
    success_log = base_out / "success" / "log"
    failed_h5 = base_out / "failed" / "h5"
    failed_log = base_out / "failed" / "log"
    for d in [success_h5, success_log, failed_h5, failed_log]:
        d.mkdir(parents=True, exist_ok=True)
    return {
        "success_h5": success_h5,
        "success_log": success_log,
        "failed_h5": failed_h5,
        "failed_log": failed_log,
    }


def copy_if_exists(src: Path, dst_dir: Path) -> bool:
    if src.exists() and src.is_file():
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        return True
    return False


def build_report(
    input_dir: Path,
    output_dir: Path,
    complex_summaries: List[ComplexSummary],
) -> Report:
    success_pdb_ids = [c.pdb_id for c in complex_summaries if c.complex_success]
    failed_pdb_ids = [c.pdb_id for c in complex_summaries if not c.complex_success]
    total_samples = sum(c.total_samples for c in complex_summaries)
    success_samples = sum(c.success_samples for c in complex_summaries)
    failed_samples = sum(c.failed_samples for c in complex_summaries)

    return Report(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        total_complexes=len(complex_summaries),
        success_complexes=len(success_pdb_ids),
        failed_complexes=len(failed_pdb_ids),
        total_samples=total_samples,
        success_samples=success_samples,
        failed_samples=failed_samples,
        success_pdb_ids=sorted(success_pdb_ids),
        failed_pdb_ids=sorted(failed_pdb_ids),
        per_complex={c.pdb_id: c for c in complex_summaries},
    )


def dataclass_to_dict(obj):
    if isinstance(obj, list):
        return [dataclass_to_dict(o) for o in obj]
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for k, v in asdict(obj).items():
            result[k] = dataclass_to_dict(v)
        return result
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def main():
    parser = argparse.ArgumentParser(description="Classify AF3 predictions by success/failed and copy files")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/proj/berzelius-2021-29/users/x_matta/abag_af3_predictions/abag_dataset",
        help="Directory containing *_log.json and corresponding *.h5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/proj/berzelius-2021-29/users/x_matta/abag_af3_predictions/abag_dataset_classified",
        help="Directory where success/failed subdirectories will be created",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not copy files; only print a summary and write report.json",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    log_files = discover_log_files(input_dir)
    if not log_files:
        print(f"No *_log.json files found in {input_dir}")
        return

    dirs = ensure_dirs(output_dir)

    complex_summaries: List[ComplexSummary] = []

    for log_path in sorted(log_files):
        m = LOG_FILENAME_PATTERN.match(log_path.name)
        assert m, f"Unexpected log filename format: {log_path.name}"
        pdb_id = m.group("pdb_id")

        log_json = read_json(log_path)
        (
            total_samples,
            success_samples,
            component_counts,
            successful_sample_names,
            failed_sample_names,
            failed_components_by_sample,
        ) = evaluate_complex(log_json)
        failed_samples = total_samples - success_samples
        complex_success = failed_samples == 0

        complex_summaries.append(
            ComplexSummary(
                pdb_id=pdb_id,
                total_samples=total_samples,
                success_samples=success_samples,
                failed_samples=failed_samples,
                complex_success=complex_success,
                component_failures=component_counts,
                successful_sample_names=successful_sample_names,
                failed_sample_names=failed_sample_names,
                failed_components_by_sample=failed_components_by_sample,
            )
        )

        # Determine destination subfolders
        dest_log_dir = dirs["success_log"] if complex_success else dirs["failed_log"]
        dest_h5_dir = dirs["success_h5"] if complex_success else dirs["failed_h5"]

        # Copy files
        if not args.dry_run:
            copied_log = copy_if_exists(log_path, dest_log_dir)
            h5_path = input_dir / f"{pdb_id}.h5"
            copied_h5 = copy_if_exists(h5_path, dest_h5_dir)

            if not copied_log:
                print(f"[WARN] Log file missing or not copied: {log_path}")
            if not copied_h5:
                print(f"[WARN] H5 file missing or not copied: {h5_path}")

    # Build and write report
    report = build_report(input_dir, output_dir, complex_summaries)
    report_path = output_dir / "report.json"
    with report_path.open("w") as f:
        json.dump(dataclass_to_dict(report), f, indent=2)

    print(f"Processed {report.total_complexes} complexes from {input_dir}")
    print(f"  Success: {report.success_complexes} | Failed: {report.failed_complexes}")
    print(f"  Samples  Success: {report.success_samples} | Failed: {report.failed_samples}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()

