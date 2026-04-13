"""Command-line interface for indexing and comparing Marcussen recordings."""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

from .dataset import MarcussenDataset, make_group_id

logger = logging.getLogger(__name__)


def _make_dataset(root: Path, *, close_only: bool = False) -> MarcussenDataset:
    return MarcussenDataset(root=root, mic_location_filter="close" if close_only else None)


def _resolve_root(root_arg: str | None) -> Path:
    root_value = root_arg or os.getenv("MARCUSSEN_DATASET_ROOT")
    if not root_value:
        raise ValueError("Dataset root is required (use --root or MARCUSSEN_DATASET_ROOT)")
    return Path(root_value)


def _write_index_csv(dataset: MarcussenDataset, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path",
        "ext",
        "raw_stem",
        "family",
        "organ_id",
        "registration_raw",
        "registration_expanded",
        "division",
        "mic_location",
        "pitch",
        "take",
        "track",
        "extras",
        "warnings",
        "group_id",
    ]

    count = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for item in dataset.iter_flat_items():
            meta = item.meta
            writer.writerow(
                {
                    "path": item.path,
                    "ext": item.ext,
                    "raw_stem": item.raw_stem,
                    "family": meta.get("family", ""),
                    "organ_id": meta.get("organ_id", ""),
                    "registration_raw": meta.get("registration_raw", ""),
                    "registration_expanded": meta.get("registration_expanded", ""),
                    "division": meta.get("division", ""),
                    "mic_location": meta.get("mic_location", ""),
                    "pitch": meta.get("pitch", ""),
                    "take": meta.get("take", ""),
                    "track": meta.get("track", ""),
                    "extras": "|".join(item.extras),
                    "warnings": "|".join(item.warnings),
                    "group_id": make_group_id(meta, dataset.group_keys),
                }
            )
            count += 1
            if count % 200 == 0:
                handle.flush()

    return count


def _path_relative_to_root(path_str: str, root: Path) -> str:
    path = Path(path_str)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_group_key_csv(dataset: MarcussenDataset, out_path: Path) -> int:
    """Write a CSV containing relative filepath and one column per group key."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filepath", *dataset.group_keys]

    count = 0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for item in dataset.iter_flat_items():
            row = {"filepath": _path_relative_to_root(item.path, dataset.root)}
            for key in dataset.group_keys:
                value = item.meta.get(key, "")
                row[key] = "" if value is None else str(value)
            writer.writerow(row)
            count += 1
            if count % 200 == 0:
                handle.flush()

    return count


def _cmd_index(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    dataset = _make_dataset(root, close_only=args.close_only)
    out = Path(args.out)
    count = _write_index_csv(dataset, out)
    print(f"Indexed {count} files -> {out}")
    return 0


def _cmd_parse(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    dataset = _make_dataset(root, close_only=args.close_only)
    out = Path(args.out)
    count = _write_group_key_csv(dataset, out)
    print(f"Parsed {count} files -> {out}")
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    root = _resolve_root(args.root)
    dataset = _make_dataset(root, close_only=args.close_only)
    sampled = dataset.sample(n=args.n, seed=args.seed)
    for item in sampled:
        group_id = make_group_id(item.meta, dataset.group_keys)
        print(f"{item.path}\t{group_id}")
    print(f"Sampled {len(sampled)} file(s)")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from .compare import run_within_group

    root = _resolve_root(args.root)
    dataset = _make_dataset(root, close_only=args.close_only)
    summary = run_within_group(
        dataset,
        out_csv_path=args.out,
        metric=args.metric,
        max_pairs=args.max_pairs,
    )
    print(
        "Compared within groups: "
        f"groups={summary['groups']} rows={summary['rows']} errors={summary['errors']} out={summary['out_csv']}"
    )
    return 0


def _cmd_prepare_listening_dataset(args: argparse.Namespace) -> int:
    from .listening_dataset import prepare_listening_dataset

    root = _resolve_root(args.root)
    dataset = _make_dataset(root, close_only=args.close_only)
    summary = prepare_listening_dataset(
        dataset,
        out_dir=args.out_dir,
        csv_name=args.csv_name,
        max_pairs=args.max_pairs,
        expand_toots=True,
        trim=args.trim,
        normalize=args.normalize,
        steady_state_seconds=args.steady_state_seconds,
        debug_first_n_per_batch=50 if args.debug_first_50_per_batch else None,
    )
    print(
        "Prepared listening dataset: "
        f"groups={summary['groups']} rows={summary['rows']} wav_files={summary['wav_files']} "
        f"out={summary['out_dir']} csv={summary['out_csv']}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build top-level argparse parser."""
    parser = argparse.ArgumentParser(prog="marcussen")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Scan dataset and write parsed index CSV")
    index_parser.add_argument("--root", type=str, default=None, help="Dataset root path")
    index_parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    index_parser.add_argument(
        "--close-only",
        action="store_true",
        help="Only include recordings with mic_location=CLOSE",
    )
    index_parser.set_defaults(func=_cmd_index)

    parse_parser = subparsers.add_parser(
        "parse",
        help="Parse all files and write CSV with filepath + current group keys",
    )
    parse_parser.add_argument("--root", type=str, default=None, help="Dataset root path")
    parse_parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    parse_parser.add_argument(
        "--close-only",
        action="store_true",
        help="Only include recordings with mic_location=CLOSE",
    )
    parse_parser.set_defaults(func=_cmd_parse)

    sample_parser = subparsers.add_parser("sample", help="Print random sample of parsed items")
    sample_parser.add_argument("--root", type=str, default=None, help="Dataset root path")
    sample_parser.add_argument("--n", type=int, default=2, help="Number of samples")
    sample_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    sample_parser.add_argument(
        "--close-only",
        action="store_true",
        help="Only include recordings with mic_location=CLOSE",
    )
    sample_parser.set_defaults(func=_cmd_sample)

    compare_parser = subparsers.add_parser("compare", help="Run within-group comparisons")
    compare_parser.add_argument("--root", type=str, default=None, help="Dataset root path")
    compare_parser.add_argument("--out", type=str, required=True, help="Output pair CSV path")
    compare_parser.add_argument("--metric", type=str, default="placeholder", help="Comparison metric")
    compare_parser.add_argument(
        "--close-only",
        action="store_true",
        help="Only include recordings with mic_location=CLOSE",
    )
    compare_parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Stop after writing this many pairs (global cap, useful for debugging)",
    )
    compare_parser.set_defaults(func=_cmd_compare)

    prepare_parser = subparsers.add_parser(
        "prepare_listening_dataset",
        help="Write toot-level listening-test pairs plus WAV exports, including same-organ controls",
    )
    prepare_parser.add_argument("--root", type=str, default=None, help="Dataset root path")
    prepare_parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    prepare_parser.add_argument(
        "--close-only",
        action="store_true",
        help="Only include recordings with mic_location=CLOSE",
    )
    prepare_parser.add_argument(
        "--csv-name",
        type=str,
        default="pairs.csv",
        help="CSV filename within the output directory",
    )
    prepare_parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Stop after writing this many pairs (global cap, useful for debugging)",
    )
    prepare_parser.add_argument(
        "--trim",
        action="store_true",
        help="Trim leading and trailing silence from exported audio",
    )
    prepare_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Peak-normalize exported audio",
    )
    prepare_parser.add_argument(
        "--steady-state-seconds",
        type=float,
        default=None,
        help="Export a centered steady-state window of this duration in seconds",
    )
    prepare_parser.add_argument(
        "--debug-first-50-per-batch",
        action="store_true",
        help="Debug mode: only write the first 50 pair rows from each batch",
    )
    prepare_parser.set_defaults(func=_cmd_prepare_listening_dataset)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Console entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    try:
        return int(args.func(args))
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
