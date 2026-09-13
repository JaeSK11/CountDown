#!/usr/bin/env python3
"""Train one ensemble member and write a reproducible run directory.

    scripts/train.py --model flow_gbdt --dataset iscx_pooled --target traffic_type
    scripts/train.py --config configs/experiments/iscx_pooled_traffic_type_gbdt.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qsentinel.config import default_config, setup_logging  # noqa: E402
from qsentinel.data import available_datasets, available_pooled  # noqa: E402
from qsentinel.models import ModelRegistry  # noqa: E402
from qsentinel.training import ExperimentConfig, train  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, help="experiment YAML; flags below override it")
    p.add_argument("--dataset", help=f"one of {available_datasets()} or a pooled alias")
    p.add_argument("--target", help="label field to train on (default: the dataset's)")
    p.add_argument("--model", help=f"one of {ModelRegistry.list()}")
    p.add_argument("--features", help="feature extractor (default: flow_stats)")
    p.add_argument("--group-key", dest="group_key",
                   help="grouping key (default: the dataset's declared group_key)")
    p.add_argument("--no-group", dest="grouped", action="store_false", default=None,
                   help="use a plain stratified split -- OPTIMISTIC, never report it")
    p.add_argument("--test-size", dest="test_size", type=float)
    p.add_argument("--val-size", dest="val_size", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--min-samples-per-class", dest="min_samples_per_class", type=int)
    p.add_argument("--top-k-classes", dest="top_k_classes", type=int)
    p.add_argument("--runs-dir", dest="runs_dir", type=Path)
    p.add_argument("--no-write", dest="write", action="store_false",
                   help="evaluate without writing a run directory")
    p.add_argument("--list", action="store_true", help="list datasets and models, then exit")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.list:
        cfg = default_config()
        print("datasets :", available_datasets())
        print("pooled   :", available_pooled(cfg))
        print("models   :", ModelRegistry.list())
        for name in available_datasets() + available_pooled(cfg):
            print(f"  group_key[{name}] = {cfg.group_key(name)}")
        return 0

    if args.config:
        cfg = ExperimentConfig.from_yaml(args.config)
    elif args.dataset:
        cfg = ExperimentConfig(dataset=args.dataset)
    else:
        build_parser().error("need --config or --dataset (or --list)")

    for field_name in (
        "dataset", "target", "model", "features", "group_key", "grouped",
        "test_size", "val_size", "seed", "min_samples_per_class", "top_k_classes",
    ):
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(cfg, field_name, value)

    train(cfg, runs_dir=args.runs_dir, write=args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
