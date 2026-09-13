"""Pooled datasets: several constituents concatenated into one ``FlowDataset``.

Pooling ISCXVPN with ISCXTor is not a convenience, it is a requirement of the label
distribution.  ISCXVPN realises 7 of the 8 ``traffic_type`` classes (the corpus contains no
``browsing`` capture at all) and its entire ``p2p`` class comes from one file, so a
group-aware split cannot place ``p2p`` on both sides.  ISCXTor supplies both.  Pooled, the
target has all 8 classes with at least 10 capture groups on the thinnest one.

Named pools live in ``configs/datasets.yaml`` under ``pooled:`` so ``load("iscx_pooled")``
and ``--dataset iscx_pooled`` work like any other dataset.
"""

from __future__ import annotations

from typing import Any, Sequence

from qsentinel.config import Config, default_config, get_logger
from qsentinel.schema import Flow, FlowDataset

log = get_logger(__name__)


def available_pooled(config: Config | None = None) -> list[str]:
    return sorted((config or default_config()).pooled)


def load_pooled(
    names: Sequence[str],
    target: str | None = None,
    config: Config | None = None,
    name: str | None = None,
    invalid_targets: Sequence[str] | None = None,
    **kwargs: Any,
) -> FlowDataset:
    """Load several datasets and concatenate their flows into one ``FlowDataset``.

    Two invariants are checked rather than assumed, because a pool that violates either
    fails silently and expensively:

    1. **Every flow carries the pooled target.**  Otherwise ``raw_labels`` raises deep
       inside training with a per-flow ``KeyError`` that names nothing useful.  Both ISCX
       loaders happen to populate the full field set already; a future constituent may not.
    2. **Group ids stay globally unique.**  Group-aware splitting is only leakage-safe if
       one capture maps to one group; two constituents that reuse a ``source_file`` string
       would silently merge two unrelated captures into one group.
    """
    from qsentinel.data.base import load as _load_one

    names = list(names)
    if not names:
        raise ValueError("load_pooled() needs at least one dataset name")
    config = config or default_config()

    nested = [m for m in names if config.pooled_spec(m) is not None]
    if nested:
        raise ValueError(
            f"cannot pool {names}: {nested} are themselves pooled aliases; list their "
            f"members directly instead"
        )

    parts: list[FlowDataset] = []
    for member in names:
        parts.append(_load_one(member, target=target, config=config, **kwargs))

    target = target or parts[0].default_target
    flows: list[Flow] = []
    seen_sources: dict[str, str] = {}

    for member, ds in zip(names, parts):
        missing = sum(1 for f in ds.flows if target not in f.label_fields)
        if missing:
            available = sorted(ds.flows[0].label_fields) if ds.flows else []
            raise ValueError(
                f"cannot pool {names}: {missing}/{len(ds)} flows from {member!r} have no "
                f"{target!r} label field (that dataset provides {available}). Pool only "
                f"datasets that share the target."
            )
        for f in ds.flows:
            src = f.meta.get("source_file")
            if src is not None:
                owner = seen_sources.setdefault(src, member)
                if owner != member:
                    raise ValueError(
                        f"cannot pool {names}: source_file {src!r} appears in both "
                        f"{owner!r} and {member!r}, so capture groups would collide and "
                        f"a group-aware split would leak across them"
                    )
        flows.extend(ds.flows)

    pooled_invalid: set[str] = set(invalid_targets or ())
    for ds in parts:
        pooled_invalid |= set(ds.invalid_targets)

    pooled = FlowDataset(
        flows,
        name=name or "+".join(names),
        default_target=target,
        config=config,
        invalid_targets=pooled_invalid,
    )
    log.info(
        "[%s] pooled %d flows from %s -> %d classes on %r",
        pooled.name, len(pooled), names, pooled.num_classes, target,
    )
    return pooled


def load_alias(
    alias: str, target: str | None = None, config: Config | None = None, **kwargs: Any
) -> FlowDataset:
    """Load a pooled dataset by its ``configs/datasets.yaml`` alias."""
    config = config or default_config()
    spec = config.pooled_spec(alias)
    if spec is None:
        raise KeyError(
            f"unknown pooled dataset {alias!r}; configured: {available_pooled(config)}"
        )
    return load_pooled(
        spec["members"],
        target=target or spec.get("default_target"),
        config=config,
        name=alias,
        invalid_targets=spec.get("invalid_targets"),
        **kwargs,
    )
