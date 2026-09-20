"""Experts: a ``(dataset, target, members)`` binding per tailored ensemble.

Decision D2 (2026-09-19) settled the shape of Axis A: not the five protocol experts of the
original ``PHASE-4.md`` (tls / vpn / tor / app / activity) but the **three target-specific
ensembles** of ``ENSEMBLE-MAP.md`` -- E1 traffic type, E2 app-ID, E3 in-app activity.
``tunnel_type`` needs no model: Stage 1 reads it deterministically.

An expert here is a *binding*, not a combiner.  It names the dataset, the target and the
members that fit that regime, and can train each member standalone through the Phase-3
harness.  Calibration, stacking and the router that picks one expert per flow are Phase 5.

Two kinds of member:

``harness``  trained by :func:`countdown.training.train.train` from an
             :class:`~countdown.training.train.ExperimentConfig` built here;
``script``   members whose input does not come from ``Flow`` objects yet (the byte
             transformer reads ET-BERT's released corpus, the graph member reads the
             ``byte_prep`` cache).  The binding records the command that trains them, so the
             expert is complete on paper and the gap is visible in code, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from countdown.models.base import ModelRegistry
from countdown.training.train import ExperimentConfig


@dataclass(frozen=True)
class MemberSpec:
    model: str
    features: str
    feature_params: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    windows: tuple[float, ...] | None = None
    role: str = ""                       # why this member belongs to this expert
    driver: str = "harness"              # "harness" | "script"
    script: str | None = None            # the command, for driver="script"

    def check(self) -> None:
        cls = ModelRegistry.get(self.model)
        if cls.input_type != self.features:
            raise ValueError(f"{self.model} consumes {cls.input_type!r}, spec says {self.features!r}")
        if self.driver not in ("harness", "script"):
            raise ValueError(f"unknown driver {self.driver!r}")
        if self.driver == "script" and not self.script:
            raise ValueError(f"{self.model}: a script member must name its command")


@dataclass(frozen=True)
class Expert:
    name: str                            # E1 / E2 / E3
    title: str
    dataset: str
    target: str
    routes_on: str                       # the Stage-1 context Phase 5 will dispatch on
    members: tuple[MemberSpec, ...]
    test_size: float = 0.2
    val_size: float = 0.1
    notes: str = ""

    def check(self) -> None:
        if not self.members:
            raise ValueError(f"{self.name} has no members")
        for m in self.members:
            m.check()

    @property
    def default_member(self) -> MemberSpec:
        """The always-included floor: the first harness member (the GBDT, by convention)."""
        return next(m for m in self.members if m.driver == "harness")

    def member(self, model: str) -> MemberSpec:
        for m in self.members:
            if m.model == model:
                return m
        raise KeyError(f"{self.name} has no member {model!r}; it has {[m.model for m in self.members]}")

    def config(self, member: MemberSpec | str | None = None, **overrides: Any) -> ExperimentConfig:
        spec = self.default_member if member is None else (
            self.member(member) if isinstance(member, str) else member)
        if spec.driver != "harness":
            raise ValueError(f"{spec.model} is trained by a script, not the harness: {spec.script}")
        kw: dict[str, Any] = dict(
            dataset=self.dataset, target=self.target, model=spec.model, features=spec.features,
            feature_params=dict(spec.feature_params), params=dict(spec.params),
            windows=list(spec.windows) if spec.windows else None, grouped=True,
            test_size=self.test_size, val_size=self.val_size,
            notes=f"{self.name} {self.title}: {spec.role}",
        )
        kw.update(overrides)
        return ExperimentConfig(**kw)

    def train(self, member: MemberSpec | str | None = None, write: bool = True,
              runs_dir: str | Path | None = None, **overrides: Any) -> dict[str, Any]:
        """Train one member standalone (default: the floor) and return the harness result."""
        from countdown.training.train import train

        return train(self.config(member, **overrides), write=write,
                     runs_dir=Path(runs_dir) if runs_dir else None)
