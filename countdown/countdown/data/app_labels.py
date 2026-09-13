"""Application-label normalisation for the ISCX capture stems.

The problem this solves, and why it is not a loader concern
-----------------------------------------------------------
Both ISCX loaders store the capture-filename stem as ``app``.  That is faithful to the
source but useless as a class set: ISCXVPN reports 135 distinct values where
``facebook_audio1a``, ``facebook_audio2`` and ``facebook_audio4`` are one app+modality
recorded three times, and ISCXTor reports 95 where ``chat_gate_skype_chat`` and
``voip_gate_skype_audio`` are Tor-gateway spellings of Skype.  With roughly one value per
source file, a ``source_file`` group-aware split is undefined -- which is exactly why
``configs/datasets.yaml`` marks ``app`` in ``invalid_targets``.

This module recovers the real class set as two *additional* fields, leaving ``app`` and
its guard untouched:

``app_norm``
    The service.  ``facebook``, ``skype``, ``hangouts``, ``youtube``, ``spotify``, ...
``app_activity``
    Service + modality, e.g. ``facebook_audio``, ``skype_video``.  The modality token
    comes from the flow's already-assigned ``traffic_type``, so the two labels cannot
    disagree.  This is the naming Okonkwo's non-VPN application task uses (Fig. 5a),
    which is what makes that experiment reproducible here.

Applied from ``DatasetLoader.postprocess()`` rather than ``discover()``.  ``discover``
writes ``label_fields`` into the parquet cache and ``Config.flow_key`` hashes the dataset
spec, so doing it there would orphan every cache and force a re-parse of the 34 GB ISCX
corpus to add a label that cannot change ``X``.  ``postprocess`` runs after the cache read,
so the rules can be edited and re-applied for free.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from countdown.config import get_logger
from countdown.schema import Flow

log = get_logger(__name__)

#: Fallback when a traffic_type has no configured modality token.
_UNKNOWN_MODALITY = "other"


class AppLabelRules:
    """Compiled, ordered stem -> service rules plus the traffic_type -> modality map."""

    def __init__(
        self,
        rules: Sequence[dict[str, str]],
        modality: dict[str, str] | None = None,
    ) -> None:
        if not rules:
            raise ValueError("AppLabelRules needs at least one rule")
        self.rules = [(re.compile(r["pattern"]), r["app"]) for r in rules]
        self.modality = dict(modality or {})

    @classmethod
    def from_config(cls, config: Any) -> "AppLabelRules":
        return cls(config.app_rules, config.app_modality)

    # -- the mapping -------------------------------------------------------------------
    def service(self, stem: str) -> str | None:
        """First matching rule's app, or ``None`` if the stem matched nothing."""
        s = str(stem).lower()
        for pattern, app in self.rules:
            if pattern.search(s):
                return app
        return None

    def modality_token(self, traffic_type: str | None) -> str:
        return self.modality.get(str(traffic_type), _UNKNOWN_MODALITY)

    def activity(self, stem: str, traffic_type: str | None) -> str | None:
        app = self.service(stem)
        if app is None:
            return None
        return f"{app}_{self.modality_token(traffic_type)}"


def normalize_app_labels(
    flows: Iterable[Flow],
    rules: AppLabelRules,
    on_unmapped: str = "error",
    dataset: str = "",
) -> list[Flow]:
    """Add ``app_norm`` / ``app_activity`` to every flow, in place.

    A stem matching no rule is reported by *stem*, not by flow -- one unmatched capture
    can contribute thousands of flows and a per-flow error message would bury the one
    fact needed to fix it (which pattern to add).
    """
    flows = list(flows)
    unmapped: dict[str, int] = {}

    for f in flows:
        stem = f.label_fields.get("app")
        if stem is None:
            continue
        app = rules.service(stem)
        if app is None:
            unmapped[str(stem)] = unmapped.get(str(stem), 0) + 1
            continue
        traffic_type = f.label_fields.get("traffic_type")
        f.label_fields["app_norm"] = app
        f.label_fields["app_activity"] = f"{app}_{rules.modality_token(traffic_type)}"

    if unmapped:
        msg = (
            f"[{dataset or 'app_labels'}] {len(unmapped)} capture stems matched no "
            f"app rule ({sum(unmapped.values())} flows): "
            f"{sorted(unmapped)[:20]} -- add a pattern to app_rules: in "
            f"configs/datasets.yaml"
        )
        if on_unmapped == "error":
            raise ValueError(msg)
        log.warning(msg)

    return flows
