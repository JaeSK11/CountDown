"""Phase 6.6: the experiments runner -- registry, and byte-for-byte reproducible tables."""

from __future__ import annotations

from pathlib import Path

import pytest

from countdown.eval import experiments

RUNS = Path(experiments.ROOT) / "runs" / "ensemble"


def test_registry_and_unknown_names():
    assert set(experiments.EXPERIMENTS) == {"triage", "cross_dataset", "ablation"}
    with pytest.raises(KeyError, match="unknown experiment"):
        experiments.run(["nope"])


@pytest.mark.slow
@pytest.mark.skipif(not RUNS.exists(), reason="needs the finished Phase-5 runs")
def test_tables_are_reproducible_byte_for_byte(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    experiments.run(["ablation"], out_dir=a)
    experiments.run(["ablation"], out_dir=b)
    for f in ("ablation.json", "ablation.md"):
        assert (a / f).read_bytes() == (b / f).read_bytes()
    text = (a / "ablation.md").read_text()
    assert "| E2 | app |" in text and "20" not in text.split("\n")[0]    # no timestamps
