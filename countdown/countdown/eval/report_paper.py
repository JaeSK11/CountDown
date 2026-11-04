"""Paper tables and figures from ``results/experiments/`` (Phase 6, task 6.7).

    countdown experiments run paper        # after triage / cross_dataset / ablation

Writes ``results/paper/tables/*.tex`` (booktabs, ``\\input``-able) and
``results/paper/figures/*.{pdf,png}``. Reads only the experiment JSONs, so the paper's numbers
are exactly the regenerated tables'. Output is deterministic: PDF/PNG metadata dates are
stripped, so a re-run reproduces the files byte for byte.

Figure colours follow the data-viz method: verdict buckets are *states*, so they take the
fixed status palette (good / warning / critical / serious) plus a neutral gray for "not
observable" -- with a legend, direct labels and a hatch on ``at_risk_ceiling`` so no state is
told apart by colour alone; series comparisons take categorical slots 1-2 (blue, orange).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "results" / "paper"

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834")                       # categorical slots 1, 2
#: Stack order keeps warning away from serious (adjacent ΔE 13.6 < 15 otherwise).
BUCKETS = [  # (key, label, colour, hatch)
    ("quantum_safe", "quantum-safe", "#0ca30c", None),
    ("pqc_offered_undecided", "PQC offered, undecided", "#fab219", None),
    ("at_risk", "exposed (negotiated)", "#d03b3b", None),
    ("at_risk_ceiling", "exposed (ClientHello only)", "#ec835a", "////"),
    ("not_observable", "not observable", "#a8a69e", None),
    ("unknown", "unparsed", "#52514e", None),
]
DATASET_NAMES = {"cstnet": "CSTNET-TLS1.3", "postquantumtls": "PostQuantumTLS",
                 "iscxvpn": "ISCX-VPN", "iscxtor": "ISCX-Tor"}


def _load(name: str) -> dict[str, Any]:
    p = EXP / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run `countdown experiments run {name}` first")
    return json.loads(p.read_text())


def _pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


# -- tables -----------------------------------------------------------------------------
def table_triage(tr: dict) -> str:
    rows = []
    for key in ("cstnet", "postquantumtls", "iscxvpn", "iscxtor"):
        t = tr["datasets"][key]
        safe = t["verdict_by_flows"].get("quantum_safe", {}).get("n", 0)
        vers = ", ".join(v for v in t["tls_versions"] if v != "None")
        rows.append(f"{DATASET_NAMES[key]} & {t['flows']:,} & {_pct(t['observable_share'])} & "
                    f"{_pct(t['exposed_share_flows'])} & {_pct(t['exposed_share_bytes'])} & "
                    f"{safe} & {vers} \\\\")
    s = tr["synthetic_pqc"]["triage"]
    rows.append(f"Synthetic (ours) & {s['flows']} & {_pct(s['observable_share'])} & "
                f"{_pct(s['exposed_share_flows'])} & {_pct(s['exposed_share_bytes'])} & "
                f"{s['verdict_by_flows'].get('quantum_safe', {}).get('n', 0)} & 1.3 \\\\")
    return _tabular(
        "PQC triage per dataset. Exposed = negotiated classical key exchange, or a ClientHello "
        "listing no post-quantum group. No public dataset contains a quantum-safe handshake; the "
        "only one is in our synthetic captures.", "tab:triage", "lrrrrrl",
        "Dataset & Flows & Observable & Exposed (flows) & Exposed (bytes) & Quantum-safe & TLS",
        rows)


def table_cross(cd: dict) -> str:
    rows = [f"{p['shift']} & {p['train']} $\\rightarrow$ {p['test']} & {len(p['shared_classes'])} & "
            f"{p['in_domain']['macro_f1']:.3f} & \\textbf{{{p['cross']['macro_f1']:.3f}}} & "
            f"{-p['drop']:+.3f} \\\\" for p in cd["pairs"]]
    return _tabular(
        "Cross-dataset generalisation of the traffic-type model (GBDT, macro-F1 over shared "
        "classes). In-domain: the training side's own grouped split.", "tab:cross", "llrrrr",
        "Shift & Train $\\rightarrow$ test & Classes & In-domain & Cross & $\\Delta$", rows)


def table_ablation(ab: dict) -> str:
    rows = []
    for name, e in ab.items():
        bs, sel = e["best_single"], e["selected"]
        rows.append(f"{name} & {e['target'].replace('_', ' ')} & {bs['members'][0].replace('_', '\\_')} "
                    f"{bs['macro_f1']:.3f} & {' + '.join(m.replace('_', '\\_') for m in sel['members'])} "
                    f"({sel['combiner'].replace('_', ' ')}) {sel['macro_f1']:.3f} & "
                    f"{'yes' if e['beats_best_single'] else 'no'} \\\\")
    return _tabular(
        "Ensemble vs best single member per expert (macro-F1, grouped test split). The "
        "combiner is chosen by cross-validation on out-of-fold predictions.", "tab:ablation",
        "llllc", "Expert & Target & Best single & Selected ensemble & $\\geq$ best", rows)


def table_fidelity(tr: dict) -> str:
    rows = []
    for name, f in tr["fidelity"].items():
        cm = f["class_mix"]
        cb = f.get("class_mix_bytes", {}).get("total_variation")
        tier = f.get("tier_mix", {}).get("total_variation")
        tb = f.get("tier_mix_bytes", {}).get("total_variation")
        pair = lambda a, b: "—" if a is None else (f"{a:.3f}" if b is None else f"{a:.3f} / {b:.3f}")  # noqa: E731
        rows.append(f"{name} & {f['exposed_test_flows']:,} & {f['accuracy_on_exposed']:.3f} & "
                    f"{pair(cm['total_variation'], cb)} & {cm['worst_class'].replace('_', '\\_')} "
                    f"({100 * cm['worst_gap']:+.1f} pts) & {pair(tier, tb)} \\\\")
    return _tabular(
        "Fidelity of the predicted breakdown of exposed traffic (total-variation distance "
        "between predicted and true class mix on held-out flows, by flows / by bytes).",
        "tab:fidelity", "lrrrlr",
        "Expert & Exposed flows & Accuracy & Class-mix TV & Worst class & Tier-mix TV", rows)


def _tabular(caption: str, label: str, cols: str, head: str, rows: list[str]) -> str:
    # \resizebox keeps every table inside the text column whatever the template (needs graphicx,
    # which the figures need anyway): unscaled, all four overran a default article column.
    return ("% Generated by countdown/eval/report_paper.py -- do not edit by hand.\n"
            "% Needs \\usepackage{booktabs,graphicx}.\n"
            "\\begin{table}[t]\n\\centering\n\\small\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            "\\resizebox{\\linewidth}{!}{%\n"
            f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


# -- figures ----------------------------------------------------------------------------
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> list[Path]:
    out = OUT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    fig.savefig(paths[0], facecolor=SURFACE, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(paths[1], dpi=200, facecolor=SURFACE, metadata={"Software": None})
    return paths


def figure_exposure(tr: dict) -> list[Path]:
    import matplotlib.pyplot as plt

    keys = ["cstnet", "postquantumtls", "iscxvpn", "iscxtor"]
    labels, shares = [], []
    for k in keys:
        t = tr["datasets"][k]
        for unit in ("flows", "bytes"):
            labels.append(f"{DATASET_NAMES[k]} · {unit}")
            shares.append({b: t[f"verdict_by_{unit}"].get(b, {}).get("share", 0.0) for b, *_ in BUCKETS})
    # A legend entry for a segment nobody can see is noise: drop buckets under 0.5 % everywhere
    # (CSTNET's 10 unparsed flows of 46k); the tables keep the exact counts.
    present = [b for b in BUCKETS if any(s[b[0]] >= 0.005 for s in shares)]
    fig, ax = plt.subplots(figsize=(6.6, 3.4), facecolor=SURFACE)
    _style(ax)
    y = list(range(len(labels)))[::-1]
    left = [0.0] * len(labels)
    for key, lab, col, hatch in present:
        w = [s[key] for s in shares]
        ax.barh(y, w, left=left, height=0.62, color=col, edgecolor=SURFACE, linewidth=1.5,
                hatch=hatch, label=lab)
        for yi, li, wi in zip(y, left, w):
            if wi >= 0.08:     # selective direct labels: only segments wide enough to read
                ax.text(li + wi / 2, yi, f"{100 * wi:.0f}%", ha="center", va="center",
                        fontsize=7, color="#ffffff" if col in ("#d03b3b", "#0ca30c", "#52514e") else INK)
        left = [a + b for a, b in zip(left, w)]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("share of traffic", color=INK2, fontsize=8)
    ax.legend(ncol=len(present), loc="lower center", bbox_to_anchor=(0.42, 1.0), frameon=False,
              fontsize=7, handlelength=1.2, columnspacing=1.0, labelcolor=INK2)
    fig.tight_layout()
    paths = _save(fig, "exposure")
    plt.close(fig)
    return paths


def _paired(ax, cats: list[str], a: list[float], b: list[float], la: str, lb: str,
            digits: int = 2) -> None:
    y = list(range(len(cats)))[::-1]
    h = 0.36
    ax.barh([v + h / 2 + 0.02 for v in y], a, height=h, color=SERIES[0], label=la)
    ax.barh([v - h / 2 - 0.02 for v in y], b, height=h, color=SERIES[1], label=lb)
    for yi, va, vb in zip(y, a, b):
        ax.text(va + 0.01, yi + h / 2 + 0.02, f"{va:.{digits}f}", va="center", fontsize=7, color=INK2)
        ax.text(vb + 0.01, yi - h / 2 - 0.02, f"{vb:.{digits}f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, cats)
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro-F1", color=INK2, fontsize=8)
    ax.legend(loc="lower right", frameon=False, fontsize=7, labelcolor=INK2)


def figure_cross(cd: dict) -> list[Path]:
    import matplotlib.pyplot as plt

    pairs = cd["pairs"]
    fig, ax = plt.subplots(figsize=(6.0, 3.3), facecolor=SURFACE)
    _style(ax)
    _paired(ax, [f"{p['train']} → {p['test']}" for p in pairs],
            [p["in_domain"]["macro_f1"] for p in pairs], [p["cross"]["macro_f1"] for p in pairs],
            "in-domain", "cross")
    fig.tight_layout()
    paths = _save(fig, "cross_dataset")
    plt.close(fig)
    return paths


def figure_ablation(ab: dict) -> list[Path]:
    import matplotlib.pyplot as plt

    names = list(ab)
    fig, ax = plt.subplots(figsize=(5.4, 2.2), facecolor=SURFACE)
    _style(ax)
    _paired(ax, [f"{n} {ab[n]['target'].replace('_', ' ')}" for n in names],
            [ab[n]["best_single"]["macro_f1"] for n in names],
            [ab[n]["selected"]["macro_f1"] for n in names], "best single member", "selected ensemble",
            digits=3)   # E1 is a tie (0.6654 vs 0.6648); two decimals would invent a gap
    fig.tight_layout()
    paths = _save(fig, "ablation")
    plt.close(fig)
    return paths


def build() -> tuple[dict[str, Any], str]:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "countdown",
                                "pdf.fonttype": 42})
    tr, cd, ab = _load("triage"), _load("cross_dataset"), _load("ablation")
    tables = {"triage": table_triage(tr), "cross_dataset": table_cross(cd),
              "ablation": table_ablation(ab), "fidelity": table_fidelity(tr)}
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    for n, tex in tables.items():
        (OUT / "tables" / f"{n}.tex").write_text(tex)
    figs = figure_exposure(tr) + figure_cross(cd) + figure_ablation(ab)
    index = {"tables": sorted(f"tables/{n}.tex" for n in tables),
             "figures": sorted(str(p.relative_to(OUT)) for p in figs)}
    md = ["# Paper tables and figures", "", "Generated from `results/experiments/` by "
          "`countdown experiments run paper`.", ""]
    md += [f"- `{p}`" for p in index["tables"] + index["figures"]]
    return index, "\n".join(md) + "\n"
