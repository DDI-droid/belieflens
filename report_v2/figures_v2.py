"""Figures for the E1 v2 report (four real harnesses, six questions).

Every figure is guarded: a failure writes a placeholder carrying its traceback
rather than breaking the LaTeX build, so a bad figure is visible in the PDF
instead of stopping it.
"""
from __future__ import annotations

import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from belieflens.questions_v2 import QUESTIONS, GROUPS, TRAJECTORY, PROBE_DATE  # noqa: E402

R = ROOT / "results" / "exp1_v2"
FIGS = ROOT / "report_v2" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

H = ["analytica_full", "blf_full", "futuresim_full", "react"]
HLAB = {"analytica_full": "analytica-full", "blf_full": "blf-full",
        "futuresim_full": "futuresim-full", "react": "react"}
C = {"analytica_full": "#2a78d6", "blf_full": "#eb6834",
     "futuresim_full": "#1baf7a", "react": "#c98500"}
INK, INK2, MUT, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PANEL = "#f4f4f0"
TRUTH = {q.id: float(q.truth) for q in QUESTIONS}
QTEXT = {q.id: q.text for q in QUESTIONS}
QSHORT = {"hockey_usa": "USA gold", "hockey_can": "Canada gold",
          "ausopen_alcaraz": "Alcaraz title", "ausopen_djokovic": "Djokovic title",
          "superbowl_sea": "Seahawks win", "superbowl_ne": "Patriots win"}
GLAB = {"hockey": "Olympic hockey gold", "ausopen": "Australian Open title",
        "superbowl": "Super Bowl LX"}
STEPS = ["D1", "D2", "D3", "D4", "D5"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUT, "ytick.color": MUT,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 8,
})

M = json.loads((R / "metrics_v2.json").read_text(encoding="utf-8"))


def jl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


SEQ = defaultdict(dict)
for r in jl(R / "e1_runs.jsonl"):
    for t in r["turns"]:
        if t.get("forecast") is not None:
            SEQ[(r["harness"], r["question_id"], r["sample"])][t["date"]] = t["forecast"]
PAR = defaultdict(list)
for r in jl(R / "e1_independent.jsonl"):
    if r.get("forecast") is not None:
        PAR[(r["harness"], r["question_id"], r["date"])].append(r["forecast"])
SAMPLES = sorted({s for _, _, s in SEQ})


def save(fig, name):
    fig.savefig(FIGS / (name + ".pdf"))
    fig.savefig(FIGS / (name + ".png"), dpi=160)
    plt.close(fig)


def slope(pts):
    xs = [a for a, _ in pts]
    ys = [b for _, b in pts]
    if len(xs) < 3 or pstdev(xs) < 1e-9:
        return None
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


# ------------------------------------------------------------------ F: flow
def _flow(group):
    qa, qb = GROUPS[group]
    fig, axes = plt.subplots(4, 2, figsize=(7.0, 8.4), sharex=True, sharey=True)
    handles = None
    for i, h in enumerate(H):
        for j, q in enumerate((qa, qb)):
            ax = axes[i][j]
            dates = TRAJECTORY[q]
            X = list(range(len(dates)))
            rolls = [[SEQ[(h, q, s)].get(d) for d in dates] for s in SAMPLES
                     if (h, q, s) in SEQ]
            cols = [[v for v in (r[k] for r in rolls) if v is not None] for k in X]
            cols = [c if c else [float("nan")] for c in cols]
            ax.fill_between(X, [min(c) for c in cols], [max(c) for c in cols],
                            color=C[h], alpha=0.16, lw=0, label="sequential range (k=5)")
            for r in rolls:
                xs = [k for k in X if r[k] is not None]
                ax.plot(xs, [r[k] for k in xs], color=C[h], lw=0.7, alpha=0.45)
            ax.plot(X, [mean(c) for c in cols], color=C[h], lw=2.2, marker="o",
                    ms=3.5, label="sequential mean")
            for k, d in enumerate(dates):
                draws = PAR.get((h, q, d), [])
                if draws:
                    ax.plot([k] * len(draws), draws, ls="", marker="o", ms=3.4,
                            mfc="none", mec=MUT, mew=1.0, alpha=0.9)
            pm = [mean(PAR[(h, q, d)]) if PAR.get((h, q, d)) else float("nan")
                  for d in dates]
            ax.plot(X, pm, color=INK2, lw=1.4, ls=(0, (4, 2)), marker="_", ms=9,
                    label="parallel mean")
            ax.axhline(TRUTH[q], color=INK2, lw=0.8, ls="-", alpha=0.35)
            ax.set_ylim(-0.03, 1.03)
            ax.set_title("%s - %s (%s)" % (HLAB[h], QSHORT[q],
                                           "YES" if TRUTH[q] else "NO"),
                         loc="left", fontsize=8.4)
            if i == 3:
                ax.set_xticks(X, [d[5:] for d in dates])
            if j == 0:
                ax.set_ylabel("P(event)")
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    save(fig, "fig_flow_" + group)


def fig_flow_hockey():
    _flow("hockey")


def fig_flow_ausopen():
    _flow("ausopen")


def fig_flow_superbowl():
    _flow("superbowl")


# ------------------------------------------------------------------ F: gain
def fig_gain():
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharex=True, sharey=True)
    for ax, h in zip(axes, H):
        pts = M["per"][h]["gain_pts"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lim = 0.45
        ax.plot([-lim, lim], [-lim, lim], color=MUT, lw=0.8, ls=(0, (4, 2)))
        ax.scatter(xs, ys, s=13, color=C[h], alpha=0.55, lw=0)
        b = slope(pts)
        if b is not None:
            ax.plot([-lim, lim], [-lim * b, lim * b], color=C[h], lw=1.9)
            ax.text(0.04, 0.92, "slope = %.2f" % b, transform=ax.transAxes,
                    color=C[h], fontweight="bold", fontsize=8.4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(HLAB[h], loc="left", fontsize=8.6)
        ax.set_xlabel(r"$\Delta$ parallel mean")
    axes[0].set_ylabel(r"$\Delta$ sequential")
    fig.tight_layout()
    save(fig, "fig_gain")


# ---------------------------------------------------------------- F: anchor
def fig_anchor():
    gap = {h: mean(M["per"][h]["gap"]) for h in H}
    lag = {h: mean(M["per"][h]["lag0"]) - mean(M["per"][h]["lag1"]) for h in H}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, vals, title in ((axes[0], gap, "signed anchor gap (sequential - parallel mean)"),
                            (axes[1], lag, "lag advantage (>0 = tracks yesterday better)")):
        ys = [vals[h] for h in H]
        span = max(abs(min(ys)), abs(max(ys))) or 1.0
        ax.bar([HLAB[h] for h in H], ys, color=[C[h] for h in H], width=0.55)
        ax.axhline(0, color="#8a8a80", lw=0.9)
        for xi, v in enumerate(ys):
            ax.text(xi, v + (span * 0.09 if v >= 0 else -span * 0.09), "%+.3f" % v,
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7.8,
                    color=INK2)
        ax.set_ylim(-span * 1.6, span * 1.6)
        ax.set_title(title, loc="left", fontsize=8.4)
        ax.tick_params(axis="x", labelsize=7.4)
    fig.tight_layout()
    save(fig, "fig_anchor")


# -------------------------------------------------------------- F: stab/pit
def fig_stab_pit():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    for h in H:
        st = M["per"][h]["stab"]
        xs = [i for i, k in enumerate(STEPS) if k in st]
        ax.plot(xs, [st[STEPS[i]] for i in xs], color=C[h], lw=1.8, marker="o",
                ms=3, label=HLAB[h])
    ax.axhline(1.0, color=MUT, lw=0.9, ls=(0, (4, 2)))
    ax.set_xticks(range(5), STEPS)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.06)
    ax.set_ylabel("sd(sequential) / sd(parallel)")
    ax.set_title("Is carrying history a variance reducer?", loc="left", fontsize=8.6)
    handles, labels = ax.get_legend_handles_labels()
    ax = axes[1]
    for i, h in enumerate(H):
        vals = M["per"][h]["pit"]
        ax.scatter(vals, [i + (k % 7 - 3) * 0.045 for k, _ in enumerate(vals)],
                   s=11, color=C[h], alpha=0.35, lw=0)
        med = sorted(vals)[len(vals) // 2]
        ax.plot([med, med], [i - 0.26, i + 0.26], color=INK, lw=2.2)
    ax.axvline(50, color=MUT, lw=0.9, ls=(0, (4, 2)))
    ax.set_yticks(range(len(H)), [HLAB[h] for h in H], fontsize=8)
    ax.set_xlim(-3, 103)
    ax.set_xlabel("percentile of sequential forecast within parallel draws")
    ax.set_title("Placement (black = median; 50 = unbiased)", loc="left", fontsize=8.6)
    fig.legend(handles, labels, loc="lower center", ncols=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    save(fig, "fig_stab_pit")


# ------------------------------------------------------------- F: coherence
def fig_coherence():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ax, g in zip(axes, GROUPS):
        for h in H:
            ser = M["per"][h]["coh_seq"][g]
            xs = [i for i, k in enumerate(STEPS) if k in ser]
            ax.plot(xs, [ser[STEPS[i]] for i in xs], color=C[h], lw=1.8,
                    marker="o", ms=3, label=HLAB[h])
        ax.axhline(1.0, color=MUT, lw=1.0, ls=(0, (4, 2)))
        ax.set_xticks(range(5), STEPS)
        ax.set_ylim(0.2, 1.25)
        ax.margins(x=0.06)
        ax.set_title(GLAB[g], loc="left", fontsize=8.6)
    axes[0].set_ylabel("P(a) + P(b)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncols=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig_coherence")


# ----------------------------------------------------------------- F: brier
def fig_brier():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, cond, ttl in ((axes[0], "brier_seq", "sequential"),
                          (axes[1], "brier_par", "parallel")):
        for h in H:
            ser = M["per"][h][cond]
            xs = [i for i, k in enumerate(STEPS) if k in ser]
            ax.plot(xs, [ser[STEPS[i]] for i in xs], color=C[h], lw=1.8,
                    marker="o", ms=3, label=HLAB[h])
        ax.axhline(0.25, color=MUT, lw=1.0, ls=(0, (4, 2)))
        ax.text(0.02, 0.255, "constant 0.5", fontsize=7, color=MUT)
        ax.set_xticks(range(5), STEPS)
        ax.margins(x=0.06)
        ax.set_title("Brier vs truth - %s" % ttl, loc="left", fontsize=8.6)
    axes[0].set_ylabel("Brier (lower is better)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncols=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig_brier")


# ---------------------------------------------------------------- F: search
def fig_search():
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    w = 0.36
    for i, h in enumerate(H):
        s, p = M["per"][h]["search_seq"], M["per"][h]["search_par"]
        ax.bar(i - w / 2, s, width=w, color=C[h])
        ax.bar(i + w / 2, p, width=w, facecolor="none", edgecolor=C[h], lw=1.7)
        ax.text(i - w / 2, s + 0.5, "%.1f" % s, ha="center", fontsize=7.6, color=INK2)
        ax.text(i + w / 2, p + 0.5, "%.1f" % p, ha="center", fontsize=7.6, color=INK2)
    ax.set_xticks(range(len(H)), [HLAB[h] for h in H], fontsize=8)
    ax.set_ylabel("mean searches per forecast")
    ax.margins(y=0.2)
    ax.set_title("Search effort - filled: sequential, outlined: parallel",
                 loc="left", fontsize=8.6)
    fig.tight_layout()
    save(fig, "fig_search")


# ----------------------------------------------------------------- F: probe
def fig_probe():
    """The outcome probe: D6 sits two days after resolution, so the corpus
    already contains the answer."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axes[0]
    qids = [q.id for q in QUESTIONS]
    yl = []
    y = 0
    for q in qids:
        for h in H:
            pr = M["per"][h]["probe"][q]
            if pr["seq_prev_mean"] is None or pr["seq_mean"] is None:
                y += 1
                yl.append("")
                continue
            a, b = pr["seq_prev_mean"], pr["seq_mean"]
            ax.annotate("", xy=(b, y), xytext=(a, y),
                        arrowprops=dict(arrowstyle="-|>", color=C[h], lw=1.6,
                                        shrinkA=0, shrinkB=0))
            ax.scatter([a], [y], s=16, color=C[h], alpha=0.5, zorder=3)
            yl.append("%s %s" % (QSHORT[q][:12], HLAB[h][:9]))
            y += 1
        ax.scatter([TRUTH[q]], [y - 2.5], marker="*", s=95, color=INK, zorder=4)
        y += 1
        yl.append("")
    ax.set_yticks(range(len(yl)), yl, fontsize=5.6)
    ax.invert_yaxis()
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("P(event):  D5  ->  D6 (answer now in the corpus)")
    ax.set_title("Does the belief move to the truth? (star = truth)",
                 loc="left", fontsize=8.4)
    ax = axes[1]
    err = [mean([v["abs_err_seq"] for v in M["per"][h]["probe"].values()
                 if v["abs_err_seq"] is not None]) for h in H]
    bars = ax.barh(range(len(H)), err, color=[C[h] for h in H], height=0.6)
    for i, (b_, e) in enumerate(zip(bars, err)):
        ax.text(e + 0.012, i, "%.3f" % e, va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(H)), [HLAB[h] for h in H], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(err) * 1.35)
    ax.set_xlabel("mean |forecast - truth| at D6")
    ax.set_title("Residual error once the answer is public", loc="left", fontsize=8.4)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_probe")


# ------------------------------------------------------- F: v1 vs v2 (gain)
def fig_replication():
    v1 = {"analytica_full": 0.05, "blf_full": -0.17}
    v2 = {h: slope(M["per"][h]["gain_pts"]) for h in H}
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    names = ["analytica_full", "blf_full"]
    w = 0.34
    for i, h in enumerate(names):
        ax.bar(i - w / 2, v1[h], width=w, color=C[h], alpha=0.45,
               label="v1: 2 questions, K=3" if i == 0 else None)
        ax.bar(i + w / 2, v2[h], width=w, color=C[h],
               label="v2: 6 questions, K=5" if i == 0 else None)
        for x, v in ((i - w / 2, v1[h]), (i + w / 2, v2[h])):
            ax.text(x, v + (0.03 if v >= 0 else -0.03), "%.2f" % v, ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)
    ax.axhline(0, color="#8a8a80", lw=1.0)
    ax.axhline(1.0, color=MUT, lw=0.9, ls=(0, (4, 2)))
    ax.text(1.42, 1.02, "1.0 = tracks evidence exactly", fontsize=7, color=MUT)
    ax.set_xticks(range(len(names)), [HLAB[h] for h in names], fontsize=8.4)
    ax.set_ylim(-0.35, 1.2)
    ax.set_ylabel("update gain")
    ax.legend(loc="upper left", fontsize=7.6)
    ax.set_title("The v1 sign on blf-full does not replicate", loc="left", fontsize=8.8)
    fig.tight_layout()
    save(fig, "fig_replication")


# --------------------------------------------------------------- F: timeline
def fig_timeline():
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.set_ylim(-0.6, 2.7)
    ax.axis("off")
    for gi, (g, (qa, _)) in enumerate(GROUPS.items()):
        dates = TRAJECTORY[qa] + [PROBE_DATE[qa]]
        y = 2 - gi
        xs = list(range(6))
        ax.plot(xs[:5], [y] * 5, color="#b9b8ae", lw=1.3, zorder=1)
        ax.plot([4, 5], [y, y], color=MUT, lw=1.1, ls=(0, (4, 3)), zorder=1)
        for k, d in enumerate(dates):
            probe = (k == 5)
            ax.scatter([k], [y], s=70 if probe else 42,
                       facecolor="white" if probe else C["analytica_full"],
                       edgecolor=INK if probe else "none", lw=1.4, zorder=3)
            ax.text(k, y - 0.30, d[5:], ha="center", fontsize=7.2, color=INK2)
        ax.text(-0.55, y, GLAB[g], ha="right", va="center", fontsize=8.4,
                fontweight="bold", color=INK)
    for k, lab in enumerate(["D1", "D2", "D3", "D4", "D5", "D6"]):
        ax.text(k, 2.5, lab, ha="center", fontsize=8, color=MUT, fontweight="bold")
    ax.text(5, 2.5, "D6", ha="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_xlim(-2.4, 5.7)
    ax.text(5.0, -0.45, "D6 = resolution + 2 days: the OUTCOME PROBE,\n"
                        "by which point the corpus contains the answer",
            ha="center", fontsize=7.2, color=INK2)
    ax.text(2, -0.45, "D1..D5 = R-39, -25, -14, -7, -2\nthe forecasting trajectory",
            ha="center", fontsize=7.2, color=INK2)
    fig.tight_layout()
    save(fig, "fig_timeline")




# ------------------------------------------------------------ F: schematics
def _box(ax, x, y, w, h, title, subs, edge, fill, tcol=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.014",
                                linewidth=1.3, edgecolor=edge, facecolor=fill, zorder=2))
    # offsets are RELATIVE to the box height: absolute ones spill out of short
    # boxes and print on top of the next row.
    ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color=tcol or edge, zorder=3)
    for i, s in enumerate(subs):
        ax.text(x + w / 2, y + h * 0.42 - i * h * 0.26, s, ha="center", va="center",
                fontsize=6.3, color=INK2, zorder=3)


def _schematic(rows, titles, fname, accent):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.0, 0.80 * n + 0.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top, gap = 0.915, 0.026
    bh = (top - 0.03 - gap * (n - 1)) / n
    ax.text(0.02, 0.965, titles[0], fontsize=8.3, fontweight="bold", color=INK)
    ax.text(0.525, 0.965, titles[1], fontsize=8.3, fontweight="bold", color=accent)
    ax.plot([0.505, 0.505], [0.02, 0.94], color=GRID, lw=1)
    for i, (L, Rr) in enumerate(rows):
        y = top - bh - i * (bh + gap)
        for spec, x0 in ((L, 0.02), (Rr, 0.525)):
            title, subs, changed = spec
            edge = accent if changed else "#b9b8ae"
            fill = "#eef4fc" if changed else PANEL
            _box(ax, x0, y, 0.455, bh, title, subs, edge, fill,
                 tcol=accent if changed else INK)
        if i < n - 1:
            for xc in (0.2475, 0.7525):
                ax.annotate("", xy=(xc, y - gap + 0.004), xytext=(xc, y - 0.004),
                            arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.0))
    fig.tight_layout()
    save(fig, fname)


def fig_arch_analytica():
    a = C["analytica_full"]
    rows = [
        (("root proposition", ["one query, built from scratch"], False),
         ("proposition tree carried from yesterday", ["state, not a fresh root"], True)),
        (("Analyzer", ["expand the tree to the leaf budget"], False),
         ("Analyzer - EDIT pass", ["add / remove leaves for the new date",
                                   "prompt: most days need no change"], True)),
        (("Grounder agents (parallel)", ["one per leaf: search, then a soft",
                                         "truth value plus a written report"], False),
         ("Grounders - DELTA-grounded", ["search window = since THIS leaf last looked",
                                         "shown its own previous value and report"], True)),
        (("Synthesizer", ["emits b0 and the beta weights"], False),
         ("Synthesizer - weights REUSED", ["re-elicited only where children changed"], True)),
        (("p = b0 + sum(beta_j * p_j)", ["computed in code, never by the model"], False),
         ("p = b0 + sum(beta_j * p_j)", ["unchanged - the paper's own resynthesis"], False)),
    ]
    _schematic(rows, ("Analytica as published - one pass",
                      "As run here - one pass per simulated date"),
               "fig_arch_analytica", a)


def fig_arch_blf():
    a = C["blf_full"]
    rows = [
        (("b0 : p = 0.5", ["a fresh belief for every question"], False),
         ("b0 = yesterday's belief JSON", ["sequential; p = 0.5 in the parallel arm"], True)),
        (("loop t = 1..Tmax = 10", ["one generation returns BOTH an action",
                                    "and the updated belief"], False),
         ("loop t = 1..Tmax = 10 - unchanged", ["LeakFilter replaced by the SQL date gate,",
                                               "a strictly stronger guarantee"], True)),
        (("K = 5 trials -> shrunken logit mean", ["clamp to [0.05, 0.95]"], False),
         ("K = 5 trials - same aggregator", ["alpha = 1/(1 + var(logits)),",
                                             "our instantiation of their alpha"], True)),
        (("hierarchical Platt calibration", ["per-source intercept offsets"], False),
         ("calibration omitted", ["needs resolved training questions;",
                                  "the window contains none"], True)),
    ]
    _schematic(rows, ("BLF as published - one forecast date",
                      "As run here - belief carried across dates"),
               "fig_arch_blf", a)


def fig_arch_futuresim():
    a = C["futuresim_full"]
    rows = [
        (("environment drives the clock", ["the agent calls next_day() when",
                                           "it decides to advance"], False),
         ("WE drive the clock", ["one session per scheduled date,",
                                 "so all four harnesses share a calendar"], True)),
        (("session: one tool call per turn", ["query_df, search_news, memory CRUD,",
                                              "submit_forecasts, next_day"], False),
         ("session - unchanged", ["same action set; query_df is a real",
                                  "pandas sandbox over the question table"], False)),
        (("may end a day WITHOUT submitting", ["the standing forecast carries over"], False),
         ("MUST submit every date", ["their declining-to-update behaviour is",
                                     "disabled to keep the tiers comparable"], True)),
        (("context cleared between sessions", ["structured memory and past",
                                               "predictions are all that survive"], False),
         ("context cleared - unchanged", ["CRUD memory plus a separate",
                                          "end-of-session memory phase"], False)),
        (("hybrid semantic + keyword retrieval", ["LanceDB over 7.36M articles"], False),
         ("BM25 over the gated index", ["the same retrieval every harness gets,",
                                        "so it is a constant not a confound"], True)),
    ]
    _schematic(rows, ("FutureSim basicAgent as published",
                      "As run here - inside our protocol"),
               "fig_arch_futuresim", a)


FIGURES = [("fig_timeline", fig_timeline),
           ("fig_arch_analytica", fig_arch_analytica),
           ("fig_arch_blf", fig_arch_blf),
           ("fig_arch_futuresim", fig_arch_futuresim),
           ("fig_flow_hockey", fig_flow_hockey),
           ("fig_flow_ausopen", fig_flow_ausopen),
           ("fig_flow_superbowl", fig_flow_superbowl),
           ("fig_gain", fig_gain),
           ("fig_replication", fig_replication),
           ("fig_anchor", fig_anchor),
           ("fig_stab_pit", fig_stab_pit),
           ("fig_coherence", fig_coherence),
           ("fig_brier", fig_brier),
           ("fig_search", fig_search),
           ("fig_probe", fig_probe)]


def placeholder(name, err):
    fig, ax = plt.subplots(figsize=(7.0, 1.6))
    ax.axis("off")
    ax.text(0.5, 0.7, "FIGURE FAILED: %s" % name, ha="center", fontsize=11,
            fontweight="bold", color="#c0392b")
    ax.text(0.5, 0.32, err[-340:], ha="center", fontsize=5.6, color=INK2)
    save(fig, name)


if __name__ == "__main__":
    ok, bad = 0, []
    for name, fn in FIGURES:
        try:
            fn()
            ok += 1
            print("wrote", name)
        except Exception:
            tb = traceback.format_exc()
            print("FAILED", name, "\n", tb)
            try:
                placeholder(name, tb)
            except Exception:
                pass
            bad.append(name)
    print("\n%d figures, %d failed%s" % (ok, len(bad), (": " + ", ".join(bad)) if bad else ""))
