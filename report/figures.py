"""Report figures (matplotlib -> vector PDF + PNG preview).

Design rules (dataviz skill): fixed categorical order, thin marks, direct
labels over legends where possible, one axis per panel, recessive grid, no
color-alone identity. The F1 redesign replaces five same-hue rollout lines
(illegible) with a min-max band + bold mean path for the sequential condition,
against open dots + dashed mean for the parallel condition.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "results" / "exp1"
FIGS = ROOT / "report" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# palette (validated; react uses slot-4's darker step for white-surface contrast)
C = {"analytica": "#2a78d6", "bayesian": "#eb6834",
     "futuresim": "#1baf7a", "react": "#c98500"}
INK, INK2, MUT, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
HARNESSES = ["analytica", "bayesian", "futuresim", "react"]
QUESTIONS = ["hockey_usa", "hockey_can"]
QLAB = {"hockey_usa": "USA gold (resolves YES)", "hockey_can": "Canada gold (resolves NO)"}
TRUTH = {"hockey_usa": 1.0, "hockey_can": 0.0}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUT, "ytick.color": MUT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "legend.frameon": False, "legend.fontsize": 7.5,
})

M = json.loads((R1 / "seqpar_metrics.json").read_text(encoding="utf-8"))
DATES = M["dates"]
DL = [d[5:] for d in DATES]
X = range(len(DATES))

seqrows: dict = {}
for name in ("e1_runs.jsonl", "e1_runs_extra.jsonl"):
    for line in (R1 / name).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["question_id"] in QUESTIONS:
            for t in r["turns"]:
                if t["forecast"] is not None:
                    seqrows.setdefault((r["harness"], r["question_id"], r["sample"]),
                                       {})[t["date"]] = t["forecast"]
par: dict = {}
for line in (R1 / "e1_independent.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r["forecast"] is not None:
        par.setdefault((r["harness"], r["question_id"], r["date"]), []).append(r["forecast"])


def save(fig, name):
    fig.savefig(FIGS / (name + ".pdf"))
    fig.savefig(FIGS / (name + ".png"), dpi=160)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ F1 flow
fig, axes = plt.subplots(4, 2, figsize=(7.4, 9.2), sharex=True, sharey=True)
for i, h in enumerate(HARNESSES):
    for j, q in enumerate(QUESTIONS):
        ax = axes[i][j]
        col = C[h]
        rolls = [[seqrows[(h, q, s)].get(d) for d in DATES]
                 for s in range(5) if (h, q, s) in seqrows]
        cols_ok = [[v for v in (r[k] for r in rolls) if v is not None] for k in X]
        band_lo = [min(v) for v in cols_ok]
        band_hi = [max(v) for v in cols_ok]
        m_seq = [mean(v) for v in cols_ok]
        ax.fill_between(X, band_lo, band_hi, color=col, alpha=0.16, lw=0,
                        label="sequential range (k=5)")
        for r in rolls:
            xs = [k for k in X if r[k] is not None]
            ax.plot(xs, [r[k] for k in xs], color=col, lw=0.7, alpha=0.45)
        ax.plot(X, m_seq, color=col, lw=2.2, marker="o", ms=3.5,
                label="sequential mean")
        pm = [mean(par[(h, q, d)]) for d in DATES]
        for k, d in enumerate(DATES):
            ax.plot([k] * len(par[(h, q, d)]), par[(h, q, d)], ls="",
                    marker="o", ms=3.6, mfc="none", mec=MUT, mew=1.0, alpha=0.9)
        ax.plot(X, pm, color=INK2, lw=1.4, ls=(0, (4, 2)), marker="_", ms=9,
                label="parallel mean")
        ax.axvline(4, color="#c3c2b7", lw=0.8, ls=":")
        ax.axhline(TRUTH[q], color=INK2, lw=0.7, ls="-", alpha=0.35)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title("%s — %s" % (h, QLAB[q]), loc="left")
        if i == 3:
            ax.set_xticks(list(X), DL)
        if j == 0:
            ax.set_ylabel("P(event)")
        if i == 0 and j == 0:
            ax.legend(loc="upper left", ncols=1)
fig.tight_layout()
save(fig, "fig_flow")

# ------------------------------------------------------------------ F2 gain
fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.7), sharex=True, sharey=True)
for ax, h in zip(axes, HARNESSES):
    pts = M["per"][h]["gain_pts"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lim = 0.5
    ax.plot([-lim, lim], [-lim, lim], color=MUT, lw=0.8, ls=(0, (4, 2)))
    ax.scatter(xs, ys, s=16, color=C[h], alpha=0.65, lw=0)
    b = M["per"][h]["gain_slope"]
    if b is not None:
        ax.plot([-lim, lim], [-lim * b, lim * b], color=C[h], lw=1.8)
        ax.text(0.05, 0.92, "slope = %.2f" % b, transform=ax.transAxes,
                color=C[h], fontweight="bold", fontsize=8.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(h, loc="left")
    ax.set_xlabel(r"$\Delta$ parallel mean")
axes[0].set_ylabel(r"$\Delta$ sequential")
fig.tight_layout()
save(fig, "fig_gain")

# ------------------------------------------------------------------ F3 bars
gap = {h: mean(M["per"][h]["gap"]) for h in HARNESSES}
lag = {h: mean(M["per"][h]["lag0"]) - mean(M["per"][h]["lag1"]) for h in HARNESSES}
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5))
for ax, vals, title in ((axes[0], gap, "signed anchor gap  (sequential − parallel mean)"),
                        (axes[1], lag, "lag advantage  (>0 ⇒ tracks yesterday better)")):
    ys = [vals[h] for h in HARNESSES]
    bars = ax.bar(HARNESSES, ys, color=[C[h] for h in HARNESSES], width=0.55)
    ax.axhline(0, color="#c3c2b7", lw=0.9)
    for b_, v in zip(bars, ys):
        ax.text(b_.get_x() + b_.get_width() / 2, v + (0.002 if v >= 0 else -0.006),
                "%+.3f" % v, ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7.5, color=INK2)
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.margins(y=0.25)
fig.tight_layout()
save(fig, "fig_anchor")

# --------------------------------------------------------------- F4/F5 pair
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
ax = axes[0]
for h in HARNESSES:
    st = M["per"][h]["stab"]
    xs = [k for k, d in enumerate(DATES) if d in st]
    ax.plot(xs, [st[DATES[k]] for k in xs], color=C[h], lw=1.8, marker="o", ms=3)
    ax.annotate(h, (xs[-1], st[DATES[xs[-1]]]), xytext=(4, 0), fontsize=7,
                textcoords="offset points", color=C[h], fontweight="bold")
ax.axhline(1.0, color=MUT, lw=0.8, ls=(0, (4, 2)))
ax.text(0.05, 0.96, "1 = same spread as fresh eyes", fontsize=7, color=MUT,
        transform=ax.transAxes, ha="left", va="top")
ax.set_xticks(list(X), DL)
ax.set_ylabel("sd(sequential) / sd(parallel)")
ax.set_title("Is history a variance reducer?", loc="left")
ax = axes[1]
for i, h in enumerate(HARNESSES):
    vals = M["per"][h]["pit"]
    ax.scatter(vals, [i + (k % 7 - 3) * 0.035 for k, _ in enumerate(vals)],
               s=14, color=C[h], alpha=0.4, lw=0)
    med = sorted(vals)[len(vals) // 2]
    ax.plot([med, med], [i - 0.22, i + 0.22], color=INK, lw=2)
ax.axvline(50, color=MUT, lw=0.8, ls=(0, (4, 2)))
ax.set_yticks(range(4), HARNESSES)
ax.set_xlim(-3, 103)
ax.set_xlabel("percentile of sequential forecast within parallel draws")
ax.set_title("Placement (black = median; 50 = unbiased)", loc="left")
fig.tight_layout()
save(fig, "fig_stab_pit")

# ----------------------------------------------------------- F6/F7 2x2
fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.6), sharex=True)
for j, (fld, ttl, ylim, ref) in enumerate(
        (("coh", "Complement coherence  P(USA)+P(CAN)", (0.4, 1.35), 1.0),
         ("brier", "Brier vs truth (lower is better)", (0.0, 0.62), None))):
    for k, cond in enumerate(("seq", "par")):
        ax = axes[j][k]
        for h in HARNESSES:
            ser = M["per"][h]["%s_%s" % (fld, cond)]
            xs = [i for i, d in enumerate(DATES) if d in ser]
            ax.plot(xs, [ser[DATES[i]] for i in xs], color=C[h], lw=1.8,
                    marker="o", ms=2.8)
            dy = {"analytica": 5, "bayesian": -5, "futuresim": 10, "react": -10}[h]
            ax.annotate(h[:4], (xs[-1], ser[DATES[xs[-1]]]), xytext=(4, dy),
                        textcoords="offset points", fontsize=6.5, color=C[h],
                        fontweight="bold", annotation_clip=False)
        if ref is not None:
            ax.axhline(ref, color=MUT, lw=0.8, ls=(0, (4, 2)))
        ax.set_ylim(*ylim)
        ax.set_title("%s — %s" % (ttl, "sequential" if cond == "seq" else "parallel"),
                     loc="left", fontsize=8.3)
        if j == 1:
            ax.set_xticks(list(X), DL)
fig.tight_layout()
save(fig, "fig_coh_brier")

# ------------------------------------------------------------- recovery
rep = json.loads((R1 / "e2_programs.json").read_text(encoding="utf-8"))
bd = json.loads((R1 / "beliefdyn.json").read_text(encoding="utf-8"))
groups = ["bayesian/hockey_usa", "bayesian/hockey_can", "futuresim/hockey_can",
          "futuresim/hockey_usa", "react/hockey_can", "react/hockey_usa",
          "analytica/hockey_usa", "analytica/hockey_can"]
ins, oos, tmp = [], [], []
for g in groups:
    v = rep[g]
    f = (v.get("fit") or {}).get("mae")
    ins.append(f if (f is not None and f == f) else None)
    o = (v.get("oos") or {}).get("test_mae")
    oos.append(o if (o is not None and o == o) else None)
    t = (bd.get("temporal_holdout") or {}).get(g, {}).get("test_mae")
    tmp.append(t if (t is not None and t == t) else None)
fig, ax = plt.subplots(figsize=(7.4, 3.0))
w = 0.26
for off, series, lab, alpha in ((-w, ins, "in-sample fit", 1.0),
                               (0, oos, "sample holdout (blind)", 0.75),
                               (w, tmp, "temporal holdout, late days (blind)", 0.5)):
    for i, v in enumerate(series):
        h = groups[i].split("/")[0]
        if v is None:
            ax.text(i + off, 0.0006, "×", ha="center", color=MUT, fontsize=9)
            continue
        ax.bar(i + off, max(v, 4e-4), width=w * 0.92, color=C[h], alpha=alpha)
ax.set_yscale("log")
ax.set_ylim(4e-4, 1.0)
ax.axhline(0.02, color=INK2, lw=0.9, ls=(0, (4, 2)))
ax.text(7.45, 0.022, "tolerance 0.02", fontsize=7, color=INK2, ha="right")
ax.set_xticks(range(len(groups)),
              [g.replace("hockey_", "").replace("/", "\n") for g in groups], fontsize=7.5)
ax.set_ylabel("MAE (log scale)")
ax.set_title("Program recovery: one literal-free program per group — fit and blind holdouts "
             "(darker→lighter: in-sample, sample-holdout, temporal)", loc="left", fontsize=8.5)
fig.tight_layout()
save(fig, "fig_recovery")

# ---------------------------------------------------------------- probe
pr = json.loads((R1 / "probe_results.json").read_text(encoding="utf-8"))
rc = json.loads((R1 / "probe_rule_check.json").read_text(encoding="utf-8"))
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
ax = axes[0]
order = [(j["key"], j["sample"], j["condition"]) for j in pr if j["condition"] != "placebo"]
ys = range(len(order))
for yi, j in enumerate([j for j in pr if j["condition"] != "placebo"]):
    h = j["key"].split("/")[0]
    act = (j.get("p_probe") or 0) - j["p_last"]
    ax.plot([j["implied"], act], [yi, yi], color=GRID, lw=1.2, zorder=1)
    ax.scatter([j["implied"]], [yi], marker="s", s=26, color=INK2, zorder=2)
    ax.scatter([act], [yi], marker="o", s=30, color=C[h], zorder=3)
ax.axvline(0, color="#c3c2b7", lw=0.8)
ax.set_yticks(list(ys), ["%s s%d %s" % (k.split("/")[0], s, c) for k, s, c in order],
              fontsize=6.2)
ax.invert_yaxis()
ax.set_xlabel(r"forecast change on probe day  ($\Delta$)")
ax.set_title("Rigidity probe: implied by own rule (square) vs actual (dot)",
             loc="left", fontsize=8.3)
ax = axes[1]
conds = ["toward", "against", "placebo"]
for i, h in enumerate(HARNESSES):
    for k, c_ in enumerate(conds):
        gs = [abs(r["gap"]) for r in rc if r["key"].startswith(h) and r["condition"] == c_]
        if gs:
            ax.bar(i + (k - 1) * 0.27, mean(gs), width=0.25, color=C[h],
                   alpha=[1.0, 0.65, 0.35][k])
ax.set_xticks(range(4), HARNESSES)
ax.set_ylabel("|actual − rule-implied|")
ax.set_title("Rule-compliance gap (dark→light: toward, against, placebo)",
             loc="left", fontsize=8.3)
fig.tight_layout()
save(fig, "fig_probe")

# ----------------------------------------------------- exemplar trajectory
fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))
tj = rep["bayesian/hockey_usa"]["trajectory"]["s0"]
dts = sorted(tj)
ax = axes[0]
for name, col in (("lr_recent_game_results", "#2a78d6"),
                  ("lr_opponent_canada_strength", "#c98500"),
                  ("prior_odds_usa_gold", "#1baf7a")):
    ax.plot(range(len(dts)), [tj[d].get(name, 0) for d in dts], lw=1.8, marker="o",
            ms=3, color=col)
    ax.annotate(name.replace("lr_", "").replace("_", " ")[:18],
                (len(dts) - 1, tj[dts[-1]].get(name, 0)), xytext=(4, 0),
                textcoords="offset points", fontsize=6.5, color=col,
                fontweight="bold", annotation_clip=False)
ax.axhline(1.0, color=MUT, lw=0.7, ls=(0, (4, 2)))
ax.set_xticks(range(len(dts)), [d[5:] for d in dts])
ax.margins(x=0.14)
ax.set_title("bayesian/USA: belief in its discovered coordinates (s0)", loc="left",
             fontsize=8.3)
ax.set_ylabel("value")
ax = axes[1]
ax.plot(range(len(dts)), [tj[d]["_target"] for d in dts], color="#eb6834", lw=2.2,
        marker="o", ms=3.5)
th = bd["temporal_holdout"].get("bayesian/hockey_usa", {})
ax.set_xticks(range(len(dts)), [d[5:] for d in dts])
ax.set_ylim(0, 0.6)
ax.set_title("…and the forecast it produces (early-theory program predicts late days "
             "at MAE %.3f)" % th.get("test_mae", float("nan")), loc="left", fontsize=8.3)
ax.set_ylabel("P(USA gold)")
fig.tight_layout()
save(fig, "fig_exemplar")

print("all figures done")
