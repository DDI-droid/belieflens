"""BeliefLens report figures -> vector PDF (+ PNG preview) in report/figs/.

Every figure is a function registered in FIGURES.  The driver calls each one
inside a guard: if a figure raises, a placeholder carrying the traceback is
written in its place, so a single bad figure can never break the LaTeX build --
the failure shows up in the compiled PDF instead, where it is visible.

Design rules (dataviz): fixed categorical order, thin marks, direct labels over
legends where possible, one axis per panel, recessive grid, never colour-alone
identity.
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "results" / "exp1"
R2 = ROOT / "results" / "exp1_real"
FIGS = ROOT / "report" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- palette
C = {"analytica": "#2a78d6", "bayesian": "#eb6834",
     "futuresim": "#1baf7a", "react": "#c98500",
     "analytica_full": "#2a78d6", "blf_full": "#eb6834"}
INK, INK2, MUT, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
PANEL = "#f4f4f0"
HARNESSES = ["analytica", "bayesian", "futuresim", "react"]
REAL = ["analytica_full", "blf_full"]
RLAB = {"analytica_full": "analytica-full", "blf_full": "blf-full"}
QUESTIONS = ["hockey_usa", "hockey_can"]
QLAB = {"hockey_usa": "USA gold (resolves YES)", "hockey_can": "Canada gold (resolves NO)"}
QSHORT = {"hockey_usa": "USA gold", "hockey_can": "Canada gold"}
TRUTH = {"hockey_usa": 1.0, "hockey_can": 0.0}

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


# ---------------------------------------------------------------- data load
def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jlines(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


M = jload(R1 / "seqpar_metrics.json")
DATES = M["dates"]
DL = [d[5:] for d in DATES]
X = list(range(len(DATES)))

seqrows: dict = {}
for _name in ("e1_runs.jsonl", "e1_runs_extra.jsonl"):
    for r in jlines(R1 / _name):
        if r["question_id"] in QUESTIONS:
            for t in r["turns"]:
                if t["forecast"] is not None:
                    seqrows.setdefault((r["harness"], r["question_id"], r["sample"]),
                                       {})[t["date"]] = t["forecast"]
par: dict = {}
for r in jlines(R1 / "e1_independent.jsonl"):
    if r["forecast"] is not None:
        par.setdefault((r["harness"], r["question_id"], r["date"]), []).append(r["forecast"])

HAVE_REAL = (R2 / "seqpar_metrics.json").exists()
if HAVE_REAL:
    M2 = jload(R2 / "seqpar_metrics.json")
    seq2: dict = {}
    for r in jlines(R2 / "e1_runs.jsonl"):
        for t in r["turns"]:
            if t["forecast"] is not None:
                seq2.setdefault((r["harness"], r["question_id"], r["sample"]),
                                {})[t["date"]] = t["forecast"]
    par2: dict = {}
    for r in jlines(R2 / "e1_independent.jsonl"):
        if r["forecast"] is not None:
            par2.setdefault((r["harness"], r["question_id"], r["date"]), []).append(r["forecast"])


def save(fig, name):
    fig.savefig(FIGS / (name + ".pdf"))
    fig.savefig(FIGS / (name + ".png"), dpi=160)
    plt.close(fig)


def rolls_of(store, h, q):
    return [[store[(h, q, s)].get(d) for d in DATES]
            for s in range(8) if (h, q, s) in store]


def flow_panel(ax, store, parstore, h, q, col, title):
    """Band + bold mean for sequential, open dots + dashed mean for parallel."""
    rolls = rolls_of(store, h, q)
    cols = [[v for v in (r[k] for r in rolls) if v is not None] for k in X]
    ax.fill_between(X, [min(v) for v in cols], [max(v) for v in cols],
                    color=col, alpha=0.16, lw=0, label="sequential range (k=5)")
    for r in rolls:
        xs = [k for k in X if r[k] is not None]
        ax.plot(xs, [r[k] for k in xs], color=col, lw=0.7, alpha=0.45)
    ax.plot(X, [mean(v) for v in cols], color=col, lw=2.2, marker="o", ms=3.5,
            label="sequential mean")
    for k, d in enumerate(DATES):
        draws = parstore.get((h, q, d), [])
        if draws:
            ax.plot([k] * len(draws), draws, ls="", marker="o", ms=3.6,
                    mfc="none", mec=MUT, mew=1.0, alpha=0.9)
    pm = [mean(parstore[(h, q, d)]) for d in DATES if (h, q, d) in parstore]
    if len(pm) == len(DATES):
        ax.plot(X, pm, color=INK2, lw=1.4, ls=(0, (4, 2)), marker="_", ms=9,
                label="parallel mean")
    ax.axvline(len(DATES) - 1, color="#c3c2b7", lw=0.8, ls=":")
    ax.axhline(TRUTH[q], color=INK2, lw=0.7, ls="-", alpha=0.35)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title, loc="left")


# ======================================================================= F1
def fig_flow():
    fig, axes = plt.subplots(4, 2, figsize=(7.0, 8.8), sharex=True, sharey=True)
    for i, h in enumerate(HARNESSES):
        for j, q in enumerate(QUESTIONS):
            ax = axes[i][j]
            flow_panel(ax, seqrows, par, h, q, C[h], "%s — %s" % (h, QLAB[q]))
            if i == 3:
                ax.set_xticks(X, DL)
            if j == 0:
                ax.set_ylabel("P(event)")
            if i == 0 and j == 0:
                ax.legend(loc="upper left", ncols=1)
    fig.tight_layout()
    save(fig, "fig_flow")


# ======================================================================= F2
def _gain_panel(ax, metrics, h, label, col):
    pts = metrics["per"][h]["gain_pts"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lim = 0.5
    ax.plot([-lim, lim], [-lim, lim], color=MUT, lw=0.8, ls=(0, (4, 2)))
    ax.scatter(xs, ys, s=16, color=col, alpha=0.65, lw=0)
    b = metrics["per"][h]["gain_slope"]
    if b is not None:
        ax.plot([-lim, lim], [-lim * b, lim * b], color=col, lw=1.8)
        ax.text(0.05, 0.92, "slope = %.2f" % b, transform=ax.transAxes,
                color=col, fontweight="bold", fontsize=8.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(label, loc="left")
    ax.set_xlabel(r"$\Delta$ parallel mean")


def fig_gain():
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharex=True, sharey=True)
    for ax, h in zip(axes, HARNESSES):
        _gain_panel(ax, M, h, h, C[h])
    axes[0].set_ylabel(r"$\Delta$ sequential")
    fig.tight_layout()
    save(fig, "fig_gain")


# ======================================================================= F3
def fig_anchor():
    gap = {h: mean(M["per"][h]["gap"]) for h in HARNESSES}
    lag = {h: mean(M["per"][h]["lag0"]) - mean(M["per"][h]["lag1"]) for h in HARNESSES}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    pairs = ((axes[0], gap, "signed anchor gap  (sequential − parallel mean)"),
             (axes[1], lag, "lag advantage  (>0 ⇒ tracks yesterday better)"))
    for ax, vals, title in pairs:
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


# ==================================================================== F4/F5
def _stab_pit(metrics, names, colmap, labmap, fname, title_sfx):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for h in names:
        st = metrics["per"][h]["stab"]
        xs = [k for k, d in enumerate(DATES) if d in st]
        if not xs:
            continue
        ax.plot(xs, [st[DATES[k]] for k in xs], color=colmap[h], lw=1.8, marker="o", ms=3)
        ax.annotate(labmap[h], (xs[-1], st[DATES[xs[-1]]]), xytext=(4, 0), fontsize=7,
                    textcoords="offset points", color=colmap[h], fontweight="bold",
                    annotation_clip=False)
    ax.axhline(1.0, color=MUT, lw=0.8, ls=(0, (4, 2)))
    ax.text(0.04, 0.96, "1 = same spread as fresh eyes", fontsize=7, color=MUT,
            transform=ax.transAxes, ha="left", va="top")
    ax.set_xticks(X, DL)
    ax.margins(x=0.16)
    ax.set_ylabel("sd(sequential) / sd(parallel)")
    ax.set_title("Is history a variance reducer?" + title_sfx, loc="left")
    ax = axes[1]
    for i, h in enumerate(names):
        vals = metrics["per"][h]["pit"]
        if not vals:
            continue
        ax.scatter(vals, [i + (k % 7 - 3) * 0.035 for k, _ in enumerate(vals)],
                   s=14, color=colmap[h], alpha=0.4, lw=0)
        med = sorted(vals)[len(vals) // 2]
        ax.plot([med, med], [i - 0.22, i + 0.22], color=INK, lw=2)
    ax.axvline(50, color=MUT, lw=0.8, ls=(0, (4, 2)))
    ax.set_yticks(range(len(names)), [labmap[h] for h in names])
    ax.set_xlim(-3, 103)
    ax.set_xlabel("percentile of sequential forecast within parallel draws")
    ax.set_title("Placement (black = median; 50 = unbiased)", loc="left")
    fig.tight_layout()
    save(fig, fname)


def fig_stab_pit():
    _stab_pit(M, HARNESSES, C, {h: h for h in HARNESSES}, "fig_stab_pit", "")


# ==================================================================== F6/F7
def _coh_brier(metrics, names, colmap, labmap, fname):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True)
    spec = (("coh", "Coherence  P(USA)+P(CAN)", (0.4, 1.35), 1.0),
            ("brier", "Brier vs truth (lower is better)", (0.0, 0.62), 0.25))
    for j, (fld, ttl, ylim, ref) in enumerate(spec):
        for k, cond in enumerate(("seq", "par")):
            ax = axes[j][k]
            for n_i, h in enumerate(names):
                ser = metrics["per"][h]["%s_%s" % (fld, cond)]
                xs = [i for i, d in enumerate(DATES) if d in ser]
                if not xs:
                    continue
                ax.plot(xs, [ser[DATES[i]] for i in xs], color=colmap[h], lw=1.8,
                        marker="o", ms=2.8)
                dy = [6, -8, 11, -12, 15, -16][n_i % 6]
                ax.annotate(labmap[h], (xs[-1], ser[DATES[xs[-1]]]), xytext=(4, dy),
                            textcoords="offset points", fontsize=6.5, color=colmap[h],
                            fontweight="bold", annotation_clip=False)
            if ref is not None:
                ax.axhline(ref, color=MUT, lw=0.8, ls=(0, (4, 2)))
            ax.set_ylim(*ylim)
            ax.margins(x=0.14)
            ax.set_title("%s — %s" % (ttl, "sequential" if cond == "seq" else "parallel"),
                         loc="left", fontsize=8.3)
            if j == 1:
                ax.set_xticks(X, DL)
    fig.tight_layout()
    save(fig, fname)


def fig_coh_brier():
    _coh_brier(M, HARNESSES, C, {h: h[:4] for h in HARNESSES}, "fig_coh_brier")


# =================================================================== search
def fig_search():
    names = HARNESSES + (REAL if HAVE_REAL else [])
    labs = [h if h in HARNESSES else RLAB[h] for h in names]
    sq = [M["per"][h]["search_seq"] if h in HARNESSES else M2["per"][h]["search_seq"]
          for h in names]
    pr = [M["per"][h]["search_par"] if h in HARNESSES else M2["per"][h]["search_par"]
          for h in names]
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    w = 0.36
    xs = list(range(len(names)))
    for i, h in enumerate(names):
        ax.bar(i - w / 2, sq[i], width=w, color=C[h],
               alpha=1.0 if h in REAL else 0.55)
        ax.bar(i + w / 2, pr[i], width=w, facecolor="none", edgecolor=C[h], lw=1.6)
        ax.text(i - w / 2, sq[i] + 0.25, "%.1f" % sq[i], ha="center", fontsize=7, color=INK2)
        ax.text(i + w / 2, pr[i] + 0.25, "%.1f" % pr[i], ha="center", fontsize=7, color=INK2)
    ax.set_xticks(xs, labs, fontsize=7.5)
    ax.set_ylabel("mean searches per turn")
    ax.margins(y=0.18)
    ax.set_title("Search effort — filled: sequential, outlined: parallel "
                 "(solid fill = real orchestration code)", loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_search")


# ================================================================= recovery
def fig_recovery():
    rep = jload(R1 / "e2_programs.json")
    bd = jload(R1 / "beliefdyn.json")
    groups = ["bayesian/hockey_usa", "bayesian/hockey_can", "futuresim/hockey_can",
              "futuresim/hockey_usa", "react/hockey_can", "react/hockey_usa",
              "analytica/hockey_usa", "analytica/hockey_can"]
    ins, oos, tmp = [], [], []
    for g in groups:
        v = rep.get(g, {})
        f = (v.get("fit") or {}).get("mae")
        ins.append(f if (f is not None and f == f) else None)
        o = (v.get("oos") or {}).get("test_mae")
        oos.append(o if (o is not None and o == o) else None)
        t = (bd.get("temporal_holdout") or {}).get(g, {}).get("test_mae")
        tmp.append(t if (t is not None and t == t) else None)
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    w = 0.26
    series = ((-w, ins, 1.0), (0, oos, 0.72), (w, tmp, 0.45))
    for off, vals, alpha in series:
        for i, v in enumerate(vals):
            h = groups[i].split("/")[0]
            if v is None:
                ax.text(i + off, 6e-4, "×", ha="center", color=MUT, fontsize=9)
                continue
            ax.bar(i + off, max(v, 4e-4), width=w * 0.92, color=C[h], alpha=alpha)
    ax.set_yscale("log")
    ax.set_ylim(4e-4, 1.0)
    ax.axhline(0.02, color=INK2, lw=0.9, ls=(0, (4, 2)))
    ax.text(7.45, 0.023, "tolerance 0.02", fontsize=7, color=INK2, ha="right")
    ax.set_xticks(range(len(groups)),
                  [g.replace("hockey_", "").replace("/", "\n") for g in groups], fontsize=7.5)
    ax.set_ylabel("MAE (log scale)")
    ax.set_title("Program recovery — one literal-free program per group. "
                 "Darker→lighter: in-sample fit, blind sample holdout, blind temporal holdout",
                 loc="left", fontsize=8.3)
    fig.tight_layout()
    save(fig, "fig_recovery")


# ==================================================================== probe
def fig_probe():
    pr = jload(R1 / "probe_results.json")
    rc = jload(R1 / "probe_rule_check.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axes[0]
    directional = [j for j in pr if j["condition"] != "placebo"]
    for yi, j in enumerate(directional):
        h = j["key"].split("/")[0]
        act = (j.get("p_probe") or 0) - j["p_last"]
        ax.plot([j["implied"], act], [yi, yi], color=GRID, lw=1.2, zorder=1)
        ax.scatter([j["implied"]], [yi], marker="s", s=26, color=INK2, zorder=2)
        ax.scatter([act], [yi], marker="o", s=30, color=C[h], zorder=3)
    ax.axvline(0, color="#c3c2b7", lw=0.8)
    ax.set_yticks(range(len(directional)),
                  ["%s s%d %s" % (j["key"].split("/")[0], j["sample"], j["condition"])
                   for j in directional], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel(r"forecast change on probe day  ($\Delta$)")
    ax.set_title("Implied by own rule (square) vs actual (dot)", loc="left", fontsize=8.3)
    ax = axes[1]
    for i, h in enumerate(HARNESSES):
        for k, c_ in enumerate(["toward", "against", "placebo"]):
            gs = [abs(r["gap"]) for r in rc
                  if r["key"].startswith(h) and r["condition"] == c_]
            if gs:
                ax.bar(i + (k - 1) * 0.27, mean(gs), width=0.25, color=C[h],
                       alpha=[1.0, 0.65, 0.35][k])
    ax.set_xticks(range(4), HARNESSES, fontsize=7.5)
    ax.set_ylabel("|actual − rule-implied|")
    ax.set_title("Rule-compliance gap (dark→light: toward, against, placebo)",
                 loc="left", fontsize=8.3)
    fig.tight_layout()
    save(fig, "fig_probe")


# ================================================================= exemplar
def fig_exemplar():
    rep = jload(R1 / "e2_programs.json")
    bd = jload(R1 / "beliefdyn.json")
    tj = rep["bayesian/hockey_usa"]["trajectory"]["s0"]
    dts = sorted(tj)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    trio = (("lr_recent_game_results", "#2a78d6"),
            ("lr_opponent_canada_strength", "#c98500"),
            ("prior_odds_usa_gold", "#1baf7a"))
    for name, col in trio:
        ax.plot(range(len(dts)), [tj[d].get(name, 0) for d in dts], lw=1.8,
                marker="o", ms=3, color=col)
        ax.annotate(name.replace("lr_", "").replace("_", " ")[:18],
                    (len(dts) - 1, tj[dts[-1]].get(name, 0)), xytext=(4, 0),
                    textcoords="offset points", fontsize=6.5, color=col,
                    fontweight="bold", annotation_clip=False)
    ax.axhline(1.0, color=MUT, lw=0.7, ls=(0, (4, 2)))
    ax.set_xticks(range(len(dts)), [d[5:] for d in dts])
    ax.margins(x=0.16)
    ax.set_title("bayesian/USA: belief in its discovered coordinates (s0)",
                 loc="left", fontsize=8.3)
    ax.set_ylabel("value")
    ax = axes[1]
    ax.plot(range(len(dts)), [tj[d]["_target"] for d in dts], color="#eb6834",
            lw=2.2, marker="o", ms=3.5)
    th = bd["temporal_holdout"].get("bayesian/hockey_usa", {})
    ax.set_xticks(range(len(dts)), [d[5:] for d in dts])
    ax.set_ylim(0, 0.6)
    ax.set_title("…and the forecast those coordinates produce (early-theory program\n"
                 "predicts the held-out late days at MAE %.3f)" % th.get("test_mae", float("nan")),
                 loc="left", fontsize=8.3)
    ax.set_ylabel("P(USA gold)")
    fig.tight_layout()
    save(fig, "fig_exemplar")


# ============================================================ E1R: real tier
def fig_flow_real():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True, sharey=True)
    for i, h in enumerate(REAL):
        for j, q in enumerate(QUESTIONS):
            ax = axes[i][j]
            flow_panel(ax, seq2, par2, h, q, C[h],
                       "%s — %s" % (RLAB[h], QLAB[q]))
            if i == 1:
                ax.set_xticks(X, DL)
            if j == 0:
                ax.set_ylabel("P(event)")
            if i == 0 and j == 0:
                ax.legend(loc="upper left", ncols=1)
    fig.tight_layout()
    save(fig, "fig_flow_real")


def fig_gain_real():
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4), sharex=True, sharey=True)
    for ax, h in zip(axes, REAL):
        _gain_panel(ax, M2, h, RLAB[h], C[h])
    axes[0].set_ylabel(r"$\Delta$ sequential")
    fig.tight_layout()
    save(fig, "fig_gain_real")


def fig_stab_pit_real():
    _stab_pit(M2, REAL, C, RLAB, "fig_stab_pit_real", " (real code)")


def fig_coh_brier_real():
    _coh_brier(M2, REAL, C, RLAB, "fig_coh_brier_real")


def fig_shape_vs_real():
    """The headline of E1R: same metric, prompt shape vs real orchestration."""
    pairs = [("analytica", "analytica_full", "Analytica lineage"),
             ("bayesian", "blf_full", "BLF lineage")]

    def tot_move(store, h):
        vals = []
        for q in QUESTIONS:
            for r in rolls_of(store, h, q):
                pts = [v for v in r if v is not None]
                if len(pts) > 1:
                    vals.append(sum(abs(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)))
        return mean(vals) if vals else 0.0

    metrics = [
        ("update gain\n(1 = tracks evidence)",
         lambda h, real: (M2 if real else M)["per"][h]["gain_slope"], True),
        ("total movement  $|\\Delta|_{tot}$",
         lambda h, real: tot_move(seq2 if real else seqrows, h), False),
        ("coherence at last date\n(1 = additive)",
         lambda h, real: (M2 if real else M)["per"][h]["coh_seq"][DATES[-1]], False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for ax, (title, fn, zeroline) in zip(axes, metrics):
        labels, vals, cols, alphas = [], [], [], []
        for shape, real, _lin in pairs:
            labels.append(shape)
            vals.append(fn(shape, False))
            cols.append(C[shape])
            alphas.append(0.5)
            labels.append(RLAB[real])
            vals.append(fn(real, True))
            cols.append(C[real])
            alphas.append(1.0)
        ys = list(range(len(vals)))[::-1]
        for y, v, col, a in zip(ys, vals, cols, alphas):
            ax.barh(y, v, height=0.62, color=col, alpha=a)
            ax.text(v + (0.012 if v >= 0 else -0.012), y, "%.2f" % v,
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=7.2, color=INK2)
        if zeroline:
            ax.axvline(0, color="#c3c2b7", lw=0.9)
            ax.axvline(1.0, color=MUT, lw=0.8, ls=(0, (4, 2)))
        ax.set_yticks(ys, labels, fontsize=7)
        ax.set_title(title, loc="left", fontsize=8.2)
        ax.margins(x=0.32)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_shape_vs_real")


# ============================================================ schematics
def _box(ax, x, y, w, h, title, subs, edge, fill, tcol=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=1.3, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=7.6,
            fontweight="bold", color=tcol or edge, zorder=3)
    for i, s in enumerate(subs):
        ax.text(x + w / 2, y + h - 0.075 - i * 0.036, s, ha="center", va="top",
                fontsize=6.4, color=INK2, zorder=3)


def _arrow(ax, x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.0,
                                shrinkA=0, shrinkB=0), zorder=1)


def _schematic(rows, titles, fname, accent):
    """rows: list of (left_spec, right_spec); spec = (title, [sublines], changed)"""
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.0, 0.86 * n + 0.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top = 0.90
    gap = 0.035
    bh = (top - 0.04 - gap * (n - 1)) / n
    ax.text(0.02, 0.965, titles[0], fontsize=8.4, fontweight="bold", color=INK)
    ax.text(0.53, 0.965, titles[1], fontsize=8.4, fontweight="bold", color=accent)
    ax.plot([0.505, 0.505], [0.02, 0.935], color=GRID, lw=1)
    for i, (L, R) in enumerate(rows):
        y = top - bh - i * (bh + gap)
        for (spec, x0, w, changed_default) in ((L, 0.02, 0.455, False), (R, 0.525, 0.455, True)):
            title, subs, changed = spec
            edge = accent if changed else "#b9b8ae"
            fill = "#eef4fc" if changed else PANEL
            _box(ax, x0, y, w, bh, title, subs, edge, fill,
                 tcol=accent if changed else INK)
        if i < n - 1:
            _arrow(ax, 0.2475, y - 0.004, y - gap + 0.004)
            _arrow(ax, 0.7525, y - 0.004, y - gap + 0.004)
    fig.tight_layout()
    save(fig, fname)


def fig_arch_analytica():
    acc = C["analytica"]
    rows = [
        (("root proposition  ρ₀", ["one query, from scratch"], False),
         ("proposition tree carried from yesterday", ["state, not a fresh root"], True)),
        (("Analyzer  A_A", ["expand tree until L_max leaves"], False),
         ("Analyzer — EDIT pass", ["add / remove leaves for the new date",
                                   "prompt: “most days need no change”"], True)),
        (("Grounder agents  A_G  (parallel)", ["every leaf: search → soft truth value",
                                               "+ written report"], False),
         ("Grounders — delta-grounded", ["search window = since THIS leaf last looked",
                                         "shown its own previous p and report"], True)),
        (("Synthesizer  A_S", ["emits β₀ and β_j as JSON"], False),
         ("Synthesizer — weights REUSED", ["re-elicited only where the child set changed"], True)),
        (("p = β₀ + Σ β_j · p_j", ["arithmetic in code, not in the model"], False),
         ("p = β₀ + Σ β_j · p_j", ["unchanged — the paper's own resynthesis"], False)),
    ]
    _schematic(rows, ("Analytica as published — one pass",
                      "As run here — one pass per simulated date"),
               "fig_arch_analytica", acc)


def fig_arch_blf():
    acc = C["bayesian"]
    rows = [
        (("b₀ :  p = 0.5", ["fresh belief every question"], False),
         ("b₀ = yesterday's belief JSON", ["sequential; p = 0.5 in the parallel condition"], True)),
        (("loop  t = 1 … T_max = 10", ["(a_t , b_t) ← LLM(m)  — one generation",
                                       "search → LeakFilter → append,  or submit"], False),
         ("loop  t = 1 … T_max = 10  — unchanged", ["LeakFilter replaced by the SQL date gate",
                                                    "(a stronger guarantee, not a weaker one)"], True)),
        (("K = 5 trials → shrunken logit mean", ["p̂ = σ(α · mean logit p_k),  clamp [0.05, 0.95]"], False),
         ("K = 3 trials → same aggregator", ["α = 1/(1 + var(logits)) — our instantiation"], True)),
        (("hierarchical Platt calibration", ["per-source intercept offsets"], False),
         ("calibration omitted", ["no resolved training questions in the window"], True)),
    ]
    _schematic(rows, ("BLF as published — one forecast date",
                      "As run here — belief carried across dates"),
               "fig_arch_blf", acc)


def fig_pipeline():
    """E2: turn -> variables -> schema -> program -> blind holdouts."""
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("forecast turn", ["reasoning text", "that day's searches"],
         "#b9b8ae", PANEL, INK),
        ("step 1 — variables", ["name = value for", "every quantity used",
                                "answer-copies", "dropped ±0.005"],
         C["analytica"], "#eef4fc", C["analytica"]),
        ("step 2 — schema", ["reconcile days into", "one canonical schema",
                             "missing keys excluded,", "never zero-filled"],
         C["analytica"], "#eef4fc", C["analytica"]),
        ("step 3 — synthesis", ["ONE literal-free", "program per group",
                                "≤4 validator and", "residual repairs"],
         C["analytica"], "#eef4fc", C["analytica"]),
        ("blind holdouts", ["sample 0 → sample 1", "first 3 dates → last 2",
                            "targets never seen"],
         C["futuresim"], "#e9f7f1", C["futuresim"]),
    ]
    w = 0.183
    gap = (1.0 - 5 * w) / 4
    y0, bh = 0.17, 0.73
    for i, (t, subs, edge, fill, tcol) in enumerate(stages):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, bh,
                                    boxstyle="round,pad=0.004,rounding_size=0.02",
                                    linewidth=1.3, edgecolor=edge, facecolor=fill,
                                    zorder=2))
        ax.text(x + w / 2, y0 + bh - 0.075, t, ha="center", va="top", fontsize=7.4,
                fontweight="bold", color=tcol, zorder=3)
        for k, s in enumerate(subs):
            ax.text(x + w / 2, y0 + bh - 0.26 - k * 0.115, s, ha="center", va="top",
                    fontsize=6.1, color=INK2, zorder=3)
        if i < 4:
            ax.annotate("", xy=(x + w + gap - 0.002, y0 + bh / 2),
                        xytext=(x + w + 0.002, y0 + bh / 2),
                        arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.0))
    ax.text(0.0, 0.045, "The literal ban forces the split: a number is either a named input "
                        "or a computation. The blind holdouts, not the fit, are the gate.",
            fontsize=6.9, color=INK2)
    fig.tight_layout()
    save(fig, "fig_pipeline")


def fig_timeline():
    fig, ax = plt.subplots(figsize=(7.0, 1.25))
    ax.set_xlim(-0.4, 5.9)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0, 4], [0.52, 0.52], color="#b9b8ae", lw=1.2, zorder=1)
    ax.plot([4, 5.1], [0.52, 0.52], color=MUT, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for i, d in enumerate(DL):
        last = (i == len(DL) - 1)
        ax.scatter([i], [0.52], s=64 if last else 40,
                   color=C["futuresim"] if last else C["analytica"], zorder=3)
        ax.text(i, 0.33, d, ha="center", fontsize=7.5, color=INK2)
    ax.scatter([5.1], [0.52], s=44, facecolor="white", edgecolor=MUT, lw=1.3, zorder=3)
    ax.text(5.1, 0.33, "02-22", ha="center", fontsize=7.5, color=MUT)
    ax.text(0, 0.75, "five forecast dates per rollout", fontsize=7.2, color=INK2, ha="left")
    ax.text(4, 0.75, "semifinals: both teams win", fontsize=7.2,
            color=C["futuresim"], ha="center", fontweight="bold")
    ax.text(5.1, 0.75, "final — outside the window", fontsize=7.2, color=MUT, ha="right")
    ax.text(0, 0.10, "Because the gold medal is decided after the last date any harness sees, "
                     "a correct forecaster raises BOTH paired beliefs toward ~0.5 on 02-20; "
                     "coherence, not separation, is the legitimate metric.",
            fontsize=6.8, color=INK2, ha="left")
    fig.tight_layout()
    save(fig, "fig_timeline")


# ---------------------------------------------------------------- driver
FIGURES = [
    ("fig_timeline", fig_timeline),
    ("fig_arch_analytica", fig_arch_analytica),
    ("fig_arch_blf", fig_arch_blf),
    ("fig_pipeline", fig_pipeline),
    ("fig_flow", fig_flow),
    ("fig_gain", fig_gain),
    ("fig_anchor", fig_anchor),
    ("fig_stab_pit", fig_stab_pit),
    ("fig_coh_brier", fig_coh_brier),
    ("fig_search", fig_search),
    ("fig_recovery", fig_recovery),
    ("fig_probe", fig_probe),
    ("fig_exemplar", fig_exemplar),
]
if HAVE_REAL:
    FIGURES += [
        ("fig_flow_real", fig_flow_real),
        ("fig_gain_real", fig_gain_real),
        ("fig_stab_pit_real", fig_stab_pit_real),
        ("fig_coh_brier_real", fig_coh_brier_real),
        ("fig_shape_vs_real", fig_shape_vs_real),
    ]


def placeholder(name, err):
    fig, ax = plt.subplots(figsize=(7.0, 1.6))
    ax.axis("off")
    ax.text(0.5, 0.72, "FIGURE FAILED TO BUILD: %s" % name, ha="center",
            fontsize=10, fontweight="bold", color="#c0392b")
    ax.text(0.5, 0.36, err[-360:], ha="center", fontsize=6, color=INK2,
            family="DejaVu Sans Mono", wrap=True)
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
    print("\n%d figures written, %d failed%s"
          % (ok, len(bad), (": " + ", ".join(bad)) if bad else ""))
