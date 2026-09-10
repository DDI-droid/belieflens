"""Sequence-vs-Parallel metrics and charts (single self-contained HTML).

Reads  results/exp1/e1_runs.jsonl (+ e1_runs_extra.jsonl)   sequential rollouts
       results/exp1/e1_independent.jsonl                    parallel rollouts
Writes results/exp1/seqpar_metrics.json  and  seqpar_charts.html

Figures (each with legend, hover tooltips, and a data-table view):
 F1 belief flow      per harness x question: parallel per-date distribution
                     (dots + mean tick) with sequential rollout paths overlaid
 F2 update gain      d(seq) vs d(parallel mean) scatter + fitted slope --
                     the conservatism coefficient per harness
 F3 anchoring        signed anchor gap and lag-advantage bars per harness
 F4 stabilization    sd(sequential)/sd(parallel) per date -- is history a
                     variance reducer?
 F5 placement (PIT)  percentile of each sequential value inside the parallel
                     distribution -- 50 = unbiased, edges = anchored bias
 F6 coherence        P(USA)+P(CAN) per date, per condition (target 1)
 F7 brier            per-date Brier vs truth, per condition
 F8 search effort    searches per turn, sequential vs parallel
"""
from __future__ import annotations

import html as _html
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/exp1")
HARNESSES = ["analytica", "bayesian", "futuresim", "react"]
QUESTIONS = ["hockey_usa", "hockey_can"]
TRUTH = {"hockey_usa": 1.0, "hockey_can": 0.0}
QLAB = {"hockey_usa": "USA gold", "hockey_can": "Canada gold"}
SEMI_DATE = "2026-02-20"

# ------------------------------------------------------------------ load

def load():
    seq = defaultdict(dict)          # (h,q,s) -> {date: p}
    seq_tools = defaultdict(list)    # (h) -> [n_searches...]
    for name in ("e1_runs.jsonl", "e1_runs_extra.jsonl"):
        p = OUT / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["question_id"] not in QUESTIONS:
                continue
            for t in r["turns"]:
                if t["forecast"] is not None:
                    seq[(r["harness"], r["question_id"], r["sample"])][t["date"]] = t["forecast"]
                seq_tools[r["harness"]].append(len(t.get("tool_calls", [])))
    par = defaultdict(list)          # (h,q,d) -> [p...]
    par_tools = defaultdict(list)
    for line in (OUT / "e1_independent.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["forecast"] is not None:
            par[(r["harness"], r["question_id"], r["date"])].append(r["forecast"])
        par_tools[r["harness"]].append(len(r.get("tool_calls", [])))
    dates = sorted({d for (_, _, d) in par})
    return seq, par, dates, seq_tools, par_tools


seq, par, DATES, seq_tools, par_tools = load()
K_SEQ = len({s for (_, _, s) in seq})
K_PAR = max((len(v) for v in par.values()), default=0)

# ------------------------------------------------------------------ metrics

def pmean(h, q, d):
    v = par.get((h, q, d), [])
    return mean(v) if v else None


def psd(h, q, d):
    v = par.get((h, q, d), [])
    return pstdev(v) if len(v) > 1 else None


M = {"k_seq": K_SEQ, "k_par": K_PAR, "dates": DATES, "per": {}}
for h in HARNESSES:
    e = {"gap": [], "lag0": [], "lag1": [], "gain_pts": [], "pit": [],
         "stab": {}, "coh_seq": {}, "coh_par": {}, "brier_seq": {}, "brier_par": {},
         "search_seq": mean(seq_tools[h]) if seq_tools[h] else 0,
         "search_par": mean(par_tools[h]) if par_tools[h] else 0}
    for q in QUESTIONS:
        for smp in sorted({s for (hh, qq, s) in seq if (hh, qq) == (h, q)}):
            tr = seq[(h, q, smp)]
            for i, d in enumerate(DATES):
                if d not in tr:
                    continue
                pm = pmean(h, q, d)
                draws = par.get((h, q, d), [])
                if draws:
                    below = sum(1 for x in draws if x < tr[d])
                    ties = sum(1 for x in draws if x == tr[d])
                    e["pit"].append(100.0 * (below + 0.5 * ties) / len(draws))
                if i >= 1 and pm is not None:
                    e["gap"].append(tr[d] - pm)
                    e["lag0"].append(abs(tr[d] - pm))
                    pm_prev = pmean(h, q, DATES[i - 1])
                    if pm_prev is not None:
                        e["lag1"].append(abs(tr[d] - pm_prev))
                    if DATES[i - 1] in tr:
                        dpar = pm - pm_prev
                        dseq = tr[d] - tr[DATES[i - 1]]
                        e["gain_pts"].append((dpar, dseq))
    for i, d in enumerate(DATES):
        sq_vals = {qq: [seq[(h, qq, s)][d] for s in range(K_SEQ)
                        if (h, qq, s) in seq and d in seq[(h, qq, s)]]
                   for qq in QUESTIONS}
        sds_s = [pstdev(v) for v in sq_vals.values() if len(v) > 1]
        sds_p = [psd(h, qq, d) for qq in QUESTIONS]
        sds_p = [x for x in sds_p if x]
        if sds_s and sds_p and mean(sds_p) > 1e-9:
            e["stab"][d] = mean(sds_s) / mean(sds_p)
        both_s = [sq_vals["hockey_usa"][j] + sq_vals["hockey_can"][j]
                  for j in range(min(len(sq_vals["hockey_usa"]), len(sq_vals["hockey_can"])))]
        if both_s:
            e["coh_seq"][d] = mean(both_s)
        pu, pc = pmean(h, "hockey_usa", d), pmean(h, "hockey_can", d)
        if pu is not None and pc is not None:
            e["coh_par"][d] = pu + pc
        bs = [(v - TRUTH[qq]) ** 2 for qq in QUESTIONS for v in sq_vals[qq]]
        bp = [(v - TRUTH[qq]) ** 2 for qq in QUESTIONS for v in par.get((h, qq, d), [])]
        if bs:
            e["brier_seq"][d] = mean(bs)
        if bp:
            e["brier_par"][d] = mean(bp)
    xs = [p[0] for p in e["gain_pts"]]
    ys = [p[1] for p in e["gain_pts"]]
    if len(xs) >= 3 and pstdev(xs) > 1e-9:
        mx, my = mean(xs), mean(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        e["gain_slope"] = b
    else:
        e["gain_slope"] = None
    M["per"][h] = e

(OUT / "seqpar_metrics.json").write_text(json.dumps(M, indent=1), encoding="utf-8")
print("k_seq=%d rollouts, k_par=%d per date" % (K_SEQ, K_PAR))
for h in HARNESSES:
    e = M["per"][h]
    print("  %-10s gap %+0.3f  |gap| %.3f  lagadv %+0.3f  gain %s  pit-med %.0f"
          % (h, mean(e["gap"]), mean(e["lag0"]), mean(e["lag0"]) - mean(e["lag1"]),
             "--" if e["gain_slope"] is None else "%.2f" % e["gain_slope"],
             sorted(e["pit"])[len(e["pit"]) // 2] if e["pit"] else float("nan")))

# ------------------------------------------------------------------ svg kit

CV = {"analytica": "var(--c1)", "bayesian": "var(--c2)",
      "futuresim": "var(--c3)", "react": "var(--c4)"}
W, HP, ML, MR, MT, MB = 470, 260, 44, 14, 22, 34


def esc(s):
    return _html.escape(str(s), quote=True)


def sx(i, n=None):
    n = n or len(DATES)
    return ML + (W - ML - MR) * (i / max(1, n - 1))


def sy(v, lo=0.0, hi=1.0):
    return MT + (HP - MT - MB) * (1 - (v - lo) / (hi - lo))


def axes(lo=0.0, hi=1.0, ylab="", xlabels=None, n=None):
    xlabels = xlabels if xlabels is not None else [d[5:] for d in DATES]
    n = n or len(xlabels)
    p = []
    ticks = [lo, (lo + hi) / 2, hi]
    for tv in ticks:
        y = sy(tv, lo, hi)
        p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>' % (ML, y, W - MR, y))
        p.append('<text x="%d" y="%.1f" class="tick" text-anchor="end">%.2g</text>' % (ML - 6, y + 3, tv))
    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="axis"/>'
             % (ML, sy(lo, lo, hi), W - MR, sy(lo, lo, hi)))
    for i, xl in enumerate(xlabels):
        p.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%s</text>'
                 % (sx(i, n), HP - MB + 16, esc(xl)))
    if ylab:
        p.append('<text x="%d" y="%d" class="tick">%s</text>' % (ML, MT - 8, esc(ylab)))
    return "".join(p)


def panel(inner, title, w=W, h=HP):
    return ('<div class="panel"><div class="ptitle">%s</div>'
            '<svg viewBox="0 0 %d %d" role="img" aria-label="%s">%s</svg></div>'
            % (esc(title), w, h, esc(title), inner))


def figure(fid, title, claim, panels, legend, table_rows, table_head):
    tbl = "<tr>" + "".join("<th>%s</th>" % esc(c) for c in table_head) + "</tr>"
    for row in table_rows:
        tbl += "<tr>" + "".join("<td>%s</td>" % esc(c) for c in row) + "</tr>"
    return ('<section id="%s"><h2>%s</h2><p class="claim">%s</p>'
            '<div class="legend">%s</div><div class="grid2">%s</div>'
            '<details><summary>data table</summary>'
            '<div class="twrap"><table>%s</table></div></details></section>'
            % (fid, esc(title), claim, legend, "".join(panels), tbl))


def leg(items):
    out = []
    for label, css, kind in items:
        sw = ('<svg width="18" height="10"><line x1="0" y1="5" x2="18" y2="5" '
              'style="stroke:%s;stroke-width:2.5"/></svg>' % css) if kind == "line" else \
             ('<svg width="18" height="10"><circle cx="9" cy="5" r="4" '
              'style="fill:none;stroke:%s;stroke-width:1.6"/></svg>' % css)
        out.append('<span class="li">%s %s</span>' % (sw, esc(label)))
    return "".join(out)


figs = []

# ---------------------------------------------------------------- F1 flow
panels = []
for h in HARNESSES:
    for q in QUESTIONS:
        p = [axes(ylab="P(%s)" % QLAB[q])]
        si = DATES.index(SEMI_DATE)
        p.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%.1f" class="vline"/>'
                 '<text x="%.1f" y="%d" class="tick" text-anchor="middle">semis</text>'
                 % (sx(si), MT, sx(si), sy(0), sx(si), MT - 8))
        ty = sy(TRUTH[q])
        p.append('<text x="%d" y="%.1f" class="truth">resolves %s &#9656;</text>'
                 % (W - MR - 74, ty + (12 if TRUTH[q] > 0.5 else -6), "YES" if TRUTH[q] else "NO"))
        for i, d in enumerate(DATES):
            draws = par.get((h, q, d), [])
            for j, v in enumerate(sorted(draws)):
                jx = sx(i) + (j - (len(draws) - 1) / 2) * 5.0
                p.append('<circle cx="%.1f" cy="%.1f" r="4" class="pardot" '
                         'data-tip="parallel %s r%d: %.2f"/>' % (jx, sy(v), d[5:], j, v))
            if draws:
                pm = mean(draws)
                p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="parmean" '
                         'data-tip="parallel mean %s: %.2f"/>'
                         % (sx(i) - 13, sy(pm), sx(i) + 13, sy(pm), d[5:], pm))
        for smp in range(K_SEQ):
            tr = seq.get((h, q, smp), {})
            pts = [(sx(i), sy(tr[d])) for i, d in enumerate(DATES) if d in tr]
            if len(pts) >= 2:
                p.append('<polyline points="%s" class="seqline" style="stroke:%s" '
                         'data-tip="sequential rollout %d"/>'
                         % (" ".join("%.1f,%.1f" % pt for pt in pts), CV[h], smp))
                p.append('<circle cx="%.1f" cy="%.1f" r="3.4" style="fill:%s"/>'
                         % (pts[-1][0], pts[-1][1], CV[h]))
        panels.append(panel("".join(p), "%s · %s" % (h, QLAB[q])))
rows = [[h, QLAB[q], d[5:],
         "%.2f" % pmean(h, q, d) if pmean(h, q, d) is not None else "--",
         "%.2f" % psd(h, q, d) if psd(h, q, d) is not None else "--",
         " / ".join("%.2f" % seq[(h, q, s)][d] for s in range(K_SEQ)
                    if (h, q, s) in seq and d in seq[(h, q, s)])]
        for h in HARNESSES for q in QUESTIONS for d in DATES]
figs.append(figure("flow", "F1 · Belief flow — sequence over the parallel distribution",
    "Colored paths are the %d sequential rollouts (history carried). Open dots are the %d parallel "
    "one-shot draws per date (fresh eyes, same evidence), with their mean as a tick. "
    "<b>Read:</b> paths riding above/below the dots = anchoring; paths matching yesterday's dots = lag; "
    "dot spread = how determinate the evidence-conditioned belief is." % (K_SEQ, K_PAR),
    panels, leg([("sequential rollout (harness hue)", "var(--ink2)", "line"),
                 ("parallel draw", "var(--muted)", "dot"),
                 ("parallel mean", "var(--ink2)", "line")]),
    rows, ["harness", "question", "date", "par mean", "par sd", "sequential rollouts"]))

# ---------------------------------------------------------------- F2 gain
panels = []
for h in HARNESSES:
    pts = M["per"][h]["gain_pts"]
    lo, hi = -0.45, 0.45
    p = []
    for tv in (-0.4, -0.2, 0.0, 0.2, 0.4):
        x = ML + (W - ML - MR) * ((tv - lo) / (hi - lo))
        y = MT + (HP - MT - MB) * (1 - (tv - lo) / (hi - lo))
        p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="grid"/>' % (ML, y, W - MR, y))
        p.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="grid"/>' % (x, MT, x, HP - MB))
        p.append('<text x="%d" y="%.1f" class="tick" text-anchor="end">%+.1f</text>' % (ML - 6, y + 3, tv))
        p.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%+.1f</text>' % (x, HP - MB + 16, tv))
    def gx(v): return ML + (W - ML - MR) * ((v - lo) / (hi - lo))
    def gy(v): return MT + (HP - MT - MB) * (1 - (v - lo) / (hi - lo))
    p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="vline"/>' % (gx(lo), gy(lo), gx(hi), gy(hi)))
    p.append('<text x="%.1f" y="%.1f" class="tick">y = x (no under-reaction)</text>' % (gx(0.16), gy(0.22)))
    for dx, dy in pts:
        p.append('<circle cx="%.1f" cy="%.1f" r="4" style="fill:%s;fill-opacity:.75" '
                 'data-tip="Δpar %+0.2f → Δseq %+0.2f"/>' % (gx(dx), gy(dy), CV[h], dx, dy))
    b = M["per"][h]["gain_slope"]
    if b is not None:
        p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" style="stroke:%s;stroke-width:2.5"/>'
                 % (gx(-0.4), gy(-0.4 * b), gx(0.4), gy(0.4 * b), CV[h]))
        p.append('<text x="%d" y="%d" class="slopelab" style="fill:%s">slope %.2f</text>'
                 % (ML + 4, MT + 12, CV[h], b))
    panels.append(panel("".join(p), "%s · update gain" % h))
figs.append(figure("gain", "F2 · Update gain — does the sequence move as much as fresh eyes do?",
    "Each dot: one date-step. x = how much the parallel (fresh-eyes) mean moved; y = how much that "
    "sequential rollout moved. <b>Read:</b> slope 1 = updates exactly as the evidence shifts; "
    "slope &lt; 1 = conservatism (under-reaction to news while anchored to history); scatter off the "
    "line = idiosyncratic path noise.",
    panels, leg([("date-step", "var(--ink2)", "dot"), ("fitted slope", "var(--ink2)", "line")]),
    [[h, "%.2f" % M["per"][h]["gain_slope"] if M["per"][h]["gain_slope"] is not None else "--",
      len(M["per"][h]["gain_pts"])] for h in HARNESSES],
    ["harness", "gain slope", "n steps"]))

# ---------------------------------------------------------------- F3 bars
def barchart(vals, title, fmt="%+.3f", lo=None, hi=None, tipfmt=None):
    lo = min(0, min(vals.values())) * 1.3 if lo is None else lo
    hi = max(0, max(vals.values())) * 1.3 + 1e-9 if hi is None else hi
    p = []
    y0 = MT + (HP - MT - MB) * (1 - (0 - lo) / (hi - lo))
    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="axis"/>' % (ML, y0, W - MR, y0))
    bw = 44
    for i, h in enumerate(HARNESSES):
        v = vals[h]
        x = ML + 30 + i * ((W - ML - MR - 60) / (len(HARNESSES) - 1)) - bw / 2
        yv = MT + (HP - MT - MB) * (1 - (v - lo) / (hi - lo))
        top, hgt = (yv, y0 - yv) if v >= 0 else (y0, yv - y0)
        p.append('<rect x="%.1f" y="%.1f" width="%d" height="%.1f" rx="4" style="fill:%s" '
                 'data-tip="%s: %s"/>' % (x, top, bw, max(1.5, hgt), CV[h], h,
                                          (tipfmt or fmt) % v))
        p.append('<text x="%.1f" y="%.1f" class="vlab" text-anchor="middle">%s</text>'
                 % (x + bw / 2, (top - 5) if v >= 0 else (top + hgt + 13), fmt % v))
        p.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%s</text>'
                 % (x + bw / 2, HP - MB + 16, h))
    return panel("".join(p), title)

gapv = {h: mean(M["per"][h]["gap"]) for h in HARNESSES}
lagv = {h: mean(M["per"][h]["lag0"]) - mean(M["per"][h]["lag1"]) for h in HARNESSES}
figs.append(figure("anchor", "F3 · Anchoring summary",
    "<b>Left:</b> signed anchor gap = sequential &minus; parallel mean (dates 2+). Positive = "
    "carrying history holds the belief above what today's evidence alone supports. "
    "<b>Right:</b> lag advantage = |gap to today's fresh belief| &minus; |gap to yesterday's|. "
    "Positive = the sequence tracks <i>yesterday's</i> evidence better than today's — it "
    "updates one date behind.",
    [barchart(gapv, "signed anchor gap"), barchart(lagv, "lag advantage (positive = lags)")],
    leg([(h, CV[h], "line") for h in HARNESSES]),
    [[h, "%+.3f" % gapv[h], "%.3f" % mean(M["per"][h]["lag0"]), "%+.3f" % lagv[h]] for h in HARNESSES],
    ["harness", "signed gap", "|gap|", "lag advantage"]))

# ---------------------------------------------------------------- F4 stab
p = [axes(0, 2.2, "sd(sequence) / sd(parallel)")]
yl1 = sy(1.0, 0, 2.2)
p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="vline"/>'
         '<text x="%d" y="%.1f" class="tick">1 = same spread</text>'
         % (ML, yl1, W - MR, yl1, ML + 4, yl1 - 5))
for h in HARNESSES:
    st = M["per"][h]["stab"]
    pts = [(sx(i), sy(min(2.15, st[d]), 0, 2.2)) for i, d in enumerate(DATES) if d in st]
    if len(pts) >= 2:
        p.append('<polyline points="%s" class="seqline" style="stroke:%s" data-tip="%s"/>'
                 % (" ".join("%.1f,%.1f" % q_ for q_ in pts), CV[h], h))
        p.append('<text x="%.1f" y="%.1f" class="slopelab" style="fill:%s">%s</text>'
                 % (pts[-1][0] - 30, pts[-1][1] - 6, CV[h], h[:4]))
figs.append(figure("stab", "F4 · Is history a variance reducer?",
    "Ratio of rollout spread: sequential sd over parallel sd, per date. <b>Read:</b> below 1 = "
    "carrying history <i>stabilises</i> the belief (memory as variance reduction, the "
    "FutureSim/Analytica claim); above 1 = path dependence — rollouts that diverge because "
    "their own early draws differed.",
    [panel("".join(p), "stabilisation ratio by date")],
    leg([(h, CV[h], "line") for h in HARNESSES]),
    [[h] + ["%.2f" % M["per"][h]["stab"][d] if d in M["per"][h]["stab"] else "--" for d in DATES]
     for h in HARNESSES],
    ["harness"] + [d[5:] for d in DATES]))

# ---------------------------------------------------------------- F5 pit
p = []
lo_, hi_ = 0, 100
for tv in (0, 25, 50, 75, 100):
    x = ML + (W - ML - MR) * (tv / 100)
    p.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="grid"/>' % (x, MT, x, HP - MB))
    p.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%d</text>' % (x, HP - MB + 16, tv))
x50 = ML + (W - ML - MR) * 0.5
p.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" class="vline"/>' % (x50, MT, x50, HP - MB))
rowy = lambda i: MT + 24 + i * 50
for i, h in enumerate(HARNESSES):
    y = rowy(i)
    p.append('<text x="%d" y="%.1f" class="tick" text-anchor="end">%s</text>' % (ML - 6, y + 3, h))
    vals = M["per"][h]["pit"]
    for v in vals:
        x = ML + (W - ML - MR) * (v / 100)
        p.append('<circle cx="%.1f" cy="%.1f" r="4" style="fill:%s;fill-opacity:.45" '
                 'data-tip="%s pct %.0f"/>' % (x, y, CV[h], h, v))
    med = sorted(vals)[len(vals) // 2]
    xm = ML + (W - ML - MR) * (med / 100)
    p.append('<rect x="%.1f" y="%.1f" width="3.5" height="16" style="fill:var(--ink)" '
             'data-tip="%s median pct %.0f"/>' % (xm - 1.7, y - 8, h, med))
figs.append(figure("pit", "F5 · Where the anchored belief sits inside the fresh-eyes distribution",
    "Every sequential forecast, placed at its percentile within that date's parallel draws. "
    "<b>Read:</b> mass at 50 = the sequence is indistinguishable from an evidence-only belief; "
    "mass pushed to one side = systematic anchoring bias; black tick = median.",
    [panel("".join(p), "percentile placement (0–100)", h=MT + 24 + 4 * 50 + 20)],
    leg([(h, CV[h], "dot") for h in HARNESSES] + [("median", "var(--ink)", "line")]),
    [[h, "%.0f" % (sorted(M["per"][h]["pit"])[len(M["per"][h]["pit"]) // 2]),
      "%.0f" % mean(M["per"][h]["pit"]), len(M["per"][h]["pit"])] for h in HARNESSES],
    ["harness", "median pct", "mean pct", "n"]))

# ---------------------------------------------------------------- F6 + F7
def lines_panel(field, title, lo, hi, ylab, refline=None):
    p = [axes(lo, hi, ylab)]
    if refline is not None:
        yr = sy(refline, lo, hi)
        p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="vline"/>' % (ML, yr, W - MR, yr))
    for h in HARNESSES:
        série = M["per"][h][field]
        pts = [(sx(i), sy(max(lo, min(hi, série[d])), lo, hi))
               for i, d in enumerate(DATES) if d in série]
        if len(pts) >= 2:
            p.append('<polyline points="%s" class="seqline" style="stroke:%s" data-tip="%s"/>'
                     % (" ".join("%.1f,%.1f" % q_ for q_ in pts), CV[h], h))
            p.append('<text x="%.1f" y="%.1f" class="slopelab" style="fill:%s">%s</text>'
                     % (pts[-1][0] - 28, pts[-1][1] - 6, CV[h], h[:4]))
    return panel("".join(p), title)

figs.append(figure("coh", "F6 · Complement coherence — P(USA)+P(CAN)",
    "The paired beliefs should sum toward 1 as the field narrows to two finalists (never above). "
    "<b>Read:</b> which condition keeps the pair coherent — fresh eyes, or the carried history?",
    [lines_panel("coh_seq", "sequential", 0.4, 1.3, "P(USA)+P(CAN)", 1.0),
     lines_panel("coh_par", "parallel", 0.4, 1.3, "P(USA)+P(CAN)", 1.0)],
    leg([(h, CV[h], "line") for h in HARNESSES]),
    [[h, d[5:], "%.2f" % M["per"][h]["coh_seq"].get(d, float("nan")),
      "%.2f" % M["per"][h]["coh_par"].get(d, float("nan"))] for h in HARNESSES for d in DATES],
    ["harness", "date", "seq sum", "par sum"]))

figs.append(figure("brier", "F7 · Accuracy cost — Brier score vs truth, by condition",
    "Mean squared error against the resolved outcomes (USA=1, CAN=0), pooled over the pair. Lower is "
    "better. <b>Read:</b> if the parallel line sits below the sequential one late, anchoring is "
    "costing measurable accuracy — the FutureSim sequential-vs-direct result, reproduced.",
    [lines_panel("brier_seq", "sequential", 0, 0.6, "Brier"),
     lines_panel("brier_par", "parallel", 0, 0.6, "Brier")],
    leg([(h, CV[h], "line") for h in HARNESSES]),
    [[h, d[5:], "%.3f" % M["per"][h]["brier_seq"].get(d, float("nan")),
      "%.3f" % M["per"][h]["brier_par"].get(d, float("nan"))] for h in HARNESSES for d in DATES],
    ["harness", "date", "seq Brier", "par Brier"]))

# ---------------------------------------------------------------- F8 search
p = []
mx = max(max(M["per"][h]["search_seq"], M["per"][h]["search_par"]) for h in HARNESSES) * 1.25
y0 = sy(0, 0, mx)
p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="axis"/>' % (ML, y0, W - MR, y0))
for i, h in enumerate(HARNESSES):
    xc = ML + 40 + i * ((W - ML - MR - 80) / (len(HARNESSES) - 1))
    for k, (fld, cls) in enumerate((("search_seq", ""), ("search_par", "parbar"))):
        v = M["per"][h][fld]
        yv = sy(v, 0, mx)
        x = xc - 22 + k * 24
        style = 'style="fill:%s"' % CV[h] if not cls else \
                'style="fill:none;stroke:%s;stroke-width:2" ' % CV[h]
        p.append('<rect x="%.1f" y="%.1f" width="20" height="%.1f" rx="4" %s data-tip="%s %s: %.1f"/>'
                 % (x, yv, y0 - yv, style, h, "seq" if k == 0 else "par", v))
        p.append('<text x="%.1f" y="%.1f" class="vlab" text-anchor="middle">%.1f</text>'
                 % (x + 10, yv - 5, v))
    p.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%s</text>'
             % (xc, HP - MB + 16, h))
for tv in (0, mx / 2):
    p.append('<text x="%d" y="%.1f" class="tick" text-anchor="end">%.0f</text>'
             % (ML - 6, sy(tv, 0, mx) + 3, tv))
figs.append(figure("search", "F8 · Search effort by condition",
    "Mean searches per turn. <b>Read:</b> does carrying memory make a harness search less (memory "
    "substituting for retrieval), and does the fresh-eyes condition compensate by searching more?",
    [panel("".join(p), "searches per turn — filled = sequential, outline = parallel")],
    leg([("sequential (filled)", "var(--ink2)", "line"), ("parallel (outline)", "var(--ink2)", "dot")]),
    [[h, "%.2f" % M["per"][h]["search_seq"], "%.2f" % M["per"][h]["search_par"]] for h in HARNESSES],
    ["harness", "seq searches/turn", "par searches/turn"]))

# ---------------------------------------------------------------- html

HTML = """<title>Sequence vs Parallel</title>
<style>
.viz-root{
  color-scheme:light;
  --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#eda100;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])) .viz-root{
    color-scheme:dark;
    --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --axis:#383835;
    --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500;
  }
}
:root[data-theme="dark"] .viz-root{
  color-scheme:dark;
  --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink2:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500;
}
body{margin:0}
.viz-root{background:var(--page);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:15px;line-height:1.55;
  padding:0 0 80px}
.wrap{max-width:1060px;margin:0 auto;padding:0 22px}
header{padding:44px 0 10px}
h1{font-size:clamp(1.7rem,3.5vw,2.4rem);margin:0 0 10px;letter-spacing:-.01em}
.sub{color:var(--ink2);max-width:72ch}
h2{font-size:1.15rem;margin:0}
section{margin-top:46px;display:flex;flex-direction:column;gap:10px}
.claim{color:var(--ink2);max-width:78ch;margin:0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px}
.panel{background:var(--surface);border:1px solid var(--grid);border-radius:8px;padding:10px 8px 6px}
.ptitle{font-size:.8rem;color:var(--ink2);padding:0 6px 4px;font-weight:600}
svg{display:block;width:100%;height:auto}
svg text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1.2}
.vline{stroke:var(--axis);stroke-width:1;stroke-dasharray:4 4}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.truth{fill:var(--ink2);font-size:11px}
.vlab{fill:var(--ink2);font-size:11px;font-variant-numeric:tabular-nums}
.slopelab{font-size:11px;font-weight:600}
.pardot{fill:none;stroke:var(--muted);stroke-width:1.6}
.parmean{stroke:var(--ink2);stroke-width:2.5}
.seqline{fill:none;stroke-width:2;stroke-opacity:.8}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:.82rem;align-items:center}
.li{display:inline-flex;gap:6px;align-items:center}
details{border:1px solid var(--grid);border-radius:8px;background:var(--surface)}
summary{cursor:pointer;padding:8px 12px;font-size:.82rem;color:var(--ink2)}
.twrap{overflow-x:auto;padding:0 10px 10px}
table{border-collapse:collapse;font-size:.8rem;width:100%}
th,td{text-align:left;padding:4px 10px;border-bottom:1px solid var(--grid);
  font-variant-numeric:tabular-nums;white-space:nowrap}
th{color:var(--muted);font-weight:600}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--page);
  padding:4px 8px;border-radius:5px;font-size:12px;opacity:0;transition:opacity .08s;z-index:9}
.note{color:var(--muted);font-size:.82rem;max-width:78ch}
</style>
<div class="viz-root"><div class="wrap">
<header>
<h1>Sequence vs Parallel</h1>
<p class="sub">Two ways of asking the same four forecasting harnesses the same questions on the same
evidence. <b>Sequence</b>: __KS__ rollouts that walk the five dates in order, carrying their own
forecasts and memory. <b>Parallel</b>: for each date, __KP__ fresh one-shot runs given everything up
to that date and no history at all &mdash; the evidence-conditioned belief as a distribution.
Every figure below is one lens on the difference; each has a hover layer and a data table.</p>
<p class="note">Data: hockey pair (USA resolves YES, Canada NO; both reached the final inside the
window, decided after it). Same model, same date-gated corpus, same tools in both conditions.
gpt-5-mini; per-date parallel n = __KP__; sequential rollouts k = __KS__.</p>
</header>
__FIGS__
</div><div id="tip"></div></div>
<script>
(function(){var t=document.getElementById('tip');
document.addEventListener('mousemove',function(e){
  var el=e.target.closest('[data-tip]');
  if(el){t.textContent=el.getAttribute('data-tip');t.style.opacity=1;
    t.style.left=(e.clientX+12)+'px';t.style.top=(e.clientY+12)+'px';}
  else t.style.opacity=0;});})();
</script>
"""
HTML = HTML.replace("__KS__", str(K_SEQ)).replace("__KP__", str(K_PAR))
HTML = HTML.replace("__FIGS__", "\n".join(figs))
(OUT / "seqpar_charts.html").write_text(HTML, encoding="utf-8")
print("wrote %s (%.0f KB)" % (OUT / "seqpar_charts.html",
                              (OUT / "seqpar_charts.html").stat().st_size / 1024))
