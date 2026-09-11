"""E1 question set v2: three matched groups, six binary questions.

Each group is one free-form question from the FutureSim benchmark split
(OpenForesight `aljazeera2026Q1`, 330 questions) converted into TWO binary
questions -- one for each finalist of a two-sided contest.  That conversion is
what makes the belief scalar, which every metric in this project needs; the
pairing is what makes coherence measurable, because the two beliefs must sum to
<= 1 and, once the field has narrowed to those two, to ~1.

The runner-up in every group is taken from the source article, not invented:

  hockey   "Which country will win the men's ice hockey gold medal at the
            Milano Cortina Winter Olympics?"            -> United States
           Both finalists reached the gold-medal game on 2026-02-20.
  ausopen  "Who will win the 2026 Australian Open men's singles title?"
            -> Carlos Alcaraz; article: beat Novak Djokovic 2-6 6-2 6-3 7-5.
  superbowl "Which NFL team will win Super Bowl LX?"
            -> Seattle Seahawks; article: beat New England 29-13.

Three resolve YES and three NO.  The previous scale-up set was all-YES because
the candidate was always chosen to be the true answer; choosing both finalists
removes that bias by construction rather than by disclosure.

DATES.  Six per question, as offsets from the resolution date R:
    D1..D5 = R-39, R-25, R-14, R-7, R-2   the forecasting trajectory
    D6     = R+2                          the OUTCOME PROBE
D6 is not part of the trajectory.  By then the corpus contains the result, so
it measures whether a harness moves to the truth when the answer is in the
news -- a per-harness ceiling check.  Update gain, anchor gap, placement and
stabilisation are computed on D1..D5 only; mixing a post-outcome date into a
belief-dynamics slope would corrupt it.

The hockey offsets reproduce the v1 dates exactly, so v1 results remain
comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

TRAJECTORY_OFFSETS = (-39, -25, -14, -7, -2)
PROBE_OFFSET = 2
CORPUS_START, CORPUS_END = "2025-12-01", "2026-03-31"


def grid(resolution: str) -> list:
    """The six dates for a question resolving on `resolution`."""
    r = date.fromisoformat(resolution)
    out = [(r + timedelta(days=d)).isoformat() for d in TRAJECTORY_OFFSETS]
    out.append((r + timedelta(days=PROBE_OFFSET)).isoformat())
    assert out[0] >= CORPUS_START and out[-1] <= CORPUS_END, \
        "grid for %s falls outside the corpus window" % resolution
    return out


@dataclass
class Q:
    id: str
    group: str
    text: str
    background: str = ""
    resolution_date: str = ""
    truth: int = 1              # 1 = it happened, 0 = it did not
    dates: list = field(default_factory=list)
    source_qid: str = ""        # row in aljazeera2026Q1 this was binarised from


HOCKEY_BG = ("The Milano Cortina 2026 Winter Olympics men's ice hockey tournament "
             "concludes with the gold medal game on 22 February 2026. NHL players "
             "are participating.")
AO_BG = ("The 2026 Australian Open men's singles final is played at Rod Laver "
         "Arena, Melbourne, on 1 February 2026.")
SB_BG = ("Super Bowl LX is played at Levi's Stadium, Santa Clara, California, on "
         "8 February 2026.")

QUESTIONS = [
    # ---- group 1: Olympic men's ice hockey gold (R = 2026-02-22)
    Q(id="hockey_usa", group="hockey",
      text=("Will the United States win the men's ice hockey gold medal at the "
            "Milano Cortina 2026 Winter Olympics?"),
      background=HOCKEY_BG, resolution_date="2026-02-22", truth=1,
      dates=grid("2026-02-22"), source_qid="hockey"),
    Q(id="hockey_can", group="hockey",
      text=("Will Canada win the men's ice hockey gold medal at the Milano "
            "Cortina 2026 Winter Olympics?"),
      background=HOCKEY_BG, resolution_date="2026-02-22", truth=0,
      dates=grid("2026-02-22"), source_qid="hockey"),

    # ---- group 2: Australian Open men's singles title (R = 2026-01-31)
    Q(id="ausopen_alcaraz", group="ausopen",
      text="Will Carlos Alcaraz win the 2026 Australian Open men's singles title?",
      background=AO_BG, resolution_date="2026-01-31", truth=1,
      dates=grid("2026-01-31"), source_qid="ausopen"),
    Q(id="ausopen_djokovic", group="ausopen",
      text="Will Novak Djokovic win the 2026 Australian Open men's singles title?",
      background=AO_BG, resolution_date="2026-01-31", truth=0,
      dates=grid("2026-01-31"), source_qid="ausopen"),

    # ---- group 3: Super Bowl LX (R = 2026-02-07)
    Q(id="superbowl_sea", group="superbowl",
      text="Will the Seattle Seahawks win Super Bowl LX?",
      background=SB_BG, resolution_date="2026-02-07", truth=1,
      dates=grid("2026-02-07"), source_qid="superbowl"),
    Q(id="superbowl_ne", group="superbowl",
      text="Will the New England Patriots win Super Bowl LX?",
      background=SB_BG, resolution_date="2026-02-07", truth=0,
      dates=grid("2026-02-07"), source_qid="superbowl"),
]

GROUPS = {"hockey": ("hockey_usa", "hockey_can"),
          "ausopen": ("ausopen_alcaraz", "ausopen_djokovic"),
          "superbowl": ("superbowl_sea", "superbowl_ne")}

TRAJECTORY = {q.id: q.dates[:len(TRAJECTORY_OFFSETS)] for q in QUESTIONS}
PROBE_DATE = {q.id: q.dates[-1] for q in QUESTIONS}

if __name__ == "__main__":
    print("%-18s %-10s %-6s %-12s  dates" % ("id", "group", "truth", "resolves"))
    for q in QUESTIONS:
        print("%-18s %-10s %-6s %-12s  %s"
              % (q.id, q.group, "YES" if q.truth else "NO",
                 q.resolution_date, " ".join(d[5:] for d in q.dates)))
    print("\nYES: %d   NO: %d" % (sum(q.truth for q in QUESTIONS),
                                  sum(1 - q.truth for q in QUESTIONS)))
