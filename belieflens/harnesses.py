"""Four harness shapes over one shared substrate.

Same model, same date-gated search tool, same sequential protocol.  The only
thing that varies is the reasoning structure -- which is the point.  Installing
four upstream repos would instead compare four repos.

  react      linear think/act loop. no memory, no structure. the floor.
  futuresim  ReAct plus the engineering the FutureSim paper describes:
             procedural forecasting guidelines, task state as a table, and a
             forced memory-update phase with per-question memory carried
             across days.
  analytica  decompose into sub-propositions, ground each by search, then
             compose with explicit weights.
  bayesian   an explicit linguistic belief state; each day states prior,
             likelihood of the new evidence, then posterior.

Every harness must end its turn with a line   FORECAST: <p>   and is asked to
write its reasoning visibly, so the trajectory is recoverable without relying
on hidden chain-of-thought.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .evidence import NewsIndex, search_tool_spec

HARNESSES = ["react", "futuresim", "analytica", "bayesian"]

_COMMON = """You are forecasting a real future event.

The current simulation date is {date}. You know nothing about events after
this date. Use search_news to gather evidence; it can only return articles
published on or before {date}.

Write your reasoning out explicitly as you go -- state the quantities you are
weighing and the numbers you assign them. Someone will later have to
reconstruct exactly how you arrived at your number.

End your final message with a line of exactly this form:
FORECAST: <probability between 0 and 1>"""

_REACT = _COMMON + """

Work in a simple loop: think about what you need, search for it, read what
comes back, think again. Stop when further searching would not change your
answer."""

_FUTURESIM = _COMMON + """

FORECASTING GUIDELINES (follow them in order):
1. Restate the resolution criterion precisely. What exactly must happen?
2. Set a base rate from reference-class outcomes before looking at news.
3. Search for evidence. Run several distinct, targeted queries, not one broad
   one. Look for disconfirming evidence specifically.
4. Weigh evidence against the base rate. State how much each item moves you
   and why.
5. Check yourself for overconfidence and for anchoring on your previous
   forecast. A forecast that never moves is as suspect as one that thrashes.
6. Give a calibrated probability.

MEMORY. Your memory from previous days is below. Before your final answer you
MUST output an updated memory block, in this exact form:

<memory>
BASE RATE: ...
KEY EVIDENCE: ...
WHAT WOULD CHANGE MY MIND: ...
LAST FORECAST AND WHY: ...
</memory>

Keep it under 200 words. It is all you will carry to the next day."""

_ANALYTICA = _COMMON + """

Work in three explicit stages, labelled.

STAGE 1 - ANALYSE. Break the question into 3 to 5 sub-propositions that are
independent of each other and each individually checkable. Number them P1, P2,
...  Do not search yet.

STAGE 2 - GROUND. For each sub-proposition in turn, search for evidence and
assign it a probability. Write   P1 = <p>   with a one-line justification.

STAGE 3 - SYNTHESISE. Assign each sub-proposition a weight showing how much it
contributes to the overall question, state the combination you are performing
with its numbers written out, and compute the result. Then give FORECAST."""

_BAYESIAN = _COMMON + """

Maintain an explicit belief state and update it in the open.

Your belief state from previous days is below. Each day, in this order:

PRIOR: state your current probability and the one-sentence summary of the
evidence it rests on.
EVIDENCE: search, then state what is new since your last update.
LIKELIHOOD: for each new item, state how much more likely it is if the event
happens than if it does not -- an explicit ratio, e.g. "about 3x more likely
under YES".
POSTERIOR: apply the update to your prior odds and state the result.

Then output an updated belief state in this exact form:

<belief>
P: <probability>
EVIDENCE SUMMARY: ...
STRONGEST COUNTER-EVIDENCE: ...
</belief>

Then give FORECAST."""

SYSTEMS = {"react": _REACT, "futuresim": _FUTURESIM,
           "analytica": _ANALYTICA, "bayesian": _BAYESIAN}


@dataclass
class Turn:
    date: str
    forecast: float | None
    text: str
    tool_calls: list = field(default_factory=list)
    carry: str = ""
    error: str = ""
    forced_stop: bool = False   # hit the tool-round ceiling and was told to answer


@dataclass
class Run:
    harness: str
    question_id: str
    sample: int
    turns: list = field(default_factory=list)

    def forecasts(self) -> list:
        return [t.forecast for t in self.turns]

    def to_dict(self) -> dict:
        return {"harness": self.harness, "question_id": self.question_id,
                "sample": self.sample,
                "turns": [{"date": t.date, "forecast": t.forecast, "text": t.text,
                           "tool_calls": t.tool_calls, "carry": t.carry,
                           "error": t.error, "forced_stop": t.forced_stop}
                          for t in self.turns]}


_FC = re.compile(r"FORECAST:\s*([0-9]+(?:[.,][0-9]+)?|[.,][0-9]+)\s*(%?)", re.I)


def parse_forecast(text: str):
    """Strict: only an explicit FORECAST line counts.

    Reviewer findings fixed here: '85%' used to clamp to 1.0; a bare-number
    fallback could grab a stray '1.' from a numbered list and fabricate
    certainty; '0,45' parsed as 0.0.  Now: percent divides by 100, comma is a
    decimal separator, any value outside [0,1] is a parse FAILURE (triggering
    the per-turn retry), and there is no fallback at all.
    """
    m = _FC.findall(text or "")
    if not m:
        return None
    num, pct = m[-1]
    try:
        v = float(num.replace(",", "."))
    except ValueError:
        return None
    if pct:
        v /= 100.0
    if not (0.0 <= v <= 1.0):
        return None
    return v


def _extract_block(text: str, tag: str) -> str:
    m = re.search(r"<%s>(.*?)</%s>" % (tag, tag), text or "", re.S | re.I)
    return m.group(1).strip() if m else ""


class HarnessRunner:
    def __init__(self, client, model: str, index: NewsIndex,
                 max_tool_rounds: int = 6, max_results: int = 6,
                 reasoning_effort: str | None = None):
        self.client = client
        self.model = model
        self.index = index
        self.max_tool_rounds = max_tool_rounds
        self.max_results = max_results
        self.reasoning_effort = reasoning_effort

    # ---------- one day ----------
    def _agent_turn(self, system: str, messages: list, sim_date: str):
        """Run the tool loop for one simulation day.  Returns (text, tool_calls)."""
        tools = [search_tool_spec()]
        msgs = [{"role": "system", "content": system}] + messages
        calls_log: list = []

        for _ in range(self.max_tool_rounds):
            kw = dict(model=self.model, messages=msgs, tools=tools)
            if self.reasoning_effort:
                kw["reasoning_effort"] = self.reasoning_effort
            resp = self.client.chat.completions.create(**kw)
            m = resp.choices[0].message
            msgs.append(m.model_dump(exclude_none=True))

            if not getattr(m, "tool_calls", None):
                return (m.content or ""), calls_log, False

            for tc in m.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                q = str(args.get("query", ""))
                try:
                    k = int(args.get("k") or self.max_results)
                except (TypeError, ValueError):
                    k = self.max_results
                k = max(1, min(k, self.max_results))  # a negative k must never mean 'unlimited'
                arts = self.index.search(q, to_date=sim_date,
                                         from_date=args.get("from_date"), k=k)
                calls_log.append({"query": q, "from_date": args.get("from_date"),
                                  "n": len(arts),
                                  "titles": [a.title for a in arts]})
                body = ("\n\n".join(a.render() for a in arts)
                        if arts else "No articles found for that query.")
                msgs.append({"role": "tool", "tool_call_id": tc.id,
                             "name": "search_news", "content": body})

        # ran out of tool rounds: force a final answer
        msgs.append({"role": "user",
                     "content": "Stop searching. Give your final reasoning and "
                                "the FORECAST line now."})
        kw = dict(model=self.model, messages=msgs, tools=tools, tool_choice="none")
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        resp = self.client.chat.completions.create(**kw)
        return (resp.choices[0].message.content or ""), calls_log, True  # forced stop

    # ---------- the sequential protocol ----------
    def run(self, harness: str, question, dates: list, sample: int = 0) -> Run:
        """K dates in order. Each day sees its own prior forecasts, which is the
        anchoring pressure the whole experiment is about."""
        run = Run(harness=harness, question_id=question.id, sample=sample)
        carry = ""
        history: list = []

        for d in dates:
            system = SYSTEMS[harness].format(date=d)
            parts = ["QUESTION: " + question.text]
            if getattr(question, "background", ""):
                parts.append("BACKGROUND: " + question.background)
            if getattr(question, "resolution_date", None):
                parts.append("RESOLVES: " + str(question.resolution_date))
            if history:
                parts.append("YOUR PREVIOUS FORECASTS:\n" + "\n".join(history))
            if carry:
                tag = "MEMORY" if harness == "futuresim" else "BELIEF STATE"
                parts.append("YOUR %s FROM YESTERDAY:\n%s" % (tag, carry))
            parts.append("Today is %s. Produce your forecast for today." % d)

            turn = Turn(date=d, forecast=None, text="")
            # One retry. A hole in the trajectory costs a whole instance
            # downstream and the sequential protocol cannot backfill it.
            for _attempt in range(2):
                try:
                    text, calls, forced = self._agent_turn(
                        system, [{"role": "user", "content": "\n\n".join(parts)}], d)
                    turn.text, turn.tool_calls, turn.forced_stop = text, calls, forced
                    turn.forecast = parse_forecast(text)
                    if harness == "futuresim":
                        turn.carry = _extract_block(text, "memory")
                    elif harness == "bayesian":
                        turn.carry = _extract_block(text, "belief")
                    if turn.forecast is not None:
                        turn.error = ""
                        break
                    turn.error = "no FORECAST line parsed"
                except Exception as e:                  # noqa: BLE001
                    turn.error = "%s: %s" % (type(e).__name__, str(e)[:300])

            run.turns.append(turn)
            if turn.carry:
                carry = turn.carry
            if turn.forecast is not None:
                history.append("  %s: %.3f" % (d, turn.forecast))
        return run


def trajectory_text(run: Run) -> str:
    """The linear reasoning route, flattened for the extractor: what it said,
    and what it looked at, in order."""
    out = []
    for t in run.turns:
        out.append("=" * 66)
        out.append("DATE %s   (forecast produced: %s)"
                   % (t.date, "none" if t.forecast is None else "%.3f" % t.forecast))
        out.append("=" * 66)
        if t.tool_calls:
            out.append("-- searches run --")
            for c in t.tool_calls:
                out.append("  query: %s  -> %d articles" % (c["query"], c["n"]))
                for ti in c["titles"][:4]:
                    out.append("      - " + ti)
        out.append("-- reasoning --")
        out.append(t.text.strip())
        if t.error:
            out.append("-- ERROR: " + t.error)
    return "\n".join(out)
