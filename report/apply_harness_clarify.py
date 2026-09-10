# Make the prompt-shape vs real-system distinction explicit everywhere.
p = "main.tex"
s = open(p, encoding="utf-8").read()

def rep(o, n, t):
    global s
    assert o in s, "MISS:" + t
    s = s.replace(o, n, 1)

rep(r"""Four reasoning
harnesses (\hr{react}, \hr{futuresim}, \hr{analytica}, \hr{bayesian}), built as
prompts over one shared substrate on a 339{,}801-article date-gated news corpus,""",
    r"""Four \emph{harness shapes}
(\hr{react}, \hr{futuresim}, \hr{analytica}, \hr{bayesian})---prompt-defined re-implementations of
each source system's reasoning structure, with shape-specific state-carry code, executed by one
shared single-agent tool loop on a 339{,}801-article date-gated news corpus---""",
    "abs")

rep(r"""Four system prompts over one substrate (same model, tools, tool budget, output contract
\texttt{FORECAST:\,$p$}); Table~\ref{tab:harness}. Installing the papers' repositories would
compare repositories; holding the substrate fixed makes reasoning shape the only variable.""",
    r"""\textbf{What these are, precisely.} All four run on \emph{one} shared code path: a
single-agent tool loop (at most five \texttt{search\_news} rounds, then a forced answer, output
contract \texttt{FORECAST:\,$p$}). What differs per harness is (i) the system prompt imposing the
reasoning structure and (ii) a small amount of shape-specific code---for \hr{futuresim} and
\hr{bayesian}, programmatic extraction of the memory/belief block and its re-injection the next
day. They are therefore \emph{prompt-level re-implementations of each source system's reasoning
shape}, not the source systems themselves: real Analytica builds its proposition tree and computes
its linear composition \emph{in orchestration code} with parallel per-leaf grounder agents; the
real FutureSim harness maintains task state as an executable dataframe with structured memory
tools; the BLF system adds multi-trial aggregation and calibration. Our \hr{react} coincides with
its source (ReAct \emph{is} a prompt pattern over a tool loop); the others approximate.
This is a deliberate control---installing the papers' repositories would compare repositories,
whereas holding the substrate fixed makes reasoning shape the only manipulated variable---but it
bounds the claims: every cross-harness result in this report is a statement about reasoning
shapes at fixed capacity, \textbf{not} an evaluation of the source systems. Table~\ref{tab:harness}
summarises.""",
    "s32")

rep(r"""\hr{analytica}/USA & 0.108 & 0.274 & 0.073$\,\to\,$0.285 & \multirow{2}{*}{structure changes daily} \\""",
    r"""\hr{analytica}/USA & 0.108 & 0.274 & 0.073$\,\to\,$0.285 & \multirow{2}{*}{\parbox{4.4cm}{structure changes daily (in-context shape; see \S\ref{sec:limits} item 8)}} \\""",
    "t3note")

rep(r"\section{Limitations}",
    r"\section{Limitations}\label{sec:limits}", "limlabel")

rep(r"""(7) Program recovery inherits the
expressiveness of the program space; ``incompressible'' means no program \emph{in this space}
within four repair attempts.""",
    r"""(7) Program recovery inherits the
expressiveness of the program space; ``incompressible'' means no program \emph{in this space}
within four repair attempts. (8) The harnesses are prompt-level shapes on a shared loop
(\S3.2), not the source systems: in particular, real Analytica enforces its tree and linear
composition \emph{in code}, so ``\hr{analytica}'s structure changes daily'' characterises a model
asked to follow that procedure in-context, and would be prevented by construction in the original
system; conversely, \hr{bayesian}'s recovered fixed formula is the more striking for being
maintained by the model with no code enforcement.""",
    "lim8")

open(p, "w", encoding="utf-8").write(s)
print("clarifications applied")
