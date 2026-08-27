"""Provider-agnostic chat wrapper with an offline mock backend.

Providers
---------
anthropic   official SDK.  Note: `temperature` is REJECTED (400) on the current
            reasoning models (Opus 5, Opus 4.7/4.8, Sonnet 5, Fable 5).  Draws
            across repeated calls are therefore ordinary sampling noise, not
            temperature-controlled noise -- see `sample_id` below.
openai      chat.completions, for GPT-family baselines.
openrouter  same wire format as openai, different base_url; the cheapest way
            to add open-weight models (Qwen, DeepSeek, GLM) as a third family.
mock        deterministic, no network.  Lets the whole pipeline and every
            metric be exercised offline before spending a cent.

Caching: every call is keyed by (provider, model, system, user) and written to
`cache/`, so re-running an analysis costs nothing and re-runs are exactly
reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path(os.environ.get("BELIEFLENS_CACHE", "cache"))


@dataclass
class Reply:
    text: str
    model: str
    cached: bool = False
    usage: dict | None = None


class LLM:
    def __init__(self, provider: str, model: str, effort: str = "high",
                 max_tokens: int = 16000, temperature: float | None = None,
                 use_cache: bool = True, sample_nonce: bool = False):
        self.provider = provider
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_cache = use_cache
        # When True, a "Sample id: k" line is appended to force independent
        # draws from providers that would otherwise serve a cached completion.
        # It perturbs the prompt, so it is a confound -- keep it off unless a
        # provider gives you identical text across draws, and say so in the paper.
        self.sample_nonce = sample_nonce
        self._client = None

    # ---------- public ----------
    def chat(self, system: str, user: str, sample_id: int = 0) -> Reply:
        if self.sample_nonce and sample_id:
            user = user + "\n\n[Sample id: %d]" % sample_id
        key = self._key(system, user, sample_id)
        if self.use_cache:
            hit = self._cache_get(key)
            if hit is not None:
                return Reply(text=hit, model=self.model, cached=True)
        text, usage = self._call(system, user, sample_id)
        if self.use_cache:
            self._cache_put(key, text)
        return Reply(text=text, model=self.model, usage=usage)

    def sample(self, system: str, user: str, k: int) -> list[Reply]:
        return [self.chat(system, user, sample_id=i) for i in range(k)]

    # ---------- caching ----------
    def _key(self, system: str, user: str, sample_id: int) -> str:
        blob = json.dumps([self.provider, self.model, self.effort, self.temperature,
                           system, user, sample_id], sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / (self.provider + "_" + self.model.replace("/", "_")) / (key + ".txt")

    def _cache_get(self, key: str):
        p = self._cache_path(key)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def _cache_put(self, key: str, text: str) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    # ---------- backends ----------
    def _call(self, system: str, user: str, sample_id: int = 0):
        for attempt in range(4):
            try:
                if self.provider == "anthropic":
                    return self._anthropic(system, user)
                if self.provider in ("openai", "openrouter"):
                    return self._openai_compatible(system, user)
                if self.provider == "mock":
                    return self._mock(system, user, sample_id), None
                raise ValueError("unknown provider " + self.provider)
            except Exception as exc:              # noqa: BLE001 - retry envelope
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt + random.random())
        raise RuntimeError("unreachable")

    def _anthropic(self, system: str, user: str):
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic()
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )
        # Streaming so a large max_tokens cannot trip the HTTP timeout.
        with self._client.messages.stream(**kwargs) as stream:
            msg = stream.get_final_message()
        if getattr(msg, "stop_reason", None) == "refusal":
            raise RuntimeError("model refused: %s" % getattr(msg, "stop_details", None))
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}
        return text, usage

    def _openai_compatible(self, system: str, user: str):
        from openai import OpenAI
        if self._client is None:
            if self.provider == "openrouter":
                self._client = OpenAI(base_url="https://openrouter.ai/api/v1",
                                      api_key=os.environ["OPENROUTER_API_KEY"])
            else:
                self._client = OpenAI()
        kwargs = dict(model=self.model,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}])
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        r = self._client.chat.completions.create(**kwargs)
        u = getattr(r, "usage", None)
        usage = {"input": getattr(u, "prompt_tokens", None),
                 "output": getattr(u, "completion_tokens", None)} if u else None
        return r.choices[0].message.content or "", usage

    # ---------- offline mock ----------
    def _mock(self, system: str, user: str, sample_id: int = 0) -> str:
        """Deterministic pseudo-model.  Produces valid BeliefSpecs and valid
        direct answers so the pipeline and every metric can be tested with no
        network and no key."""
        rng = random.Random(hashlib.sha256(("%d|" % sample_id + system[:200] + user).encode()).hexdigest())
        question = _field(user, "QUESTION:") or "unspecified"
        as_of = _field(user, "DATE:") or "2026-01-01"

        # Update prompts: echo the incoming spec back with a plausible move, so
        # the locality / rigidity / follow-through metrics are exercised offline.
        if "CURRENT BELIEFSPEC:" in user:
            return self._mock_update(system, user, rng)

        wants_spec = "BeliefSpec" in system and "nothing else" in system
        if not wants_spec:
            p = round(rng.uniform(0.2, 0.8), 3)
            return ('Prior reasoning, no evidence available.\n'
                    '{"p": %.3f, "wep": "%s"}' % (p, _wep_for(p)))

        n_leaves = rng.randint(3, 5)
        nodes = [{"id": "n0", "kind": "root", "claim": question, "p": 0.5,
                  "wep": "possible", "parent": None, "w": 0.0}]
        ws = [round(rng.uniform(0.15, 0.3), 3) for _ in range(n_leaves)]
        scale = min(1.0, 0.9 / sum(ws))
        ws = [round(w * scale, 3) for w in ws]
        for i, w in enumerate(ws, start=1):
            p = round(rng.uniform(0.15, 0.9), 3)
            nodes.append({
                "id": "n%d" % i, "kind": "leaf",
                "claim": "Driver %d bearing on: %s" % (i, question[:60]),
                "p": p, "wep": _wep_for(p), "parent": "n0", "w": w,
                "sensitivity": [
                    {"if": "a credible report confirms driver %d" % i,
                     "p_to": round(min(0.97, p + rng.uniform(0.1, 0.3)), 3)},
                    {"if": "a credible report contradicts driver %d" % i,
                     "p_to": round(max(0.03, p - rng.uniform(0.1, 0.3)), 3)},
                ],
            })
        b0 = round(max(0.0, 1.0 - sum(ws)) * rng.uniform(0.0, 0.6), 3)
        root_p = round(min(1.0, b0 + sum(w * n["p"] for w, n in zip(ws, nodes[1:]))), 3)
        nodes[0]["p"] = root_p
        nodes[0]["wep"] = _wep_for(root_p)
        spec = {"question": question, "as_of": as_of, "root": "n0",
                "rule": "linear", "b0": {"n0": b0}, "nodes": nodes}
        return "```json\n" + json.dumps(spec, indent=2) + "\n```"


    def _mock_update(self, system: str, user: str, rng: random.Random) -> str:
        """Simulate a partially-rigid updater: move the cued node most of the
        way, nudge a couple of neighbours, and (for the free-form protocol)
        shave a little off the weights."""
        from .dsl import extract_json
        head = user.split("NEW INFORMATION:")[0].split("CURRENT BELIEFSPEC:", 1)[-1]
        spec = extract_json(head)
        evidence = user.split("NEW INFORMATION:", 1)[-1].lower()
        for n in spec.get("nodes", []):
            claim = str(n.get("claim", "")).lower()
            cues = [str(c.get("if", "")).lower() for c in n.get("sensitivity", [])]
            hit = any(c and c.rstrip(".") in evidence for c in cues)
            if hit:
                targets = [c["p_to"] for c in n["sensitivity"]
                           if str(c.get("if", "")).lower().rstrip(".") in evidence]
                tgt = targets[0]
                # follow through only ~70% of the way: a deliberate, detectable shortfall
                n["p"] = round(n["p"] + 0.7 * (tgt - n["p"]), 3)
                n["note"] = "moved by the reported evidence"
            elif _overlap(claim, evidence) > 0.6:
                n["p"] = round(min(1.0, max(0.0, n["p"] + rng.uniform(-0.05, 0.05))), 3)
        if "may not add nodes" not in system:   # free protocol only
            for n in spec.get("nodes", []):
                if n.get("parent"):
                    n["w"] = round(max(0.0, n["w"] * rng.uniform(0.9, 1.05)), 3)
        return "```json\n" + json.dumps(spec, indent=2) + "\n```"


def _overlap(a: str, b: str) -> float:
    ta = {w for w in a.split() if len(w) > 3}
    if not ta:
        return 0.0
    tb = {w for w in b.split() if len(w) > 3}
    return len(ta & tb) / len(ta)


def _field(text: str, tag: str) -> str:
    for line in (text or "").splitlines():
        if line.startswith(tag):
            return line[len(tag):].strip()
    return ""


def _wep_for(p: float) -> str:
    for label, cut in (("confirmed", 0.95), ("almost certain", 0.85),
                       ("probable", 0.65), ("possible", 0.4),
                       ("unlikely", 0.2), ("doubtful", 0.0)):
        if p >= cut:
            return label
    return "unknown"


def from_config(cfg: dict) -> LLM:
    return LLM(provider=cfg.get("provider", "mock"),
               model=cfg.get("model", "mock-1"),
               effort=cfg.get("effort", "high"),
               max_tokens=int(cfg.get("max_tokens", 16000)),
               temperature=cfg.get("temperature"),
               use_cache=bool(cfg.get("cache", True)),
               sample_nonce=bool(cfg.get("sample_nonce", False)))
