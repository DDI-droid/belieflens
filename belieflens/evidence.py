"""Date-gated news search over the real FutureSim corpus.

FutureSim's own tool is hybrid semantic + keyword over a 48 GB prebuilt
embedding index.  We use the same articles but a local SQLite FTS5 (BM25)
index instead.  Retrieval is weaker; it is also *identical for all four
harnesses*, so it is a constant of the experiment rather than a confound in
it.  The thing under comparison is reasoning structure, not retrieval.

The date gate is the part that must not be got wrong: at simulation date D an
agent may only see articles dated <= D.  Enforced in SQL, not in a prompt.
"""
from __future__ import annotations

import glob
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

DB_PATH = os.environ.get("BL_INDEX", "data/news.db")


@dataclass
class Article:
    id: str
    title: str
    source: str
    date: str
    url: str
    snippet: str

    def render(self, n: int = 900) -> str:
        body = self.snippet[:n].replace("\n", " ").strip()
        return "[%s | %s] %s\n%s" % (self.date, self.source, self.title, body)


def build(corpus_dir: str = "data/corpus", db_path: str = DB_PATH,
          content_chars: int = 6000, verbose: bool = True) -> int:
    import pyarrow.parquet as pq

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("""CREATE TABLE art(
        rid INTEGER PRIMARY KEY, id TEXT, title TEXT, source TEXT,
        date TEXT, url TEXT, content TEXT)""")
    con.execute("CREATE INDEX art_date ON art(date)")
    con.execute("CREATE VIRTUAL TABLE fts USING fts5(text, content='')")

    files = sorted(glob.glob(os.path.join(corpus_dir, "*", "*", "*", "*.parquet")))
    total = 0
    for fi, f in enumerate(files):
        t = pq.read_table(f, columns=["id", "title", "source", "date", "url",
                                      "content", "description"])
        rows = t.to_pylist()
        batch, fbatch = [], []
        for r in rows:
            total += 1
            content = (r.get("content") or "")[:content_chars]
            desc = r.get("description") or ""
            batch.append((total, r.get("id") or "", r.get("title") or "",
                          r.get("source") or "", r.get("date") or "",
                          r.get("url") or "", content))
            fbatch.append((total, " ".join([r.get("title") or "", desc, content])))
        con.executemany("INSERT INTO art VALUES(?,?,?,?,?,?,?)", batch)
        con.executemany("INSERT INTO fts(rowid, text) VALUES(?,?)", fbatch)
        if verbose and fi % 20 == 0:
            print("  %4d/%d files, %d articles" % (fi + 1, len(files), total), flush=True)
    con.commit()
    con.execute("INSERT INTO fts(fts) VALUES('optimize')")
    con.commit()
    con.close()
    if verbose:
        print("indexed %d articles -> %s (%.0f MB)"
              % (total, db_path, os.path.getsize(db_path) / 1e6))
    return total


_TOKEN = re.compile(r"[A-Za-z0-9']+")


def _fts_query(q: str) -> str:
    """FTS5 has its own query syntax and raises on stray punctuation.  Quote
    every token so an agent can type anything without crashing the tool."""
    toks = _TOKEN.findall(q or "")
    toks = [t for t in toks if len(t) > 1][:24]
    if not toks:
        return '""'
    return " OR ".join('"%s"' % t for t in toks)


class NewsIndex:
    """Thread-safe reader.

    A single sqlite3.Connection is NOT safe to use from several threads at
    once, even with check_same_thread=False: concurrent execute() on one
    connection returns corrupted rows and raises InterfaceError. The harnesses
    run in a thread pool, so every thread gets its own connection.
    """

    def __init__(self, db_path: str = DB_PATH):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                "no index at %s -- run: python scripts/build_index.py" % db_path)
        self.db_path = db_path
        self._local = threading.local()

    @property
    def con(self):
        c = getattr(self._local, "con", None)
        if c is None:
            c = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.con = c
        return c

    def search(self, query: str, to_date: str, from_date: str | None = None,
               k: int = 6) -> list:
        """Date-gated BM25 search.  `to_date` is the hard wall: nothing later
        than the current simulation date is retrievable, ever."""
        sql = """SELECT a.id, a.title, a.source, a.date, a.url, a.content
                 FROM fts JOIN art a ON a.rid = fts.rowid
                 WHERE fts MATCH ? AND a.date <= ?"""
        params: list = [_fts_query(query), to_date]
        if from_date:
            sql += " AND a.date >= ?"
            params.append(from_date)
        sql += " ORDER BY bm25(fts) LIMIT ?"
        # a negative or zero k must never reach LIMIT: LIMIT -1 means unlimited
        params.append(max(1, min(int(k), 50)))
        try:
            rows = self.con.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            # masked failures silently distort forecasts -- at least say so
            import sys
            print("WARNING search failed (%s) query=%r" % (e, query[:80]),
                  file=sys.stderr, flush=True)
            return []
        return [Article(*r) for r in rows]

    def day_headlines(self, date: str, k: int = 40) -> list:
        rows = self.con.execute(
            "SELECT id,title,source,date,url,substr(content,1,400) FROM art "
            "WHERE date = ? LIMIT ?", (date, k)).fetchall()
        return [Article(*r) for r in rows]

    def span(self) -> tuple:
        return self.con.execute("SELECT min(date), max(date), count(*) FROM art").fetchone()


def search_tool_spec() -> dict:
    """The one tool every harness gets.  Identical across all four."""
    return {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": ("Search dated news articles. You can only retrieve articles "
                            "published on or before the current simulation date; future "
                            "articles do not exist yet. Use several targeted queries "
                            "rather than one broad one."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "keywords to search for"},
                    "from_date": {"type": "string",
                                  "description": "optional earliest date, YYYY-MM-DD"},
                    "k": {"type": "integer", "description": "how many articles, max 6"},
                },
                "required": ["query"],
            },
        },
    }
