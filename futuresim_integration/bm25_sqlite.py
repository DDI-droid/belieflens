"""SQLite-FTS5 (BM25) search tool (BeliefLens adapter).

Implements FutureSim's BaseSearchTool over the BeliefLens news index
(data/news.db: table `art` + FTS5 table `fts`), replacing the 48 GB LanceDB
hybrid index with local keyword search. Date gating is enforced here in SQL,
in addition to the SearchHandler's own ceiling.
"""

import os
import re
import sqlite3
import threading
from datetime import date, datetime
from typing import List, Optional

from .base import Article, BaseSearchTool, SearchResult

_TOKEN = re.compile(r"[A-Za-z0-9']+")


def _fts_query(q: str) -> str:
    toks = [t for t in _TOKEN.findall(q or "") if len(t) > 1][:24]
    return " OR ".join('"%s"' % t for t in toks) if toks else '""'


def _iso(d) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, (date, datetime)):
        return d.strftime("%Y-%m-%d")
    return str(d)[:10]


def _to_date(s) -> Optional[date]:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


class BM25SqliteSearchTool(BaseSearchTool):
    def __init__(self, db_path: str, snippet_chars: int = 1200):
        self.db_path = db_path
        self.snippet_chars = snippet_chars
        self._local = threading.local()

    @property
    def _con(self):
        c = getattr(self._local, "con", None)
        if c is None:
            c = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.con = c
        return c

    @property
    def is_available(self) -> bool:
        return os.path.exists(self.db_path)

    def search(self, query: str, max_results: int = 10,
               max_date: Optional[date] = None, search_type: str = "hybrid",
               min_date: Optional[date] = None) -> List[SearchResult]:
        sql = ("SELECT a.id, a.title, a.source, a.date, a.url, a.content "
               "FROM fts JOIN art a ON a.rid = fts.rowid WHERE fts MATCH ?")
        params = [_fts_query(query)]
        if max_date is not None:
            sql += " AND a.date <= ?"
            params.append(_iso(max_date))
        if min_date is not None:
            sql += " AND a.date >= ?"
            params.append(_iso(min_date))
        sql += " ORDER BY bm25(fts) LIMIT ?"
        params.append(max(1, min(int(max_results), 50)))
        try:
            rows = self._con.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        out = []
        for rid, title, source, d, url, content in rows:
            out.append(SearchResult(
                article_id=str(rid), title=title or "", source=source or "",
                date=_to_date(d), date_publish=None,
                snippet=(content or "")[: self.snippet_chars],
                score=0.0, url=url or ""))
        return out

    def get_article(self, article_id: str) -> Optional[Article]:
        row = self._con.execute(
            "SELECT id, title, source, date, url, content FROM art WHERE rid = ?",
            (article_id,)).fetchone()
        if row is None:
            return None
        aid, title, source, d, url, content = row
        return Article(id=str(aid), title=title or "", source=source or "",
                       date=_to_date(d), content=content or "", url=url or "")

    def count_articles(self, min_date: Optional[date] = None,
                       max_date: Optional[date] = None) -> Optional[int]:
        sql = "SELECT count(*) FROM art WHERE 1=1"
        params = []
        if min_date is not None:
            sql += " AND date >= ?"
            params.append(_iso(min_date))
        if max_date is not None:
            sql += " AND date <= ?"
            params.append(_iso(max_date))
        return self._con.execute(sql, params).fetchone()[0]
