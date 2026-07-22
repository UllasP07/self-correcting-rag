"""SQL risk assessment for the human-in-the-loop gate (Milestone 6).

Our SQL is already read-only (M5), so "risky" here isn't about destruction — it's
about *exposure*. Two things warrant a human's sign-off before a query runs:

  1. SENSITIVE COLUMNS — the query reads something we've marked confidential
     (e.g. `salary`). Even a SELECT can leak sensitive data.
  2. BROAD SCANS — the query has no WHERE filter and no LIMIT and isn't an
     aggregate, so it would dump whole rows of the table (bulk exfiltration).

This is a pure function (no LLM, no DB), so it's trivially testable and the
policy is auditable at a glance — exactly what you want gating a human approval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import settings

_AGG = re.compile(r"\b(count|sum|avg|min|max|group\s+by)\b", re.IGNORECASE)
_HAS_WHERE = re.compile(r"\bwhere\b", re.IGNORECASE)
_HAS_LIMIT = re.compile(r"\blimit\b", re.IGNORECASE)


@dataclass
class RiskAssessment:
    risky: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "no risk flags"


def assess_risk(
    sql: str,
    sensitive_columns: tuple[str, ...] | None = None,
    flag_broad_scans: bool | None = None,
) -> RiskAssessment:
    """Flag a (read-only) query that needs admin approval before it runs."""
    cols = (settings.sensitive_columns if sensitive_columns is None
            else tuple(c.lower() for c in sensitive_columns))
    flag_broad = (settings.hitl_flag_broad_scans if flag_broad_scans is None
                  else flag_broad_scans)

    lowered = sql.lower()
    reasons: list[str] = []

    # 1. sensitive column access (word-boundary match to avoid false hits)
    for col in cols:
        if re.search(rf"\b{re.escape(col)}\b", lowered):
            reasons.append(f"reads sensitive column '{col}'")

    # 2. unbounded scan: no WHERE, no LIMIT, and not an aggregate rollup
    if flag_broad:
        is_aggregate = bool(_AGG.search(sql))
        if not is_aggregate and not _HAS_WHERE.search(sql) and not _HAS_LIMIT.search(sql):
            reasons.append("unbounded scan (no WHERE/LIMIT) could return whole rows")

    return RiskAssessment(risky=bool(reasons), reasons=reasons)
