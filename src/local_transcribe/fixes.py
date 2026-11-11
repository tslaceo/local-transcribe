"""Apply corrections to a text so that none of them can miss silently.

    local-transcribe fix TEXT.txt FIXES.json [-o OUT.txt]

``FIXES.json`` is a list of ``{"old": "...", "new": "...", "count": 1}``; ``count`` is how
many times ``old`` must occur and defaults to 1. The fixes are applied in order, each to
the result of the ones before it. If any ``old`` occurs a different number of times than
stated, nothing is written and every miss is listed.

Why a tool for search-and-replace: a long transcript takes hundreds of corrections, made
over several passes. A correction whose ``old`` no longer matches (the text moved under it,
a typo, an earlier fix ate it) does nothing and says nothing, and one that matches twice
changes a place nobody looked at. Stating the count turns both into errors. Keep the fixes
file next to the raw text: together they are the record of what was heard and what was
changed.
"""

from __future__ import annotations

from typing import Any, Sequence


def apply_fixes(text: str, fixes: Sequence[dict[str, Any]]) -> tuple[str, list[str]]:
    """``text`` with the fixes applied in order, and a line for every fix that did not match as stated."""
    misses: list[str] = []
    for n, fix in enumerate(fixes, 1):
        old, new, count = fix["old"], fix["new"], int(fix.get("count", 1))
        if not old:
            misses.append(f"fix {n}: empty 'old'")
            continue
        found = text.count(old)
        if found != count:
            misses.append(f"fix {n}: {old[:60]!r} occurs {found} times, expected {count}")
            continue
        text = text.replace(old, new)
    return text, misses
