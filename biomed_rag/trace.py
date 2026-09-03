"""Flow tracing for the RAG pipeline.

Levels (case-insensitive):

    debug    every step: retrieval scores, node counts, per-query metrics,
             per-call retrieve/query/answer tracing, dataset load counts, ...
    info     headline milestones only: slow model loads, "N documents -> N
             nodes", "N vector(s) written", index build, final eval metrics
    warning  (default) near-silent — only genuine problems

Output goes to stderr as ``LEVEL  biomed_rag.<module> | message`` so it never
mixes into a script's stdout results. Set ``BIOMED_RAG_LOG_JSON=1`` for
one-JSON-object-per-line instead (handy for piping into `jq`).
"""
from __future__ import annotations

import json
import logging
import os
import sys

_ROOT = "biomed_rag"
_configured = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        return json.dumps(payload)


def _configure() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.environ.get("BIOMED_RAG_LOG", "warning").strip().upper()
    level = getattr(logging, level_name, logging.WARNING)

    handler = logging.StreamHandler(sys.stderr)
    if os.environ.get("BIOMED_RAG_LOG_JSON", "").strip() in ("1", "true", "yes"):
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(levelname)-5s %(name)s | %(message)s")
        )

    root = logging.getLogger(_ROOT)
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:

    _configure()
    if not name or name == "__main__":
        name = f"{_ROOT}.main"
    elif not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)
