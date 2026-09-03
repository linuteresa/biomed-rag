"""Flow tracing for the RAG pipeline.

Every module calls ``log = get_logger(__name__)`` and emits ``log.debug(...)`` at
each step of its logic. Nothing prints by default; turn it on with an env var:

    BIOMED_RAG_LOG=debug  python scripts/answer.py "does IL-6 drive M2 ..."
    BIOMED_RAG_LOG=info   python scripts/eval.py

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
    root.handlers[:] = [handler]
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``biomed_rag`` namespace.

    ``name`` is normally ``__name__`` (e.g. ``biomed_rag.eval.harness``); a bare
    name is namespaced automatically so ``get_logger("scratch")`` also works.
    """
    _configure()
    if not name or name == "__main__":
        name = f"{_ROOT}.main"
    elif not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)
