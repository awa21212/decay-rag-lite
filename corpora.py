# -*- coding: utf-8 -*-
"""Corpus registry — one index per corpus, no cross-contamination.

Four independent corpora:

- ``code``    : project conventions, debugging pitfalls, toolchain notes
- ``finance`` : factors, backtest reports, financial logic
- ``life``    : chats, journal entries, personal notes
- ``task``    : current status, hand-offs, environment snapshots

Paths come from ``config.json`` (copy ``config.example.json`` and edit it).
Every corpus maps to its own index directory ``index_<name>`` next to this file.

Usage::

    cp config.example.json config.json     # then edit the paths
    python build.py --corpus code
    python ask.py --corpus finance "turnover vs fitness"
    python preflight.py "docker build"      # searches code/finance/task by default
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(BASE, "config.json")
EXAMPLE = os.path.join(BASE, "config.example.json")

# External vector store (optional, read-only). Leave empty to disable --st.
ST_VECTORS = ""

# Corpora searched by preflight.py unless told otherwise.
# ``life`` is excluded on purpose: journal chatter would outrank technical cards.
DEFAULT_SCOPE = ("code", "finance", "task")


def _load_config():
    path = CONFIG if os.path.exists(CONFIG) else EXAMPLE
    if not os.path.exists(path):
        raise SystemExit("missing config.json — copy config.example.json and edit it")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def _build(entry):
    sources = []
    for s in entry.get("sources", []):
        if isinstance(s, str):
            sources.append((_abs(s), None))
        else:
            sources.append((_abs(s["path"]), s.get("include")))
    return {
        "index": os.path.join(BASE, "index_" + entry["name"]),
        "desc": entry.get("desc", ""),
        "sources": sources,
    }


_cfg = _load_config()
CORPUS = {e["name"]: _build(e) for e in _cfg["corpora"]}
ST_VECTORS = _cfg.get("st_vectors", "") or ST_VECTORS
DEFAULT_SCOPE = tuple(_cfg.get("default_scope", DEFAULT_SCOPE))

# Old names from earlier versions: fail with a hint instead of a traceback.
RETIRED = {"main": "finance", "sop": "code", "now": "task"}


def get(name):
    if name in CORPUS:
        return CORPUS[name]
    if name in RETIRED:
        raise SystemExit("corpus %s was retired, use %s (e.g. python ask.py --corpus %s ...)"
                         % (name, RETIRED[name], RETIRED[name]))
    raise SystemExit("unknown corpus: %s (available: %s)" % (name, ", ".join(CORPUS)))


def names():
    return tuple(CORPUS)
