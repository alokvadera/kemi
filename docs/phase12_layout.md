# Phase 12 — kemi library layout reorg

## 1. Goals

- **30+ top-level files** in `src/kemi/` made the codebase hard to navigate.
- **3 different "core" layer files** (`core.py`, `_memory_core.py`, `_memory_impl.py`) confused contributors.
- **cli.py (1116 LOC)** and **api_server.py (1916 LOC)** were the two biggest files, buried among small helpers.
- Group files by **domain**: memory subsystem, infrastructure, NLP pipeline, and user-facing interfaces.

## 2. New tree

```
src/kemi/
├── __init__.py              (public API — unchanged)
├── __main__.py              (unchanged)
├── core.py                  (back-compat shim)
├── exceptions.py            (unchanged)
│
├── memory/
│   ├── __init__.py
│   ├── core.py              (was _memory_core.py)
│   ├── facade.py            (was _memory_impl.py)
│   ├── service.py           (was memory_service.py)
│   ├── model.py             (was models.py)
│   ├── lifecycle.py         (unchanged move)
│   ├── scoring.py           (unchanged move)
│   ├── dedup.py             (unchanged move)
│   ├── sanitize.py          (unchanged move)
│   ├── chunker.py           (unchanged move)
│   ├── versions.py          (unchanged move)
│   ├── entities.py          (unchanged move)
│   ├── adaptive.py          (unchanged move)
│   ├── consolidation.py     (unchanged move)
│   ├── procedures.py        (unchanged move)
│   └── formation.py         (was memory_formation.py)
│
├── infra/
│   ├── __init__.py
│   ├── audit.py             (unchanged move)
│   ├── background_tasks.py  (unchanged move)
│   ├── encryption.py        (unchanged move)
│   ├── webhooks.py          (unchanged move)
│   ├── api_keys.py          (unchanged move)
│   └── observability.py     (unchanged move)
│
├── interfaces/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py           (was api_server.py)
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py          (was cli.py + main.py)
│   │   └── writer.py        (was cli_writer.py)
│   └── mcp/
│       ├── __init__.py
│       └── server.py        (was mcp_server.py)
│
├── nlp/
│   ├── __init__.py
│   ├── formation.py         (was memory_formation.py — renamed from memory context to NLP domain)
│   ├── decomposer.py        (unchanged move)
│   ├── topics.py            (unchanged move)
│   ├── summarizer.py        (unchanged move)
│   ├── reranker.py          (unchanged move)
│   └── graph.py             (unchanged move)
│
├── pipeline/                (unchanged)
│   ├── ingestion.py
│   ├── retrieval.py
│   └── _steps.py
│
├── services/                (unchanged)
│   ├── read_service.py
│   ├── write_service.py
│   └── admin_service.py
│
├── operations/              (unchanged)
│   ├── _io.py
│   ├── _ops_*.py
│   └── _query_cache.py
│
├── plugins/                 (unchanged)
│   ├── builtin.py
│   ├── protocols.py
│   └── registry.py
│
├── adapters/                (unchanged)
│   ├── storage/
│   ├── embedding/
│   └── base.py
│
└── integrations/            (unchanged)
    └── langchain.py
```

## 3. Mapping table

| Old path | New path | Reason |
|---|---|---|
| `_memory_core.py` | `memory/core.py` | Core domain state object; lives in memory/ subpackage |
| `memory_service.py` | `memory/service.py` | MemoryService class; memory/ subpackage |
| `_memory_impl.py` | `memory/facade.py` | Back-compat Memory alias; was a thin subclass shim |
| `models.py` | `memory/model.py` | All domain model types; belongs in memory/ |
| `lifecycle.py` | `memory/lifecycle.py` | Memory lifecycle evaluation |
| `scoring.py` | `memory/scoring.py` | Search scoring functions |
| `dedup.py` | `memory/dedup.py` | Memory deduplication |
| `sanitize.py` | `memory/sanitize.py` | Prompt injection detection |
| `chunker.py` | `memory/chunker.py` | Semantic text chunking |
| `versions.py` | `memory/versions.py` | Memory versioning/rollback |
| `entities.py` | `memory/entities.py` | Entity extraction for retrieval |
| `adaptive.py` | `memory/adaptive.py` | Adaptive retrieval weights |
| `consolidation.py` | `memory/consolidation.py` | Memory consolidation/summarization |
| `procedures.py` | `memory/procedures.py` | Procedural memory helpers |
| `memory_formation.py` | `memory/formation.py` | Memory formation (NLP pipeline entry) |
| `audit.py` | `infra/audit.py` | Audit trail — infrastructure concern |
| `background_tasks.py` | `infra/background_tasks.py` | Background task runner — infra |
| `encryption.py` | `infra/encryption.py` | Encryption layer — infrastructure |
| `webhooks.py` | `infra/webhooks.py` | Webhook dispatch — infrastructure |
| `api_keys.py` | `infra/api_keys.py` | API key management — infrastructure |
| `observability.py` | `infra/observability.py` | Metrics collection — infrastructure |
| `api_server.py` | `interfaces/api/app.py` | FastAPI server — user-facing interface |
| `mcp_server.py` | `interfaces/mcp/server.py` | MCP server — user-facing interface |
| `cli.py` + `main.py` | `interfaces/cli/main.py` | CLI entry — user-facing interface |
| `cli_writer.py` | `interfaces/cli/writer.py` | CLI output writer |
| `decomposer.py` | `nlp/decomposer.py` | Query decomposition — NLP pipeline |
| `topics.py` | `nlp/topics.py` | Topic clustering — NLP pipeline |
| `summarizer.py` | `nlp/summarizer.py` | LLM summarization — NLP pipeline |
| `reranker.py` | `nlp/reranker.py` | Cross-encoder reranking — NLP pipeline |
| `graph.py` | `nlp/graph.py` | Memory graph extraction — NLP pipeline |

## 4. Naming renaming

- `_memory_core.py` → `memory/core.py` — drop leading underscore (private-by-convention), drop `.py` suffix.
  Chosen `core.py` over `runtime.py` because it matches the class name `_MemoryCore`.
- `_memory_impl.py` → `memory/facade.py` — this file was always a thin subclass shim (the `Memory` class). "Facade" names its role better than "impl".
- `memory_formation.py` → `memory/formation.py` — dropped the `memory_` prefix since it's already under `memory/`.

## 5. Back-compat shims (REMOVED in v0.4.0)

The 24 back-compat shim files listed below were **removed in v0.4.0**.
They were pure re-exports of the canonical subpackage modules. Use the
**New canonical path** column for all imports.

| Old path | New canonical path |
|---|---|
| `src/kemi/models.py` | `kemi.memory.model` |
| `src/kemi/entities.py` | `kemi.memory.entities` |
| `src/kemi/procedures.py` | `kemi.memory.procedures` |
| `src/kemi/memory_formation.py` | `kemi.memory.formation` |
| `src/kemi/versions.py` | `kemi.memory.versions` |
| `src/kemi/lifecycle.py` | `kemi.memory.lifecycle` |
| `src/kemi/scoring.py` | `kemi.memory.scoring` |
| `src/kemi/dedup.py` | `kemi.memory.dedup` |
| `src/kemi/sanitize.py` | `kemi.memory.sanitize` |
| `src/kemi/chunker.py` | `kemi.memory.chunker` |
| `src/kemi/adaptive.py` | `kemi.memory.adaptive` |
| `src/kemi/consolidation.py` | `kemi.memory.consolidation` |
| `src/kemi/audit.py` | `kemi.infra.audit` |
| `src/kemi/encryption.py` | `kemi.infra.encryption` |
| `src/kemi/webhooks.py` | `kemi.infra.webhooks` |
| `src/kemi/api_keys.py` | `kemi.infra.api_keys` |
| `src/kemi/background_tasks.py` | `kemi.infra.background_tasks` |
| `src/kemi/observability.py` | `kemi.infra.observability` |
| `src/kemi/decomposer.py` | `kemi.nlp.decomposer` |
| `src/kemi/topics.py` | `kemi.nlp.topics` |
| `src/kemi/summarizer.py` | `kemi.nlp.summarizer` |
| `src/kemi/reranker.py` | `kemi.nlp.reranker` |
| `src/kemi/graph.py` | `kemi.nlp.graph` |
| `src/kemi/cli_writer.py` | `kemi.interfaces.cli.writer` |

`core.py` already exists as a re-export shim pointing at `kemi.memory.facade` and `kemi.memory.service`.

`__init__.py` already imports from new paths (`kemi.memory.*`, `kemi.memory.entities`, etc.).

## 6. Public API impact

- `from kemi import <X>` — unchanged. `__init__.py` imports from new canonical paths.
- `from kemi.<old_module> import <Y>` — **no longer works** as of v0.4.0. The back-compat
  shims were removed. Use the canonical subpackage path instead (see section 5).
- `from kemi.<new_module> import <Y>` — canonical path; use in all code.

## 7. Out of scope

These directories are NOT moved:

- `pipeline/` — `ingestion.py`, `retrieval.py`, `_steps.py`
- `services/` — `read_service.py`, `write_service.py`, `admin_service.py`
- `operations/` — `_io.py`, `_ops_*.py`, `_query_cache.py`
- `plugins/` — `builtin.py`, `protocols.py`, `registry.py`
- `adapters/` — `storage/`, `embedding/`, `base.py`
- `integrations/` — `langchain.py`
- `exceptions.py` — stays in root, imported widely
- `__init__.py`, `__main__.py` — stay in root

## 8. Risk and rollback

- **26 files moved**. All 1539 tests are the safety net.
- If something breaks, moves can be reverted with `mv` (file moves are atomic).
- The main agent does incremental moves with test runs between each.
- Back-compat shims ensure zero downstream breakage during migration.
