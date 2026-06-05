# Changelog

## [0.4.0] - Unreleased
### Breaking changes
- `pip install kemi` now ships **zero hard dependencies** (`chromadb` and
  `qdrant-client` moved to optional `[chroma]` and `[qdrant]` extras).
  This is the largest install-size reduction since v0.1.0.
- Removed `src/kemi/versioning.py` (639 LOC) and the orphan test suite
  `tests/test_versioning_new.py` (37 tests). These were duplicates of
  the canonical `src/kemi/versions.py` module and were never imported
  by production code.
- Internal: `src/kemi/core.py` was renamed to `src/kemi/_memory_impl.py`
  to make room for the new `src/kemi/operations/` subpackage. The public
  `kemi.core.Memory` re-export is preserved, so `from kemi.core import
  Memory` continues to work. This is invisible to users.

### Added
- **API stability promise** — see `docs/API_STABILITY.md`. Public
  classes (`Memory`, `MemoryObject`, `MemoryConfig`, enums) follow a
  no-breaking-changes-without-deprecation-cycle policy.
- **CLI `--json` and `--quiet` flags** — every `kemi` subcommand
  supports machine-readable JSON output (one object per line) and a
  silent mode that suppresses info output but keeps errors. Default
  behaviour remains human-readable.
- **New `kemi.operations` subpackage** — extracted operations from
  the monolithic `core.py`. Public API unchanged; internal structure
  is now `Memory` (orchestrator) + free functions in
  `kemi/operations/_ops_*.py`.
- **Coverage report configuration** — `pyproject.toml` now documents
  why optional adapters are excluded (their tests only run when the
  corresponding dependency is installed). The `92%` badge now clearly
  refers to the *core path* (SQLite + fastembed), not a global figure.
- **Honest failure messages** — broad `except Exception` clauses in
  the core orchestrator now have explanatory comments. Where the
  catch is unavoidable (storage adapters, external HTTP calls), the
  rationale is documented inline.
- **Chunker edge-case tests** — added 7 regression tests for
  abbreviation handling (Dr., Mr., Prof., e.g.), decimal numbers
  (3.14), ellipsis (...), and the short-fragment merge logic.

### Fixed
- **Chunker sentence boundary detection** — `_is_sentence_boundary`
  now checks if the first word of the previous sentence is an
  abbreviation, fixing false-positive boundaries after "Dr. Smith",
  "Mr. Jones", and similar. `split_into_sentences` no longer merges
  complete short sentences (e.g. "First sentence.") into the next
  one, only true fragments without terminators.
- **SQLite migration on legacy databases** — the
  `idx_memories_expires_at` index is now created in a `try/except`
  so legacy databases (schema versions 2-6) initialise cleanly
  instead of crashing with `no such column: expires_at`.

## [0.3.0] - 2026-04-19
### Added
- CLI — kemi list, recall, forget, export, import, stats commands
- Installable as console script: pip install kemi then use kemi command directly

## [0.2.1] - 2026-04-19
### Fixed
- Replaced deprecated asyncio.get_event_loop() in all async methods
- Added full test coverage for export() and import_from()
- CustomStorageAdapter.get_all() now raises clear NotImplementedError instead of silently crashing

## [0.2.0] - 2026-04-19
### Added
- MCP server (kemi[mcp]) — expose kemi as an MCP tool server, startable with: python -m kemi.mcp_server
- Export/Import — memory.export("backup.json") and memory.import_from("backup.json") with async versions
- LangChain adapter (kemi[langchain]) — KemiMemory class with save_context() and load_memory_variables()
- First-run download warning — clear stderr message before 130MB model download

## [0.1.7] - 2026-04-06
### Added
- Input validation with descriptive error messages on all core methods
- Embedding dimension mismatch detection in recall()
- Default database path moved to ~/.kemi/memories.db

## [0.1.6] - 2026-04-05
### Added
- Async methods: aremember(), arecall(), aforget(), acontext_block()
- FastAPI async example in README and docs

## [0.1.5] - 2026-04-05
### Fixed
- Sentiment flip detection — "I love coffee" and "I hate coffee" no longer incorrectly merged

## [0.1.4] - 2026-04-04
### Fixed
- fastembed numpy array conversion — embeddings now correctly converted to Python lists

## [0.1.3] - 2026-04-04
### Fixed
- Source field deserialization bug in SQLite and JSON adapters

## [0.1.2] - 2026-04-04
### Fixed
- SQLite in-memory database support via shared connection

## [0.1.1] - 2026-04-03
### Added
- Coverage exclusions for untestable optional adapters
- Ruff linting fixes

## [0.1.0] - 2026-04-03
### Added
- Initial release
- remember(), recall(), forget() core methods
- FastEmbed local embeddings (default)
- SQLite storage (default)
- JSON storage adapter
- Custom embedding and storage adapters
- Semantic deduplication with dual-threshold conflict detection
- Importance-weighted scoring with temporal decay
- Lifecycle state management
- Optional prompt injection sanitization
- Full test suite (95 tests, 94% coverage)
- CI pipeline (Python 3.9-3.12)
