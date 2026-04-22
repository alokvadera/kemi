# Changelog

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