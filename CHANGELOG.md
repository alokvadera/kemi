# Changelog

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