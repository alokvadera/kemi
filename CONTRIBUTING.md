# Contributing to Kemi

Thank you for your interest in contributing! This document covers the conventions we follow for changelog maintenance and release notes.

## Changelog Maintenance

We follow the [Keep a Changelog](https://keepachangelog.com/) format. Every user-facing change must include a corresponding entry in `CHANGELOG.md`.

### When to add an entry

Add a changelog entry when your PR or commit introduces a change that affects users of the library or CLI:

- **Breaking changes** — any change that could break existing code
- **Added** — new features, adapters, commands, or configuration options
- **Fixed** — bug fixes, race-condition patches, or correctness improvements
- **Changed** — refactors, performance improvements, or behavioural tweaks that are not breaking

Internal-only changes (pure test additions, CI-only changes, or comment/docstring updates with no user-visible effect) do **not** need a changelog entry.

### Where to add the entry

All unreleased changes go under the top-most heading:

```markdown
## [X.Y.Z] - Unreleased
```

If that heading does not exist yet, create it at the top of the file immediately below `# Changelog`. The version number should be the **next anticipated release** (usually a minor bump for new features or a patch bump for fixes).

### Entry style

- Write entries as full sentences ending with a period.
- Use back-ticks for code symbols (`Memory.recall`, `ScoreConfig`, `--json`).
- Keep the first line of an entry concise; add a second sentence only if the "why" is not obvious.
- Group entries under the correct category heading (`Breaking changes`, `Added`, `Fixed`, `Changed`).
- If an entry relates to a specific issue or PR, append the reference inline: `Fixed memory leak in batch recall (#123).`

Example:

```markdown
## [0.4.1] - Unreleased
### Added
- `ScoreConfig.from_memory_config()` classmethod for constructing scoring configs from `MemoryConfig`.

### Fixed
- SQLite migration now skips missing `expires_at` columns on legacy databases instead of crashing.
```

### Before a release

When cutting a release, the maintainer updates the top heading from `Unreleased` to the release date (`YYYY-MM-DD`) and creates a new empty `Unreleased` section above it:

```markdown
# Changelog

## [0.5.0] - Unreleased

## [0.4.0] - 2026-06-06
### Added
- ...
```

Do not retroactively edit the content of a shipped release section; if a post-release correction is needed, add it under the next release heading instead.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html). By participating, you are expected to uphold this code. Please report unacceptable behaviour to the project maintainers.

## Security Reporting

If you discover a security vulnerability in Kemi, please report it responsibly instead of opening a public issue.

- **Email**: security@kemi.dev *(or contact the maintainers directly if this address is not yet active)*
- **What to include**: A description of the vulnerability, the affected version(s), steps to reproduce, and any suggested mitigation.

We will acknowledge receipt within 48 hours and aim to provide a timeline for a fix within 7 days. Once the issue is resolved, we will publish a security advisory and credit the reporter (unless they wish to remain anonymous).
