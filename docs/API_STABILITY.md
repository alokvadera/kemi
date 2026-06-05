# Kemi API Stability Policy

This document describes Kemi's API stability promise, what is and isn't
covered, and the migration path for any breaking change.

## Stability tiers

| Tier | Examples | Stability |
|---|---|---|
| **Stable** | `Memory.remember`, `Memory.recall`, `Memory.forget`, `Memory.update`, `MemoryConfig`, `MemoryObject`, all enums, the CLI subcommands | Follows the rules below. No breaking changes without a deprecation cycle. |
| **Additive** | New optional parameters, new methods, new optional backends, new `MemoryType` values | Always backward-compatible. New code can use them; old code is unaffected. |
| **Experimental** | `MemoryType.PROCEDURAL`, webhook payload schema, `MemoryObject.agent_id` / `run_id` / `app_id` (multi-tenant tracing) | May change in minor versions. Documented as experimental in the docstring. |
| **Internal** | `kemi._memory_impl`, `kemi.operations._ops_*`, `kemi.core._QueryCache` shim, the `kemi.versions` module's internal classes | No stability promise. May change in any release. |

## Rules for `Stable` APIs

1. **No breaking signature changes without a deprecation cycle.** Removing a
   parameter, changing a parameter's type, renaming a public method, or
   changing a return type is a breaking change and requires:
   - A deprecation warning emitted at the old call site for at least one
     minor version.
   - A `CHANGELOG.md` entry under the next `## [Unreleased]` section.
   - Migration instructions in the deprecation message itself.

2. **New optional parameters are always safe.** Adding a new keyword
   argument with a default value is not a breaking change. Existing
   callers continue to work unchanged.

3. **New methods are always safe.** Adding a new method to `Memory` is
   non-breaking. It is, however, the maintainer's responsibility to
   ensure the new method doesn't shadow user-defined methods on a
   subclass.

4. **`MemoryObject` field policy.** Fields are **additive only**:
   - Adding a new field is non-breaking. Default values are used when
     loading older records from storage.
   - Renaming a field is breaking. Don't do it.
   - Removing a field is breaking. Don't do it.
   - Changing a field's type is breaking. Don't do it.
   - Exception: a field can be deprecated (kept but ignored, with a
     warning) for one minor version before removal.

5. **Enum value policy.** Adding a new enum value is non-breaking (it's
   essentially a new constant). Removing or renaming is breaking.

6. **Storage adapter contract.** A storage adapter that implements the
   `StorageAdapter` ABC and the search/recall contract continues to work
   across minor versions. New optional methods may be added to the ABC
   with a default `NotImplementedError` body; old adapters don't have to
   implement them.

## What is currently `Experimental`

### `MemoryType.PROCEDURAL`
Added in v0.4.0 to support "how-to" memories. The semantics around how
procedural memories interact with the recall pipeline (reranking,
consolidation, clustering) are still being finalised. Pin to
`MemoryType.EPISODIC` or `MemoryType.SEMANTIC` for production use.

### `MemoryObject.agent_id` / `run_id` / `app_id`
Added in v0.4.0 for multi-tenant tracing (which agent, which run, which
app produced/owns the memory). These fields are first-class on the
dataclass (not folded into `metadata`) so they can be indexed at the SQL
layer. The indexing strategy may evolve; for now they are simply stored
as columns.

If you need additional cross-cutting fields, request them via an issue.
Don't fold them into `metadata` — that breaks the index.

### Webhook payload schema (`kemi.webhooks.build_payload`)
The webhook payload is a JSON dict whose top-level keys may change as
we add more event types. Treat it as opaque on the consumer side.

## What is `Internal` and may change

- `kemi._memory_impl` — the implementation of `Memory`. Imported only
  by `kemi.core` and `kemi.operations`. Do not import directly.
- `kemi.operations._ops_*` — extracted free functions. Used by
  `kemi._memory_impl` to keep the orchestrator slim. May be reorganised
  freely.
- `kemi.core._QueryCache` — re-export shim. The canonical location is
  `kemi.operations._query_cache._QueryCache`. The shim may be removed
  in a future major version.
- `kemi.versions` dataclasses (`VersionSnapshot`, `DiffResult`,
  `RollbackResult`, `MemoryVersionStore`) — the shape of the
  *return values* is stable, but the class internals are not part of
  the API.

## Migration paths

If a breaking change is unavoidable, the deprecation message will look
like:

```
DeprecationWarning: `Memory.some_method` is deprecated and will be removed
in v0.7.0. Use `Memory.other_method` instead. See
https://github.com/alokvadera/kemi/blob/main/docs/API_STABILITY.md
for migration instructions.
```

## Versioning

Kemi follows [Semantic Versioning](https://semver.org/):

- **Patch** (0.4.1) — bug fixes only. No API change.
- **Minor** (0.5.0) — additive changes. New optional parameters, new
  methods, new backends. May include deprecation warnings for upcoming
  removals.
- **Major** (1.0.0) — breaking changes, after a deprecation cycle.

Kemi is currently in **0.x** (`0.4.0` at the time of writing). The 0.x
series reserves the right to make breaking changes with a single
minor-version deprecation cycle. Once `1.0.0` ships, the rules above
are binding.
