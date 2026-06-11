# Codebase Problems

Audit findings for the `kemi` codebase. Generated from a full sweep of `src/kemi/`,
`tests/`, and `pyproject.toml`. Issues are grouped by severity.

---

## Critical

### 1. Silent double-failure in `update()` — data integrity hazard
- **File:** `src/kemi/operations/_io.py:185-193`
- **Category:** Correctness / Data integrity
- **Why it matters:** If `record_before_update` and the manual fallback both fail, the
  memory is updated in the store with no version row written and no error raised.
  The `memory_versions` table silently diverges from the `memories` table.

### 2. `previous_state` passed to `memory.updated` webhook is the post-state
- **File:** `src/kemi/operations/_io.py:169-202`
- **Category:** Correctness / API contract
- **Why it matters:** Docstring says "capture pre-update state BEFORE mutating" but
  the capture happens after `memory.content`, `embedding`, `importance` are
  mutated. `previous_state == current_state` for every consumer.

### 3. Fernet key derived from a single unsalted SHA-256
- **File:** `src/kemi/infra/encryption.py:127-134`
- **Category:** Security
- **Why it matters:** No salt, no iteration count, no PBKDF2/scrypt/Argon2. A
  passphrase like `hunter2` is one rainbow table away from decrypted. The
  docstring acknowledges the simplification.

### 4. API keys hashed with plain unsalted SHA-256
- **File:** `src/kemi/infra/api_keys.py:85-86`
- **Category:** Security
- **Why it matters:** Should be `scrypt`, `argon2`, or `pbkdf2_hmac` with a unique
  per-row salt. With unsalted SHA-256, dumping the `api_keys.hashed_key` column
  is rainbow-tableable at billions of guesses per second.

### 5. Webhook writes never commit
- **File:** `src/kemi/infra/webhooks.py:178, 198, 254, 264`
- **Category:** Correctness / Data loss
- **Why it matters:** `with self._get_connection() as conn` closes the connection
  without `commit()`. Every `create()`, `update()`, `delete()` is silently rolled
  back. Connections are also opened+closed on every CRUD call (no pooling,
  ~5 syscalls per op).

### 6. Webhook payload leaks `user_id` to attacker-controlled URL (SSRF + PII)
- **File:** `src/kemi/infra/webhooks.py:96-127, 335-338`
- **Category:** Security
- **Why it matters:** Combined with the fact that `WebhookDispatcher.dispatch_sync`
  is called from `update()` / `forget()` (synchronous retry loop) and the URL is
  not validated (no scheme check, no localhost/private-IP block, no domain
  allow-list), a user with permission to register a webhook can register
  `http://internal-service/...` and exfiltrate other users' `memory_id` and
  `user_id`, and trigger SSRF.

### 7. `None.tzinfo` crash in read stream
- **File:** `src/kemi/services/read_service.py:194`
- **Category:** Correctness
- **Why it matters:** `memory.last_accessed_at.tzinfo` raises `AttributeError` when
  `last_accessed_at is None` (e.g. some ingestion paths leave it unset, or rows
  imported from a different store). The exception propagates through the
  generator and crashes the streaming call after the first yielded memory.

### 8. Resource leak per background task
- **File:** `src/kemi/infra/background_tasks.py:177-208, 259-299, 471-503`
- **Category:** Performance / Resource leak
- **Why it matters:** Each background task constructs a fresh `Memory()` opening
  a new SQLite connection, plus an embedder, plus a memory core. The connection
  is never closed. Under sustained load, the background thread leaks file
  descriptors until the process runs out.

### 9. `cancel_task` is a lie
- **File:** `src/kemi/infra/background_tasks.py:390-410`
- **Category:** Correctness / API contract
- **Why it matters:** Only `PENDING` tasks can be "cancelled". `RUNNING` tasks
  return `False` and the coroutine keeps executing. A `remember_many` of 10,000
  items will keep going despite a cancel request.

---

## High

### 10. Public `record_operation()` is a no-op
- **File:** `src/kemi/infra/observability.py:331-339`
- **Category:** Correctness / API contract
- **Why it matters:** Docstring says it records a completed operation with
  timing. Body is `pass`. Anyone calling `metrics.record_operation("remember",
  0.5)` thinks they're recording a metric. They are not.

### 11. No-commit schema init
- **File:** `src/kemi/adapters/storage/sqlite.py:137-200` and
  `src/kemi/adapters/storage/sqlite_vec.py:114-173`
- **Category:** Correctness
- **Why it matters:** DDL runs without a transaction and without commit. For a
  fresh DB this happens to work (autocommit + idempotent DDL), but on disk-full
  mid-`ALTER TABLE` the state is left inconsistent. `_transaction()` is
  available but unused.

### 12. Per-request DB write in API middleware
- **File:** `src/kemi/interfaces/api/app.py:497-538`
- **Category:** Performance / Correctness
- **Why it matters:** Every authenticated request does
  `BEGIN; UPDATE api_keys SET last_used_at = ?; COMMIT;` This serialises all
  requests through SQLite's RESERVED lock on `api_keys` and is a DoS vector. The
  shared connection across requests also causes `OperationalError: database is
  locked` under concurrency.

### 13. vec0 extension race
- **File:** `src/kemi/adapters/storage/sqlite_vec.py:101-110`
- **Category:** Correctness
- **Why it matters:** `_vec_loaded` is set on the instance, but
  `_get_connection()` is called per-thread (each thread creates a new
  connection). Thread 2 spawns first, loads extension, sets `_vec_loaded = True`;
  thread 1 spawns second, sees `_vec_loaded = True`, but its fresh connection
  never had the extension loaded. Thread 1 silently falls through to
  brute-force.

### 14. Test suite is broken when run together
- **Files:** `tests/` (suite-wide)
- **Category:** Correctness / Test reliability
- **Why it matters:** 77 failures + 34 errors out of ~1700 tests. Most are
  fixture-isolation / shared-global issues. Tests pass individually but fail
  as a suite because there is no `conftest.py`-level isolation for
  `_MemoryCore` singletons (`_api_key_manager`, `_task_manager`,
  `_global_collector`).
- **Additional:** `tests/test_vec_adapter.py::test_store_search_delete` fails
  deterministically in isolation: search returns the wrong memory (real
  ANN/HNSW bug).
- **Additional:** `tests/adapters/test_chroma.py` and
  `tests/adapters/test_qdrant.py` run by default with no `pytest.importorskip`
  guard.

### 15. `forget_many` / `aforget_many` skip hooks, webhooks, metrics, audit
- **File:** `src/kemi/operations/_io.py:366-393, 211-212`
- **Category:** Correctness / Compliance
- **Why it matters:** Batch deletes leave NO audit trail, NO webhook, NO metrics,
  NO hooks. A user calling `forget_many([100_ids])` deletes 100 memories
  untraceably. The same user calling `forget()` 100 times gets full
  instrumentation. Quiet compliance gap.

### 16. `bm25_score` is just TF, no IDF
- **File:** `src/kemi/memory/scoring.py:91-132`
- **Category:** Correctness / Quality
- **Why it matters:** Function is named `bm25_score` and the docstring claims
  BM25, but it has no IDF component — it's a normalised term-frequency sum.
  When `_io.py:1228` calls `bm25_score_corpus` only when
  `corpus and len(corpus) > 1`, single-candidate or empty-corpus calls silently
  degrade to this TF-only score. Silent quality regression in hybrid search.

### 17. Encryption silently disabled on bad env
- **File:** `src/kemi/memory/service.py:58-65`
- **Category:** Correctness / Security
- **Why it matters:** A misconfigured `KEMI_ENCRYPTION_KEY` (typo, wrong format,
  expired) results in encryption silently being turned off. No log line, no
  warning. An operator who set the env var to enable encryption will not know
  it failed.

### 18. Two-connection versioning
- **File:** `src/kemi/memory/versions.py:152-289, 419`
- **Category:** Correctness / Data integrity
- **Why it matters:** Version store opens a separate SQLite connection from the
  main store (defaults to the same file `~/.kemi/memories.db`). The two
  connections are independent transactions on the same file → different write
  locks. If `memory_versions` write succeeds and the main `memories` table
  write is then rolled back, the version history is now ahead of reality.

### 19. Type-incompatible `__getattr__` / `__setattr__` proxy
- **File:** `src/kemi/services/service.py:188-226`
- **Category:** Maintainability / Types
- **Why it matters:** "We don't know what's on the core, just proxy it" pattern
  that breaks IDE autocomplete, silently accepts typos, makes class contracts
  impossible to reason about, and accounts for 21+ `arg-type` mypy errors
  between `_MemoryCore` and `MemoryService`.

### 20. CI gates that nobody runs
- **File:** `pyproject.toml:104-156, 159, 60-69, 199-208`
- **Category:** Maintainability / Process
- **Why it matters:** `mypy strict = true` is set on every `kemi.*` module but
  174 mypy errors still exist. `--cov-fail-under=80` is enforced but actual
  coverage is ~9.4%. The `dev` extra containing `mypy`/`ruff` is not in
  `dependency-groups`, so `uv sync --group dev` does not install lint/typecheck
  tools.

---

## Medium

### 21. Test-isolation problems
- **Files:** `tests/` (suite-wide)
- **Category:** Test reliability
- **Why it matters:** No fixture isolation for `_api_key_manager`,
  `_task_manager`, `_global_collector` singletons. A test that creates
  `Memory()` in one file mutates globals seen by another.

### 22. Busy-wait loop for loop startup
- **File:** `src/kemi/infra/background_tasks.py:101-104`
- **Category:** Performance
- **Why it matters:** `while self._loop is None: time.sleep(0.01)` polls up to
  10/sec. Adds 0.01s latency to the hot path and ties up the calling thread.

### 23. `shutdown()` doesn't `join()` the thread
- **File:** `src/kemi/infra/background_tasks.py:360-370`
- **Category:** Correctness
- **Why it matters:** The daemon thread is dereferenced but never joined. The
  Python interpreter may shut down before in-flight tasks finish, dropping
  data mid-write. Setting `self._loop = None` before the thread has stopped is
  also unsafe (pending coroutines will raise `RuntimeError: Event loop is
  closed`).

### 24. Embedding round-trip silently returns `None` on corruption
- **File:** `src/kemi/memory/versions.py:43-64`
- **Category:** Correctness
- **Why it matters:** If the blob length is neither `% 4` nor `% 8`, the
  function returns `None` with no exception or log. The caller then has a
  `None` embedding where one existed. Semantic-search results become
  non-deterministic for that memory.

### 25. `Memory` deprecated alias kept around
- **File:** `src/kemi/memory/facade.py` and 20 other files
- **Category:** Maintainability
- **Why it matters:** The deprecation warning is emitted ~120 times during the
  test run. Every internal call has two valid entry points, the proxy
  `__getattr__`/`__setattr__` in `service.py` exists primarily to make this
  work, and the type system can't distinguish the two classes.

### 26. Adaptive-retrieval path swallows all errors
- **File:** `src/kemi/pipeline/retrieval.py`,
  `src/kemi/pipeline/_steps.py:150-154, 220-225`
- **Category:** Correctness
- **Why it matters:** `except Exception: logger.debug("...failed",
  exc_info=True)` swallows ALL errors (not just "expected" ones), and the user
  never sees the failure. If the adaptive retriever is the cause of every
  recall being 10x slower, there's no way to know without enabling DEBUG.

### 27. `OpenAIMemoryExtractor` swallows every error and returns `[]`
- **File:** `src/kemi/memory/formation.py:230-232`
- **Category:** Correctness
- **Why it matters:** A 200 OK response with garbage content silently produces
  zero candidates. A network error produces zero candidates. A 429 rate-limit
  error produces zero candidates. The user gets the same empty list in every
  case. Should distinguish failure modes.

### 28. `bm25_score_corpus` paths: see also #16.
- Covered in #16.

### 29. `_known_namespaces` samples 1000 rows and silently fails
- **File:** `src/kemi/operations/_io.py:526-541`
- **Category:** Correctness
- **Why it matters:** `get_all(limit=1000)` returns by `created_at DESC`; for a
  user with mixed namespaces the sampler can miss entire namespaces. The
  `except Exception: pass` also masks all backend errors. `prune_expired` and
  `backfill_entities` for `namespace=None` will silently leave data undeleted.

### 30. `get_history` / `diff_versions` / `rollback_memory` swallow all errors
- **File:** `src/kemi/operations/_io.py:1511-1561`
- **Category:** Correctness
- **Why it matters:** Same pattern as #26: `except (OSError,
  sqlite3.DatabaseError, AttributeError): logger.debug("get_history failed for
  %s", memory_id, exc_info=True); return []`. If the version store is broken,
  the user gets "no history" rather than an error — they have no way to know
  their data is silently lost.

### 31. Broken docstring in `get_memory_graph`
- **File:** `src/kemi/operations/_io.py:743-750`
- **Category:** Style / Quality
- **Why it matters:** Garbage copy-paste artifact in the docstring
  (`from kemi.nlp import graph`). Indicates a botched edit. Signals sloppy
  maintenance of this large file.

### 32. Async batch operations drop `metric_namespace`, webhooks, metrics
- **File:** `src/kemi/operations/_io.py:1333-1342, 387-393`
- **Category:** Correctness
- **Why it matters:** `aforget_many`, `arecall_since`, `arecall_by_tag` don't
  call `ctx.track_operation`, don't fire webhooks, don't increment metrics.
  Same compliance gap as #15 in async.

### 33. `import_from` doesn't validate schema
- **File:** `src/kemi/operations/_io.py:1476-1492`
- **Category:** Correctness
- **Why it matters:** A malformed export file (partial export from an older
  version, hand-edited) raises `KeyError: 'memory_id'` from inside a loop
  that has stored N-1 memories already. No schema validation or transactional
  rollback.

### 34. `OpenAI(api_key=api_key)` allows `api_key=None`
- **File:** `src/kemi/memory/formation.py:188-194`
- **Category:** Correctness / UX
- **Why it matters:** The OpenAI client silently looks for `OPENAI_API_KEY` in
  env. If also unset, the first `chat.completions.create()` returns a vague
  auth error. No upfront configuration validation.

### 35. `to_prometheus()` reads counters without the lock
- **File:** `src/kemi/infra/observability.py:60-62, 109-117, 145-148`
- **Category:** Correctness
- **Why it matters:** The `inc/observe/set` methods acquire `self._lock`, but
  the export methods read `self._value`, `self._bucket_counts`, etc. without
  it. A long scrape during heavy traffic will return torn or non-monotonic
  counter values.

### 36. Plugin protocol `PascalCase` aliased to `_XX` for isinstance checks
- **File:** `src/kemi/services/admin_service.py:264-301`
- **Category:** Style / Types
- **Why it matters:** N814 ruff violations on every `from kemi.plugins import X
  as _X`. Pattern repeated 4 times. Indicates the plugin protocol and the
  `add_*` methods are working against the type system, not with it.

### 37. `has_sentiment_flip` is dead code
- **File:** `src/kemi/memory/dedup.py:62-83`
- **Category:** Quality
- **Why it matters:** Entire function body wrapped in `# pragma: no cover
  (edge case)`. Called from `find_duplicates` (line 108) which is also
  wrapped. Never exercised in tests. The `SENTIMENT_SHIFT_PAIRS` list is
  hand-curated, and the negation detection is by word presence (so "I never
  had a not unpleasant moment" flips). Ships but doesn't actually run.

### 38. Tags CSV escape: see also #34.
- **File:** `src/kemi/adapters/storage/sqlite.py:50`
- **Category:** Correctness
- **Why it matters:** `[t.replace("\\,", ",") for t in tags_csv.split(",")]`
  rule is wrong if the input ever has `\\,` (backslash followed by comma).
  Tests don't exercise this.

### 39. `_QueryCache` shim uses `__new__` for proxy
- **File:** `src/kemi/services/service.py:455-468`
- **Category:** Maintainability
- **Why it matters:** Hack to preserve a deprecated import path. The new
  location is canonical, the shim should just be removed.

### 40. 67 `except Exception:` blocks across the codebase
- **Files:** multiple (notable: `src/kemi/memory/versions.py:198, 220`)
- **Category:** Maintainability
- **Why it matters:** Broad catches that hide bugs. Notably the rollback path
  in `versions.py` swallows ALL exceptions in `_transaction()` rollback,
  including the rollback failure itself.

---

## Recommended first fixes

Highest impact, lowest blast radius:

1. **#1** silent update failure — data integrity
2. **#2** `previous_state` is post-state — webhook contract
3. **#3** single-SHA256 KDF — security
4. **#5** no commit on webhook writes — data loss
5. **#7** `None.tzinfo` crash — correctness
6. **#15** `forget_many` audit gap — compliance

---

# Second-Pass Audit (deeper sweep)

Areas the first pass under-covered or missed entirely. Items C1–C17 are new
Critical/High findings; M18–M25 are Medium; T26–T27 are Tests; D28–D30 are
Docs/Process.

---

## Critical (2nd pass)

### C1. Webhook signature body mismatch — receivers will always fail HMAC verification
- **Files:** `src/kemi/infra/webhooks.py:141` (sign) vs `:354, :425` (send)
- **Category:** Security / Correctness
- **Why it matters:** `sign_payload()` computes HMAC over
  `json.dumps(payload, sort_keys=True, separators=(",",":"))`, but
  `dispatch_sync()` and `dispatch_async()` send `json.dumps(payload)` — different
  key order, different whitespace. The `X-Kemi-Signature` header is computed
  over a body that is *never sent*. Any honest receiver will reject every
  webhook as forged. Webhook signing is currently a no-op for security.
- **Fix:** Pass the canonical body bytes to the dispatcher; or remove the
  canonicalization from `sign_payload`.

### C2. SSRF via webhook URL — no scheme/host validation
- **Files:** `src/kemi/infra/webhooks.py:192`,
  `src/kemi/interfaces/cli/main.py:1017`,
  `src/kemi/interfaces/api/app.py:1827`
- **Category:** Security
- **Why it matters:** A caller can register
  `http://169.254.169.254/latest/meta-data/iam/...`,
  `http://localhost:6379/...`, `file:///etc/passwd`, or any internal HTTP
  service. The dispatcher will POST memory content to whatever URL is given.
- **Fix:** Validate scheme is `https://` (or `http://` with explicit opt-in),
  block link-local/loopback/private CIDRs, reject non-HTTP schemes.

### C3. Two admin FTS endpoints have no authorization check at all
- **File:** `src/kemi/interfaces/api/app.py:1152` (`/admin/fts/rebuild`) and
  `:1251` (`/admin/fts/verify`)
- **Category:** Security
- **Why it matters:** These endpoints mutate the full-text search index and
  (in verify's auto-repair branch) DELETE rows from `memories_fts`. They sit in
  the `/admin/*` namespace, are rate-limit-exempt, never call
  `_resolve_user_id`, and check neither API key nor `KEMI_API_KEY_REQUIRED`.
  Any anonymous caller in default-config mode can rebuild the index or wipe
  orphaned FTS rows.
- **Fix:** Require auth (or a separate admin token) and rate-limit; gate
  auto-repair behind a `verify_only` default.

### C4. `/memories/{memory_id}/history` leaks any user's version history
- **File:** `src/kemi/interfaces/api/app.py:1776`
- **Category:** Security
- **Why it matters:** No `existing = mem._store.get(memory_id)` lookup, no
  `_resolve_user_id`, no comparison to `authed`. In default config
  (`KEMI_API_KEY_REQUIRED=false`) any unauthenticated caller can read full
  version history — including prior `content` snapshots — of any memory by ID.
- **Fix:** Look up the memory, confirm the requester owns it, then return
  history.

### C5. `FernetEncryptor` derives its key via a single unsalted SHA-256
- **File:** `src/kemi/infra/encryption.py:130-134`
- **Category:** Security
- **Why it matters:** `digest = hashlib.sha256(key.encode("utf-8")).digest()` —
  one round, no salt, no iteration count. If a user passes a passphrase, it is
  brute-forceable at GPU rates (10⁹+ guesses/sec on a single GPU). This makes
  the "encrypted at rest" promise hollow for any non-random key.
- **Fix:** Use PBKDF2-HMAC-SHA256 (≥600k iters) or scrypt/Argon2; or document
  that the input must already be a random 32-byte key.

### C6. `BackgroundTaskManager` workers create a *different* `Memory` instance
- **Files:** `src/kemi/infra/background_tasks.py:177`, `:259`, `:472`
- **Category:** Correctness
- **Why it matters:** `_run_embed_batch`, `_run_rebuild_fts`, and
  `_run_ttl_sweep` each call `Memory()` with default config — not the Memory
  the API request was served by. The DB path comes from default env (not the
  custom store the user passed to `create_app(memory=...)`), the embedding
  adapter is fastembed (not the user's OpenAI/custom), the encryption config
  is ignored, the metrics collector is a fresh singleton. Background writes
  effectively target a different database. A user who configured
  `Memory(store=PostgresStorageAdapter(...))` will have `embed_batch` writes
  silently go to the local SQLite.
- **Fix:** Pass the active `Memory` (or its core deps) into `submit_*` and
  reuse it on the worker thread.

### C7. CI matrix runs Python 3.9, but pyproject requires >=3.10
- **Files:** `.github/workflows/ci.yml:8`, `pyproject.toml:11`
- **Category:** Process
- **Why it matters:** `matrix: ["3.9", "3.10", "3.11", "3.12"]` vs
  `requires-python = ">=3.10"`. Every push on 3.9 fails (or has been silently
  broken). CHANGELOG 0.1.0 brags "CI pipeline (Python 3.9-3.12)" — that
  capability has been removed.
- **Fix:** Drop 3.9 from the matrix (or relax `requires-python`).

### C8. CI installs `[dev]` but `dependency-groups.dev` is missing ruff and mypy
- **Files:** `pyproject.toml:60-69` vs `:198-208`, `.github/workflows/ci.yml:14`
- **Category:** Process
- **Why it matters:** `[project.optional-dependencies].dev` lists ruff/mypy;
  `[dependency-groups].dev` does not. CI runs `pip install -e ".[dev]"` so
  today it works, but anyone using `uv sync` gets no lint/typecheck tools and
  the two definitions drift on every dev-dep change. Two "dev" sections is a
  footgun.
- **Fix:** Pick one (PEP 735 `dependency-groups` is the modern way) and
  remove the other.

---

## High (2nd pass)

### C9. `_OperationTracker` attributes unknown operations to `embed_latency` histogram
- **File:** `src/kemi/infra/observability.py:493-496`
- **Category:** Correctness
- **Why it matters:** `__exit__` does
  `getattr(self._collector, hist_name, self._collector.embed_latency)`.
  Operations not in `_HISTOGRAM_MAP` ("prune", "consolidate", "export",
  "import", "migrate", "feedback", anything user-added) silently get reported
  as embedding latency. Prometheus dashboards built on this data are wrong.
- **Fix:** Add a dedicated `_other_latency` histogram, or raise on unknown
  operations.

### C10. `MetricsCollector` is a process-wide singleton shared by every `Memory`
- **Files:** `src/kemi/infra/observability.py:506-520`,
  `src/kemi/memory/service.py:97-101`
- **Category:** Correctness / Tests
- **Why it matters:** `get_metrics_collector()` returns the same instance to
  every `Memory` ever constructed. Two `Memory()` objects in the same process
  will report merged counters (and there is already a known `xfail` test in
  `tests/test_core.py:252` whose reason literally reads "Singleton metrics
  collector state bleed between tests — needs isolation fix"). Anyone running
  multi-tenant will get cross-talk.
- **Fix:** Per-instance collectors (a `MetricsCollector` attribute on
  `_MemoryCore`), with a global default opt-in.

### C11. N+1 query in `/admin/users`
- **File:** `src/kemi/interfaces/api/app.py:1903-1912`
- **Category:** Performance
- **Why it matters:** For each user returned by `list_users()` the endpoint
  calls `store.count(uid)` (1 round-trip per user) and `get_last_active(uid)`
  (another per user). With 10k users that's 20k queries. Also the
  `getattr(store, "get_last_active", lambda _u: None)` default lambda is
  *never* used if the attribute exists — it doesn't actually guard against
  `get_last_active` raising with a different signature.
- **Fix:** `SELECT user_id, COUNT(*), MAX(last_accessed_at) FROM memories GROUP
  BY user_id`.

### C12. N+1 query in `prune_expired`
- **File:** `src/kemi/operations/_io.py:603-635`
- **Category:** Performance
- **Why it matters:** For every user, calls `get_all_by_user(uid,
  namespace=ns)` for every known namespace, then in Python scans each memory.
  With M users × N namespaces this is O(M·N) round-trips loading entire
  result sets into Python.
- **Fix:** `SELECT memory_id FROM memories WHERE expires_at <= ? AND
  lifecycle_state IN (...)` — single query.

### C13. `submit_*` race allows exceeding `max_concurrent_tasks`
- **Files:** `src/kemi/infra/background_tasks.py:132-156` (and analogous
  blocks at `:222-247`, `:430-455`)
- **Category:** Correctness
- **Why it matters:** The `_running_count >= _max_concurrent` check is inside
  `with self._lock`, but the counter is **not** incremented before the lock
  is released and the coroutine is scheduled. N threads can each pass the
  check and submit; the bound is advisory at best.
- **Fix:** Increment `_running_count` under the lock; decrement in the task
  body. (Same race in all three `submit_*` methods.)

### C14. `BackgroundTaskManager.shutdown()` leaks the daemon thread
- **File:** `src/kemi/infra/background_tasks.py:360-370`
- **Category:** Resource management
- **Why it matters:** `shutdown()` schedules `loop.stop`, nulls
  `self._loop` and `self._thread`, and returns — but never joins the thread,
  and `_loop.stop()` doesn't cancel running tasks or close the loop. The
  daemon thread is left running, the loop is left in a stopped state with no
  one to close it. A subsequent `submit_*` creates a second loop on a second
  thread while the first is still alive.
- **Fix:** `self._thread.join(timeout=...)`, then `self._loop.close()`.

### C15. CI excludes `tests/test_api_integration.py` from the 80% coverage gate
- **File:** `.github/workflows/ci.yml:17`
- **Category:** Tests
- **Why it matters:** The integration test (`pytest.mark.slow`, 405 LOC) is
  the only file that exercises the real FastAPI app end-to-end. By excluding
  it from `--cov-fail-under=80`, every other test can pass even if the entire
  API surface breaks.
- **Fix:** Run the integration tests on a non-default Python version with a
  separate coverage threshold, or speed them up and include in the main
  matrix.

### C16. `Memory(...)` emits `DeprecationWarning` on *every* construction
- **File:** `src/kemi/memory/facade.py:36-43`
- **Category:** Correctness / UX
- **Why it matters:** The warning is unconditional, fired inside `__init__`
  with `stacklevel=2`. Frameworks, test suites, and any code path that
  constructs a `Memory` once per request (a common pattern in agent loops)
  will spam user logs. The README still teaches `from kemi import Memory` as
  the primary entry point, so the warning fires on the documented happy path.
- **Fix:** Only emit when `warnings.filterwarnings` hasn't suppressed, or push
  the warning to a once-per-process hook, or remove it now that the
  deprecation period is over.

### C17. API server default auth off is a footgun
- **Files:** `src/kemi/interfaces/api/app.py:77` (`KEMI_API_KEY_REQUIRED=false`
  default), `README.md` Quickstart
- **Category:** Security / Docs
- **Why it matters:** The server starts completely open by default. New users
  running the `kemi api-server` CLI in any non-dev environment will expose
  every endpoint unauthenticated because they didn't know to set the env
  var. The README and the FastAPI app.py docstring both note the default but
  a "fail-closed" default would be safer.
- **Fix:** Default to `KEMI_API_KEY_REQUIRED=true` if
  `KEMI_API_KEYS_BOOTSTRAP` is set; emit a startup warning if it's false and
  `KEMI_TRUSTED_HOSTS` is not set.

---

## Medium (2nd pass)

### M18. `_io.py` is 1607 lines / 46 top-level functions across 7 unrelated concerns
- **File:** `src/kemi/operations/_io.py`
- **Category:** Maintainability
- **Why it matters:** The module docstring says "Direct memory I/O operations"
  but the file contains `update`, `forget`, `context_block`, `stats`,
  `list_users`, `prune`, `prune_expired`, `consolidate`, `cluster_topics`,
  `extract_entities`, `get_memory_graph`, `feedback`, `backfill_entities`,
  `run_maintenance`, `recall_*` (10+ variants), `migrate`, `export`,
  `import_from`, `get_history`, `diff_versions`, `rollback_memory`. Six of
  these have *no* I/O at all (recall/migrate operate on in-memory objects).
  The naming is misleading and the file is over the 1500-line readability
  cliff.
- **Fix:** Split by domain: `_crud.py`, `_recall.py`, `_maintenance.py`,
  `_export_import.py`, `_versioning_io.py`.

### M19. `KEMI_PROJECT_CONTEXT.md` describes the pre-Phase-12 flat layout
- **File:** `KEMI_PROJECT_CONTEXT.md:7, :287-319`
- **Category:** Docs
- **Why it matters:** Lists `kemi/scoring.py`, `kemi/dedup.py`,
  `kemi/lifecycle.py`, `kemi/topics.py`, `kemi/graph.py`,
  `kemi/consolidation.py`, `kemi/chunker.py`, `kemi/decomposer.py`,
  `kemi/reranker.py`, `kemi/versions.py`, `kemi/webhooks.py`,
  `kemi/api_keys.py`, `kemi/background_tasks.py`, `kemi/adaptive.py`,
  `kemi/audit.py`, `kemi/encryption.py`, `kemi/observability.py`,
  `kemi/api_server.py`, `kemi/mcp_server.py`, `kemi/cli.py` — all of which
  moved to subpackages in Phase 12. Says the project is version 0.3.0
  (pyproject is 0.4.0). It is the "authoritative architecture overview" but
  it does not describe the actual architecture.
- **Fix:** Update or delete; point to `docs/phase12_layout.md` and
  `docs/ARCHITECTURE.md`.

### M20. `prune()` skips the `min_importance` check after matching `max_age_days`
- **File:** `src/kemi/operations/_io.py:566-577`
- **Category:** Correctness
- **Why it matters:** The loop has
  `if max_age_days is not None: ... if age_days > max_age_days:
  to_delete.append(...); continue`. When `max_age_days` triggers, the
  `min_importance` check is skipped via `continue`. A caller passing
  `max_age_days=30, min_importance=0.9` will delete 30-day-old memories even
  if they're 0.99 important. The `continue` is intentional-looking but
  produces a surprising precedence rule. There is no test for "both filters
  together".
- **Fix:** Drop the `continue`; or document the precedence explicitly in the
  docstring + a test.

### M21. `aforget_many` and `forget_many` skip event hooks — inconsistent with `forget`
- **File:** `src/kemi/operations/_io.py:366-393` vs `311-363`
- **Category:** Correctness
- **Why it matters:** Single `forget` calls pre+post hooks + dispatches a
  webhook per delete. `forget_many` (sync) explicitly skips hooks "for batch
  performance". `aforget_many` (async) uses `asyncio.gather` and also skips
  hooks. Result: batch deletes silently bypass user-registered
  `add_event_hook("pre", "forget", ...)` callbacks. Hook authors have no
  warning.
- **Fix:** At minimum, fire one pre+post hook for the batch with `count=`;
  document the perf choice in the docstring.

### M22. Webhook secrets stored in plaintext in SQLite
- **File:** `src/kemi/infra/webhooks.py:155` (DDL
  `secret TEXT NOT NULL DEFAULT ''`)
- **Category:** Security
- **Why it matters:** Anyone with read access to the kemi DB (or a backup)
  gets the HMAC secret for every webhook destination. Combined with the
  plaintext log of webhook IDs in the audit log, an attacker can forge
  events to all configured endpoints.
- **Fix:** Hash at rest (Argon2/bcrypt) or encrypt with the user's
  `KEMI_ENCRYPTION_KEY`.

### M23. CLI webhook commands hardcode `~/.kemi/memories.db`, ignoring `KEMI_DB_PATH`
- **File:** `src/kemi/interfaces/cli/main.py:1021, 1050, 1078`
- **Category:** Correctness
- **Why it matters:** `db_path = os.path.expanduser("~/.kemi/memories.db")` —
  the rest of the codebase honors `KEMI_DB_PATH` (and `Memory` does too). On
  a system with multiple kemi instances in different locations, the CLI's
  webhook management will silently target the wrong database.
- **Fix:** Use `os.environ.get("KEMI_DB_PATH", ...)` everywhere.

### C24. 20+ `E302` violations in CLI — ruff should fail CI but doesn't
- **File:** `src/kemi/interfaces/cli/main.py` (lines 428, 505, 532, 598, 620,
  624, 640, 644, 660, 707, 719, 773, 786, 807, 825, 848, 888, 921, 981, 1017)
- **Category:** Process / Style
- **Why it matters:** `pyproject.toml` enables
  `select = ["E", ...]`, which includes `E302`. Either ruff is being
  skipped, the `src/` glob in CI is missing `interfaces/cli/`, or
  `per-file-ignores` is silently turning it off. Strong evidence the lint
  gate is bypassed.
- **Fix:** Run `ruff check src/ tests/` locally and investigate why these
  aren't surfacing.

### M25. `_check_cache` and `_cache_results` rebuild the cache key twice per recall
- **Files:** `src/kemi/pipeline/retrieval.py:226, 352`,
  `src/kemi/operations/_query_cache.py:28-55`
- **Category:** Performance
- **Why it matters:** `_make_key` is called with 9 arguments (including a
  `tuple(sorted(...))` over the lifecycle enum values) twice on every
  cache-miss path. The key string is identical the second time. Plus
  `_make_key` is private (`_` prefix) and called from the pipeline, which is
  fragile coupling.
- **Fix:** Compute the key once and pass to both methods; make `_make_key`
  part of the protocol.

---

## Tests (2nd pass)

### T26. `xfail` test for known metrics-bleed issue left in the suite
- **File:** `tests/test_core.py:252-254`
- **Category:** Tests
- **Why it matters:** `pytest.mark.xfail(reason="Singleton metrics collector
  state bleed between tests — needs isolation fix")` — the issue is
  identified but not fixed, so any future change that depends on per-test
  metric values will silently no-op. Combined with C10, this means
  `MetricsCollector` is effectively untestable.
- **Fix:** Resolve the singleton (C10), then remove the `xfail` marker.

### T27. 19 test functions with zero assertions
- **Files:** e.g. `tests/test_lifecycle.py:118-138`,
  `tests/test_pipeline_steps.py:77, 80, 89, 193, 196`,
  `tests/test_main.py:97`, `tests/test_versions.py:107`,
  `tests/test_core.py:69`
- **Category:** Tests
- **Why it matters:** They have a `pass` body or call functions whose return
  is ignored. They will never fail regardless of the code being tested. Most
  are labeled "no-op" in the name — they exist to ensure import works. They
  should be removed or upgraded.
- **Fix:** Remove the no-op tests, or add a real assertion.

---

## Docs / Process (2nd pass)

### D28. CHANGELOG 0.4.0 calls `MemoryService` the new API but it isn't in the README
- **Files:** `CHANGELOG.md:60-64` ("Phase 8: facade split — `MemoryService` is
  now a 478-LOC delegation shell"), `README.md` (no mention)
- **Category:** Docs
- **Why it matters:** The CHANGELOG treats `MemoryService` as the canonical
  entry point. The README still exclusively documents `Memory`. A user
  reading README + CHANGELOG gets no clear path to the new API.
- **Fix:** Update README Quick Start to show
  `from kemi import MemoryService`; demote `Memory` to "compatibility shim".

### D29. No pre-commit hooks; quality gates exist only on CI
- **File:** repo root (no `.pre-commit-config.yaml`)
- **Category:** Process
- **Why it matters:** With 20+ ruff violations already in
  `interfaces/cli/main.py` and 40+ `# type: ignore` in `src/`, an
  editor/lint-on-save gate would have caught most of these at write time. CI
  catches them only after push.
- **Fix:** Add a `.pre-commit-config.yaml` with `ruff`, `ruff-format`,
  `mypy`, and the same `pytest` smoke test.

### D30. `kemi/__init__.py` fallback `__version__ = "0.3.0"` is wrong
- **File:** `src/kemi/__init__.py:6`
- **Category:** Docs
- **Why it matters:** `pyproject.toml` declares 0.4.0, the API server banner
  shows 0.3.0 (`app.py:470`), the CHANGELOG says 0.4.0, but the
  import-fallback path in `__init__.py` still says 0.3.0. The hardcoded
  literal will silently desync again next release.
- **Fix:** Drop the fallback constant or generate it at build time.

---

## Top-of-list from 2nd pass (correctness/security, low blast radius)

1. **C1** webhook signature body mismatch — every signed webhook is forged-looking
2. **C2** SSRF via webhook URL — no scheme/host validation
3. **C3** `/admin/fts/*` endpoints have no auth at all
4. **C4** `/memories/{id}/history` leaks any user's history
5. **C6** background workers create a *different* `Memory` instance — data goes to wrong DB
6. **C9** unknown operations silently attributed to `embed_latency`
7. **C10** `MetricsCollector` singleton → multi-tenant cross-talk + known xfail
8. **C13** `submit_*` race exceeds `max_concurrent_tasks`
9. **C16** `Memory()` deprecation warning fires on every construction
