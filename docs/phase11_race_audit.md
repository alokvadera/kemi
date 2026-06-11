# Phase 11 — Versioning Race Audit

Documenting seven concurrency defects in the memory versioning subsystem.
All line numbers refer to `src/kemi/versions.py` and `src/kemi/operations/_io.py` as of the current commit.

---

## R1 — INSERT OR REPLACE clobbers concurrent pre-snapshots

**Location:** `src/kemi/operations/_io.py:169` (call site), `src/kemi/versions.py:305-360` (`record_before_update`)

**Severity:** High

**Summary:** `record_before_update` stores the pre-update snapshot with `INSERT OR REPLACE` keyed on `(memory_id, version)`. Because `update()` in `_io.py` calls `record_before_update` on a fresh connection that is **not** wrapped around `store.update()`, two concurrent updates for the same memory start from the same `memory.version`. Both threads therefore attempt to insert a pre-snapshot at the same version number. The second thread’s `INSERT OR REPLACE` silently overwrites the first thread’s pre-snapshot.

**Reproduction:**
1. Thread A reads memory `m` at `version=1`.
2. Thread B reads memory `m` at `version=1`.
3. Thread A calls `record_before_update` → inserts pre-snapshot at `v=1`, post-snapshot at `v=2`.
4. Thread B calls `record_before_update` → `INSERT OR REPLACE` overwrites A’s pre-snapshot at `v=1`, inserts post-snapshot at `v=3`.
5. Result: only three rows exist in `memory_versions` (one pre, two post) instead of four.

**Proposed fix:** Make the entire update + versioning sequence atomic. The simplest path is to perform `store.update()` and `record_before_update` inside the same `BEGIN IMMEDIATE` transaction (or to have `record_before_update` return the pre/post version numbers and let `_io.py` defer the store write until after the version numbers are reserved). Alternatively, change the pre-snapshot insert to a plain `INSERT` and catch the unique violation as a no-op (the pre-snapshot already exists from an earlier concurrent writer).

---

## R2 — Store row version drifts from version-store version after update

**Location:** `src/kemi/operations/_io.py:175` (`memory.version += 1`)

**Severity:** High

**Summary:** `record_before_update` already advances `memory_after.version` to the next free slot (e.g. `2`). `_io.py` then increments it again (`memory.version += 1`, e.g. to `3`) before calling `store.update()`. The store row therefore has a version number that does not exist in `memory_versions`, making the audit trail inconsistent.

**Reproduction:**
1. Create a memory at `version=1`.
2. Call `memory.update()` once.
3. `record_before_update` writes pre at `v=1`, post at `v=2`, and sets `memory.version = 2`.
4. `_io.py` executes `memory.version += 1` → `3`.
5. `store.update()` persists `version=3`.
6. `memory_versions` contains `v=1` and `v=2`; the store row claims `v=3`.

**Proposed fix:** Remove the redundant `memory.version += 1` in `_io.py:175`. `record_before_update` already assigns the correct next version to `memory_after.version`; the caller should trust that value and persist it directly.

---

## R3 — auto_prune_versions runs outside the version-store transaction

**Location:** `src/kemi/operations/_io.py:170` (`ctx.auto_prune_versions(memory_id)`)

**Severity:** Medium

**Summary:** `auto_prune_versions` is invoked immediately after `record_before_update` but before `store.update()`. It opens its own connection, lists all versions, and deletes the oldest ones. Because it is not inside the same `BEGIN IMMEDIATE` transaction as the snapshot recording, a concurrent prune can delete rows that another thread just inserted, or a concurrent writer can insert rows that the prune thread did not see.

**Reproduction:**
1. Thread A begins `update()`, calls `record_before_update` (commits pre+post snapshots).
2. Thread B begins `update()`, also calls `record_before_update` (commits its own pre+post).
3. Thread A proceeds to `auto_prune_versions`. It lists versions and sees four rows, but `max_versions_per_memory=3`.
4. Thread A deletes the oldest row — which happens to be Thread B’s pre-snapshot.
5. Result: Thread B’s pre-snapshot is lost even though Thread B’s update has not finished yet.

**Proposed fix:** Move `auto_prune_versions` inside the same transaction as `record_before_update`, or have `record_before_update` itself prune excess versions before committing. The prune must see the same snapshot of `memory_versions` that the insert sees.

---

## R4 — Concurrent rollbacks clobber each other's snapshot

**Location:** `src/kemi/versions.py:617-621` (`rollback`)

**Severity:** High

**Summary:** `rollback` computes the next version number with `get_latest_version_number()` (a read on a separate connection), then adds one, then calls `store.update()`, and finally opens yet another connection for the `INSERT OR REPLACE` into `memory_versions`. The read and the write are not atomic. Two threads rolling back to the same target can compute the identical `new_version`, causing one rollback’s version snapshot to overwrite the other’s. Depending on timing and SQLite locking, this can also surface as an `IntegrityError` when the second writer races the first.

**Reproduction:**
1. Memory has versions `1, 2, 3` in `memory_versions`; store row is at `v=3`.
2. Thread A calls `rollback(..., target_version=1)`.
3. Thread B calls `rollback(..., target_version=1)` simultaneously.
4. Both threads read `latest=3`, compute `new_version=4`.
5. Thread A writes store row `v=4`, then inserts snapshot at `v=4`.
6. Thread B writes store row `v=4`, then `INSERT OR REPLACE` overwrites A’s snapshot at `v=4`.
7. Only one rollback snapshot survives; the audit trail is missing a row.

**Proposed fix:** Acquire a single `BEGIN IMMEDIATE` transaction at the start of `rollback` and hold it until both `store.update()` and the `memory_versions` insert complete. Use `_next_version_number` (which already has collision detection) instead of the manual `get_latest_version_number() + 1` arithmetic.

---

## R5 — Rollback bypasses _next_version_number collision logic

**Location:** `src/kemi/versions.py:617` (`rollback`)

**Severity:** Medium

**Summary:** `rollback` re-implements version bumping as `get_latest_version_number() + 1` instead of calling `_next_version_number`, which already handles the collision case gracefully. This is a code-quality issue rather than a distinct race, but it means any fix for R4 should also eliminate this duplication.

**Reproduction:** Same as R4; the root cause is the ad-hoc version computation.

**Proposed fix:** Replace the manual `get_latest_version_number() + 1` logic with a call to `_next_version_number` inside the rollback transaction.

---

## R6 — verify_sequential_versions reads without transaction

**Location:** `src/kemi/versions.py:465-475` (`verify_sequential_versions`)

**Severity:** Low

**Summary:** `verify_sequential_versions` opens a connection, runs `SELECT version ... ORDER BY version ASC`, fetches all rows into a Python list, and then checks for a contiguous sequence. If a concurrent writer inserts or deletes a version between the `fetchall()` and the assertion, the check can return `False` for a database that is actually valid, or `True` for a database that has just become invalid.

**Reproduction:**
1. Thread A calls `verify_sequential_versions("m")`.
2. Thread A fetches rows `[1, 2, 3]`.
3. Thread B inserts version `4`.
4. Thread A computes `range(1, 4) == [1, 2, 3]` → `False`, incorrectly flagging the database as corrupt.

**Proposed fix:** Run the SELECT inside a read transaction (`BEGIN`) so the snapshot is isolated from concurrent writers.

---

## R7 — List/get version queries lack read transactions

**Location:** `src/kemi/versions.py:481-549` (`list_versions`, `get_version`, `get_latest_version_number`)

**Severity:** Low

**Summary:** All read helpers (`list_versions`, `get_version`, `get_latest_version_number`) open a connection, run one or more queries, and close. Without an explicit transaction, each query sees the committed state at the moment it executes. If a concurrent writer commits mid-way through a multi-query read (e.g. `get_version` doing two lookups), the results can be inconsistent.

**Reproduction:**
1. Thread A calls `list_versions("m")`.
2. SQLite reads rows `v=3, v=2`.
3. Thread B inserts `v=4` and commits.
4. SQLite continues reading and sees `v=1`.
5. Thread A returns `[3, 2, 1]` — note that `v=4` is missing because the cursor snapshot was taken before the insert, but the iteration is not atomic.
(While SQLite cursors are generally stable for a single SELECT, explicit `BEGIN` guarantees the snapshot.)

**Proposed fix:** Wrap each read helper in `BEGIN` … `COMMIT` (or simply open the connection and rely on SQLite’s auto-commit read isolation, which is sufficient for a single SELECT). For `list_versions` and other single-query methods, the current behavior is usually safe in practice; the fix is mostly defensive.

---

## Resolution status (Phase 11 close-out)

| Race | Severity | Status | Fix |
|------|----------|--------|-----|
| R1 | High | **Fixed** | Pre-snapshot insert changed from `INSERT OR REPLACE` to `INSERT OR IGNORE`; pre+post wrapped in a single `BEGIN IMMEDIATE` transaction. |
| R2 | High | **Fixed** | `_io.update` no longer does `memory.version += 1` after `record_before_update`; the post-version assigned inside the version-store transaction is used directly for the store write. |
| R3 | Medium | **Fixed (primary path)** | `record_and_update` (new method) does the prune step inside its own `BEGIN IMMEDIATE` transaction. The legacy fallback path in `_io.update` (only hit when `record_and_update` itself throws) still calls `ctx.auto_prune_versions`, which is not transactional — but this path is unreachable in normal operation. |
| R4 | High | **Fixed** | `rollback` now opens a single `BEGIN IMMEDIATE` transaction and uses `_next_version_number` (collision-safe) for the post-version, so two concurrent rollbacks serialise and never collide on the primary key. |
| R5 | Medium | **Fixed** | Subsumed by the R4 fix. |
| R6 | Low | **Fixed** | `verify_sequential_versions` now runs inside a read transaction (`BEGIN … COMMIT`) so the check sees a stable snapshot. |
| R7 | Low | **Fixed** | `list_versions`, `get_version`, and `get_latest_version_number` now use a `_read_transaction` context manager (plain `BEGIN`, not `BEGIN IMMEDIATE`) so reads give a consistent snapshot without blocking other readers. |

### New helpers added to `MemoryVersionStore`
- `_transaction()` — context manager that opens a connection, runs `BEGIN IMMEDIATE`, commits on success or rolls back on exception, and always closes the connection.
- `_read_transaction()` — same shape, but uses plain `BEGIN` so multiple readers can run concurrently.
- `_insert_snapshot()` — DRY helper for `INSERT` / `INSERT OR IGNORE` / `INSERT OR REPLACE` into `memory_versions`.
- `record_and_update(memory_before, memory_after, store, *, changed_by, keep_count)` — unified write path for `_io.update`. Runs the pre+post snapshots in one transaction, then writes the store, then prunes in a second transaction.

### Test coverage
`tests/test_versioning_race.py` now contains seven tests:
- `TestR1ConcurrentPreSnapshot` — two threads racing on `record_before_update` must both leave a distinct post-snapshot; the pre-snapshot is preserved.
- `TestR2VersionDrift` — after `memory.update()`, the store row's `version` equals the highest version in the version store.
- `TestR4ConcurrentRollback` — two threads rolling back to the same target produce two distinct rollback snapshots, with no `sqlite3.IntegrityError`.
- `TestR3AutoPruneInsideTransaction` — two tests proving prune runs inside the transaction and the surviving versions stay contiguous.
- `TestR6ReadStability` — `verify_sequential_versions` is stable under concurrent writes.
- `TestR7ListSnapshot` — `list_versions` never returns duplicate versions even under load.
