#!/usr/bin/env python3
"""Test what HNSW parameters sqlite-vec actually supports.

Tries multiple syntaxes found in different sources to discover what works.
"""

import sqlite3
import sqlite_vec


def try_syntax(label, sql, conn):
    try:
        conn.execute(sql)
        print(f"  ✅ {label}")
        return True
    except sqlite3.OperationalError as e:
        print(f"  ❌ {label}: {e}")
        return False


def main():
    print("=" * 60)
    print("  sqlite-vec HNSW Parameter Discovery")
    print("=" * 60)

    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    # Get version
    version = db.execute("select vec_version()").fetchone()[0]
    print(f"\n  Version: {version}\n")

    # Test 1: Basic vec0 table (no params) - baseline
    print("Test 1: Basic table (no params)")
    try_syntax("basic", "CREATE VIRTUAL TABLE t1 USING vec0(embedding float[4])", db)

    # Test 2: Syntax from documentation - hnsw(...) block in table def
    print("\nTest 2: HNSW block in table definition")
    try_syntax(
        "hnsw() after column",
        "CREATE VIRTUAL TABLE t2 USING vec0(embedding float[4], hnsw(m=16))",
        db,
    )

    # Test 3: HNSW block with ef_construction
    print("\nTest 3: HNSW with ef_construction")
    try_syntax(
        "hnsw(m=16, ef_construction=200)",
        "CREATE VIRTUAL TABLE t3 USING vec0(embedding float[4], hnsw(m=16, ef_construction=200))",
        db,
    )

    # Test 4: Parameters as string-valued options
    print("\nTest 4: Parameters as string options")
    try_syntax(
        "m=16 as string",
        "CREATE VIRTUAL TABLE t4 USING vec0(embedding float[4], m='16')",
        db,
    )

    # Test 5: Parameters as table-level options (FTS5 style)
    print("\nTest 5: Table-level comma-separated options")
    try_syntax(
        "comma options",
        "CREATE VIRTUAL TABLE t5 USING vec0(embedding float[4], m=16, ef_construction=200)",
        db,
    )

    # Test 6: Different column order
    print("\nTest 6: Different orders")
    try_syntax(
        "hnsw block first",
        "CREATE VIRTUAL TABLE t6 USING vec0(hnsw(m=24, ef_construction=300), embedding float[4])",
        db,
    )

    # Test 7: Try WITH-style parameters
    print("\nTest 7: Query-time parameters")
    db.execute("CREATE VIRTUAL TABLE t7 USING vec0(embedding float[4])")
    db.execute("INSERT INTO t7(rowid, embedding) VALUES (1, '[0.1,0.2,0.3,0.4]')")
    db.execute("INSERT INTO t7(rowid, embedding) VALUES (2, '[0.9,0.8,0.7,0.6]')")
    try_syntax(
        "MATCH with k parameter",
        "SELECT rowid, distance FROM t7 WHERE embedding MATCH '[0.12,0.22,0.32,0.42]' AND k=10 ORDER BY distance LIMIT 2",
        db,
    )

    # Test 8: Try ef_search as pragma
    print("\nTest 8: Setting ef_search")
    try_syntax(
        "PRAGMA vec_ef_search=200",
        "PRAGMA vec_ef_search=200",
        db,
    )

    # Test 9: Try setting parameter via SQL function
    print("\nTest 9: SET command")
    try_syntax(
        "SET vec_ef_search=200",
        "SET vec_ef_search=200",
        db,
    )

    # Test 10: Check if sqlite-vec source code has these
    print("\nTest 10: List registered virtual table modules")
    rows = db.execute("SELECT * FROM pragma_module_list").fetchall()
    for row in rows:
        print(f"  Module: {row[0]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
