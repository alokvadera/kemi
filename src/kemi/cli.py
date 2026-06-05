"""Command-line interface for kemi."""

import argparse
import json
import os
import sys

from kemi import Memory
from kemi.cli_writer import Writer, make_writer
from kemi.models import LifecycleState

# Module-level writer — set by main() based on --json/--quiet flags.
# All subcommands read this via the get_writer() helper so we don't have
# to thread it through every function signature.
_writer: Writer | None = None


def get_writer() -> Writer:
    """Return the active CLI writer, falling back to a default ConsoleWriter."""
    global _writer
    if _writer is None:
        _writer = make_writer()
    return _writer


def main() -> None:
    parser = argparse.ArgumentParser(description="kemi CLI - persistent memory for AI agents")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one JSON object per line (NDJSON) on stdout.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level output; errors and warnings still appear.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    list_parser = subparsers.add_parser("list", help="List all memories for a user")
    list_parser.add_argument("user_id", help="User ID")
    list_parser.add_argument("--namespace", default="default", help="Namespace to filter by")
    list_parser.add_argument(
        "--lifecycle-filter",
        help=(
            "Filter by lifecycle state(s), comma-separated. "
            "Values: active,decaying,archived,deleted"
        ),
    )
    list_parser.add_argument("--session-id", help="Session ID filter")
    list_parser.add_argument("--tags", help="Filter by tag (comma-separated, OR logic)")

    # store
    store_parser = subparsers.add_parser("store", help="Store a new memory")
    store_parser.add_argument("user_id", help="User ID")
    store_parser.add_argument("content", help="Memory content")
    store_parser.add_argument("--importance", type=float, default=0.5, help="Importance (0.0-1.0)")
    store_parser.add_argument("--namespace", default="default", help="Memory namespace")
    store_parser.add_argument("--session-id", help="Session ID")
    store_parser.add_argument("--agent-id", help="Agent ID")
    store_parser.add_argument("--run-id", help="Run ID")
    store_parser.add_argument("--app-id", help="App ID")
    store_parser.add_argument("--metadata", help="Metadata as JSON string")
    store_parser.add_argument("--tags", help="Comma-separated tags")
    store_parser.add_argument("--chunk", action="store_true", help="Automatically split content into semantic chunks before storing")
    store_parser.add_argument("--max-chunk-tokens", type=int, default=256, help="Max tokens per chunk (used with --chunk)")
    store_parser.add_argument("--chunk-overlap", type=int, default=1, help="Sentence overlap between chunks (used with --chunk)")

    # recall
    recall_parser = subparsers.add_parser("recall", help="Search memories for a user")
    recall_parser.add_argument("user_id", help="User ID")
    recall_parser.add_argument("query", nargs="?", default="", help="Search query")
    recall_parser.add_argument("--namespace", default="default", help="Memory namespace")
    recall_parser.add_argument("--session-id", help="Session ID filter")
    recall_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return"
    )
    recall_parser.add_argument(
        "--metadata-filter", help='Metadata filter as JSON string, e.g. \'{"key":"value"}\''
    )

    # recall-stream
    recall_stream_parser = subparsers.add_parser(
        "recall-stream",
        help="Stream recall results progressively. "
        "Each result prints to stdout as it becomes available "
        "instead of waiting for full ranking.",
    )
    recall_stream_parser.add_argument("user_id", help="User ID")
    recall_stream_parser.add_argument("query", help="Search query")
    recall_stream_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return"
    )
    recall_stream_parser.add_argument("--namespace", default="default", help="Memory namespace")
    recall_stream_parser.add_argument("--session-id", help="Session ID filter")
    recall_stream_parser.add_argument(
        "--hybrid-search",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Use hybrid search (true/false, default: from config)",
    )
    recall_stream_parser.add_argument(
        "--metadata-filter", help='Metadata filter as JSON string, e.g. \'{"key":"value"}\''
    )

    # forget
    forget_parser = subparsers.add_parser("forget", help="Delete all memories for a user")
    forget_parser.add_argument("user_id", help="User ID")
    forget_parser.add_argument("--memory-id", help="Specific memory ID to delete")

    # forget-many
    forget_many_parser = subparsers.add_parser("forget-many", help="Delete multiple memories by ID")
    forget_many_parser.add_argument("memory_ids", nargs="+", help="Memory IDs to delete")

    # update-many
    update_many_parser = subparsers.add_parser(
        "update-many", help="Update multiple memories at once"
    )
    update_many_parser.add_argument("memory_ids", nargs="+", help="Memory IDs to update")
    update_many_parser.add_argument("--content", help="New content for all")
    update_many_parser.add_argument("--importance", type=float, help="New importance (0.0-1.0)")
    update_many_parser.add_argument("--confidence", type=float, help="New confidence (0.0-1.0)")
    update_many_parser.add_argument(
        "--memory-type", choices=["episodic", "semantic"], help="New memory type"
    )

    # recall-many
    recall_many_parser = subparsers.add_parser(
        "recall-many", help="Batch recall for multiple users"
    )
    recall_many_parser.add_argument(
        "user_queries", nargs="+", help="User:query pairs, e.g. user1:food user2:travel"
    )
    recall_many_parser.add_argument("--namespace", default="default", help="Memory namespace")
    recall_many_parser.add_argument("--session-id", help="Session ID filter")
    recall_many_parser.add_argument("--metadata-filter", help="Metadata filter as JSON string")

    # export
    export_parser = subparsers.add_parser("export", help="Export all memories to a file")
    export_parser.add_argument("file", help="Output file path")

    # import
    import_parser = subparsers.add_parser("import", help="Import memories from a file")
    import_parser.add_argument("file", help="Input file path")

    # stats
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.add_argument("user_id", nargs="?", help="User ID (optional)")
    stats_parser.add_argument(
        "--lifecycle-filter",
        help=(
            "Filter by lifecycle state(s), comma-separated. "
            "Values: active,decaying,archived,deleted"
        ),
    )
    stats_parser.add_argument("--session-id", help="Session ID filter")

    # list-users
    subparsers.add_parser("list-users", help="List all users with memory counts")

    # update
    update_parser = subparsers.add_parser("update", help="Update a memory")
    update_parser.add_argument("memory_id", help="Memory ID to update")
    update_parser.add_argument("--content", help="New content")
    update_parser.add_argument("--importance", type=float, help="New importance (0.0-1.0)")
    update_parser.add_argument("--confidence", type=float, help="New confidence (0.0-1.0)")
    update_parser.add_argument(
        "--memory-type", choices=["episodic", "semantic"], help="New memory type"
    )
    update_parser.add_argument("--metadata", help="Metadata as JSON string to merge")
    update_parser.add_argument("--tags", help="Comma-separated tags (replaces existing)")

    # prune
    prune_parser = subparsers.add_parser("prune", help="Prune old or low-importance memories")
    prune_parser.add_argument("user_id", help="User ID")
    prune_parser.add_argument(
        "--max-age-days", type=float, help="Delete memories older than N days"
    )
    prune_parser.add_argument(
        "--min-importance", type=float, help="Delete memories below importance"
    )
    prune_parser.add_argument("--namespace", default="default", help="Memory namespace")

    # consolidate
    consolidate_parser = subparsers.add_parser("consolidate", help="Consolidate old memories")
    consolidate_parser.add_argument("user_id", help="User ID")
    consolidate_parser.add_argument("--namespace", default="default", help="Memory namespace")
    consolidate_parser.add_argument(
        "--with-summary",
        action="store_true",
        default=False,
        help="Use LLM-powered abstractive summarization instead of extractive",
    )

    # topics
    topics_parser = subparsers.add_parser("topics", help="Cluster memories into topics")
    topics_parser.add_argument("user_id", help="User ID")
    topics_parser.add_argument("--n-clusters", type=int, default=3, help="Number of clusters")
    topics_parser.add_argument("--namespace", default="default", help="Memory namespace")

    # graph
    graph_parser = subparsers.add_parser("graph", help="Show memory graph")
    graph_parser.add_argument("user_id", help="User ID")
    graph_parser.add_argument("--namespace", default="default", help="Memory namespace")

    # chunk
    chunk_parser = subparsers.add_parser("chunk", help="Preview how content would be semantically chunked")
    chunk_parser.add_argument("content", help="Content to preview chunking on")
    chunk_parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per chunk")
    chunk_parser.add_argument("--overlap", type=int, default=1, help="Sentence overlap between chunks")

    # explain
    explain_parser = subparsers.add_parser("explain", help="Recall with score explanations")
    explain_parser.add_argument("user_id", help="User ID")
    explain_parser.add_argument("query", help="Search query")
    explain_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    explain_parser.add_argument("--namespace", default="default", help="Memory namespace")

    # decompose
    decompose_parser = subparsers.add_parser("decompose", help="Decompose a query into sub-queries and show fusion")
    decompose_parser.add_argument("user_id", help="User ID")
    decompose_parser.add_argument("query", help="Complex multi-aspect query")
    decompose_parser.add_argument("--strategy", default="simple",
                                   choices=["simple", "expand", "both", "none"],
                                   help="Decomposition strategy (default: simple)")
    decompose_parser.add_argument("--top-k", type=int, default=5, help="Results per sub-query")
    decompose_parser.add_argument("--namespace", default="default", help="Memory namespace")

    # rerank
    rerank_parser = subparsers.add_parser("rerank", help="Rerank recall results with cross-encoder scoring")
    rerank_parser.add_argument("user_id", help="User ID")
    rerank_parser.add_argument("query", help="Search query")
    rerank_parser.add_argument("--top-k", type=int, default=10, help="Number of results to rerank")
    rerank_parser.add_argument("--provider", default="fallback", help="Reranker provider (fallback/nomic)")
    rerank_parser.add_argument("--namespace", default="default", help="Memory namespace")

    # webhook
    webhook_parser = subparsers.add_parser("webhook", help="Manage webhook callbacks")
    webhook_sub = webhook_parser.add_subparsers(dest="webhook_command", help="Webhook subcommands")

    webhook_add_parser = webhook_sub.add_parser("add", help="Register a new webhook endpoint")
    webhook_add_parser.add_argument("--url", required=True, help="Webhook endpoint URL")
    webhook_add_parser.add_argument(
        "--events", required=True,
        help="Comma-separated event types: remembered,updated,forgotten,deleted,conflict,consolidated",
    )
    webhook_add_parser.add_argument("--secret", default="", help="HMAC signing secret")
    webhook_add_parser.add_argument(
        "--no-active", action="store_false", dest="active", default=True,
        help="Register but keep disabled",
    )

    webhook_list_parser = webhook_sub.add_parser("list", help="List registered webhooks")

    webhook_delete_parser = webhook_sub.add_parser("delete", help="Delete a webhook")
    webhook_delete_parser.add_argument("webhook_id", help="Webhook ID to delete")

    # history
    history_parser = subparsers.add_parser("history", help="Show version history of a memory")
    history_parser.add_argument("memory_id", help="Memory ID")
    history_parser.add_argument("--limit", type=int, default=100, help="Maximum versions to show")

    # version (subcommand: diff)
    version_parser = subparsers.add_parser("version", help="Version management commands")
    version_sub = version_parser.add_subparsers(dest="version_command", help="Version subcommands")
    diff_parser = version_sub.add_parser("diff", help="Show diff between two versions")
    diff_parser.add_argument("memory_id", help="Memory ID")
    diff_parser.add_argument("--v1", type=int, required=True, help="Starting version number")
    diff_parser.add_argument("--v2", type=int, required=True, help="Ending version number")

    # rollback
    rollback_parser = subparsers.add_parser("rollback", help="Rollback a memory to a previous version")
    rollback_parser.add_argument("memory_id", help="Memory ID")
    rollback_parser.add_argument("--to-version", type=int, required=True, dest="target_version", help="Target version number")

    hook_group = parser.add_mutually_exclusive_group()
    hook_group.add_argument(
        "--hooks-raise-on-error",
        action="store_true",
        default=None,
        dest="hooks_raise_on_error",
        help="Abort operations when event hooks fail (default: True from config)",
    )
    hook_group.add_argument(
        "--no-hooks-raise-on-error",
        action="store_false",
        default=None,
        dest="hooks_raise_on_error",
        help="Log and swallow event hook failures instead of aborting",
    )

    args = parser.parse_args()

    # Set up the active writer for subcommands to read via get_writer().
    global _writer
    _writer = make_writer(json_mode=args.json, quiet=args.quiet)

    if args.command == "list":
        list_memories(args)
    elif args.command == "store":
        store_memory(args)
    elif args.command == "recall":
        recall_memories(args)
    elif args.command == "recall-stream":
        recall_stream_memories(args)
    elif args.command == "recall-many":
        recall_many_memories(args)
    elif args.command == "forget":
        forget_memories(args)
    elif args.command == "forget-many":
        forget_many_memories(args)
    elif args.command == "export":
        export_memories(args)
    elif args.command == "import":
        import_memories(args)
    elif args.command == "stats":
        show_stats(args)
    elif args.command == "list-users":
        list_users(args)
    elif args.command == "update":
        update_memory(args)
    elif args.command == "update-many":
        update_many_memories(args)
    elif args.command == "prune":
        prune_memories(args)
    elif args.command == "consolidate":
        consolidate_memories(args)
    elif args.command == "topics":
        topics_memories(args)
    elif args.command == "graph":
        graph_memories(args)
    elif args.command == "explain":
        explain_memories(args)
    elif args.command == "decompose":
        decompose_and_recall(args)
    elif args.command == "rerank":
        rerank_recall(args)
    elif args.command == "webhook":
        cmd = getattr(args, "webhook_command", None)
        if cmd == "add":
            webhook_add(args)
        elif cmd == "list":
            webhook_list(args)
        elif cmd == "delete":
            webhook_delete(args)
        else:
            webhook_parser.print_help()
    elif args.command == "history":
        show_history(args)
    elif args.command == "version":
        cmd = getattr(args, "version_command", None)
        if cmd == "diff":
            show_version_diff(args)
        else:
            version_parser.print_help()
    elif args.command == "rollback":
        rollback_memory(args)
    elif args.command == "chunk":
        preview_chunk(args)
    else:
        parser.print_help()


def get_memory(args: argparse.Namespace | None = None) -> Memory:
    """Get a Memory instance, handling db not existing yet."""
    db_path = os.path.expanduser("~/.kemi/memories.db")
    if not os.path.exists(db_path):
        w = get_writer()
        w.write("No memory database found yet.")
        w.write(f"Location: {db_path}")
        w.write("Run 'kemi list <user_id>' after storing some memories.")
        sys.exit(1)

    if args is not None and getattr(args, "hooks_raise_on_error", None) is not None:
        from kemi.models import MemoryConfig

        config = MemoryConfig(hooks_raise_on_error=args.hooks_raise_on_error)
        return Memory(config=config)
    return Memory()


def list_memories(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    storage = memory._store

    lifecycle_filter = None
    if args.lifecycle_filter:
        states = [s.strip().lower() for s in args.lifecycle_filter.split(",")]
        try:
            lifecycle_filter = [LifecycleState(s) for s in states]
        except ValueError:
            valid = [s.value for s in LifecycleState]
            get_writer().error(f"invalid lifecycle state. Valid values: {valid}")
            sys.exit(2)

    results = storage.get_all_by_user(
        args.user_id,
        namespace=args.namespace,
        lifecycle_filter=lifecycle_filter,
        session_id=args.session_id,
    )

    # Filter by tags if specified (OR logic: memory matches if it has any of the specified tags)
    tag_filter = None
    if getattr(args, "tags", None):
        tag_filter = [t.strip().lower() for t in args.tags.split(",")]
        results = [
            r for r in results
            if r.tags and any(t.lower() in tag_filter for t in r.tags)
        ]

    if not results:
        get_writer().write(f"No memories found for user: {args.user_id}")
        return

    get_writer().write(f"Memories for user: {args.user_id}")
    get_writer().write("-" * 80)
    for r in results:
        state = r.lifecycle_state.value if r.lifecycle_state else "unknown"
        get_writer().write(f"ID: {r.memory_id}")
        get_writer().write(f"Content: {r.content}")
        get_writer().write(f"Importance: {r.importance:.2f}")
        get_writer().write(f"State: {state}")
        if r.tags:
            get_writer().write(f"Tags: {', '.join(r.tags)}")
        get_writer().write("-" * 80)
def store_memory(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            get_writer().error(f"invalid metadata JSON: {e}")
            sys.exit(2)
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None

    if getattr(args, "chunk", False):
        # Chunk the content before storing
        from kemi.chunker import semantic_chunks
        chunks = semantic_chunks(
            args.content,
            memory._embed,
            max_tokens=getattr(args, "max_chunk_tokens", 256),
            overlap_sentences=getattr(args, "chunk_overlap", 1),
        )
        if not chunks:
            get_writer().error(f"chunking produced no results")
            sys.exit(1)
        if len(chunks) == 1:
            # Single chunk — just store normally
            mid = memory.remember(
                args.user_id,
                args.content,
                importance=args.importance,
                namespace=args.namespace,
                session_id=args.session_id,
                agent_id=args.agent_id,
                run_id=args.run_id,
                app_id=args.app_id,
                metadata=metadata,
                tags=tags,
            )
            get_writer().write(f"Stored memory: {mid} (1 chunk)")
        else:
            # Store each chunk as a separate memory with parent link
            from kemi.chunker import CHUNK_META_KEY
            parent_id = None
            for i, chunk in enumerate(chunks):
                chunk_meta = {
                    CHUNK_META_KEY: chunk.chunk_info.to_dict() if chunk.chunk_info else {}
                }
                merged_meta = {**(metadata or {}), **chunk_meta}
                mid = memory.remember(
                    args.user_id,
                    chunk.content,
                    importance=args.importance,
                    namespace=args.namespace,
                    session_id=args.session_id,
                    agent_id=args.agent_id,
                    run_id=args.run_id,
                    app_id=args.app_id,
                    metadata=merged_meta,
                    tags=tags,
                )
                if i == 0:
                    parent_id = mid
                get_writer().write(f"  Stored chunk {i + 1}/{len(chunks)}: {mid}")
            get_writer().write(f"Stored {len(chunks)} chunks as memories (parent: {parent_id})")
    else:
        mid = memory.remember(
            args.user_id,
            args.content,
            importance=args.importance,
            namespace=args.namespace,
            session_id=args.session_id,
            agent_id=args.agent_id,
            run_id=args.run_id,
            app_id=args.app_id,
            metadata=metadata,
            tags=tags,
        )
        get_writer().write(f"Stored memory: {mid}")
def recall_memories(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    metadata_filter = None
    if args.metadata_filter:
        try:
            metadata_filter = json.loads(args.metadata_filter)
        except json.JSONDecodeError as e:
            get_writer().error(f"invalid metadata-filter JSON: {e}")
            sys.exit(2)
    results = memory.recall(
        args.user_id,
        args.query,
        top_k=args.top_k,
        namespace=args.namespace,
        session_id=args.session_id,
        metadata_filter=metadata_filter,
    )

    if not results:
        get_writer().write(f"No memories found for: {args.query or 'all'}")
        return

    get_writer().write(f"Results for: {args.query}")
    get_writer().write("-" * 80)
    for r in results:
        get_writer().write(f"Score: {r.score:.3f} | {r.content}")
    get_writer().write("-" * 80)
def recall_stream_memories(args: argparse.Namespace) -> None:
    """Stream recall results progressively, printing each as it arrives."""
    import asyncio

    memory = get_memory(args)

    metadata_filter = None
    if getattr(args, "metadata_filter", None):
        try:
            metadata_filter = json.loads(args.metadata_filter)
        except json.JSONDecodeError as e:
            get_writer().error(f"invalid metadata-filter JSON: {e}")
            sys.exit(2)

    async def _stream() -> None:
        count = 0
        async for mem in memory.recall_stream(
            user_id=args.user_id,
            query=args.query,
            top_k=args.top_k,
            namespace=args.namespace,
            session_id=args.session_id,
            hybrid_search=args.hybrid_search,
            metadata_filter=metadata_filter,
        ):
            count += 1
            get_writer().write(f"#{count:>3} | Score: {mem.score:.3f} | {mem.content}", flush=True)
        if count == 0:
            get_writer().write(f"No memories found for: {args.query}")
            return

        get_writer().write("-" * 80)
        get_writer().write(f"Streamed {count} result(s)")
    asyncio.run(_stream())


def recall_many_memories(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    metadata_filter = None
    if args.metadata_filter:
        try:
            metadata_filter = json.loads(args.metadata_filter)
        except json.JSONDecodeError as e:
            get_writer().error(f"invalid metadata-filter JSON: {e}")
            sys.exit(2)
    user_ids = []
    queries = []
    for pair in args.user_queries:
        if ":" not in pair:
            get_writer().error(f"user_queries must be in 'user:query' format, got: {pair}")
            sys.exit(2)
        uid, q = pair.split(":", 1)
        user_ids.append(uid)
        queries.append(q)
    results = memory.recall_many(
        user_ids,
        queries,
        namespace=args.namespace,
        session_id=args.session_id,
        metadata_filter=metadata_filter,
    )
    for uid, mems in results.items():
        get_writer().write(f"\nResults for {uid}:")
        get_writer().write("-" * 40)
        for m in mems:
            get_writer().write(f"  {m.score:.3f} | {m.content}")
def forget_memories(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    if args.memory_id:
        deleted = memory.forget(args.user_id, memory_id=args.memory_id)
        get_writer().write(f"Deleted {deleted} memory.")
        return

    count = memory._store.count(args.user_id)

    if count == 0:
        get_writer().write(f"No memories found for user: {args.user_id}")
        return

    get_writer().write(f"This will delete {count} memories for user: {args.user_id}")
    get_writer().write("Are you sure? (y/n): ", end="")
    response = input().strip().lower()

    if response == "y":
        deleted = memory.forget(args.user_id)
        get_writer().write(f"Deleted {deleted} memories.")
    else:
        get_writer().write("Cancelled.")
def forget_many_memories(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    deleted = memory.forget_many(args.memory_ids)
    get_writer().write(f"Deleted {deleted} memories.")
def update_many_memories(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    from kemi.models import MemoryType

    mtype = None
    if args.memory_type:
        mtype = MemoryType(args.memory_type)

    updated = memory.update_many(
        args.memory_ids,
        content=args.content,
        importance=args.importance,
        confidence=args.confidence,
        memory_type=mtype,
    )
    get_writer().write(f"Updated {len(updated)} memories.")
def export_memories(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    count = memory.export(args.file)
    get_writer().write(f"Exported {count} memories to: {args.file}")
def import_memories(args: argparse.Namespace) -> None:
    if not os.path.exists(args.file):
        get_writer().write(f"File not found: {args.file}")
        sys.exit(1)

    memory = get_memory(args)
    with open(args.file) as f:
        data = json.load(f)

    total = len(data)
    imported = memory.import_from(args.file)
    skipped = total - imported

    get_writer().write("Import complete:")
    get_writer().write(f"  Imported: {imported}")
    get_writer().write(f"  Skipped (duplicates): {skipped}")
def show_stats(args: argparse.Namespace) -> None:
    db_path = os.path.expanduser("~/.kemi/memories.db")

    if not os.path.exists(db_path):
        get_writer().write("No memory database found.")
        return

    memory = get_memory(args)

    if args.user_id:
        lifecycle_filter = None
        if args.lifecycle_filter:
            states = [s.strip().lower() for s in args.lifecycle_filter.split(",")]
            try:
                lifecycle_filter = [LifecycleState(s) for s in states]
            except ValueError:
                valid = [s.value for s in LifecycleState]
                get_writer().error(f"invalid lifecycle state. Valid values: {valid}")
                sys.exit(2)

        stats_data = memory.stats(
            args.user_id,
            lifecycle_filter=lifecycle_filter,
            session_id=args.session_id,
        )
        get_writer().write(f"Statistics for user: {args.user_id}")
        get_writer().write("=" * 40)
        get_writer().write(f"Total memories: {stats_data['total']}")
        get_writer().write(f"By lifecycle: {stats_data['by_lifecycle']}")
        get_writer().write(f"By source: {stats_data['by_source']}")
        get_writer().write(f"Avg importance: {stats_data['avg_importance']:.2f}")
        get_writer().write(f"Tags: {stats_data['tag_counts']}")
        get_writer().write(f"With tags: {stats_data['total_with_tags']}")
        get_writer().write(f"Without tags: {stats_data['total_without_tags']}")
    else:
        all_memories = memory._store.get_all()
        total_memories = len(all_memories)
        users = set(m.user_id for m in all_memories)
        total_users = len(users)
        db_size = os.path.getsize(db_path) / (1024 * 1024)

        get_writer().write("kemi Statistics")
        get_writer().write("=" * 40)
        get_writer().write(f"Database: {db_path}")
        get_writer().write(f"Total users: {total_users}")
        get_writer().write(f"Total memories: {total_memories}")
        get_writer().write(f"Database size: {db_size:.2f} MB")
def list_users(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    users = memory.list_users()

    if not users:
        get_writer().write("No users found.")
        return

    get_writer().write("Users:")
    for user_id in users:
        count = memory._store.count(user_id)
        get_writer().write(f"  {user_id}: {count} memories")
def update_memory(args: argparse.Namespace) -> None:
    if (
        not args.content
        and args.importance is None
        and args.confidence is None
        and not args.memory_type
        and not getattr(args, "metadata", None)
        and not getattr(args, "tags", None)
    ):
        get_writer().error(f"must specify at least one field to update")
        sys.exit(1)

    memory = get_memory(args)
    from kemi.models import MemoryType

    mtype = None
    if args.memory_type:
        mtype = MemoryType(args.memory_type)

    metadata = None
    if getattr(args, "metadata", None):
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            get_writer().error(f"invalid metadata JSON: {e}")
            sys.exit(2)

    tags = [t.strip() for t in getattr(args, "tags", None).split(",")] if getattr(args, "tags", None) else None

    try:
        memory.update(
            args.memory_id,
            content=args.content,
            importance=args.importance,
            confidence=args.confidence,
            memory_type=mtype,
            metadata=metadata,
            tags=tags,
        )
        get_writer().write(f"Updated memory: {args.memory_id}")
    except ValueError as e:
        get_writer().error(f"{e}")
        sys.exit(1)


def prune_memories(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    deleted = memory.prune(
        args.user_id,
        max_age_days=args.max_age_days,
        min_importance=args.min_importance,
        namespace=args.namespace,
    )
    get_writer().write(f"Pruned {deleted} memories.")
def consolidate_memories(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    with_summary = getattr(args, "with_summary", False)
    mid = memory.consolidate(
        args.user_id,
        namespace=args.namespace,
        with_llm_summary=with_summary,
    )
    if mid:
        label = "LLM summary" if with_summary else "extractive summary"
        get_writer().write(f"Consolidated into memory: {mid} ({label})")
    else:
        get_writer().write("No consolidation needed.")
def topics_memories(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    try:
        clusters = memory.cluster_topics(
            args.user_id,
            n_clusters=args.n_clusters,
            namespace=args.namespace,
        )
    except ImportError as e:
        get_writer().error(f"{e}")
        sys.exit(1)
        return  # unreachable when sys.exit is real, but needed when mocked in tests

    if not clusters:
        get_writer().write("No topics found.")
        return

    for label, mems in clusters.items():
        get_writer().write(f"\n{label} ({len(mems)} memories):")
        for mem in mems[:5]:
            get_writer().write(f"  - {mem.content[:80]}...")
def graph_memories(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    graph_data = memory.get_memory_graph(args.user_id, namespace=args.namespace)

    get_writer().write("Entities:")
    entities = graph_data.get("entities", [])
    if entities:
        for ent in entities[:20]:
            get_writer().write(f"  - {ent['text']} ({ent['label']})")
    else:
        get_writer().write("  None")
    get_writer().write("\nRelations:")
    relations = graph_data.get("relations", [])
    if relations:
        for rel in relations[:20]:
            get_writer().write(f"  - {rel['subject']} --{rel['predicate']}--> {rel['object']}")
    else:
        get_writer().write("  None")
def explain_memories(args: argparse.Namespace) -> None:
    memory = get_memory(args)
    explained = memory.recall_explain(
        args.user_id,
        args.query,
        top_k=args.top_k,
        namespace=args.namespace,
    )

    if not explained:
        get_writer().write("No memories found.")
        return

    for item in explained:
        mem = item["memory"]
        exp = item["explanation"]
        get_writer().write(f"\nScore: {exp['final_score']:.4f}")
        get_writer().write(f"  Content: {mem.content}")
        get_writer().write(
            f"  Semantic: {exp['semantic_score']} | Recency: {exp['recency_score']} "
            f"| BM25: {exp.get('bm25_score', 'n/a')}"
        )
        get_writer().write(f"  Weights: {exp['weights']}")
def decompose_and_recall(args: argparse.Namespace) -> None:
    """Decompose a complex query into sub-queries, execute each, fuse with RRF."""
    from kemi.decomposer import decompose_query, fused_recall

    memory = get_memory(args)

    # First show decomposition
    decomposed = decompose_query(args.query, strategy=args.strategy)
    get_writer().write(f"Original query: {decomposed.original_query}")
    get_writer().write(f"Strategy: {decomposed.strategy}")
    get_writer().write(f"Sub-queries ({len(decomposed.sub_queries)}):")
    for i, sq in enumerate(decomposed.sub_queries, 1):
        get_writer().write(f"  {i}. {sq}")
    if len(decomposed.sub_queries) == 1:
        get_writer().write("\n(Single sub-query, skipping fusion)")
        results = memory.recall(
            args.user_id,
            decomposed.sub_queries[0],
            top_k=args.top_k,
            namespace=args.namespace,
        )
    else:
        get_writer().write("\nFusing results with Reciprocal Rank Fusion (k=60)...")
        fusion_results = fused_recall(
            memory,
            args.user_id,
            decomposed.sub_queries,
            top_k=args.top_k,
            namespace=args.namespace,
        )
        results = [fr.memory for fr in fusion_results]

    if not results:
        get_writer().write("\nNo memories found.")
        return

    get_writer().write(f"\nResults (fused top {len(results)}):")
    get_writer().write("-" * 80)
    for r in results:
        get_writer().write(f"  {r.score:.3f} | {r.content[:80]}")
def rerank_recall(args: argparse.Namespace) -> None:
    """Recall memories then rerank using cross-encoder scoring."""
    from kemi.reranker import rerank_results, RerankerConfig

    memory = get_memory(args)

    # Initial retrieval
    initial = memory.recall(
        args.user_id,
        args.query,
        top_k=args.top_k,
        namespace=args.namespace,
    )

    if not initial:
        get_writer().write("No memories found.")
        return

    get_writer().write(f"Initial recall: {len(initial)} results")
    # Rerank
    config = RerankerConfig(provider=args.provider)
    reranked = rerank_results(
        initial,
        args.query,
        config,
        embed_fn=memory._embed,
    )

    get_writer().write(f"\nReranked results ({args.provider} reranker):")
    get_writer().write("-" * 80)
    for r in reranked:
        ce_score = getattr(r, "cross_encoder_score", 0.0)
        get_writer().write(f"  {ce_score:.4f} | {r.content[:80]}")
def show_history(args: argparse.Namespace) -> None:
    """Show version history for a memory (`kemi history <memory_id>`)."""
    try:
        memory = get_memory(args)
    except SystemExit:
        return

    try:
        memory.configure_versioning()
        history = memory.get_history(args.memory_id, limit=args.limit)
    except RuntimeError as e:
        get_writer().error(f"{e}")
        return

    if not history:
        get_writer().write(f"No version history found for: {args.memory_id}")
        return

    get_writer().write(f"Version history for: {args.memory_id}")
    get_writer().write(f"Total versions: {len(history)}")
    get_writer().write("-" * 80)
    for snap in history:
        changed = f"by {snap.changed_by}" if snap.changed_by else ""
        ts = snap.changed_at.strftime('%Y-%m-%d %H:%M:%S') if snap.changed_at else "unknown"
        get_writer().write(f"v{snap.version} | {ts} | {changed}")
        get_writer().write(f"  Content: {snap.content[:70]}...")
        get_writer().write(f"  Importance: {snap.importance:.2f}")
        if snap.tags:
            get_writer().write(f"  Tags: {snap.tags}")
        print()


def show_version_diff(args: argparse.Namespace) -> None:
    """Show diff between two versions (`kemi version diff <memory_id> --v1 N --v2 M`)."""
    try:
        memory = get_memory(args)
    except SystemExit:
        return

    try:
        memory.configure_versioning()
        diff_result = memory.diff_versions(args.memory_id, args.v1, args.v2)
    except RuntimeError as e:
        get_writer().error(f"{e}")
        return

    if diff_result is None:
        get_writer().write(f"One or both versions not found for: {args.memory_id}")
        return

    if not diff_result.field_changes:
        get_writer().write(f"No differences between v{args.v1} and v{args.v2}")
        return

    get_writer().write(f"Diff v{args.v1} → v{args.v2} for: {args.memory_id}")
    get_writer().write("-" * 60)
    for field, (old_val, new_val) in diff_result.field_changes.items():
        get_writer().write(f"  {field}:")
        get_writer().write(f"    old: {str(old_val)[:200]}")
        get_writer().write(f"    new: {str(new_val)[:200]}")
def rollback_memory(args: argparse.Namespace) -> None:
    """Rollback a memory to a previous version (`kemi rollback <memory_id> --to-version N`)."""
    try:
        memory = get_memory(args)
    except SystemExit:
        return

    try:
        memory.configure_versioning()
    except RuntimeError as e:
        get_writer().error(f"{e}")
        return

    # Accept either `target_version` (canonical, from the CLI argument
    # `--to-version`) or `to_version` (alias used by some test harnesses).
    target_version = getattr(args, "target_version", None)
    if target_version is None:
        target_version = getattr(args, "to_version", None)
    if target_version is None:
        get_writer().error(f"missing --to-version argument")
        return

    # Check version exists first
    history = memory.get_history(args.memory_id, limit=1000)
    target = next((s for s in history if s.version == target_version), None)
    if not target:
        get_writer().write(f"Version {target_version} not found for: {args.memory_id}")
        return

    result = memory.rollback_memory(args.memory_id, target_version)

    if result:
        get_writer().write(f"Rolled back {args.memory_id} from v{result.from_version} to v{result.to_version}")
        get_writer().write(f"  Memory restored to: {target.content[:60]}...")
    else:
        get_writer().write("Rollback failed.")
def webhook_add(args: argparse.Namespace) -> None:
    """Register a new webhook endpoint."""
    from kemi.webhooks import WebhookConfig, WebhookEventType, WebhookStore

    db_path = os.path.expanduser("~/.kemi/memories.db")
    if not os.path.exists(db_path):
        get_writer().write("No memory database found. Run 'kemi store' first.")
        return

    try:
        events = [WebhookEventType.from_string(e.strip()) for e in args.events.split(",")]
    except ValueError as e:
        get_writer().error(f"{e}")
        return

    cfg = WebhookConfig(
        webhook_id="",
        url=args.url,
        events=events,
        secret=args.secret,
        active=args.active,
    )

    store = WebhookStore(db_path=db_path)
    wh_id = store.create(cfg)
    get_writer().write(f"Registered webhook: {wh_id}")
    get_writer().write(f"  URL: {args.url}")
    get_writer().write(f"  Events: {', '.join(e.value for e in events)}")
    get_writer().write(f"  Active: {args.active}")
def webhook_list(args: argparse.Namespace) -> None:
    """List registered webhooks."""
    from kemi.webhooks import WebhookStore

    db_path = os.path.expanduser("~/.kemi/memories.db")
    if not os.path.exists(db_path):
        get_writer().write("No memory database found.")
        return

    store = WebhookStore(db_path=db_path)
    configs = store.list_all(active_only=False)

    if not configs:
        get_writer().write("No webhooks registered.")
        return

    get_writer().write(f"Registered webhooks ({len(configs)}):")
    get_writer().write("-" * 80)
    for c in configs:
        status = "active" if c.active else "disabled"
        events = ", ".join(e.value for e in c.events)
        get_writer().write(f"  ID: {c.webhook_id}")
        get_writer().write(f"  URL: {c.url}")
        get_writer().write(f"  Events: {events}")
        get_writer().write(f"  Status: {status}")
        print()


def webhook_delete(args: argparse.Namespace) -> None:
    """Delete a webhook registration."""
    from kemi.webhooks import WebhookStore

    db_path = os.path.expanduser("~/.kemi/memories.db")
    if not os.path.exists(db_path):
        get_writer().write("No memory database found.")
        return

    store = WebhookStore(db_path=db_path)
    if store.delete(args.webhook_id):
        get_writer().write(f"Deleted webhook: {args.webhook_id}")
    else:
        get_writer().write(f"Webhook not found: {args.webhook_id}")
def preview_chunk(args: argparse.Namespace) -> None:
    """Preview how content would be semantically chunked."""
    from kemi.chunker import semantic_chunks

    memory = get_memory(args)
    chunks = semantic_chunks(
        args.content,
        memory._embed,
        max_tokens=args.max_tokens,
        overlap_sentences=args.overlap,
    )

    if not chunks:
        get_writer().write("No chunks produced (empty content?)")
        return

    get_writer().write(f"Content: {args.content[:80]}...")
    get_writer().write(f"Produced {len(chunks)} chunk(s):\n")
    get_writer().write("-" * 80)
    for i, chunk in enumerate(chunks, 1):
        info = chunk.chunk_info
        strength = f"{info.boundary_strength:.2f}" if info else "N/A"
        get_writer().write(f"Chunk {i}/{len(chunks)} [{chunk.token_count_estimate()} tokens, strength={strength}]")
        get_writer().write(f"  {chunk.content}")
        print()


if __name__ == "__main__":
    main()
