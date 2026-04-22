"""Command-line interface for kemi."""

import argparse
import json
import os
import sys

from kemi import Memory
from kemi.models import LifecycleState


def main():
    parser = argparse.ArgumentParser(description="kemi CLI - persistent memory for AI agents")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("list", help="List all memories for a user")
    subparsers.add_parser("recall", help="Search memories for a user")
    subparsers.add_parser("forget", help="Delete all memories for a user")
    subparsers.add_parser("export", help="Export all memories to a file")
    subparsers.add_parser("import", help="Import memories from a file")
    subparsers.add_parser("stats", help="Show memory statistics")
    subparsers.add_parser("list-users", help="List all users with memory counts")
    update_parser = subparsers.add_parser("update", help="Update a memory")
    update_parser.add_argument("memory_id", help="Memory ID to update")
    update_parser.add_argument("--content", help="New content")
    update_parser.add_argument("--importance", type=float, help="New importance (0.0-1.0)")

    args, remaining = parser.parse_known_args()

    if args.command == "list":
        list_memories(remaining)
    elif args.command == "recall":
        recall_memories(remaining)
    elif args.command == "forget":
        forget_memories(remaining)
    elif args.command == "export":
        export_memories(remaining)
    elif args.command == "import":
        import_memories(remaining)
    elif args.command == "stats":
        show_stats()
    elif args.command == "list-users":
        list_users(remaining)
    elif args.command == "update":
        update_memory(remaining)
    else:
        parser.print_help()


def get_memory():
    """Get a Memory instance, handling db not existing yet."""
    db_path = os.path.expanduser("~/.kemi/memories.db")
    if not os.path.exists(db_path):
        print("No memory database found yet.")
        print(f"Location: {db_path}")
        print("Run 'kemi list <user_id>' after storing some memories.")
        sys.exit(1)
    return Memory()


def list_memories(args):
    parser = argparse.ArgumentParser("kemi list <user_id>")
    parser.add_argument("user_id", help="User ID")
    parsed = parser.parse_args(args)

    memory = get_memory()
    results = memory.recall(parsed.user_id, "", top_k=1000)

    if not results:
        print(f"No memories found for user: {parsed.user_id}")
        return

    print(f"Memories for user: {parsed.user_id}")
    print("-" * 80)
    for r in results:
        state = r.lifecycle_state.value if r.lifecycle_state else "unknown"
        print(f"ID: {r.memory_id}")
        print(f"Content: {r.content}")
        print(f"Importance: {r.importance:.2f}")
        print(f"State: {state}")
        print(f"Score: {r.score:.3f}")
        print("-" * 80)


def recall_memories(args):
    parser = argparse.ArgumentParser("kemi recall <user_id> <query>")
    parser.add_argument("user_id", help="User ID")
    parser.add_argument("query", nargs="?", default="", help="Search query")
    parsed = parser.parse_args(args)

    memory = get_memory()
    results = memory.recall(parsed.user_id, parsed.query)

    if not results:
        print(f"No memories found for: {parsed.query or 'all'}")
        return

    print(f"Results for: {parsed.query}")
    print("-" * 80)
    for r in results:
        print(f"Score: {r.score:.3f} | {r.content}")
    print("-" * 80)


def forget_memories(args):
    parser = argparse.ArgumentParser("kemi forget <user_id>")
    parser.add_argument("user_id", help="User ID")
    parsed = parser.parse_args(args)

    memory = get_memory()
    count = memory._store.count(parsed.user_id)

    if count == 0:
        print(f"No memories found for user: {parsed.user_id}")
        return

    print(f"This will delete {count} memories for user: {parsed.user_id}")
    print("Are you sure? (y/n): ", end="")
    response = input().strip().lower()

    if response == "y":
        deleted = memory.forget(parsed.user_id)
        print(f"Deleted {deleted} memories.")
    else:
        print("Cancelled.")


def export_memories(args):
    parser = argparse.ArgumentParser("kemi export <file>")
    parser.add_argument("file", help="Output file path")
    parsed = parser.parse_args(args)

    memory = get_memory()
    count = memory.export(parsed.file)
    print(f"Exported {count} memories to: {parsed.file}")


def import_memories(args):
    parser = argparse.ArgumentParser("kemi import <file>")
    parser.add_argument("file", help="Input file path")
    parsed = parser.parse_args(args)

    if not os.path.exists(parsed.file):
        print(f"File not found: {parsed.file}")
        sys.exit(1)

    memory = get_memory()
    with open(parsed.file, "r") as f:
        data = json.load(f)

    total = len(data)
    imported = memory.import_from(parsed.file)
    skipped = total - imported

    print(f"Import complete:")
    print(f"  Imported: {imported}")
    print(f"  Skipped (duplicates): {skipped}")


def show_stats():
    db_path = os.path.expanduser("~/.kemi/memories.db")

    if not os.path.exists(db_path):
        print("No memory database found.")
        return

    memory = get_memory()

    all_memories = memory._store.get_all()
    total_memories = len(all_memories)

    users = set(m.user_id for m in all_memories)
    total_users = len(users)

    db_size = os.path.getsize(db_path) / (1024 * 1024)

    print("kemi Statistics")
    print("=" * 40)
    print(f"Database: {db_path}")
    print(f"Total users: {total_users}")
    print(f"Total memories: {total_memories}")
    print(f"Database size: {db_size:.2f} MB")


def list_users(args):
    parser = argparse.ArgumentParser("kemi list-users")
    parsed = parser.parse_args(args)

    memory = get_memory()
    users = memory.list_users()

    if not users:
        print("No users found.")
        return

    print("Users:")
    for user_id in users:
        count = memory._store.count(user_id)
        print(f"  {user_id}: {count} memories")


def update_memory(args):
    parser = argparse.ArgumentParser(
        "kemi update <memory_id> --content <text> --importance <float>"
    )
    parser.add_argument("memory_id", help="Memory ID to update")
    parser.add_argument("--content", help="New content")
    parser.add_argument("--importance", type=float, help="New importance (0.0-1.0)")
    parsed = parser.parse_args(args)

    if not parsed.content and not parsed.importance:
        print("Error: must specify --content and/or --importance")
        sys.exit(1)

    memory = get_memory()

    try:
        memory.update(parsed.memory_id, content=parsed.content, importance=parsed.importance)
        print(f"Updated memory: {parsed.memory_id}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
