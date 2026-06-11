#!/usr/bin/env python3
"""Auto-fix E501 line-too-long errors using ruff JSON output."""

import json
import re
import subprocess
import sys
from pathlib import Path

LINE_LENGTH = 100


def get_e501_violations() -> list[dict]:
    """Run ruff and return list of E501 violations."""
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "src/", "tests/", "--select", "E501", "--output-format", "json"],
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        return []
    return json.loads(result.stdout)


def break_string_line(line: str, max_len: int = LINE_LENGTH) -> str | None:
    """Try to break a line containing a long string."""
    # Don't break if it's already under limit
    if len(line) <= max_len:
        return line

    # Pattern: variable = "very long string"
    m = re.match(r'^(\s*)(\w+)\s*=\s*(".*")\s*$', line)
    if m:
        indent, var, s = m.groups()
        # Try implicit string concatenation with parentheses
        if len(s) > max_len - len(indent) - len(var) - 5:
            new_line = f'{indent}{var} = (\n{indent}    {s}\n{indent})'
            return new_line

    # Pattern: return "very long string"
    m = re.match(r'^(\s*return\s+)(".*")\s*$', line)
    if m:
        prefix, s = m.groups()
        if len(s) > max_len - len(prefix):
            indent = re.match(r'^(\s*)', line).group(1)
            return f'{prefix}(\n{indent}    {s}\n{indent})'

    # Pattern: "string" in function call
    # e.g., raise ConfigurationError("...") from e
    m = re.match(r'^(\s*raise\s+\w+\()("[^"]+")(,.+)$', line)
    if m:
        indent = re.match(r'^(\s*)', line).group(1)
        prefix, s, suffix = m.groups()
        # Put string on next line with proper indent
        return f'{prefix}\n{indent}    {s}{suffix}'

    return None


def break_function_call(line: str, max_len: int = LINE_LENGTH) -> str | None:
    """Try to break a long function call."""
    if len(line) <= max_len:
        return line

    # Pattern: func(arg1, arg2, ...) — break args onto new lines
    m = re.match(r'^(\s*)([\w\.]+)\((.*)\)\s*$', line)
    if m:
        indent, func_name, args = m.groups()
        if len(args) < 20:  # Too short to bother breaking
            return None
        # Split by commas, but respect nested parens/brackets
        parts = []
        current = ""
        depth = 0
        for char in args:
            if char in "([{":
                depth += 1
            elif char in ")}]":
                depth -= 1
            if char == "," and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += char
        if current:
            parts.append(current)

        if len(parts) > 1:
            new_lines = [f'{indent}{func_name}(']
            for i, part in enumerate(parts):
                part = part.strip()
                if i < len(parts) - 1:
                    new_lines.append(f'{indent}    {part},')
                else:
                    new_lines.append(f'{indent}    {part},')
            new_lines.append(f'{indent})')
            result = "\n".join(new_lines)
            # Check that the longest line is under limit
            max_line_len = max(len(l) for l in result.split("\n"))
            if max_line_len <= max_len:
                return result

    return None


def break_function_def(line: str, max_len: int = LINE_LENGTH) -> str | None:
    """Try to break a long function definition."""
    if len(line) <= max_len:
        return line

    # Pattern: def name(args) -> type:
    m = re.match(r'^(\s*def\s+\w+)\((.*)\)(\s*->\s*[^:]+:)$', line)
    if m:
        prefix, args, suffix = m.groups()
        indent = " " * (len(prefix) + 4)

        # Try to split args by comma
        parts = []
        current = ""
        depth = 0
        for char in args:
            if char in "([{":
                depth += 1
            elif char in ")}]":
                depth -= 1
            if char == "," and depth == 0:
                parts.append(current)
                current = ""
            else:
                current += char
        if current:
            parts.append(current)

        if len(parts) > 1:
            new_lines = [f'{prefix}(']
            for part in parts:
                new_lines.append(f'{indent}{part.strip()},')
            new_lines.append(f'{indent[:-4]}){suffix}')
            return "\n".join(new_lines)

    return None


def break_dict_or_list(line: str, max_len: int = LINE_LENGTH) -> str | None:
    """Try to break a long dict/list literal."""
    if len(line) <= max_len:
        return line

    stripped = line.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        # Dict literal
        content = stripped[1:-1].strip()
        if len(content) < 30:
            return None
        indent = re.match(r'^(\s*)', line).group(1)
        items = []
        current = ""
        depth = 0
        for char in content:
            if char in "([{":
                depth += 1
            elif char in ")}]":
                depth -= 1
            if char == "," and depth == 0:
                items.append(current)
                current = ""
            else:
                current += char
        if current:
            items.append(current)

        if len(items) > 1:
            new_lines = [f'{indent}{{']
            for item in items:
                new_lines.append(f'{indent}    {item.strip()},')
            new_lines.append(f'{indent}}}')
            result = "\n".join(new_lines)
            max_line_len = max(len(l) for l in result.split("\n"))
            if max_line_len <= max_len:
                return result

    return None


def break_comment(line: str, max_len: int = LINE_LENGTH) -> str | None:
    """Try to break a long comment line."""
    if len(line) <= max_len:
        return line

    m = re.match(r'^(\s*#\s*)(.*)$', line)
    if m:
        indent, text = m.groups()
        words = text.split()
        lines = [indent]
        for word in words:
            if len(lines[-1]) + 1 + len(word) > max_len:
                lines.append(indent + word)
            else:
                lines[-1] += " " + word if lines[-1] != indent else word
        return "\n".join(lines)

    return None


def try_fix_line(line: str, max_len: int = LINE_LENGTH) -> str | None:
    """Try to fix a single long line. Returns fixed line(s) or None if can't fix."""
    if len(line) <= max_len:
        return line

    # Skip lines that are data (test strings, URLs, etc.)
    stripped = line.strip()
    if stripped.startswith("http"):
        return None  # Can't break URLs meaningfully
    if re.match(r'^\s*["\']', stripped) and not re.search(r'\w+\s*=', stripped):
        # A bare string line - likely test data
        return None

    # Try different strategies
    for strategy in [
        break_function_def,
        break_function_call,
        break_dict_or_list,
        break_string_line,
        break_comment,
    ]:
        result = strategy(line, max_len)
        if result is not None:
            # Verify all resulting lines are under limit
            for l in result.split("\n"):
                if len(l) > max_len:
                    break
            else:
                return result

    return None


def fix_file(filepath: Path, violations: list[dict]) -> bool:
    """Fix all E501 violations in a file. Returns True if modified."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    modified = False
    # Process violations in reverse order of line number to preserve indices
    sorted_v = sorted(violations, key=lambda v: v["location"]["row"], reverse=True)

    for v in sorted_v:
        row = v["location"]["row"]
        idx = row - 1
        if idx >= len(lines):
            continue

        line = lines[idx]
        if len(line) <= LINE_LENGTH:
            continue

        # Try to fix the line
        fixed = try_fix_line(line.rstrip("\n"), LINE_LENGTH)
        if fixed is not None:
            lines[idx] = fixed + "\n"
            modified = True
        else:
            # Add noqa comment
            if not line.rstrip().endswith("# noqa: E501"):
                lines[idx] = line.rstrip("\n") + "  # noqa: E501\n"
                modified = True

    if modified:
        with open(filepath, "w") as f:
            f.writelines(lines)

    return modified


def main() -> None:
    violations = get_e501_violations()
    if not violations:
        print("No E501 violations found.")
        return

    # Group by file
    by_file: dict[Path, list[dict]] = {}
    for v in violations:
        path = Path(v["filename"])
        by_file.setdefault(path, []).append(v)

    fixed_count = 0
    noqa_count = 0
    for filepath, file_violations in by_file.items():
        modified = fix_file(filepath, file_violations)
        if modified:
            # Count how many were fixed vs noqa'd by re-checking
            with open(filepath, "r") as f:
                new_lines = f.readlines()
            for v in file_violations:
                row = v["location"]["row"]
                idx = row - 1
                if idx < len(new_lines) and len(new_lines[idx]) > LINE_LENGTH:
                    if "# noqa: E501" in new_lines[idx]:
                        noqa_count += 1
                else:
                    fixed_count += 1
            print(f"  Fixed {filepath}")

    print(f"\nFixed {fixed_count} lines, added noqa to {noqa_count} lines.")

    # Re-check
    result = subprocess.run(
        ["uv", "run", "ruff", "check", "src/", "tests/", "--select", "E501", "--output-format", "json"],
        capture_output=True,
        text=True,
    )
    remaining = json.loads(result.stdout) if result.stdout.strip() else []
    print(f"Remaining E501 violations: {len(remaining)}")


if __name__ == "__main__":
    main()
