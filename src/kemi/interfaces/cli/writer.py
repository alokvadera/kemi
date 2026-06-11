"""CLI output writer abstraction.

Provides three writers that share a uniform ``write``/``error`` interface:

- :class:`ConsoleWriter` — human-readable, the default.
- :class:`JsonWriter` — one JSON object per line (NDJSON), for scripts.
- :class:`SilentWriter` — drops ``info`` messages, keeps ``error`` and
  ``warn``.

The CLI entry point parses ``--json`` / ``--quiet`` flags and passes
the chosen writer to every subcommand as ``args.writer``.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Protocol, TextIO


class Writer(Protocol):
    """Protocol every CLI writer satisfies."""

    def write(
                 self,
                 message: str,
                 *,
                 kind: str = "info",
                 flush: bool = False,
                 end: str = "\n",
             ) -> None:
        """Emit a message. ``kind`` is one of info/warn/error.

        ``flush`` is honoured for streaming use cases (recall-stream).
        ``end`` controls the trailing newline (default ``"\n"``; pass
        ``""`` for interactive prompts).
        """
        ...

    def error(self, message: str) -> None:
        """Emit an error message. Always shown, even in --quiet mode."""
        ...

    def warn(self, message: str) -> None:
        """Emit a warning message."""
        ...


class ConsoleWriter:
    """Default human-readable writer. Prints to ``stream`` (default stdout).

    The default stream is looked up lazily so that ``capsys``/``capfd``
    redirections from ``pytest`` take effect.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._explicit_stream = stream

    def _stream(self) -> TextIO:
        return self._explicit_stream or sys.stdout

    def write(
                 self,
                 message: str,
                 *,
                 kind: str = "info",
                 flush: bool = False,
                 end: str = "\n",
             ) -> None:
        s = self._stream()
        print(message, file=s, flush=flush)

    def error(self, message: str) -> None:
        print(f"Error: {message}", file=sys.stderr)

    def warn(self, message: str) -> None:
        print(f"Warning: {message}", file=sys.stderr)


class JsonWriter:
    """NDJSON writer — one JSON object per line, on stdout.

    Useful for piping into ``jq`` or other tools. Use ``kind`` to
    distinguish info/warn/error entries.
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def _emit(self, payload: dict[str, Any]) -> None:
        self._stream.write(json.dumps(payload, default=str) + "\n")
        self._stream.flush()

    def write(
                 self,
                 message: str,
                 *,
                 kind: str = "info",
                 flush: bool = False,
                 end: str = "\n",
             ) -> None:
        self._emit({"level": kind, "message": message})

    def error(self, message: str) -> None:
        self._emit({"level": "error", "message": message})

    def warn(self, message: str) -> None:
        self._emit({"level": "warn", "message": message})


class SilentWriter:
    """Drops ``info`` messages; keeps ``error`` and ``warn``."""

    def write(
                 self,
                 message: str,
                 *,
                 kind: str = "info",
                 flush: bool = False,
                 end: str = "\n",
             ) -> None:
        if kind != "info":
            print(message, file=sys.stderr)

    def error(self, message: str) -> None:
        print(f"Error: {message}", file=sys.stderr)

    def warn(self, message: str) -> None:
        print(f"Warning: {message}", file=sys.stderr)


def make_writer(json_mode: bool = False, quiet: bool = False) -> Writer:
    """Construct the writer chosen by ``--json`` and ``--quiet`` flags.

    - ``json_mode=True, quiet=False``  → JsonWriter
    - ``json_mode=False, quiet=True``  → SilentWriter
    - ``json_mode=True, quiet=True``   → JsonWriter (quiet ignored)
    - default                          → ConsoleWriter
    """
    if json_mode:
        return JsonWriter()
    if quiet:
        return SilentWriter()
    return ConsoleWriter()
