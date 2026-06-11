"""Tests for src/kemi/cli_writer.py — CLI output writers."""

import json
from io import StringIO

import pytest

from kemi.interfaces.cli.writer import (
    ConsoleWriter,
    JsonWriter,
    SilentWriter,
    make_writer,
)


class TestConsoleWriter:
    """Tests for ConsoleWriter."""

    def test_write_defaults_to_stdout(self, capsys: pytest.CaptureFixture) -> None:
        writer = ConsoleWriter()
        writer.write("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_write_to_explicit_stream(self) -> None:
        stream = StringIO()
        writer = ConsoleWriter(stream=stream)
        writer.write("hello")
        assert "hello" in stream.getvalue()

    def test_write_with_flush(self) -> None:
        stream = StringIO()
        writer = ConsoleWriter(stream=stream)
        writer.write("no newline", end="", flush=True)
        # StringIO doesn't flush, but we verify no error is raised
        assert "no newline" in stream.getvalue()

    def test_error_goes_to_stderr(self, capsys: pytest.CaptureFixture) -> None:
        writer = ConsoleWriter()
        writer.error("bad thing")
        captured = capsys.readouterr()
        assert "Error: bad thing" in captured.err

    def test_warn_goes_to_stderr(self, capsys: pytest.CaptureFixture) -> None:
        writer = ConsoleWriter()
        writer.warn("careful")
        captured = capsys.readouterr()
        assert "Warning: careful" in captured.err

    def test_write_info_kind(self, capsys: pytest.CaptureFixture) -> None:
        writer = ConsoleWriter()
        writer.write("info msg", kind="info")
        captured = capsys.readouterr()
        assert "info msg" in captured.out

    def test_write_warn_kind_goes_to_stdout(self, capsys: pytest.CaptureFixture) -> None:
        writer = ConsoleWriter()
        writer.write("warn msg", kind="warn")
        captured = capsys.readouterr()
        assert "warn msg" in captured.out

    def test_write_error_kind_goes_to_stdout(self, capsys: pytest.CaptureFixture) -> None:
        writer = ConsoleWriter()
        writer.write("err msg", kind="error")
        captured = capsys.readouterr()
        assert "err msg" in captured.out

    def test_write_empty_string(self, capsys: pytest.CaptureFixture) -> None:
        writer = ConsoleWriter()
        writer.write("")
        captured = capsys.readouterr()
        assert captured.out == "\n"


class TestJsonWriter:
    """Tests for JsonWriter."""

    def test_write_outputs_json(self) -> None:
        stream = StringIO()
        writer = JsonWriter(stream=stream)
        writer.write("hello")
        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["level"] == "info"
        assert data["message"] == "hello"

    def test_error_outputs_json(self) -> None:
        stream = StringIO()
        writer = JsonWriter(stream=stream)
        writer.error("bad thing")
        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["level"] == "error"
        assert data["message"] == "bad thing"

    def test_warn_outputs_json(self) -> None:
        stream = StringIO()
        writer = JsonWriter(stream=stream)
        writer.warn("careful")
        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["level"] == "warn"
        assert data["message"] == "careful"

    def test_multiple_writes_are_ndjson(self) -> None:
        stream = StringIO()
        writer = JsonWriter(stream=stream)
        writer.write("msg1")
        writer.write("msg2")
        lines = stream.getvalue().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["message"] == "msg1"
        assert json.loads(lines[1])["message"] == "msg2"

    def test_write_empty_message(self) -> None:
        stream = StringIO()
        writer = JsonWriter(stream=stream)
        writer.write("")
        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["message"] == ""

    def test_write_with_kind(self) -> None:
        stream = StringIO()
        writer = JsonWriter(stream=stream)
        writer.write("hello", kind="warn")
        output = stream.getvalue().strip()
        data = json.loads(output)
        assert data["level"] == "warn"

    def test_write_flush_param_noop(self) -> None:
        # JsonWriter._emit always flushes; flush param is accepted but ignored
        stream = StringIO()
        writer = JsonWriter(stream=stream)
        writer.write("flush test", flush=True)
        output = stream.getvalue().strip()
        assert json.loads(output)["message"] == "flush test"

    def test_write_end_param_noop(self) -> None:
        # JsonWriter._emit always adds newline; end param is accepted but ignored
        stream = StringIO()
        writer = JsonWriter(stream=stream)
        writer.write("end test", end="")
        output = stream.getvalue().strip()
        assert json.loads(output)["message"] == "end test"


class TestSilentWriter:
    """Tests for SilentWriter."""

    def test_write_info_is_silenced(self, capsys: pytest.CaptureFixture) -> None:
        writer = SilentWriter()
        writer.write("hello")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_write_warn_is_shown(self, capsys: pytest.CaptureFixture) -> None:
        writer = SilentWriter()
        writer.write("careful", kind="warn")
        captured = capsys.readouterr()
        assert "careful" in captured.err

    def test_write_error_is_shown(self, capsys: pytest.CaptureFixture) -> None:
        writer = SilentWriter()
        writer.write("bad", kind="error")
        captured = capsys.readouterr()
        assert "bad" in captured.err

    def test_error_always_shown(self, capsys: pytest.CaptureFixture) -> None:
        writer = SilentWriter()
        writer.error("always visible")
        captured = capsys.readouterr()
        assert "Error: always visible" in captured.err

    def test_warn_always_shown(self, capsys: pytest.CaptureFixture) -> None:
        writer = SilentWriter()
        writer.warn("warn visible")
        captured = capsys.readouterr()
        assert "Warning: warn visible" in captured.err

    def test_write_with_flush_and_end(self, capsys: pytest.CaptureFixture) -> None:
        writer = SilentWriter()
        writer.write("warn msg", kind="warn", flush=True, end="")
        captured = capsys.readouterr()
        assert "warn msg" in captured.err


class TestMakeWriter:
    """Tests for make_writer factory."""

    def test_default_returns_console(self) -> None:
        writer = make_writer()
        assert isinstance(writer, ConsoleWriter)

    def test_json_mode_returns_json(self) -> None:
        writer = make_writer(json_mode=True)
        assert isinstance(writer, JsonWriter)

    def test_quiet_returns_silent(self) -> None:
        writer = make_writer(quiet=True)
        assert isinstance(writer, SilentWriter)

    def test_json_and_quiet_prefers_json(self) -> None:
        writer = make_writer(json_mode=True, quiet=True)
        assert isinstance(writer, JsonWriter)

    def test_false_flags_return_console(self) -> None:
        writer = make_writer(json_mode=False, quiet=False)
        assert isinstance(writer, ConsoleWriter)
