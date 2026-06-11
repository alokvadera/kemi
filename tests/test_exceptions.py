"""Tests for src/kemi/exceptions.py."""

from __future__ import annotations

import pytest

from kemi.exceptions import (
    ConfigurationError,
    EmbeddingError,
    EncryptionError,
    IncompatibleSchemaError,
    KemiError,
    MigrationError,
    NotFoundError,
    StorageError,
    ValidationError,
)


class TestKemiErrorInheritance:
    def test_kemi_error_is_subclass_of_exception(self) -> None:
        assert issubclass(KemiError, Exception)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationError,
            ValidationError,
            NotFoundError,
            EmbeddingError,
            StorageError,
            MigrationError,
            IncompatibleSchemaError,
            EncryptionError,
        ],
    )
    def test_subclass_is_subclass_of_kemi_error(self, exc_cls: type[KemiError]) -> None:
        assert issubclass(exc_cls, KemiError)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationError,
            ValidationError,
            NotFoundError,
            EmbeddingError,
            StorageError,
            MigrationError,
            IncompatibleSchemaError,
            EncryptionError,
        ],
    )
    def test_subclass_is_transitively_subclass_of_exception(
        self, exc_cls: type[KemiError]
    ) -> None:
        assert issubclass(exc_cls, Exception)


class TestKemiErrorStrFormat:
    def test_str_no_context(self) -> None:
        e = KemiError("something went wrong")
        assert str(e) == "something went wrong"

    def test_str_with_context(self) -> None:
        e = KemiError("bad thing", foo=123, bar="baz")
        s = str(e)
        assert s == "bad thing (foo=123, bar='baz')"

    def test_str_with_single_context_entry(self) -> None:
        e = KemiError("oops", key="value")
        assert str(e) == "oops (key='value')"

    def test_str_empty_context_same_as_message(self) -> None:
        e = KemiError("simple")
        assert str(e) == e.message


class TestKemiErrorAttributes:
    def test_message_preserved(self) -> None:
        e = KemiError("test message")
        assert e.message == "test message"

    def test_context_is_dict(self) -> None:
        e = KemiError("msg", a=1, b=2)
        assert isinstance(e.context, dict)

    def test_context_empty_by_default(self) -> None:
        e = KemiError("msg")
        assert e.context == {}

    def test_context_populated_from_kwargs(self) -> None:
        e = KemiError("msg", x=10, y="hello")
        assert e.context == {"x": 10, "y": "hello"}


class TestSubclassRaiseAndCatch:
    @pytest.mark.parametrize(
        "exc_cls,msg",
        [
            (ConfigurationError, "config error"),
            (ValidationError, "validation error"),
            (NotFoundError, "not found error"),
            (EmbeddingError, "embedding error"),
            (StorageError, "storage error"),
            (MigrationError, "migration error"),
            (IncompatibleSchemaError, "schema error"),
            (EncryptionError, "encryption error"),
        ],
    )
    def test_can_raise_and_catch_as_self(
        self, exc_cls: type[KemiError], msg: str
    ) -> None:
        with pytest.raises(exc_cls) as exc_info:
            raise exc_cls(msg)
        assert exc_info.value.message == msg

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationError,
            ValidationError,
            NotFoundError,
            EmbeddingError,
            StorageError,
            MigrationError,
            IncompatibleSchemaError,
            EncryptionError,
        ],
    )
    def test_can_catch_as_kemi_error(self, exc_cls: type[KemiError]) -> None:
        with pytest.raises(KemiError):
            raise exc_cls("base-catch test")

    @pytest.mark.parametrize(
        "exc_cls,msg",
        [
            (ConfigurationError, "config with context"),
            (ValidationError, "validation with context"),
            (NotFoundError, "memory not found"),
            (EmbeddingError, "embed failed"),
            (StorageError, "sqlite error"),
            (MigrationError, "migration failed"),
            (IncompatibleSchemaError, "schema mismatch"),
            (EncryptionError, "decrypt failed"),
        ],
    )
    def test_subclass_context_carries_through(
        self, exc_cls: type[KemiError], msg: str
    ) -> None:
        e = exc_cls(msg, x=1, y="two")
        assert e.message == msg
        assert e.context == {"x": 1, "y": "two"}


class TestShadowing:
    def test_two_instances_dont_shadow(self) -> None:
        e1 = NotFoundError("first", id="a")
        e2 = NotFoundError("second", id="b")
        assert str(e1) == "first (id='a')"
        assert str(e2) == "second (id='b')"
        assert e1.context != e2.context


class TestAllExports:
    def test_all_exports_exactly_expected_names(self) -> None:
        from kemi import exceptions

        expected = {
            "KemiError",
            "ConfigurationError",
            "ValidationError",
            "NotFoundError",
            "EmbeddingError",
            "StorageError",
            "MigrationError",
            "IncompatibleSchemaError",
            "EncryptionError",
            "CompatibilityError",
        }
        assert set(exceptions.__all__) == expected

    def test_all_names_are_public(self) -> None:
        from kemi import exceptions

        for name in exceptions.__all__:
            assert hasattr(exceptions, name)

class TestBaseCatchUseCase:
    """Demonstrates the use case for the unified KemiError base class."""

    def test_except_kemi_error_catches_every_subclass(self) -> None:
        """A single 'except KemiError' catches every kemi-specific error."""
        for exc in (
            ConfigurationError("x"),
            ValidationError("x"),
            NotFoundError("x"),
            EmbeddingError("x"),
            StorageError("x"),
            MigrationError("x"),
            IncompatibleSchemaError("x"),
            EncryptionError("x"),
        ):
            try:
                raise exc
            except KemiError as caught:
                assert caught is exc
            else:
                pytest.fail(f"expected KemiError to catch {type(exc).__name__}")

    def test_except_kemi_error_does_not_catch_unrelated_exceptions(self) -> None:
        """KemiError is a leaf-like base — must not swallow Python builtins."""
        for unrelated in (KeyError("x"), TypeError("x"), RuntimeError("x")):
            with pytest.raises(type(unrelated)):
                try:
                    raise unrelated
                except KemiError:
                    pytest.fail(f"KemiError swallowed unrelated {type(unrelated).__name__}")

    def test_except_value_error_still_catches_validation_error(self) -> None:
        """Backward-compat: legacy 'except ValueError' keeps working."""
        try:
            raise ValidationError("bad input")
        except ValueError as caught:
            assert isinstance(caught, KemiError)

    def test_except_runtime_error_still_catches_embedding_error(self) -> None:
        """Backward-compat: legacy 'except RuntimeError' keeps working."""
        try:
            raise EmbeddingError("model down")
        except RuntimeError as caught:
            assert isinstance(caught, KemiError)

    def test_except_oserror_still_catches_storage_error(self) -> None:
        """Backward-compat: 'except OSError' catches StorageError (file-mirror)."""
        try:
            raise StorageError("disk full")
        except OSError as caught:
            assert isinstance(caught, KemiError)

    def test_except_lookup_error_still_catches_not_found_error(self) -> None:
        """Backward-compat: 'except LookupError' catches NotFoundError."""
        try:
            raise NotFoundError("missing")
        except LookupError as caught:
            assert isinstance(caught, KemiError)

    def test_kemi_error_is_subclass_of_exception(self) -> None:
        """KemiError can be caught with the bare 'except:' in generic handlers."""
        try:
            raise KemiError("oops")
        except Exception as caught:
            assert isinstance(caught, KemiError)
