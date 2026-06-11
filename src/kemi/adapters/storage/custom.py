import inspect
from collections.abc import Callable
from typing import Any, cast

from kemi.adapters.base import StorageAdapter
from kemi.memory.model import LifecycleState, MemoryObject

_FnMap = dict[
    str,
    Callable[..., Any] | None,
]


def _accepts_param(fn: Callable[..., Any], param: str) -> bool:
    """Check if a callable accepts a given parameter name."""
    try:
        sig = inspect.signature(fn)
        return param in sig.parameters
    except (ValueError, TypeError):
        return False


class CustomStorageAdapter(StorageAdapter):
    """Custom storage adapter that delegates to user-provided functions.

    Zero external dependencies.
    """

    def __init__(
        self,
        store_fn: Callable[[MemoryObject], None] | None = None,
        search_fn: Callable[..., list[MemoryObject]] | None = None,
        get_fn: Callable[[str], MemoryObject | None] | None = None,
        update_fn: Callable[[MemoryObject], None] | None = None,
        delete_by_user_fn: Callable[[str], int] | None = None,
        delete_by_id_fn: Callable[[str], bool] | None = None,
        get_all_by_user_fn: Callable[..., list[MemoryObject]] | None = None,
        get_all_fn: Callable[[], list[MemoryObject]] | None = None,
        count_fn: Callable[[str], int] | None = None,
        upgrade_schema_fn: Callable[[int, int], None] | None = None,
    ) -> None:
        self._fns: _FnMap = {
            "store": store_fn,
            "search": search_fn,
            "get": get_fn,
            "update": update_fn,
            "delete_by_user": delete_by_user_fn,
            "delete_by_id": delete_by_id_fn,
            "get_all_by_user": get_all_by_user_fn,
            "get_all": get_all_fn,
            "count": count_fn,
            "upgrade_schema": upgrade_schema_fn,
        }

    def _get_fn(self, name: str) -> Callable[..., Any]:
        fn = self._fns.get(name)
        if fn is None:
            raise NotImplementedError(
                f"CustomStorageAdapter: method '{name}' not provided. Pass it to __init__."
            )
        return fn

    def store(self, memory: MemoryObject) -> None:
        self._get_fn("store")(memory)

    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
    ) -> list[MemoryObject]:
        fn = self._get_fn("search")
        kwargs: dict[str, Any] = {}
        if _accepts_param(fn, "namespace"):
            kwargs["namespace"] = namespace
        if _accepts_param(fn, "session_id"):
            kwargs["session_id"] = session_id
        return cast(
            "list[MemoryObject]",
            fn(user_id, query_embedding, top_k, lifecycle_filter, **kwargs),
        )

    def get(self, memory_id: str) -> MemoryObject | None:
        return cast(
            "MemoryObject | None",
            self._get_fn("get")(memory_id),
        )

    def update(self, memory: MemoryObject) -> None:
        self._get_fn("update")(memory)

    def delete_by_user(self, user_id: str) -> int:
        return cast(int, self._get_fn("delete_by_user")(user_id))

    def delete_by_id(self, memory_id: str) -> bool:
        return cast(bool, self._get_fn("delete_by_id")(memory_id))

    def get_all_by_user(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
        session_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        fn = self._get_fn("get_all_by_user")
        kwargs: dict[str, Any] = {}
        if _accepts_param(fn, "namespace"):
            kwargs["namespace"] = namespace
        if _accepts_param(fn, "session_id"):
            kwargs["session_id"] = session_id
        results = cast(
            "list[MemoryObject]",
            fn(user_id, lifecycle_filter, **kwargs),
        )
        if offset is not None:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]
        return results

    def count(self, user_id: str) -> int:
        return cast(int, self._get_fn("count")(user_id))

    def get_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[MemoryObject]:
        fn = self._fns.get("get_all")
        if fn is None:
            raise NotImplementedError(
                "get_all not implemented in your CustomStorageAdapter. "
                "Add a get_all function to use export()"
            )
        results = cast("list[MemoryObject]", fn())
        if offset is not None:
            results = results[offset:]
        if limit is not None:
            results = results[:limit]
        return results

    def get_all_users(self) -> list[str]:
        fn = self._fns.get("get_all_users")
        if fn is None:
            fn = self._fns.get("get_all")
            if fn is None:
                raise NotImplementedError(
                    "get_all_users not implemented in your CustomStorageAdapter. "
                    "Add a get_all_users function to use list_users()"
                )
            all_memories = cast("list[MemoryObject]", fn())
            return list(set(m.user_id for m in all_memories))
        return cast("list[str]", fn())

    def get_by_tag(
        self,
        user_id: str,
        tag: str,
        lifecycle_filter: list[LifecycleState] | None = None,
        namespace: str = "default",
    ) -> list[MemoryObject]:
        fn = self._get_fn("get_by_tag")
        kwargs: dict[str, Any] = {}
        if _accepts_param(fn, "namespace"):
            kwargs["namespace"] = namespace
        return cast(
            "list[MemoryObject]",
            fn(user_id, tag, lifecycle_filter, **kwargs),
        )

    def upgrade_schema(
        self, from_version: int | None = None, to_version: int | None = None
    ) -> int:
        fn = self._get_fn("upgrade_schema")
        from_v = from_version if from_version is not None else 0
        to_v = to_version if to_version is not None else 1
        fn(from_v, to_v)
        return to_v
