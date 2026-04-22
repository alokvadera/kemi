from typing import Callable

from kemi.adapters.base import StorageAdapter
from kemi.models import LifecycleState, MemoryObject


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
    ):
        self._fns = {
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

    def _get_fn(self, name: str):
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
    ) -> list[MemoryObject]:
        return self._get_fn("search")(user_id, query_embedding, top_k, lifecycle_filter)

    def get(self, memory_id: str) -> MemoryObject | None:
        return self._get_fn("get")(memory_id)

    def update(self, memory: MemoryObject) -> None:
        self._get_fn("update")(memory)

    def delete_by_user(self, user_id: str) -> int:
        return self._get_fn("delete_by_user")(user_id)

    def delete_by_id(self, memory_id: str) -> bool:
        return self._get_fn("delete_by_id")(memory_id)

    def get_all_by_user(
        self,
        user_id: str,
        lifecycle_filter: list[LifecycleState] | None = None,
    ) -> list[MemoryObject]:
        return self._get_fn("get_all_by_user")(user_id, lifecycle_filter)

    def count(self, user_id: str) -> int:
        return self._get_fn("count")(user_id)

    def get_all(self) -> list[MemoryObject]:
        fn = self._fns.get("get_all")
        if fn is None:
            raise NotImplementedError(
                "get_all not implemented in your CustomStorageAdapter. "
                "Add a get_all function to use export()"
            )
        return fn()

    def get_all_users(self) -> list[str]:
        fn = self._fns.get("get_all_users")
        if fn is None:
            fn = self._fns.get("get_all")
            if fn is None:
                raise NotImplementedError(
                    "get_all_users not implemented in your CustomStorageAdapter. "
                    "Add a get_all_users function to use list_users()"
                )
            all_memories = fn()
            return list(set(m.user_id for m in all_memories))
        return fn()

    def upgrade_schema(self, from_version: int, to_version: int) -> None:
        self._get_fn("upgrade_schema")(from_version, to_version)
