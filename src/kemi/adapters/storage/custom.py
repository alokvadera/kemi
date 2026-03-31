from typing import Any, Callable, Dict, List, Optional

from kemi.adapters.base import StorageAdapter
from kemi.models import LifecycleState, MemoryObject


class CustomStorageAdapter(StorageAdapter):
    """Custom storage adapter that delegates to user-provided functions.

    Zero external dependencies.
    """

    def __init__(
        self,
        store_fn: Optional[Callable[[MemoryObject], None]] = None,
        search_fn: Optional[Callable[..., List[MemoryObject]]] = None,
        get_fn: Optional[Callable[[str], Optional[MemoryObject]]] = None,
        update_fn: Optional[Callable[[MemoryObject], None]] = None,
        delete_by_user_fn: Optional[Callable[[str], int]] = None,
        delete_by_id_fn: Optional[Callable[[str], bool]] = None,
        get_all_by_user_fn: Optional[Callable[..., List[MemoryObject]]] = None,
        count_fn: Optional[Callable[[str], int]] = None,
        upgrade_schema_fn: Optional[Callable[[int, int], None]] = None,
    ):
        self._fns = {
            "store": store_fn,
            "search": search_fn,
            "get": get_fn,
            "update": update_fn,
            "delete_by_user": delete_by_user_fn,
            "delete_by_id": delete_by_id_fn,
            "get_all_by_user": get_all_by_user_fn,
            "count": count_fn,
            "upgrade_schema": upgrade_schema_fn,
        }

    def _get_fn(self, name: str):
        fn = self._fns.get(name)
        if fn is None:
            raise NotImplementedError(
                f"CustomStorageAdapter: method '{name}' not provided. "
                f"Pass it to __init__."
            )
        return fn

    def store(self, memory: MemoryObject) -> None:
        self._get_fn("store")(memory)

    def search(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        lifecycle_filter: Optional[List[LifecycleState]] = None,
    ) -> List[MemoryObject]:
        return self._get_fn("search")(user_id, query_embedding, top_k, lifecycle_filter)

    def get(self, memory_id: str) -> Optional[MemoryObject]:
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
        lifecycle_filter: Optional[List[LifecycleState]] = None,
    ) -> List[MemoryObject]:
        return self._get_fn("get_all_by_user")(user_id, lifecycle_filter)

    def count(self, user_id: str) -> int:
        return self._get_fn("count")(user_id)

    def upgrade_schema(self, from_version: int, to_version: int) -> None:
        self._get_fn("upgrade_schema")(from_version, to_version)
