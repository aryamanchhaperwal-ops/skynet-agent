"""Long-term memory for Skynet (Phase P2).

Public surface: :class:`MemoryManager` (the single doorway), the
:class:`agency.memory.models` domain models, storage backends behind
:class:`agency.memory.stores.MemoryStore`, deterministic extraction from
observations/conversations/experiences, and the ``memory_*`` actions.
"""

from agency.memory.manager import MemoryManager
from agency.memory.models import MemoryRecord, MemoryStats, MemoryType, ScoredMemory
from agency.memory.stores import MemoryStore, build_memory_store

__all__ = [
    "MemoryManager",
    "MemoryRecord",
    "MemoryStats",
    "MemoryStore",
    "MemoryType",
    "ScoredMemory",
    "build_memory_store",
]
