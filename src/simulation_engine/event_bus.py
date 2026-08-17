"""
event_bus.py
Lightweight in-process pub/sub event bus.
The simulation engine publishes SimulationSnapshot every tick;
all other modules subscribe and react asynchronously.
"""
from __future__ import annotations
import asyncio
import queue
import threading
import logging
from typing import Callable, Dict, List, Any

logger = logging.getLogger(__name__)


class EventBus:
    """
    Thread-safe synchronous pub/sub bus.
    Suitable for MVP single-process operation.

    For production scale: swap internals for Redis Pub/Sub
    while keeping the same subscriber API.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._history: List[Dict[str, Any]] = []
        self._max_history: int = 200   # keep last N snapshots in memory

    def subscribe(self, event_type: str, callback: Callable):
        """Register a callback for an event type."""
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    cb for cb in self._subscribers[event_type] if cb != callback
                ]

    def publish(self, event_type: str, data: Any):
        """
        Publish an event synchronously.
        All subscribers for this event_type are called in registration order.
        """
        with self._lock:
            subscribers = list(self._subscribers.get(event_type, []))

        for cb in subscribers:
            try:
                cb(data)
            except Exception as e:
                logger.warning(f"EventBus: subscriber {cb.__name__} raised {e}")

        # Cache simulation snapshots for API polling
        if event_type == "simulation.tick":
            with self._lock:
                self._history.append(data)
                if len(self._history) > self._max_history:
                    self._history.pop(0)

    def latest_snapshot(self) -> Any:
        with self._lock:
            return self._history[-1] if self._history else None

    def recent_snapshots(self, n: int = 60) -> List[Any]:
        with self._lock:
            return list(self._history[-n:])

    def clear_history(self):
        with self._lock:
            self._history.clear()


# ── Async wrapper for FastAPI WebSocket streaming ─────────────────────────────
class AsyncEventQueue:
    """
    Wraps EventBus with an asyncio.Queue so FastAPI WebSocket
    handlers can await new snapshots without blocking.
    """

    def __init__(self, bus: EventBus, event_type: str = "simulation.tick",
                 maxsize: int = 50):
        self._q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        bus.subscribe(event_type, self._enqueue)

    def _enqueue(self, data: Any):
        try:
            self._q.put_nowait(data)
        except asyncio.QueueFull:
            pass   # drop oldest if consumer is slow

    async def get(self) -> Any:
        return await self._q.get()


# Singleton bus used across the application
_bus: EventBus = EventBus()

def get_bus() -> EventBus:
    return _bus
