from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from .config import BotConfig
from .execution_types import ExitResult
from .models import SymbolState
from .paper import PaperAccount

MessageHandler = Callable[[dict[str, Any]], None]


class AbstractMarketDataWS(ABC):
    """WebSocket market data stream -- exchange-specific implementation."""

    @abstractmethod
    def __init__(self, cfg: BotConfig, symbols: list[str], on_message: MessageHandler) -> None: ...

    @abstractmethod
    async def run_forever(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


class AbstractExecutionEngine(ABC):
    """Order execution -- exchange-specific implementation."""

    @abstractmethod
    def __init__(self, cfg: BotConfig, paper_account: PaperAccount | None = None) -> None: ...

    @abstractmethod
    async def initialize(self) -> dict[str, Any]: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def place_entry_limit(self, state: SymbolState, side: str) -> float: ...

    @abstractmethod
    async def place_exit(self, state: SymbolState) -> ExitResult: ...

    @abstractmethod
    async def fetch_open_positions(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def fetch_account_balance(self) -> float: ...

    @abstractmethod
    async def fetch_position_qty(self, symbol: str) -> float:
        """Return the absolute position size on exchange (0.0 if no position)."""
        ...


class AbstractMessageParser(ABC):
    """Parse raw WS messages into SymbolState -- exchange-specific implementation.

    Note: methods are defined as regular abstract methods rather than @staticmethod
    to ensure Python's ABC machinery enforces the contract on subclasses.
    Concrete implementations may still be stateless (no instance state required).
    """

    @abstractmethod
    def extract_symbol(self, msg: dict[str, Any]) -> str | None: ...

    @abstractmethod
    def ingest(self, state: SymbolState, msg: dict[str, Any]) -> None: ...
