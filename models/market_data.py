"""Market data models — Observed data structures."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Tick(BaseModel):
    """Single tick from MT5 — OBSERVED data."""
    
    time: datetime
    bid: float
    ask: float
    last: float = 0.0
    volume: int = 0
    flags: int = 0
    spread: float = Field(default=0.0, description="Computed: ask - bid")
    
    def model_post_init(self, __context) -> None:
        if self.spread == 0.0 and self.ask > 0 and self.bid > 0:
            object.__setattr__(self, 'spread', self.ask - self.bid)


class Candle(BaseModel):
    """OHLC candle from MT5 — OBSERVED data."""
    
    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int
    real_volume: int = 0


class TickBuffer(BaseModel):
    """Buffer of recent ticks for processing."""
    
    ticks: list[Tick] = Field(default_factory=list)
    max_size: int = 10000
    
    def add(self, tick: Tick) -> None:
        self.ticks.append(tick)
        if len(self.ticks) > self.max_size:
            self.ticks = self.ticks[-self.max_size:]
    
    @property
    def count(self) -> int:
        return len(self.ticks)
    
    @property
    def latest(self) -> Optional[Tick]:
        return self.ticks[-1] if self.ticks else None


class MarketSnapshot(BaseModel):
    """Current market state — combination of OBSERVED and DERIVED data."""
    
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    spread: float
    last_price: float
    tick_volume_1m: int = 0
    atr_14: float = 0.0
    session: str = "unknown"
    
    # Metadata
    data_category: str = "mixed"  # observed + derived