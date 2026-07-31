"""Data models for the Assistant layer."""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class SetupType(str, Enum):
    LIQUIDITY_REVERSAL = "liquidity_reversal"
    STOP_HUNT_REVERSAL = "stop_hunt_reversal"
    BREAKOUT_CONTINUATION = "breakout_continuation"
    ABSORPTION_REVERSAL = "absorption_reversal"
    EXHAUSTION = "exhaustion"


class State(str, Enum):
    IDLE = "idle"
    WATCH = "watch"
    ALERT = "alert"
    ACTION = "action"


@dataclass
class MarketContext:
    """Consolidated market data snapshot."""
    price: float = 0.0
    trend: str = "neutral"       # "bullish", "bearish", "neutral"
    session: str = "unknown"
    liquidity_zones: List[dict] = field(default_factory=list)
    stop_hunt_signals: List[dict] = field(default_factory=list)
    absorption_signals: List[dict] = field(default_factory=list)
    fake_breakout_signals: List[dict] = field(default_factory=list)
    volume_profile: List[dict] = field(default_factory=list)
    tick_volume: float = 0.0
    avg_volume: float = 0.0
    atr: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SetupResult:
    """Result of setup detection."""
    setup_type: Optional[SetupType] = None
    direction: str = "neutral"    # "buy" or "sell"
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class AssistantOutput:
    """Final assistant output."""
    state: State = State.IDLE
    setup: Optional[SetupType] = None
    confidence: int = 0           # 0–100
    price: float = 0.0
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    reasons: List[str] = field(default_factory=list)
    raw_score: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    setup_direction: str = "neutral"