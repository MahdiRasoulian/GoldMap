"""Signal Engine — Aggregates and prioritizes signals from all engines.

Combines outputs from Volume, Liquidity, Absorption, Stop Hunt,
and Fake Breakout engines into a unified signal stream.
"""

from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from config.loader import CONFIG
from engines.volume_engine import VolumeEngine
from engines.liquidity_engine import LiquidityEngine
from engines.absorption_engine import AbsorptionEngine
from engines.stop_hunt_engine import StopHuntEngine
from engines.fake_breakout_engine import FakeBreakoutEngine
from models.signals import Signal, SignalType, DataCategory


class SignalEngine:
    """Orchestrates all intelligence engines and produces unified signals.
    
    This is the central coordination point that:
    1. Feeds data to individual engines
    2. Collects their outputs
    3. Correlates signals across engines
    4. Produces a prioritized signal stream
    """
    
    def __init__(self):
        self.volume_engine = VolumeEngine()
        self.liquidity_engine = LiquidityEngine()
        self.absorption_engine = AbsorptionEngine()
        self.stop_hunt_engine = StopHuntEngine()
        self.fake_breakout_engine = FakeBreakoutEngine()
        
        self._signal_history: list[Signal] = []
        self._max_history = 1000
    
    def process(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all engines and return comprehensive analysis.
        
        Args:
            df: OHLC DataFrame with tick_volume.
            
        Returns:
            Dictionary with all engine outputs and unified signals.
        """
        if df.empty:
            return self._empty_result()
        
        # Run each engine
        volume_signals = self.volume_engine.analyze(df)
        liquidity_zones = self.liquidity_engine.analyze(df)
        absorption_signals = self.absorption_engine.analyze(df)
        
        # Stop hunt uses liquidity levels
        liquidity_levels = [z.midpoint for z in liquidity_zones[:20]]
        stop_hunt_signals = self.stop_hunt_engine.analyze(
            df, liquidity_levels
        )
        
        fake_breakout_signals = self.fake_breakout_engine.analyze(df)
        
        # Volume profile
        volume_profile = self.volume_engine.get_volume_profile(df)
        
        # Generate unified signals
        unified_signals = self._unify_signals(
            volume_signals,
            absorption_signals,
            stop_hunt_signals,
            fake_breakout_signals,
        )
        
        # Store in history
        self._signal_history.extend(unified_signals)
        if len(self._signal_history) > self._max_history:
            self._signal_history = self._signal_history[-self._max_history:]
        
        return {
            "timestamp": datetime.now(),
            "volume_signals": volume_signals,
            "liquidity_zones": liquidity_zones,
            "absorption_signals": absorption_signals,
            "stop_hunt_signals": stop_hunt_signals,
            "fake_breakout_signals": fake_breakout_signals,
            "volume_profile": volume_profile,
            "unified_signals": unified_signals,
            "active_alerts": self._get_active_alerts(unified_signals),
        }
    
    def _unify_signals(
        self,
        volume_signals,
        absorption_signals,
        stop_hunt_signals,
        fake_breakout_signals,
    ) -> list[Signal]:
        """Convert all engine outputs to unified Signal format."""
        unified = []
        
        for vs in volume_signals:
            if vs.is_spike:
                unified.append(Signal(
                    timestamp=vs.timestamp,
                    signal_type=SignalType.VOLUME_SPIKE,
                    price_level=vs.price,
                    confidence=min(1.0, vs.relative_volume / 5.0),
                    data_category=DataCategory.DERIVED,
                    description=(
                        f"Volume spike: {vs.relative_volume}x average"
                    ),
                    metadata={"relative_volume": vs.relative_volume},
                ))
        
        for ab in absorption_signals:
            unified.append(Signal(
                timestamp=ab.timestamp,
                signal_type=SignalType.ABSORPTION,
                price_level=ab.price_level,
                confidence=ab.confidence,
                data_category=DataCategory.ESTIMATED,
                description=(
                    f"{ab.direction.title()} absorption at "
                    f"{ab.price_level} (defended {ab.defense_count}x)"
                ),
                metadata={
                    "direction": ab.direction,
                    "defense_count": ab.defense_count,
                },
            ))
        
        for sh in stop_hunt_signals:
            unified.append(Signal(
                timestamp=sh.timestamp,
                signal_type=SignalType.STOP_HUNT,
                price_level=sh.trigger_price,
                confidence=sh.confidence,
                data_category=DataCategory.ESTIMATED,
                description=(
                    f"Stop hunt {sh.hunt_direction} {sh.trigger_price} "
                    f"(extreme: {sh.extreme_price})"
                ),
                metadata={
                    "direction": sh.hunt_direction,
                    "extreme": sh.extreme_price,
                    "reversal_speed": sh.reversal_speed,
                },
            ))
        
        for fb in fake_breakout_signals:
            unified.append(Signal(
                timestamp=fb.timestamp,
                signal_type=SignalType.FAKE_BREAKOUT,
                price_level=fb.breakout_price,
                confidence=fb.confidence,
                data_category=DataCategory.ESTIMATED,
                description=(
                    f"Fake breakout {fb.breakout_direction} at "
                    f"{fb.breakout_price} (returned to {fb.return_price})"
                ),
                metadata={
                    "direction": fb.breakout_direction,
                    "bars_outside": fb.bars_outside,
                },
            ))
        
        # Sort by confidence
        unified.sort(key=lambda s: s.confidence, reverse=True)
        
        return unified
    
    def _get_active_alerts(
        self, signals: list[Signal]
    ) -> list[dict]:
        """Get high-priority alerts from recent signals."""
        alerts = []
        
        for signal in signals:
            if signal.confidence >= 0.6:
                alerts.append({
                    "type": signal.signal_type.value,
                    "price": signal.price_level,
                    "confidence": signal.confidence,
                    "category": signal.data_category.value,
                    "message": signal.description,
                    "time": signal.timestamp.isoformat(),
                })
        
        return alerts[:10]  # Top 10 alerts
    
    def _empty_result(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(),
            "volume_signals": [],
            "liquidity_zones": [],
            "absorption_signals": [],
            "stop_hunt_signals": [],
            "fake_breakout_signals": [],
            "volume_profile": pd.DataFrame(),
            "unified_signals": [],
            "active_alerts": [],
        }