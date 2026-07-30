"""Stop Hunt Engine — Detects potential stop hunt patterns.

Data Classification:
- Input: OBSERVED (OHLC, tick_volume)
- Output: ESTIMATED (stop hunt signals)

Concept:
A stop hunt occurs when price aggressively moves to trigger stop-loss
orders clustered at a level, then immediately reverses. Pattern:
1. Price approaches a known level (swing high/low)
2. Aggressive breakout move (spike through level)
3. Immediate and strong reversal

Limitations:
- Cannot confirm actual stop-loss triggers
- Similar pattern can occur from legitimate breakout failures
- No way to see pending stop orders
- Reversal could be coincidental, not caused by stop triggering
"""

import numpy as np
import pandas as pd
from loguru import logger

from config.loader import CONFIG
from models.signals import StopHuntSignal, DataCategory


class StopHuntEngine:
    """Detects potential stop hunt patterns.
    
    All outputs are ESTIMATED — we infer stop hunts from
    characteristic price patterns.
    """
    
    def __init__(self):
        cfg = CONFIG["engines"]["stop_hunt"]
        self.breakout_pips = cfg["breakout_pips"] * 0.01
        self.reversal_pips = cfg["reversal_pips"] * 0.01
        self.time_window = cfg["time_window_bars"]
        self.volume_multiplier = cfg["volume_multiplier"]
    
    def analyze(
        self,
        df: pd.DataFrame,
        liquidity_levels: list[float] | None = None,
    ) -> list[StopHuntSignal]:
        """Detect stop hunt patterns.
        
        Args:
            df: OHLC DataFrame with tick_volume.
            liquidity_levels: Known liquidity levels to monitor.
                If None, uses swing highs/lows.
                
        Returns:
            List of StopHuntSignal objects.
        """
        if df.empty or len(df) < 50:
            return []
        
        signals = []
        
        # Get reference levels
        if liquidity_levels is None:
            liquidity_levels = self._find_reference_levels(df)
        
        # Calculate average volume for comparison
        avg_volume = df["tick_volume"].rolling(20).mean()
        
        # Scan for stop hunt patterns
        for i in range(self.time_window + 1, len(df)):
            window = df.iloc[i - self.time_window:i + 1]
            
            # Check each liquidity level
            for level in liquidity_levels:
                signal = self._check_stop_hunt_at_level(
                    window, level, avg_volume.iloc[i]
                )
                if signal:
                    signals.append(signal)
        
        # Deduplicate
        signals = self._deduplicate(signals)
        
        return signals
    
    def _find_reference_levels(self, df: pd.DataFrame) -> list[float]:
        """Find swing highs/lows as potential stop-loss cluster levels."""
        levels = []
        lookback = 10
        
        for i in range(lookback, len(df) - lookback):
            # Swing high
            if df["high"].iloc[i] == df["high"].iloc[
                i - lookback:i + lookback + 1
            ].max():
                levels.append(df["high"].iloc[i])
            
            # Swing low
            if df["low"].iloc[i] == df["low"].iloc[
                i - lookback:i + lookback + 1
            ].min():
                levels.append(df["low"].iloc[i])
        
        return levels
    
    def _check_stop_hunt_at_level(
        self,
        window: pd.DataFrame,
        level: float,
        avg_volume: float,
    ) -> StopHuntSignal | None:
        """Check if a stop hunt occurred at a specific level.
        
        Pattern for upside stop hunt:
        1. Price was below level
        2. Spike above level (breakout)
        3. Quick reversal back below
        
        Pattern for downside stop hunt:
        1. Price was above level
        2. Spike below level (breakout)
        3. Quick reversal back above
        """
        if window.empty or len(window) < 3:
            return None
        
        # Check upside stop hunt
        signal = self._check_upside_hunt(window, level, avg_volume)
        if signal:
            return signal
        
        # Check downside stop hunt
        signal = self._check_downside_hunt(window, level, avg_volume)
        if signal:
            return signal
        
        return None
    
    def _check_upside_hunt(
        self,
        window: pd.DataFrame,
        level: float,
        avg_volume: float,
    ) -> StopHuntSignal | None:
        """Detect upside stop hunt (hunt above level, reverse down)."""
        # Find the bar that broke above the level
        for i in range(1, len(window)):
            prev_close = window["close"].iloc[i - 1]
            current_high = window["high"].iloc[i]
            
            # Was below, spiked above
            if prev_close < level and current_high > level + self.breakout_pips:
                # Check for reversal in subsequent bars
                remaining = window.iloc[i:]
                
                if len(remaining) < 2:
                    continue
                
                extreme = remaining["high"].max()
                final_close = remaining["close"].iloc[-1]
                
                # Reversal condition: closed back below level
                reversal_size = extreme - final_close
                
                if (final_close < level and 
                    reversal_size > self.reversal_pips):
                    
                    # Volume check
                    hunt_volume = remaining["tick_volume"].max()
                    vol_ratio = hunt_volume / max(avg_volume, 1)
                    
                    # Calculate reversal speed
                    bars_for_reversal = len(remaining)
                    reversal_speed = (reversal_size * 100) / max(bars_for_reversal, 1)
                    
                    confidence = self._calculate_confidence(
                        reversal_size, vol_ratio, bars_for_reversal
                    )
                    
                    if confidence > 0.3:
                        return StopHuntSignal(
                            timestamp=window.index[i],
                            trigger_price=level,
                            extreme_price=round(extreme, 2),
                            reversal_price=round(final_close, 2),
                            hunt_direction="above",
                            volume_at_extreme=round(vol_ratio, 2),
                            reversal_speed=round(reversal_speed, 1),
                            confidence=round(confidence, 2),
                        )
        
        return None
    
    def _check_downside_hunt(
        self,
        window: pd.DataFrame,
        level: float,
        avg_volume: float,
    ) -> StopHuntSignal | None:
        """Detect downside stop hunt (hunt below level, reverse up)."""
        for i in range(1, len(window)):
            prev_close = window["close"].iloc[i - 1]
            current_low = window["low"].iloc[i]
            
            # Was above, spiked below
            if prev_close > level and current_low < level - self.breakout_pips:
                remaining = window.iloc[i:]
                
                if len(remaining) < 2:
                    continue
                
                extreme = remaining["low"].min()
                final_close = remaining["close"].iloc[-1]
                
                # Reversal condition
                reversal_size = final_close - extreme
                
                if (final_close > level and 
                    reversal_size > self.reversal_pips):
                    
                    hunt_volume = remaining["tick_volume"].max()
                    vol_ratio = hunt_volume / max(avg_volume, 1)
                    
                    bars_for_reversal = len(remaining)
                    reversal_speed = (reversal_size * 100) / max(bars_for_reversal, 1)
                    
                    confidence = self._calculate_confidence(
                        reversal_size, vol_ratio, bars_for_reversal
                    )
                    
                    if confidence > 0.3:
                        return StopHuntSignal(
                            timestamp=window.index[i],
                            trigger_price=level,
                            extreme_price=round(extreme, 2),
                            reversal_price=round(final_close, 2),
                            hunt_direction="below",
                            volume_at_extreme=round(vol_ratio, 2),
                            reversal_speed=round(reversal_speed, 1),
                            confidence=round(confidence, 2),
                        )
        
        return None
    
    def _calculate_confidence(
        self,
        reversal_size: float,
        volume_ratio: float,
        bars_for_reversal: int,
    ) -> float:
        """Calculate confidence for stop hunt signal."""
        # Reversal size component (0-0.4)
        rev_score = min(0.4, reversal_size / (self.reversal_pips * 3))
        
        # Volume component (0-0.3)
        vol_score = min(0.3, (volume_ratio - 1.0) * 0.15)
        
        # Speed component (0-0.3) — faster reversal = more likely hunt
        speed_score = min(0.3, (1.0 / max(bars_for_reversal, 1)) * 0.6)
        
        return min(1.0, rev_score + vol_score + speed_score)
    
    def _deduplicate(
        self, signals: list[StopHuntSignal]
    ) -> list[StopHuntSignal]:
        """Remove duplicate signals at same level."""
        if len(signals) <= 1:
            return signals
        
        seen_levels = set()
        unique = []
        
        for signal in signals:
            key = (
                round(signal.trigger_price, 1),
                signal.hunt_direction,
            )
            if key not in seen_levels:
                seen_levels.add(key)
                unique.append(signal)
        
        return unique