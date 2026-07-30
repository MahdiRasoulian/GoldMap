"""Fake Breakout Engine — Detects failed breakout patterns.

Data Classification:
- Input: OBSERVED (OHLC, tick_volume)
- Output: ESTIMATED (fake breakout signals)

Concept:
A fake breakout occurs when price breaks a significant level but
fails to sustain the move and returns into the prior range:
1. Price breaks above/below a level
2. Weak continuation (low volume, small follow-through)
3. Price returns back into the range

Limitations:
- Cannot distinguish fake breakout from legitimate pullback
- Breakout may still succeed after initial failure
- No way to confirm if breakout was "engineered"
- Pattern recognition is probabilistic, not deterministic
"""

import numpy as np
import pandas as pd
from loguru import logger

from config.loader import CONFIG
from models.signals import FakeBreakoutSignal, DataCategory


class FakeBreakoutEngine:
    """Detects potential fake breakout patterns.
    
    All outputs are ESTIMATED — we infer fake breakouts from
    characteristic price/volume patterns.
    """
    
    def __init__(self):
        cfg = CONFIG["engines"]["fake_breakout"]
        self.breakout_threshold = cfg["breakout_threshold_pips"] * 0.01
        self.return_threshold = cfg["return_threshold_pips"] * 0.01
        self.continuation_bars = cfg["continuation_bars"]
        self.weak_continuation_ratio = cfg["weak_continuation_atr_ratio"]
    
    def analyze(self, df: pd.DataFrame) -> list[FakeBreakoutSignal]:
        """Detect fake breakout patterns.
        
        Args:
            df: OHLC DataFrame with tick_volume.
            
        Returns:
            List of FakeBreakoutSignal objects.
        """
        if df.empty or len(df) < 100:
            return []
        
        signals = []
        
        # Calculate ATR for context
        atr = self._calculate_atr(df, period=14)
        
        # Find consolidation ranges
        ranges = self._find_ranges(df)
        
        # Check each range for fake breakouts
        for range_info in ranges:
            range_signals = self._check_range_breakouts(
                df, range_info, atr
            )
            signals.extend(range_signals)
        
        return signals
    
    def _find_ranges(self, df: pd.DataFrame) -> list[dict]:
        """Identify consolidation ranges in price data.
        
        A range is defined as a period where price oscillates
        between defined high and low boundaries.
        """
        ranges = []
        window = 20  # Bars to define a range
        
        for i in range(window, len(df) - self.continuation_bars - 1):
            segment = df.iloc[i - window:i]
            
            range_high = segment["high"].max()
            range_low = segment["low"].min()
            range_size = range_high - range_low
            
            # Check if it's actually a range (not trending)
            # Range condition: price stayed within bounds for most bars
            bars_in_range = 0
            for _, row in segment.iterrows():
                if row["high"] <= range_high and row["low"] >= range_low:
                    bars_in_range += 1
            
            containment_ratio = bars_in_range / window
            
            if containment_ratio > 0.8:  # 80% of bars within range
                ranges.append({
                    "high": range_high,
                    "low": range_low,
                    "size": range_size,
                    "end_idx": i,
                    "start_idx": i - window,
                })
        
        # Deduplicate overlapping ranges
        ranges = self._deduplicate_ranges(ranges)
        
        return ranges
    
    def _check_range_breakouts(
        self,
        df: pd.DataFrame,
        range_info: dict,
        atr: pd.Series,
    ) -> list[FakeBreakoutSignal]:
        """Check if breakouts from a range failed."""
        signals = []
        
        end_idx = range_info["end_idx"]
        range_high = range_info["high"]
        range_low = range_info["low"]
        
        # Look at bars after the range
        post_range = df.iloc[end_idx:end_idx + self.continuation_bars + 5]
        
        if len(post_range) < self.continuation_bars:
            return signals
        
        # Check upside fake breakout
        signal = self._check_upside_fake(
            post_range, range_high, range_low, atr.iloc[end_idx]
        )
        if signal:
            signals.append(signal)
        
        # Check downside fake breakout
        signal = self._check_downside_fake(
            post_range, range_high, range_low, atr.iloc[end_idx]
        )
        if signal:
            signals.append(signal)
        
        return signals
    
    def _check_upside_fake(
        self,
        post_range: pd.DataFrame,
        range_high: float,
        range_low: float,
        current_atr: float,
    ) -> FakeBreakoutSignal | None:
        """Check for failed upside breakout."""
        # Did price break above range?
        max_high = post_range["high"].max()
        breakout_distance = max_high - range_high
        
        if breakout_distance < self.breakout_threshold:
            return None  # No breakout occurred
        
        # Find the breakout bar
        breakout_idx = post_range["high"].idxmax()
        breakout_bar_pos = post_range.index.get_loc(breakout_idx)
        
        # Check continuation after breakout
        after_breakout = post_range.iloc[breakout_bar_pos:]
        
        if len(after_breakout) < 2:
            return None
        
        # Measure continuation strength
        continuation_distance = after_breakout["close"].iloc[-1] - range_high
        continuation_ratio = continuation_distance / max(current_atr, 0.01)
        
        # Weak continuation check
        if continuation_ratio > self.weak_continuation_ratio:
            return None  # Breakout is holding — not fake
        
        # Did price return into range?
        final_close = after_breakout["close"].iloc[-1]
        returned = final_close < range_high + self.return_threshold
        
        if not returned:
            return None
        
        # Count bars outside range
        bars_outside = sum(
            1 for _, row in after_breakout.iterrows()
            if row["close"] > range_high
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            breakout_distance, continuation_ratio, bars_outside, current_atr
        )
        
        return FakeBreakoutSignal(
            timestamp=breakout_idx,
            breakout_price=round(range_high, 2),
            extreme_price=round(max_high, 2),
            return_price=round(final_close, 2),
            breakout_direction="up",
            continuation_strength=round(
                abs(continuation_ratio), 3
            ),
            bars_outside=bars_outside,
            confidence=round(confidence, 2),
        )
    
    def _check_downside_fake(
        self,
        post_range: pd.DataFrame,
        range_high: float,
        range_low: float,
        current_atr: float,
    ) -> FakeBreakoutSignal | None:
        """Check for failed downside breakout."""
        min_low = post_range["low"].min()
        breakout_distance = range_low - min_low
        
        if breakout_distance < self.breakout_threshold:
            return None
        
        breakout_idx = post_range["low"].idxmin()
        breakout_bar_pos = post_range.index.get_loc(breakout_idx)
        
        after_breakout = post_range.iloc[breakout_bar_pos:]
        
        if len(after_breakout) < 2:
            return None
        
        continuation_distance = range_low - after_breakout["close"].iloc[-1]
        continuation_ratio = continuation_distance / max(current_atr, 0.01)
        
        if continuation_ratio > self.weak_continuation_ratio:
            return None
        
        final_close = after_breakout["close"].iloc[-1]
        returned = final_close > range_low - self.return_threshold
        
        if not returned:
            return None
        
        bars_outside = sum(
            1 for _, row in after_breakout.iterrows()
            if row["close"] < range_low
        )
        
        confidence = self._calculate_confidence(
            breakout_distance, continuation_ratio, bars_outside, current_atr
        )
        
        return FakeBreakoutSignal(
            timestamp=breakout_idx,
            breakout_price=round(range_low, 2),
            extreme_price=round(min_low, 2),
            return_price=round(final_close, 2),
            breakout_direction="down",
            continuation_strength=round(
                abs(continuation_ratio), 3
            ),
            bars_outside=bars_outside,
            confidence=round(confidence, 2),
        )
    
    def _calculate_atr(
        self, df: pd.DataFrame, period: int = 14
    ) -> pd.Series:
        """Calculate ATR."""
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)
        
        tr = pd.concat([
            high - low,
            abs(high - close),
            abs(low - close),
        ], axis=1).max(axis=1)
        
        return tr.rolling(period).mean().fillna(tr.mean())
    
    def _calculate_confidence(
        self,
        breakout_distance: float,
        continuation_ratio: float,
        bars_outside: int,
        atr: float,
    ) -> float:
        """Calculate confidence for fake breakout signal."""
        # Breakout size relative to ATR (0-0.3)
        # Smaller breakout relative to ATR = more likely fake
        bo_ratio = breakout_distance / max(atr, 0.01)
        bo_score = max(0, 0.3 - bo_ratio * 0.1)
        
        # Weak continuation (0-0.4)
        cont_score = min(0.4, (1.0 - abs(continuation_ratio)) * 0.4)
        
        # Few bars outside (0-0.3)
        bars_score = max(0, 0.3 - bars_outside * 0.06)
        
        return min(1.0, bo_score + cont_score + bars_score)
    
    def _deduplicate_ranges(self, ranges: list[dict]) -> list[dict]:
        """Remove overlapping range definitions."""
        if not ranges:
            return []
        
        # Keep ranges that don't overlap significantly
        unique = [ranges[0]]
        
        for r in ranges[1:]:
            last = unique[-1]
            
            # Check if ranges overlap in time
            if r["start_idx"] > last["end_idx"] - 5:
                unique.append(r)
            else:
                # Keep the one with tighter range
                if r["size"] < last["size"]:
                    unique[-1] = r
        
        return unique