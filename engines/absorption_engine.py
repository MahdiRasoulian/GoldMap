"""Absorption Engine — Detects institutional absorption patterns.

Data Classification:
- Input: OBSERVED (tick_volume, OHLC)
- Output: ESTIMATED (absorption signals)

Concept:
Absorption occurs when large orders absorb selling/buying pressure
without allowing price to move significantly. This manifests as:
- High volume
- Low price movement (small candle bodies)
- Repeated defense of a price level

Limitations:
- Cannot confirm actual institutional orders
- High volume + low movement could also be consolidation
- No way to distinguish absorption from balanced flow
- Confidence scores are heuristic-based
"""

import numpy as np
import pandas as pd
from loguru import logger

from config.loader import CONFIG
from models.signals import AbsorptionSignal, DataCategory


class AbsorptionEngine:
    """Detects potential absorption patterns in price action.
    
    All outputs are ESTIMATED — we infer absorption from
    observable price/volume characteristics.
    """
    
    def __init__(self):
        cfg = CONFIG["engines"]["absorption"]
        self.volume_threshold = cfg["volume_threshold"]
        self.movement_threshold = cfg["movement_threshold"]
        self.min_candles = cfg["min_candles"]
        self.defense_tolerance = cfg["defense_tolerance_pips"] * 0.01
    
    def analyze(self, df: pd.DataFrame) -> list[AbsorptionSignal]:
        """Detect absorption patterns in recent price action.
        
        Args:
            df: OHLC DataFrame with tick_volume column.
            
        Returns:
            List of AbsorptionSignal objects.
        """
        if df.empty or len(df) < 50:
            return []
        
        signals = []
        
        df = df.copy()
        
        # Calculate metrics
        df["body_size"] = abs(df["close"] - df["open"])
        df["range_size"] = df["high"] - df["low"]
        df["avg_volume"] = df["tick_volume"].rolling(50).mean()
        df["avg_range"] = df["range_size"].rolling(50).mean()
        df["relative_volume"] = df["tick_volume"] / df["avg_volume"]
        df["relative_range"] = df["range_size"] / df["avg_range"]
        
        # Scan for absorption patterns
        window = self.min_candles
        
        for i in range(window, len(df)):
            segment = df.iloc[i - window:i + 1]
            
            # Check conditions
            avg_rel_vol = segment["relative_volume"].mean()
            avg_rel_range = segment["relative_range"].mean()
            
            # High volume condition
            if avg_rel_vol < self.volume_threshold:
                continue
            
            # Low movement condition (relative to volume)
            volume_to_movement = avg_rel_vol / max(avg_rel_range, 0.01)
            if volume_to_movement < 2.0:
                continue
            
            # Check for level defense
            price_level = segment["close"].mean()
            defense_count = self._count_defenses(
                segment, price_level
            )
            
            if defense_count < self.min_candles:
                continue
            
            # Determine direction
            direction = self._determine_direction(segment)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                avg_rel_vol, volume_to_movement, defense_count
            )
            
            # Price movement in pips
            total_movement = abs(
                segment["close"].iloc[-1] - segment["close"].iloc[0]
            ) * 100  # Convert to pips
            
            signals.append(AbsorptionSignal(
                timestamp=df.index[i],
                price_level=round(price_level, 2),
                volume_ratio=round(avg_rel_vol, 2),
                price_movement=round(total_movement, 1),
                defense_count=defense_count,
                confidence=round(confidence, 2),
                direction=direction,
            ))
        
        # Remove duplicate/overlapping signals
        signals = self._deduplicate(signals)
        
        return signals
    
    def _count_defenses(
        self, segment: pd.DataFrame, level: float
    ) -> int:
        """Count how many times a price level was defended."""
        count = 0
        
        for _, row in segment.iterrows():
            # Price touched the level but didn't break through
            if (abs(row["low"] - level) < self.defense_tolerance or
                abs(row["high"] - level) < self.defense_tolerance):
                count += 1
        
        return count
    
    def _determine_direction(self, segment: pd.DataFrame) -> str:
        """Determine if absorption is bullish or bearish.
        
        Bullish absorption: Price being held up despite selling pressure
        Bearish absorption: Price being held down despite buying pressure
        """
        # Look at where wicks are relative to bodies
        upper_wicks = segment["high"] - segment[["open", "close"]].max(axis=1)
        lower_wicks = segment[["open", "close"]].min(axis=1) - segment["low"]
        
        avg_upper = upper_wicks.mean()
        avg_lower = lower_wicks.mean()
        
        if avg_lower > avg_upper * 1.5:
            return "bullish"  # Selling being absorbed (long lower wicks)
        elif avg_upper > avg_lower * 1.5:
            return "bearish"  # Buying being absorbed (long upper wicks)
        else:
            return "neutral"
    
    def _calculate_confidence(
        self,
        volume_ratio: float,
        vol_to_movement: float,
        defense_count: int,
    ) -> float:
        """Calculate confidence score for absorption signal."""
        # Volume component (0-0.4)
        vol_score = min(0.4, (volume_ratio - self.volume_threshold) * 0.2)
        
        # Volume-to-movement ratio component (0-0.3)
        vtm_score = min(0.3, (vol_to_movement - 2.0) * 0.1)
        
        # Defense count component (0-0.3)
        def_score = min(0.3, defense_count * 0.1)
        
        return min(1.0, vol_score + vtm_score + def_score)
    
    def _deduplicate(
        self, signals: list[AbsorptionSignal]
    ) -> list[AbsorptionSignal]:
        """Remove signals that are too close together."""
        if len(signals) <= 1:
            return signals
        
        deduplicated = [signals[0]]
        
        for signal in signals[1:]:
            last = deduplicated[-1]
            
            # If same level and within 10 bars, keep the stronger one
            price_diff = abs(signal.price_level - last.price_level)
            if price_diff < self.defense_tolerance * 2:
                if signal.confidence > last.confidence:
                    deduplicated[-1] = signal
            else:
                deduplicated.append(signal)
        
        return deduplicated