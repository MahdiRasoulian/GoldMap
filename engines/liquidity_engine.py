"""Liquidity Engine — Identifies potential liquidity zones.

Data Classification:
- Input: OBSERVED (OHLC, session times)
- Output: ESTIMATED (liquidity zones, strength, ranking)

Limitations:
- No access to real order book or pending orders
- Liquidity zones are INFERRED from price structure
- Cannot see actual stop-loss clusters
- Zone strength is a heuristic, not measured liquidity
"""

from datetime import datetime, time as dt_time
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.loader import CONFIG
from models.signals import LiquidityZone, DataCategory


class LiquidityEngine:
    """Estimates liquidity zones from price structure.
    
    Methodology:
    1. Identify swing highs/lows (potential stop-loss clusters)
    2. Identify session highs/lows (institutional reference points)
    3. Score zones by number of touches and recency
    4. Rank zones by estimated liquidity strength
    
    All outputs are ESTIMATED — we infer where liquidity MIGHT be.
    """
    
    def __init__(self):
        cfg = CONFIG["engines"]["liquidity"]
        self.swing_lookback = cfg["swing_lookback"]
        self.min_touches = cfg["min_touches"]
        self.zone_tolerance = cfg["zone_tolerance_pips"] * 0.01  # Convert to price
        self.session_times = cfg["session_times"]
    
    def analyze(self, df: pd.DataFrame) -> list[LiquidityZone]:
        """Identify liquidity zones from price data.
        
        Args:
            df: OHLC DataFrame with datetime index.
            
        Returns:
            List of estimated liquidity zones, ranked by strength.
        """
        if df.empty or len(df) < self.swing_lookback * 2:
            return []
        
        zones = []
        
        # 1. Swing-based liquidity zones
        swing_zones = self._find_swing_zones(df)
        zones.extend(swing_zones)
        
        # 2. Session-based liquidity zones
        session_zones = self._find_session_zones(df)
        zones.extend(session_zones)
        
        # 3. Round number zones
        round_zones = self._find_round_number_zones(df)
        zones.extend(round_zones)
        
        # 4. Merge overlapping zones
        zones = self._merge_zones(zones)
        
        # 5. Rank by strength
        zones.sort(key=lambda z: z.strength, reverse=True)
        
        return zones
    
    def _find_swing_zones(self, df: pd.DataFrame) -> list[LiquidityZone]:
        """Find zones around swing highs and lows.
        
        Rationale: Swing points often have stop-loss clusters
        just beyond them. These are ESTIMATED liquidity pools.
        """
        zones = []
        lookback = self.swing_lookback
        
        highs = df["high"].values
        lows = df["low"].values
        times = df.index
        
        for i in range(lookback, len(df) - lookback):
            # Swing high detection
            if highs[i] == max(highs[i - lookback:i + lookback + 1]):
                zone = LiquidityZone(
                    price_low=highs[i],
                    price_high=highs[i] + self.zone_tolerance,
                    strength=0.0,  # Will be calculated
                    zone_type="resistance",
                    touches=1,
                    last_touch=times[i],
                )
                zones.append(zone)
            
            # Swing low detection
            if lows[i] == min(lows[i - lookback:i + lookback + 1]):
                zone = LiquidityZone(
                    price_low=lows[i] - self.zone_tolerance,
                    price_high=lows[i],
                    strength=0.0,
                    zone_type="support",
                    touches=1,
                    last_touch=times[i],
                )
                zones.append(zone)
        
        # Count touches for each zone
        for zone in zones:
            touches = 0
            last_touch = zone.last_touch
            
            for _, row in df.iterrows():
                if (row["high"] >= zone.price_low and 
                    row["low"] <= zone.price_high):
                    touches += 1
                    last_touch = row.name
            
            zone.touches = touches
            zone.last_touch = last_touch
            
            # Strength based on touches and recency
            recency_factor = 1.0  # Could decay with time
            zone.strength = min(1.0, (touches / 10) * recency_factor)
        
        return zones
    
    def _find_session_zones(self, df: pd.DataFrame) -> list[LiquidityZone]:
        """Find zones at session highs/lows.
        
        Institutional traders often target previous session extremes.
        """
        zones = []
        
        for session_name, (start_str, end_str) in self.session_times.items():
            start_time = dt_time(*map(int, start_str.split(":")))
            end_time = dt_time(*map(int, end_str.split(":")))
            
            # Filter candles within session
            session_mask = df.index.map(
                lambda t: start_time <= t.time() < end_time
            )
            session_df = df[session_mask]
            
            if session_df.empty:
                continue
            
            # Get session high and low
            session_high = session_df["high"].max()
            session_low = session_df["low"].min()
            
            # Session high zone
            zones.append(LiquidityZone(
                price_low=session_high,
                price_high=session_high + self.zone_tolerance,
                strength=0.7,  # Session levels are significant
                zone_type=f"session_{session_name}_high",
                touches=1,
                last_touch=session_df.index[-1],
            ))
            
            # Session low zone
            zones.append(LiquidityZone(
                price_low=session_low - self.zone_tolerance,
                price_high=session_low,
                strength=0.7,
                zone_type=f"session_{session_name}_low",
                touches=1,
                last_touch=session_df.index[-1],
            ))
        
        return zones
    
    def _find_round_number_zones(
        self, df: pd.DataFrame
    ) -> list[LiquidityZone]:
        """Find zones at psychological round numbers.
        
        For XAUUSD: $2600, $2650, $2700, etc.
        """
        zones = []
        
        price_min = df["low"].min()
        price_max = df["high"].max()
        
        # Major levels (every $50)
        major_step = 50.0
        level = np.floor(price_min / major_step) * major_step
        
        while level <= price_max + major_step:
            zones.append(LiquidityZone(
                price_low=level - self.zone_tolerance,
                price_high=level + self.zone_tolerance,
                strength=0.5,
                zone_type="round_number",
                touches=0,
            ))
            level += major_step
        
        # Minor levels (every $10)
        minor_step = 10.0
        level = np.floor(price_min / minor_step) * minor_step
        
        while level <= price_max + minor_step:
            # Skip if already covered by major
            if level % major_step != 0:
                zones.append(LiquidityZone(
                    price_low=level - self.zone_tolerance * 0.5,
                    price_high=level + self.zone_tolerance * 0.5,
                    strength=0.3,
                    zone_type="round_number_minor",
                    touches=0,
                ))
            level += minor_step
        
        return zones
    
    def _merge_zones(
        self, zones: list[LiquidityZone]
    ) -> list[LiquidityZone]:
        """Merge overlapping zones into single stronger zones."""
        if not zones:
            return []
        
        # Sort by price_low
        zones.sort(key=lambda z: z.price_low)
        
        merged = [zones[0]]
        
        for zone in zones[1:]:
            last = merged[-1]
            
            # Check overlap
            if zone.price_low <= last.price_high + self.zone_tolerance:
                # Merge: extend range, combine strength
                merged[-1] = LiquidityZone(
                    price_low=min(last.price_low, zone.price_low),
                    price_high=max(last.price_high, zone.price_high),
                    strength=min(1.0, last.strength + zone.strength * 0.5),
                    zone_type=last.zone_type,
                    touches=last.touches + zone.touches,
                    last_touch=max(
                        last.last_touch or datetime.min,
                        zone.last_touch or datetime.min,
                    ) or None,
                )
            else:
                merged.append(zone)
        
        return merged