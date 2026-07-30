"""Volume Engine — Analyzes tick volume patterns.

Data Classification:
- Input: OBSERVED (tick_volume, OHLC from MT5)
- Output: DERIVED (relative volume, ATR) and ESTIMATED (volume imbalance)

Limitations:
- MT5 tick volume ≠ real traded volume
- Volume imbalance is ESTIMATED from price action, not real order flow
- No access to actual buy/sell volume split
"""

import numpy as np
import pandas as pd
from loguru import logger

from config.loader import CONFIG
from models.signals import VolumeSignal, DataCategory


class VolumeEngine:
    """Processes tick volume data to detect significant volume events.
    
    Methods:
    - Relative volume calculation (DERIVED)
    - Volume spike detection (DERIVED)
    - Volume imbalance estimation (ESTIMATED — inferred from price action)
    """
    
    def __init__(self):
        cfg = CONFIG["engines"]["volume"]
        self.spike_threshold = cfg["spike_threshold"]
        self.atr_period = cfg["atr_period"]
        self.lookback_bars = cfg["lookback_bars"]
    
    def analyze(self, df: pd.DataFrame) -> list[VolumeSignal]:
        """Run full volume analysis on candle DataFrame.
        
        Args:
            df: DataFrame with columns [open, high, low, close, tick_volume].
                Index should be datetime.
        
        Returns:
            List of VolumeSignal objects for significant events.
        """
        if df.empty or len(df) < self.lookback_bars:
            return []
        
        signals = []
        
        # Calculate derived metrics
        df = df.copy()
        df["atr"] = self._calculate_atr(df)
        df["relative_volume"] = self._calculate_relative_volume(df)
        df["imbalance"] = self._estimate_imbalance(df)
        
        # Detect spikes in recent data
        recent = df.tail(20)
        for idx, row in recent.iterrows():
            is_spike = row["relative_volume"] > self.spike_threshold
            
            if is_spike or abs(row["imbalance"]) > 0.6:
                signals.append(VolumeSignal(
                    timestamp=idx,
                    price=row["close"],
                    tick_volume=int(row["tick_volume"]),
                    relative_volume=round(row["relative_volume"], 2),
                    is_spike=is_spike,
                    imbalance=round(row["imbalance"], 3),
                ))
        
        return signals
    
    def get_volume_profile(
        self, df: pd.DataFrame, bins: int = 50
    ) -> pd.DataFrame:
        """Calculate volume profile (volume at price levels).
        
        This is DERIVED data — aggregation of observed tick volumes
        at observed price levels.
        
        Args:
            df: OHLC DataFrame with tick_volume.
            bins: Number of price bins.
            
        Returns:
            DataFrame with columns [price_level, volume, normalized_volume].
        """
        if df.empty:
            return pd.DataFrame()
        
        price_min = df["low"].min()
        price_max = df["high"].max()
        
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        volume_at_price = np.zeros(bins)
        
        for _, row in df.iterrows():
            # Distribute volume across the candle's range
            candle_low = row["low"]
            candle_high = row["high"]
            candle_vol = row["tick_volume"]
            
            # Find bins that overlap with this candle
            for i in range(bins):
                bin_low = bin_edges[i]
                bin_high = bin_edges[i + 1]
                
                # Calculate overlap
                overlap_low = max(candle_low, bin_low)
                overlap_high = min(candle_high, bin_high)
                
                if overlap_high > overlap_low:
                    candle_range = candle_high - candle_low
                    if candle_range > 0:
                        proportion = (overlap_high - overlap_low) / candle_range
                        volume_at_price[i] += candle_vol * proportion
        
        # Normalize
        max_vol = volume_at_price.max() if volume_at_price.max() > 0 else 1
        
        return pd.DataFrame({
            "price_level": bin_centers,
            "volume": volume_at_price,
            "normalized_volume": volume_at_price / max_vol,
        })
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range — DERIVED data."""
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def _calculate_relative_volume(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume relative to moving average — DERIVED data."""
        vol = df["tick_volume"].astype(float)
        avg_vol = vol.rolling(window=self.lookback_bars).mean()
        
        relative = vol / avg_vol.replace(0, 1)
        return relative.fillna(1.0)
    
    def _estimate_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Estimate buy/sell volume imbalance — ESTIMATED data.
        
        WARNING: This is an ESTIMATION based on price action heuristics.
        We do NOT have access to real buy/sell volume data.
        
        Method: Uses close position within the bar range as a proxy
        for buying/selling pressure (similar to Money Flow concept).
        
        Limitations:
        - Assumes close position correlates with order flow direction
        - Does not account for large limit orders
        - Cannot detect iceberg orders or hidden liquidity
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # Money Flow Multiplier: where close is within the range
        # +1 = closed at high (buying pressure)
        # -1 = closed at low (selling pressure)
        bar_range = high - low
        bar_range = bar_range.replace(0, np.nan)
        
        mf_multiplier = ((close - low) - (high - close)) / bar_range
        mf_multiplier = mf_multiplier.fillna(0)
        
        # Smooth with short EMA for noise reduction
        imbalance = mf_multiplier.ewm(span=5).mean()
        
        return imbalance.clip(-1, 1)