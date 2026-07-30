"""Helper utility functions for Goldmap platform."""

from datetime import datetime, time as dt_time
from typing import Optional

import numpy as np
import pandas as pd


def pips_to_price(pips: float, symbol: str = "XAUUSD") -> float:
    """Convert pips to price movement.
    
    For XAUUSD: 1 pip = $0.01
    """
    if symbol == "XAUUSD":
        return pips * 0.01
    return pips * 0.0001  # Default forex


def price_to_pips(price_diff: float, symbol: str = "XAUUSD") -> float:
    """Convert price difference to pips."""
    if symbol == "XAUUSD":
        return price_diff * 100
    return price_diff * 10000


def get_current_session() -> str:
    """Determine current trading session."""
    hour = datetime.utcnow().hour
    
    if 0 <= hour < 8:
        return "asian"
    elif 8 <= hour < 13:
        return "london"
    elif 13 <= hour < 17:
        return "newyork_london_overlap"
    elif 17 <= hour < 21:
        return "newyork"
    else:
        return "off_hours"


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    
    tr = pd.concat([
        high - low,
        abs(high - prev_close),
        abs(low - prev_close),
    ], axis=1).max(axis=1)
    
    return tr.rolling(window=period).mean()


def find_swing_points(
    df: pd.DataFrame,
    lookback: int = 5,
) -> tuple[list[tuple], list[tuple]]:
    """Find swing highs and lows.
    
    Returns:
        Tuple of (swing_highs, swing_lows) where each is a list
        of (index, price) tuples.
    """
    highs = []
    lows = []
    
    for i in range(lookback, len(df) - lookback):
        # Swing high
        window_high = df["high"].iloc[i - lookback:i + lookback + 1]
        if df["high"].iloc[i] == window_high.max():
            highs.append((df.index[i], df["high"].iloc[i]))
        
        # Swing low
        window_low = df["low"].iloc[i - lookback:i + lookback + 1]
        if df["low"].iloc[i] == window_low.min():
            lows.append((df.index[i], df["low"].iloc[i]))
    
    return highs, lows


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a series to 0-1 range."""
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    
    return (series - min_val) / (max_val - min_val)


def ewma_volatility(
    df: pd.DataFrame,
    span: int = 20,
) -> pd.Series:
    """Calculate EWMA volatility from close prices."""
    returns = df["close"].pct_change()
    return returns.ewm(span=span).std() * np.sqrt(252)


def detect_divergence(
    price: pd.Series,
    indicator: pd.Series,
    lookback: int = 10,
) -> list[dict]:
    """Detect price/indicator divergences.
    
    Returns list of divergence events with type and location.
    """
    divergences = []
    
    for i in range(lookback * 2, len(price)):
        # Check for bullish divergence (price lower low, indicator higher low)
        price_window = price.iloc[i - lookback:i + 1]
        ind_window = indicator.iloc[i - lookback:i + 1]
        
        if (price_window.iloc[-1] < price_window.iloc[0] and
            ind_window.iloc[-1] > ind_window.iloc[0]):
            divergences.append({
                "type": "bullish",
                "index": price.index[i],
                "price": price.iloc[i],
            })
        
        # Bearish divergence (price higher high, indicator lower high)
        if (price_window.iloc[-1] > price_window.iloc[0] and
            ind_window.iloc[-1] < ind_window.iloc[0]):
            divergences.append({
                "type": "bearish",
                "index": price.index[i],
                "price": price.iloc[i],
            })
    
    return divergences