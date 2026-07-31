"""Builds MarketContext by directly using core components (no HTTP)."""

import pandas as pd
from datetime import datetime, timezone
from loguru import logger

from config.loader import CONFIG
from core.mt5_connector import MT5Connector
from engines.signal_engine import SignalEngine

from .models import MarketContext


class ContextBuilder:
    """Fetches and aggregates data directly from core components."""

    def __init__(self):
        self.connector = MT5Connector()
        self.signal_engine = SignalEngine()
        # Ensure connection (if not already)
        if not self.connector.connected:
            self.connector.connect()

    def build(self, force_refresh: bool = False) -> MarketContext:
        """Build a fresh MarketContext from current data."""
        # 1. Get candles
        df = self._get_candles()
        if df.empty:
            logger.warning("No candle data available for assistant")
            return MarketContext()

        # 2. Get analysis (signals, zones, profile)
        analysis = self._get_analysis(df)

        # 3. Get snapshot (price)
        snapshot = self._get_snapshot()

        # 4. Build context
        ctx = MarketContext()
        ctx.timestamp = datetime.now(timezone.utc)

        # Price
        if snapshot:
            ctx.price = snapshot.get("bid", 0.0)

        # Trend (simple: compare price to 20-period SMA)
        if len(df) >= 20:
            sma = df["close"].rolling(20).mean().iloc[-1]
            last_close = df["close"].iloc[-1]
            if last_close > sma * 1.002:
                ctx.trend = "bullish"
            elif last_close < sma * 0.998:
                ctx.trend = "bearish"
            else:
                ctx.trend = "neutral"

        # Session
        ctx.session = self._detect_session()

        # Signals
        ctx.liquidity_zones = analysis.get("liquidity_zones", [])
        ctx.stop_hunt_signals = analysis.get("stop_hunt_signals", [])
        ctx.absorption_signals = analysis.get("absorption_signals", [])
        ctx.fake_breakout_signals = analysis.get("fake_breakout_signals", [])

        # Volume
        ctx.volume_profile = analysis.get("volume_profile", [])
        ctx.tick_volume = df["tick_volume"].iloc[-1] if not df.empty else 0
        ctx.avg_volume = df["tick_volume"].rolling(20).mean().iloc[-1] if len(df) >= 20 else ctx.tick_volume

        # Volatility (ATR)
        if len(df) >= 14:
            high, low, close = df["high"], df["low"], df["close"]
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            ctx.atr = tr.rolling(14).mean().iloc[-1]
        else:
            ctx.atr = 0.0

        return ctx

    # ---- Internal helpers using core components ----

    def _get_candles(self) -> pd.DataFrame:
        """Get candles directly from MT5 connector."""
        df = self.connector.get_candles_df(
            timeframe=CONFIG["mt5"]["timeframe"],
            count=200
        )
        # Ensure UTC-naive for consistency
        if not df.empty and df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        return df

    def _get_analysis(self, df: pd.DataFrame) -> dict:
        """Run signal engine directly."""
        if df.empty:
            return {}
        result = self.signal_engine.process(df)
        # Convert to dict for easy consumption
        return {
            "liquidity_zones": [z.model_dump() for z in result.get("liquidity_zones", [])],
            "stop_hunt_signals": [s.model_dump() for s in result.get("stop_hunt_signals", [])],
            "absorption_signals": [s.model_dump() for s in result.get("absorption_signals", [])],
            "fake_breakout_signals": [s.model_dump() for s in result.get("fake_breakout_signals", [])],
            "volume_profile": result.get("volume_profile", []),
        }

    def _get_snapshot(self) -> dict:
        """Get snapshot from connector."""
        price_data = self.connector.get_current_price()
        if price_data:
            return price_data
        return {}

    def _detect_session(self) -> str:
        """Detect trading session based on UTC hour."""
        hour = datetime.now(timezone.utc).hour
        if 0 <= hour < 8:
            return "asian"
        elif 8 <= hour < 16:
            return "london"
        elif 16 <= hour < 21:
            return "newyork"
        else:
            return "off_hours"