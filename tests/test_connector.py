"""Tests for MT5 connector (mock mode)."""

from datetime import datetime

import pytest

from core.mt5_connector import MT5Connector
from models.market_data import Tick, Candle


class TestMT5Connector:
    """Test MT5Connector in mock mode."""
    
    def setup_method(self):
        self.connector = MT5Connector()
        self.connector.connect()
    
    def teardown_method(self):
        self.connector.disconnect()
    
    def test_connect(self):
        assert self.connector.connected is True
    
    def test_get_ticks(self):
        ticks = self.connector.get_ticks(count=100)
        assert len(ticks) == 100
        assert all(isinstance(t, Tick) for t in ticks)
    
    def test_tick_properties(self):
        ticks = self.connector.get_ticks(count=10)
        for tick in ticks:
            assert tick.bid > 0
            assert tick.ask > 0
            assert tick.ask > tick.bid
            assert tick.spread > 0
    
    def test_get_candles(self):
        candles = self.connector.get_candles(count=50)
        assert len(candles) == 50
        assert all(isinstance(c, Candle) for c in candles)
    
    def test_candle_properties(self):
        candles = self.connector.get_candles(count=10)
        for candle in candles:
            assert candle.high >= candle.open
            assert candle.high >= candle.close
            assert candle.low <= candle.open
            assert candle.low <= candle.close
            assert candle.tick_volume > 0
    
    def test_get_candles_df(self):
        df = self.connector.get_candles_df(count=100)
        assert not df.empty
        assert len(df) == 100
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "tick_volume" in df.columns
    
    def test_get_current_price(self):
        price = self.connector.get_current_price()
        assert price is not None
        assert "bid" in price
        assert "ask" in price
        assert "spread" in price
        assert price["ask"] > price["bid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])