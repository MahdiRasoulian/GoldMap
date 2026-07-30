"""Unit tests for intelligence engines."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from engines.volume_engine import VolumeEngine
from engines.liquidity_engine import LiquidityEngine
from engines.absorption_engine import AbsorptionEngine
from engines.stop_hunt_engine import StopHuntEngine
from engines.fake_breakout_engine import FakeBreakoutEngine


def generate_test_df(n_bars: int = 200, base_price: float = 2650.0) -> pd.DataFrame:
    """Generate test OHLC DataFrame."""
    np.random.seed(42)
    
    dates = [datetime.now() - timedelta(minutes=n_bars - i) for i in range(n_bars)]
    
    price = base_price
    data = []
    
    for i in range(n_bars):
        volatility = 1.5
        open_price = price
        moves = np.random.randn(4) * volatility
        
        high_price = open_price + abs(moves[0])
        low_price = open_price - abs(moves[1])
        close_price = open_price + moves[2] * 0.5
        
        high_price = max(open_price, close_price, high_price)
        low_price = min(open_price, close_price, low_price)
        
        # Volume with occasional spikes
        if np.random.random() < 0.05:
            vol = int(np.random.uniform(400, 1000))
        else:
            vol = int(np.random.exponential(150) + 50)
        
        data.append({
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "tick_volume": vol,
            "spread": 3,
            "real_volume": 0,
        })
        
        price = close_price
    
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
    return df


class TestVolumeEngine:
    """Tests for VolumeEngine."""
    
    def setup_method(self):
        self.engine = VolumeEngine()
        self.df = generate_test_df()
    
    def test_analyze_returns_signals(self):
        signals = self.engine.analyze(self.df)
        assert isinstance(signals, list)
    
    def test_analyze_empty_df(self):
        signals = self.engine.analyze(pd.DataFrame())
        assert signals == []
    
    def test_volume_profile(self):
        profile = self.engine.get_volume_profile(self.df, bins=20)
        assert not profile.empty
        assert "price_level" in profile.columns
        assert "volume" in profile.columns
        assert "normalized_volume" in profile.columns
        assert len(profile) == 20
    
    def test_relative_volume_calculation(self):
        signals = self.engine.analyze(self.df)
        for signal in signals:
            assert signal.relative_volume > 0
            assert signal.data_category.value == "derived"
    
    def test_spike_detection(self):
        # Create data with obvious spike
        df = self.df.copy()
        df.iloc[-1, df.columns.get_loc("tick_volume")] = 5000  # Huge spike
        
        signals = self.engine.analyze(df)
        spikes = [s for s in signals if s.is_spike]
        assert len(spikes) > 0


class TestLiquidityEngine:
    """Tests for LiquidityEngine."""
    
    def setup_method(self):
        self.engine = LiquidityEngine()
        self.df = generate_test_df(n_bars=500)
    
    def test_analyze_returns_zones(self):
        zones = self.engine.analyze(self.df)
        assert isinstance(zones, list)
    
    def test_zones_have_correct_category(self):
        zones = self.engine.analyze(self.df)
        for zone in zones:
            assert zone.data_category.value == "estimated"
    
    def test_zones_are_sorted_by_strength(self):
        zones = self.engine.analyze(self.df)
        if len(zones) > 1:
            strengths = [z.strength for z in zones]
            assert strengths == sorted(strengths, reverse=True)
    
    def test_zone_properties(self):
        zones = self.engine.analyze(self.df)
        for zone in zones:
            assert zone.price_high >= zone.price_low
            assert 0 <= zone.strength <= 1
            assert zone.midpoint == (zone.price_low + zone.price_high) / 2


class TestAbsorptionEngine:
    """Tests for AbsorptionEngine."""
    
    def setup_method(self):
        self.engine = AbsorptionEngine()
        self.df = generate_test_df()
    
    def test_analyze_returns_signals(self):
        signals = self.engine.analyze(self.df)
        assert isinstance(signals, list)
    
    def test_signals_have_correct_category(self):
        signals = self.engine.analyze(self.df)
        for signal in signals:
            assert signal.data_category.value == "estimated"
            assert 0 <= signal.confidence <= 1
    
    def test_empty_df(self):
        signals = self.engine.analyze(pd.DataFrame())
        assert signals == []


class TestStopHuntEngine:
    """Tests for StopHuntEngine."""
    
    def setup_method(self):
        self.engine = StopHuntEngine()
        self.df = generate_test_df(n_bars=300)
    
    def test_analyze_returns_signals(self):
        signals = self.engine.analyze(self.df)
        assert isinstance(signals, list)
    
    def test_signals_have_correct_fields(self):
        signals = self.engine.analyze(self.df)
        for signal in signals:
            assert signal.data_category.value == "estimated"
            assert signal.hunt_direction in ["above", "below"]
            assert 0 <= signal.confidence <= 1


class TestFakeBreakoutEngine:
    """Tests for FakeBreakoutEngine."""
    
    def setup_method(self):
        self.engine = FakeBreakoutEngine()
        self.df = generate_test_df(n_bars=300)
    
    def test_analyze_returns_signals(self):
        signals = self.engine.analyze(self.df)
        assert isinstance(signals, list)
    
    def test_signals_have_correct_fields(self):
        signals = self.engine.analyze(self.df)
        for signal in signals:
            assert signal.data_category.value == "estimated"
            assert signal.breakout_direction in ["up", "down"]
            assert 0 <= signal.confidence <= 1
            assert signal.bars_outside >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])