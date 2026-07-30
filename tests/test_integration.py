"""Integration tests — end-to-end pipeline testing."""

from datetime import datetime

import pandas as pd
import pytest

from core.mt5_connector import MT5Connector
from engines.signal_engine import SignalEngine


class TestFullPipeline:
    """Test the complete analysis pipeline."""
    
    def setup_method(self):
        self.connector = MT5Connector()
        self.connector.connect()
        self.signal_engine = SignalEngine()
    
    def teardown_method(self):
        self.connector.disconnect()
    
    def test_full_analysis_pipeline(self):
        """Test: Connector -> DataFrame -> SignalEngine -> Results."""
        # Get data
        df = self.connector.get_candles_df(count=500)
        assert not df.empty
        
        # Run analysis
        result = self.signal_engine.process(df)
        
        # Verify structure
        assert "timestamp" in result
        assert "volume_signals" in result
        assert "liquidity_zones" in result
        assert "absorption_signals" in result
        assert "stop_hunt_signals" in result
        assert "fake_breakout_signals" in result
        assert "volume_profile" in result
        assert "unified_signals" in result
        assert "active_alerts" in result
    
    def test_volume_profile_output(self):
        """Test volume profile generation."""
        df = self.connector.get_candles_df(count=200)
        result = self.signal_engine.process(df)
        
        vp = result["volume_profile"]
        if not vp.empty:
            assert "price_level" in vp.columns
            assert "volume" in vp.columns
            assert all(vp["normalized_volume"] >= 0)
            assert all(vp["normalized_volume"] <= 1)
    
    def test_signals_have_data_categories(self):
        """Verify all signals are properly categorized."""
        df = self.connector.get_candles_df(count=500)
        result = self.signal_engine.process(df)
        
        # Unified signals should all have data_category
        for signal in result["unified_signals"]:
            assert signal.data_category in [
                "observed", "derived", "estimated"
            ]
    
    def test_alerts_format(self):
        """Test alert output format."""
        df = self.connector.get_candles_df(count=500)
        result = self.signal_engine.process(df)
        
        for alert in result["active_alerts"]:
            assert "type" in alert
            assert "price" in alert
            assert "confidence" in alert
            assert "category" in alert
            assert "message" in alert
            assert 0 <= alert["confidence"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])