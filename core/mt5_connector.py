"""MetaTrader 5 connector — handles connection lifecycle and data retrieval."""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    logger.warning("MetaTrader5 package not available. Using mock mode.")

from config.loader import CONFIG
from models.market_data import Tick, Candle


class MT5Connector:
    """Manages MetaTrader 5 connection and raw data retrieval.
    
    All data returned is OBSERVED — directly from MT5.
    """
    
    def __init__(self):
        self.symbol = CONFIG["mt5"]["symbol"]
        self.connected = False
        self._mock_mode = mt5 is None
    
    def connect(self) -> bool:
        """Initialize MT5 connection.
        
        Returns:
            True if connected successfully.
        """
        if self._mock_mode:
            logger.info("Running in MOCK mode — no MT5 connection")
            self.connected = True
            return True
        
        if not mt5.initialize(
            login=CONFIG["mt5"]["login"],
            password=CONFIG["mt5"]["password"],
            server=CONFIG["mt5"]["server"],
            timeout=CONFIG["mt5"]["timeout"],
        ):
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            return False
        
        # Verify symbol is available
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            logger.error(f"Symbol {self.symbol} not found")
            mt5.shutdown()
            return False
        
        if not symbol_info.visible:
            mt5.symbol_select(self.symbol, True)
        
        self.connected = True
        logger.info(f"Connected to MT5 — Symbol: {self.symbol}")
        return True
    
    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if not self._mock_mode and self.connected:
            mt5.shutdown()
        self.connected = False
        logger.info("MT5 disconnected")
    
    def get_ticks(
        self,
        count: int = 1000,
        from_date: Optional[datetime] = None,
    ) -> list[Tick]:
        """Retrieve recent ticks — OBSERVED data.
        
        Args:
            count: Number of ticks to retrieve.
            from_date: Start date for tick retrieval.
            
        Returns:
            List of Tick objects.
        """
        if self._mock_mode:
            return self._generate_mock_ticks(count)
        
        if from_date is None:
            from_date = datetime.now() - timedelta(minutes=5)
        
        ticks_raw = mt5.copy_ticks_from(
            self.symbol, from_date, count, mt5.COPY_TICKS_ALL
        )
        
        if ticks_raw is None or len(ticks_raw) == 0:
            logger.warning("No ticks received from MT5")
            return []
        
        ticks = []
        for t in ticks_raw:
            ticks.append(Tick(
                time=datetime.fromtimestamp(t['time']),
                bid=float(t['bid']),
                ask=float(t['ask']),
                last=float(t.get('last', 0)),
                volume=int(t.get('volume', 0)),
                flags=int(t.get('flags', 0)),
            ))
        
        return ticks
    
    def get_candles(
        self,
        timeframe: str = "M1",
        count: int = 500,
    ) -> list[Candle]:
        """Retrieve OHLC candles — OBSERVED data.
        
        Args:
            timeframe: MT5 timeframe string (M1, M5, M15, H1, H4, D1).
            count: Number of candles to retrieve.
            
        Returns:
            List of Candle objects.
        """
        if self._mock_mode:
            return self._generate_mock_candles(count)
        
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
        
        rates = mt5.copy_rates_from_pos(self.symbol, mt5_tf, 0, count)
        
        if rates is None or len(rates) == 0:
            logger.warning("No candles received from MT5")
            return []
        
        candles = []
        for r in rates:
            candles.append(Candle(
                time=datetime.fromtimestamp(r['time']),
                open=float(r['open']),
                high=float(r['high']),
                low=float(r['low']),
                close=float(r['close']),
                tick_volume=int(r['tick_volume']),
                spread=int(r['spread']),
                real_volume=int(r.get('real_volume', 0)),
            ))
        
        return candles
    
    def get_candles_df(
        self,
        timeframe: str = "M1",
        count: int = 500,
    ) -> pd.DataFrame:
        """Get candles as DataFrame for engine processing.
        
        Returns:
            DataFrame with columns: time, open, high, low, close, 
            tick_volume, spread, real_volume.
        """
        candles = self.get_candles(timeframe, count)
        if not candles:
            return pd.DataFrame()
        
        data = [c.model_dump() for c in candles]
        df = pd.DataFrame(data)
        df.set_index("time", inplace=True)
        return df
    
    def get_current_price(self) -> Optional[dict]:
        """Get current bid/ask — OBSERVED data."""
        if self._mock_mode:
            price = 2650.0 + np.random.randn() * 2
            return {
                "bid": round(price, 2),
                "ask": round(price + 0.30, 2),
                "spread": 0.30,
                "time": datetime.now(),
            }
        
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": round(tick.ask - tick.bid, 2),
            "time": datetime.fromtimestamp(tick.time),
        }
    
    # --- Mock data generators for development/testing ---
    
    def _generate_mock_ticks(self, count: int) -> list[Tick]:
        """Generate realistic mock tick data for testing."""
        base_price = 2650.0
        ticks = []
        current_time = datetime.now() - timedelta(seconds=count * 0.1)
        
        price = base_price
        for i in range(count):
            # Random walk with mean reversion
            price += np.random.randn() * 0.15
            price = price * 0.999 + base_price * 0.001  # Mean reversion
            
            spread = 0.20 + abs(np.random.randn() * 0.05)
            
            ticks.append(Tick(
                time=current_time + timedelta(milliseconds=i * 100),
                bid=round(price, 2),
                ask=round(price + spread, 2),
                volume=max(1, int(np.random.exponential(3))),
            ))
        
        return ticks
    
    def _generate_mock_candles(self, count: int) -> list[Candle]:
        """Generate realistic mock OHLC data for testing."""
        base_price = 2650.0
        candles = []
        current_time = datetime.now() - timedelta(minutes=count)
        
        price = base_price
        for i in range(count):
            # Simulate realistic candle
            volatility = 1.5 + abs(np.random.randn() * 0.5)
            open_price = price
            
            # Intrabar movement
            moves = np.random.randn(4) * volatility
            high_price = open_price + abs(moves[0])
            low_price = open_price - abs(moves[1])
            close_price = open_price + moves[2] * 0.5
            
            # Ensure OHLC consistency
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)
            
            # Volume with occasional spikes
            base_vol = 150
            if np.random.random() < 0.05:  # 5% chance of spike
                vol = int(base_vol * np.random.uniform(3, 8))
            else:
                vol = int(np.random.exponential(base_vol) + 50)
            
            candles.append(Candle(
                time=current_time + timedelta(minutes=i),
                open=round(open_price, 2),
                high=round(high_price, 2),
                low=round(low_price, 2),
                close=round(close_price, 2),
                tick_volume=vol,
                spread=3,
            ))
            
            price = close_price
        
        return candles