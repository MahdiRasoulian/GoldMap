"""Data collector — continuous tick and candle collection from MT5."""

import asyncio
import time
from datetime import datetime
from typing import Callable, Optional

from loguru import logger

from config.loader import CONFIG
from core.mt5_connector import MT5Connector
from models.market_data import Tick, TickBuffer, MarketSnapshot
from storage.database import Database


class DataCollector:
    """Collects and buffers market data from MT5.
    
    Responsibilities:
    - Continuous tick collection
    - Tick buffering for engine consumption
    - Periodic candle updates
    - Market snapshot generation
    """
    
    def __init__(self, connector: MT5Connector, db: Database):
        self.connector = connector
        self.db = db
        self.tick_buffer = TickBuffer(
            max_size=CONFIG["collector"]["tick_buffer_size"]
        )
        self.running = False
        self._callbacks: list[Callable] = []
        self._last_candle_update = 0.0
    
    def register_callback(self, callback: Callable) -> None:
        """Register a callback for new tick data."""
        self._callbacks.append(callback)
    
    async def start(self) -> None:
        """Start continuous data collection."""
        if not self.connector.connected:
            if not self.connector.connect():
                logger.error("Cannot start collector — MT5 not connected")
                return
        
        self.running = True
        logger.info("Data collector started")
        
        interval = CONFIG["collector"]["collection_interval_ms"] / 1000.0
        
        while self.running:
            try:
                await self._collect_cycle()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Collection error: {e}")
                await asyncio.sleep(1.0)
    
    def stop(self) -> None:
        """Stop data collection."""
        self.running = False
        logger.info("Data collector stopped")
    
    async def _collect_cycle(self) -> None:
        """Single collection cycle."""
        # Collect ticks
        ticks = self.connector.get_ticks(count=100)
        
        for tick in ticks:
            self.tick_buffer.add(tick)
        
        # Store ticks in database
        if ticks:
            await self.db.store_ticks(ticks)
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(ticks)
                else:
                    callback(ticks)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        # Periodic candle update (every 5 seconds)
        now = time.time()
        if now - self._last_candle_update > 5.0:
            await self._update_candles()
            self._last_candle_update = now
    
    async def _update_candles(self) -> None:
        """Update candle data in database."""
        candles = self.connector.get_candles(
            timeframe=CONFIG["mt5"]["timeframe"],
            count=CONFIG["collector"]["candle_history_bars"],
        )
        if candles:
            await self.db.store_candles(candles)
    
    def get_snapshot(self) -> Optional[MarketSnapshot]:
        """Get current market snapshot."""
        price_data = self.connector.get_current_price()
        if price_data is None:
            return None
        
        # Determine session
        hour = datetime.now().hour
        if 0 <= hour < 8:
            session = "asian"
        elif 8 <= hour < 16:
            session = "london"
        elif 13 <= hour < 21:
            session = "newyork"
        else:
            session = "off-hours"
        
        return MarketSnapshot(
            timestamp=price_data["time"],
            symbol=self.connector.symbol,
            bid=price_data["bid"],
            ask=price_data["ask"],
            spread=price_data["spread"],
            last_price=(price_data["bid"] + price_data["ask"]) / 2,
            session=session,
        )


# Entry point for standalone collector
async def main():
    """Run collector as standalone process."""
    from storage.database import Database
    
    db = Database()
    await db.initialize()
    
    connector = MT5Connector()
    collector = DataCollector(connector, db)
    
    try:
        await collector.start()
    except KeyboardInterrupt:
        collector.stop()
        connector.disconnect()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())