"""SQLite database for persistent storage of ticks, candles, and signals."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite
from loguru import logger

from config.loader import CONFIG
from models.market_data import Tick, Candle


class Database:
    """Async SQLite database manager for Goldmap.
    
    Stores:
    - Tick data (time-series)
    - Candle data (OHLC)
    - Signals and alerts
    """
    
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or CONFIG["database"]["path"]
        self.max_tick_rows = CONFIG["database"]["max_tick_rows"]
        self._db: Optional[aiosqlite.Connection] = None
    
    async def initialize(self) -> None:
        """Create database and tables."""
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        
        await self._create_tables()
        logger.info(f"Database initialized: {self.db_path}")
    
    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None
    
    async def _create_tables(self) -> None:
        """Create all required tables."""
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TIMESTAMP NOT NULL,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                volume INTEGER DEFAULT 0,
                spread REAL DEFAULT 0,
                flags INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TIMESTAMP NOT NULL UNIQUE,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                tick_volume INTEGER NOT NULL,
                spread INTEGER DEFAULT 0,
                real_volume INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TIMESTAMP NOT NULL,
                signal_type TEXT NOT NULL,
                price_level REAL NOT NULL,
                confidence REAL NOT NULL,
                data_category TEXT NOT NULL,
                description TEXT,
                metadata TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_ticks_time ON ticks(time);
            CREATE INDEX IF NOT EXISTS idx_candles_time ON candles(time);
            CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(time);
            CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type);
        """)
        await self._db.commit()
    
    async def store_ticks(self, ticks: list[Tick]) -> None:
        """Store tick data."""
        if not ticks or not self._db:
            return
        
        data = [
            (t.time.isoformat(), t.bid, t.ask, t.volume, t.spread, t.flags)
            for t in ticks
        ]
        
        await self._db.executemany(
            "INSERT INTO ticks (time, bid, ask, volume, spread, flags) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            data,
        )
        await self._db.commit()
        
        # Cleanup old data if needed
        await self._cleanup_ticks()
    
    async def store_candles(self, candles: list[Candle]) -> None:
        """Store candle data (upsert)."""
        if not candles or not self._db:
            return
        
        data = [
            (
                c.time.isoformat(), c.open, c.high, c.low, c.close,
                c.tick_volume, c.spread, c.real_volume,
            )
            for c in candles
        ]
        
        await self._db.executemany(
            "INSERT OR REPLACE INTO candles "
            "(time, open, high, low, close, tick_volume, spread, real_volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            data,
        )
        await self._db.commit()
    
    async def get_candles(
        self,
        limit: int = 500,
        since: Optional[datetime] = None,
    ) -> list[dict]:
        """Retrieve candles from database."""
        if not self._db:
            return []
        
        if since:
            cursor = await self._db.execute(
                "SELECT * FROM candles WHERE time >= ? "
                "ORDER BY time DESC LIMIT ?",
                (since.isoformat(), limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM candles ORDER BY time DESC LIMIT ?",
                (limit,),
            )
        
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        return [dict(zip(columns, row)) for row in rows]
    
    async def get_recent_ticks(self, limit: int = 1000) -> list[dict]:
        """Get most recent ticks."""
        if not self._db:
            return []
        
        cursor = await self._db.execute(
            "SELECT * FROM ticks ORDER BY time DESC LIMIT ?",
            (limit,),
        )
        
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        return [dict(zip(columns, row)) for row in rows]
    
    async def store_signal(self, signal: dict) -> None:
        """Store a signal/alert."""
        if not self._db:
            return
        
        import json
        
        await self._db.execute(
            "INSERT INTO signals "
            "(time, signal_type, price_level, confidence, "
            "data_category, description, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                signal.get("time", datetime.now().isoformat()),
                signal.get("signal_type", ""),
                signal.get("price_level", 0),
                signal.get("confidence", 0),
                signal.get("data_category", "estimated"),
                signal.get("description", ""),
                json.dumps(signal.get("metadata", {})),
            ),
        )
        await self._db.commit()
    
    async def get_signals(
        self,
        signal_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Retrieve stored signals."""
        if not self._db:
            return []
        
        if signal_type:
            cursor = await self._db.execute(
                "SELECT * FROM signals WHERE signal_type = ? "
                "ORDER BY time DESC LIMIT ?",
                (signal_type, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM signals ORDER BY time DESC LIMIT ?",
                (limit,),
            )
        
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        return [dict(zip(columns, row)) for row in rows]
    
    async def _cleanup_ticks(self) -> None:
        """Remove old tick data to prevent database bloat."""
        cursor = await self._db.execute("SELECT COUNT(*) FROM ticks")
        count = (await cursor.fetchone())[0]
        
        if count > self.max_tick_rows:
            excess = count - self.max_tick_rows
            await self._db.execute(
                "DELETE FROM ticks WHERE id IN "
                "(SELECT id FROM ticks ORDER BY time ASC LIMIT ?)",
                (excess,),
            )
            await self._db.commit()
            logger.debug(f"Cleaned up {excess} old tick rows")