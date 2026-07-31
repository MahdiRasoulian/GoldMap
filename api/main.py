"""FastAPI application — REST API and WebSocket for real-time data."""

import asyncio
import json
import math  # ← اضافه شده برای تشخیص NaN
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from loguru import logger

from config.loader import CONFIG
from core.mt5_connector import MT5Connector
from core.collector import DataCollector
from engines.signal_engine import SignalEngine
from storage.database import Database

# ── Assistant and Notifier ──────────────────────────────────────────────
from assistant.main import get_assistant_text, set_notifier, run_assistant
from assistant.notifier import Notifier


# ── Helper: clean NaN ────────────────────────────────────────────────────

def clean_nan(obj):
    """Recursively replace NaN with None in dict/list."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj


# ── Global instances ─────────────────────────────────────────────────────

db = Database()
connector = MT5Connector()
collector = DataCollector(connector, db)
signal_engine = SignalEngine()
ws_clients: list[WebSocket] = []


# ── Background task for assistant ──────────────────────────────────────

async def assistant_loop():
    """Run assistant every 30 seconds and send Telegram notifications."""
    while True:
        try:
            # اجرای Assistant و ارسال نوتیفیکیشن در صورت لزوم
            run_assistant(force_refresh=True, send_notification=True)
        except Exception as e:
            logger.error(f"Assistant loop error: {e}")
        await asyncio.sleep(30)  # هر ۳۰ ثانیه


# ── Lifespan ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    await db.initialize()
    connector.connect()
    
    # Start collector in background
    collector_task = asyncio.create_task(collector.start())
    
    # ── راه‌اندازی Assistant Notifier ──────────────────────────────
    notifier = Notifier(websocket_manager=None)   # در حال حاضر WebSocket نداریم
    set_notifier(notifier)
    assistant_task = asyncio.create_task(assistant_loop())
    logger.info("Assistant notifier and background loop started")
    
    logger.info("Goldmap API started")
    yield
    
    # Shutdown
    collector.stop()
    collector_task.cancel()
    assistant_task.cancel()
    connector.disconnect()
    await db.close()
    logger.info("Goldmap API stopped")


# ── FastAPI app ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Goldmap API",
    description="Gold Market Intelligence Platform — XAUUSD Analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ──────────────────────────────────────────────────────

@app.get("/")
async def root():
    """API health check."""
    return {
        "status": "running",
        "platform": "Goldmap",
        "symbol": CONFIG["mt5"]["symbol"],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/snapshot")
async def get_snapshot():
    """Get current market snapshot."""
    snapshot = collector.get_snapshot()
    if snapshot is None:
        return {"error": "No data available"}
    return snapshot.model_dump()


@app.get("/api/candles")
async def get_candles(limit: int = 500, timeframe: str = "M1"):
    """Get OHLC candle data."""
    candles = connector.get_candles(timeframe=timeframe, count=limit)
    return {
        "data_category": "observed",
        "count": len(candles),
        "candles": [c.model_dump() for c in candles],
    }


@app.get("/api/analysis")
async def get_analysis(limit: int = 300):
    """Run full analysis and return all engine outputs."""
    # محدود کردن تعداد کندل‌ها برای جلوگیری از پردازش سنگین
    limit = max(50, min(limit, 500))

    df = connector.get_candles_df(
        timeframe=CONFIG["mt5"]["timeframe"],
        count=limit,
    )

    if df.empty:
        return {"error": "No data available"}

    result = signal_engine.process(df)

    # پردازش volume profile و جایگزینی NaN با 0
    profile = result["volume_profile"]
    if not profile.empty:
        profile = profile.tail(50)          # فقط ۵۰ ردیف آخر
        profile = profile.fillna(0)         # ← جایگزینی NaN با 0

    # ساخت پاسخ
    response = {
        "timestamp": result["timestamp"].isoformat(),
        "volume_signals": [s.model_dump() for s in result["volume_signals"]],
        "liquidity_zones": [z.model_dump() for z in result["liquidity_zones"]],
        "absorption_signals": [s.model_dump() for s in result["absorption_signals"]],
        "stop_hunt_signals": [s.model_dump() for s in result["stop_hunt_signals"]],
        "fake_breakout_signals": [s.model_dump() for s in result["fake_breakout_signals"]],
        "volume_profile": profile.to_dict(orient="records") if not profile.empty else [],
        "active_alerts": result["active_alerts"],
    }

    # پاکسازی نهایی (در صورت وجود NaN در سایر بخش‌ها)
    return clean_nan(response)


@app.get("/api/volume-profile")
async def get_volume_profile(bins: int = 50):
    """Get volume profile data."""
    df = connector.get_candles_df(count=500)
    
    if df.empty:
        return {"error": "No data available"}
    
    profile = signal_engine.volume_engine.get_volume_profile(df, bins=bins)
    
    return {
        "data_category": "derived",
        "bins": bins,
        "profile": profile.to_dict(orient="records"),
    }


@app.get("/api/liquidity-zones")
async def get_liquidity_zones():
    """Get estimated liquidity zones."""
    df = connector.get_candles_df(count=1000)
    
    if df.empty:
        return {"error": "No data available"}
    
    zones = signal_engine.liquidity_engine.analyze(df)
    
    return {
        "data_category": "estimated",
        "disclaimer": (
            "Liquidity zones are ESTIMATED from price structure. "
            "No real order book data is available."
        ),
        "zones": [z.model_dump() for z in zones],
    }


@app.get("/api/signals")
async def get_signals(signal_type: str | None = None, limit: int = 50):
    """Get stored signals from database."""
    signals = await db.get_signals(signal_type=signal_type, limit=limit)
    return {
        "data_category": "estimated",
        "count": len(signals),
        "signals": signals,
    }


# ── WebSocket ────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming."""
    await websocket.accept()
    ws_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(ws_clients)}")
    
    try:
        while True:
            # Send periodic updates
            snapshot = collector.get_snapshot()
            if snapshot:
                await websocket.send_json({
                    "type": "snapshot",
                    "data": snapshot.model_dump(),
                })
            
            # Run analysis periodically
            df = connector.get_candles_df(count=200)
            if not df.empty:
                result = signal_engine.process(df)
                
                if result["active_alerts"]:
                    await websocket.send_json({
                        "type": "alerts",
                        "data": result["active_alerts"],
                    })
            
            await asyncio.sleep(
                CONFIG["dashboard"]["update_interval_ms"] / 1000.0
            )
    
    except WebSocketDisconnect:
        ws_clients.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(ws_clients)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in ws_clients:
            ws_clients.remove(websocket)


# ── Assistant endpoint ──────────────────────────────────────────────────

@app.get("/api/assistant")
async def get_assistant(force: bool = False):
    """Get assistant analysis as plain text."""
    text = get_assistant_text(force_refresh=force)
    return PlainTextResponse(content=text, media_type="text/plain")


# ── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=CONFIG["api"]["host"],
        port=CONFIG["api"]["port"],
        reload=True,
    )