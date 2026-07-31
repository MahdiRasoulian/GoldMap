"""Goldmap Dashboard — Institutional-grade trading terminal interface.

Built with Dash and Plotly for real-time visualization.

Timezone Strategy: All times displayed in UTC (broker time) with clear labeling.
Session detection aligned with UTC times.
"""

import json
import math
from datetime import datetime, timezone

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, callback, dcc, html, no_update
from dash.exceptions import PreventUpdate
from loguru import logger

from config.loader import CONFIG

# --- Configuration ---
# Set to "UTC" for broker time, or "Asia/Tehran" for local time
DISPLAY_TIMEZONE = "UTC"
TIMEZONE_LABEL = "UTC"  # Label shown on charts

# NaN cleanup helper
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


# Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Goldmap — XAUUSD Intelligence",
    update_title=None,
)

# --- Helper functions to fetch data from API ---

API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 15
CANDLE_LIMIT = 200

# Simple cache
_cache = {
    "candles": None,
    "analysis": None,
    "snapshot": None,
    "timestamp": None,
}


def _get_session_from_utc_hour(hour: int) -> tuple[str, str, str]:
    """Determine trading session from UTC hour.
    
    Returns: (session_name, emoji_label, css_class)
    """
    if 0 <= hour < 8:
        return "asian", "🌏 Asian Session", "text-info"
    elif 8 <= hour < 13:
        return "london", "🇬🇧 London Session", "text-primary"
    elif 13 <= hour < 17:
        return "overlap", "🇬🇧🇺🇸 London/NY Overlap", "text-success"
    elif 17 <= hour < 21:
        return "newyork", "🇺🇸 New York Session", "text-success"
    else:
        return "off_hours", "🌙 Off-Hours", "text-muted"


def fetch_candles(limit=CANDLE_LIMIT, force=False):
    """Fetch candle data from the running API.
    
    All times are kept in UTC for consistency with broker data.
    """
    if not force and _cache["candles"] is not None:
        return _cache["candles"]
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/candles?limit={limit}",
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            candles = data.get("candles", [])
            logger.info(f"Fetched {len(candles)} candles from API")
            if candles:
                df = pd.DataFrame(candles)
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
                
                # Keep times in UTC — do NOT convert to local timezone
                # This ensures consistency with broker time and session detection
                try:
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                except Exception as e:
                    logger.warning(f"Timezone localization note: {e}")
                
                for col in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                
                logger.info(f"DataFrame shape: {df.shape}")
                _cache["candles"] = df
                _cache["timestamp"] = datetime.now(timezone.utc)
                return df
        else:
            logger.error(f"API returned status {response.status_code}")
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching candles from API")
    except Exception as e:
        logger.error(f"Error fetching candles from API: {e}")
    return _cache.get("candles", pd.DataFrame())


def fetch_analysis(force=False):
    """Fetch full analysis from API."""
    if not force and _cache["analysis"] is not None:
        return _cache["analysis"]
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/analysis",
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            data = clean_nan(data)
            logger.info(f"Analysis keys: {list(data.keys())}")
            if data.get("volume_profile"):
                logger.info(f"Volume profile length: {len(data['volume_profile'])}")
            if data.get("liquidity_zones"):
                logger.info(f"Liquidity zones count: {len(data['liquidity_zones'])}")
            logger.info("Analysis fetched successfully")
            _cache["analysis"] = data
            _cache["timestamp"] = datetime.now(timezone.utc)
            return data
        else:
            logger.error(f"Analysis API returned status {response.status_code}")
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching analysis from API")
    except Exception as e:
        logger.error(f"Error fetching analysis from API: {e}")
    return _cache.get("analysis", None)


def fetch_snapshot(force=False):
    """Fetch current market snapshot from API."""
    if not force and _cache["snapshot"] is not None:
        return _cache["snapshot"]
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/snapshot",
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            _cache["snapshot"] = data
            _cache["timestamp"] = datetime.now(timezone.utc)
            return data
    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching snapshot from API")
    except Exception as e:
        logger.error(f"Error fetching snapshot from API: {e}")
    return _cache.get("snapshot", None)


# --- Layout ---

app.layout = dbc.Container(
    fluid=True,
    className="p-2",
    children=[
        # Header
        dbc.Row([
            dbc.Col([
                html.H2(
                    "⚡ GOLDMAP",
                    className="text-warning mb-0",
                    style={"fontWeight": "bold"},
                ),
                html.Small(
                    "XAUUSD Market Intelligence Platform",
                    className="text-muted",
                ),
            ], width=3),
            dbc.Col([
                html.Div(id="live-price", className="text-end"),
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div(id="session-info", className="text-end"),
                    html.Br(),
                    html.Small(
                        id="update-timestamp",
                        className="text-muted",
                        style={"fontSize": "10px"},
                    ),
                ]),
            ], width=3),
            dbc.Col([
                html.Div([
                    dbc.Button(
                        "❄️ Freeze",
                        id="freeze-btn",
                        color="secondary",
                        size="sm",
                        className="me-2",
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "🔄 Refresh",
                        id="refresh-btn",
                        color="primary",
                        size="sm",
                        n_clicks=0,
                    ),
                    html.Div(
                        id="freeze-status",
                        className="text-muted",
                        style={"fontSize": "10px", "marginTop": "4px"},
                    ),
                ], className="text-end"),
            ], width=3),
        ], className="mb-3 border-bottom border-secondary pb-2"),
        
        # Main content
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Price Action & Liquidity Zones",
                        html.Span(
                            f" (Time: {TIMEZONE_LABEL})",
                            className="text-muted",
                            style={"fontSize": "10px"},
                        ),
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-main",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id="main-chart",
                                    config={"displayModeBar": False},
                                    style={"height": "400px"},
                                )
                            ]
                        ),
                    ]),
                ], className="mb-2", color="dark", outline=True),
                
                dbc.Card([
                    dbc.CardHeader("Volume Profile"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-volume",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id="volume-profile-chart",
                                    config={"displayModeBar": False},
                                    style={"height": "200px"},
                                )
                            ]
                        ),
                    ]),
                ], className="mb-2", color="dark", outline=True),
            ], width=8),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("🚨 Alert Center", className="text-danger"),
                        html.Span(
                            " (Auto-refresh: 60s)",
                            className="text-muted",
                            style={"fontSize": "10px"},
                        ),
                    ]),
                    dbc.CardBody(
                        id="alert-center",
                        style={
                            "maxHeight": "200px",
                            "overflowY": "auto",
                        },
                    ),
                ], className="mb-2", color="dark", outline=True),
                
                dbc.Card([
                    dbc.CardHeader([
                        "Detection Engines",
                        html.Span(
                            " (Data: Estimated)",
                            className="text-muted",
                            style={"fontSize": "10px"},
                        ),
                    ]),
                    dbc.CardBody([
                        html.Div(id="absorption-status"),
                        html.Hr(className="border-secondary my-2"),
                        html.Div(id="stop-hunt-status"),
                        html.Hr(className="border-secondary my-2"),
                        html.Div(id="fake-breakout-status"),
                    ]),
                ], className="mb-2", color="dark", outline=True),
                
                dbc.Card([
                    dbc.CardHeader("Statistics"),
                    dbc.CardBody(id="statistics-panel"),
                ], className="mb-2", color="dark", outline=True),
                
                
                
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("🤖 Trading Assistant", className="text-info"),
                        html.Span(" (Auto-refresh: 30s)", className="text-muted", style={"fontSize": "10px"}),
                    ]),
                    dbc.CardBody(
                        id="assistant-output",
                        style={
                            "fontFamily": "monospace",
                            "fontSize": "12px",
                            "whiteSpace": "pre-wrap",
                            "maxHeight": "350px",
                            "overflowY": "auto",
                            "background": "rgba(0,0,0,0.3)",
                            "borderRadius": "4px",
                            "padding": "8px",
                        }
                    ),
                ], className="mb-2", color="dark", outline=True),                
                
                
                
                
                
                # Liquidity Zones Legend Panel
                dbc.Card([
                    dbc.CardHeader([
                        "📊 Top Liquidity Zones",
                        html.Span(
                            " (Estimated)",
                            className="text-danger",
                            style={"fontSize": "10px"},
                        ),
                    ]),
                    dbc.CardBody(id="liquidity-legend"),
                ], className="mb-2", color="dark", outline=True),
                
                dbc.Card([
                    dbc.CardBody([
                        html.Small([
                            html.Strong("⚠️ Data Classification"),
                            html.Br(),
                            html.Span("🟢 Observed ", className="text-success"),
                            html.Span("🟡 Derived ", className="text-warning"),
                            html.Span("🔴 Estimated", className="text-danger"),
                            html.Br(),
                            html.Span(
                                "No real order book data available.",
                                className="text-muted",
                            ),
                        ]),
                    ]),
                ], color="dark", outline=True),
            ], width=4),
        ]),
        
        # Bottom row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Liquidity Heatmap",
                        html.Span(
                            f" ({TIMEZONE_LABEL})",
                            className="text-muted",
                            style={"fontSize": "10px"},
                        ),
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-heatmap",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id="heatmap-chart",
                                    config={"displayModeBar": False},
                                    style={"height": "250px"},
                                )
                            ]
                        ),
                    ]),
                ], color="dark", outline=True),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Session Volume Map",
                        html.Span(
                            f" ({TIMEZONE_LABEL})",
                            className="text-muted",
                            style={"fontSize": "10px"},
                        ),
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-session",
                            type="circle",
                            children=[
                                dcc.Graph(
                                    id="session-chart",
                                    config={"displayModeBar": False},
                                    style={"height": "250px"},
                                )
                            ]
                        ),
                    ]),
                ], color="dark", outline=True),
            ], width=6),
        ], className="mt-2"),
        
        # Timers
        dcc.Interval(
            id="snapshot-timer",
            interval=5000,
            n_intervals=0,
        ),
        
        dcc.Interval(
            id="assistant-timer",
            interval=30000,  # 30 seconds
            n_intervals=0,
        ),        
        
        
        dcc.Interval(
            id="chart-timer",
            interval=15000,
            n_intervals=0,
        ),
        dcc.Interval(
            id="analysis-timer",
            interval=60000,
            n_intervals=0,
        ),
    ],
)


# --- Callbacks ---

@callback(
    [
        Output("live-price", "children"),
        Output("session-info", "children"),
        Output("update-timestamp", "children"),
        Output("freeze-status", "children"),
    ],
    [
        Input("snapshot-timer", "n_intervals"),
        Input("refresh-btn", "n_clicks"),
    ],
    State("freeze-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_snapshot(n_intervals, refresh_clicks, freeze_clicks):
    if freeze_clicks and freeze_clicks % 2 == 1:
        raise PreventUpdate
    
    force = (refresh_clicks is not None and refresh_clicks > 0)
    snapshot = fetch_snapshot(force=force)
    
    now_utc = datetime.now(timezone.utc)
    timestamp = f"Last update: {now_utc.strftime('%H:%M:%S')} {TIMEZONE_LABEL}"
    
    live_price = _build_live_price(snapshot)
    session_info = _build_session_info(now_utc)
    freeze_status = html.Span("🟢 Live", className="text-success")
    
    return live_price, session_info, timestamp, freeze_status


@callback(
    [
        Output("main-chart", "figure"),
        Output("heatmap-chart", "figure"),
        Output("session-chart", "figure"),
    ],
    [
        Input("chart-timer", "n_intervals"),
        Input("refresh-btn", "n_clicks"),
    ],
    State("freeze-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_charts(n_intervals, refresh_clicks, freeze_clicks):
    if freeze_clicks and freeze_clicks % 2 == 1:
        raise PreventUpdate
    
    force = (refresh_clicks is not None and refresh_clicks > 0)
    df = fetch_candles(force=force)
    if df.empty:
        empty_fig = _empty_dark_figure()
        return empty_fig, empty_fig, empty_fig
    
    analysis = _cache.get("analysis")
    main_chart = _build_main_chart(df, analysis)
    heatmap = _build_heatmap(df, analysis)
    session_chart = _build_session_chart(df)
    return main_chart, heatmap, session_chart


@callback(
    [
        Output("volume-profile-chart", "figure"),
        Output("alert-center", "children"),
        Output("absorption-status", "children"),
        Output("stop-hunt-status", "children"),
        Output("fake-breakout-status", "children"),
        Output("statistics-panel", "children"),
        Output("liquidity-legend", "children"),
    ],
    [
        Input("analysis-timer", "n_intervals"),
        Input("refresh-btn", "n_clicks"),
    ],
    State("freeze-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_analysis_and_alerts(n_intervals, refresh_clicks, freeze_clicks):
    if freeze_clicks and freeze_clicks % 2 == 1:
        raise PreventUpdate
    
    force = (refresh_clicks is not None and refresh_clicks > 0)
    analysis = fetch_analysis(force=force)
    df = fetch_candles(force=force)
    if df.empty:
        return (no_update, no_update, no_update, no_update,
                no_update, no_update, no_update)
    
    volume_chart = _build_volume_profile(df, analysis)
    alerts = _build_alerts(analysis)
    absorption = _build_absorption_status(analysis)
    stop_hunt = _build_stop_hunt_status(analysis)
    fake_breakout = _build_fake_breakout_status(analysis)
    stats = _build_statistics(df, analysis)
    liquidity_legend = _build_liquidity_legend(analysis)
    
    return (volume_chart, alerts, absorption, stop_hunt,
            fake_breakout, stats, liquidity_legend)



@callback(
    Output("assistant-output", "children"),
    [Input("assistant-timer", "n_intervals"), Input("refresh-btn", "n_clicks")],
    State("freeze-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_assistant(n_intervals, refresh_clicks, freeze_clicks):
    if freeze_clicks and freeze_clicks % 2 == 1:
        raise PreventUpdate

    try:
        response = requests.get(
            f"{API_BASE_URL}/api/assistant?force={bool(refresh_clicks)}",
            timeout=10
        )
        if response.status_code == 200:
            return html.Pre(
                response.text,
                style={
                    "margin": 0,
                    "fontSize": "11px",
                    "whiteSpace": "pre-wrap",
                    "color": "#00ffaa" if "ALERT" in response.text or "ACTION" in response.text else "#ffffff",
                }
            )
        else:
            return html.Span("Assistant unavailable", className="text-muted")
    except Exception as e:
        logger.error(f"Error fetching assistant: {e}")
        return html.Span("Error loading assistant", className="text-danger")


# --- Helper: Empty figure ---

def _empty_dark_figure(message="No data available"):
    """Create an empty dark-themed figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=10, b=30),
        annotations=[dict(
            text=message,
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="gray", size=14),
        )]
    )
    return fig


# --- Common axis formatting ---

def _standard_xaxis(title=None):
    """Standard x-axis config for consistent time display."""
    config = dict(
        tickformat="%H:%M",
        tickangle=45,
        tickfont=dict(size=8),
        nticks=12,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
    )
    if title:
        config["title"] = title
    return config


def _standard_yaxis(title=None):
    """Standard y-axis config with automargin to prevent cutoff."""
    config = dict(
        tickformat=".2f",
        tickfont=dict(size=8),
        automargin=True,
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
    )
    if title:
        config["title"] = title
    return config


# --- Data-quality helpers ---
#
# The two helpers below fix real bugs found in the previous version:
#
# 1. `_normalize_zone_strengths` — the analysis engine frequently returns a
#    `strength` value clipped at 1.0 for *every* liquidity zone, which is why
#    the old dashboard showed "100%" (or a flat "10%") on every single row of
#    the "Top Liquidity Zones" panel. That is a backend scoring bug (ideally
#    fixed in the detection engine itself), but the frontend should never
#    display a meaningless, undifferentiated number — so we defensively
#    rescale the values here purely for *display* purposes, while leaving the
#    original data untouched for anything else that consumes it.
#
# 2. `_filter_signals_in_range` — stop-hunt/absorption/fake-breakout signals
#    can reference timestamps outside the currently fetched candle window
#    (e.g. an older signal that predates the last `CANDLE_LIMIT` candles).
#    Plotting those anyway silently extends the x-axis and produces the
#    stray, disconnected marker dots seen to the left of the candlesticks.
#    We drop anything outside the visible candle range before rendering.

def _normalize_zone_strengths(zones, floor=0.30, ceiling=0.98):
    """Rescale liquidity-zone strengths into a readable, differentiated range.

    - If the incoming values already vary, min-max rescale them into
      [floor, ceiling] so the strongest visible zone reads near-ceiling and
      the weakest reads near-floor.
    - If the incoming values are (near) identical — the observed bug —
      fall back to rank-based spacing so zones remain visually distinguishable
      instead of all showing the same inflated percentage.
    """
    if not zones:
        return zones

    raw_values = [float(z.get("strength", 0) or 0) for z in zones]
    lo, hi = min(raw_values), max(raw_values)
    spread = hi - lo

    rescaled = []
    if spread < 0.05:
        n = max(len(zones) - 1, 1)
        for i, z in enumerate(zones):
            z = dict(z)
            z["strength"] = ceiling - (i / n) * (ceiling - floor)
            rescaled.append(z)
    else:
        for z, raw in zip(zones, raw_values):
            z = dict(z)
            z["strength"] = floor + ((raw - lo) / spread) * (ceiling - floor)
            rescaled.append(z)
    return rescaled


def _filter_signals_in_range(signals, start, end, ts_key="timestamp"):
    """Keep only signals whose timestamp falls within [start, end] (UTC)."""
    kept = []
    for sig in signals or []:
        if not isinstance(sig, dict):
            continue
        ts = sig.get(ts_key)
        if not ts:
            continue
        try:
            ts_parsed = pd.to_datetime(ts)
            if ts_parsed.tzinfo is None:
                ts_parsed = ts_parsed.tz_localize("UTC")
            if start <= ts_parsed <= end:
                kept.append(sig)
        except Exception:
            continue
    return kept


# --- Build functions ---

def _build_live_price(snapshot):
    """Build live price display."""
    try:
        if snapshot is None or "error" in snapshot:
            return html.Span("Waiting for data...", className="text-muted")
        bid = snapshot.get("bid", 0)
        ask = snapshot.get("ask", 0)
        spread = snapshot.get("spread", 0)
        return html.Div([
            html.H3(
                f"${bid:.2f}",
                className="text-warning mb-0",
                style={"fontFamily": "monospace"},
            ),
            html.Small(
                f"Spread: {spread:.2f} | Ask: ${ask:.2f}",
                className="text-muted",
            ),
        ])
    except Exception as e:
        logger.error(f"Error in _build_live_price: {e}")
        return html.Span("Error", className="text-danger")


def _build_session_info(now_utc: datetime):
    """Build session information display using UTC time."""
    _, session_label, color_class = _get_session_from_utc_hour(now_utc.hour)
    
    return html.Div([
        html.Span(session_label, className=color_class),
        html.Br(),
        html.Small(
            f"{now_utc.strftime('%H:%M:%S')} {TIMEZONE_LABEL}",
            className="text-muted",
        ),
    ])


def _build_main_chart(df, analysis):
    """Build main candlestick chart with clean overlays.
    
    Fixes applied:
    - No inline text labels on liquidity zones (only colored bands)
    - Smaller markers for signals (9pt)
    - Liquidity zone legend moved to separate panel
    - Increased margins for readability
    - Consistent time formatting
    """
    fig = go.Figure()
    try:
        if df.empty:
            return _empty_dark_figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="XAUUSD",
            increasing_line_color="#00d4aa",
            decreasing_line_color="#ff4757",
        ))
        
        # Chart-visible time range — used to clamp the x-axis and to drop
        # any signal/zone timestamps that fall outside the fetched candles.
        chart_start, chart_end = df.index.min(), df.index.max()

        # Liquidity Zones as clean colored bands (NO text annotations).
        # Strength is normalized here (see _normalize_zone_strengths) so
        # zones are visually differentiated instead of all rendering at the
        # same fixed opacity when the engine returns identical/clipped scores.
        if analysis and analysis.get("liquidity_zones"):
            raw_zones = [z for z in analysis["liquidity_zones"] if isinstance(z, dict)]
            zones = _normalize_zone_strengths(raw_zones)[:10]
            for zone in zones:
                try:
                    price_low = zone.get("price_low")
                    price_high = zone.get("price_high")
                    if price_low is None or price_high is None:
                        continue

                    strength = zone.get("strength", 0.5)
                    opacity = min(0.20, strength * 0.20)
                    zone_type = zone.get("zone_type", "support")

                    if zone_type in ["resistance", "round_number"]:
                        color = f"rgba(255, 193, 7, {opacity})"
                    else:
                        color = f"rgba(0, 212, 170, {opacity})"

                    # Clean rectangle — NO annotation_text, so nothing is
                    # drawn on top of the candlesticks. Zone details live in
                    # the dedicated "Top Liquidity Zones" side panel instead.
                    fig.add_hrect(
                        y0=price_low,
                        y1=price_high,
                        fillcolor=color,
                        line_width=0,
                    )
                except Exception as e:
                    logger.warning(f"Error adding liquidity zone: {e}")

        # Stop Hunt markers — small font (9pt), subtle red/orange.
        # Only signals whose timestamp falls inside the visible candle
        # window are plotted, preventing stray disconnected markers.
        if analysis:
            stop_hunts = _filter_signals_in_range(
                analysis.get("stop_hunt_signals", []), chart_start, chart_end
            )
            for sh in stop_hunts:
                try:
                    extreme = sh.get("extreme_price")
                    ts = sh.get("timestamp")
                    if ts and extreme is not None:
                        fig.add_annotation(
                            x=pd.to_datetime(ts),
                            y=extreme,
                            text="🎯",
                            showarrow=False,
                            font=dict(size=9, color="#ff6b6b"),
                        )
                except Exception as e:
                    logger.warning(f"Error adding stop hunt marker: {e}")

            # Absorption markers — small font (9pt), subtle amber.
            absorptions = _filter_signals_in_range(
                analysis.get("absorption_signals", []), chart_start, chart_end
            )
            for ab in absorptions:
                try:
                    price = ab.get("price_level")
                    ts = ab.get("timestamp")
                    if ts and price is not None:
                        marker = "🛡️" if ab.get("direction") == "bullish" else "⚔️"
                        fig.add_annotation(
                            x=pd.to_datetime(ts),
                            y=price,
                            text=marker,
                            showarrow=False,
                            font=dict(size=9, color="#ffd93d"),
                        )
                except Exception as e:
                    logger.warning(f"Error adding absorption marker: {e}")

        # Layout with proper margins and consistent formatting.
        # NOTE: the old "top strongest zones" text used to be duplicated
        # here as small labels along the right edge of the plot — that is
        # now shown once, cleanly, in the "Top Liquidity Zones" side panel
        # (see _build_liquidity_legend), so it has been removed from the
        # chart itself to eliminate overlap/clutter.
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_rangeslider_visible=False,
            margin=dict(l=70, r=30, t=10, b=40),
            showlegend=False,
            yaxis=dict(
                tickformat=".2f",
                tickfont=dict(size=9),
                side="left",
                automargin=True,
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
            ),
            xaxis={**_standard_xaxis(title=f"Time ({TIMEZONE_LABEL})"),
                   "range": [chart_start, chart_end]},
        )

    except Exception as e:
        logger.error(f"Error in _build_main_chart: {e}", exc_info=True)
        return _empty_dark_figure("Chart error")

    return fig


def _build_volume_profile(df, analysis):
    """Build volume profile chart."""
    fig = go.Figure()
    try:
        if analysis and analysis.get("volume_profile"):
            vp_data = analysis["volume_profile"]
            if vp_data and isinstance(vp_data, list) and len(vp_data) > 0:
                if "price_level" in vp_data[0] and "normalized_volume" in vp_data[0]:
                    vp = pd.DataFrame(vp_data)
                    fig.add_trace(go.Bar(
                        y=vp["price_level"],
                        x=vp["normalized_volume"],
                        orientation="h",
                        marker_color=[
                            "#00d4aa" if v and v > 0.7
                            else "#ffc107" if v and v > 0.4
                            else "#6c757d"
                            for v in vp["normalized_volume"]
                        ],
                        name="Volume",
                    ))
    except Exception as e:
        logger.error(f"Error in _build_volume_profile: {e}", exc_info=True)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=10, b=30),
        showlegend=False,
        xaxis=dict(
            title="Relative Volume",
            tickfont=dict(size=8),
            automargin=True,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=_standard_yaxis(title="Price"),
    )
    return fig


def _build_heatmap(df, analysis):
    """Build liquidity heatmap with consistent time formatting."""
    fig = go.Figure()
    try:
        if df.empty:
            return _empty_dark_figure()
        
        n_time_bins = min(50, len(df))
        n_price_bins = 30
        time_indices = np.linspace(0, len(df) - 1, n_time_bins, dtype=int)
        price_min = df["low"].min()
        price_max = df["high"].max()
        
        if price_min == price_max:
            price_min -= 0.1
            price_max += 0.1
        
        price_bins = np.linspace(price_min, price_max, n_price_bins)
        heatmap_data = np.zeros((n_price_bins - 1, n_time_bins))
        
        for t_idx, df_idx in enumerate(time_indices):
            row = df.iloc[df_idx]
            for p_idx in range(n_price_bins - 1):
                if price_bins[p_idx] <= row["high"] and price_bins[p_idx + 1] >= row["low"]:
                    heatmap_data[p_idx, t_idx] = row["tick_volume"]
        
        fig.add_trace(go.Heatmap(
            z=heatmap_data,
            x=df.index[time_indices],
            y=price_bins[:-1],
            colorscale=[
                [0, "rgba(0,0,0,0)"],
                [0.2, "rgba(0,0,139,0.5)"],
                [0.4, "rgba(0,100,200,0.7)"],
                [0.6, "rgba(255,193,7,0.8)"],
                [0.8, "rgba(255,100,0,0.9)"],
                [1.0, "rgba(255,0,0,1)"],
            ],
            showscale=False,
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60, r=20, t=10, b=40),
            xaxis=_standard_xaxis(title=f"Time ({TIMEZONE_LABEL})"),
            yaxis=_standard_yaxis(title="Price"),
        )
    except Exception as e:
        logger.error(f"Error in _build_heatmap: {e}", exc_info=True)
        return _empty_dark_figure("Heatmap error")
    
    return fig


def _build_session_chart(df):
    """Build session volume chart with consistent formatting."""
    fig = go.Figure()
    try:
        if df.empty:
            return _empty_dark_figure()
        
        avg_vol = df["tick_volume"].mean()
        colors = [
            "#ff4757" if v > avg_vol * 2.5
            else "#ffc107" if v > avg_vol * 1.5
            else "#00d4aa" if v > avg_vol
            else "#6c757d"
            for v in df["tick_volume"]
        ]
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=df["tick_volume"],
            marker_color=colors,
            name="Volume",
        ))
        
        fig.add_hline(
            y=avg_vol,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            annotation_text=f"Avg: {avg_vol:.0f}",
            annotation_position="top right",
            annotation_font=dict(size=8, color="rgba(255,255,255,0.6)"),
        )
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=50, r=20, t=10, b=40),
            showlegend=False,
            xaxis=_standard_xaxis(title=f"Time ({TIMEZONE_LABEL})"),
            yaxis=dict(
                title="Tick Volume",
                tickfont=dict(size=8),
                automargin=True,
                showgrid=True,
                gridcolor="rgba(255,255,255,0.05)",
            ),
        )
    except Exception as e:
        logger.error(f"Error in _build_session_chart: {e}", exc_info=True)
        return _empty_dark_figure("Session chart error")
    
    return fig


def _build_alerts(analysis):
    """Build alert center content."""
    try:
        if not analysis or not analysis.get("active_alerts"):
            return html.Span("No active alerts", className="text-muted")
        
        alerts = []
        for alert in analysis["active_alerts"][:5]:
            confidence = alert.get("confidence", 0)
            badge_color = (
                "danger" if confidence > 0.7
                else "warning" if confidence > 0.5
                else "info"
            )
            category_badge = {
                "observed": "🟢",
                "derived": "🟡",
                "estimated": "🔴",
            }.get(alert.get("category", "estimated"), "⚪")
            
            alerts.append(dbc.Alert([
                html.Span(f"{category_badge} "),
                dbc.Badge(
                    f"{confidence:.0%}",
                    color=badge_color,
                    className="me-2",
                ),
                html.Span(alert.get("message", ""), className="small"),
            ], color=badge_color, className="py-1 px-2 mb-1"))
        
        return html.Div(alerts)
    except Exception as e:
        logger.error(f"Error in _build_alerts: {e}")
        return html.Span("Error loading alerts", className="text-danger")


def _build_absorption_status(analysis):
    """Build absorption detector status."""
    try:
        if not analysis:
            return html.Span("No data", className="text-muted")
        signals = analysis.get("absorption_signals", [])
        if not signals:
            return html.Div([
                html.Strong("🛡️ Absorption: ", className="text-info"),
                html.Span("None detected", className="text-muted"),
            ])
        latest = signals[-1]
        return html.Div([
            html.Strong("🛡️ Absorption: ", className="text-info"),
            html.Span(
                f"{latest.get('direction', 'neutral').title()} at "
                f"${latest.get('price_level', 0):.2f}"
            ),
            html.Br(),
            html.Small(
                f"Confidence: {latest.get('confidence', 0):.0%} | "
                f"Defended: {latest.get('defense_count', 0)}x",
                className="text-muted",
            ),
            html.Br(),
            html.Small("🔴 ESTIMATED", className="text-danger"),
        ])
    except Exception as e:
        logger.error(f"Error in _build_absorption_status: {e}")
        return html.Span("Error", className="text-danger")


def _build_stop_hunt_status(analysis):
    """Build stop hunt detector status."""
    try:
        if not analysis:
            return html.Span("No data", className="text-muted")
        signals = analysis.get("stop_hunt_signals", [])
        if not signals:
            return html.Div([
                html.Strong("🎯 Stop Hunt: ", className="text-warning"),
                html.Span("None detected", className="text-muted"),
            ])
        latest = signals[-1]
        return html.Div([
            html.Strong("🎯 Stop Hunt: ", className="text-warning"),
            html.Span(
                f"{latest.get('hunt_direction', '').title()} "
                f"${latest.get('trigger_price', 0):.2f}"
            ),
            html.Br(),
            html.Small(
                f"Extreme: ${latest.get('extreme_price', 0):.2f} | "
                f"Confidence: {latest.get('confidence', 0):.0%}",
                className="text-muted",
            ),
            html.Br(),
            html.Small("🔴 ESTIMATED", className="text-danger"),
        ])
    except Exception as e:
        logger.error(f"Error in _build_stop_hunt_status: {e}")
        return html.Span("Error", className="text-danger")


def _build_fake_breakout_status(analysis):
    """Build fake breakout detector status."""
    try:
        if not analysis:
            return html.Span("No data", className="text-muted")
        signals = analysis.get("fake_breakout_signals", [])
        if not signals:
            return html.Div([
                html.Strong("💥 Fake Breakout: ", className="text-danger"),
                html.Span("None detected", className="text-muted"),
            ])
        latest = signals[-1]
        return html.Div([
            html.Strong("💥 Fake Breakout: ", className="text-danger"),
            html.Span(
                f"{latest.get('breakout_direction', '').title()} at "
                f"${latest.get('breakout_price', 0):.2f}"
            ),
            html.Br(),
            html.Small(
                f"Returned to: ${latest.get('return_price', 0):.2f} | "
                f"Confidence: {latest.get('confidence', 0):.0%}",
                className="text-muted",
            ),
            html.Br(),
            html.Small("🔴 ESTIMATED", className="text-danger"),
        ])
    except Exception as e:
        logger.error(f"Error in _build_fake_breakout_status: {e}")
        return html.Span("Error", className="text-danger")


def _build_statistics(df, analysis):
    """Build statistics panel."""
    try:
        if df.empty:
            return html.Div([
                html.Span("No data", className="text-muted"),
                html.Br(),
                html.Small("Waiting for data...", className="text-muted"),
            ])
        
        current_price = df["close"].iloc[-1]
        daily_range = df["high"].max() - df["low"].min()
        avg_volume = df["tick_volume"].mean()
        max_volume = df["tick_volume"].max()
        n_alerts = len(analysis.get("active_alerts", [])) if analysis else 0
        
        stats = [
            html.Div([
                html.Small("Current: ", className="text-muted"),
                html.Span(f"${current_price:.2f}", className="text-warning"),
            ]),
            html.Div([
                html.Small("Range: ", className="text-muted"),
                html.Span(f"${daily_range:.2f}"),
            ]),
            html.Div([
                html.Small("Avg Vol: ", className="text-muted"),
                html.Span(f"{avg_volume:.0f}"),
            ]),
            html.Div([
                html.Small("Max Vol: ", className="text-muted"),
                html.Span(f"{max_volume}", className="text-danger"),
            ]),
            html.Div([
                html.Small("Active Alerts: ", className="text-muted"),
                html.Span(
                    str(n_alerts),
                    className="text-danger" if n_alerts > 0 else "",
                ),
            ]),
        ]
        return html.Div(stats)
    except Exception as e:
        logger.error(f"Error in _build_statistics: {e}")
        return html.Span("Error", className="text-danger")


def _build_liquidity_legend(analysis):
    """Build a separate liquidity zones legend panel.
    
    This replaces inline chart labels to prevent overlapping.
    Shows top 5 strongest zones with price levels and strength.
    """
    try:
        if not analysis or not analysis.get("liquidity_zones"):
            return html.Span("No zones detected", className="text-muted")

        raw_zones = [z for z in analysis["liquidity_zones"] if isinstance(z, dict)]
        # Normalize BEFORE filtering/sorting so the displayed percentages are
        # real, differentiated values rather than the engine's clipped scores.
        zones = _normalize_zone_strengths(raw_zones)
        strong_zones = sorted(
            [z for z in zones if z.get("strength", 0) > 0.2],
            key=lambda z: z.get("strength", 0),
            reverse=True,
        )[:5]
        
        if not strong_zones:
            return html.Span("No significant zones", className="text-muted")
        
        rows = []
        for zone in strong_zones:
            price_low = zone.get("price_low", 0)
            price_high = zone.get("price_high", 0)
            strength = zone.get("strength", 0)
            zone_type = zone.get("zone_type", "support")
            
            # Color indicator
            if zone_type in ["resistance", "round_number"]:
                color_class = "text-warning"
                icon = "▲"
            else:
                color_class = "text-info"
                icon = "▼"
            
            rows.append(html.Div([
                html.Span(f"{icon} ", className=color_class),
                html.Span(
                    f"${price_low:.2f} - ${price_high:.2f}",
                    style={"fontFamily": "monospace", "fontSize": "11px"},
                ),
                html.Span(
                    f" ({strength:.0%})",
                    className=color_class,
                    style={"fontSize": "10px"},
                ),
            ], className="mb-1"))
        
        return html.Div(rows)
    except Exception as e:
        logger.error(f"Error in _build_liquidity_legend: {e}")
        return html.Span("Error", className="text-danger")


# --- Run ---

if __name__ == "__main__":
    app.run(
        host=CONFIG["dashboard"]["host"],
        port=CONFIG["dashboard"]["port"],
        debug=False,
    )