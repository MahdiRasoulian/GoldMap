"""Goldmap Dashboard — Institutional-grade trading terminal interface.

Built with Dash and Plotly for real-time visualization.
"""

import json
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from loguru import logger

from config.loader import CONFIG
from core.mt5_connector import MT5Connector
from engines.signal_engine import SignalEngine

# Initialize
connector = MT5Connector()
connector.connect()
signal_engine = SignalEngine()

# Dash app with dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Goldmap — XAUUSD Intelligence",
    update_title=None,
)

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
            ], width=4),
            dbc.Col([
                html.Div(id="live-price", className="text-end"),
            ], width=4),
            dbc.Col([
                html.Div(id="session-info", className="text-end"),
            ], width=4),
        ], className="mb-3 border-bottom border-secondary pb-2"),
        
        # Main content
        dbc.Row([
            # Left panel — Charts
            dbc.Col([
                # Price chart with heatmap overlay
                dbc.Card([
                    dbc.CardHeader("Price Action & Liquidity Heatmap"),
                    dbc.CardBody([
                        dcc.Graph(
                            id="main-chart",
                            config={"displayModeBar": False},
                            style={"height": "400px"},
                        ),
                    ]),
                ], className="mb-2", color="dark", outline=True),
                
                # Volume profile
                dbc.Card([
                    dbc.CardHeader("Volume Profile"),
                    dbc.CardBody([
                        dcc.Graph(
                            id="volume-profile-chart",
                            config={"displayModeBar": False},
                            style={"height": "200px"},
                        ),
                    ]),
                ], className="mb-2", color="dark", outline=True),
            ], width=8),
            
            # Right panel — Signals & Alerts
            dbc.Col([
                # Alert center
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("🚨 Alert Center", className="text-danger"),
                    ]),
                    dbc.CardBody(
                        id="alert-center",
                        style={
                            "maxHeight": "200px",
                            "overflowY": "auto",
                        },
                    ),
                ], className="mb-2", color="dark", outline=True),
                
                # Detectors
                dbc.Card([
                    dbc.CardHeader("Detection Engines"),
                    dbc.CardBody([
                        html.Div(id="absorption-status"),
                        html.Hr(className="border-secondary my-2"),
                        html.Div(id="stop-hunt-status"),
                        html.Hr(className="border-secondary my-2"),
                        html.Div(id="fake-breakout-status"),
                    ]),
                ], className="mb-2", color="dark", outline=True),
                
                # Statistics
                dbc.Card([
                    dbc.CardHeader("Statistics"),
                    dbc.CardBody(id="statistics-panel"),
                ], className="mb-2", color="dark", outline=True),
                
                # Data disclaimer
                dbc.Card([
                    dbc.CardBody([
                        html.Small([
                            html.Strong("⚠️ Data Classification"),
                            html.Br(),
                            html.Span(
                                "🟢 Observed ", className="text-success"
                            ),
                            html.Span(
                                "🟡 Derived ", className="text-warning"
                            ),
                            html.Span(
                                "🔴 Estimated", className="text-danger"
                            ),
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
        
        # Bottom row — Session map and heatmap
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Liquidity Heatmap"),
                    dbc.CardBody([
                        dcc.Graph(
                            id="heatmap-chart",
                            config={"displayModeBar": False},
                            style={"height": "250px"},
                        ),
                    ]),
                ], color="dark", outline=True),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Session Map"),
                    dbc.CardBody([
                        dcc.Graph(
                            id="session-chart",
                            config={"displayModeBar": False},
                            style={"height": "250px"},
                        ),
                    ]),
                ], color="dark", outline=True),
            ], width=6),
        ], className="mt-2"),
        
        # Update interval
        dcc.Interval(
            id="update-interval",
            interval=CONFIG["dashboard"]["update_interval_ms"],
            n_intervals=0,
        ),
    ],
)


# --- Callbacks ---

@callback(
    [
        Output("live-price", "children"),
        Output("session-info", "children"),
        Output("main-chart", "figure"),
        Output("volume-profile-chart", "figure"),
        Output("heatmap-chart", "figure"),
        Output("session-chart", "figure"),
        Output("alert-center", "children"),
        Output("absorption-status", "children"),
        Output("stop-hunt-status", "children"),
        Output("fake-breakout-status", "children"),
        Output("statistics-panel", "children"),
    ],
    Input("update-interval", "n_intervals"),
)
def update_dashboard(n):
    """Main update callback — refreshes all dashboard components."""
    
    # Get data
    df = connector.get_candles_df(count=500)
    snapshot = None
    
    price_data = connector.get_current_price()
    if price_data:
        snapshot = price_data
    
    # Run analysis
    analysis = signal_engine.process(df) if not df.empty else None
    
    # Build components
    live_price = _build_live_price(snapshot)
    session_info = _build_session_info()
    main_chart = _build_main_chart(df, analysis)
    volume_chart = _build_volume_profile(df, analysis)
    heatmap = _build_heatmap(df, analysis)
    session_chart = _build_session_chart(df)
    alerts = _build_alerts(analysis)
    absorption = _build_absorption_status(analysis)
    stop_hunt = _build_stop_hunt_status(analysis)
    fake_breakout = _build_fake_breakout_status(analysis)
    stats = _build_statistics(df, analysis)
    
    return [
        live_price, session_info, main_chart, volume_chart,
        heatmap, session_chart, alerts, absorption,
        stop_hunt, fake_breakout, stats,
    ]


def _build_live_price(snapshot):
    """Build live price display."""
    if snapshot is None:
        return html.Span("Waiting for data...", className="text-muted")
    
    bid = snapshot["bid"]
    ask = snapshot["ask"]
    spread = snapshot["spread"]
    
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


def _build_session_info():
    """Build session information display."""
    hour = datetime.now().hour
    
    if 0 <= hour < 8:
        session = "🌏 Asian Session"
        color = "text-info"
    elif 8 <= hour < 13:
        session = "🇬🇧 London Session"
        color = "text-primary"
    elif 13 <= hour < 21:
        session = "🇺🇸 NY Session"
        color = "text-success"
    else:
        session = "🌙 Off-Hours"
        color = "text-muted"
    
    return html.Div([
        html.Span(session, className=color),
        html.Br(),
        html.Small(
            datetime.now().strftime("%H:%M:%S UTC"),
            className="text-muted",
        ),
    ])


def _build_main_chart(df, analysis):
    """Build main candlestick chart with overlays."""
    fig = go.Figure()
    
    if df.empty:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    
    # Candlestick
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
    
    # Add liquidity zones as horizontal bands
    if analysis and analysis.get("liquidity_zones"):
        for zone in analysis["liquidity_zones"][:10]:
            opacity = zone.strength * 0.3
            color = (
                "rgba(255, 193, 7, {})".format(opacity)
                if zone.zone_type in ["resistance", "round_number"]
                else "rgba(0, 212, 170, {})".format(opacity)
            )
            
            fig.add_hrect(
                y0=zone.price_low,
                y1=zone.price_high,
                fillcolor=color,
                line_width=0,
                annotation_text=f"LIQ ({zone.strength:.0%})",
                annotation_position="right",
            )
    
    # Add signal markers
    if analysis:
        # Stop hunt markers
        for sh in analysis.get("stop_hunt_signals", []):
            fig.add_annotation(
                x=sh.timestamp,
                y=sh.extreme_price,
                text="🎯",
                showarrow=False,
                font=dict(size=16),
            )
        
        # Absorption markers
        for ab in analysis.get("absorption_signals", []):
            marker = "🛡️" if ab.direction == "bullish" else "⚔️"
            fig.add_annotation(
                x=ab.timestamp,
                y=ab.price_level,
                text=marker,
                showarrow=False,
                font=dict(size=14),
            )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=20, t=10, b=30),
        showlegend=False,
    )
    
    return fig


def _build_volume_profile(df, analysis):
    """Build volume profile chart."""
    fig = go.Figure()
    
    if analysis and not analysis.get("volume_profile", pd.DataFrame()).empty:
        vp = analysis["volume_profile"]
        
        fig.add_trace(go.Bar(
            y=vp["price_level"],
            x=vp["normalized_volume"],
            orientation="h",
            marker_color=[
                "#00d4aa" if v > 0.7 else "#ffc107" if v > 0.4 else "#6c757d"
                for v in vp["normalized_volume"]
            ],
            name="Volume",
        ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=20, t=10, b=30),
        xaxis_title="Relative Volume",
        yaxis_title="Price",
        showlegend=False,
    )
    
    return fig


def _build_heatmap(df, analysis):
    """Build liquidity heatmap visualization."""
    fig = go.Figure()
    
    if df.empty:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    
    # Create heatmap from volume distribution over time
    n_time_bins = 50
    n_price_bins = 30
    
    time_indices = np.linspace(0, len(df) - 1, n_time_bins, dtype=int)
    price_min = df["low"].min()
    price_max = df["high"].max()
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
        margin=dict(l=60, r=20, t=10, b=30),
        xaxis_title="Time",
        yaxis_title="Price",
    )
    
    return fig


def _build_session_chart(df):
    """Build session range chart."""
    fig = go.Figure()
    
    if df.empty:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig
    
    # Volume bars colored by relative volume
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
    
    # Average line
    fig.add_hline(
        y=avg_vol,
        line_dash="dash",
        line_color="white",
        opacity=0.5,
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=10, b=30),
        showlegend=False,
    )
    
    return fig


def _build_alerts(analysis):
    """Build alert center content."""
    if not analysis or not analysis.get("active_alerts"):
        return html.Span("No active alerts", className="text-muted")
    
    alerts = []
    for alert in analysis["active_alerts"][:5]:
        badge_color = (
            "danger" if alert["confidence"] > 0.7
            else "warning" if alert["confidence"] > 0.5
            else "info"
        )
        
        category_badge = {
            "observed": "🟢",
            "derived": "🟡",
            "estimated": "🔴",
        }.get(alert["category"], "⚪")
        
        alerts.append(
            dbc.Alert([
                html.Span(f"{category_badge} "),
                dbc.Badge(
                    f"{alert['confidence']:.0%}",
                    color=badge_color,
                    className="me-2",
                ),
                html.Span(alert["message"], className="small"),
            ], color=badge_color, className="py-1 px-2 mb-1")
        )
    
    return html.Div(alerts)


def _build_absorption_status(analysis):
    """Build absorption detector status."""
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
            f"{latest.direction.title()} at ${latest.price_level:.2f}",
        ),
        html.Br(),
        html.Small(
            f"Confidence: {latest.confidence:.0%} | "
            f"Defended: {latest.defense_count}x",
            className="text-muted",
        ),
        html.Br(),
        html.Small("🔴 ESTIMATED", className="text-danger"),
    ])


def _build_stop_hunt_status(analysis):
    """Build stop hunt detector status."""
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
            f"{latest.hunt_direction.title()} ${latest.trigger_price:.2f}",
        ),
        html.Br(),
        html.Small(
            f"Extreme: ${latest.extreme_price:.2f} | "
            f"Confidence: {latest.confidence:.0%}",
            className="text-muted",
        ),
        html.Br(),
        html.Small("🔴 ESTIMATED", className="text-danger"),
    ])


def _build_fake_breakout_status(analysis):
    """Build fake breakout detector status."""
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
            f"{latest.breakout_direction.title()} at "
            f"${latest.breakout_price:.2f}",
        ),
        html.Br(),
        html.Small(
            f"Returned to: ${latest.return_price:.2f} | "
            f"Confidence: {latest.confidence:.0%}",
            className="text-muted",
        ),
        html.Br(),
        html.Small("🔴 ESTIMATED", className="text-danger"),
    ])


def _build_statistics(df, analysis):
    """Build statistics panel."""
    if df.empty:
        return html.Span("No data", className="text-muted")
    
    # Basic stats
    current_price = df["close"].iloc[-1]
    daily_range = df["high"].max() - df["low"].min()
    avg_volume = df["tick_volume"].mean()
    max_volume = df["tick_volume"].max()
    
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
    ]
    
    if analysis:
        n_alerts = len(analysis.get("active_alerts", []))
        stats.append(html.Div([
            html.Small("Active Alerts: ", className="text-muted"),
            html.Span(
                str(n_alerts),
                className="text-danger" if n_alerts > 0 else "",
            ),
        ]))
    
    return html.Div(stats)


# --- Run ---

if __name__ == "__main__":
    app.run_server(
        host=CONFIG["dashboard"]["host"],
        port=CONFIG["dashboard"]["port"],
        debug=True,
    )