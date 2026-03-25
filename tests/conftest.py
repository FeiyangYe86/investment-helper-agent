"""
Shared pytest fixtures for investment-agent test suite.

Design principles:
  - All fixtures are pure offline — no real HTTP calls in unit/contract tests.
  - Factories return callables so tests can generate varied payloads easily.
  - Fixtures are scoped to "function" by default (safe, no cross-test leakage).
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────────────────────────────────────
# FRED API Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_fred_observations(values: list[float], base_date: str = "2026-01-01") -> list[dict]:
    """
    Build a list of fake FRED observation dicts, newest first.
    Mirrors the shape returned by _fetch_fred_series().
    """
    start = datetime.strptime(base_date, "%Y-%m-%d")
    result = []
    for i, v in enumerate(values):
        date = start - timedelta(days=i)
        result.append({"date": date.strftime("%Y-%m-%d"), "value": str(v)})
    return result  # newest first


@pytest.fixture
def fred_audusd_rising():
    """AUD/USD trending upward — 90 observations, newest = 0.68 vs oldest = 0.64."""
    base = 0.64
    values = [base + (i * 0.0004) for i in range(90)]
    values.reverse()  # newest first
    return make_fred_observations(values)


@pytest.fixture
def fred_audusd_falling():
    """AUD/USD trending downward — newest = 0.60 vs 90-day-ago = 0.68."""
    base = 0.68
    values = [base - (i * 0.0009) for i in range(90)]
    values.reverse()
    return make_fred_observations(values)


@pytest.fixture
def fred_audusd_flat():
    """AUD/USD flat — all values ≈ 0.65."""
    return make_fred_observations([0.65] * 90)


@pytest.fixture
def fred_rba_rising():
    """RBA rate rising: current 4.35, previous 4.10."""
    return make_fred_observations([4.35, 4.10, 3.85])


@pytest.fixture
def fred_rba_falling():
    """RBA rate falling: current 3.85, previous 4.10."""
    return make_fred_observations([3.85, 4.10, 4.35])


@pytest.fixture
def fred_rba_flat():
    """RBA rate unchanged for 3 periods."""
    return make_fred_observations([4.35, 4.35, 4.35])


@pytest.fixture
def fred_commodity_rising():
    """Commodity rising 5% over 21 periods."""
    values = [100.0 + (i * 0.25) for i in range(90)]
    values.reverse()
    return make_fred_observations(values)


@pytest.fixture
def fred_commodity_falling():
    """Commodity falling 5% over 21 periods."""
    values = [100.0 - (i * 0.25) for i in range(90)]
    values.reverse()
    return make_fred_observations(values)


# ─────────────────────────────────────────────────────────────────────────────
# ASX Announcements Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_asx_api_response(announcements: list[dict]) -> dict:
    """Build a fake MarkitDigital ASX API JSON response body.

    Confirmed schema (March 2026):
        { "data": { "items": [...], "symbol": "TST", "displayName": "...", "xid": "..." } }
    Each item: { "headline", "date", "isPriceSensitive", "announcementType", "documentKey", ... }
    """
    return {
        "data": {
            "displayName": "TEST COMPANY LTD",
            "symbol": "TST",
            "xid": "000000",
            "items": [
                {
                    "headline": ann["title"],
                    "date": ann.get("date", "2026-03-20T10:00:00.000Z"),
                    "isPriceSensitive": ann.get("market_sensitive", False),
                    "announcementType": ann.get("announcement_type", "COMPANY ADMINISTRATION"),
                    "documentKey": ann.get("documentKey", f"2924-00000000-{i}"),
                    "fileSize": "100KB",
                    "url": "",
                }
                for i, ann in enumerate(announcements)
            ],
        }
    }


@pytest.fixture
def asx_positive_announcements():
    return make_asx_api_response([
        {"title": "CBA announces record dividend payment to shareholders", "market_sensitive": True},
        {"title": "New partnership agreement signed with major retailer"},
        {"title": "Contract win: $200M government infrastructure project"},
    ])


@pytest.fixture
def asx_red_flag_single():
    """One red flag — should produce signal='warning'."""
    return make_asx_api_response([
        {"title": "ASIC commences investigation into lending practices", "market_sensitive": True},
        {"title": "Quarterly update: operations on track"},
    ])


@pytest.fixture
def asx_red_flag_multiple():
    """Multiple red flags — should produce signal='negative'."""
    return make_asx_api_response([
        {"title": "Capital raise: $150M placement at $2.40 per share", "market_sensitive": True},
        {"title": "CEO resignation effective immediately", "market_sensitive": True},
        {"title": "ASIC investigation update"},
    ])


@pytest.fixture
def asx_neutral_announcements():
    return make_asx_api_response([
        {"title": "Change in substantial holder"},
        {"title": "Appendix 3B — new issue of securities"},
        {"title": "Notice of Annual General Meeting"},
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Shortman HTML Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_shortman_html(short_pct: float, prev_pct: float = None, date: str = "2026-03-20") -> str:
    """Build a minimal HTML page that mimics Shortman's structure."""
    prev_section = f"<p>Previous: {prev_pct}%</p>" if prev_pct is not None else ""
    return f"""
    <html><body>
        <h1>Short Interest for CBA</h1>
        <p>Short: {short_pct}%</p>
        {prev_section}
        <p>Data as of: {date}</p>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Short position</td><td>{short_pct}%</td></tr>
        </table>
    </body></html>
    """


@pytest.fixture
def shortman_low_short():
    """2.5% short — should produce bullish signal."""
    return make_shortman_html(2.5, prev_pct=2.8)


@pytest.fixture
def shortman_high_short():
    """16.0% short — should produce warning signal."""
    return make_shortman_html(16.0, prev_pct=14.5)


@pytest.fixture
def shortman_moderate_short():
    """7.0% short, decreasing — should produce bullish signal."""
    return make_shortman_html(7.0, prev_pct=8.5)


@pytest.fixture
def shortman_bearish_short():
    """11.0% short — should produce bearish signal."""
    return make_shortman_html(11.0, prev_pct=9.0)


@pytest.fixture
def shortman_no_data():
    """Page with no parseable percentage."""
    return "<html><body><p>No data available for this ticker.</p></body></html>"


# ─────────────────────────────────────────────────────────────────────────────
# News / RSS Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_newsapi_response(articles: list[dict]) -> dict:
    return {
        "status": "ok",
        "totalResults": len(articles),
        "articles": [
            {
                "title": a["title"],
                "publishedAt": a.get("date", "2026-03-20T10:00:00Z"),
                "url": a.get("url", "https://example.com/article"),
                "source": {"name": "Reuters"},
            }
            for a in articles
        ],
    }


def make_yahoo_rss(items: list[dict]) -> bytes:
    items_xml = "\n".join(
        f"<item>"
        f"<title>{i['title']}</title>"
        f"<pubDate>{i.get('date', 'Mon, 20 Mar 2026 10:00:00 GMT')}</pubDate>"
        f"<link>{i.get('url', 'https://finance.yahoo.com/news/article')}</link>"
        f"</item>"
        for i in items
    )
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Yahoo Finance</title>
        {items_xml}
      </channel>
    </rss>"""
    return xml.encode("utf-8")


@pytest.fixture
def newsapi_positive_response():
    return make_newsapi_response([
        {"title": "CBA reports record annual profit beating analyst expectations"},
        {"title": "Commonwealth Bank announces special dividend for shareholders"},
        {"title": "CBA launches new digital banking partnership expanding growth"},
    ])


@pytest.fixture
def newsapi_red_flag_response():
    return make_newsapi_response([
        {"title": "ASIC launches investigation into Commonwealth Bank lending practices"},
        {"title": "CBA faces class action over alleged misleading conduct"},
    ])


@pytest.fixture
def newsapi_empty_response():
    return {"status": "ok", "totalResults": 0, "articles": []}


@pytest.fixture
def yahoo_rss_positive():
    return make_yahoo_rss([
        {"title": "CBA posts record quarterly profit above expectations"},
        {"title": "CBA dividend yield upgraded by major analyst"},
    ])


@pytest.fixture
def yahoo_rss_negative():
    return make_yahoo_rss([
        {"title": "CBA earnings miss estimates; guidance cut for next year"},
        {"title": "Commonwealth Bank faces rising bad debts amid economic weakness"},
    ])


@pytest.fixture
def yahoo_rss_empty():
    return make_yahoo_rss([])


# ─────────────────────────────────────────────────────────────────────────────
# Price DataFrame Fixture (for metrics_calculator)
# ─────────────────────────────────────────────────────────────────────────────

def make_price_df(
    n_days: int = 300,
    start_price: float = 100.0,
    drift: float = 0.001,
    volatility: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame of n_days for testing compute_metrics().
    Uses a simple geometric random walk so technical indicators have realistic values.
    """
    np.random.seed(seed)
    returns = np.random.normal(drift, volatility, n_days)
    closes = start_price * np.cumprod(1 + returns)

    # Synthetic OHLV around close
    highs = closes * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    opens = closes * (1 + np.random.normal(0, 0.003, n_days))
    volumes = np.random.randint(500_000, 5_000_000, n_days).astype(float)

    dates = pd.bdate_range(end=datetime.today(), periods=n_days)

    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
    }, index=dates)

    return df


@pytest.fixture
def synthetic_price_df():
    """300-day synthetic OHLCV DataFrame — enough for all technical indicators."""
    return make_price_df(n_days=300)


@pytest.fixture
def short_price_df():
    """Only 20 days — insufficient for 1-month returns and some indicators."""
    return make_price_df(n_days=20)


@pytest.fixture
def sample_fundamentals():
    """Typical yfinance info dict with all common fields populated."""
    return {
        "marketCap": 180_000_000_000,
        "trailingPE": 14.5,
        "priceToBook": 2.1,
        "priceToSalesTrailing12Months": 3.2,
        "returnOnEquity": 0.145,
        "profitMargins": 0.22,
        "revenueGrowth": 0.08,
        "earningsGrowth": 0.12,
        "debtToEquity": 45.0,
        "currentRatio": 1.5,
        "dividendYield": 0.045,
        "sector": "Financial Services",
        "industry": "Banks—Regional",
    }


@pytest.fixture
def empty_fundamentals():
    """Minimal yfinance info dict — simulates data-sparse tickers."""
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Mock HTTP Response Helper
# ─────────────────────────────────────────────────────────────────────────────

def make_mock_response(json_data=None, text=None, content=None, status_code=200):
    """Create a mock requests.Response with configurable content."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        mock.raise_for_status.side_effect = HTTPError(f"HTTP {status_code}")
    if json_data is not None:
        mock.json = MagicMock(return_value=json_data)
    if text is not None:
        mock.text = text
    if content is not None:
        mock.content = content
    return mock
