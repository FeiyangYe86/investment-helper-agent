"""
Macroeconomic context tool for ASX investment analysis.

Data sources (in priority order):
  1. yfinance live prices — zero lag, exchange-traded futures:
       AUD/USD  → AUDUSD=X
       WTI Oil  → CL=F
       Brent    → BZ=F
       Gold     → GC=F
  2. FRED API — authoritative but lagged (days to weeks):
       RBA Cash Rate  → IRSTCB01AUM156N  (monthly, no yfinance equivalent)
       Iron Ore       → PIORECRUSDM      (monthly, no yfinance equivalent)

  The old approach used FRED for oil/gold/AUD too, which caused multi-week lag.
  During fast-moving events (e.g. Iran-US conflict, Hormuz closure) that lag
  would report "oil neutral" while Brent was actually up 30%+ in real time.

Adaptive lookback:
  During high-volatility regimes (5-day oil move > 2× its 30-day average daily
  move), the trend window shrinks from 20 days to 5 days so shocks register
  immediately rather than being averaged away.

Geopolitical risk layer:
  NewsAPI is queried with macro supply-shock keywords (war, sanctions, Hormuz,
  OPEC, rate decision…). The resulting headline scores are rolled up into a
  geopolitical_risk_level: "elevated" | "moderate" | "low".
  Elevated geopolitical risk is surfaced explicitly in macro_notes and pushes
  the signal toward "unfavorable" unless commodities are clearly rising (in
  which case it stays elevated but with a nuanced note).

Usage:
    from tools.ticker.macro_tools import get_macro_context
    result = get_macro_context(tool_call_id="test_123")
"""

import os
import sys
import traceback
from datetime import datetime, timedelta
from typing import Any, Annotated

import requests
import yfinance as yf
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langchain.tools import tool
from typing_extensions import TypedDict


# ─────────────────────────────────────────────────────────────────────────────
# TypedDicts
# ─────────────────────────────────────────────────────────────────────────────

class MacroContext(TypedDict):
    """Structured macroeconomic context for ASX investing."""
    audusd_current: float | None
    audusd_1m_change_pct: float | None
    audusd_3m_change_pct: float | None
    audusd_trend: str                    # "strengthening" | "weakening" | "flat"
    rba_cash_rate: float | None
    rba_rate_direction: str              # "rising" | "falling" | "flat"
    gold_current: float | None
    gold_1m_change_pct: float | None
    gold_trend: str                      # "rising" | "falling" | "flat"
    oil_wti_current: float | None
    oil_brent_current: float | None
    oil_1m_change_pct: float | None
    oil_trend: str                       # "rising" | "falling" | "flat"
    oil_5d_change_pct: float | None      # short-term move for volatility detection
    iron_ore_trend: str                  # "rising" | "falling" | "flat"
    geopolitical_risk_level: str         # "elevated" | "moderate" | "low"
    geopolitical_risk_notes: list[str]   # headline summaries driving the risk level
    macro_signal: str                    # "favorable" | "unfavorable" | "neutral"
    macro_notes: list[str]


class ToolResult(TypedDict):
    """Standard tool result container."""
    data: Any
    errors_fatal: list[str]
    errors_non_fatal: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"
SERIES_RBA_RATE  = "IRSTCB01AUM156N"    # RBA Cash Rate Target (monthly) — FRED only
SERIES_IRON_ORE  = "PIORECRUSDM"        # Iron Ore price index (monthly) — FRED only

# yfinance tickers for live prices (zero publication lag)
YF_AUDUSD = "AUDUSD=X"
YF_WTI    = "CL=F"
YF_BRENT  = "BZ=F"
YF_GOLD   = "GC=F"

# Geopolitical macro queries — NewsAPI searches for supply-shock events
GEO_QUERIES = [
    "oil supply disruption war sanctions",
    "Strait Hormuz closure tanker",
    "OPEC production cut oil",
    "Iran oil sanctions conflict",
    "Federal Reserve rate decision",
    "RBA rate decision Australia",
    "China economic slowdown demand",
    "global recession risk",
]
GEO_RISK_KEYWORDS = [
    "war", "conflict", "sanctions", "blockade", "closure", "attack",
    "supply disruption", "embargo", "escalation", "invasion",
    "rate hike", "recession", "default", "crisis", "shutdown",
]
GEO_RELIEF_KEYWORDS = [
    "ceasefire", "peace deal", "reopening", "rate cut", "stimulus",
    "trade deal", "production increase", "de-escalation", "truce",
]

NEWSAPI_BASE = "https://newsapi.org/v2/everything"
REQUEST_TIMEOUT = 10


# ─────────────────────────────────────────────────────────────────────────────
# Live price helpers (yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_yf_history(symbol: str, days: int = 40) -> list[float]:
    """
    Fetch closing prices for a yfinance symbol.

    Returns a list of floats, oldest first, covering up to `days` calendar days.
    Returns [] on any failure.
    """
    try:
        df = yf.Ticker(symbol).history(period=f"{days}d")
        if df.empty:
            return []
        return [float(v) for v in df["Close"].dropna().tolist()]
    except Exception:
        return []


def _pct_change(prices: list[float], lookback: int) -> float | None:
    """Return % change from prices[-lookback] to prices[-1]. None if insufficient data."""
    if len(prices) < lookback + 1:
        return None
    base = prices[-lookback - 1]
    if base == 0:
        return None
    return (prices[-1] - base) / abs(base)


def _is_high_volatility(prices: list[float], short_window: int = 5) -> bool:
    """
    Return True if the most recent `short_window` days show abnormally high daily moves.
    Specifically: 5-day realised vol > 2× the 30-day realised vol.
    Used to trigger adaptive (shorter) lookback.
    """
    if len(prices) < 30:
        return False
    daily_returns = [abs((prices[i] - prices[i - 1]) / prices[i - 1])
                     for i in range(1, len(prices))]
    vol_5d  = sum(daily_returns[-short_window:]) / short_window
    vol_30d = sum(daily_returns[-30:]) / 30
    return vol_5d > 2 * vol_30d if vol_30d > 0 else False


def _get_trend(change_pct: float | None, threshold: float = 0.01) -> str:
    if change_pct is None:
        return "flat"
    if abs(change_pct) <= threshold:
        return "flat"
    return "rising" if change_pct > 0 else "falling"


# ─────────────────────────────────────────────────────────────────────────────
# FRED helpers (RBA rate + iron ore only)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_fred_series(series_id: str, api_key: str | None, limit: int = 4) -> list[dict]:
    params = {"series_id": series_id, "file_type": "json", "sort_order": "desc", "limit": limit}
    if api_key:
        params["api_key"] = api_key
    try:
        resp = requests.get(FRED_API_BASE, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("observations", [])
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Geopolitical risk layer (NewsAPI)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_geopolitical_risk(news_api_key: str | None) -> tuple[str, list[str]]:
    """
    Query NewsAPI for macro supply-shock headlines and return
    (risk_level, notes) where risk_level is "elevated" | "moderate" | "low".

    Falls back to "low" (with a note) if NewsAPI key is absent or call fails.
    """
    if not news_api_key:
        return "low", ["NewsAPI key not set — geopolitical risk not assessed"]

    risk_score = 0
    relief_score = 0
    matched_headlines: list[str] = []

    for query in GEO_QUERIES[:4]:   # limit to 4 queries to stay within free-tier rate limits
        try:
            resp = requests.get(
                NEWSAPI_BASE,
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 5,
                    "apiKey": news_api_key,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            articles = resp.json().get("articles", [])
            for art in articles:
                title = (art.get("title") or "").lower()
                if any(kw in title for kw in GEO_RISK_KEYWORDS):
                    risk_score += 1
                    matched_headlines.append(art.get("title", "")[:100])
                if any(kw in title for kw in GEO_RELIEF_KEYWORDS):
                    relief_score += 1
        except Exception:
            continue

    # Deduplicate headlines (same story across queries)
    unique_headlines = list(dict.fromkeys(matched_headlines))[:5]

    net = risk_score - relief_score
    if net >= 4:
        level = "elevated"
    elif net >= 2:
        level = "moderate"
    else:
        level = "low"

    notes = unique_headlines if unique_headlines else ["No significant geopolitical supply-shock headlines found"]
    return level, notes


# ─────────────────────────────────────────────────────────────────────────────
# Signal synthesis
# ─────────────────────────────────────────────────────────────────────────────

def _compute_macro_signal(macro: MacroContext) -> tuple[str, list[str]]:
    """
    Synthesize all indicators into a macro signal.

    Scoring:
      AUD weakening         → +2 favorable   (ASX exporters benefit)
      AUD strengthening     → +2 unfavorable
      RBA falling           → +1 favorable
      RBA rising            → +2 unfavorable
      Commodities ≥2 rising → +1 favorable
      Commodities ≥2 falling→ +1 unfavorable
      Geo risk elevated     → +2 unfavorable  (unless oil is rising, then note instead)
      Geo risk moderate     → +1 unfavorable

    Signal: favorable if favorable≥3, unfavorable if unfavorable≥2, else neutral.
    """
    notes = []
    scores = {"favorable": 0, "unfavorable": 0}

    # AUD
    if macro["audusd_trend"] == "weakening":
        scores["favorable"] += 2
        notes.append(f"AUD weakening ({macro['audusd_1m_change_pct']:.1%} 1M) — favorable for ASX exporters"
                     if macro["audusd_1m_change_pct"] is not None else "AUD weakening — favorable for ASX exporters")
    elif macro["audusd_trend"] == "strengthening":
        scores["unfavorable"] += 2
        notes.append("AUD strengthening — headwind for ASX exporters")
    else:
        notes.append("AUD stable (neutral)")

    # RBA
    if macro["rba_rate_direction"] == "falling":
        scores["favorable"] += 1
        notes.append("RBA rates falling — supportive for equity valuations")
    elif macro["rba_rate_direction"] == "rising":
        scores["unfavorable"] += 2
        notes.append("RBA rates rising — headwind for equity valuations")
    else:
        notes.append("RBA rates flat (neutral)")

    # Commodities
    commodity_trends = [macro["gold_trend"], macro["oil_trend"], macro["iron_ore_trend"]]
    rising  = commodity_trends.count("rising")
    falling = commodity_trends.count("falling")
    if rising >= 2:
        scores["favorable"] += 1
        notes.append(f"Commodities broadly rising ({rising}/3) — bullish for resource stocks")
    elif falling >= 2:
        scores["unfavorable"] += 1
        notes.append(f"Commodities broadly falling ({falling}/3) — bearish for resource stocks")
    else:
        notes.append("Commodities mixed (neutral)")

    # Oil-specific note (high-value signal for energy stocks)
    if macro["oil_1m_change_pct"] is not None:
        oil_chg = macro["oil_1m_change_pct"]
        wti  = f"WTI ${macro['oil_wti_current']:.2f}" if macro.get("oil_wti_current") else ""
        brent = f"Brent ${macro['oil_brent_current']:.2f}" if macro.get("oil_brent_current") else ""
        price_str = " / ".join(filter(None, [wti, brent]))
        notes.append(f"Oil {oil_chg:+.1%} over 1 month ({price_str}) — "
                     + ("tailwind for energy stocks" if oil_chg > 0.05 else
                        "headwind for energy stocks" if oil_chg < -0.05 else "neutral for energy"))

    # Geopolitical risk
    geo = macro["geopolitical_risk_level"]
    if geo == "elevated":
        if macro["oil_trend"] == "rising":
            # Rising oil during conflict = net positive for energy stocks but risk remains
            notes.append(
                "⚠️  GEOPOLITICAL RISK ELEVATED — conflict/sanctions detected in macro news. "
                "Oil rising (beneficial for energy stocks) but binary ceasefire risk present. "
                "Monitor for rapid de-escalation which could reverse oil gains."
            )
            scores["unfavorable"] += 1   # partial, not full +2, because commodity benefit offsets
        else:
            scores["unfavorable"] += 2
            notes.append(
                "⚠️  GEOPOLITICAL RISK ELEVATED — conflict/sanctions detected. "
                "Supply disruption risk without compensating commodity price rise."
            )
    elif geo == "moderate":
        scores["unfavorable"] += 1
        notes.append("⚠️  Geopolitical risk moderate — monitor for escalation")

    # Final signal
    if scores["favorable"] >= 3:
        signal = "favorable"
    elif scores["unfavorable"] >= 2:
        signal = "unfavorable"
    else:
        signal = "neutral"

    return signal, notes


# ─────────────────────────────────────────────────────────────────────────────
# Core implementation
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_macro_context(
    fred_api_key: str | None = None,
    news_api_key: str | None = None,
) -> ToolResult:
    errors_fatal: list[str] = []
    errors_non_fatal: list[str] = []

    macro: MacroContext = {
        "audusd_current": None,
        "audusd_1m_change_pct": None,
        "audusd_3m_change_pct": None,
        "audusd_trend": "flat",
        "rba_cash_rate": None,
        "rba_rate_direction": "flat",
        "gold_current": None,
        "gold_1m_change_pct": None,
        "gold_trend": "flat",
        "oil_wti_current": None,
        "oil_brent_current": None,
        "oil_1m_change_pct": None,
        "oil_trend": "flat",
        "oil_5d_change_pct": None,
        "iron_ore_trend": "flat",
        "geopolitical_risk_level": "low",
        "geopolitical_risk_notes": [],
        "macro_signal": "neutral",
        "macro_notes": [],
    }

    # ── AUD/USD (live via yfinance) ──────────────────────────────────────────
    try:
        aud_prices = _fetch_yf_history(YF_AUDUSD, days=70)
        if aud_prices:
            macro["audusd_current"] = aud_prices[-1]
            macro["audusd_1m_change_pct"] = _pct_change(aud_prices, lookback=21)
            macro["audusd_3m_change_pct"] = _pct_change(aud_prices, lookback=63)
            chg = macro["audusd_1m_change_pct"]
            if chg is not None:
                macro["audusd_trend"] = (
                    "strengthening" if chg > 0.01 else "weakening" if chg < -0.01 else "flat"
                )
        else:
            errors_non_fatal.append("No AUD/USD data from yfinance (AUDUSD=X)")
    except Exception as e:
        errors_non_fatal.append(f"AUD/USD fetch error: {e}")

    # ── RBA Cash Rate (FRED — monthly, no real-time alternative) ────────────
    try:
        rba_obs = _fetch_fred_series(SERIES_RBA_RATE, fred_api_key, limit=3)
        if rba_obs:
            macro["rba_cash_rate"] = float(rba_obs[0]["value"])
            if len(rba_obs) >= 2:
                curr, prev = float(rba_obs[0]["value"]), float(rba_obs[1]["value"])
                macro["rba_rate_direction"] = (
                    "rising" if curr > prev + 0.01 else "falling" if curr < prev - 0.01 else "flat"
                )
        else:
            errors_non_fatal.append("No RBA Cash Rate data from FRED (IRSTCB01AUM156N)")
    except Exception as e:
        errors_non_fatal.append(f"RBA rate fetch error: {e}")

    # ── Gold (live via yfinance) ─────────────────────────────────────────────
    try:
        gold_prices = _fetch_yf_history(YF_GOLD, days=40)
        if gold_prices:
            macro["gold_current"] = gold_prices[-1]
            macro["gold_1m_change_pct"] = _pct_change(gold_prices, lookback=21)
            macro["gold_trend"] = _get_trend(macro["gold_1m_change_pct"])
        else:
            errors_non_fatal.append("No gold price data from yfinance (GC=F)")
    except Exception as e:
        errors_non_fatal.append(f"Gold fetch error: {e}")

    # ── Oil — WTI + Brent (live via yfinance, adaptive lookback) ────────────
    try:
        wti_prices    = _fetch_yf_history(YF_WTI,   days=40)
        brent_prices  = _fetch_yf_history(YF_BRENT, days=40)

        # Use WTI as primary; fall back to Brent for trend if WTI empty
        primary_prices = wti_prices if wti_prices else brent_prices

        if wti_prices:
            macro["oil_wti_current"] = wti_prices[-1]
        if brent_prices:
            macro["oil_brent_current"] = brent_prices[-1]

        if primary_prices:
            # Adaptive lookback: 5d in high-vol regime, 20d normally
            high_vol = _is_high_volatility(primary_prices)
            lookback = 5 if high_vol else 20
            macro["oil_1m_change_pct"]  = _pct_change(primary_prices, lookback=lookback)
            macro["oil_5d_change_pct"]  = _pct_change(primary_prices, lookback=5)
            macro["oil_trend"]          = _get_trend(macro["oil_1m_change_pct"], threshold=0.02)
            if high_vol:
                errors_non_fatal.append(
                    f"Oil in high-volatility regime — using {lookback}-day adaptive lookback "
                    f"(5d move: {macro['oil_5d_change_pct']:.1%})"
                    if macro["oil_5d_change_pct"] is not None else
                    f"Oil in high-volatility regime — using {lookback}-day adaptive lookback"
                )
        else:
            errors_non_fatal.append("No oil price data from yfinance (CL=F / BZ=F)")
    except Exception as e:
        errors_non_fatal.append(f"Oil fetch error: {e}")

    # ── Iron Ore (FRED — monthly, no futures equivalent) ────────────────────
    try:
        iron_obs = _fetch_fred_series(SERIES_IRON_ORE, fred_api_key, limit=4)
        if len(iron_obs) >= 2:
            curr, prev = float(iron_obs[0]["value"]), float(iron_obs[1]["value"])
            chg = (curr - prev) / abs(prev) if prev != 0 else 0
            macro["iron_ore_trend"] = _get_trend(chg)
        else:
            errors_non_fatal.append("No Iron Ore data from FRED (PIORECRUSDM)")
    except Exception as e:
        errors_non_fatal.append(f"Iron ore fetch error: {e}")

    # ── Geopolitical risk layer ──────────────────────────────────────────────
    try:
        geo_level, geo_notes = _fetch_geopolitical_risk(news_api_key)
        macro["geopolitical_risk_level"] = geo_level
        macro["geopolitical_risk_notes"] = geo_notes
    except Exception as e:
        errors_non_fatal.append(f"Geopolitical risk fetch error: {e}")

    # ── Synthesize signal ────────────────────────────────────────────────────
    signal, notes = _compute_macro_signal(macro)
    macro["macro_signal"] = signal
    macro["macro_notes"]  = notes

    # Fatal only if we have zero useful data
    if all(macro[k] is None for k in ["audusd_current", "rba_cash_rate", "gold_current", "oil_wti_current"]):
        errors_fatal.append(
            "Failed to fetch any macroeconomic data. "
            "Check network connectivity and API keys (FRED_API_KEY, NEWS_API_KEY)."
        )

    return ToolResult(data=macro, errors_fatal=errors_fatal, errors_non_fatal=errors_non_fatal)


# ─────────────────────────────────────────────────────────────────────────────
# LangChain tool wrapper
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_macro_context(
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> ToolMessage:
    """Fetch real-time macroeconomic context relevant to ASX investing.

    Data sources:
      - yfinance (live, zero lag): AUD/USD, WTI oil, Brent oil, Gold
      - FRED API (authoritative but lagged): RBA Cash Rate, Iron Ore
      - NewsAPI (geopolitical risk): supply-shock headlines (war, sanctions, OPEC, Hormuz...)

    Indicators returned:
      - AUD/USD rate and trend (weakening = favorable for ASX exporters)
      - RBA Cash Rate and direction (falling = favorable for equity valuations)
      - Gold price trend (safe-haven demand signal)
      - WTI and Brent crude oil prices and trend (energy inflation signal;
        uses adaptive 5-day lookback during high-volatility regimes like
        geopolitical supply shocks)
      - Iron Ore trend (China demand proxy, key for ASX miners)
      - geopolitical_risk_level: "elevated" | "moderate" | "low"
        Elevated = active conflicts, sanctions, or major supply disruptions
        detected in macro news. Always read geopolitical_risk_notes alongside
        the macro_signal — an elevated risk with rising oil affects energy
        stocks very differently from elevated risk with falling oil.

    IMPORTANT — when interpreting results:
      - If geopolitical_risk_level is "elevated", flag this explicitly in your
        analysis. Explain how the risk affects each ticker's sector (energy,
        materials, financials) before making a recommendation.
      - For energy stocks (WDS, STO, VEA): elevated geo risk + rising oil is
        a near-term tailwind but a binary ceasefire risk. State this trade-off.
      - Do not describe oil as "neutral" if oil_5d_change_pct shows a large
        short-term move — report both the 5-day and 1-month figures.

    Args:
        tool_call_id: Injected by LangChain; identifies this tool invocation.

    Returns:
        ToolMessage with:
          content: Human-readable macro summary
          artifact: ToolResult {
            data: MacroContext dict,
            errors_fatal: list (set if ALL data fetches fail),
            errors_non_fatal: list (partial failures, context still usable)
          }
    """
    fred_api_key = os.environ.get("FRED_API_KEY")
    news_api_key = os.environ.get("NEWS_API_KEY")

    result = _fetch_macro_context(fred_api_key, news_api_key)
    macro  = result["data"]

    parts = [f"Macro signal: {macro.get('macro_signal', 'unknown').upper()}"]
    if macro.get("audusd_current"):
        parts.append(f"AUD/USD {macro['audusd_current']:.4f} ({macro['audusd_trend']})")
    if macro.get("rba_cash_rate") is not None:
        parts.append(f"RBA {macro['rba_cash_rate']:.2f}% ({macro['rba_rate_direction']})")
    if macro.get("oil_wti_current"):
        parts.append(f"WTI ${macro['oil_wti_current']:.1f} ({macro['oil_trend']})")
    if macro.get("oil_brent_current"):
        parts.append(f"Brent ${macro['oil_brent_current']:.1f}")
    parts.append(f"Geo risk: {macro.get('geopolitical_risk_level', 'low').upper()}")

    return ToolMessage(
        content=" | ".join(parts),
        artifact=result,
        tool_call_id=tool_call_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Macroeconomic Context Tool — Standalone Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    result = _fetch_macro_context(
        fred_api_key=os.environ.get("FRED_API_KEY"),
        news_api_key=os.environ.get("NEWS_API_KEY"),
    )

    if result["errors_fatal"]:
        print("\n[FATAL]")
        for e in result["errors_fatal"]:
            print(f"  - {e}")
    else:
        m = result["data"]
        print(f"\n  AUD/USD:      {m['audusd_current']:.4f} ({m['audusd_trend']}, 1M {m['audusd_1m_change_pct']:.1%})"
              if m['audusd_current'] and m['audusd_1m_change_pct'] is not None else
              f"\n  AUD/USD:      {m['audusd_current'] or 'N/A'}")
        print(f"  RBA Rate:     {m['rba_cash_rate']:.2f}% ({m['rba_rate_direction']})"
              if m['rba_cash_rate'] is not None else "  RBA Rate:     N/A")
        print(f"  Gold:         ${m['gold_current']:.0f} ({m['gold_trend']}, 1M {m['gold_1m_change_pct']:.1%})"
              if m['gold_current'] and m['gold_1m_change_pct'] is not None else
              f"  Gold:         {m['gold_current'] or 'N/A'}")
        print(f"  WTI Oil:      ${m['oil_wti_current']:.2f}" if m['oil_wti_current'] else "  WTI Oil:      N/A")
        print(f"  Brent:        ${m['oil_brent_current']:.2f}" if m['oil_brent_current'] else "  Brent:        N/A")
        print(f"  Oil trend:    {m['oil_trend']}  (1M: {m['oil_1m_change_pct']:.1%}, 5D: {m['oil_5d_change_pct']:.1%})"
              if m['oil_1m_change_pct'] is not None and m['oil_5d_change_pct'] is not None else
              f"  Oil trend:    {m['oil_trend']}")
        print(f"  Iron Ore:     {m['iron_ore_trend']}")
        print(f"\n  Geo Risk:     {m['geopolitical_risk_level'].upper()}")
        for note in m['geopolitical_risk_notes']:
            print(f"    - {note}")
        print(f"\n  Signal:       {m['macro_signal'].upper()}")
        for note in m['macro_notes']:
            print(f"    - {note}")

    if result["errors_non_fatal"]:
        print("\n[WARNINGS]")
        for e in result["errors_non_fatal"]:
            print(f"  - {e}")
    print("\n" + "=" * 70)
