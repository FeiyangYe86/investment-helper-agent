"""
ASX Short Interest (Short Selling) Data Tool

Fetches weekly ASIC short position reports for ASX-listed stocks.
Rising short interest against bullish technical signals = warning flag.
Falling short interest during uptrend = confirms the move.

Data source: ASIC publishes weekly short position reports.
Shortman.com.au (shortman.com.au/stock?q=TICKER) aggregates this data
with historical trends and technical integration.

Usage:
    from tools.ticker.short_interest_tools import get_short_interest
    result = get_short_interest(["CBA.AX", "BHP.AX"])

Requirements:
    - requests: for HTTP calls
    - beautifulsoup4: for HTML parsing
    - Standard library: re, html.parser, datetime, concurrent.futures
"""

from datetime import datetime
import sys
import traceback
import re
import time
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langchain.tools import tool
from typing import Annotated
from typing_extensions import TypedDict

import requests
from bs4 import BeautifulSoup


class ToolResult(TypedDict):
    """Standard tool result format."""
    data: Any
    errors_fatal: list[str]
    errors_non_fatal: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Short Interest Data Model
# ─────────────────────────────────────────────────────────────────────────────

class ShortInterestData(TypedDict, total=False):
    """Per-ticker short interest metrics."""
    ticker: str
    short_pct: float | None
    short_pct_prev: float | None
    short_change_pct: float | None
    short_trend: str  # "increasing" | "decreasing" | "flat" | "unknown"
    short_signal: str  # "bullish" | "bearish" | "warning" | "neutral" | "unknown"
    data_date: str | None
    fetch_error: str | None


# ─────────────────────────────────────────────────────────────────────────────
# Signal Logic
# ─────────────────────────────────────────────────────────────────────────────

def compute_short_signal(
    short_pct: float | None,
    short_pct_prev: float | None,
    short_change: float | None,
    short_trend: str,
) -> str:
    """
    Determine bullish/bearish/warning/neutral signal from short interest metrics.

    Signal logic:
    - "warning"  : short_pct > 15% (extreme short position)
    - "bearish"  : short_pct > 10% OR (trend == "increasing" AND change > 1.5%)
    - "bullish"  : short_pct < 3% OR (trend == "decreasing" AND short_pct < 8%)
    - "neutral"  : otherwise
    - "unknown"  : any required metrics are None
    """
    if short_pct is None:
        return "unknown"

    # Extreme short position: major warning
    if short_pct > 15.0:
        return "warning"

    # Bearish signals
    if short_pct > 10.0:
        return "bearish"
    if short_trend == "increasing" and short_change is not None and short_change > 1.5:
        return "bearish"

    # Bullish signals
    if short_pct < 3.0:
        return "bullish"
    if short_trend == "decreasing" and short_pct < 8.0:
        return "bullish"

    return "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# Shortman.com.au Scraper
# ─────────────────────────────────────────────────────────────────────────────

def fetch_shortman_data(ticker: str) -> ShortInterestData:
    """
    Fetch short interest data from shortman.com.au for a single ticker.

    Args:
        ticker: ASX ticker (without .AX suffix, e.g. "CBA")

    Returns:
        ShortInterestData dict with parsed metrics or error info.
    """
    # Strip .AX suffix if present
    clean_ticker = ticker.replace(".AX", "").upper()

    result: ShortInterestData = {
        "ticker": ticker,
        "short_pct": None,
        "short_pct_prev": None,
        "short_change_pct": None,
        "short_trend": "unknown",
        "short_signal": "unknown",
        "data_date": None,
        "fetch_error": None,
    }

    try:
        url = f"https://www.shortman.com.au/stock?q={clean_ticker}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        }

        # 0.5s delay between requests (rate limiting)
        time.sleep(0.5)

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Try to find the main short interest percentage from page content
        # Shortman typically displays this in a table or summary section
        short_pct = _parse_shortman_percentage(soup)
        short_pct_prev = _parse_shortman_previous(soup)
        data_date = _parse_shortman_date(soup)

        if short_pct is None:
            result["fetch_error"] = (
                f"Could not parse short % from shortman.com.au for {clean_ticker}. "
                "Page structure may have changed."
            )
            return result

        result["short_pct"] = short_pct
        result["short_pct_prev"] = short_pct_prev
        result["data_date"] = data_date

        # Calculate week-over-week change
        if short_pct_prev is not None:
            result["short_change_pct"] = short_pct - short_pct_prev
            if short_pct > short_pct_prev + 0.1:
                result["short_trend"] = "increasing"
            elif short_pct < short_pct_prev - 0.1:
                result["short_trend"] = "decreasing"
            else:
                result["short_trend"] = "flat"
        else:
            result["short_trend"] = "unknown"

        # Compute signal
        result["short_signal"] = compute_short_signal(
            short_pct,
            short_pct_prev,
            result.get("short_change_pct"),
            result["short_trend"],
        )

        return result

    except requests.RequestException as e:
        result["fetch_error"] = f"HTTP request failed for {ticker}: {str(e)}"
        return result
    except Exception as e:
        result["fetch_error"] = f"Unexpected error parsing {ticker}: {str(e)}"
        return result


def _parse_shortman_percentage(soup: BeautifulSoup) -> float | None:
    """
    Extract current short interest percentage from Shortman HTML.

    Looks for patterns like "3.5%" or "Short: 3.5%" in the page content.
    """
    # Try to find text matching percentage patterns
    text = soup.get_text()

    # Look for "Short:" label followed by percentage
    match = re.search(r"Short\s*:?\s*([\d.]+)%", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Try tables — look for rows with "Short" and percentage values
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                cell_text = " ".join([cell.get_text(strip=True) for cell in cells])
                if "short" in cell_text.lower():
                    # Extract percentage from this row
                    match = re.search(r"([\d.]+)%", cell_text)
                    if match:
                        try:
                            return float(match.group(1))
                        except ValueError:
                            pass

    # Fallback: look for any percentage in specific contexts
    # e.g., in a span or div with "short" in class/id
    for elem in soup.find_all(["span", "div", "p"]):
        if elem.get("class"):
            elem_class = " ".join(elem.get("class", []))
            if "short" in elem_class.lower():
                text = elem.get_text()
                match = re.search(r"([\d.]+)%", text)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass

    return None


def _parse_shortman_previous(soup: BeautifulSoup) -> float | None:
    """
    Extract previous week's short interest percentage from Shortman HTML.

    Looks for patterns like "Previous: 3.2%" or "Last Week: 3.2%".
    """
    text = soup.get_text()

    # Look for "Previous:" or "Last Week:" labels
    for pattern in [r"Previous\s*:?\s*([\d.]+)%", r"Last\s+Week\s*:?\s*([\d.]+)%"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

    return None


def _parse_shortman_date(soup: BeautifulSoup) -> str | None:
    """
    Extract data publication date from Shortman HTML.

    Looks for date patterns like "2024-03-22" or "22 Mar 2024".
    """
    text = soup.get_text()

    # Try ISO format first
    match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)

    # Try DD Mon YYYY format
    match = re.search(r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})", text, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# ASIC Fallback (if Shortman fails)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_asic_data(ticker: str) -> ShortInterestData:
    """
    Fallback to fetch short interest data from ASIC directly.

    Note: ASIC publishes weekly CSV reports. This is a placeholder
    for more robust parsing if Shortman becomes unavailable.
    """
    clean_ticker = ticker.replace(".AX", "").upper()

    result: ShortInterestData = {
        "ticker": ticker,
        "short_pct": None,
        "short_pct_prev": None,
        "short_change_pct": None,
        "short_trend": "unknown",
        "short_signal": "unknown",
        "data_date": None,
        "fetch_error": None,
    }

    try:
        # ASIC short position reports table
        url = "https://www.asic.gov.au/regulatory-resources/markets/short-selling/short-position-reports-table/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        time.sleep(0.5)
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Look for CSV download links or data tables on the ASIC page
        # This is a simplified attempt; ASIC's structure varies
        text = soup.get_text()

        # If we can find our ticker mentioned, try to extract related data
        if clean_ticker in text:
            # Search for percentage patterns near the ticker name
            parts = text.split(clean_ticker)
            if len(parts) > 1:
                context = parts[1][:200]  # 200 chars after ticker
                match = re.search(r"([\d.]+)%", context)
                if match:
                    try:
                        result["short_pct"] = float(match.group(1))
                        result["short_signal"] = compute_short_signal(
                            result["short_pct"], None, None, "unknown"
                        )
                        return result
                    except ValueError:
                        pass

        result["fetch_error"] = f"ASIC data not found for {clean_ticker}"
        return result

    except Exception as e:
        result["fetch_error"] = f"ASIC fallback failed for {ticker}: {str(e)}"
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Main Tool Implementation
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_short_interest(tickers: list[str], max_workers: int = 5) -> ToolResult:
    """
    Core implementation for fetching short interest data.

    Args:
        tickers: List of ASX tickers (e.g., ["CBA.AX", "BHP.AX"])
        max_workers: Max concurrent workers (conservative to avoid rate limits)

    Returns:
        ToolResult with per-ticker short interest metrics
    """
    errors_fatal: list[str] = []
    errors_non_fatal: list[str] = []
    all_data: list[ShortInterestData] = []

    if not tickers:
        errors_fatal.append("No tickers provided")
        return ToolResult(data=[], errors_fatal=errors_fatal, errors_non_fatal=[])

    print(f"\n  Fetching short interest data for {len(tickers)} ticker(s)...")

    # Use ThreadPoolExecutor for concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(fetch_shortman_data, ticker): ticker
            for ticker in tickers
        }

        # Collect results as they complete
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
                if result["fetch_error"]:
                    errors_non_fatal.append(result["fetch_error"])
                    # Try ASIC fallback
                    fallback = fetch_asic_data(ticker)
                    if not fallback["fetch_error"]:
                        all_data.append(fallback)
                    else:
                        all_data.append(result)
                else:
                    all_data.append(result)
            except Exception as e:
                error_msg = f"Exception fetching {ticker}: {str(e)}"
                errors_non_fatal.append(error_msg)
                # Add a minimal result
                all_data.append({
                    "ticker": ticker,
                    "short_pct": None,
                    "short_pct_prev": None,
                    "short_change_pct": None,
                    "short_trend": "unknown",
                    "short_signal": "unknown",
                    "data_date": None,
                    "fetch_error": error_msg,
                })

    return ToolResult(
        data=all_data,
        errors_fatal=errors_fatal,
        errors_non_fatal=errors_non_fatal,
    )


@tool
def get_short_interest(
    tickers: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> ToolMessage:
    """
    Fetch ASX short interest (short selling) data for a list of tickers.

    Use this tool to:
    - Monitor short position sentiment for stocks under analysis
    - Validate bullish moves: falling short interest = shorts covering (bullish confirmation)
    - Identify warning flags: rising short interest against bullish technicals = caution
    - Detect extreme short positions (>15%) that may signal distress or opportunity

    The tool returns current short %, previous week's %, week-over-week change, and
    a signal classification (bullish/bearish/warning/neutral) based on ASIC data
    aggregated by shortman.com.au.

    IMPORTANT — ticker format and exchange:
        This tool is for ASX-listed stocks only. Do not use it for US or other exchange tickers.
        Pass tickers exactly as you have them — both "WDS.AX" and "WDS" are accepted.
        The tool strips the '.AX' suffix internally. Do NOT manually remove it before calling.

    Args:
        tickers: List of ASX stock tickers to analyze. Both formats accepted:
                 e.g. ["CBA.AX", "BHP.AX"] or ["CBA", "BHP"].
                 Tickers with no available short data are returned with signal="unknown".

    Returns:
        A ToolMessage where:
          - content: human-readable summary of short interest data availability
          - artifact: a ToolResult dict with:
              - data: list of ShortInterestData dicts, one per ticker. Each includes:
                  ticker, short_pct, short_pct_prev, short_change_pct, short_trend,
                  short_signal, data_date, fetch_error
              - errors_fatal: set if ALL tickers failed (data unavailable). Non-fatal.
              - errors_non_fatal: set for individual ticker failures. Analysis can
                  continue on available data.

    Signal meanings:
    - "bullish": short_pct < 3% or (decreasing trend and < 8%)
    - "bearish": short_pct > 10% or (increasing trend and change > 1.5%)
    - "warning": short_pct > 15% (extreme short position)
    - "neutral": short_pct in moderate range with stable trend
    - "unknown": data unavailable for this ticker

    Note: Short interest data updates weekly (ASIC reports). Current data may be
    up to 7 days old.
    """
    print(f'===> get_short_interest tool_call_id: {tool_call_id}')

    result = _fetch_short_interest(tickers)

    # Count valid results
    valid_count = sum(1 for d in result["data"] if d.get("short_pct") is not None)
    error_count = len([d for d in result["data"] if d.get("fetch_error")])

    summary = (
        f"Short interest data for {valid_count}/{len(tickers)} tickers. "
        f"Signal breakdown: {_count_signals(result['data'])}"
    )

    return ToolMessage(
        content=summary,
        artifact=result,
        tool_call_id=tool_call_id,
    )


def _count_signals(data: list[ShortInterestData]) -> str:
    """Count signal occurrences for display."""
    signals = {}
    for item in data:
        sig = item.get("short_signal", "unknown")
        signals[sig] = signals.get(sig, 0) + 1
    return " | ".join(f"{k}={v}" for k, v in sorted(signals.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Test Block
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: fetch short interest for major ASX stocks
    test_tickers = ["CBA.AX", "BHP.AX", "NAB.AX", "AMP.AX"]

    print("=" * 70)
    print("  SHORT INTEREST TOOL TEST")
    print("=" * 70)

    result = _fetch_short_interest(test_tickers)

    print(f"\nFetched {len(result['data'])} results")
    print(f"Fatal errors: {result['errors_fatal']}")
    print(f"Non-fatal errors: {len(result['errors_non_fatal'])}")

    print("\n" + "=" * 70)
    print("  PER-TICKER RESULTS")
    print("=" * 70 + "\n")

    for item in result["data"]:
        print(f"\nTicker: {item['ticker']}")
        print(f"  Short %: {item['short_pct']}%")
        print(f"  Prev %: {item['short_pct_prev']}%")
        print(f"  Change: {item['short_change_pct']:+.2f}%" if item['short_change_pct'] else "  Change: N/A")
        print(f"  Trend: {item['short_trend']}")
        print(f"  Signal: {item['short_signal']}")
        print(f"  Date: {item['data_date']}")
        if item['fetch_error']:
            print(f"  Error: {item['fetch_error']}")
