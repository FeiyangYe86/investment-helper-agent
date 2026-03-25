"""
ASX Company Announcements Fetcher — Material news & sentiment classification.

Requirements: requests (standard library: concurrent.futures, json, time, datetime)

Key design decisions:
  - Fetch announcements from ASX public API (no authentication required)
  - Classify by sentiment/impact using keyword matching on titles
  - Concurrent fetch via ThreadPoolExecutor with small sleep for rate limiting
  - Handle per-ticker failures gracefully (non-fatal unless all fail)
  - Return structured signal ("positive", "negative", "neutral", "warning")

Usage:
    from tools.ticker.announcements_tools import get_asx_announcements
    result = get_asx_announcements(["CBA.AX", "BHP.AX"])
"""

import json
import time
import traceback
import sys
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langchain.tools import tool
from typing import Annotated
from typing_extensions import TypedDict


class ToolResult(TypedDict):
    """Standard result envelope for tool responses."""
    data: Any
    errors_fatal: list[str]
    errors_non_fatal: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Constants & Configuration
# ─────────────────────────────────────────────────────────────────────────────

RED_FLAG_KEYWORDS = [
    "capital raise", "placement", "rights issue", "ASIC", "class action",
    "investigation", "cease", "winding up", "administration",
    "CEO", "managing director", "CFO", "resign", "terminate"
]

POSITIVE_KEYWORDS = [
    "contract win", "agreement", "partnership", "acquisition",
    "dividend", "buyback", "record", "milestone", "upgrade"
]

# The old /asx/1/company endpoint returns 404 as of 2024.
# The ASX website now uses the MarkitDigital API internally.
# Format: https://asx.api.markitdigital.com/asx-research/1.0/companies/{TICKER}/announcements
ASX_API_BASE = "https://asx.api.markitdigital.com/asx-research/1.0/companies"
REQUEST_TIMEOUT = 10
RATE_LIMIT_SLEEP = 0.3
MAX_WORKERS = 10
DAYS_LOOKBACK = 30

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ─────────────────────────────────────────────────────────────────────────────
# Announcement Fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_announcements_for_ticker(ticker: str) -> dict:
    """
    Fetch announcements for a single ticker from ASX API.
    Returns a dict with parsed announcements or error info.

    Args:
        ticker: ASX ticker code (with or without .AX suffix; will be stripped)

    Returns:
        dict with keys:
            - ticker: normalized ticker (without .AX)
            - success: bool indicating if fetch succeeded
            - announcements: list of announcement dicts (if success)
            - error: error message string (if not success)
    """
    # Normalize ticker: strip .AX if present; MarkitDigital API uses lowercase
    clean_ticker = ticker.replace(".AX", "").replace(".ax", "").upper()
    api_ticker = clean_ticker.lower()  # MarkitDigital expects lowercase

    headers = {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "application/json",
        "Referer": "https://www.asx.com.au/",
    }
    cutoff_date = (datetime.now() - timedelta(days=DAYS_LOOKBACK)).date()

    all_announcements = []
    fetch_errors: list[str] = []
    seen_keys: set[str] = set()  # deduplicate by documentKey across both calls

    # MarkitDigital API schema (confirmed):
    #   GET /asx-research/1.0/companies/{ticker}/announcements?pageSize=20&market_sensitive=true
    #   Response: { "data": { "items": [...], "symbol": "WDS", "displayName": "...", ... } }
    #   Each item: { "headline", "date", "isPriceSensitive", "announcementType", "documentKey", "fileSize", "url" }
    #
    # We call twice (market_sensitive=true then false) to get both price-sensitive and
    # general announcements, deduplicating by documentKey.
    for market_sensitive in [True, False]:
        variant_label = f"market_sensitive={market_sensitive}"
        try:
            url = (
                f"{ASX_API_BASE}/{api_ticker}/announcements"
                f"?pageSize=20&market_sensitive={str(market_sensitive).lower()}"
            )

            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

            if response.status_code != 200:
                fetch_errors.append(
                    f"[{variant_label}] HTTP {response.status_code} from {url}"
                )
                continue

            data = response.json()

            if not isinstance(data, dict) or "data" not in data:
                fetch_errors.append(
                    f"[{variant_label}] Missing top-level 'data' key. "
                    f"Got: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}"
                )
                continue

            inner = data["data"]
            if not isinstance(inner, dict) or "items" not in inner:
                fetch_errors.append(
                    f"[{variant_label}] Expected data.data to be a dict with 'items'. "
                    f"Got type={type(inner).__name__}, "
                    f"keys={list(inner.keys()) if isinstance(inner, dict) else 'n/a'}"
                )
                continue

            for ann in inner["items"]:
                if not isinstance(ann, dict):
                    fetch_errors.append(
                        f"[{variant_label}] items[] entry is {type(ann).__name__}, not dict: {ann!r}"
                    )
                    break

                doc_key = ann.get("documentKey", "")
                if doc_key in seen_keys:
                    continue
                seen_keys.add(doc_key)

                ann_date_str = ann.get("date", "")
                title = ann.get("headline", "")
                is_price_sensitive = ann.get("isPriceSensitive", False)

                if ann_date_str:
                    try:
                        ann_date = datetime.fromisoformat(
                            ann_date_str.replace("Z", "+00:00")
                        ).date()
                        if ann_date >= cutoff_date:
                            all_announcements.append({
                                "date": ann_date_str,
                                "title": title,
                                "market_sensitive": is_price_sensitive,
                                "announcement_type": ann.get("announcementType", ""),
                            })
                    except (ValueError, AttributeError):
                        pass

            time.sleep(RATE_LIMIT_SLEEP)

        except requests.exceptions.Timeout:
            fetch_errors.append(f"[{variant_label}] Request timed out after {REQUEST_TIMEOUT}s")
            continue
        except requests.exceptions.HTTPError as e:
            fetch_errors.append(f"[{variant_label}] HTTP error: {e}")
            continue
        except json.JSONDecodeError as e:
            fetch_errors.append(f"[{variant_label}] JSON decode error: {e}")
            continue
        except Exception as e:
            fetch_errors.append(f"[{variant_label}] Unexpected error: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stdout)
            continue

    if not all_announcements:
        error_detail = "; ".join(fetch_errors) if fetch_errors else "API returned empty data"
        return {
            "ticker": clean_ticker,
            "success": False,
            "error": f"No announcements found for {clean_ticker}. Details: {error_detail}",
        }

    return {
        "ticker": clean_ticker,
        "success": True,
        "announcements": all_announcements,
    }


def classify_announcements(ticker: str, announcements: list[dict]) -> dict:
    """
    Classify announcements by type and sentiment.

    Args:
        ticker: ASX ticker code (normalized, without .AX)
        announcements: list of announcement dicts with 'title', 'date', 'market_sensitive'

    Returns:
        dict with classification results:
            - ticker
            - announcements_count
            - market_sensitive_count
            - red_flags: list of titles matching red-flag keywords
            - positive_signals: list of titles matching positive keywords
            - latest_announcement: dict with date, title, market_sensitive
            - announcement_signal: "positive" | "negative" | "neutral" | "warning"
    """
    red_flags = []
    positive_signals = []
    market_sensitive_count = 0

    # Sort by date descending to find latest
    sorted_anns = sorted(
        announcements,
        key=lambda x: x.get("date", ""),
        reverse=True
    )
    latest = sorted_anns[0] if sorted_anns else None

    # Classify each announcement
    for ann in announcements:
        title_lower = ann.get("title", "").lower()

        if ann.get("market_sensitive"):
            market_sensitive_count += 1

        # Check for red flags
        for keyword in RED_FLAG_KEYWORDS:
            if keyword.lower() in title_lower:
                red_flags.append(ann.get("title", ""))
                break

        # Check for positive signals
        for keyword in POSITIVE_KEYWORDS:
            if keyword.lower() in title_lower:
                positive_signals.append(ann.get("title", ""))
                break

    # Determine sentiment signal
    if red_flags:
        if len(red_flags) > 1:
            signal = "negative"
        else:
            signal = "warning"
    elif positive_signals:
        signal = "positive"
    else:
        signal = "neutral"

    return {
        "ticker": ticker,
        "announcements_count": len(announcements),
        "market_sensitive_count": market_sensitive_count,
        "red_flags": red_flags,
        "positive_signals": positive_signals,
        "latest_announcement": {
            "date": latest.get("date"),
            "title": latest.get("title"),
            "market_sensitive": latest.get("market_sensitive"),
        } if latest else None,
        "announcement_signal": signal,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Tool Implementation
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_announcements_impl(tickers: list[str]) -> ToolResult:
    """
    Core implementation for fetching and classifying announcements.
    Returns a ToolResult with per-ticker announcement data.

    Args:
        tickers: list of ASX ticker codes (e.g., ["CBA.AX", "BHP.AX"])

    Returns:
        ToolResult with:
            - data: list of classification dicts, one per ticker
            - errors_fatal: set if ALL tickers fail
            - errors_non_fatal: set if SOME tickers fail
    """
    errors_fatal: list[str] = []
    errors_non_fatal: list[str] = []
    results = []

    # Normalize tickers
    clean_tickers = [t.replace(".AX", "").upper() for t in tickers]

    # Fetch announcements concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(fetch_announcements_for_ticker, t): t
            for t in clean_tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                fetch_result = future.result(timeout=REQUEST_TIMEOUT + 5)

                if not fetch_result.get("success"):
                    errors_non_fatal.append(
                        f"Failed to fetch announcements for '{ticker}': {fetch_result.get('error', 'Unknown error')}"
                    )
                    continue

                # Classify the fetched announcements
                classification = classify_announcements(
                    ticker,
                    fetch_result.get("announcements", [])
                )
                results.append(classification)

            except Exception as e:
                error_msg = f"Exception processing '{ticker}': {str(e)}"
                errors_non_fatal.append(error_msg)
                traceback.print_exc(file=sys.stdout)

    # If all tickers failed, mark as fatal
    if not results and clean_tickers:
        errors_fatal = errors_non_fatal
        errors_non_fatal = []

    return ToolResult(
        data=results,
        errors_fatal=errors_fatal,
        errors_non_fatal=errors_non_fatal,
    )


@tool
def get_asx_announcements(
    tickers: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> ToolMessage:
    """Fetch and classify recent ASX company announcements for a list of tickers.

    This tool is for ASX-listed stocks only. Do not use it for US or other exchange tickers.

    Use this tool when you need to identify material announcements (capital raises,
    acquisitions, resignations, investigations, dividends, etc.) that may impact
    a stock's short-term direction. This surfaces news that price data alone won't capture.

    The tool fetches the last 30 days of announcements from the ASX public API and
    classifies each by sentiment impact using keyword matching:
    - Red flags: capital raises, placements, ASIC issues, investigations, executive changes
    - Positive signals: contract wins, partnerships, acquisitions, dividends, buybacks
    - Returns a risk signal: "positive" | "negative" | "neutral" | "warning"

    IMPORTANT — ticker format:
        Pass tickers exactly as you have them — both "WDS.AX" and "WDS" are accepted.
        The tool strips the '.AX' suffix internally before querying the ASX API.
        Do NOT manually remove '.AX' before calling this tool.

    Args:
        tickers: List of ASX ticker symbols. Both formats are accepted:
                 e.g. ["CBA.AX", "BHP.AX"] or ["CBA", "BHP"] — the tool handles either.

    Returns:
        A ToolMessage where:
          - content: human-readable summary of announcement data fetched.
          - artifact: a ToolResult dict with:
              - data: list of classification dicts, one per ticker. Each includes:
                  - ticker: ASX code (normalized, no .AX)
                  - announcements_count: total announcements in last 30 days
                  - market_sensitive_count: number flagged as market-sensitive
                  - red_flags: list of announcement titles matching risk keywords
                  - positive_signals: list of announcement titles matching positive keywords
                  - latest_announcement: dict with date, title, and market_sensitive flag
                  - announcement_signal: "positive" | "negative" | "neutral" | "warning"
              - errors_fatal: set when ALL tickers fail to fetch. Indicates a network
                  or API issue that requires investigation.
              - errors_non_fatal: set when SOME (but not all) tickers fail. Analysis
                  can continue on valid tickers.
    """
    result = _fetch_announcements_impl(tickers)

    summary = f"Fetched announcements for {len(result['data'])} tickers."
    if result['errors_non_fatal']:
        summary += f" ({len(result['errors_non_fatal'])} had partial failures.)"

    return ToolMessage(
        content=summary,
        artifact=result,
        tool_call_id=tool_call_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test Block
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  ASX Announcements Fetcher — Test Run")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    test_tickers = ["CBA.AX", "BHP.AX"]

    # Use a dummy tool_call_id for testing
    test_tool_call_id = "test-call-123"

    result = get_asx_announcements(test_tickers, tool_call_id=test_tool_call_id)

    print(f"Tool Response Content:\n{result.content}\n")

    if result.artifact and result.artifact.get("data"):
        print("Results by ticker:")
        for ticker_data in result.artifact["data"]:
            print(f"\n  {ticker_data['ticker']}:")
            print(f"    Total announcements (30d): {ticker_data['announcements_count']}")
            print(f"    Market-sensitive: {ticker_data['market_sensitive_count']}")
            print(f"    Signal: {ticker_data['announcement_signal']}")

            if ticker_data.get('latest_announcement'):
                latest = ticker_data['latest_announcement']
                print(f"    Latest: {latest['date']} — {latest['title']}")

            if ticker_data.get('red_flags'):
                print(f"    Red flags ({len(ticker_data['red_flags'])}):")
                for flag in ticker_data['red_flags'][:3]:  # Show first 3
                    print(f"      - {flag}")

            if ticker_data.get('positive_signals'):
                print(f"    Positive signals ({len(ticker_data['positive_signals'])}):")
                for sig in ticker_data['positive_signals'][:3]:  # Show first 3
                    print(f"      - {sig}")

    if result.artifact and result.artifact.get("errors_fatal"):
        print(f"\nFatal errors: {result.artifact['errors_fatal']}")

    if result.artifact and result.artifact.get("errors_non_fatal"):
        print(f"\nNon-fatal errors: {result.artifact['errors_non_fatal']}")

    print(f"\n{'='*60}\n")
