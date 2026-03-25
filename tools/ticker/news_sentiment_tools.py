"""
News Sentiment Analysis Tool — fetch recent headlines and analyze sentiment.

Requirements: requests, xml.etree.ElementTree, os, datetime, concurrent.futures

Key design decisions:
  - Dual-source fetching: NewsAPI (with auth) + Yahoo Finance RSS (no auth)
  - Concurrent fetch via ThreadPoolExecutor to speed up ticker processing
  - Simple keyword-based sentiment scorer (no ML libraries)
  - Per-ticker failures handled gracefully (non-fatal unless all fail)
  - Red flag detection for material corporate news
  - Return structured signal ("positive", "negative", "warning", "neutral", "insufficient_data")

Usage:
    from tools.ticker.news_sentiment_tools import get_news_sentiment
    result = get_news_sentiment(["CBA.AX", "BHP.AX"], company_names={"CBA.AX": "Commonwealth Bank"})
"""

import json
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
from urllib.parse import quote

import requests
import yfinance as yf
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

POSITIVE_KEYWORDS = [
    "profit", "growth", "record", "upgrade", "beat", "dividend", "acquire",
    "partnership", "win", "contract", "strong", "surge", "rally", "expand",
    "launch", "approved", "outperform", "raise guidance", "above expectations"
]

NEGATIVE_KEYWORDS = [
    "loss", "decline", "miss", "downgrade", "investigation", "asic",
    "class action", "resign", "layoff", "restructure", "debt", "warning",
    "below expectations", "cuts guidance", "risk", "concern", "weak",
    "disappointing", "breach", "fine", "penalty", "recall"
]

RED_FLAG_KEYWORDS = [
    "asic", "class action", "investigation", "fraud", "misleading",
    "capital raise", "placement", "rights issue", "ceo resign", "cfo resign",
    "managing director resign", "administration", "receivership", "winding up"
]

NEWSAPI_BASE = "https://newsapi.org/v2/everything"
YAHOO_FINANCE_RSS_TEMPLATE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=AU&lang=en-AU"

REQUEST_TIMEOUT = 10
MAX_WORKERS = 10
RATE_LIMIT_SLEEP = 0.2

BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


# ─────────────────────────────────────────────────────────────────────────────
# Company name resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_company_name(ticker: str) -> Optional[str]:
    """
    Look up the official company long name from yfinance.

    Uses the .AX suffix variant for ASX tickers so yfinance returns the correct
    listed entity (e.g. SDR.AX → "SiteMinder Limited", not "Santos Limited").
    Returns None if the lookup fails or returns no useful name.
    """
    # Ensure ASX tickers are queried with the .AX suffix
    yf_ticker = ticker if ticker.endswith(".AX") else f"{ticker}.AX"
    try:
        info = yf.Ticker(yf_ticker).info
        name = info.get("longName") or info.get("shortName")
        # Reject obviously wrong results (empty, or just the ticker repeated)
        if name and name.strip() and name.strip().upper() != ticker.upper():
            return name.strip()
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_headline(title: str) -> tuple[float, list[str]]:
    """
    Score a headline using keyword-based sentiment analysis.

    Args:
        title: headline text to analyze

    Returns:
        tuple of (sentiment_score, red_flags_found)
        - sentiment_score: float in range -1.0 to 1.0
        - red_flags_found: list of red flag keywords detected
    """
    title_lower = title.lower()
    red_flags_found = []

    # Check for red flags first
    for keyword in RED_FLAG_KEYWORDS:
        if keyword.lower() in title_lower:
            red_flags_found.append(keyword)

    # Count positive and negative keyword hits
    positive_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw.lower() in title_lower)
    negative_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw.lower() in title_lower)

    total_hits = max(1, positive_hits + negative_hits)
    sentiment_score = (positive_hits - negative_hits) / total_hits

    return sentiment_score, red_flags_found


def get_sentiment_label(score: float) -> str:
    """Convert sentiment score to label."""
    if score > 0.1:
        return "positive"
    elif score < -0.1:
        return "negative"
    else:
        return "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# NewsAPI Source
# ─────────────────────────────────────────────────────────────────────────────

def fetch_newsapi_articles(
    ticker: str,
    company_name: Optional[str] = None,
    days_lookback: int = 30
) -> list[dict]:
    """
    Fetch articles from NewsAPI for a ticker or company name.

    Args:
        ticker: stock ticker (e.g., "CBA" or "CBA.AX")
        company_name: optional company name for better search (e.g., "Commonwealth Bank")
        days_lookback: how many days of history to fetch

    Returns:
        list of article dicts with keys: title, date, source, url
    """
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        return []

    articles = []
    cutoff_date = (datetime.now() - timedelta(days=days_lookback)).isoformat()

    # Normalize ticker: remove .AX suffix for search
    clean_ticker = ticker.replace(".AX", "").upper()

    # Try both ticker and company name
    search_terms = [clean_ticker]
    if company_name:
        search_terms.append(company_name)

    for search_term in search_terms:
        try:
            params = {
                "q": search_term,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "from": cutoff_date,
                "apiKey": api_key
            }

            response = requests.get(
                NEWSAPI_BASE,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": BROWSER_USER_AGENT}
            )
            response.raise_for_status()

            data = response.json()

            if data.get("status") == "ok" and data.get("articles"):
                for article in data["articles"]:
                    article_dict = {
                        "title": article.get("title", ""),
                        "date": article.get("publishedAt", ""),
                        "source": "NewsAPI",
                        "url": article.get("url", "")
                    }
                    articles.append(article_dict)

            time.sleep(RATE_LIMIT_SLEEP)

        except requests.exceptions.RequestException:
            # Non-fatal; continue to next search term
            continue
        except (json.JSONDecodeError, KeyError):
            # Malformed response; non-fatal
            continue
        except Exception:
            # Any other error; non-fatal
            continue

    return articles


# ─────────────────────────────────────────────────────────────────────────────
# Yahoo Finance RSS Source
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yahoo_rss_articles(ticker: str) -> list[dict]:
    """
    Fetch articles from Yahoo Finance RSS feed for an ASX ticker.

    Args:
        ticker: ASX ticker (e.g., "CBA.AX")

    Returns:
        list of article dicts with keys: title, date, source, url
    """
    articles = []
    clean_ticker = ticker.replace(".AX", "").upper()

    try:
        url = YAHOO_FINANCE_RSS_TEMPLATE.format(ticker=clean_ticker)
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": BROWSER_USER_AGENT}
        )
        response.raise_for_status()

        # Parse RSS XML
        root = ET.fromstring(response.content)

        # Define namespace for RSS parsing
        ns = {
            'content': 'http://purl.org/rss/1.0/modules/content/'
        }

        # Extract last 10 items from the RSS feed
        items = root.findall('.//item')[:10]

        for item in items:
            title_elem = item.find('title')
            pub_date_elem = item.find('pubDate')
            link_elem = item.find('link')

            if title_elem is not None:
                article_dict = {
                    "title": title_elem.text or "",
                    "date": pub_date_elem.text if pub_date_elem is not None else "",
                    "source": "Yahoo Finance",
                    "url": link_elem.text if link_elem is not None else ""
                }
                articles.append(article_dict)

        time.sleep(RATE_LIMIT_SLEEP)

    except ET.ParseError:
        # XML parsing error; non-fatal
        pass
    except requests.exceptions.RequestException:
        # Network error; non-fatal
        pass
    except Exception:
        # Any other error; non-fatal
        pass

    return articles


# ─────────────────────────────────────────────────────────────────────────────
# Per-Ticker Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_news_sentiment_for_ticker(
    ticker: str,
    company_name: Optional[str] = None,
    days_lookback: int = 30
) -> dict:
    """
    Fetch news from multiple sources and compute sentiment for a single ticker.

    Args:
        ticker: stock ticker
        company_name: optional company name for NewsAPI search
        days_lookback: days of history to fetch

    Returns:
        dict with sentiment analysis results for the ticker
    """
    # Fetch from both sources
    newsapi_articles = fetch_newsapi_articles(ticker, company_name, days_lookback)
    yahoo_articles = fetch_yahoo_rss_articles(ticker)

    # Merge results, avoiding duplicates by title
    all_articles = []
    seen_titles = set()

    for article in newsapi_articles + yahoo_articles:
        title = article.get("title", "").strip()
        if title and title not in seen_titles:
            all_articles.append(article)
            seen_titles.add(title)

    # Analyze sentiment for each article
    article_scores = []
    all_red_flags = []
    red_flag_articles = []

    for article in all_articles:
        score, flags = score_headline(article["title"])
        article_scores.append(score)

        if flags:
            all_red_flags.extend(flags)
            red_flag_articles.append(article["title"])

    # Compute aggregate sentiment
    if article_scores:
        sentiment_score = sum(article_scores) / len(article_scores)
    else:
        sentiment_score = 0.0

    sentiment_score = max(-1.0, min(1.0, sentiment_score))
    sentiment_label = get_sentiment_label(sentiment_score)

    # Get top 3 most recent headlines
    top_headlines = all_articles[:3] if all_articles else []

    # Determine news signal
    if red_flag_articles:
        news_signal = "warning"
    elif len(all_articles) < 2:
        news_signal = "insufficient_data"
    elif sentiment_score > 0.3 and len(all_articles) >= 3:
        news_signal = "positive"
    elif sentiment_score < -0.3 and len(all_articles) >= 3:
        news_signal = "negative"
    else:
        news_signal = "neutral"

    # Determine which sources provided data
    data_sources_used = []
    if newsapi_articles:
        data_sources_used.append("NewsAPI")
    if yahoo_articles:
        data_sources_used.append("Yahoo Finance RSS")

    clean_ticker = ticker.replace(".AX", "").upper()

    return {
        "ticker": clean_ticker,
        "articles_found": len(all_articles),
        "sentiment_score": round(sentiment_score, 3),
        "sentiment_label": sentiment_label,
        "red_flags": list(set(all_red_flags)),  # Deduplicate
        "top_headlines": top_headlines,
        "news_signal": news_signal,
        "data_sources_used": data_sources_used,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Tool Implementation
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_news_sentiment_impl(
    tickers: list[str],
    company_names: Optional[dict[str, str]] = None,
    days_lookback: int = 30
) -> ToolResult:
    """
    Core implementation for fetching and analyzing news sentiment.
    Returns a ToolResult with per-ticker sentiment data.

    Args:
        tickers: list of stock tickers (e.g., ["CBA.AX", "BHP.AX"])
        company_names: optional dict mapping ticker to company name
        days_lookback: days of history to fetch (default 30)

    Returns:
        ToolResult with:
            - data: list of sentiment dicts, one per ticker
            - errors_fatal: set if ALL tickers fail
            - errors_non_fatal: set if SOME tickers fail
    """
    if company_names is None:
        company_names = {}

    errors_fatal: list[str] = []
    errors_non_fatal: list[str] = []
    results = []

    # Normalize tickers
    clean_tickers = [t.replace(".AX", "").upper() for t in tickers]

    # Resolve company names from yfinance for any ticker not supplied by the caller.
    # This prevents LLM hallucination (e.g. SDR.AX being mapped to "Santos Limited"
    # instead of the correct "SiteMinder Limited").
    resolved_names: dict[str, Optional[str]] = {}
    for t in clean_tickers:
        caller_name = company_names.get(t) or company_names.get(f"{t}.AX")
        if caller_name:
            # Still override with yfinance to guard against hallucinated names
            yf_name = _resolve_company_name(t)
            if yf_name and yf_name.lower() != caller_name.lower():
                errors_non_fatal.append(
                    f"[{t}] Caller supplied company_name='{caller_name}' but yfinance "
                    f"returned '{yf_name}'. Using yfinance name to avoid hallucination."
                )
                resolved_names[t] = yf_name
            else:
                resolved_names[t] = caller_name
        else:
            resolved_names[t] = _resolve_company_name(t)

    # Fetch sentiment concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {
            executor.submit(
                analyze_news_sentiment_for_ticker,
                t,
                resolved_names.get(t),
                days_lookback
            ): t
            for t in clean_tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                sentiment_result = future.result(timeout=REQUEST_TIMEOUT + 5)
                results.append(sentiment_result)

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
def get_news_sentiment(
    tickers: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
    company_names: dict[str, str] | None = None,
    days_lookback: int = 30
) -> ToolMessage:
    """Fetch and analyze sentiment from recent news headlines for a list of tickers.

    Use this tool to surface material news that isn't captured in price or fundamental data.
    It fetches recent headlines from multiple sources (NewsAPI and Yahoo Finance RSS),
    analyzes sentiment using keyword matching, and flags material corporate news.

    The tool detects two types of signals:
    - Red flags: regulatory issues (ASIC, investigations), corporate restructuring,
      executive departures, capital raises, class actions
    - Sentiment: positive (growth, dividends, partnerships), negative (losses, downgrades,
      layoffs), or neutral

    IMPORTANT — ticker format:
        Pass tickers exactly as you have them — both "WDS.AX" and "WDS" are accepted.
        The tool normalises the ticker internally. Do NOT manually remove '.AX' before calling.
        This tool works for both ASX and non-ASX tickers (it searches news by symbol name).

    Args:
        tickers: List of stock ticker symbols. Both formats accepted:
                 e.g. ["CBA.AX", "BHP.AX"] or ["CBA", "BHP"].
        company_names: DO NOT populate this from memory. The tool automatically
                       resolves the correct company name from yfinance for each ticker,
                       which prevents hallucination errors (e.g. SDR.AX is SiteMinder,
                       not Santos). Only pass this if you have a verified, authoritative
                       name from a prior tool result. Leave it as None in all other cases.
        days_lookback: Number of days of news history to fetch (default: 30).

    Returns:
        A ToolMessage where:
          - content: human-readable summary of how many tickers were analyzed.
          - artifact: a ToolResult dict with:
              - data: list of sentiment dicts, one per ticker. Each includes:
                  - ticker: normalized ticker code (no .AX suffix)
                  - articles_found: total articles retrieved
                  - sentiment_score: aggregate sentiment in range [-1.0, 1.0]
                  - sentiment_label: "positive", "negative", or "neutral"
                  - red_flags: list of red-flag keywords found in articles
                  - top_headlines: list of 3 most recent articles with {title, date, source, url}
                  - news_signal: "warning" (red flags), "positive", "negative", "neutral",
                      or "insufficient_data" (< 2 articles)
                  - data_sources_used: which sources returned data (NewsAPI, Yahoo Finance RSS)
              - errors_fatal: set when ALL tickers fail to fetch. Check NEWS_API_KEY
                  environment variable and network connectivity.
              - errors_non_fatal: set when SOME (but not all) tickers fail. Analysis
                  can continue on valid tickers.

    Notes:
        - NewsAPI requires the NEWS_API_KEY environment variable. If not set, only Yahoo
          Finance RSS will be used (free tier, no auth needed).
        - Sentiment analysis is keyword-based (no ML). Check top_headlines for context.
        - Red flags require human review before making investment decisions.
    """
    result = _fetch_news_sentiment_impl(tickers, company_names, days_lookback)

    summary = f"Analyzed news sentiment for {len(result['data'])} tickers."
    if result['errors_non_fatal']:
        summary += f" ({len(result['errors_non_fatal'])} had failures.)"

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
    print(f"  News Sentiment Analysis Tool — Test Run")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    test_tickers = ["CBA.AX", "BHP.AX"]
    test_company_names = {
        "CBA.AX": "Commonwealth Bank",
        "BHP.AX": "BHP Group"
    }

    # Use a dummy tool_call_id for testing
    test_tool_call_id = "test-call-123"

    result = get_news_sentiment(
        test_tickers,
        tool_call_id=test_tool_call_id,
        company_names=test_company_names,
        days_lookback=30
    )

    print(f"Tool Response Content:\n{result.content}\n")

    if result.artifact and result.artifact.get("data"):
        print("Results by ticker:")
        for ticker_data in result.artifact["data"]:
            print(f"\n  {ticker_data['ticker']}:")
            print(f"    Articles found: {ticker_data['articles_found']}")
            print(f"    Sentiment score: {ticker_data['sentiment_score']}")
            print(f"    Sentiment label: {ticker_data['sentiment_label']}")
            print(f"    News signal: {ticker_data['news_signal']}")
            print(f"    Sources: {', '.join(ticker_data['data_sources_used'])}")

            if ticker_data.get('red_flags'):
                print(f"    Red flags: {', '.join(ticker_data['red_flags'])}")

            if ticker_data.get('top_headlines'):
                print(f"    Top headlines:")
                for headline in ticker_data['top_headlines'][:3]:
                    print(f"      [{headline.get('source')}] {headline['title'][:70]}...")

    if result.artifact and result.artifact.get("errors_fatal"):
        print(f"\nFatal errors: {result.artifact['errors_fatal']}")

    if result.artifact and result.artifact.get("errors_non_fatal"):
        print(f"\nNon-fatal errors: {result.artifact['errors_non_fatal']}")

    print(f"\n{'='*60}\n")
