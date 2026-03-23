"""
ASX Stock Screener — bulk data fetch edition
Requirements: pip install yfinance pandas ta

Key design decisions vs a naive per-ticker loop:
  - Price/volume history : single bulk request via yf.Tickers.history()
  - Fundamental info     : concurrent fetch via ThreadPoolExecutor (no sleep needed)
  - Indicator calculation: pure CPU, no network — runs after both fetches complete

Usage:
    python asx_screener.py
    python asx_screener.py --source asx200 --top 30
    python asx_screener.py --min-turnover 1000000 --no-export
"""
from datetime import datetime
import sys
import traceback

from langchain.tools import tool
import pandas as pd

from tools.ticker.utils.code_fetcher import bulk_download_prices, bulk_fetch_fundamentals, get_tickers
from tools.ticker.utils.exporter import export_results
from tools.ticker.utils.metrics_calculator import compute_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Scoring model
# ─────────────────────────────────────────────────────────────────────────────

def score_stock(row: dict) -> float:
    """
    Score a stock 0-100 across four dimensions.
    Weights are intentionally transparent — edit the numbers to match your style.

    Momentum    30 pts  — recent price performance
    Technical   30 pts  — indicator signals
    Fundamental 30 pts  — financial health
    Risk        10 pts  — volatility discount
    """
    score = 0.0

    # -- Momentum (max 30) ------------------------------------------------
    momentum = 0.0
    if row.get("ret_1m") is not None:
        momentum += min(max(row["ret_1m"] * 100, -5), 10)          # 1-month return
    if row.get("ret_3m") is not None:
        momentum += min(max(row["ret_3m"] * 100 * 0.5, -5), 10)    # 3-month return (half weight)
    pct = row.get("pct_from_52w_high")
    if pct is not None:
        if   -0.05 <= pct <= 0:    momentum += 10  # near 52-week high
        elif -0.20 <= pct < -0.05: momentum += 5
        elif -0.40 <= pct < -0.20: momentum += 2
    score += min(momentum, 30)

    # -- Technical (max 30) -----------------------------------------------
    tech = 0.0
    rsi = row.get("rsi_14")
    if rsi is not None:
        if   50 <= rsi <= 70: tech += 10   # strong but not overbought
        elif 40 <= rsi < 50:  tech += 5
        elif rsi > 70:        tech += 3    # overbought — small penalty
        elif rsi < 30:        tech += 4    # oversold — watch for bounce

    if row.get("macd_crossover") == 1:                              # fresh bullish cross
        tech += 10
    elif row.get("macd_hist") and row["macd_hist"] > 0:             # histogram positive
        tech += 5

    if row.get("above_ema50"):  tech += 5
    if row.get("golden_cross"): tech += 5                           # strong trend signal

    bb = row.get("bb_position")
    if bb is not None:
        if   0.5 <= bb <= 0.8: tech += 5   # mid-to-upper band, trending up
        elif bb > 0.8:          tech += 2   # near upper band, risk of pullback
    score += min(tech, 30)

    # -- Fundamental (max 30) ---------------------------------------------
    fundamental = 0.0
    pe = row.get("pe_ratio")
    if pe and pe > 0:
        if   5  <= pe <= 20: fundamental += 8   # reasonable valuation
        elif 20 < pe <= 35:  fundamental += 5
        elif pe > 35:        fundamental += 2

    roe = row.get("roe")
    if roe is not None:
        if   roe > 0.20: fundamental += 8
        elif roe > 0.10: fundamental += 5
        elif roe > 0:    fundamental += 2

    rg = row.get("revenue_growth")
    if rg is not None:
        if   rg > 0.15: fundamental += 7
        elif rg > 0.05: fundamental += 4
        elif rg > 0:    fundamental += 2

    de = row.get("debt_to_equity")
    if de is not None:
        if   de < 50:  fundamental += 4    # low leverage
        elif de < 100: fundamental += 2
        elif de > 200: fundamental -= 3    # high leverage penalty

    if row.get("profit_margin") and row["profit_margin"] > 0.15:
        fundamental += 3
    score += min(fundamental, 30)

    # -- Risk discount (max 10) -------------------------------------------
    vol = row.get("volatility_30d")
    if vol is not None:
        if   vol < 0.20: score += 10    # low volatility bonus
        elif vol < 0.35: score += 5
        elif vol < 0.55: score += 2
        else:            score -= 5     # extreme volatility penalty

    return round(min(max(score, 0), 100), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Hard filters (must pass before scoring)
# ─────────────────────────────────────────────────────────────────────────────

def passes_hard_filters(row: dict, config: dict) -> tuple[bool, str]:
    """
    Return (passed, reason_if_failed).
    All thresholds are driven by the config dict so callers can tune them.
    """
    turnover = row.get("avg_turnover_30d")
    if turnover is None or pd.isna(turnover):
        return False, "no turnover data"
    if turnover < config["min_turnover"]:
        return False, f"insufficient turnover (${turnover:,.0f})"

    cap = row.get("market_cap")
    if cap is not None and not pd.isna(cap) and cap < config["min_market_cap"]:
        return False, f"market cap too small (${cap:,.0f})"

    price = row.get("current_price")
    if price is not None and not pd.isna(price) and price < config["min_price"]:
        return False, f"price too low (${price:.3f})"

    rsi = row.get("rsi_14")
    if rsi is not None and not pd.isna(rsi) and rsi > config.get("max_rsi", 82):
        return False, f"RSI overbought ({rsi:.1f})"

    vol = row.get("volatility_30d")
    if vol is not None and not pd.isna(vol) and vol > config.get("max_volatility", 0.90):
        return False, f"volatility too high ({vol:.1%})"

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Main screening pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_screener(
    tickers: list[str],
    top_n: int = 20,
    config: dict | None = None,
    fundamental_workers: int = 20,
) -> pd.DataFrame:
    if config is None:
        config = {
            "min_turnover":   500_000,
            "min_market_cap": 100_000_000,
            "min_price":      0.10,
            "max_rsi":        82,
            "max_volatility": 0.90,
        }

    print(f"\n{'='*60}")
    print(f"  ASX Stock Screener — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Universe: {len(tickers)} tickers  |  Target: Top {top_n}")
    print(f"{'='*60}\n")

    # Step 1: get all metrics
    all_metrics = get_metrics_for_tickers(tickers, fundamental_workers)

    # Step 4: filter + score
    records, rejected = [], []
    for metrics in all_metrics:
        passed, reason = passes_hard_filters(metrics, config)
        if not passed:
            rejected.append({"ticker": metrics["ticker"], "reason": reason})
            continue
        metrics["score"] = score_stock(metrics)
        records.append(metrics)

    if not records:
        print("\n[WARNING] No stocks passed the filters. Check parameters or network.")
        return pd.DataFrame()

    df = (
        pd.DataFrame(records)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    df.index += 1  # rank starts at 1
    top_df = df.head(top_n)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  Done — passed: {len(records)}  |  filtered out: {len(rejected)}")
    print(f"  Top {min(top_n, len(top_df))} candidates:")
    print(f"{'='*60}")

    display_cols = ["ticker", "score", "current_price", "ret_1m", "ret_3m",
                    "rsi_14", "volatility_30d", "pe_ratio", "roe", "sector"]
    print_df = top_df[[c for c in display_cols if c in top_df.columns]].copy()
    for col in ["ret_1m", "ret_3m", "volatility_30d", "roe"]:
        if col in print_df.columns:
            print_df[col] = print_df[col].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
            )
    for col in ["pe_ratio", "rsi_14"]:
        if col in print_df.columns:
            print_df[col] = print_df[col].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
            )
    print_df["current_price"] = print_df["current_price"].apply(
        lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
    )
    print(print_df.to_string())
    print()

    return top_df

@tool
def get_metrics_for_tickers(
    tickers: list[str],
    fundamental_workers: int = 20
) -> list[dict]:
    """Fetch and compute technical and fundamental metrics for a list of stock tickers.

    Use this tool when you need quantitative data to analyze or compare stocks. It returns
    a rich set of indicators — price trends, momentum, volatility, valuation ratios, and
    growth metrics — that are directly useful for screening, ranking, or evaluating tickers.

    Args:
        tickers: List of stock ticker symbols to analyze (e.g. ["AAPL", "MSFT", "NVDA"]).
                 Tickers with no available price data are silently skipped.
        fundamental_workers: Number of concurrent workers for fetching fundamental data.
                             Defaults to 20. Increase for larger ticker lists.

    Returns:
        A list of metric dicts, one per valid ticker. Each dict includes fields such as:
          - ticker: str — the ticker symbol
          - price, price_change_*: current price and % change over various windows
          - sma_*, ema_*: simple and exponential moving averages
          - rsi, macd, bollinger_*: momentum and volatility indicators
          - volume, avg_volume_*, relative_volume: volume statistics
          - market_cap, pe_ratio, pb_ratio, ps_ratio, peg_ratio: valuation multiples
          - revenue_growth, earnings_growth, profit_margin, roe, roa: fundamentals
          - debt_to_equity, current_ratio, free_cash_flow: balance sheet metrics
        Tickers for which metric computation fails are skipped and not included in output.
    """
    # Step 1: bulk price download
    try:
        price_data = bulk_download_prices(tickers)
        valid_tickers = list(price_data.keys())
        print(f"\n  Price data OK: {len(valid_tickers)}  |  Skipped (no data): {len(tickers) - len(valid_tickers)}\n")
    except Exception as downloadError:
        # print("  Failed to download prices —", downloadError)
        traceback.print_exception(downloadError, file=sys.stdout)

    # Step 2: concurrent fundamental fetch
    fundamentals = bulk_fetch_fundamentals(valid_tickers, max_workers=fundamental_workers)

    # Step 3: compute indicators
    print(f"\n  Computing technical and fundamental metrics...")
    all_metrics = []
    for ticker in valid_tickers:
        try:
            metrics = compute_metrics(
                ticker=ticker,
                price_df=price_data[ticker],
                info=fundamentals.get(ticker, {}),
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  [SKIP {ticker}] metric computation failed: {e}")
    
    return all_metrics

@tool
def run_l1_screener(
    top: int = 20,
    minTurnover: int = 500_000,
    minCap: int = 100_000_000,
    source: str = 'manual'
):
    """
    Scans the Australian Securities Exchange (ASX) to identify high-potential stocks using a multi-factor quantitative model.
    Do NOT use this if the user is only asking about certain tickers.

    This tool performs the Stage 1 (L1) screening process. It integrates technical momentum 
    indicators (RSI, MACD, EMA), fundamental health metrics (PE Ratio, ROE), and liquidity 
    risk controls. It is designed to filter out market noise and pinpoint stocks that are 
    currently in a confirmed uptrend with solid financial backing.

    Args:
        top (int, optional): The number of top-ranked candidate stocks to return. 
            Defaults to 20.
        minTurnover (int, optional): The minimum 30-day average daily turnover threshold 
            in AUD. Used to filter out illiquid small-cap stocks to ensure ease of entry 
            and exit. Defaults to 500,000.
        minCap (int, optional): The minimum market capitalization threshold in AUD. 
            For ASX 200-level universe screening, 100,000,000 is recommended. 
            Defaults to 100,000,000.
        source (str, optional): The source of the ticker universe.
            'manual': Uses a pre-defined list of blue-chip and high-conviction sectors.
            'asx200': Scrapes the latest S&P/ASX 200 index constituents in real-time.
            Defaults to 'manual'.

    Returns:
        str: The file path to the generated 'asx_candidates_for_llm.json' containing 
             detailed metrics for the top candidates, or a warning message if no stocks 
             passed the hard filters.
    """
    tickers = get_tickers(source)

    results = run_screener(
        tickers=tickers,
        top_n=top,
        config={
            "min_turnover":   minTurnover,
            "min_market_cap": minCap,
            "min_price":      0.10,
            "max_rsi":        82,
            "max_volatility": 0.90,
        },
        fundamental_workers=20,
    )

    if not results.empty:
        export_results(results)
        print("\nNext step (LLM deep analysis):")
        print("  Feed asx_candidates_for_llm.json to your LLM together with:")
        print("  - Latest earnings summaries / analyst reports")
        print("  - Commodity price moves relevant to each sector")
        print("  - Recent macro data and geopolitical developments\n")
