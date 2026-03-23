import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker metric computation (pure CPU, no network)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(ticker: str, price_df: pd.DataFrame, info: dict) -> dict:
    """
    Compute all screening metrics for one ticker given its price DataFrame
    and its fundamental info dict. No network calls are made here.
    """
    close  = price_df["Close"]
    volume = price_df["Volume"]
    high   = price_df["High"]
    low    = price_df["Low"]

    result: dict = {"ticker": ticker}

    # -- Liquidity --------------------------------------------------------
    result["avg_volume_30d"]   = volume.tail(30).mean()
    result["avg_turnover_30d"] = (close.tail(30) * volume.tail(30)).mean()
    result["current_price"]    = close.iloc[-1]
    result["market_cap"]       = info.get("marketCap")

    # -- Price momentum ---------------------------------------------------
    result["ret_1m"] = (close.iloc[-1] / close.iloc[-22]  - 1) if len(close) >= 22  else None
    result["ret_3m"] = (close.iloc[-1] / close.iloc[-66]  - 1) if len(close) >= 66  else None
    result["ret_6m"] = (close.iloc[-1] / close.iloc[-132] - 1) if len(close) >= 132 else None
    result["ret_1y"] = (close.iloc[-1] / close.iloc[0]    - 1)

    result["high_52w"]          = close.tail(252).max()
    result["low_52w"]           = close.tail(252).min()
    result["pct_from_52w_high"] = (close.iloc[-1] / result["high_52w"]) - 1

    # -- Volatility -------------------------------------------------------
    daily_ret = close.pct_change().dropna()
    result["volatility_30d"] = daily_ret.tail(30).std() * (252 ** 0.5)  # annualised

    # -- Fundamentals (sourced from yfinance info) ------------------------
    result["pe_ratio"]        = info.get("trailingPE")
    result["pb_ratio"]        = info.get("priceToBook")
    result["ps_ratio"]        = info.get("priceToSalesTrailing12Months")
    result["roe"]             = info.get("returnOnEquity")
    result["profit_margin"]   = info.get("profitMargins")
    result["revenue_growth"]  = info.get("revenueGrowth")
    result["earnings_growth"] = info.get("earningsGrowth")
    result["debt_to_equity"]  = info.get("debtToEquity")
    result["current_ratio"]   = info.get("currentRatio")
    result["dividend_yield"]  = info.get("dividendYield")
    result["sector"]          = info.get("sector", "Unknown")
    result["industry"]        = info.get("industry", "Unknown")

    # -- Technical indicators ---------------------------------------------

    # RSI(14): overbought > 70, oversold < 30
    rsi = RSIIndicator(close, window=14).rsi()
    result["rsi_14"] = rsi.iloc[-1] if not rsi.empty else None

    # MACD histogram and crossover signal (+1 bullish cross, -1 bearish, 0 none)
    macd_obj  = MACD(close)
    macd_hist = macd_obj.macd_diff()
    result["macd_hist"] = macd_hist.iloc[-1] if not macd_hist.empty else None
    if len(macd_hist.dropna()) >= 2:
        result["macd_crossover"] = (
             1 if (macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0) else
            -1 if (macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0) else 0
        )
    else:
        result["macd_crossover"] = 0

    # EMA trend: price above EMA50 signals bullish trend
    ema50  = EMAIndicator(close, window=50).ema_indicator()
    ema200 = EMAIndicator(close, window=200).ema_indicator()
    result["above_ema50"] = bool(close.iloc[-1] > ema50.iloc[-1])

    # Golden cross: EMA50 crosses above EMA200 (strong bullish signal)
    ema200_clean = ema200.dropna()
    result["golden_cross"] = (
        bool(ema50.iloc[-1] > ema200.iloc[-1] and ema50.iloc[-2] <= ema200.iloc[-2])
        if len(ema200_clean) >= 2 else False
    )

    # Bollinger Band position: 0 = at lower band, 1 = at upper band
    bb     = BollingerBands(close, window=20, window_dev=2)
    bb_rng = bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]
    result["bb_position"] = (
        (close.iloc[-1] - bb.bollinger_lband().iloc[-1]) / bb_rng
        if bb_rng > 0 else None
    )

    # ATR as a percentage of price — normalised measure of daily range
    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
    result["atr_pct"] = (atr.iloc[-1] / close.iloc[-1]) if close.iloc[-1] > 0 else None

    return result