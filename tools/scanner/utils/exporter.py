import pandas as pd


def export_results(df: pd.DataFrame, filename: str = "asx_candidates.csv"):
    """Write full results to CSV and a trimmed JSON ready for LLM ingestion."""
    if df.empty:
        return
    df.to_csv(filename, index=True)
    print(f"[EXPORT] CSV -> {filename}")

    # JSON subset: only the fields an LLM needs for second-stage analysis
    json_cols = [
        "ticker", "score", "sector", "industry",
        "current_price", "market_cap",
        "ret_1m", "ret_3m", "ret_6m", "ret_1y",
        "volatility_30d", "rsi_14", "macd_crossover",
        "above_ema50", "golden_cross", "bb_position",
        "pe_ratio", "pb_ratio", "roe", "profit_margin",
        "revenue_growth", "earnings_growth", "debt_to_equity",
        "dividend_yield",
    ]
    json_file = filename.replace(".csv", "_for_llm.json")
    df[[c for c in json_cols if c in df.columns]].to_json(
        json_file, orient="records", indent=2
    )
    print(f"[EXPORT] LLM JSON -> {json_file}")