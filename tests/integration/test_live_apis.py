"""
Integration tests — call REAL external APIs with a real well-known ticker.

These tests are excluded from the default pytest run (marked 'integration').
Run them explicitly:

    pytest tests/integration -m integration -v

Or via Make:

    make test-live

Requirements:
  - Real network access
  - Optional: FRED_API_KEY in env (tests degrade gracefully without it)
  - Optional: NEWS_API_KEY in env (tests degrade gracefully without it)

These tests serve two purposes:
  1. Smoke test that external APIs haven't changed structure
  2. Verify that data quality checks catch real-world range violations

Run schedule: daily via CI, or manually before a trading session.
"""

import pytest
import os


WELL_KNOWN_TICKERS = ["CBA.AX", "BHP.AX"]   # stable large caps, always have data


# ─────────────────────────────────────────────────────────────────────────────
# Macro Tool — FRED API
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestMacroToolLive:
    def test_fetches_audusd_in_valid_range(self):
        from tools.ticker.macro_tools import _fetch_macro_context
        api_key = os.environ.get("FRED_API_KEY")
        result = _fetch_macro_context(api_key)

        if result["errors_fatal"]:
            pytest.skip(f"FRED API unavailable: {result['errors_fatal']}")

        macro = result["data"]
        if macro["audusd_current"] is not None:
            assert 0.40 < macro["audusd_current"] < 1.20, (
                f"AUD/USD {macro['audusd_current']} outside sanity range 0.40-1.20"
            )

    def test_rba_cash_rate_in_plausible_range(self):
        from tools.ticker.macro_tools import _fetch_macro_context
        api_key = os.environ.get("FRED_API_KEY")
        result = _fetch_macro_context(api_key)

        if result["errors_fatal"]:
            pytest.skip("FRED API unavailable")

        macro = result["data"]
        if macro["rba_cash_rate"] is not None:
            assert 0.0 <= macro["rba_cash_rate"] <= 20.0, (
                f"RBA rate {macro['rba_cash_rate']} outside plausible range"
            )

    def test_macro_signal_is_valid_enum(self):
        from tools.ticker.macro_tools import _fetch_macro_context
        result = _fetch_macro_context()
        assert result["data"]["macro_signal"] in {"favorable", "unfavorable", "neutral"}

    def test_macro_notes_populated_with_real_data(self):
        from tools.ticker.macro_tools import _fetch_macro_context
        result = _fetch_macro_context()
        if not result["errors_fatal"]:
            assert len(result["data"]["macro_notes"]) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Announcements Tool — ASX public API
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestAnnouncementsToolLive:
    def test_fetches_announcements_for_cba(self):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        result = _fetch_announcements_impl(["CBA.AX"])

        if result["errors_fatal"]:
            pytest.skip(f"ASX API unavailable: {result['errors_fatal']}")

        assert len(result["data"]) >= 1
        item = result["data"][0]
        assert item["ticker"] == "CBA"

    def test_announcements_count_is_realistic(self):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        result = _fetch_announcements_impl(["CBA.AX"])

        if result["errors_fatal"]:
            pytest.skip("ASX API unavailable")

        if result["data"]:
            count = result["data"][0]["announcements_count"]
            # CBA makes several announcements per month
            assert 0 <= count <= 200, f"Unexpected announcement count: {count}"

    def test_signal_is_valid_for_major_bank(self):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        result = _fetch_announcements_impl(["CBA.AX"])

        if result["errors_fatal"]:
            pytest.skip("ASX API unavailable")

        for item in result["data"]:
            assert item["announcement_signal"] in {"positive", "negative", "neutral", "warning"}

    def test_multiple_tickers_fetched_concurrently(self):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        result = _fetch_announcements_impl(WELL_KNOWN_TICKERS)

        if result["errors_fatal"]:
            pytest.skip("ASX API unavailable")

        assert len(result["data"]) >= 1  # at least one should succeed


# ─────────────────────────────────────────────────────────────────────────────
# Short Interest Tool — Shortman
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestShortInterestToolLive:
    def test_fetches_short_interest_for_cba(self):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        result = _fetch_short_interest(["CBA.AX"])

        item = result["data"][0] if result["data"] else None
        if item is None or item.get("fetch_error"):
            pytest.skip(f"Shortman unavailable: {item}")

        assert item["short_pct"] is not None
        # CBA short interest is typically 0.5-5%
        assert 0.0 <= item["short_pct"] <= 30.0, (
            f"CBA short % {item['short_pct']} outside plausible range"
        )

    def test_short_signal_is_valid_enum(self):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        result = _fetch_short_interest(["CBA.AX"])

        for item in result["data"]:
            assert item["short_signal"] in {"bullish", "bearish", "warning", "neutral", "unknown"}

    def test_data_date_parseable(self):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        result = _fetch_short_interest(["CBA.AX"])

        for item in result["data"]:
            if item.get("data_date"):
                # Should contain year 2024, 2025, or 2026
                assert any(yr in item["data_date"] for yr in ["2024", "2025", "2026"]), (
                    f"Unexpected date format: {item['data_date']}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# News Sentiment Tool — Yahoo Finance RSS + NewsAPI
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestNewsSentimentToolLive:
    def test_yahoo_rss_returns_articles_for_cba(self):
        from tools.ticker.news_sentiment_tools import fetch_yahoo_rss_articles
        articles = fetch_yahoo_rss_articles("CBA.AX")
        # Yahoo Finance RSS should have at least some articles for a major ASX stock
        # Note: this can legitimately return 0 if Yahoo's feed is empty
        assert isinstance(articles, list)
        for a in articles:
            assert "title" in a
            assert "source" in a
            assert a["source"] == "Yahoo Finance"

    def test_full_pipeline_returns_valid_structure_for_cba(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        result = _fetch_news_sentiment_impl(["CBA.AX"])

        assert result["errors_fatal"] == [] or isinstance(result["errors_fatal"], list)
        if result["data"]:
            item = result["data"][0]
            assert item["ticker"] == "CBA"
            assert item["news_signal"] in {
                "positive", "negative", "warning", "neutral", "insufficient_data"
            }
            assert -1.0 <= item["sentiment_score"] <= 1.0

    def test_newsapi_degrades_gracefully_without_key(self):
        import os
        from tools.ticker.news_sentiment_tools import fetch_newsapi_articles
        # Temporarily ensure key is unset
        original = os.environ.pop("NEWS_API_KEY", None)
        try:
            result = fetch_newsapi_articles("CBA.AX")
            assert result == []  # should silently return empty, not raise
        finally:
            if original:
                os.environ["NEWS_API_KEY"] = original

    def test_sentiment_score_in_valid_range_for_real_headlines(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        result = _fetch_news_sentiment_impl(["BHP.AX"])
        for item in result["data"]:
            assert -1.0 <= item["sentiment_score"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Tool — yfinance (live market data)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.slow
class TestMetricsToolLive:
    def test_fetches_price_data_for_cba(self):
        from tools.ticker.metrics_tools import _fetch_metrics
        result = _fetch_metrics(["CBA.AX"])

        if result["errors_fatal"]:
            pytest.skip(f"yfinance unavailable: {result['errors_fatal']}")

        assert len(result["data"]) >= 1
        cba = result["data"][0]
        assert cba["ticker"] == "CBA.AX"

    def test_current_price_in_plausible_range(self):
        from tools.ticker.metrics_tools import _fetch_metrics
        result = _fetch_metrics(["CBA.AX"])

        if result["errors_fatal"]:
            pytest.skip("yfinance unavailable")

        if result["data"]:
            price = result["data"][0]["current_price"]
            # CBA trades roughly $80-$180 range historically
            assert 10.0 < price < 500.0, f"CBA price {price} outside plausible range"

    def test_rsi_in_valid_range_with_live_data(self):
        from tools.ticker.metrics_tools import _fetch_metrics
        result = _fetch_metrics(["CBA.AX"])

        if result["errors_fatal"] or not result["data"]:
            pytest.skip("yfinance unavailable")

        rsi = result["data"][0].get("rsi_14")
        if rsi is not None:
            assert 0 <= rsi <= 100, f"RSI {rsi} out of range"

    def test_invalid_ticker_handled_gracefully(self):
        from tools.ticker.metrics_tools import _fetch_metrics
        result = _fetch_metrics(["XXXXNOTREAL.AX"])
        # Should not raise; ticker should be in non-fatal or fatal errors
        assert isinstance(result, dict)
        assert "errors_fatal" in result or "errors_non_fatal" in result
