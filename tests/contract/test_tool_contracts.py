"""
Contract tests — enforce the ToolResult schema and signal enum contracts
across ALL tools. These are the tests that catch breaking changes.

If you rename a field, change an allowed signal value, or restructure
ToolResult, these tests break immediately — before the LLM silently
hallucinates over a missing key.

Run these after every change to any tool file.
"""

import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import (
    make_mock_response, make_fred_observations,
    make_asx_api_response, make_shortman_html, make_yahoo_rss
)


# ─────────────────────────────────────────────────────────────────────────────
# ToolResult schema contract (shared across all tools)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_RESULT_REQUIRED_KEYS = {"data", "errors_fatal", "errors_non_fatal"}


def assert_valid_tool_result(result: dict, label: str = ""):
    """Assert the ToolResult envelope matches the contract."""
    assert isinstance(result, dict), f"{label}: ToolResult must be a dict"
    assert TOOL_RESULT_REQUIRED_KEYS == set(result.keys()), (
        f"{label}: ToolResult keys mismatch. "
        f"Expected {TOOL_RESULT_REQUIRED_KEYS}, got {set(result.keys())}"
    )
    assert isinstance(result["errors_fatal"], list), f"{label}: errors_fatal must be a list"
    assert isinstance(result["errors_non_fatal"], list), f"{label}: errors_non_fatal must be a list"
    assert isinstance(result["data"], (list, dict, type(None))), (
        f"{label}: data must be list, dict, or None"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Macro Tool Contracts
# ─────────────────────────────────────────────────────────────────────────────

MACRO_SIGNAL_VALUES = {"favorable", "unfavorable", "neutral"}
MACRO_TREND_VALUES = {"rising", "falling", "flat"}
MACRO_AUD_TREND_VALUES = {"strengthening", "weakening", "flat"}

class TestMacroToolContract:
    def test_tool_result_schema(self):
        from tools.ticker.macro_tools import _fetch_macro_context
        with patch("tools.ticker.macro_tools._fetch_fred_series", return_value=[]):
            result = _fetch_macro_context()
        assert_valid_tool_result(result, "get_macro_context")

    def test_macro_context_required_keys(self):
        from tools.ticker.macro_tools import _fetch_macro_context
        with patch("tools.ticker.macro_tools._fetch_fred_series", return_value=[]):
            result = _fetch_macro_context()
        macro = result["data"]
        required = {
            "audusd_current", "audusd_1m_change_pct", "audusd_3m_change_pct",
            "audusd_trend", "rba_cash_rate", "rba_rate_direction",
            "gold_1m_change_pct", "gold_trend", "oil_1m_change_pct", "oil_trend",
            "iron_ore_trend", "macro_signal", "macro_notes",
        }
        for key in required:
            assert key in macro, f"MacroContext missing key: {key}"

    def test_macro_signal_always_valid_enum(self):
        from tools.ticker.macro_tools import _fetch_macro_context
        with patch("tools.ticker.macro_tools._fetch_fred_series", return_value=[]):
            result = _fetch_macro_context()
        assert result["data"]["macro_signal"] in MACRO_SIGNAL_VALUES

    def test_audusd_trend_always_valid_enum(self):
        from tools.ticker.macro_tools import _fetch_macro_context
        with patch("tools.ticker.macro_tools._fetch_fred_series", return_value=[]):
            result = _fetch_macro_context()
        assert result["data"]["audusd_trend"] in MACRO_AUD_TREND_VALUES

    def test_rba_direction_always_valid_enum(self):
        from tools.ticker.macro_tools import _fetch_macro_context
        with patch("tools.ticker.macro_tools._fetch_fred_series", return_value=[]):
            result = _fetch_macro_context()
        assert result["data"]["rba_rate_direction"] in MACRO_TREND_VALUES

    def test_macro_notes_is_list(self):
        from tools.ticker.macro_tools import _fetch_macro_context
        with patch("tools.ticker.macro_tools._fetch_fred_series", return_value=[]):
            result = _fetch_macro_context()
        assert isinstance(result["data"]["macro_notes"], list)

    def test_fatal_implies_all_data_is_none(self):
        """When errors_fatal is set, all key macro values must be None —
        the agent should not trust any data field in that state.
        Note: non_fatal may also be populated simultaneously; they carry
        per-series diagnostics while fatal carries the overall verdict."""
        from tools.ticker.macro_tools import _fetch_macro_context
        with patch("tools.ticker.macro_tools._fetch_fred_series", return_value=[]):
            result = _fetch_macro_context()
        if result["errors_fatal"]:
            macro = result["data"]
            for key in ["audusd_current", "rba_cash_rate", "gold_1m_change_pct", "oil_1m_change_pct"]:
                assert macro[key] is None, (
                    f"Key '{key}' should be None when fatal error is set, got {macro[key]}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Announcements Tool Contracts
# ─────────────────────────────────────────────────────────────────────────────

ANNOUNCEMENT_SIGNAL_VALUES = {"positive", "negative", "neutral", "warning"}

class TestAnnouncementsToolContract:
    def test_tool_result_schema(self):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        mock_resp = make_mock_response(status_code=503)
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = _fetch_announcements_impl(["CBA.AX"])
        assert_valid_tool_result(result, "get_asx_announcements")

    def test_per_ticker_required_keys(self):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        mock_resp = make_mock_response(json_data=make_asx_api_response([
            {"title": "Quarterly update on progress"}
        ]))
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = _fetch_announcements_impl(["CBA.AX"])
        if result["data"]:
            item = result["data"][0]
            required_keys = {
                "ticker", "announcements_count", "market_sensitive_count",
                "red_flags", "positive_signals", "latest_announcement",
                "announcement_signal",
            }
            for key in required_keys:
                assert key in item, f"Announcement result missing key: {key}"

    def test_signal_always_valid_enum(self):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        mock_resp = make_mock_response(json_data=make_asx_api_response([
            {"title": "Dividend declared"}
        ]))
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = _fetch_announcements_impl(["CBA.AX"])
        for item in result["data"]:
            assert item["announcement_signal"] in ANNOUNCEMENT_SIGNAL_VALUES

    def test_red_flags_is_list(self):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        mock_resp = make_mock_response(json_data=make_asx_api_response([
            {"title": "Normal quarterly update"}
        ]))
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = _fetch_announcements_impl(["CBA.AX"])
        for item in result["data"]:
            assert isinstance(item["red_flags"], list)
            assert isinstance(item["positive_signals"], list)

    def test_counts_are_non_negative_integers(self):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        mock_resp = make_mock_response(json_data=make_asx_api_response([
            {"title": "Update", "market_sensitive": True}
        ]))
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = _fetch_announcements_impl(["CBA.AX"])
        for item in result["data"]:
            assert isinstance(item["announcements_count"], int)
            assert item["announcements_count"] >= 0
            assert item["market_sensitive_count"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Short Interest Tool Contracts
# ─────────────────────────────────────────────────────────────────────────────

SHORT_SIGNAL_VALUES = {"bullish", "bearish", "warning", "neutral", "unknown"}
SHORT_TREND_VALUES = {"increasing", "decreasing", "flat", "unknown"}

class TestShortInterestToolContract:
    def test_tool_result_schema(self):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        with patch("tools.ticker.short_interest_tools.requests.get",
                   side_effect=ConnectionError("down")):
            result = _fetch_short_interest(["CBA.AX"])
        assert_valid_tool_result(result, "get_short_interest")

    def test_per_ticker_required_keys(self):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        mock_resp = make_mock_response(text=make_shortman_html(5.0, 5.5))
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp):
            result = _fetch_short_interest(["CBA.AX"])
        if result["data"]:
            item = result["data"][0]
            required_keys = {
                "ticker", "short_pct", "short_pct_prev", "short_change_pct",
                "short_trend", "short_signal", "data_date", "fetch_error",
            }
            for key in required_keys:
                assert key in item, f"ShortInterestData missing key: {key}"

    def test_signal_always_valid_enum(self):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        mock_resp = make_mock_response(text=make_shortman_html(5.0, 5.5))
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp):
            result = _fetch_short_interest(["CBA.AX"])
        for item in result["data"]:
            assert item["short_signal"] in SHORT_SIGNAL_VALUES

    def test_trend_always_valid_enum(self):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        mock_resp = make_mock_response(text=make_shortman_html(5.0, 5.5))
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp):
            result = _fetch_short_interest(["CBA.AX"])
        for item in result["data"]:
            assert item["short_trend"] in SHORT_TREND_VALUES

    def test_short_pct_is_float_or_none(self):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        mock_resp = make_mock_response(text=make_shortman_html(7.25))
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp):
            result = _fetch_short_interest(["CBA.AX"])
        for item in result["data"]:
            assert item["short_pct"] is None or isinstance(item["short_pct"], float)

    def test_entry_per_ticker_even_on_failure(self):
        """Every ticker gets an entry — signals 'unknown' on failure, doesn't disappear."""
        from tools.ticker.short_interest_tools import _fetch_short_interest
        with patch("tools.ticker.short_interest_tools.requests.get",
                   side_effect=ConnectionError("down")):
            result = _fetch_short_interest(["CBA.AX", "BHP.AX"])
        assert len(result["data"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# News Sentiment Tool Contracts
# ─────────────────────────────────────────────────────────────────────────────

NEWS_SIGNAL_VALUES = {"positive", "negative", "warning", "neutral", "insufficient_data"}
SENTIMENT_LABEL_VALUES = {"positive", "negative", "neutral"}

class TestNewsSentimentToolContract:
    def test_tool_result_schema(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        with patch("tools.ticker.news_sentiment_tools.fetch_newsapi_articles", return_value=[]):
            with patch("tools.ticker.news_sentiment_tools.fetch_yahoo_rss_articles", return_value=[]):
                result = _fetch_news_sentiment_impl(["CBA.AX"])
        assert_valid_tool_result(result, "get_news_sentiment")

    def test_per_ticker_required_keys(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        with patch("tools.ticker.news_sentiment_tools.fetch_newsapi_articles", return_value=[]):
            with patch("tools.ticker.news_sentiment_tools.fetch_yahoo_rss_articles", return_value=[]):
                result = _fetch_news_sentiment_impl(["CBA.AX"])
        if result["data"]:
            item = result["data"][0]
            required_keys = {
                "ticker", "articles_found", "sentiment_score", "sentiment_label",
                "red_flags", "top_headlines", "news_signal", "data_sources_used",
            }
            for key in required_keys:
                assert key in item, f"News sentiment result missing key: {key}"

    def test_news_signal_always_valid_enum(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        with patch("tools.ticker.news_sentiment_tools.fetch_newsapi_articles", return_value=[]):
            with patch("tools.ticker.news_sentiment_tools.fetch_yahoo_rss_articles", return_value=[]):
                result = _fetch_news_sentiment_impl(["CBA.AX"])
        for item in result["data"]:
            assert item["news_signal"] in NEWS_SIGNAL_VALUES

    def test_sentiment_label_always_valid_enum(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        with patch("tools.ticker.news_sentiment_tools.fetch_newsapi_articles", return_value=[]):
            with patch("tools.ticker.news_sentiment_tools.fetch_yahoo_rss_articles", return_value=[]):
                result = _fetch_news_sentiment_impl(["CBA.AX"])
        for item in result["data"]:
            assert item["sentiment_label"] in SENTIMENT_LABEL_VALUES

    def test_sentiment_score_in_valid_range(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        with patch("tools.ticker.news_sentiment_tools.fetch_newsapi_articles", return_value=[]):
            with patch("tools.ticker.news_sentiment_tools.fetch_yahoo_rss_articles", return_value=[]):
                result = _fetch_news_sentiment_impl(["CBA.AX"])
        for item in result["data"]:
            assert -1.0 <= item["sentiment_score"] <= 1.0

    def test_red_flags_is_list(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        with patch("tools.ticker.news_sentiment_tools.fetch_newsapi_articles", return_value=[]):
            with patch("tools.ticker.news_sentiment_tools.fetch_yahoo_rss_articles", return_value=[]):
                result = _fetch_news_sentiment_impl(["CBA.AX"])
        for item in result["data"]:
            assert isinstance(item["red_flags"], list)

    def test_top_headlines_is_list(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        with patch("tools.ticker.news_sentiment_tools.fetch_newsapi_articles", return_value=[]):
            with patch("tools.ticker.news_sentiment_tools.fetch_yahoo_rss_articles", return_value=[]):
                result = _fetch_news_sentiment_impl(["CBA.AX"])
        for item in result["data"]:
            assert isinstance(item["top_headlines"], list)
            assert len(item["top_headlines"]) <= 3

    def test_articles_found_is_non_negative_int(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        with patch("tools.ticker.news_sentiment_tools.fetch_newsapi_articles", return_value=[]):
            with patch("tools.ticker.news_sentiment_tools.fetch_yahoo_rss_articles", return_value=[]):
                result = _fetch_news_sentiment_impl(["CBA.AX"])
        for item in result["data"]:
            assert isinstance(item["articles_found"], int)
            assert item["articles_found"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Cross-tool data sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossToolSanity:
    """Verify that signals across all tools use consistent string values
    (e.g. no tool uses 'POSITIVE' vs 'positive' inconsistency)."""

    def test_all_signal_values_are_lowercase(self):
        """Signal enum values must be lowercase strings — LLM prompts depend on this."""
        all_signal_sets = [
            MACRO_SIGNAL_VALUES,
            ANNOUNCEMENT_SIGNAL_VALUES,
            SHORT_SIGNAL_VALUES,
            NEWS_SIGNAL_VALUES,
            SENTIMENT_LABEL_VALUES,
        ]
        for signal_set in all_signal_sets:
            for val in signal_set:
                assert val == val.lower(), f"Signal value '{val}' is not lowercase"

    def test_no_signal_set_contains_none(self):
        """None should never appear as a signal value — use 'unknown' or 'insufficient_data'."""
        all_signal_sets = [
            MACRO_SIGNAL_VALUES, ANNOUNCEMENT_SIGNAL_VALUES,
            SHORT_SIGNAL_VALUES, NEWS_SIGNAL_VALUES,
        ]
        for signal_set in all_signal_sets:
            assert None not in signal_set
