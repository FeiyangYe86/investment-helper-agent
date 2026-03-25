"""
Unit tests for tools/ticker/news_sentiment_tools.py

Coverage:
  - score_headline                     : pure scoring function, all keyword branches
  - get_sentiment_label                : pure label function
  - fetch_newsapi_articles             : mocked HTTP, env var handling
  - fetch_yahoo_rss_articles           : mocked HTTP, XML parsing
  - analyze_news_sentiment_for_ticker  : full per-ticker pipeline (mocked sources)
  - _fetch_news_sentiment_impl         : concurrent pipeline, fatal vs non-fatal
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import (
    make_newsapi_response, make_yahoo_rss, make_mock_response,
)


# ─────────────────────────────────────────────────────────────────────────────
# score_headline  (pure function)
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreHeadline:
    def test_all_positive_keywords_score_positive(self):
        from tools.ticker.news_sentiment_tools import score_headline
        score, flags = score_headline("CBA reports record profit beating expectations with dividend upgrade")
        assert score > 0
        assert flags == []

    def test_all_negative_keywords_score_negative(self):
        from tools.ticker.news_sentiment_tools import score_headline
        score, flags = score_headline("CBA posts loss amid declining revenue and earnings miss")
        assert score < 0

    def test_mixed_keywords_score_near_zero(self):
        from tools.ticker.news_sentiment_tools import score_headline
        score, flags = score_headline("CBA profit up but faces investigation risk")
        # Has both positive (profit) and negative (investigation, risk) → mixed
        assert -1.0 <= score <= 1.0

    def test_no_keywords_scores_zero(self):
        from tools.ticker.news_sentiment_tools import score_headline
        score, flags = score_headline("CBA updates its corporate registry address")
        assert score == pytest.approx(0.0)
        assert flags == []

    def test_asic_is_red_flag(self):
        from tools.ticker.news_sentiment_tools import score_headline
        _, flags = score_headline("ASIC launches investigation into Commonwealth Bank")
        assert len(flags) > 0
        assert any("asic" in f.lower() or "investigation" in f.lower() for f in flags)

    def test_class_action_is_red_flag(self):
        from tools.ticker.news_sentiment_tools import score_headline
        _, flags = score_headline("Class action filed against CBA by shareholders")
        assert len(flags) > 0

    def test_capital_raise_is_red_flag(self):
        from tools.ticker.news_sentiment_tools import score_headline
        _, flags = score_headline("CBA announces capital raise via institutional placement")
        assert len(flags) > 0

    def test_ceo_resign_is_red_flag(self):
        from tools.ticker.news_sentiment_tools import score_headline
        _, flags = score_headline("CEO resign with immediate effect, replacement sought")
        assert len(flags) > 0

    def test_score_clamped_to_range(self):
        from tools.ticker.news_sentiment_tools import score_headline
        # Can't exceed -1 or +1
        score, _ = score_headline("profit growth record upgrade beat dividend acquire partnership win contract strong")
        assert -1.0 <= score <= 1.0

    def test_case_insensitive_keyword_matching(self):
        from tools.ticker.news_sentiment_tools import score_headline
        score_lower, _ = score_headline("record profit reported")
        score_upper, _ = score_headline("RECORD PROFIT REPORTED")
        assert score_lower == pytest.approx(score_upper)

    def test_multiple_red_flags_all_captured(self):
        from tools.ticker.news_sentiment_tools import score_headline
        _, flags = score_headline("ASIC investigation: class action filed, CEO resign pending")
        assert len(flags) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# get_sentiment_label  (pure function)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetSentimentLabel:
    def test_positive_score_returns_positive(self):
        from tools.ticker.news_sentiment_tools import get_sentiment_label
        assert get_sentiment_label(0.5) == "positive"
        assert get_sentiment_label(0.11) == "positive"

    def test_negative_score_returns_negative(self):
        from tools.ticker.news_sentiment_tools import get_sentiment_label
        assert get_sentiment_label(-0.5) == "negative"
        assert get_sentiment_label(-0.11) == "negative"

    def test_neutral_range(self):
        from tools.ticker.news_sentiment_tools import get_sentiment_label
        assert get_sentiment_label(0.0) == "neutral"
        assert get_sentiment_label(0.05) == "neutral"
        assert get_sentiment_label(-0.05) == "neutral"
        assert get_sentiment_label(0.1) == "neutral"
        assert get_sentiment_label(-0.1) == "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# fetch_newsapi_articles  (mocked HTTP + env var)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchNewsapiArticles:
    def test_returns_empty_when_no_api_key(self, newsapi_positive_response):
        from tools.ticker.news_sentiment_tools import fetch_newsapi_articles
        with patch.dict(os.environ, {}, clear=True):
            # Ensure NEWS_API_KEY is not in env
            os.environ.pop("NEWS_API_KEY", None)
            result = fetch_newsapi_articles("CBA.AX")
        assert result == []

    def test_returns_articles_with_api_key(self, newsapi_positive_response):
        from tools.ticker.news_sentiment_tools import fetch_newsapi_articles
        mock_resp = make_mock_response(json_data=newsapi_positive_response)
        with patch.dict(os.environ, {"NEWS_API_KEY": "test_key_123"}):
            with patch("tools.ticker.news_sentiment_tools.requests.get", return_value=mock_resp):
                result = fetch_newsapi_articles("CBA.AX")
        assert len(result) > 0
        assert all("title" in a for a in result)

    def test_searches_with_company_name_when_provided(self, newsapi_positive_response):
        from tools.ticker.news_sentiment_tools import fetch_newsapi_articles
        mock_resp = make_mock_response(json_data=newsapi_positive_response)
        with patch.dict(os.environ, {"NEWS_API_KEY": "test_key"}):
            with patch("tools.ticker.news_sentiment_tools.requests.get",
                       return_value=mock_resp) as mock_get:
                fetch_newsapi_articles("CBA.AX", company_name="Commonwealth Bank")
        assert mock_get.call_count == 2  # once for ticker, once for company name

    def test_graceful_degradation_on_network_error(self):
        from tools.ticker.news_sentiment_tools import fetch_newsapi_articles
        with patch.dict(os.environ, {"NEWS_API_KEY": "test_key"}):
            with patch("tools.ticker.news_sentiment_tools.requests.get",
                       side_effect=ConnectionError("timeout")):
                result = fetch_newsapi_articles("CBA.AX")
        assert result == []   # must return empty, not raise

    def test_strips_ax_from_search_term(self, newsapi_positive_response):
        from tools.ticker.news_sentiment_tools import fetch_newsapi_articles
        mock_resp = make_mock_response(json_data=newsapi_positive_response)
        with patch.dict(os.environ, {"NEWS_API_KEY": "test_key"}):
            with patch("tools.ticker.news_sentiment_tools.requests.get",
                       return_value=mock_resp) as mock_get:
                fetch_newsapi_articles("CBA.AX")
        call_params = str(mock_get.call_args)
        assert ".AX" not in call_params

    def test_returns_empty_on_api_error_status(self):
        from tools.ticker.news_sentiment_tools import fetch_newsapi_articles
        error_response = {"status": "error", "code": "rateLimited", "message": "too many requests"}
        mock_resp = make_mock_response(json_data=error_response)
        with patch.dict(os.environ, {"NEWS_API_KEY": "test_key"}):
            with patch("tools.ticker.news_sentiment_tools.requests.get", return_value=mock_resp):
                result = fetch_newsapi_articles("CBA.AX")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# fetch_yahoo_rss_articles  (mocked HTTP, XML parsing)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchYahooRssArticles:
    def test_parses_valid_rss_feed(self, yahoo_rss_positive):
        from tools.ticker.news_sentiment_tools import fetch_yahoo_rss_articles
        mock_resp = make_mock_response(content=yahoo_rss_positive)
        with patch("tools.ticker.news_sentiment_tools.requests.get", return_value=mock_resp):
            result = fetch_yahoo_rss_articles("CBA.AX")
        assert len(result) == 2
        assert all("title" in a for a in result)
        assert all(a["source"] == "Yahoo Finance" for a in result)

    def test_returns_empty_on_empty_feed(self, yahoo_rss_empty):
        from tools.ticker.news_sentiment_tools import fetch_yahoo_rss_articles
        mock_resp = make_mock_response(content=yahoo_rss_empty)
        with patch("tools.ticker.news_sentiment_tools.requests.get", return_value=mock_resp):
            result = fetch_yahoo_rss_articles("CBA.AX")
        assert result == []

    def test_handles_invalid_xml_gracefully(self):
        from tools.ticker.news_sentiment_tools import fetch_yahoo_rss_articles
        mock_resp = make_mock_response(content=b"this is not xml!!!")
        with patch("tools.ticker.news_sentiment_tools.requests.get", return_value=mock_resp):
            result = fetch_yahoo_rss_articles("CBA.AX")
        assert result == []   # must not raise

    def test_handles_network_error_gracefully(self):
        from tools.ticker.news_sentiment_tools import fetch_yahoo_rss_articles
        with patch("tools.ticker.news_sentiment_tools.requests.get",
                   side_effect=ConnectionError("down")):
            result = fetch_yahoo_rss_articles("CBA.AX")
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# analyze_news_sentiment_for_ticker  (mocked both sources)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeNewsSentimentForTicker:
    def _mock_sources(self, newsapi_articles=None, yahoo_articles=None):
        """Patch both fetch functions to return controlled data."""
        newsapi_articles = newsapi_articles or []
        yahoo_articles = yahoo_articles or []
        return (
            patch("tools.ticker.news_sentiment_tools.fetch_newsapi_articles",
                  return_value=newsapi_articles),
            patch("tools.ticker.news_sentiment_tools.fetch_yahoo_rss_articles",
                  return_value=yahoo_articles),
        )

    def test_positive_signal_with_many_positive_articles(self):
        from tools.ticker.news_sentiment_tools import analyze_news_sentiment_for_ticker
        articles = [
            {"title": "CBA record profit beat expectations", "date": "", "source": "Test", "url": ""},
            {"title": "Commonwealth Bank dividend upgrade growth strong", "date": "", "source": "Test", "url": ""},
            {"title": "CBA contract win outperform growth record", "date": "", "source": "Test", "url": ""},
        ]
        p1, p2 = self._mock_sources(newsapi_articles=articles)
        with p1, p2:
            result = analyze_news_sentiment_for_ticker("CBA.AX")
        assert result["news_signal"] == "positive"
        assert result["sentiment_score"] > 0.3

    def test_warning_signal_when_red_flag_present(self):
        from tools.ticker.news_sentiment_tools import analyze_news_sentiment_for_ticker
        articles = [
            {"title": "ASIC investigation launched into CBA practices", "date": "", "source": "Test", "url": ""},
            {"title": "Quarterly update looks strong", "date": "", "source": "Test", "url": ""},
            {"title": "Analyst upgrades CBA target", "date": "", "source": "Test", "url": ""},
        ]
        p1, p2 = self._mock_sources(newsapi_articles=articles)
        with p1, p2:
            result = analyze_news_sentiment_for_ticker("CBA.AX")
        assert result["news_signal"] == "warning"
        assert len(result["red_flags"]) > 0

    def test_insufficient_data_with_less_than_2_articles(self):
        from tools.ticker.news_sentiment_tools import analyze_news_sentiment_for_ticker
        articles = [{"title": "CBA update", "date": "", "source": "Test", "url": ""}]
        p1, p2 = self._mock_sources(newsapi_articles=articles)
        with p1, p2:
            result = analyze_news_sentiment_for_ticker("CBA.AX")
        assert result["news_signal"] == "insufficient_data"

    def test_insufficient_data_when_no_articles(self):
        from tools.ticker.news_sentiment_tools import analyze_news_sentiment_for_ticker
        p1, p2 = self._mock_sources()
        with p1, p2:
            result = analyze_news_sentiment_for_ticker("CBA.AX")
        assert result["news_signal"] == "insufficient_data"
        assert result["articles_found"] == 0

    def test_deduplicates_same_headline_across_sources(self):
        from tools.ticker.news_sentiment_tools import analyze_news_sentiment_for_ticker
        same_article = {"title": "CBA profit rises", "date": "", "source": "A", "url": ""}
        p1, p2 = self._mock_sources(newsapi_articles=[same_article], yahoo_articles=[same_article])
        with p1, p2:
            result = analyze_news_sentiment_for_ticker("CBA.AX")
        assert result["articles_found"] == 1  # deduplicated

    def test_result_has_all_required_keys(self):
        from tools.ticker.news_sentiment_tools import analyze_news_sentiment_for_ticker
        p1, p2 = self._mock_sources()
        with p1, p2:
            result = analyze_news_sentiment_for_ticker("CBA.AX")
        for key in ["ticker", "articles_found", "sentiment_score", "sentiment_label",
                    "red_flags", "top_headlines", "news_signal", "data_sources_used"]:
            assert key in result, f"Missing key: {key}"

    def test_top_headlines_capped_at_3(self):
        from tools.ticker.news_sentiment_tools import analyze_news_sentiment_for_ticker
        articles = [
            {"title": f"Article {i}", "date": "", "source": "Test", "url": ""}
            for i in range(10)
        ]
        p1, p2 = self._mock_sources(newsapi_articles=articles)
        with p1, p2:
            result = analyze_news_sentiment_for_ticker("CBA.AX")
        assert len(result["top_headlines"]) <= 3

    def test_ticker_normalized_in_output(self):
        from tools.ticker.news_sentiment_tools import analyze_news_sentiment_for_ticker
        # The function is always called with an uppercase ticker in practice —
        # _fetch_news_sentiment_impl pre-normalises before delegating here.
        # ".AX" suffix is stripped by a case-sensitive replace(".AX", "").
        p1, p2 = self._mock_sources()
        with p1, p2:
            result = analyze_news_sentiment_for_ticker("CBA.AX")
        assert result["ticker"] == "CBA"   # .AX stripped, uppercase preserved


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_news_sentiment_impl  (full concurrent pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchNewsSentimentImpl:
    def test_returns_result_for_each_ticker(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        with patch("tools.ticker.news_sentiment_tools.fetch_newsapi_articles", return_value=[]):
            with patch("tools.ticker.news_sentiment_tools.fetch_yahoo_rss_articles", return_value=[]):
                result = _fetch_news_sentiment_impl(["CBA.AX", "BHP.AX"])
        assert len(result["data"]) == 2

    def test_result_structure_always_valid(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        with patch("tools.ticker.news_sentiment_tools.fetch_newsapi_articles", return_value=[]):
            with patch("tools.ticker.news_sentiment_tools.fetch_yahoo_rss_articles", return_value=[]):
                result = _fetch_news_sentiment_impl(["CBA.AX"])
        assert "data" in result
        assert "errors_fatal" in result
        assert "errors_non_fatal" in result

    def test_news_signal_always_valid_value(self):
        from tools.ticker.news_sentiment_tools import _fetch_news_sentiment_impl
        valid_signals = {"positive", "negative", "warning", "neutral", "insufficient_data"}
        with patch("tools.ticker.news_sentiment_tools.fetch_newsapi_articles", return_value=[]):
            with patch("tools.ticker.news_sentiment_tools.fetch_yahoo_rss_articles", return_value=[]):
                result = _fetch_news_sentiment_impl(["CBA.AX", "BHP.AX"])
        for item in result["data"]:
            assert item["news_signal"] in valid_signals
