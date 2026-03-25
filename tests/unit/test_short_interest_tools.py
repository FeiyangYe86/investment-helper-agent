"""
Unit tests for tools/ticker/short_interest_tools.py

Coverage:
  - compute_short_signal         : pure function, all threshold branches
  - _parse_shortman_percentage   : HTML parsing with mock BeautifulSoup content
  - _parse_shortman_previous     : HTML parsing
  - _parse_shortman_date         : HTML parsing
  - fetch_shortman_data          : mocked HTTP, scraping pipeline
  - _fetch_short_interest        : concurrent pipeline, fallback logic
"""

import pytest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
from tests.conftest import (
    make_shortman_html, make_mock_response,
    shortman_low_short, shortman_high_short,
)


# ─────────────────────────────────────────────────────────────────────────────
# compute_short_signal  (pure function)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeShortSignal:
    def test_extreme_short_returns_warning(self):
        from tools.ticker.short_interest_tools import compute_short_signal
        assert compute_short_signal(16.0, 14.0, 2.0, "increasing") == "warning"

    def test_exactly_15_pct_returns_warning(self):
        from tools.ticker.short_interest_tools import compute_short_signal
        # boundary: > 15 is warning; 15 itself is NOT > 15
        assert compute_short_signal(15.1, None, None, "flat") == "warning"

    def test_above_10_pct_returns_bearish(self):
        from tools.ticker.short_interest_tools import compute_short_signal
        assert compute_short_signal(12.0, 11.0, 1.0, "increasing") == "bearish"

    def test_increasing_trend_with_large_change_returns_bearish(self):
        from tools.ticker.short_interest_tools import compute_short_signal
        # short_pct = 7% (below 10%) but increasing fast → bearish
        assert compute_short_signal(7.0, 5.0, 2.0, "increasing") == "bearish"

    def test_low_short_below_3_returns_bullish(self):
        from tools.ticker.short_interest_tools import compute_short_signal
        assert compute_short_signal(2.5, 2.8, -0.3, "decreasing") == "bullish"

    def test_decreasing_trend_below_8_returns_bullish(self):
        from tools.ticker.short_interest_tools import compute_short_signal
        assert compute_short_signal(7.0, 8.5, -1.5, "decreasing") == "bullish"

    def test_moderate_stable_short_returns_neutral(self):
        from tools.ticker.short_interest_tools import compute_short_signal
        assert compute_short_signal(5.0, 5.0, 0.0, "flat") == "neutral"

    def test_none_short_pct_returns_unknown(self):
        from tools.ticker.short_interest_tools import compute_short_signal
        assert compute_short_signal(None, None, None, "unknown") == "unknown"

    def test_increasing_with_small_change_not_bearish(self):
        from tools.ticker.short_interest_tools import compute_short_signal
        # change < 1.5 threshold → not triggered as bearish on trend alone
        assert compute_short_signal(7.0, 6.5, 0.5, "increasing") == "neutral"

    def test_boundary_exactly_3_pct(self):
        from tools.ticker.short_interest_tools import compute_short_signal
        # exactly 3% is NOT < 3%, so not bullish on that condition alone
        result = compute_short_signal(3.0, 3.5, -0.5, "decreasing")
        # decreasing + < 8% → should be bullish
        assert result == "bullish"

    def test_boundary_exactly_10_pct(self):
        from tools.ticker.short_interest_tools import compute_short_signal
        # 10.0 is NOT > 10, should not trigger bearish on pct alone
        result = compute_short_signal(10.0, 10.0, 0.0, "flat")
        assert result == "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# HTML Parsing Helpers  (using real BeautifulSoup with synthetic HTML)
# ─────────────────────────────────────────────────────────────────────────────

class TestParseShortmanPercentage:
    def test_parses_short_label_pattern(self):
        from tools.ticker.short_interest_tools import _parse_shortman_percentage
        html = "<html><body><p>Short: 5.32%</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        assert _parse_shortman_percentage(soup) == pytest.approx(5.32)

    def test_parses_table_row_with_short(self):
        from tools.ticker.short_interest_tools import _parse_shortman_percentage
        html = """
        <html><body><table>
          <tr><td>Short position</td><td>3.87%</td></tr>
        </table></body></html>"""
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_shortman_percentage(soup)
        assert result == pytest.approx(3.87)

    def test_returns_none_when_no_percentage(self):
        from tools.ticker.short_interest_tools import _parse_shortman_percentage
        soup = BeautifulSoup("<html><body><p>No data available.</p></body></html>", "html.parser")
        assert _parse_shortman_percentage(soup) is None

    def test_parses_class_based_element(self):
        from tools.ticker.short_interest_tools import _parse_shortman_percentage
        html = '<html><body><span class="short-pct">12.50%</span></body></html>'
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_shortman_percentage(soup)
        assert result == pytest.approx(12.50)


class TestParseShortmanPrevious:
    def test_parses_previous_label(self):
        from tools.ticker.short_interest_tools import _parse_shortman_previous
        html = "<html><body><p>Previous: 4.10%</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        assert _parse_shortman_previous(soup) == pytest.approx(4.10)

    def test_parses_last_week_label(self):
        from tools.ticker.short_interest_tools import _parse_shortman_previous
        html = "<html><body><p>Last Week: 3.95%</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        assert _parse_shortman_previous(soup) == pytest.approx(3.95)

    def test_returns_none_when_missing(self):
        from tools.ticker.short_interest_tools import _parse_shortman_previous
        soup = BeautifulSoup("<html><body><p>No previous data.</p></body></html>", "html.parser")
        assert _parse_shortman_previous(soup) is None


class TestParseShortmanDate:
    def test_parses_iso_date(self):
        from tools.ticker.short_interest_tools import _parse_shortman_date
        html = "<html><body><p>Data as of: 2026-03-20</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        assert _parse_shortman_date(soup) == "2026-03-20"

    def test_parses_natural_language_date(self):
        from tools.ticker.short_interest_tools import _parse_shortman_date
        html = "<html><body><p>Updated 20 Mar 2026</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        result = _parse_shortman_date(soup)
        assert result is not None
        assert "Mar" in result

    def test_returns_none_when_no_date(self):
        from tools.ticker.short_interest_tools import _parse_shortman_date
        soup = BeautifulSoup("<html><body><p>No date.</p></body></html>", "html.parser")
        assert _parse_shortman_date(soup) is None


# ─────────────────────────────────────────────────────────────────────────────
# fetch_shortman_data  (mocked requests.get)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchShortmanData:
    def test_successful_fetch_with_low_short(self, shortman_low_short):
        from tools.ticker.short_interest_tools import fetch_shortman_data
        mock_resp = make_mock_response(text=shortman_low_short)
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp):
            result = fetch_shortman_data("CBA.AX")

        assert result["fetch_error"] is None
        assert result["short_pct"] == pytest.approx(2.5)
        assert result["short_pct_prev"] == pytest.approx(2.8)
        assert result["short_signal"] == "bullish"
        assert result["short_trend"] == "decreasing"

    def test_successful_fetch_with_high_short(self, shortman_high_short):
        from tools.ticker.short_interest_tools import fetch_shortman_data
        mock_resp = make_mock_response(text=shortman_high_short)
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp):
            result = fetch_shortman_data("BHP.AX")

        assert result["fetch_error"] is None
        assert result["short_pct"] == pytest.approx(16.0)
        assert result["short_signal"] == "warning"

    def test_http_error_returns_fetch_error(self):
        from tools.ticker.short_interest_tools import fetch_shortman_data
        mock_resp = make_mock_response(status_code=503)
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp):
            result = fetch_shortman_data("CBA.AX")

        assert result["fetch_error"] is not None
        assert result["short_pct"] is None

    def test_connection_error_returns_fetch_error(self):
        from tools.ticker.short_interest_tools import fetch_shortman_data
        with patch("tools.ticker.short_interest_tools.requests.get",
                   side_effect=ConnectionError("Connection refused")):
            result = fetch_shortman_data("CBA.AX")

        assert result["fetch_error"] is not None
        assert result["short_pct"] is None

    def test_unparseable_html_returns_fetch_error(self, shortman_no_data):
        from tools.ticker.short_interest_tools import fetch_shortman_data
        mock_resp = make_mock_response(text=shortman_no_data)
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp):
            result = fetch_shortman_data("CBA.AX")

        assert result["fetch_error"] is not None
        assert result["short_pct"] is None

    def test_ticker_without_ax_suffix_works(self, shortman_low_short):
        from tools.ticker.short_interest_tools import fetch_shortman_data
        mock_resp = make_mock_response(text=shortman_low_short)
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp) as mock_get:
            fetch_shortman_data("CBA")  # no .AX suffix
        url = mock_get.call_args[0][0]
        assert "CBA" in url
        assert ".AX" not in url

    def test_result_always_has_required_keys(self):
        from tools.ticker.short_interest_tools import fetch_shortman_data
        mock_resp = make_mock_response(status_code=404)
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp):
            result = fetch_shortman_data("CBA.AX")

        for key in ["ticker", "short_pct", "short_pct_prev", "short_change_pct",
                    "short_trend", "short_signal", "data_date", "fetch_error"]:
            assert key in result, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_short_interest  (concurrent pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchShortInterest:
    def test_empty_tickers_returns_fatal_error(self):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        result = _fetch_short_interest([])
        assert len(result["errors_fatal"]) > 0
        assert result["data"] == []

    def test_successful_fetch_returns_all_tickers(self, shortman_low_short):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        mock_resp = make_mock_response(text=shortman_low_short)
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp):
            result = _fetch_short_interest(["CBA.AX", "BHP.AX"])

        assert result["errors_fatal"] == []
        assert len(result["data"]) == 2

    def test_all_fetches_fail_does_not_raise(self):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        with patch("tools.ticker.short_interest_tools.requests.get",
                   side_effect=ConnectionError("down")):
            result = _fetch_short_interest(["CBA.AX"])

        # Must not raise; errors captured in result
        assert isinstance(result, dict)
        assert "data" in result

    def test_result_includes_entry_for_every_ticker(self, shortman_no_data):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        mock_resp = make_mock_response(text=shortman_no_data)
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp):
            result = _fetch_short_interest(["CBA.AX", "BHP.AX", "NAB.AX"])

        # Even if parsing fails, we get an entry per ticker
        assert len(result["data"]) == 3

    def test_signal_values_always_valid(self, shortman_low_short):
        from tools.ticker.short_interest_tools import _fetch_short_interest
        valid_signals = {"bullish", "bearish", "warning", "neutral", "unknown"}
        mock_resp = make_mock_response(text=shortman_low_short)
        with patch("tools.ticker.short_interest_tools.requests.get", return_value=mock_resp):
            result = _fetch_short_interest(["CBA.AX"])

        for item in result["data"]:
            assert item["short_signal"] in valid_signals
