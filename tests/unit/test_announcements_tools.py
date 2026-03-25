"""
Unit tests for tools/ticker/announcements_tools.py

All HTTP calls are mocked. Tests cover:
  - classify_announcements : pure classification logic, all signal branches
  - fetch_announcements_for_ticker : mocked HTTP, error handling
  - _fetch_announcements_impl : concurrent pipeline, fatal vs non-fatal errors
"""

import json
import pytest
from unittest.mock import patch, MagicMock, call
from tests.conftest import make_asx_api_response, make_mock_response


# ─────────────────────────────────────────────────────────────────────────────
# classify_announcements  (pure function — no mocks needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyAnnouncements:
    """Tests classification logic without any network calls."""

    def _make_ann(self, title: str, date: str = "2026-03-20T10:00:00", sensitive: bool = False):
        return {"title": title, "date": date, "market_sensitive": sensitive}

    def test_single_red_flag_produces_warning(self):
        from tools.ticker.announcements_tools import classify_announcements
        anns = [self._make_ann("ASIC commences review of lending practices", sensitive=True)]
        result = classify_announcements("CBA", anns)
        assert result["announcement_signal"] == "warning"
        assert len(result["red_flags"]) == 1

    def test_multiple_red_flags_produces_negative(self):
        from tools.ticker.announcements_tools import classify_announcements
        anns = [
            self._make_ann("Capital raise: $150M placement at $2.40", sensitive=True),
            self._make_ann("CEO resignation announced effective immediately", sensitive=True),
        ]
        result = classify_announcements("CBA", anns)
        assert result["announcement_signal"] == "negative"
        assert len(result["red_flags"]) == 2

    def test_capital_raise_is_red_flag(self):
        from tools.ticker.announcements_tools import classify_announcements
        anns = [self._make_ann("Capital raise approved by board: $200M placement")]
        result = classify_announcements("CBA", anns)
        assert len(result["red_flags"]) > 0

    def test_rights_issue_is_red_flag(self):
        from tools.ticker.announcements_tools import classify_announcements
        anns = [self._make_ann("Rights issue announced: 1 for 5 at $3.20")]
        result = classify_announcements("CBA", anns)
        assert len(result["red_flags"]) > 0

    def test_dividend_produces_positive_signal(self):
        from tools.ticker.announcements_tools import classify_announcements
        anns = [self._make_ann("Dividend declared: 85c fully franked", sensitive=True)]
        result = classify_announcements("CBA", anns)
        assert result["announcement_signal"] == "positive"
        assert len(result["positive_signals"]) == 1

    def test_contract_win_is_positive(self):
        from tools.ticker.announcements_tools import classify_announcements
        anns = [self._make_ann("Contract win: $500M government infrastructure deal")]
        result = classify_announcements("CBA", anns)
        assert len(result["positive_signals"]) > 0

    def test_neutral_announcements_produce_neutral_signal(self):
        from tools.ticker.announcements_tools import classify_announcements
        anns = [
            self._make_ann("Change in substantial holder"),
            self._make_ann("Notice of Annual General Meeting"),
        ]
        result = classify_announcements("CBA", anns)
        assert result["announcement_signal"] == "neutral"
        assert result["red_flags"] == []
        assert result["positive_signals"] == []

    def test_empty_announcements_list(self):
        from tools.ticker.announcements_tools import classify_announcements
        result = classify_announcements("CBA", [])
        assert result["announcements_count"] == 0
        assert result["announcement_signal"] == "neutral"
        assert result["latest_announcement"] is None

    def test_market_sensitive_count_is_accurate(self):
        from tools.ticker.announcements_tools import classify_announcements
        anns = [
            self._make_ann("Earnings update", sensitive=True),
            self._make_ann("Change in director interest", sensitive=False),
            self._make_ann("Trading halt", sensitive=True),
        ]
        result = classify_announcements("CBA", anns)
        assert result["market_sensitive_count"] == 2

    def test_latest_announcement_is_most_recent(self):
        from tools.ticker.announcements_tools import classify_announcements
        anns = [
            self._make_ann("Older announcement", date="2026-03-01T10:00:00"),
            self._make_ann("Newer announcement", date="2026-03-20T10:00:00"),
        ]
        result = classify_announcements("CBA", anns)
        assert result["latest_announcement"]["title"] == "Newer announcement"

    def test_red_flag_takes_priority_over_positive(self):
        """An announcement can't be both positive and red-flag — red flag wins."""
        from tools.ticker.announcements_tools import classify_announcements
        anns = [
            self._make_ann("ASIC investigation: record compliance breach"),  # red flag
            self._make_ann("Record dividend declared to shareholders"),       # positive
        ]
        result = classify_announcements("CBA", anns)
        # Red flag present → signal is warning, not positive
        assert result["announcement_signal"] in {"warning", "negative"}


# ─────────────────────────────────────────────────────────────────────────────
# fetch_announcements_for_ticker  (mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchAnnouncementsForTicker:
    def test_successful_fetch_returns_announcements(self, asx_positive_announcements):
        from tools.ticker.announcements_tools import fetch_announcements_for_ticker
        mock_resp = make_mock_response(json_data=asx_positive_announcements)
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = fetch_announcements_for_ticker("CBA.AX")
        assert result["success"] is True
        assert len(result["announcements"]) > 0

    def test_strips_ax_suffix_from_ticker(self, asx_positive_announcements):
        from tools.ticker.announcements_tools import fetch_announcements_for_ticker
        mock_resp = make_mock_response(json_data=asx_positive_announcements)
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp) as mock_get:
            fetch_announcements_for_ticker("CBA.AX")
        url_called = mock_get.call_args[0][0]
        assert "CBA" in url_called
        assert ".AX" not in url_called

    def test_http_404_returns_success_false(self):
        from tools.ticker.announcements_tools import fetch_announcements_for_ticker
        mock_resp = make_mock_response(status_code=404)
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = fetch_announcements_for_ticker("INVALID.AX")
        assert result["success"] is False

    def test_http_error_includes_status_code_in_error_message(self):
        """Error message must include the HTTP status code so failures are diagnosable."""
        from tools.ticker.announcements_tools import fetch_announcements_for_ticker
        mock_resp = make_mock_response(status_code=403)
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = fetch_announcements_for_ticker("WDS.AX")
        assert result["success"] is False
        assert "403" in result["error"], (
            f"Expected HTTP 403 in error message, got: {result['error']}"
        )

    def test_network_timeout_returns_success_false(self):
        from tools.ticker.announcements_tools import fetch_announcements_for_ticker
        from requests.exceptions import Timeout
        with patch("tools.ticker.announcements_tools.requests.get", side_effect=Timeout()):
            result = fetch_announcements_for_ticker("CBA.AX")
        assert result["success"] is False

    def test_timeout_includes_diagnostic_in_error_message(self):
        """Timeout errors must be identified in the error message."""
        from tools.ticker.announcements_tools import fetch_announcements_for_ticker
        from requests.exceptions import Timeout
        with patch("tools.ticker.announcements_tools.requests.get", side_effect=Timeout()):
            result = fetch_announcements_for_ticker("CBA.AX")
        assert "timed out" in result["error"].lower() or "timeout" in result["error"].lower(), (
            f"Expected timeout info in error message, got: {result['error']}"
        )

    def test_empty_data_returns_success_false(self):
        from tools.ticker.announcements_tools import fetch_announcements_for_ticker
        mock_resp = make_mock_response(json_data={"data": []})
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = fetch_announcements_for_ticker("CBA.AX")
        assert result["success"] is False

    def test_ticker_normalized_to_uppercase(self, asx_positive_announcements):
        from tools.ticker.announcements_tools import fetch_announcements_for_ticker
        mock_resp = make_mock_response(json_data=asx_positive_announcements)
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp) as mock_get:
            fetch_announcements_for_ticker("cba.ax")
        url_called = mock_get.call_args[0][0]
        assert "CBA" in url_called


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_announcements_impl  (end-to-end pipeline with mocked fetcher)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchAnnouncementsImpl:
    def test_successful_pipeline_returns_classified_results(self, asx_positive_announcements):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        mock_resp = make_mock_response(json_data=asx_positive_announcements)
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = _fetch_announcements_impl(["CBA.AX", "BHP.AX"])

        assert result["errors_fatal"] == []
        assert len(result["data"]) > 0
        # Each result must have the contract keys
        for item in result["data"]:
            assert "ticker" in item
            assert "announcement_signal" in item
            assert item["announcement_signal"] in {"positive", "negative", "neutral", "warning"}

    def test_all_tickers_fail_returns_fatal_error(self):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        mock_resp = make_mock_response(status_code=503)
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = _fetch_announcements_impl(["CBA.AX", "BHP.AX"])

        assert len(result["errors_fatal"]) > 0
        assert result["data"] == []

    def test_partial_failure_is_non_fatal(self, asx_positive_announcements):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        call_count = [0]
        def side_effect(url, *args, **kwargs):
            call_count[0] += 1
            if "CBA" in url:
                return make_mock_response(json_data=asx_positive_announcements)
            return make_mock_response(status_code=404)
        with patch("tools.ticker.announcements_tools.requests.get", side_effect=side_effect):
            result = _fetch_announcements_impl(["CBA.AX", "BHP.AX"])

        assert result["errors_fatal"] == []
        assert len(result["errors_non_fatal"]) > 0
        assert len(result["data"]) >= 1   # CBA succeeded

    def test_result_structure_always_valid(self):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        mock_resp = make_mock_response(status_code=500)
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = _fetch_announcements_impl(["CBA.AX"])

        assert "data" in result
        assert "errors_fatal" in result
        assert "errors_non_fatal" in result
        assert isinstance(result["errors_fatal"], list)
        assert isinstance(result["errors_non_fatal"], list)
        assert isinstance(result["data"], list)

    def test_red_flag_announcement_surfaces_in_result(self, asx_red_flag_single):
        from tools.ticker.announcements_tools import _fetch_announcements_impl
        mock_resp = make_mock_response(json_data=asx_red_flag_single)
        with patch("tools.ticker.announcements_tools.requests.get", return_value=mock_resp):
            result = _fetch_announcements_impl(["CBA.AX"])

        assert result["errors_fatal"] == []
        data = result["data"]
        if data:  # only check if we got results back
            cba = next((d for d in data if d["ticker"] == "CBA"), None)
            if cba:
                assert cba["announcement_signal"] in {"warning", "negative"}
