"""
Unit tests for tools/ticker/macro_tools.py

Tests are 100% offline — all HTTP calls are mocked at the requests.get level.

Coverage:
  - _compute_change_pct   : pure function, no mocks
  - _get_trend            : pure function, no mocks
  - _compute_macro_signal : pure function, all signal branches
  - _fetch_fred_series    : mocked HTTP, verify parsing and error handling
  - _fetch_macro_context  : mocked _fetch_fred_series, end-to-end pipeline
"""

import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import make_fred_observations, make_mock_response


# ─────────────────────────────────────────────────────────────────────────────
# _compute_change_pct  (pure function — no mocks needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeChangePct:
    def test_basic_positive_change(self):
        from tools.ticker.macro_tools import _compute_change_pct
        # newest=110, oldest (period=1)=100 → +10%
        obs = make_fred_observations([110.0, 100.0])
        result = _compute_change_pct(obs, periods=1)
        assert result == pytest.approx(0.10, rel=1e-3)

    def test_basic_negative_change(self):
        from tools.ticker.macro_tools import _compute_change_pct
        obs = make_fred_observations([90.0, 100.0])
        result = _compute_change_pct(obs, periods=1)
        assert result == pytest.approx(-0.10, rel=1e-3)

    def test_returns_none_when_insufficient_data(self):
        from tools.ticker.macro_tools import _compute_change_pct
        obs = make_fred_observations([1.0, 2.0])
        assert _compute_change_pct(obs, periods=5) is None

    def test_returns_none_when_empty(self):
        from tools.ticker.macro_tools import _compute_change_pct
        assert _compute_change_pct([], periods=1) is None

    def test_returns_none_when_oldest_is_zero(self):
        from tools.ticker.macro_tools import _compute_change_pct
        obs = make_fred_observations([1.0, 0.0])
        assert _compute_change_pct(obs, periods=1) is None

    def test_handles_non_numeric_values_gracefully(self):
        from tools.ticker.macro_tools import _compute_change_pct
        obs = [{"date": "2026-01-02", "value": "."}, {"date": "2026-01-01", "value": "100.0"}]
        assert _compute_change_pct(obs, periods=1) is None

    def test_large_period(self):
        from tools.ticker.macro_tools import _compute_change_pct
        obs = make_fred_observations([0.65] * 90)
        # 63-period change on flat series → 0%
        result = _compute_change_pct(obs, periods=63)
        assert result == pytest.approx(0.0, abs=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# _get_trend  (pure function)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTrend:
    def test_rising(self):
        from tools.ticker.macro_tools import _get_trend
        assert _get_trend(0.05) == "rising"

    def test_falling(self):
        from tools.ticker.macro_tools import _get_trend
        assert _get_trend(-0.05) == "falling"

    def test_flat_within_threshold(self):
        from tools.ticker.macro_tools import _get_trend
        assert _get_trend(0.005) == "flat"   # < default 1% threshold
        assert _get_trend(-0.005) == "flat"

    def test_exactly_at_threshold_is_flat(self):
        from tools.ticker.macro_tools import _get_trend
        assert _get_trend(0.01) == "flat"
        assert _get_trend(-0.01) == "flat"

    def test_none_returns_flat(self):
        from tools.ticker.macro_tools import _get_trend
        assert _get_trend(None) == "flat"

    def test_custom_threshold(self):
        from tools.ticker.macro_tools import _get_trend
        # 2% threshold — 1.5% should still be flat
        assert _get_trend(0.015, threshold=0.02) == "flat"
        assert _get_trend(0.025, threshold=0.02) == "rising"


# ─────────────────────────────────────────────────────────────────────────────
# _compute_macro_signal  (pure function — tests all branches)
# ─────────────────────────────────────────────────────────────────────────────

def make_macro_context(**overrides):
    """Build a neutral MacroContext, apply overrides."""
    base = {
        "audusd_current": 0.65,
        "audusd_1m_change_pct": 0.0,
        "audusd_3m_change_pct": 0.0,
        "audusd_trend": "flat",
        "rba_cash_rate": 4.35,
        "rba_rate_direction": "flat",
        "gold_1m_change_pct": 0.0,
        "gold_trend": "flat",
        "oil_1m_change_pct": 0.0,
        "oil_trend": "flat",
        "iron_ore_trend": "flat",
        "macro_signal": "neutral",
        "macro_notes": [],
    }
    base.update(overrides)
    return base


class TestComputeMacroSignal:
    def test_fully_favorable(self):
        from tools.ticker.macro_tools import _compute_macro_signal
        ctx = make_macro_context(
            audusd_trend="weakening",       # +2 favorable
            rba_rate_direction="falling",   # +1 favorable
            gold_trend="rising",
            oil_trend="rising",
            iron_ore_trend="rising",        # 3/3 rising → +1 favorable
        )
        signal, notes = _compute_macro_signal(ctx)
        assert signal == "favorable"
        assert any("weakening" in n for n in notes)
        assert any("falling" in n for n in notes)

    def test_fully_unfavorable(self):
        from tools.ticker.macro_tools import _compute_macro_signal
        ctx = make_macro_context(
            audusd_trend="strengthening",   # +2 unfavorable
            rba_rate_direction="rising",    # +2 unfavorable
            gold_trend="falling",
            oil_trend="falling",
            iron_ore_trend="falling",       # 3/3 falling → +1 unfavorable
        )
        signal, notes = _compute_macro_signal(ctx)
        assert signal == "unfavorable"
        assert any("strengthening" in n for n in notes)
        assert any("rising" in n for n in notes)

    def test_mixed_returns_neutral(self):
        from tools.ticker.macro_tools import _compute_macro_signal
        # favorable=2 (below >=3 threshold), unfavorable=0 (below >=2 threshold) → neutral
        # Note: the thresholds are asymmetric by design — unfavorable fires at >=2,
        # favorable only at >=3. So "weakening AUD + rising RBA" actually resolves
        # as "unfavorable" (unfavorable hits 2). Use flat RBA to stay truly neutral.
        ctx = make_macro_context(
            audusd_trend="weakening",       # +2 favorable → favorable = 2
            rba_rate_direction="flat",      # +0 to either → unfavorable = 0
            gold_trend="flat",
            oil_trend="flat",
            iron_ore_trend="flat",          # all flat → no commodity signal
        )
        signal, notes = _compute_macro_signal(ctx)
        assert signal == "neutral"

    def test_notes_always_populated(self):
        from tools.ticker.macro_tools import _compute_macro_signal
        ctx = make_macro_context()  # all flat
        signal, notes = _compute_macro_signal(ctx)
        assert len(notes) >= 3   # one note per factor group

    def test_two_of_three_commodities_rising(self):
        from tools.ticker.macro_tools import _compute_macro_signal
        ctx = make_macro_context(
            audusd_trend="weakening",
            rba_rate_direction="flat",
            gold_trend="rising",
            oil_trend="rising",
            iron_ore_trend="flat",          # 2/3 → still favorable commodity signal
        )
        signal, notes = _compute_macro_signal(ctx)
        assert any("2/3" in n for n in notes)


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_fred_series  (mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchFredSeries:
    def test_returns_observations_on_success(self):
        from tools.ticker.macro_tools import _fetch_fred_series
        mock_response = make_mock_response(json_data={
            "observations": [
                {"date": "2026-03-20", "value": "0.6500"},
                {"date": "2026-03-19", "value": "0.6480"},
            ]
        })
        with patch("tools.ticker.macro_tools.requests.get", return_value=mock_response):
            result = _fetch_fred_series("DEXUSAL")
        assert len(result) == 2
        assert result[0]["value"] == "0.6500"

    def test_returns_empty_list_on_http_error(self):
        from tools.ticker.macro_tools import _fetch_fred_series
        mock_response = make_mock_response(status_code=429)
        with patch("tools.ticker.macro_tools.requests.get", return_value=mock_response):
            result = _fetch_fred_series("DEXUSAL")
        assert result == []

    def test_returns_empty_list_on_connection_error(self):
        from tools.ticker.macro_tools import _fetch_fred_series
        with patch("tools.ticker.macro_tools.requests.get", side_effect=ConnectionError("timeout")):
            result = _fetch_fred_series("DEXUSAL")
        assert result == []

    def test_includes_api_key_in_params(self):
        from tools.ticker.macro_tools import _fetch_fred_series
        mock_response = make_mock_response(json_data={"observations": []})
        with patch("tools.ticker.macro_tools.requests.get", return_value=mock_response) as mock_get:
            _fetch_fred_series("DEXUSAL", api_key="test_key_123")
        call_kwargs = mock_get.call_args
        params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs[0][1]
        assert "api_key" in str(call_kwargs)

    def test_omits_api_key_when_none(self):
        from tools.ticker.macro_tools import _fetch_fred_series
        mock_response = make_mock_response(json_data={"observations": []})
        with patch("tools.ticker.macro_tools.requests.get", return_value=mock_response) as mock_get:
            _fetch_fred_series("DEXUSAL", api_key=None)
        assert "api_key" not in str(mock_get.call_args)


# ─────────────────────────────────────────────────────────────────────────────
# _fetch_macro_context  (mocked _fetch_fred_series — end-to-end)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchMacroContext:
    def _patch_fred(self, audusd, rba, gold, oil, iron):
        """Return a side_effect list for _fetch_fred_series calls in order."""
        call_map = {
            "DEXUSAL": audusd,
            "IRSTCB01AUM156N": rba,
            "GOLDAMGBD228NLBM": gold,
            "DCOILWTICO": oil,
            "PIORECRUSDM": iron,
        }
        def side_effect(series_id, *args, **kwargs):
            return call_map.get(series_id, [])
        return side_effect

    def test_favorable_macro_environment(self, fred_audusd_falling, fred_rba_falling,
                                          fred_commodity_rising):
        from tools.ticker.macro_tools import _fetch_macro_context
        side_effect = self._patch_fred(
            audusd=fred_audusd_falling,
            rba=fred_rba_falling,
            gold=fred_commodity_rising,
            oil=fred_commodity_rising,
            iron=fred_commodity_rising[:4],
        )
        with patch("tools.ticker.macro_tools._fetch_fred_series", side_effect=side_effect):
            result = _fetch_macro_context()

        assert result["errors_fatal"] == []
        macro = result["data"]
        assert macro["audusd_trend"] == "weakening"
        assert macro["rba_rate_direction"] == "falling"
        assert macro["macro_signal"] == "favorable"

    def test_unfavorable_macro_environment(self, fred_audusd_rising, fred_rba_rising,
                                            fred_commodity_falling):
        from tools.ticker.macro_tools import _fetch_macro_context
        side_effect = self._patch_fred(
            audusd=fred_audusd_rising,
            rba=fred_rba_rising,
            gold=fred_commodity_falling,
            oil=fred_commodity_falling,
            iron=fred_commodity_falling[:4],
        )
        with patch("tools.ticker.macro_tools._fetch_fred_series", side_effect=side_effect):
            result = _fetch_macro_context()

        assert result["errors_fatal"] == []
        macro = result["data"]
        assert macro["audusd_trend"] == "strengthening"
        assert macro["rba_rate_direction"] == "rising"
        assert macro["macro_signal"] == "unfavorable"

    def test_all_series_fail_returns_fatal_error(self):
        from tools.ticker.macro_tools import _fetch_macro_context
        with patch("tools.ticker.macro_tools._fetch_fred_series", return_value=[]):
            result = _fetch_macro_context()

        assert len(result["errors_fatal"]) > 0
        assert "FRED" in result["errors_fatal"][0]

    def test_partial_series_failure_is_non_fatal(self, fred_audusd_flat, fred_rba_flat):
        from tools.ticker.macro_tools import _fetch_macro_context
        def side_effect(series_id, *args, **kwargs):
            # Only AUD/USD and RBA succeed; others return empty
            if series_id == "DEXUSAL":
                return fred_audusd_flat
            if series_id == "IRSTCB01AUM156N":
                return fred_rba_flat
            return []  # gold, oil, iron all fail
        with patch("tools.ticker.macro_tools._fetch_fred_series", side_effect=side_effect):
            result = _fetch_macro_context()

        # Should NOT be fatal since we got some data
        assert result["errors_fatal"] == []
        assert len(result["errors_non_fatal"]) > 0   # some commodity warnings
        assert result["data"]["audusd_current"] is not None

    def test_result_always_has_required_keys(self, fred_audusd_flat, fred_rba_flat):
        from tools.ticker.macro_tools import _fetch_macro_context
        with patch("tools.ticker.macro_tools._fetch_fred_series", return_value=[]):
            result = _fetch_macro_context()

        assert "data" in result
        assert "errors_fatal" in result
        assert "errors_non_fatal" in result
        macro = result["data"]
        for key in ["audusd_trend", "rba_rate_direction", "gold_trend",
                    "oil_trend", "iron_ore_trend", "macro_signal", "macro_notes"]:
            assert key in macro, f"Missing key: {key}"

    def test_macro_signal_always_valid_value(self):
        from tools.ticker.macro_tools import _fetch_macro_context
        with patch("tools.ticker.macro_tools._fetch_fred_series", return_value=[]):
            result = _fetch_macro_context()
        assert result["data"]["macro_signal"] in {"favorable", "unfavorable", "neutral"}
