"""
Unit tests for tools/ticker/utils/metrics_calculator.py and
the scoring/filter model in tools/ticker/metrics_tools.py.

Uses synthetic OHLCV DataFrames — no network calls needed.
"""

import pytest
import numpy as np
import pandas as pd
from tests.conftest import make_price_df


# ─────────────────────────────────────────────────────────────────────────────
# compute_metrics  (pure CPU, uses synthetic DataFrames)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_ticker_key(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        assert result["ticker"] == "CBA.AX"

    def test_returns_all_expected_technical_keys(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        technical_keys = [
            "rsi_14", "macd_hist", "macd_crossover", "above_ema50",
            "golden_cross", "bb_position", "atr_pct"
        ]
        for key in technical_keys:
            assert key in result, f"Missing technical key: {key}"

    def test_returns_all_expected_fundamental_keys(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        fundamental_keys = [
            "pe_ratio", "pb_ratio", "roe", "profit_margin", "revenue_growth",
            "earnings_growth", "debt_to_equity", "current_ratio", "dividend_yield",
            "sector", "industry"
        ]
        for key in fundamental_keys:
            assert key in result, f"Missing fundamental key: {key}"

    def test_returns_all_momentum_keys(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        for key in ["ret_1m", "ret_3m", "ret_6m", "ret_1y", "high_52w", "low_52w",
                    "pct_from_52w_high"]:
            assert key in result, f"Missing momentum key: {key}"

    def test_rsi_in_valid_range(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        if result["rsi_14"] is not None:
            assert 0 <= result["rsi_14"] <= 100

    def test_macd_crossover_is_valid_enum(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        assert result["macd_crossover"] in {-1, 0, 1}

    def test_above_ema50_is_boolean(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        assert isinstance(result["above_ema50"], bool)

    def test_golden_cross_is_boolean(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        assert isinstance(result["golden_cross"], bool)

    def test_bb_position_in_valid_range_when_not_none(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        if result["bb_position"] is not None:
            # BB position can go slightly outside [0,1] in volatile markets
            assert -0.5 <= result["bb_position"] <= 1.5

    def test_short_history_returns_none_for_1m_return(self, short_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        # Only 20 days of data — can't compute 22-day 1-month return
        result = compute_metrics("CBA.AX", short_price_df, sample_fundamentals)
        assert result["ret_1m"] is None

    def test_short_history_returns_none_for_3m_return(self, short_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", short_price_df, sample_fundamentals)
        assert result["ret_3m"] is None

    def test_fundamentals_sourced_from_info_dict(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        assert result["pe_ratio"] == pytest.approx(14.5)
        assert result["roe"] == pytest.approx(0.145)
        assert result["sector"] == "Financial Services"

    def test_empty_fundamentals_returns_none_for_all_fundamental_fields(
        self, synthetic_price_df, empty_fundamentals
    ):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, empty_fundamentals)
        assert result["pe_ratio"] is None
        assert result["roe"] is None
        assert result["market_cap"] is None
        assert result["sector"] == "Unknown"

    def test_volatility_is_annualised_positive(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        assert result["volatility_30d"] > 0  # must be positive
        # Typical ASX stock: 10–80% annualised
        assert 0.05 <= result["volatility_30d"] <= 2.0

    def test_avg_turnover_is_positive(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        assert result["avg_turnover_30d"] > 0

    def test_current_price_matches_last_close(self, synthetic_price_df, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        result = compute_metrics("CBA.AX", synthetic_price_df, sample_fundamentals)
        expected_price = synthetic_price_df["Close"].iloc[-1]
        assert result["current_price"] == pytest.approx(expected_price)

    def test_trending_up_stock_above_ema50(self, sample_fundamentals):
        from tools.ticker.utils.metrics_calculator import compute_metrics
        # Create a monotonically rising price series — should be above EMA50
        n = 300
        prices = np.linspace(50, 150, n)  # steady uptrend
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
        df = pd.DataFrame({
            "Open": prices, "High": prices * 1.01, "Low": prices * 0.99,
            "Close": prices, "Volume": [1_000_000] * n,
        }, index=dates)
        result = compute_metrics("TEST.AX", df, sample_fundamentals)
        assert result["above_ema50"] is True


# ─────────────────────────────────────────────────────────────────────────────
# score_stock  (pure scoring function in metrics_tools.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreStock:
    def _make_row(self, **overrides):
        """Build a baseline metrics row with neutral/mediocre values."""
        base = {
            "ret_1m": 0.02,
            "ret_3m": 0.05,
            "pct_from_52w_high": -0.15,
            "rsi_14": 55.0,
            "macd_hist": 0.5,
            "macd_crossover": 0,
            "above_ema50": True,
            "golden_cross": False,
            "bb_position": 0.6,
            "pe_ratio": 15.0,
            "roe": 0.15,
            "revenue_growth": 0.08,
            "debt_to_equity": 40.0,
            "profit_margin": 0.18,
            "volatility_30d": 0.25,
        }
        base.update(overrides)
        return base

    def test_score_always_between_0_and_100(self):
        from tools.ticker.metrics_tools import score_stock
        row = self._make_row()
        score = score_stock(row)
        assert 0 <= score <= 100

    def test_perfect_row_scores_high(self):
        from tools.ticker.metrics_tools import score_stock
        perfect = self._make_row(
            ret_1m=0.15, ret_3m=0.30, pct_from_52w_high=-0.02,
            rsi_14=65, macd_crossover=1, above_ema50=True, golden_cross=True,
            bb_position=0.7, pe_ratio=12.0, roe=0.25, revenue_growth=0.20,
            debt_to_equity=20.0, profit_margin=0.25, volatility_30d=0.15,
        )
        assert score_stock(perfect) >= 60

    def test_terrible_row_scores_low(self):
        from tools.ticker.metrics_tools import score_stock
        terrible = self._make_row(
            ret_1m=-0.20, ret_3m=-0.40, pct_from_52w_high=-0.60,
            rsi_14=75, macd_crossover=-1, above_ema50=False, golden_cross=False,
            pe_ratio=None, roe=-0.10, revenue_growth=-0.15,
            debt_to_equity=300.0, profit_margin=-0.05, volatility_30d=0.80,
        )
        assert score_stock(terrible) <= 40

    def test_empty_row_returns_valid_score(self):
        from tools.ticker.metrics_tools import score_stock
        assert 0 <= score_stock({}) <= 100

    def test_macd_crossover_bonus_applied(self):
        from tools.ticker.metrics_tools import score_stock
        with_cross = score_stock(self._make_row(macd_crossover=1))
        without_cross = score_stock(self._make_row(macd_crossover=0))
        assert with_cross > without_cross

    def test_golden_cross_bonus_applied(self):
        from tools.ticker.metrics_tools import score_stock
        with_gc = score_stock(self._make_row(golden_cross=True))
        without_gc = score_stock(self._make_row(golden_cross=False))
        assert with_gc > without_gc

    def test_high_volatility_penalised(self):
        from tools.ticker.metrics_tools import score_stock
        low_vol = score_stock(self._make_row(volatility_30d=0.15))
        high_vol = score_stock(self._make_row(volatility_30d=0.70))
        assert low_vol > high_vol

    def test_high_debt_penalised(self):
        from tools.ticker.metrics_tools import score_stock
        low_debt = score_stock(self._make_row(debt_to_equity=30.0))
        high_debt = score_stock(self._make_row(debt_to_equity=250.0))
        assert low_debt > high_debt


# ─────────────────────────────────────────────────────────────────────────────
# passes_hard_filters  (pure filter function)
# ─────────────────────────────────────────────────────────────────────────────

class TestPassesHardFilters:
    def _config(self):
        return {
            "min_turnover": 500_000,
            "min_market_cap": 100_000_000,
            "min_price": 0.10,
            "max_rsi": 82,
            "max_volatility": 0.90,
        }

    def test_good_stock_passes(self):
        from tools.ticker.metrics_tools import passes_hard_filters
        row = {
            "avg_turnover_30d": 2_000_000,
            "market_cap": 500_000_000,
            "current_price": 5.50,
            "rsi_14": 55.0,
            "volatility_30d": 0.25,
        }
        passed, reason = passes_hard_filters(row, self._config())
        assert passed is True
        assert reason == ""

    def test_insufficient_turnover_fails(self):
        from tools.ticker.metrics_tools import passes_hard_filters
        row = {
            "avg_turnover_30d": 100_000,  # below 500k threshold
            "market_cap": 500_000_000,
            "current_price": 5.50,
            "rsi_14": 55.0,
            "volatility_30d": 0.25,
        }
        passed, reason = passes_hard_filters(row, self._config())
        assert passed is False
        assert "turnover" in reason.lower()

    def test_none_turnover_fails(self):
        from tools.ticker.metrics_tools import passes_hard_filters
        row = {"avg_turnover_30d": None, "market_cap": 500_000_000, "current_price": 5.50}
        passed, reason = passes_hard_filters(row, self._config())
        assert passed is False

    def test_small_cap_fails(self):
        from tools.ticker.metrics_tools import passes_hard_filters
        row = {
            "avg_turnover_30d": 2_000_000,
            "market_cap": 50_000_000,   # below 100M threshold
            "current_price": 5.50,
            "rsi_14": 55.0,
            "volatility_30d": 0.25,
        }
        passed, reason = passes_hard_filters(row, self._config())
        assert passed is False
        assert "market cap" in reason.lower()

    def test_penny_stock_fails(self):
        from tools.ticker.metrics_tools import passes_hard_filters
        row = {
            "avg_turnover_30d": 2_000_000,
            "market_cap": 500_000_000,
            "current_price": 0.05,   # below 10c threshold
            "rsi_14": 55.0,
            "volatility_30d": 0.25,
        }
        passed, reason = passes_hard_filters(row, self._config())
        assert passed is False
        assert "price" in reason.lower()

    def test_overbought_rsi_fails(self):
        from tools.ticker.metrics_tools import passes_hard_filters
        row = {
            "avg_turnover_30d": 2_000_000,
            "market_cap": 500_000_000,
            "current_price": 5.50,
            "rsi_14": 85.0,   # above 82 threshold
            "volatility_30d": 0.25,
        }
        passed, reason = passes_hard_filters(row, self._config())
        assert passed is False
        assert "rsi" in reason.lower() or "overbought" in reason.lower()

    def test_excessive_volatility_fails(self):
        from tools.ticker.metrics_tools import passes_hard_filters
        row = {
            "avg_turnover_30d": 2_000_000,
            "market_cap": 500_000_000,
            "current_price": 5.50,
            "rsi_14": 55.0,
            "volatility_30d": 0.95,   # above 90% threshold
        }
        passed, reason = passes_hard_filters(row, self._config())
        assert passed is False
        assert "volatility" in reason.lower()

    def test_none_market_cap_passes_cap_filter(self):
        """If market cap data is unavailable, filter should not block."""
        from tools.ticker.metrics_tools import passes_hard_filters
        row = {
            "avg_turnover_30d": 2_000_000,
            "market_cap": None,        # no data
            "current_price": 5.50,
            "rsi_14": 55.0,
            "volatility_30d": 0.25,
        }
        passed, _ = passes_hard_filters(row, self._config())
        assert passed is True
