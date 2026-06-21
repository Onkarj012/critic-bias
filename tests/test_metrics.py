"""Tests for CRITIQ-BIAS SOTA metrics."""

import pytest
import numpy as np

from app.metrics.statistical import StatisticalTests
from app.metrics.bps import _resolve_condition_value


class TestStatisticalTests:
    def test_bootstrap_ci_basic(self):
        data = [5.0, 6.0, 7.0, 8.0, 9.0]
        point, ci_lower, ci_upper = StatisticalTests.bootstrap_ci(data)
        assert ci_lower <= point <= ci_upper
        assert 5.0 <= point <= 9.0

    def test_cohens_d_same_groups(self):
        d = StatisticalTests.cohens_d([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        assert d == 0.0

    def test_cohens_d_different_groups(self):
        d = StatisticalTests.cohens_d([8.0, 9.0, 10.0], [2.0, 3.0, 4.0])
        assert abs(d) > 2.0

    def test_one_way_anova_significant(self):
        groups = [
            [8.0, 9.0, 8.5, 9.5, 8.0],
            [3.0, 4.0, 3.5, 4.5, 3.0],
        ]
        result = StatisticalTests.one_way_anova(groups)
        assert result.significant
        assert result.p_value < 0.05
        assert result.effect_size > 0

    def test_one_way_anova_not_significant(self):
        groups = [
            [5.0, 5.1, 4.9, 5.0, 5.1],
            [5.0, 4.9, 5.1, 5.0, 4.9],
        ]
        result = StatisticalTests.one_way_anova(groups)
        assert not result.significant

    def test_one_way_anova_constant_groups(self):
        groups = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        result = StatisticalTests.one_way_anova(groups)
        assert not result.significant
        assert result.p_value == 1.0

    def test_one_way_anova_singleton_group_rejected(self):
        result = StatisticalTests.one_way_anova([[1.0], [2.0, 3.0]])
        assert not result.significant
        assert result.p_value == 1.0

    def test_mfi_significant_favored(self):
        assert StatisticalTests.mfi_significant(1.1, 1.3) == "significantly_favored"

    def test_mfi_significant_disfavored(self):
        assert StatisticalTests.mfi_significant(0.7, 0.9) == "significantly_disfavored"

    def test_mfi_no_bias(self):
        assert StatisticalTests.mfi_significant(0.9, 1.1) == "no_significant_bias"


class TestBPSConditionNames:
    """Verify BPS handles both v1 and v2 condition naming."""

    def test_visible_blind_keys(self):
        vals = {"visible": 1.2, "blind": 1.0}
        visible_val = _resolve_condition_value(vals, "visible", "source_visible")
        blind_val = _resolve_condition_value(vals, "blind", "source_blind")
        assert visible_val == 1.2
        assert blind_val == 1.0
        assert abs(visible_val - blind_val) == pytest.approx(0.2)

    def test_legacy_keys(self):
        vals = {"source_visible": 1.15, "source_blind": 1.05}
        visible_val = _resolve_condition_value(vals, "visible", "source_visible")
        blind_val = _resolve_condition_value(vals, "blind", "source_blind")
        assert visible_val == 1.15
        assert blind_val == 1.05

    def test_zero_mfi_not_treated_as_missing(self):
        vals = {"visible": 0.0, "blind": 1.0}
        visible_val = _resolve_condition_value(vals, "visible", "source_visible")
        blind_val = _resolve_condition_value(vals, "blind", "source_blind")
        assert visible_val == 0.0
        assert blind_val == 1.0
