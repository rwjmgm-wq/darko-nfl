"""
Unit tests for time-weighted metrics
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from features.time_weighted_metrics import TimeWeightedMetrics


class TestTimeWeightedMetrics:
    """Test suite for TimeWeightedMetrics class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.twm = TimeWeightedMetrics()

    def test_default_decay_rates(self):
        """Test that default decay rates are reasonable."""
        assert 0 < self.twm.decay_rates['epa'] < 1
        assert 0 < self.twm.decay_rates['cpoe'] < 1
        assert self.twm.decay_rates['default'] == 0.996

    def test_calculate_time_weights(self):
        """Test time weight calculation."""
        # Create dates
        dates = pd.Series([
            datetime(2024, 1, 1),
            datetime(2024, 1, 8),
            datetime(2024, 1, 15)
        ])
        reference_date = datetime(2024, 1, 15)

        # Calculate weights
        weights = self.twm.calculate_time_weights(dates, beta=0.99, reference_date=reference_date)

        # Most recent should have weight 1.0
        assert weights.iloc[-1] == pytest.approx(1.0)

        # Older games should have lower weights
        assert weights.iloc[0] < weights.iloc[-1]
        assert weights.iloc[1] < weights.iloc[-1]

        # Check exponential decay
        # 7 days ago with beta=0.99: 0.99^7 ≈ 0.932
        assert weights.iloc[1] == pytest.approx(0.99 ** 7, rel=0.01)

    def test_aggregate_player_metrics(self):
        """Test player metric aggregation."""
        # Create sample data
        data = pd.DataFrame({
            'game_date': pd.date_range('2024-09-01', periods=10, freq='7D'),
            'passer_player_id': ['player1'] * 10,
            'passer_player_name': ['Test Player'] * 10,
            'epa_per_play': np.random.randn(10) * 0.1 + 0.15,
            'cpoe': np.random.randn(10) * 3 + 2,
            'attempts': [30] * 10,
            'season': [2024] * 10
        })

        # Aggregate
        result = self.twm.aggregate_player_metrics(
            data,
            metrics=['epa_per_play', 'cpoe']
        )

        # Check output
        assert len(result) == 1
        assert 'epa_per_play_weighted' in result.columns
        assert 'cpoe_weighted' in result.columns
        assert result['total_attempts'].iloc[0] == 300

    def test_weighted_metric_gives_more_weight_to_recent(self):
        """Test that weighted metrics properly favor recent games."""
        # Create data with clear trend
        dates = pd.date_range('2024-01-01', periods=5, freq='7D')
        data = pd.DataFrame({
            'game_date': dates,
            'passer_player_id': ['player1'] * 5,
            'passer_player_name': ['Test Player'] * 5,
            'epa_per_play': [0.0, 0.0, 0.0, 0.5, 0.5],  # Recent games much better
            'attempts': [30] * 5,
            'season': [2024] * 5
        })

        # Calculate weighted metric
        result = self.twm.aggregate_player_metrics(data, metrics=['epa_per_play'])
        weighted_epa = result['epa_per_play_weighted'].iloc[0]

        # Weighted average should be higher than simple average
        simple_avg = data['epa_per_play'].mean()  # 0.2
        assert weighted_epa > simple_avg

    def test_decay_weights_summary(self):
        """Test decay weights summary generation."""
        summary = self.twm.get_decay_weights_summary(days=365)

        # Check structure
        assert 'metric' in summary.columns
        assert 'beta' in summary.columns
        assert 'day_0' in summary.columns
        assert 'day_365' in summary.columns

        # Day 0 should always be 1.0
        assert all(summary['day_0'] == 1.0)

        # Day 365 should be less than 1.0
        assert all(summary['day_365'] < 1.0)


class TestRollingTimeWeightedMetrics:
    """Test suite for RollingTimeWeightedMetrics class."""

    def test_basic_update(self):
        """Test basic rolling update."""
        from features.time_weighted_metrics import RollingTimeWeightedMetrics

        rolling = RollingTimeWeightedMetrics(decay_rate=0.99)

        # First observation
        result1 = rolling.update('player1', metric_value=0.2, attempts=30, days_since_last_update=0)
        assert result1 == pytest.approx(0.2)

        # Second observation after 7 days
        result2 = rolling.update('player1', metric_value=0.3, attempts=30, days_since_last_update=7)

        # Should be between 0.2 and 0.3, closer to 0.3 (more recent)
        assert 0.2 < result2 < 0.3

    def test_get_current_value(self):
        """Test retrieving current value."""
        from features.time_weighted_metrics import RollingTimeWeightedMetrics

        rolling = RollingTimeWeightedMetrics()
        rolling.update('player1', metric_value=0.25, attempts=100)

        current = rolling.get_current_value('player1')
        assert current == pytest.approx(0.25)

        # Unknown player should return 0
        unknown = rolling.get_current_value('unknown_player')
        assert unknown == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
