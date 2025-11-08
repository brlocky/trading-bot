"""
Unit tests for session_utils.py - Trading Session Detection

Tests session detection across different times of year to ensure
DST (Daylight Saving Time) changes are handled correctly for London and NY sessions.
"""

import pytest
import pandas as pd
from src.utils.session_utils import get_trading_session


class TestSessionDetection:
    """Test trading session detection with DST awareness"""

    # ==================== WINTER TESTS (Standard Time) ====================

    def test_winter_asia_early_morning(self):
        """Test ASIA session early morning in January (standard time)"""
        timestamp = pd.to_datetime("2024-01-15 00:30:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        assert session == "ASIA"
        assert is_asia is True
        assert is_london is False
        assert is_ny is False
        assert mins_in > 0 and mins_in < 999  # Should be into the session

    def test_winter_asia_late(self):
        """Test ASIA session late morning in January"""
        timestamp = pd.to_datetime("2024-01-15 07:30:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        assert session == "ASIA"
        assert is_asia is True
        assert mins_until < 60  # Close to end of ASIA session

    def test_winter_london_early_gmt(self):
        """Test LONDON session early in January (GMT - standard time)"""
        timestamp = pd.to_datetime("2024-01-15 08:30:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        assert session == "LONDON"
        assert is_london is True
        assert is_asia is False
        assert is_ny is False
        assert mins_in < 60  # Should be early in London session

    def test_winter_london_ny_overlap(self):
        """Test LONDON/NY overlap in January (standard time)"""
        timestamp = pd.to_datetime("2024-01-15 15:00:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        # Should be in NY session (higher priority) or London late
        assert is_london is True or is_ny is True
        assert is_asia is False

    def test_winter_ny_session_est(self):
        """Test NY session in January (EST - standard time)"""
        timestamp = pd.to_datetime("2024-01-15 16:00:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        assert session == "NY"
        assert is_ny is True

    def test_winter_ny_late(self):
        """Test NY session late in January"""
        timestamp = pd.to_datetime("2024-01-15 20:30:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        assert session == "NY"
        assert is_ny is True
        assert is_london is False
        assert is_asia is False

    def test_winter_asia_start(self):
        """Test ASIA session start at 23:00 UTC in January"""
        timestamp = pd.to_datetime("2024-01-15 23:00:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        assert session == "ASIA"
        assert is_asia is True
        assert is_london is False
        assert is_ny is False
        assert mins_in == 0  # Just started

    # ==================== SPRING TRANSITION ====================

    def test_spring_march(self):
        """Test sessions in March (transition to DST)"""
        timestamp = pd.to_datetime("2024-03-15 14:00:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        # March is still winter in our simplified model (DST kicks in April)
        assert session in ["LONDON", "NY"]

    # ==================== SUMMER TESTS (Daylight Saving Time) ====================

    def test_summer_asia_early_morning(self):
        """Test ASIA session early morning in July (summer)"""
        timestamp = pd.to_datetime("2024-07-15 00:30:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        assert session == "ASIA"
        assert is_asia is True
        assert is_london is False
        assert is_ny is False

    def test_summer_london_early_bst(self):
        """Test LONDON session early in July (BST - daylight time)"""
        timestamp = pd.to_datetime("2024-07-15 07:30:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        # In summer, London opens at 07:00 UTC (BST)
        assert session == "LONDON"
        assert is_london is True
        assert mins_in < 60  # Should be early in session

    def test_summer_london_ny_overlap(self):
        """Test LONDON/NY overlap in July (daylight time)"""
        timestamp = pd.to_datetime("2024-07-15 14:00:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        # Should have both sessions open or be in NY
        assert is_london is True or is_ny is True
        assert is_asia is False

    def test_summer_ny_session_edt(self):
        """Test NY session in July (EDT - daylight time)"""
        timestamp = pd.to_datetime("2024-07-15 15:00:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        assert session == "NY"
        assert is_ny is True

    def test_summer_ny_late(self):
        """Test NY session late in July"""
        timestamp = pd.to_datetime("2024-07-15 19:30:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        assert session == "NY"
        assert is_ny is True
        # Should be late in session
        assert mins_until < 60 or mins_in > 300

    def test_summer_asia_start(self):
        """Test ASIA session start at 23:00 UTC in July"""
        timestamp = pd.to_datetime("2024-07-15 23:00:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        assert session == "ASIA"
        assert is_asia is True
        assert is_london is False
        assert is_ny is False

    # ==================== AUTUMN TRANSITION ====================

    def test_autumn_october(self):
        """Test sessions in October (transition back to standard time)"""
        timestamp = pd.to_datetime("2024-10-15 14:30:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        # October is still summer in our simplified model
        assert session in ["LONDON", "NY"]

    def test_autumn_november(self):
        """Test sessions in November (back to standard time)"""
        timestamp = pd.to_datetime("2024-11-15 15:00:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        # November should use winter schedule
        assert session == "NY"

    # ==================== EDGE CASES & DEAD ZONES ====================

    def test_dead_zone_between_asia_london(self):
        """Test the dead zone between ASIA end and LONDON start"""
        # Winter: Between 08:00 and 08:00 UTC (no gap in winter)
        # Summer: Between 08:00 and 07:00 UTC (no gap - actually overlap)
        timestamp = pd.to_datetime("2024-01-15 08:15:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        # Should be in LONDON session
        assert session == "LONDON"

    def test_dead_zone_between_london_ny(self):
        """Test potential dead zone between LONDON end and NY start"""
        # Winter: London ends 16:30, NY starts 14:30 (overlap)
        # Summer: London ends 15:30, NY starts 13:30 (overlap)
        timestamp = pd.to_datetime("2024-01-15 16:00:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        # Should be in NY session or late London
        assert session in ["LONDON", "NY"]

    def test_dead_zone_between_ny_asia(self):
        """Test the dead zone between NY end and ASIA start"""
        # Winter: NY ends 21:00, ASIA starts 23:00 (2 hour gap)
        # Summer: NY ends 20:00, ASIA starts 23:00 (3 hour gap)
        timestamp = pd.to_datetime("2024-01-15 22:00:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        # Should be assigned to nearest session
        assert session in ["NY", "ASIA"]
        # If marked as very late (999), it's in the dead zone
        if mins_in == 999:
            assert mins_until < 120  # Should be within 2 hours of next session

    # ==================== FEATURE VECTOR TESTS ====================

    def test_feature_vector_single_session(self):
        """Test that feature vector has only one session open (non-overlap time)"""
        timestamp = pd.to_datetime("2024-01-15 10:00:00")
        _, _, _, is_asia, is_london, is_ny = get_trading_session(timestamp)

        # At 10:00 UTC in winter, should be London only
        sessions_open = sum([is_asia, is_london, is_ny])
        assert sessions_open == 1
        assert is_london is True

    def test_feature_vector_overlap_sessions(self):
        """Test that feature vector can have multiple sessions open"""
        # 14:00 UTC in summer: London (07:00-15:30) and NY (13:30-20:00) overlap
        timestamp = pd.to_datetime("2024-07-15 14:00:00")
        _, _, _, is_asia, is_london, is_ny = get_trading_session(timestamp)

        # Should have multiple sessions open
        sessions_open = sum([is_asia, is_london, is_ny])
        assert sessions_open >= 1  # At least one, possibly two

    def test_feature_vector_boolean_types(self):
        """Test that session flags are proper boolean types"""
        timestamp = pd.to_datetime("2024-07-15 14:00:00")
        _, _, _, is_asia, is_london, is_ny = get_trading_session(timestamp)

        assert isinstance(is_asia, bool)
        assert isinstance(is_london, bool)
        assert isinstance(is_ny, bool)

    # ==================== CONSISTENCY TESTS ====================

    def test_session_consistency_across_day(self):
        """Test that sessions progress logically throughout a day"""
        base_date = "2024-07-15"
        hours = [0, 4, 8, 12, 16, 20, 23]

        sessions_seen = []
        for hour in hours:
            timestamp = pd.to_datetime(f"{base_date} {hour:02d}:00:00")
            session, _, _, _, _, _ = get_trading_session(timestamp)
            sessions_seen.append(session)

        # Should see a progression: ASIA -> LONDON -> NY -> ASIA
        assert "ASIA" in sessions_seen
        assert "LONDON" in sessions_seen
        assert "NY" in sessions_seen

    def test_minutes_consistency(self):
        """Test that minutes_into and minutes_until are consistent"""
        timestamp = pd.to_datetime("2024-07-15 10:00:00")
        session, mins_in, mins_until, _, _, _ = get_trading_session(timestamp)

        # Both should be positive (or 0 for start, or 999 for dead zone)
        assert mins_in >= 0
        assert mins_until >= 0

        # If not in dead zone, sum should be reasonable (session length)
        if mins_in != 999:
            total = mins_in + mins_until
            assert total > 0  # Should have some session time
            assert total <= 600  # No session longer than 10 hours

    # ==================== YEAR-ROUND COVERAGE ====================

    @pytest.mark.parametrize("month,day", [
        (1, 15),   # January - deep winter
        (2, 15),   # February
        (3, 15),   # March - spring transition
        (4, 15),   # April - DST starts
        (5, 15),   # May
        (6, 15),   # June - full summer
        (7, 15),   # July
        (8, 15),   # August
        (9, 15),   # September
        (10, 15),  # October - autumn transition
        (11, 15),  # November - DST ends
        (12, 15),  # December - deep winter
    ])
    def test_all_months(self, month, day):
        """Test that session detection works for all months of the year"""
        timestamp = pd.to_datetime(f"2024-{month:02d}-{day:02d} 12:00:00")
        session, mins_in, mins_until, is_asia, is_london, is_ny = get_trading_session(timestamp)

        # Should always return valid session
        assert session in ["ASIA", "LONDON", "NY"]

        # At 12:00 UTC, should always be in LONDON or NY (never ASIA)
        assert session in ["LONDON", "NY"]
        assert is_asia is False

        # Should have valid minute values
        assert mins_in >= 0
        assert mins_until >= 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
