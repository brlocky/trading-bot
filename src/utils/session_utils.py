"""
Trading Session Detection Utilities

Handles detection of major trading sessions (ASIA, LONDON, NY) with DST awareness.
"""

import pandas as pd


def get_trading_session(timestamp):
    """
    Determine which trading session the current time belongs to.
    Handles daylight saving time changes for London and NY sessions.

    Sessions (in local time):
    - ASIA (Tokyo): 09:00-18:00 JST (00:00-09:00 UTC, no DST)
    - LONDON: 08:00-16:30 GMT/BST (08:00-16:30 UTC winter / 07:00-15:30 UTC summer)
    - NY: 09:30-16:00 EST/EDT (14:30-21:00 UTC winter / 13:30-20:00 UTC summer)

    For simplicity with crypto 24/7 trading, using approximate UTC ranges:
    - ASIA: 23:00 (prev day) - 08:00 UTC (Tokyo morning session)
    - LONDON: 07:00 - 16:00 UTC (includes both GMT/BST, overlaps with Asia end)
    - NY: 13:00 - 21:00 UTC (includes both EST/EDT, overlaps with London)

    Args:
        timestamp: pandas Timestamp or datetime object

    Returns:
        tuple: (session_name, minutes_into_session, minutes_until_end, is_asia_open, is_london_open, is_ny_open)
    """
    if isinstance(timestamp, pd.Timestamp):
        dt = timestamp
    else:
        dt = pd.to_datetime(timestamp)

    # Get current time in UTC
    if dt.tz is None:
        # Assume UTC if no timezone
        current_time = dt.time()
    else:
        # Convert to UTC
        dt_utc = dt.tz_convert('UTC')
        current_time = dt_utc.time()

    # Convert to minutes since midnight for easier comparison
    current_minutes = current_time.hour * 60 + current_time.minute

    # Check for DST (approximate - London last Sun March to last Sun Oct, NY similar)
    month = dt.month
    is_summer = 4 <= month <= 10  # Rough approximation

    # Session boundaries (in minutes from midnight UTC)
    if is_summer:
        # Summer: DST active for London (BST) and NY (EDT)
        ASIA_START = 23 * 60  # 23:00 UTC (08:00 JST next day)
        ASIA_END = 8 * 60     # 08:00 UTC (17:00 JST)
        LONDON_START = 7 * 60   # 07:00 UTC (08:00 BST)
        LONDON_END = 15 * 60 + 30  # 15:30 UTC (16:30 BST)
        NY_START = 13 * 60 + 30    # 13:30 UTC (09:30 EDT)
        NY_END = 20 * 60           # 20:00 UTC (16:00 EDT)
    else:
        # Winter: Standard time for London (GMT) and NY (EST)
        ASIA_START = 23 * 60  # 23:00 UTC (08:00 JST next day)
        ASIA_END = 8 * 60     # 08:00 UTC (17:00 JST)
        LONDON_START = 8 * 60      # 08:00 UTC (08:00 GMT)
        LONDON_END = 16 * 60 + 30  # 16:30 UTC (16:30 GMT)
        NY_START = 14 * 60 + 30    # 14:30 UTC (09:30 EST)
        NY_END = 21 * 60           # 21:00 UTC (16:00 EST)

    # Check which sessions are open (boolean flags for features)
    is_asia_open = (current_minutes >= ASIA_START) or (current_minutes < ASIA_END)
    is_london_open = LONDON_START <= current_minutes < LONDON_END
    is_ny_open = NY_START <= current_minutes < NY_END

    # Determine primary session with priority: NY > LONDON > ASIA (in case of overlaps)
    # This prioritizes the most active/volatile sessions
    if NY_START <= current_minutes < NY_END:
        session = 'NY'
        minutes_into_session = current_minutes - NY_START
        minutes_until_end = NY_END - current_minutes

    elif LONDON_START <= current_minutes < LONDON_END:
        session = 'LONDON'
        minutes_into_session = current_minutes - LONDON_START
        minutes_until_end = LONDON_END - current_minutes

    elif ASIA_START <= current_minutes or current_minutes < ASIA_END:
        session = 'ASIA'
        if current_minutes >= ASIA_START:
            # Late evening (23:00-24:00)
            minutes_into_session = current_minutes - ASIA_START
            minutes_until_end = (24 * 60 - current_minutes) + ASIA_END
        else:
            # Early morning (00:00-08:00)
            minutes_into_session = (24 * 60 - ASIA_START) + current_minutes
            minutes_until_end = ASIA_END - current_minutes

    else:
        # Between sessions (dead zone) - assign to nearest upcoming session
        if current_minutes < LONDON_START:
            # Between ASIA end and LONDON start (08:00-07:00/08:00)
            session = 'ASIA'  # Tail end of Asia
            minutes_into_session = 999  # Mark as very late
            minutes_until_end = LONDON_START - current_minutes
        elif current_minutes < NY_START:
            # Between LONDON end and NY start
            session = 'LONDON'  # Tail end of London
            minutes_into_session = 999
            minutes_until_end = NY_START - current_minutes
        else:
            # Between NY end and ASIA start (21:00-23:00)
            session = 'NY'  # Tail end of NY
            minutes_into_session = 999
            if current_minutes < ASIA_START:
                minutes_until_end = ASIA_START - current_minutes
            else:
                minutes_until_end = (24*60 - current_minutes + ASIA_START)

    return session, minutes_into_session, minutes_until_end, is_asia_open, is_london_open, is_ny_open
