"""
Print Volume Profile features used in the trading environment
"""

print("=" * 80)
print("VOLUME PROFILE FEATURES USED IN OBSERVATION SPACE")
print("=" * 80)

print("\n📊 FEATURE GROUP: volume_profile")
print("   Shape: (lookback_window=288, 10)")
print("   Source: EnhancedVolumeProfile class\n")

print("Feature breakdown (10 features per bar):")
print("-" * 80)

print("\n[0] Distance to Current POC (Point of Control)")
print("    - Distance from current price to highest volume price level")
print("    - Normalized by price range")
print("    - POC is the price with maximum traded volume")

print("\n[1] Distance to Previous Day VAH (Value Area High)")
print("    - Distance from current price to previous day's value area high")
print("    - Normalized by current price (percentage)")
print("    - VAH = upper bound of 70% volume area from previous session")

print("\n[2] Distance to Previous Day VAL (Value Area Low)")
print("    - Distance from current price to previous day's value area low")
print("    - Normalized by current price (percentage)")
print("    - VAL = lower bound of 70% volume area from previous session")

print("\n[3] Distance to Previous Day POC")
print("    - Distance from current price to previous day's POC")
print("    - Normalized by current price (percentage)")
print("    - Reference level from previous trading session")

print("\n[4] Inside Value Area (Binary)")
print("    - 1.0 if current price is between prev_day_VAL and prev_day_VAH")
print("    - 0.0 otherwise")
print("    - Indicates if trading inside previous day's acceptance area")

print("\n[5] Above Value Area (Binary)")
print("    - 1.0 if current price is above prev_day_VAH")
print("    - 0.0 otherwise")
print("    - Indicates if price broke above previous day's acceptance")

print("\n[6] Distance to Previous Day High")
print("    - Distance from current price to previous day's highest price")
print("    - Normalized by current price (percentage)")

print("\n[7] Distance to Previous Day Low")
print("    - Distance from current price to previous day's lowest price")
print("    - Normalized by current price (percentage)")

print("\n[8] Distance to Weekly POC")
print("    - Distance from current price to previous week's POC")
print("    - Normalized by current price (percentage)")
print("    - Longer timeframe reference level")

print("\n[9] Local Volume Concentration")
print("    - Sum of volume weights in ±2 bins around current price")
print("    - Range: 0.0 to 1.0 (clamped)")
print("    - High values indicate strong support/resistance zones")

print("\n" + "=" * 80)
print("ADDITIONAL VP FEATURES IN PRICE_CONTEXT GROUP")
print("=" * 80)

print("\nThese 5 VP features are also included in the 'price_context' group:")
print("-" * 80)

print("\n[0] distance_to_vah (from current session)")
print("    - Real-time distance to current session's Value Area High")

print("\n[1] distance_to_val (from current session)")
print("    - Real-time distance to current session's Value Area Low")

print("\n[2] distance_to_poc (from current session)")
print("    - Real-time distance to current session's Point of Control")

print("\n[3] inside_value_area (Binary)")
print("    - 1.0 if price is inside current session's value area")

print("\n[4] above_value_area (Binary)")
print("    - 1.0 if price is above current session's value area")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nTotal VP-related features in observation: 15")
print("  - 10 features in 'volume_profile' group (previous session + weekly references)")
print("  - 5 features in 'price_context' group (current session levels)")

print("\nKey concepts:")
print("  • POC (Point of Control): Price level with highest traded volume")
print("  • VAH/VAL: Upper/lower bounds containing 70% of session volume")
print("  • Value Area: Price range where 70% of volume was traded")
print("  • Session: Daily or weekly period for calculating VP levels")

print("\nUse cases:")
print("  • POC acts as magnet - price tends to return to high volume areas")
print("  • VAH/VAL are key support/resistance levels")
print("  • Trading above VAH = bullish, below VAL = bearish")
print("  • Inside value area = balanced/consolidation")
print("  • Local volume concentration shows immediate support/resistance")

print("\n" + "=" * 80)
