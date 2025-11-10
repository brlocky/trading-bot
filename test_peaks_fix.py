"""
Test script to verify find_peaks_troughs improvements
"""
import numpy as np
import matplotlib.pyplot as plt
from src.utils.divergence_detector import find_peaks_troughs

# Create test data with known peaks and troughs


def test_basic_peaks_troughs():
    """Test basic peak/trough detection"""
    # Create synthetic price data with clear peaks and troughs
    data = np.array([10, 12, 15, 20, 18, 16, 14, 15, 18, 22, 20, 18, 16, 12, 10, 8, 10, 12, 15, 12, 10, 8, 5, 8, 10])

    peaks, troughs = find_peaks_troughs(data, order=3)

    print("Test 1: Basic Peak/Trough Detection")
    print(f"Data: {data}")
    print(f"Peaks found: {peaks}")
    print(f"Troughs found: {troughs}")
    print()

    # Visualize
    plt.figure(figsize=(14, 5))
    plt.plot(data, 'b-', label='Data', linewidth=2)
    if peaks:
        peak_x = [p[0] for p in peaks]
        peak_y = [p[1] for p in peaks]
        plt.plot(peak_x, peak_y, 'rv', markersize=12, label='Peaks')
    if troughs:
        trough_x = [t[0] for t in troughs]
        trough_y = [t[1] for t in troughs]
        plt.plot(trough_x, trough_y, 'g^', markersize=12, label='Troughs')
    plt.legend()
    plt.title('Test 1: Basic Peak/Trough Detection')
    plt.grid(alpha=0.3)
    plt.show()

    return peaks, troughs


def test_with_highs_lows():
    """Test with separate high/low arrays (candlestick data)"""
    # Simulate OHLC data
    closes = np.array([100, 102, 105, 108, 106, 104, 102, 105, 110, 108, 106, 104, 100, 98, 96, 98, 100, 102, 100, 98, 95, 98, 100])
    highs = closes + 2  # Highs are 2 points above close
    lows = closes - 2   # Lows are 2 points below close

    # Add some variation to highs and lows to make them realistic
    highs[3] = 112  # Make a prominent peak
    lows[14] = 93   # Make a prominent trough

    peaks, troughs = find_peaks_troughs(closes, order=3, highs=highs, lows=lows)

    print("Test 2: Peak/Trough with OHLC Data")
    print(f"Closes: {closes}")
    print(f"Highs: {highs}")
    print(f"Lows: {lows}")
    print(f"Peaks found: {peaks}")
    print(f"Troughs found: {troughs}")
    print()

    # Visualize
    plt.figure(figsize=(14, 5))
    plt.plot(closes, 'b-', label='Close', linewidth=2, alpha=0.7)
    plt.plot(highs, 'r--', label='High', linewidth=1, alpha=0.5)
    plt.plot(lows, 'g--', label='Low', linewidth=1, alpha=0.5)

    if peaks:
        peak_x = [p[0] for p in peaks]
        peak_y = [highs[p[0]] for p in peaks]  # Use actual high values
        plt.plot(peak_x, peak_y, 'rv', markersize=12, label='Detected Peaks')
    if troughs:
        trough_x = [t[0] for t in troughs]
        trough_y = [lows[t[0]] for t in troughs]  # Use actual low values
        plt.plot(trough_x, trough_y, 'g^', markersize=12, label='Detected Troughs')

    plt.legend()
    plt.title('Test 2: Peak/Trough Detection with OHLC Data')
    plt.grid(alpha=0.3)
    plt.show()

    return peaks, troughs


def test_flat_peaks():
    """Test with flat peaks/troughs (should now be detected)"""
    data = np.array([10, 12, 15, 20, 20, 20, 18, 15, 12, 10, 8, 5, 5, 5, 8, 10, 12])

    peaks, troughs = find_peaks_troughs(data, order=2)

    print("Test 3: Flat Peak/Trough Detection")
    print(f"Data: {data}")
    print(f"Peaks found (should include flat peak at 20): {peaks}")
    print(f"Troughs found (should include flat trough at 5): {troughs}")
    print()

    # Visualize
    plt.figure(figsize=(14, 5))
    plt.plot(data, 'b-', label='Data', linewidth=2)
    if peaks:
        peak_x = [p[0] for p in peaks]
        peak_y = [p[1] for p in peaks]
        plt.plot(peak_x, peak_y, 'rv', markersize=12, label='Peaks')
    if troughs:
        trough_x = [t[0] for t in troughs]
        trough_y = [t[1] for t in troughs]
        plt.plot(trough_x, trough_y, 'g^', markersize=12, label='Troughs')
    plt.legend()
    plt.title('Test 3: Flat Peak/Trough Detection')
    plt.grid(alpha=0.3)
    plt.show()

    return peaks, troughs


def test_different_orders():
    """Test with different order values"""
    data = np.array([10, 15, 20, 25, 22, 18, 15, 18, 22, 28, 25, 22, 18, 14, 10, 8, 12, 15, 18, 15, 12, 8, 5, 8, 12])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    orders = [2, 5, 8]

    for idx, order in enumerate(orders):
        peaks, troughs = find_peaks_troughs(data, order=order)

        print(f"Test 4.{idx+1}: Order={order}")
        print(f"Peaks: {peaks}")
        print(f"Troughs: {troughs}")
        print()

        axes[idx].plot(data, 'b-', label='Data', linewidth=2)
        if peaks:
            peak_x = [p[0] for p in peaks]
            peak_y = [p[1] for p in peaks]
            axes[idx].plot(peak_x, peak_y, 'rv', markersize=10, label='Peaks')
        if troughs:
            trough_x = [t[0] for t in troughs]
            trough_y = [t[1] for t in troughs]
            axes[idx].plot(trough_x, trough_y, 'g^', markersize=10, label='Troughs')
        axes[idx].legend()
        axes[idx].set_title(f'Order={order}')
        axes[idx].grid(alpha=0.3)

    plt.suptitle('Test 4: Different Order Values', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("="*60)
    print("Testing find_peaks_troughs improvements")
    print("="*60)
    print()

    test_basic_peaks_troughs()
    test_with_highs_lows()
    test_flat_peaks()
    test_different_orders()

    print("="*60)
    print("All tests completed!")
    print("="*60)
