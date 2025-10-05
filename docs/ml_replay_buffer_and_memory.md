# ML Replay Buffer and Memory-Enhanced Dataset Design

## 1. Data Structure
Each row in your dataset should include:
- **Market features:** price, indicators, volume, trend slope, momentum, etc.
- **Trade features (optional):** last trade outcome, profit/loss, distance from last entry.
- **Target labels:** entry price, stop loss, take profit (the values you want to predict).
- **Memory flag:** boolean or weight indicating a "good trade" (e.g., profitable > threshold).

## 2. Memory / Replay Buffer
- Maintain a separate buffer of your best trades (profitable trades above a threshold).
- Each entry is a full feature row + target.
- Use weighting or oversampling so the model sees these trades more often during training.

## 3. Dataset Building Workflow
1. **Historical trend extraction:**
   - Identify trends using your signal (e.g., moving average slope, breakouts).
   - Group rows per trend (do not split mid-trend).
2. **Feature engineering per trend:**
   - Compute technical indicators, price deltas, momentum.
   - Add "memory features" for past profitable trades in this trend or prior trends.
3. **Combine memory and trend data:**
   - Merge the replay buffer with the current trend.
   - Optionally augment memory trades (slight shifts) to increase sample size.
4. **Train/Test split:**
   - Chronological / trend-aware split: train on past trends, test on later trends.
   - Each trend is either fully in train or test.

## 4. When to Use Memory
- **During training:** Always include memory trades with current trend data.
- **During inference:** Use "memory features" as input (e.g., last N good trades) to guide predictions.

---

**Benefits:**
- Learn from few trend-start samples.
- Keep "good trade patterns" alive in memory.
- Avoid leakage by respecting trend structure.
