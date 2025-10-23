# Copilot Instructions for AI Agents

## Project Overview
- **Domain:** Reinforcement Learning (RL) for algorithmic trading using PPO agents.
- **Major Components:**
  - `src/` — Core source code, including RL model, data loading, technical analysis, environments, and reporting.
  - `models/` — Saved RL models and logs.
  - `data/` — Historical OHLCV data (JSON) for multiple symbols/timeframes.
  - `tests/` — Unit and functional tests for trading logic and broker.

## Key Architectural Patterns
- **Modular Design:**
  - `src/prediction/rl_predictor.py`: Main entry for RL model training/inference. Uses PPO (Stable-Baselines3/RecurrentPPO).
  - `src/training/data_loader.py`: Loads and preprocesses historical data, supports multiple timeframes.
  - `src/ta/technical_analysis.py` & `src/ta/middlewares/`: Middleware pattern for feature engineering (e.g., zigzag, volume profile).
  - `src/environments/trading_environment.py`: Custom Gymnasium environment for RL agent, uses `TradingBroker` for trade simulation.
  - `src/reporting/`: Tools for visualizing and reporting training results.

## Developer Workflows
- **Training:**
  - Use notebooks (e.g., `RL_Trading_Model_Training.ipynb`) or scripts to train models via `RLPredictor`.
  - Data is loaded from `data/`, features are precomputed or generated on-the-fly.
- **Feature Precomputation:**
  - Run `python src/scripts/precompute_features.py` to cache slow TA features to Parquet for faster training.
- **Data Extraction:**
  - Run `python src/scripts/binance_data_extractor.py` to fetch/update OHLCV data from Binance.
- **Testing:**
  - Run all tests: `python tests/run_tests.py` (uses pytest, verbose output).
- **Reporting:**
  - Use `src/reporting/model_training_report.py` and `src/reporting/training_log_visualizer.py` for analysis/visualization.

## Project-Specific Conventions
- **Data:**
  - All raw data in `data/{SYMBOL}-{TIMEFRAME}.json`.
  - Precomputed features in `data/levels_cache/` (Parquet).
- **Models:**
  - Saved in `models/rl_optimized/` or similar, with logs for TensorBoard/CSV.
- **Feature Engineering:**
  - Middleware pattern: add new TA features as a middleware in `src/ta/middlewares/`.
- **Config:**
  - Feature and environment configs are code-based (see `src/core/normalization_config.py`).
- **GPU Support:**
  - Training and feature computation can use GPU if available (torch, numpy).

## Integration Points
- **Stable-Baselines3** for RL agent (PPO, RecurrentPPO).
- **Gymnasium** for environment interface.
- **Pandas/Numpy** for data manipulation.
- **Matplotlib/Plotly** for reporting/visualization.

## Examples
- To train a model: see `RL_Trading_Model_Training.ipynb` or use `RLPredictor` directly.
- To add a new technical indicator: implement as a middleware in `src/ta/middlewares/` and register in `technical_analysis.py`.
- To run all tests: `python tests/run_tests.py`.

---

**For AI agents:**
- Prefer code-based configuration over static config files.
- Follow the modular structure; do not hardcode paths.
- Use the provided data/model directories for I/O.
- When in doubt, check the relevant module docstrings for usage patterns.
