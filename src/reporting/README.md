# Reporting Module

Comprehensive reporting and analysis tools for trading model training and evaluation.

## Overview

The reporting module provides classes and utilities for generating detailed training reports, performance analysis, and visualization for ML/RL trading models.

## Main Components

### `ModelTrainingReport`

The core class for generating comprehensive RL model training reports.

#### Features:
- **Model Status Checking**: Verifies model files exist and can be loaded
- **Training Metrics Creation**: Extracts and organizes training data
- **Performance Analysis**: Analyzes trading signals and strategy performance
- **Advanced Visualizations**: Creates training dashboards and performance charts
- **Session Documentation**: Generates complete training session summaries

#### Basic Usage:

```python
from src.reporting import ModelTrainingReport

# Initialize report generator
report = ModelTrainingReport(model_dir='models/rl_demo')

# Check if model was saved correctly
model_exists = report.check_model_status()

# Create training metrics
metrics = report.create_training_metrics(features_df, feature_columns)

# Analyze trading performance
performance = report.analyze_trading_performance(signals_df)

# Generate visualizations
report.create_training_visualization()      # Training dashboard
report.create_trading_analysis(signals_df)  # Trading performance charts

# Create final session summary
summary = report.create_session_summary()
```

#### One-Step Report Generation:

```python
from src.reporting import ModelTrainingReport

report = ModelTrainingReport(model_dir='models/rl_demo')
session_summary = report.generate_complete_report(
    features_df=features_df,
    feature_columns=feature_columns,
    signals_df=signals_df
)
```

### Convenience Functions

#### `create_training_report()`

Quick way to create a `ModelTrainingReport` instance:

```python
from src.reporting import create_training_report

report = create_training_report('models/my_model')
report.check_model_status()
```

#### `quick_report()`

Generate a complete report in one function call:

```python
from src.reporting import quick_report

summary = quick_report(
    features_df, 
    feature_columns, 
    signals_df, 
    model_dir='models/rl_demo'
)
```

## Generated Files

The reporting module automatically creates these files in your model directory:

### Training Data
- `training_metrics.json` - Comprehensive training metrics and configuration
- `metrics.json` - Visualization-ready metrics for plotting

### Visualizations
- `training_analysis.png` - 4-panel training dashboard (Balance, Rewards, FPS, Entropy)
- `rl_performance_analysis.png` - Trading performance charts with signals

### Documentation
- `session_summary.json` - Complete training session summary with metadata

## Example Integration

### In Jupyter Notebooks:

```python
# Import the reporting module
from src.reporting import ModelTrainingReport

# After training your RL model...
report_generator = ModelTrainingReport(model_dir='models/rl_demo')

# Generate step-by-step analysis
model_saved = report_generator.check_model_status()
if model_saved:
    metrics = report_generator.create_training_metrics(features_df, feature_columns)
    performance = report_generator.analyze_trading_performance(signals_df)
    report_generator.create_training_visualization()
    report_generator.create_trading_analysis(signals_df)
    summary = report_generator.create_session_summary()
```

### In Scripts:

```python
from src.reporting import quick_report

def train_and_report():
    # ... your training code ...
    
    # Generate complete report
    summary = quick_report(features_df, feature_columns, signals_df)
    print(f"Training complete! Report saved to: {summary['model_info']['filename']}")
```

## Customization

### Custom Training Logs

You can pass custom training logs to override defaults:

```python
custom_logs = {
    'total_timesteps': 50000,
    'device': 'cuda',
    'progression': [
        {'iteration': 1, 'timesteps': 10000, 'fps': 2000, 'entropy_loss': -1.5},
        # ... more iterations
    ],
    'final_performance': {
        'fps': 1800,
        'policy_loss': -0.002,
        'value_loss': 0.0005,
        # ... other metrics
    }
}

metrics = report.create_training_metrics(
    features_df, 
    feature_columns, 
    training_logs=custom_logs
)
```

### Extending the Reporter

You can subclass `ModelTrainingReport` to add custom analysis methods:

```python
class CustomTrainingReport(ModelTrainingReport):
    def create_risk_analysis(self, signals_df):
        # Your custom risk analysis
        pass
    
    def compare_models(self, other_model_dir):
        # Compare with another model
        pass
```

## Requirements

The reporting module requires:
- `pandas` - Data manipulation
- `matplotlib` - Plotting and visualization
- `stable_baselines3` - For loading PPO models
- `pathlib` - File system operations
- `json` - Data serialization

## Version History

- **1.0.0** - Initial release with complete RL training report functionality