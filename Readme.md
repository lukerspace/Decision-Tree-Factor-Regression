## Multi-Factor Decision Tree Backtesting Framework
### Purpose:
--------------------------------
This program builds a complete quantitative experiment pipeline:
1. Generate **simulated multi-factor stock data** (pseudo panel).
2. Train a **Decision Tree Regressor** to predict next-month returns.
3. Perform **walk-forward backtesting** to simulate real-world out-of-sample prediction.
4. Construct **long-short portfolios** based on predicted returns.
5. Evaluate model stability using **Mean Squared Error (MSE)**, **R²**, and **feature importance**.


### Key Concepts:
--------------------------------
- **Multi-factor model**: Uses factors (value, momentum, size, etc.) to predict stock returns. (historic ground value)
- **Walk-forward validation**: Simulates a time series backtest without look-ahead bias.
- **Cross-validation**: Tunes model hyperparameters to minimize out-of-sample error (MSE).
- **Decision Tree**: Captures nonlinear relationships among financial factors.

### Workflow Steps:
--------------------------------
1. Data Simulation → `generate_pseudo_factor_panel`
2. Model Training + Tuning → `train_and_backtest_decision_tree_model`
3. Portfolio Construction → Long-Short ranking by predicted return
4. Performance Evaluation → Save metrics, visualize MSE & return trends
