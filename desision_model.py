import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_SEED = 42
NUM_STOCKS = 100              # Number of simulated stocks 100 here 
NUM_MONTHS = 120              # 10 years of monthly data
BACKTEST_START_MONTH = 84     # First backtest point (after 7 years of training)
TREE_DEPTH_CANDIDATES = [3, 5, 7]
MIN_LEAF_SAMPLES_CANDIDATES = [30, 50, 100]
TOP_QUANTILE = 0.1            # Top 10% for long
BOTTOM_QUANTILE = 0.1         # Bottom 10% for short
np.random.seed(RANDOM_SEED)


def generate_pseudo_factor_panel(num_stocks=100, num_months=120, seed=42):
    rng = np.random.default_rng(seed)
    monthly_dates = pd.date_range("2015-01-31", periods=num_months, freq="M")
    stock_ids = np.arange(1, num_stocks + 1)
    all_monthly_data = []

    # Define true hidden factor weights (market's "true beta" fake data here)
    true_factor_weights = {
        "value": 0.34,
        "momentum": 0.39,
        "size": -0.21,
        "quality": 0.25,
        "volatility": -0.14,
        "liquidity": 0.11
    }

    for current_date in monthly_dates:
        # Generate cross-sectional factor values and normalize 
        def zscore(x): return (x - x.mean()) / (x.std(ddof=0) + 1e-9)
        factor_dict = {
            factor: zscore(rng.normal(0, 1, num_stocks))
            for factor in true_factor_weights.keys()
        }

        # Introduce nonlinear terms + noise
        nonlinear_effect = (
            0.10 * (factor_dict["momentum"] * (-factor_dict["volatility"])) -
            0.05 * (factor_dict["liquidity"] ** 2) +
            0.08 * np.maximum(factor_dict["value"], 0) * np.maximum(factor_dict["quality"], 0)
        )
        random_noise = rng.normal(0, 0.25, num_stocks)

        # Generate next-month returns based on linear + nonlinear structure
        linear_combination = sum(
            true_factor_weights[f] * factor_dict[f] for f in true_factor_weights.keys()
        )
        next_month_return = (linear_combination + nonlinear_effect + random_noise) * 0.02

        # Build DataFrame for this month
        monthly_df = pd.DataFrame({
            "date": current_date,
            "stock_id": stock_ids,
            **factor_dict,
            "next_month_return": next_month_return
        })
        all_monthly_data.append(monthly_df)

    factor_panel_df = pd.concat(all_monthly_data, ignore_index=True)
    return factor_panel_df #產生全部資料



def train_and_backtest_decision_tree_model(
    factor_panel_df: pd.DataFrame,
    start_backtest_month: int = 84,
    depth_candidates: list = [3, 5, 7],
    leaf_candidates: list = [30, 50, 100],
    top_q: float = 0.1,
    bot_q: float = 0.1
):
    
    factor_columns = ["value", "momentum", "size", "quality", "volatility", "liquidity"]
    monthly_perf_records, model_info_records, feature_importance_records = [], [], []

    factor_panel_df = factor_panel_df.sort_values(["date", "stock_id"]).reset_index(drop=True)
    all_dates = factor_panel_df["date"].drop_duplicates().sort_values().to_list()
    tscv = TimeSeriesSplit(n_splits=5)

    # Rolling month-by-month backtest
    for month_index, current_date in enumerate(all_dates):
        if month_index < start_backtest_month:
            continue

        # Split training and test sets
        train_df = factor_panel_df[factor_panel_df["date"] < current_date]
        test_df = factor_panel_df[factor_panel_df["date"] == current_date]

        X_train, y_train = train_df[factor_columns].values, train_df["next_month_return"].values
        X_test, y_test = test_df[factor_columns].values, test_df["next_month_return"].values

        # ---- Hyperparameter tuning (cross-validation) ----
        best_cv_mse, best_params = float("inf"), {}
        for max_depth in depth_candidates:
            for min_leaf in leaf_candidates:
                fold_mses = []
                for tr_idx, val_idx in tscv.split(X_train):
                    model = DecisionTreeRegressor(
                        criterion="squared_error",
                        max_depth=max_depth,
                        min_samples_leaf=min_leaf,
                        random_state=RANDOM_SEED
                    )
                    model.fit(X_train[tr_idx], y_train[tr_idx])
                    fold_mses.append(mean_squared_error(y_train[val_idx], model.predict(X_train[val_idx])))
                avg_mse = np.mean(fold_mses)
                if avg_mse < best_cv_mse:
                    best_cv_mse = avg_mse
                    best_params = {"max_depth": max_depth, "min_samples_leaf": min_leaf}

        # ---- Final model training ----
        final_model = DecisionTreeRegressor(
            criterion="squared_error",
            max_depth=best_params["max_depth"],
            min_samples_leaf=best_params["min_samples_leaf"],
            random_state=RANDOM_SEED
        )
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        # ---- Model performance metrics ----
        in_mse = mean_squared_error(y_train, final_model.predict(X_train))
        out_mse = mean_squared_error(y_test, y_pred)
        in_r2 = r2_score(y_train, final_model.predict(X_train))
        out_r2 = r2_score(y_test, y_pred)

        # ---- Portfolio construction (top/bottom quantile) ----
        test_df = test_df.assign(predicted_return=y_pred)
        top_threshold = test_df["predicted_return"].quantile(1 - top_q)
        bottom_threshold = test_df["predicted_return"].quantile(bot_q)

        long_df = test_df[test_df["predicted_return"] >= top_threshold]
        short_df = test_df[test_df["predicted_return"] <= bottom_threshold]

        long_ret = long_df["next_month_return"].mean() if len(long_df) > 0 else 0.0
        short_ret = short_df["next_month_return"].mean() if len(short_df) > 0 else 0.0
        long_short_ret = long_ret - short_ret

        monthly_perf_records.append({
            "date": current_date,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "long_short_ret": long_short_ret
        })
        model_info_records.append({
            "date": current_date,
            "in_sample_mse": in_mse,
            "out_sample_mse": out_mse,
            "in_sample_r2": in_r2,
            "out_r2": out_r2,
            "max_depth": best_params["max_depth"],
            "min_samples_leaf": best_params["min_samples_leaf"],
            "num_tree_nodes": final_model.tree_.node_count
        })
        feature_importance_records.append(
            dict(date=current_date, **{f"importance_{f}": imp for f, imp in zip(factor_columns, final_model.feature_importances_)})
        )

    # Combine into DataFrames
    perf_df = pd.DataFrame(monthly_perf_records).set_index("date")
    model_diagnostics_df = pd.DataFrame(model_info_records).set_index("date")
    feature_importance_df = pd.DataFrame(feature_importance_records).set_index("date")

    # Compute cumulative performance
    perf_df["cum_long_short_ret"] = (1 + perf_df["long_short_ret"]).cumprod() - 1
    return perf_df, model_diagnostics_df, feature_importance_df



# 主程式段落 利用假資料 成立多空組 用決策數將預測最好跟最差的分群做spread~
if __name__ == "__main__":
    # Step 1. Generate pseudo market data
    factor_panel_df = generate_pseudo_factor_panel(NUM_STOCKS, NUM_MONTHS, seed=RANDOM_SEED)

    # Step 2. Train model and run backtest
    perf_df, model_diagnostics_df, feature_importance_df = train_and_backtest_decision_tree_model(
        factor_panel_df,
        start_backtest_month=BACKTEST_START_MONTH,
        depth_candidates=TREE_DEPTH_CANDIDATES,
        leaf_candidates=MIN_LEAF_SAMPLES_CANDIDATES,
        top_q=TOP_QUANTILE,
        bot_q=BOTTOM_QUANTILE
    )

    # Step 3. Summary report
    print("\n=== Portfolio Performance Summary ===")
    print(perf_df[["long_ret", "short_ret", "long_short_ret"]].mean().rename("Mean Monthly Return"))

    print("\n=== Final Cumulative Return ===")
    print(perf_df["cum_long_short_ret"].iloc[-1])

    print("\n=== Model Diagnostics (last 5 months) ===")
    print(model_diagnostics_df.tail(30))
    

    print("\n=== Average Feature Importances ===")
    print(feature_importance_df.mean().sort_values(ascending=False))

    # === Long-Short spread significance test (Out-of-Sample) ===
    # H0: mean(long_short_ret) = 0  vs.  H1: mean != 0
    oos_spread = perf_df["long_short_ret"].dropna().values
    n = len(oos_spread)
    mean_m = float(np.mean(oos_spread))
    std_m = float(np.std(oos_spread, ddof=1))
    se_m = std_m / np.sqrt(n) if n > 0 else np.nan
    t_stat = mean_m / se_m if se_m > 0 else np.nan

    try:
        from scipy.stats import t as student_t
        p_value = 2 * student_t.sf(abs(t_stat), df=n-1)
        p_note = "p-value (Student-t)"
    except Exception:
        # Normal approximation fallback if scipy not installed
        from math import erf, sqrt
        z = abs(t_stat)
        p_value = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
        p_note = "p-value (Normal approx)"

    # Annualized metrics (approximate)
    ann_mean = mean_m * 12
    ann_vol = std_m * np.sqrt(12)
    ann_sharpe = ann_mean / ann_vol if ann_vol > 0 else np.nan

    print("\n=== Long-Short Spread Significance (Out-of-Sample) ===")
    print(f"Samples (months): {n}")
    print(f"Monthly mean spread: {mean_m:.6f}")
    print(f"Monthly std:         {std_m:.6f}")
    print(f"t-statistic:         {t_stat:.3f}")
    print(f"{p_note}:            {p_value:.4g}")
    print(f"Annualized mean:     {ann_mean:.6f}")
    print(f"Annualized vol:      {ann_vol:.6f}")
    print(f"Annualized Sharpe:   {ann_sharpe:.3f}")
    if p_value < 0.05:
        print("=> 結論：在 5% 顯著水準下，長期多空 spread 的平均『顯著不為 0%』。")
    else:
        print("=> 結論：無法在 5% 顯著水準下拒絕平均為 0% 的假設。")

    try:
        import matplotlib.pyplot as plt

        # Plot cumulative long-short performance
        plt.figure(figsize=(10, 6))
        perf_df["cum_long_short_ret"].plot(title="Cumulative Long-Short Portfolio Return")
        plt.xlabel("Date"); plt.ylabel("Cumulative Return")
        plt.tight_layout(); plt.show()

        # Plot average factor importance
        plt.figure(figsize=(10, 6))
        feature_importance_df.mean().plot(kind="bar", title="Average Feature Importances")
        plt.tight_layout(); plt.show()

        # Plot out-of-sample MSE trend
        plt.figure(figsize=(10, 6))
        model_diagnostics_df["out_sample_mse"].plot(title="Out-of-Sample MSE Over Time")
        plt.xlabel("Date"); plt.ylabel("Out-of-Sample MSE")
        plt.tight_layout(); plt.show()

    except Exception as e:
        print("\n[Visualization skipped due to error]", e)
