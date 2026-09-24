"""
FinRL Tutorial — Train RL agents on stock trading data.

Architecture note (per FRO-8 review):
- Lives in research/ only. Not imported by system core.
- SB3 (StableBaselines3) is the primary dependency.
- FinRL StockTradingEnv (gym.Env subclass) used directly — not FinRL's
  high-level wrappers — as the arch review specified.
- PyTorch backend (not TF).
- Data fetched via yfinance directly (FinRL's YahooDownloader incompatible
  with yfinance >0.2).

After running, see DESIGN_NOTES.md for ETS agent lifecycle mapping.
"""

from __future__ import annotations

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt

import yfinance as yf
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

os.makedirs("research/finrl-experiments/models", exist_ok=True)
os.makedirs("research/finrl-experiments/results", exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
TRAIN_START = "2015-01-01"
TRAIN_END   = "2022-01-01"
TEST_START  = "2022-01-01"
TEST_END    = "2023-06-30"

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]

INITIAL_AMOUNT = 100_000
TRANSACTION_COST_PCT = 0.001
HMAX = 100
TIMESTEPS = 50_000

# Technical indicators to compute manually (replacing FinRL's FeatureEngineer)
TECH_INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30"]


# ── Step 1: Download historical data via yfinance directly ─────────────────
print("=" * 60)
print("Step 1: Downloading historical price data via yfinance")
print("=" * 60)

all_frames = []
for ticker in TICKERS:
    raw = yf.download(ticker, start=TRAIN_START, end=TEST_END, auto_adjust=True, progress=False)
    raw = raw.reset_index()
    raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in raw.columns]
    raw["tic"] = ticker
    raw = raw.rename(columns={"date": "date"})
    all_frames.append(raw)

df = pd.concat(all_frames, ignore_index=True)
df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
df = df.sort_values(["date", "tic"]).reset_index(drop=True)

print(f"  Downloaded {len(df)} rows, {df['tic'].nunique()} tickers")
print(f"  Date range: {df['date'].min()} → {df['date'].max()}")


# ── Step 2: Compute technical indicators ───────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Computing technical indicators")
print("=" * 60)

def compute_indicators(ticker_df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD, RSI, CCI, DX for one ticker."""
    close = ticker_df["close"]
    high  = ticker_df["high"]
    low   = ticker_df["low"]

    # MACD = EMA12 - EMA26
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ticker_df["macd"] = ema12 - ema26

    # RSI-30
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(30).mean()
    loss   = (-delta.clip(upper=0)).rolling(30).mean()
    rs     = gain / (loss + 1e-8)
    ticker_df["rsi_30"] = 100 - (100 / (1 + rs))

    # CCI-30
    tp = (high + low + close) / 3
    ma = tp.rolling(30).mean()
    md = tp.rolling(30).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    ticker_df["cci_30"] = (tp - ma) / (0.015 * md + 1e-8)

    # DX-30 (simplified: Wilder's +DI/-DI)
    tr  = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(30).mean()
    ticker_df["dx_30"] = (atr / (close + 1e-8)) * 100

    return ticker_df

processed = []
for tic in TICKERS:
    sub = df[df["tic"] == tic].copy().reset_index(drop=True)
    sub = compute_indicators(sub)
    processed.append(sub)

processed_df = pd.concat(processed).sort_values(["date", "tic"]).reset_index(drop=True)
processed_df = processed_df.fillna(0).replace([np.inf, -np.inf], 0)

print(f"  Indicators computed: {TECH_INDICATORS}")
print(f"  Final shape: {processed_df.shape}")


# ── Step 3: Train/test split ────────────────────────────────────────────────
def apply_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """Set integer day index required by StockTradingEnv.
    All tickers on the same date share the same index value so
    df.loc[day] returns a DataFrame (multi-row) for multi-stock mode.
    """
    dates = sorted(df["date"].unique())
    date_to_day = {d: i for i, d in enumerate(dates)}
    out = df.copy()
    out.index = out["date"].map(date_to_day)
    return out

train_data = apply_day_index(processed_df[processed_df["date"] < TRAIN_END].copy())
test_data  = apply_day_index(processed_df[processed_df["date"] >= TEST_START].copy())
print(f"\n  Train rows: {len(train_data)} | Test rows: {len(test_data)}")


# ── Step 4: Build trading environment ──────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Building StockTradingEnv (gym.Env via SB3/FinRL)")
print("=" * 60)

stock_dim  = len(TICKERS)
state_space = 1 + 2 * stock_dim + len(TECH_INDICATORS) * stock_dim
action_space = stock_dim

env_kwargs = {
    "hmax": HMAX,
    "initial_amount": INITIAL_AMOUNT,
    "num_stock_shares": [0] * stock_dim,
    "buy_cost_pct": [TRANSACTION_COST_PCT] * stock_dim,
    "sell_cost_pct": [TRANSACTION_COST_PCT] * stock_dim,
    "state_space": state_space,
    "stock_dim": stock_dim,
    "tech_indicator_list": TECH_INDICATORS,
    "action_space": action_space,
    "reward_scaling": 1e-4,
}

train_env = DummyVecEnv([lambda: StockTradingEnv(df=train_data, **env_kwargs)])

print(f"  State space dim: {state_space}")
print(f"  Action space dim: {action_space}")
print(f"  Training period: {TRAIN_START} → {TRAIN_END}")


# ── Step 5: Train agents ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 5: Training agents ({TIMESTEPS:,} timesteps each)")
print("=" * 60)

# A2C
print("\n  [A2C] Training...")
a2c_model = A2C("MlpPolicy", train_env, learning_rate=7e-4, n_steps=5, verbose=0)
a2c_model.learn(total_timesteps=TIMESTEPS)
a2c_model.save("research/finrl-experiments/models/a2c_stock_trader")
print("  [A2C] Done")

# PPO
print("\n  [PPO] Training...")
ppo_model = PPO("MlpPolicy", train_env, learning_rate=3e-4, n_steps=2048, batch_size=64, verbose=0)
ppo_model.learn(total_timesteps=TIMESTEPS)
ppo_model.save("research/finrl-experiments/models/ppo_stock_trader")
print("  [PPO] Done")

# DDPG
print("\n  [DDPG] Training...")
n_actions = train_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
ddpg_model = DDPG("MlpPolicy", train_env, action_noise=action_noise, learning_rate=1e-3, batch_size=100, verbose=0)
ddpg_model.learn(total_timesteps=TIMESTEPS)
ddpg_model.save("research/finrl-experiments/models/ddpg_stock_trader")
print("  [DDPG] Done")


# ── Step 6: Backtesting ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6: Backtesting on held-out test data")
print("=" * 60)

results: dict[str, dict] = {}

def backtest(model, model_name: str) -> None:
    env_raw  = StockTradingEnv(df=test_data, **env_kwargs)
    env_test = DummyVecEnv([lambda: StockTradingEnv(df=test_data, **env_kwargs)])

    obs = env_test.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env_test.step(action)

    portfolio_values = env_raw.asset_memory if hasattr(env_raw, "asset_memory") else []

    # Use actual env via manual step for portfolio tracking
    env2 = StockTradingEnv(df=test_data, **env_kwargs)
    obs2, _ = env2.reset() if hasattr(env2.reset(), "__len__") else (env2.reset(), None)
    # Gymnasium-style reset returns (obs, info); old gym returns obs
    reset_out = env2.reset()
    obs2 = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    portfolio_track = [INITIAL_AMOUNT]
    done2 = False
    while not done2:
        obs_arr = np.array(obs2) if not isinstance(obs2, np.ndarray) else obs2
        act, _ = model.predict(obs_arr)
        step_result = env2.step(act)
        if len(step_result) == 5:
            obs2, rew, term, trunc, info = step_result
            done2 = bool(term) or bool(trunc)
        else:
            obs2, rew, done2, info = step_result
            done2 = bool(done2)
        portfolio_track.append(env2.asset_memory[-1])

    final_value  = portfolio_track[-1]
    total_return = (final_value - INITIAL_AMOUNT) / INITIAL_AMOUNT * 100
    daily_returns = np.diff(portfolio_track) / (np.array(portfolio_track[:-1]) + 1e-8)
    sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)) * np.sqrt(252)

    print(f"  [{model_name}]")
    print(f"    Final portfolio: ${final_value:>12,.0f}")
    print(f"    Total return:    {total_return:>8.1f}%")
    print(f"    Sharpe ratio:    {sharpe:>8.2f}")

    results[model_name] = {
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(float(sharpe), 4),
        "portfolio_values": portfolio_track,
    }

backtest(a2c_model,  "A2C")
backtest(ppo_model,  "PPO")
backtest(ddpg_model, "DDPG")


# ── Step 7: Plot and save ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 7: Saving backtest chart")
print("=" * 60)

fig, ax = plt.subplots(figsize=(12, 6))
for name, data in results.items():
    ax.plot(data["portfolio_values"], label=f"{name} (ret={data['total_return_pct']:.1f}%, Sharpe={data['sharpe_ratio']:.2f})")

ax.axhline(y=INITIAL_AMOUNT, color="gray", linestyle="--", alpha=0.5, label="Buy-and-hold baseline ($100k)")
ax.set_title(f"FinRL Tutorial Backtest — {TEST_START} to {TEST_END}\nA2C / PPO / DDPG on {', '.join(TICKERS)}")
ax.set_xlabel("Trading Days")
ax.set_ylabel("Portfolio Value ($)")
ax.legend()
ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

chart_path = "research/finrl-experiments/results/backtest_comparison.png"
plt.tight_layout()
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"  Chart saved: {chart_path}")

summary_df = pd.DataFrame([
    {"model": name, "final_value": d["final_value"], "total_return_pct": d["total_return_pct"], "sharpe_ratio": d["sharpe_ratio"]}
    for name, d in results.items()
])
csv_path = "research/finrl-experiments/results/backtest_summary.csv"
summary_df.to_csv(csv_path, index=False)
print(f"  Summary CSV:  {csv_path}")
print("\n" + summary_df.to_string(index=False))


print("\n" + "=" * 60)
print("TUTORIAL COMPLETE")
print("=" * 60)
print(f"  Agents trained:  A2C, PPO, DDPG ({TIMESTEPS:,} steps each)")
print(f"  Models:          research/finrl-experiments/models/")
print(f"  Results:         research/finrl-experiments/results/")
print("\nSee DESIGN_NOTES.md for ETS lifecycle mapping.")
