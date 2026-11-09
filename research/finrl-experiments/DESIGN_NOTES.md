---
title: FinRL → ETS Agent Lifecycle Mapping
date: 2026-09-24
tags: [finrl, rl, ets, design, architecture]
---

## Question

How does a FinRL-trained policy map to the ETS (Evolutionary Trading System) agent lifecycle?

## FinRL's Model of the World

A FinRL agent is a single SB3 policy trained against a `gym.Env` (specifically `StockTradingEnv`):

```
policy = PPO.train(env)      # one policy, one training run
action = policy.predict(obs) # deterministic inference
```

The policy is a neural network. It is frozen after training. There is no concept of competition, cloning, mutation, or elimination.

## ETS's Model of the World

```
agents = [Agent1, Agent2, ..., AgentN]
each agent: { policy, capital_allocation, fitness_score }

per epoch:
  - all agents trade
  - rank by performance (Sharpe, returns, drawdown)
  - top K → clone with mutation (new policy = parent.weights + noise)
  - bottom M → eliminate
  - reallocate capital proportional to fitness
```

## The Mapping

| FinRL concept | ETS concept |
|---|---|
| `PPO("MlpPolicy", env)` | The **policy architecture** used by each agent |
| Trained policy weights | An **agent's genome** — what gets mutated on clone |
| `policy.predict(obs)` | Each agent's **action** method |
| Episode reward | **Fitness score** for the epoch |
| Training timesteps | **One epoch** of the competition loop |

## Key Design Decisions for ETS

### 1. Policy as genome

Each ETS agent holds SB3 policy weights as its genome. Mutation = apply Gaussian noise to policy parameters:

```python
def mutate(parent_policy: nn.Module, sigma: float = 0.01) -> nn.Module:
    child = copy.deepcopy(parent_policy)
    with torch.no_grad():
        for param in child.parameters():
            param.add_(torch.randn_like(param) * sigma)
    return child
```

### 2. The ETS epoch ≠ FinRL training

FinRL's `learn(timesteps)` is **gradient-descent training**. ETS epochs are **evolutionary selection** — no gradient, just run-and-score. Each agent runs inference-only in its epoch; only the population composition changes between epochs.

For ETS, the training loop should use SB3's `policy.predict()` without calling `.learn()`:

```python
for agent in agents:
    obs = env.reset()
    for step in range(epoch_length):
        action, _ = agent.policy.predict(obs)
        obs, reward, done, _ = env.step(action)
        agent.fitness += reward
```

### 3. Bootstrapping — seeding the first generation

Use FinRL (or SB3 directly) to pre-train a population seed. Run A2C/PPO/DDPG for ~50k steps to get a non-random starting policy, then spawn N clones with mutations as generation 0. This avoids cold-start random behavior for the first epoch.

### 4. Use SB3 env interface directly

The `StockTradingEnv` is a `gymnasium.Env` subclass. Use it directly for ETS without FinRL's high-level wrappers:

```python
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
# Not: from finrl.train import train
# Not: from finrl.meta.data_processor import DataProcessor
```

FinRL's data pipeline (YahooDownloader, FeatureEngineer) can be replaced with direct yfinance + pandas indicator code (as done in `train_tutorial.py`) for compatibility and control.

### 5. Action space compatibility

`StockTradingEnv` uses `Box(-1, 1, shape=(n_stocks,))` for actions. This works for continuous RL agents (PPO, DDPG). The ETS mutation operator must preserve this constraint (e.g., clip or tanh output layer).

## What FinRL Is NOT Suitable For in ETS

- **Multi-agent competition** — no native support; must be built around it
- **Capital reallocation** — not in FinRL's scope; ETS-native concern
- **Clone lifecycle** — FinRL has no agent registry
- **Real-time trading** — FinRL's paper trading (AlpacaPaperTrading) depends on a broker API; ETS paper trading should use a lightweight local simulator

## Next Steps

1. Implement `EvolAgent` wrapper around SB3 policy with `mutate()` and `fitness` tracking
2. Implement `EvolPopulation` with epoch loop, ranking, and capital allocation
3. Write `StockTradingEnv`-compatible multi-agent runner
4. Use FinRL tutorial results as generation-0 seed policies
