Link to Colab File (for easier viewing):

https://colab.research.google.com/drive/1eE7JzppeJrcPkVWnaFo_16kr4Ld96pZ-?usp=sharing 

# Portfolio Optimization using Reinforcement Learning
### Srihan Kakarlapudi
## Introduction

In this project, we investigate whether contemporary reinforcement learning (RL) techniques can reliably convert a comprehensive library of predictive “alpha” signals into a robust portfolio allocation strategy. Our work builds on the “101 Formulaic Alphas” framework introduced by Kakushadze and Lee (2016), which codifies a diverse set of quantitative equity factors—momentum, mean-reversion, cross-sectional, and liquidity measures—into closed-form formulas. We aim to test whether these formulaic alphas, many predating modern machine learning in finance, still carry exploitable predictive power when ingested by an RL agent.

To do so, we built a custom trading environment in Python, leveraging pandas and NumPy for data wrangling and computation, Gym (via Stable-Baselines3) to structure reinforcement-learning episodes, and SciPy for statistical diagnostics. We sourced individual stock data from Investing.com CSVs (parsed locally) and used a Kaggle S&P 500 dataset for universe-wide tests. Our study unfolds across four progressively challenging phases:

- Phase 1 – train a PPO agent on TSLA alone, uncover look-ahead bias in 60 % in-sample gains, then correct it to achieve ~3 % out-of-sample return when tested on 2024 data (and ~20 % when shifting the training window to 2021–2024).
- Phase 2 – scale to ~170 S&P 500 constituents, compute and standardize ~70 alphas per stock per day, and train PPO in-sample (2020–2023) with no meaningful out-of-sample edge in 2024.
- Phase 3 – fit a least-squares “mega-alpha” by regressing returns on the 70 alphas, diagnose its information coefficient (≈ 0) and hit rate (~ 50 %), and conclude that no RL training is warranted.
- Phase 4 – reduce to ten well-known technical indicators on TSLA, redesign the environment with volatility and drawdown penalties, train on 2010–2023, and achieve an out-of-sample Sharpe ≈ 1.3 and max drawdown ≈ 12 % on 2024 data.

### Brief overview of how Proximal Policy Optimization works:

Proximal Policy Optimization (PPO) is an on-policy, actor-critic method that balances stable learning with implementation simplicity. In each iteration, you first execute the current policy in the environment to collect a batch of state-action-reward trajectories. You then compute an “advantage” for each action—a statistical estimate of how much better that action turned out compared to the policy’s own value prediction. The policy update maximizes a surrogate objective that uses the ratio of new to old action probabilities, but “clips” that ratio within a small interval (e.g. ±10 %) so that no single update can move the policy too far and introduce instability. Simultaneously, you fit a value function by least squares to the observed discounted returns, and you can optionally add a small entropy term to encourage exploration. By alternating between collecting fresh data and performing multiple epochs of these clipped, gradient-based updates, PPO achieves reliable policy improvements without requiring complex second-order optimization or hard trust-region constraints.

## Data Ingestion & 101 Formulaic Alphas

1. Raw data loading
  - Read OHLCV CSVs (Investing.com) into pandas, parse Date as datetime index.
  - Standardize column names to Open, High, Low, Close, Volume.
  - Forward-fill missing Close/Open/… to maintain continuity.
2. Return calculation
`df['return'] = df['Close'].pct_change().fillna(0.0)`
3. Alpha computation
  - Implemented time-series helpers: ts_sum, sma, stddev, sliding correlations/covariances, ts_rank/rank, decay_linear, delta, etc., matching Kakushadze & Lee’s formulas.
  - Encapsulated in a FormulaicAlphas class; each alpha###() returns a pandas Series.
  - Skipped any alpha requiring unavailable data.
  - Assembled surviving signals into a DataFrame df_alphas of shape T×M, M≤101.
4. Preprocessing
  - Filled any remaining NaNs/infs with zero.
  - Standardized each column to zero mean and unit variance.

## ​​Trading Environment Design

#### Our TradingEnv (subclass of gym.Env) defines:
- Observation:
  st=[at-1,wt-1]​
  where at-1RM is the standardized alpha vector at t−1.
- Action:
  wt[-1,1]M, raw portfolio weights over M signals, then normalized:
  wtwti|wt,i| to enforce |wt,i|=1.
- Transition & PnL:
  Scalar trade signal xt=at-1 wt [-1,1].
  PnL on day t:
  PnLt=balancet-1xtrt, with rt= next-day percent change.
- Reward:
-   Phases 1–2:
    reward = PnLt(no extra penalties).
-   Phases 4:
    Rt​=PnLt​−λvol​σt​−μdd​DDt​
    With σt​ = recent volatility, DDt​ = peak-to-trough drawdown.
- Tracking: balance history, cumulative returns, drawdown time series for analysis.

## Phase 1: Single-Ticker PPO Backtests

In Phase 1, we validate our end‐to‐end alpha‐to‐trade pipeline on Tesla (TSLA). We begin by loading a CSV of TSLA’s daily OHLCV data—including parsing “M” and “K” suffixes in the Volume column—and forward-filling any gaps to avoid introducing look-ahead bias. We compute simple daily returns as the close-to-close percent change (pct_change().fillna(0.0)); these returns drive the environment’s PnL.

Next, we instantiate FormulaicAlphas, passing in the five core series (Open, High, Low, Close, and Volume×100 for numerical stability). Using over forty custom helper functions—rolling sums (ts_sum), moving averages (sma), standard deviations (stddev), correlations/covariances, rank transforms (ts_rank, rank), linear decays (decay_linear), deltas, delays, and more—we dynamically call every alpha###() method via Python’s inspect. Any alpha that raises an error (due to missing data or incompatible functions) is skipped, yielding an N×M matrix of factors (for a 2024‐only file, N≈252 days and M typically ~70 alphas). We then replace any remaining NaNs or infinities with zero and standardize each column to zero mean and unit variance.

Our custom TradingEnv uses this standardized alpha matrix as its observation space. At each step tt, the agent observes αt-1, outputs a raw weight vector wt[-1,1]M, and we normalize wt​ so i|wt,i| =1. The scalar trade signal xt=at-1wt is then multiplied by the current balance and TSLA’s next‐day return to compute PnL; the environment’s reward equals this PnL, with no additional drawdown or risk penalties in Phase 1.

An initial in-sample run on TSLA data covering 2024 produced an apparent +60% return by early May—but a detailed audit revealed look-ahead bias in several alpha formulas that used same-day closing prices. After shifting the entire alpha matrix by one day (so actions at t depend only on data through t−1), retraining PPO on the same dataset yielded a much more modest ~3% gain when evaluated on early 2025, and training on 2021–2024 produced ~20% in early 2025. Replicating this setup on SPY and other single equities, however, resulted in performance indistinguishable from random, underscoring that TSLA’s single‐asset regime alone cannot sustain a generalizable RL policy without broader diversification or stronger feature engineering

## Phase 2: Cross-Sectional S&P 500 PPO

In Phase 2, we applied our full “101-alpha” pipeline to a multi‐asset context by training and evaluating a PPO agent on ~70 alphas computed on ~170 SP500 stocks. We began by loading a CSV of SP500 daily OHLCV data from Kaggle—parsing the Date column into a datetime index, standardizing column names to Open, High, Low, Close, and Volume, converting “M”/“K” volume suffixes into numeric floats, sorting by date, and forward‐filling any missing values. We then computed next‐day returns as simple percentage changes on Close (pct_change().fillna(0.0)) and stored them in the environment for reward calculation. Using our FormulaicAlphas class, we vectorized the computation of every alpha### method via rolling sums, moving averages, standard deviations, sliding correlations and covariances, rank transforms, linear decays, deltas, delays, and more. Python’s inspect module dynamically invoked each alpha, filtering out those that errored, and we utilized a cross sectional percentile rank scaling method to utilize only meaningful alpha/stock pairs. We then replaced all NaNs and infinites with zeros, standardized each column to zero mean and unit variance, and cast the result to a NumPy float32 matrix. In our custom TradingEnv, at each time step the agent observes αt−1αt−1​, outputs a raw weight vector wt[-1,1]Mwhich is clipped and normalized so that i|wi| =1, and computes a scalar signal xt=wtat-1. Contrary to phase 1, the weight vector output weights for each stock, which would combine with alpha indicators to produce trade signals for each stock, allocating parts of the portfolio to different trades. The daily reward equals the product of the agent’s current balance, this signal, and total next‐day return; no additional drawdown or volatility penalties are applied in this phase. We trained Stable-Baselines3’s PPO for 100,000 timesteps on SPY data from 2020–2023 and then tested the saved model on 2024. The resulting equity curve remained essentially flat through the year, with an annualized Sharpe ratio near zero and drawdowns comparable to a random‐weight baseline.




These results prompted an Information Coefficient (IC) and Hit-rate analysis, which revealed that almost all of the alphas by themselves had little predictive value, and had hit rates that were equivalent or worse than random guessing.


## Phase 3: Information-Weighted Mega-Alpha

In Phase 3, we adopted a straightforward, computationally efficient approach to synthesize our universe of alphas into a single composite signal. Research suggests that combining multiple weak alphas into a single alpha may increase predictive value. First, we z-scored each individual alpha across the entire historical period, ensuring that every factor had zero mean and unit variance over time. We then defined the mega-alpha on each day simply as the arithmetic average of these standardized scores. Although we reviewed Z. Kakushadze et al. 's information-weighted mega-alpha methodology which dynamically assigns weights proportional to each alpha’s rolling Information Coefficient, we deliberately chose the unweighted z-score average to keep our implementation lightweight and realistic for live trading scenarios.

Prior to training our PPO agent, we evaluated this mega-alpha with an Information Coefficient (IC) and hit-rate study against next-day returns. The results, however, were disappointing. The composite exhibited an IC hovering around zero and a hit rate worse than 50 percent, effectively underperforming random guessing. These findings indicate that, despite the elegance of a simple z-score ensemble, naïvely averaging decayed alpha signals fails to generate a predictive edge for reinforcement-learning–based portfolio allocation.

## Phase 4: Prototype with Ten Core Indicators

As phases 1-3 demonstrated, the 101 formulaic alphas may have lost their edge and predictive value, which could be due to the fact that the paper was published over a decade ago, thus the edge has diminished over time. Additionally, given the proprietary nature of hedge funds, it wouldn’t be a surprise if this paper consisted of indicators that are no longer in use, to serve as a peek inside the life of a quant, while preserving the valuable secrets that make the fund successful. As a final proof-of-concept to explore the power of RL in investing, we narrowed our universe down to ten well-established technical indicators—momentum, moving averages, volatility measures, and so on—rather than the full 101 alphas. Each day, we compute these ten signals for TSLA over the period 2010–2024 and standardize them to ensure comparability. During training (2010–2023), the PPO agent ingests the ten-dimensional z-scored vector and learns to output a weight for each indicator, effectively taking multiple simultaneous positions in the same stock according to its learned policy. By constraining the action space to just ten features, we dramatically reduce complexity and speed up convergence. 

Additionally, in this phase, we crafted a much better reward function:

Rt​=PnLt​−λvol​σt​−μdd​DDt​

This reward makes sure to include penalties for volatility/risk and drawdown, which helps to make our PPO policy prioritize wins while penalizing risk, creating a conservative yet consistently winning policy.

When we evaluated the trained policy on the 2024 out-of-sample period, the PPO agent delivered a shaky, yet ultimately winning return. While the strongest individual indicator (Strat_04) more than doubled its balance—ending 2024 at $227936 with a Sharpe of 1.31—it did so with a maximum drawdown of 66.63%! By contrast, our RL agent grew an initial $100 000 to $191625, achieving a very comparable annualized Sharpe of 1.27 and limiting its largest drawdown to just 34%. Three of the ten strategies under-performed, two with negative Sharpe ratios and minimal upside. Although many were positive, none were near our PPO policy or strat4. Our agent consistently blended these diverse signals into a balanced portfolio that captured upside while controlling risk, and minimizing drawdown. Although Strat4 outperformed our agent, it did so with nearly double the maximum drawdown, indicating that our agent can win consistently while being aware of its mistakes, and logically proactive, something that modern quantitative indicators and strategies cannot do. 

## SPY RL Agent Evaluation

To assess the generality of our reinforcement‐learning framework beyond a single stock, we applied the same PPO agent—trained on ten core technical indicators from 2010–2023—to the SPY ETF and evaluated it throughout 2024. The image below plots the daily equity curves for the RL agent (black line) alongside each of the ten benchmark strategies (colored lines), all starting from $100 000 on January 1, 2024.

At year‐end, the RL agent grew its account to approximately $120924, achieving an annualized Sharpe ratio of 1.643 and limiting its maximum drawdown to 5.64%. By comparison, the best single indicator (Strat_04) finished at $151018 (Sharpe 1.4, drawdown 15.5%), while several others underperformed or were insignificant.  Although Strat_04 produced higher absolute returns, it did so with nearly triple the drawdown, and a worse Sharpe ratio, highlighting the RL agent’s superior risk‐adjusted discipline. Most other individual strategies clustered near flat or negative PnL, reinforcing that no single indicator offers a consistently reliable signal on SPY in 2024.

Overall, the SPY evaluation demonstrates that our PPO agent can robustly combine diverse, low-dimensional inputs into a cohesive portfolio policy that outperforms many standalone strategies. While pure momentum (Strat_04) still led in raw performance, its severe volatility underscores the value of the RL agent’s ability to moderate exposure dynamically and navigate drawdowns more effectively. It’s worth noting that the SP500 was up 25% in 2024, so our agent didn’t beat the market, and although strategy 4 did, this could be due to the bullish market. Our agent's performance in 2024 on SPY highlights that it cannot beat the market alone, however it does serve as a reliable way to generate relatively risk-averse returns. 

## Conclusion

Our multi‐phase study reveals a clear trajectory in the application of reinforcement learning (RL) to quantitative portfolio construction. Phase 1 demonstrated that even a strong in‐sample performance—an apparent 60 % gain on TSLA—can evaporate once look‐ahead bias is eliminated, leaving only modest out‐of‐sample returns (3–20 % depending on training window) and emphasizing the necessity of strict data causality. Moving to Phase 2, we scaled to roughly 170 S&P 500 stocks and ingested ~70 percentile‐scaled formulaic alphas per ticker; here, the PPO agent delivered a –3 % return, a –0.85 Sharpe, and a 4.7 % drawdown in 2024, signaling that decadal‐old alphas have largely decayed when deployed broadly without further refinement.

In Phase 3, our simple mega‐alpha—an unweighted average of z-scored alphas—proved even less effective, with Information Coefficients around zero and hit rates below 50 %, underscoring that naïve ensembling of weak signals offers no predictive edge. By contrast, Phase 4 illustrated the power of selective feature engineering: training PPO on just ten well‐known technical indicators produced a TSLA policy that achieved a 1.64 Sharpe and contained drawdowns to 5.6 %, and when applied to SPY in 2024, yielded a 1.27 Sharpe with a reasonable 34 % peak‐to‐trough drop—outperforming almost all standalone strategies in both stability and risk‐adjusted return.

Together, these findings highlight that successful RL in modern markets hinges not on algorithmic novelty alone, but on meticulous data hygiene, dimensionality control, and robust cross‐validation. By carefully curating input features, enforcing strict causality, and benchmarking against simple baselines, RL agents can indeed extract meaningful patterns—and potentially deliver a genuine trading edge—in today’s competitive landscape. The motto of quantitative traders is to remove all emotion, however our agent has the ability to mathematically update its policies based on its mistakes, essentially blending the line between quantitative discipline, and behavioral awareness. This makes for a risk-aware and drawdown conscious policy which maximizes returns. 
