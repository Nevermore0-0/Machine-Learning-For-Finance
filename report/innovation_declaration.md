# Innovation Declaration

## Strategy Identity

The final submitted strategy is `phase2_g5_05_expanding`: a daily, dollar-neutral, close-to-open long-short equity strategy using a volatility-scaled overnight-return target with expanding-window feature-weight estimation. It operates on the top-1000 US common-equity universe by market capitalisation, entering positions at the closing auction and exiting at the next morning's opening auction.

## Human Decisions That Shaped the Strategy

### 1. Volatility-scaled target (human economic reasoning)

The baseline strategy trained on raw next-overnight returns, which are dominated by high-volatility names. We replaced this with a risk-adjusted label: `overnight_next / (vol20 / sqrt(252))`, where `vol20` is trailing 20-day close-to-close volatility shifted by one trading day. The economic rationale is that a 50-basis-point overnight move in a low-volatility utility stock carries more signal than the same move in a high-volatility biotech name. This choice is motivated by the Sharpe-ratio interpretation of the target: features should predict *risk-adjusted* overnight alpha, not raw dollar returns.

### 2. Expanding training window (stability argument)

The Phase 1 champion used a fixed 4-year rolling window, discarding older data. We chose an expanding window based on the observation that the overnight return premium, while time-varying, is persistent. Older data still contains useful cross-sectional information about which features predict overnight moves. The expanding window uses all available history before each scored year, which stabilises the transparent correlation-based weight learner without adding model complexity. This was validated by comparing validation-period (2019-2022) and internal holdout (2023-2024) Sharpe ratios across both window types.

### 3. Feature prior design (human financial intuition)

The 27-feature prior weight vector was designed from economic intuition, not learned from data. Short-horizon reversal features (1-day, 5-day overnight returns) receive the largest negative priors, reflecting the well-documented mean-reversion effect in overnight returns. Fundamental quality features (Piotroski score, gross profit margin) receive positive priors. Short-interest stress features receive negative priors (high short interest predicts negative overnight performance). These priors serve as a regularisation anchor: 50% of the final weight comes from the prior and 50% from data-learned correlations. This prevents the model from making extreme bets on noisy correlation estimates.

### 4. LightGBM comparison (ML experiment)

We explicitly tested a LightGBM gradient-boosted tree model using the same 27 rank features and walk-forward protocol. Conservative hyperparameters were chosen (31 leaves, depth 6, 200 trees, 500 minimum child samples) to avoid overfitting. This comparison is a legitimate ML contribution for the course: it documents whether non-linear feature interactions improve overnight-return prediction beyond the transparent linear model. The finding itself -- whether trees help or not -- is informative regardless of direction, as it characterises the alpha signal's structure.

### 5. Feature ablation interpretation (honest disclosure)

We ran leave-one-group-out ablation and discovered that removing the short-interest/borrow-stress and earnings/revision groups actually *improves* net Sharpe. Rather than hiding this, we discuss it transparently: these features may serve as risk-control or cost-awareness inputs rather than pure alpha contributors, or their contribution may be time-varying and weak in the full sample.

## What AI Tools Did vs. What Humans Decided

AI tools (Claude, Codex) were used for code generation, debugging, and boilerplate automation. All economic design decisions -- the volatility-scaled target, expanding window choice, prior weight vector, feature selection, borrow tier thresholds, and the decision to compare LightGBM against the linear model -- were made by the team based on financial reasoning and coursework requirements. The AI did not choose the strategy; it implemented the strategy the team designed.

## What Did Not Change

The promotion did not introduce external data, intraday data, 2025+ data, new feature families, lower transaction costs, lower borrow costs, a relaxed participation cap, or different execution timing. Commission (0.5 bps/leg), auction slippage (1.5 bps/leg), Tier A/B/C borrow rates, the 5% ADV20 cap, and close-to-open execution all remain fixed at the brief's specification.
