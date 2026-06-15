# India Portfolio Risk Analytics

A quantitative portfolio risk analytics framework designed to study dynamic risk behaviour, cross-sector contagion, and tail-risk propagation in Indian equity markets using Monte Carlo simulation and statistical risk modelling.

Built around a 20-stock NSE universe spanning IT, Financials, Energy, FMCG, and Healthcare sectors, the project attempts to move beyond conventional mean-variance portfolio analysis into a more structural examination of market instability, diversification decay, and higher-order risk dynamics.

---

# Objective

Traditional portfolio analysis often assumes stable correlations, Gaussian return distributions, and static diversification benefits.

This project explores how these assumptions weaken during stressed market environments by modelling:

* correlation clustering,
* volatility regime transitions,
* tail-risk asymmetries,
* contagion behaviour,
* and dynamic sector interactions.

The framework combines stochastic simulation, econometric diagnostics, and statistical risk decomposition into a unified portfolio analytics pipeline.

---

# Core Methodology

## Monte Carlo Simulation Engine

The portfolio is simulated using correlated geometric Brownian motion (GBM) processes:

* Cholesky decomposition is used to generate correlated asset paths
* Sector-sensitive correlation structures are imposed dynamically
* 5-year daily return paths are simulated across 20 NSE equities

The simulation framework models both:

* systematic co-movement,
* and idiosyncratic volatility behaviour.

---

## Portfolio Construction

* 20-stock equal-weight portfolio
* Multi-sector allocation:

  * IT
  * Financials
  * Energy
  * FMCG
  * Healthcare

The equal-weight structure allows the project to isolate differences between:

* capital allocation,
* and true underlying risk contribution.

---

# Risk Analytics Framework

## Standard Risk Metrics

The framework computes:

* Annualized return
* Annualized volatility
* Sharpe ratio
* Maximum drawdown
* Diversification ratio
* Rolling performance statistics

---

## Tail Risk Modelling

Beyond volatility-based risk measures, the project evaluates downside asymmetry using:

* Value-at-Risk (VaR)
* Conditional Value-at-Risk (CVaR)
* Pain Index
* Ulcer Index
* Omega Ratio

This enables analysis of:

* fat-tail exposure,
* downside clustering,
* and extreme-loss behaviour.

---

# Regime Detection

Volatility regimes are classified dynamically into:

* Calm
* Normal
* Stress

using rolling realized volatility thresholds.

This allows the framework to study:

* regime persistence,
* volatility clustering,
* and diversification breakdown during stressed environments.

---

# Correlation & Contagion Analysis

A key focus of the project is studying how correlation structures evolve through time.

The framework includes:

* Rolling cross-sector correlation analysis
* Correlation heatmaps
* Hierarchical clustering
* Dynamic contagion diagnostics

One important finding:
cross-sector diversification weakens materially during stress periods, particularly between Financials and IT sectors.

---

# Efficient Frontier Simulation

The project generates randomized portfolio clouds to evaluate:

* return-volatility tradeoffs,
* Sharpe-optimal allocations,
* and diversification efficiency.

Over 600 randomized portfolios are simulated to benchmark the equal-weight portfolio against alternative allocation structures.

---

# Higher Moment Diagnostics

The framework evaluates non-Gaussian behaviour using:

* Skewness
* Excess Kurtosis
* Fat-tail mapping

This helps identify assets whose risk characteristics are not adequately captured by variance alone.

---

# Novel Components

## Vol-of-Vol Modelling

The project computes volatility-of-volatility (VoV) to study:

* regime instability,
* uncertainty transitions,
* and potential hedging environments.

## Rolling Beta Dynamics

Rolling beta decomposition is used to evaluate:

* time-varying systematic exposure,
* and risk concentration drift.

## Sector Risk Contribution Decomposition

The framework compares:

* capital weights,
* versus true marginal portfolio risk contribution.

This reveals hidden concentration risks inside seemingly diversified portfolios.

---

# Technology Stack

* Python
* NumPy
* SciPy
* Matplotlib
* ReportLab

---

# Repository Structure

```plaintext
india-portfolio-risk-analytics/
│
├── india_portfolio_risk.py
├── india_portfolio_risk_v2.pdf
├── requirements.txt
├── README.md
└── images/
```

---

# Sample Visuals

## Portfolio Overview

![Portfolio Overview](images/page1.png)

## Regime & Correlation Analysis

![Regime Analysis](images/page2.png)

## Tail Risk & Clustering

![Tail Risk](images/page3.png)

---

# Key Insights

* Equal-weight portfolios do not imply equal risk contribution
* Diversification benefits compress during volatility spikes
* Tail-risk asymmetries become materially significant during stressed regimes
* Cross-sector contagion increases dynamically during instability
* Variance alone fails to capture structural downside exposure

---

# Future Extensions

Planned improvements include:

* Real NSE data integration
* GARCH-based volatility modelling
* Hidden Markov regime-switching models
* Black-Litterman optimization
* Options-implied volatility integration
* Interactive Streamlit dashboard deployment

---

# Author

Sahej Verma
Madras School of Economics
M.A. Financial Economics
