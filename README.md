# Heston Model and Implied Volatility Calculation

## Overview
This repository contains Python scripts and a Jupyter notebook for simulating stock prices using the Heston model and 
calculating implied volatilities from the simulated data. The Heston model is a stochastic volatility model that can describe the evolution of stock prices with a stochastic volatility component differnet to Black Scholes which assumes constant volatility. 

### Model Dynamics
The Heston model specifies the following dynamics for the asset price $\(S(t)\)$ and variance $\(v(t)\)$:

- **Price Dynamics**:
  $$dS(t) = \mu S(t) \, dt + \sqrt{v(t)} S(t) \, dW_s(t)$$
  
  Here, $\mu is the risk-free rate, $S(t)$ is the spot price of the asset, $v(t)$ is the variance, and $dW_s(t)$ is the increment of a Wiener process (Brownian motion) driving the asset price.

- **Variance Dynamics**:
  $$dv(t) = \kappa (\theta - v(t)) \, dt + \sigma \sqrt{v(t)} \, dW_v(t)$$

  $\kappa$ represents the rate of reversion, $\theta$ is the long-term variance mean, $\sigma$ is the volatility of volatility, and $dW_v(t)$ is the increment of a second, correlated Wiener process.

### Correlation Structure
The model allows for a correlation between the two Wiener processes, $dW_s(t)$ and $dW_v(t)$, with:
  $$dW_s(t) \cdot dW_v(t) = \rho \, dt$$
where $\rho$ is the correlation coefficient, which typically is negative, implying an inverse relationship between asset returns and changes in volatility (known as the leverage effect).

## Structure
The project is structured into three main components:
- `Heston_Model.py`: Implements the Heston model for simulating stock price paths and volatility.
- `Implied_Vol.ipynb`: Jupyter notebook for calculating the implied volatilities from the simulated stock prices using the Heston model.
- `utils.py`: Contains utility functions to fetch and prepare data required for the simulations.

## Key Features
### Heston Model Simulation
- **Script**: `Heston_Model.py`
- **Functionality**: Simulates multiple paths of stock prices and volatility using the Heston model. Users can specify the stock ticker, simulation period, number of simulations, and the discretization method (Euler or Milstein).
- **Output**: The script outputs the simulated price paths, which are then visualized or further analyzed for financial metrics.

### Implied Volatility Calculation
- **Functionality**: Uses the results from the Heston model to calculate European put and call option prices at various strike prices. Then, it computes the implied volatilities for these options and plots the volatility smile.
- **Visualization**: The notebook includes plots of the implied volatility curves (volatility smile) and histograms showing the probability density function of the stock prices at expiration.

### Utility Functions
- **Functionality**: Provides essential data fetching and preprocessing functions used by the Heston model simulation. This includes fetching stock and volatility data from Yahoo Finance, processing the data to compute initial values and correlations needed for the model.
  
## Dependencies
- Python 3.x
- NumPy
- Matplotlib
- pandas
- yfinance
- py_vollib_vectorized
