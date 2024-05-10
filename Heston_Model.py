from utils import get_data
import numpy as np
import matplotlib.pyplot as plt


def heston_method(ticker, period="1y", simulations=1000, discretization_method="euler"):
    """
    Simulate the Heston Model for SPY & VIX using either the Euler or Milstein Stochastic DE Approximation methods
    """
    # Get the data from the custom utils class
    utils_obj = get_data(ticker, period, simulations)
    data = utils_obj()
    print(data)

    # Initialize the prices and vol arrays with 253 days (252 trading days + 1 initial day)
    prices = np.full(shape=(253, simulations), fill_value=data["S0"])
    vol = np.full(shape=(253, simulations), fill_value=data["v0"])

    # Get normal samples for the correlated brownian motions
    Z = np.random.multivariate_normal([0, 0], data["correlation"], (252, simulations))
    dt = 1 / 252

    for i in range(1, 252 + 1):
        prices[i] = prices[i - 1] * np.exp(
            (data["rf_rate"] - 0.5 * vol[i - 1]) * dt
            + np.sqrt(vol[i - 1] * dt) * Z[i - 1, :, 0]
        )

        if discretization_method == "euler":
            vol_calc = data["kappa"] * (data["theta"] - vol[i - 1]) * dt
            vol_calc_p2 = data["sigma"] * np.sqrt(vol[i - 1] * dt) * Z[i - 1, :, 1]
            vol[i] = np.maximum(0, vol[i - 1] + vol_calc + vol_calc_p2)

        elif discretization_method == "milstein":
            vol[i] = np.maximum(0, 
                vol[i - 1]
                + data["kappa"] * (data["theta"] - vol[i - 1]) * dt
                + data["sigma"] * np.sqrt(vol[i - 1] * dt) * Z[i - 1, :, 1]
                + 0.25 * data["sigma"] ** 2 * dt * (Z[i - 1, :, 1] ** 2 - 1)
            )
    return prices, vol

'''

def euler_method(ticker, period="1y", simulations=1000):
    """
    Simulate Heston Model for SPY + VIX Using the Euler Approximation method
    """
    utils_obj = utils(ticker, period, simulations)
    data = utils_obj()
    print(data)
    prices = np.full(shape=(252 + 1, simulations), fill_value=data["S0"])
    vol = np.full(shape=(252 + 1, simulations), fill_value=data["v0"])
    Z = np.random.multivariate_normal([0, 0], data["correlation"], (252, simulations))
    dt = 1 / 252

    for i in range(1, 252 + 1):
        prices[i] = prices[i - 1] * np.exp(
            (data["rf_rate"] - 0.5 * vol[i - 1]) * dt
            + np.sqrt(vol[i - 1] * dt) * Z[i - 1, :, 0]
        )

        # Use reflection method to ensure variance is positive
        vol_calc = data["kappa"] * (data["theta"] - vol[i - 1]) * dt
        vol_calc_p2 = data["sigma"] * np.sqrt(vol[i - 1] * dt) * Z[i - 1, :, 1]
        vol[i] = np.abs(vol[i - 1] + vol_calc + vol_calc_p2)
    return prices, vol


def milstein_method(ticker, period="1y", simulations=1000):
    """
    Simulate Heston Model for SPY + VIX Using the Milstein Stochastic DE Approximation method
    """
    utils_obj = utils(ticker, period, simulations)
    data = utils_obj()
    print(data)
    prices = np.full(shape=(252 + 1, simulations), fill_value=data["S0"])
    vol = np.full(shape=(252 + 1, simulations), fill_value=data["v0"])
    dt = 1 / 252  # daily time step

    # get normal samples for the correlated brownian motions
    Z = np.random.multivariate_normal([0, 0], data["correlation"], (252, simulations))
    dt = 1 / 252

    for i in range(1, 253):
        prices[i] = prices[i - 1] * np.exp(
            (data["rf_rate"] - 0.5 * vol[i - 1]) * dt
            + np.sqrt(vol[i - 1] * dt) * Z[i - 1, :, 0]
        )
        vol[i] = np.abs(
            vol[i - 1]
            + data["kappa"] * (data["theta"] - vol[i - 1]) * dt
            + data["sigma"] * np.sqrt(vol[i - 1] * dt) * Z[i - 1, :, 1]
            + 0.25 * data["sigma"] ** 2 * dt * (Z[i - 1, :, 1] ** 2 - 1)
        )
    return prices, vol
'''

def plots(prices, vol):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    time = np.linspace(0, len(prices)/253, 253)
    ax1.plot(time, prices)
    ax1.set_title("Heston Model Projected Asset Prices)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Asset Prices")

    ax2.plot(time, vol)
    ax2.set_title("Heston Model Project Variance (SPY)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Variance")

    plt.show()

if __name__ == "__main__":
    np.random.seed(20901908)
    plots(*heston_method("SPY", period="3y", simulations=10000, discretization_method="euler"))
    plots(*heston_method("SPY", period="3y", simulations=10000, discretization_method="milstein"))

