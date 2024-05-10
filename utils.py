import yfinance as yf
import numpy as np
import pandas as pd


class get_data:
    def __init__(self, ticker, period, simulations, kappa=3):
        self.ticker = ticker
        self.period = period
        self.simulations = simulations
        self.kappa = kappa
        self.refresh_data()
        self.get_corr()

    def __call__(self):
        self.full_results_data = {
            "model": "Heston",
            "period": self.period,
            "simulations": self.simulations,
            "ticker": self.ticker,
            "correlation": self.corr,
            "S0": self.data["Close"].iloc[0],
            "v0": self.vol["Close"].iloc[0]/100,
            "sigma": self.vol_of_vol.mean()/100,
            "rf_rate": self.rf_rate,
            "kappa": self.kappa,
            "theta": self.theta,
        }
        return self.full_results_data

    def refresh_data(self):
        self.vol_of_vol = yf.download("^VVIX", period=self.period)["Close"]
        self.vol = yf.download("^VIX", period=self.period)
        self.theta = self.vol["Close"].mean()/100
        self.data = yf.download(self.ticker, period=self.period)
        self.rf_rate = yf.download("^TNX", period=self.period)["Close"].iloc[-1]/100
        print("Data Refreshed From Yfinance...")

    def get_corr(self):
        stock_returns = self.data["Close"].pct_change().dropna()
        vol_returns = self.vol["Close"].pct_change().dropna()
        combined_data = pd.DataFrame({"stock_returns": stock_returns, "vol_returns": vol_returns}).dropna()
        self.corr = np.corrcoef(combined_data["vol_returns"], combined_data["stock_returns"])
