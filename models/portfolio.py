# models/portfolio.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

class Portfolio:
    def __init__(self):
        self.assets = []

    def add_asset(self, ticker, sector, asset_class, quantity, purchase_price):
        asset = {
            "Ticker": ticker.upper(),
            "Sector": sector,
            "Asset Class": asset_class,
            "Quantity": quantity,
            "Purchase Price": purchase_price
        }
        self.assets.append(asset)

    def get_portfolio_dataframe(self):
        if not self.assets:
            return None, 0  # Always return a tuple

        df = pd.DataFrame(self.assets)

        if df.empty:
            return None, 0

        current_values = []
        for _, row in df.iterrows():
            try:
                stock = yf.Ticker(row['Ticker'])
                current_price = stock.info.get('currentPrice', None)
                if current_price is None:
                    raise ValueError("Price not available")
                value = current_price * row['Quantity']
            except Exception as e:
                print(f"Error fetching price for {row['Ticker']}: {e}")
                value = 0
            current_values.append(value)

        df['Current Value'] = current_values
        total_value = df['Current Value'].sum()

        if total_value == 0:
            return None, 0

        df['Weight (%)'] = df['Current Value'] / total_value * 100

        return df, total_value

    
    def view_summary(self):
        if not self.assets:
            print("Portfolio is empty.")
            return

        df = pd.DataFrame(self.assets)
    
        # Fetch current prices and compute current values
        current_values = []
        for _, row in df.iterrows():
            try:
                stock = yf.Ticker(row['Ticker'])
                current_price = stock.info.get('currentPrice', None)
                if current_price is None:
                    raise ValueError("Price not available")
                value = current_price * row['Quantity']
            except Exception as e:
                print(f"Error fetching price for {row['Ticker']}: {e}")
                value = 0
            current_values.append(value)

        df['Current Value'] = current_values

        total_value = df['Current Value'].sum()
        if total_value == 0:
            print("Could not calculate total portfolio value.")
            return

        # Weights per asset
        df['Weight (%)'] = df['Current Value'] / total_value * 100

        # Group by Asset Class and Sector
        by_class = df.groupby('Asset Class')['Current Value'].sum()
        by_class_weights = by_class / total_value * 100

        by_sector = df.groupby('Sector')['Current Value'].sum()
        by_sector_weights = by_sector / total_value * 100

        # Output
        print("\n--- Portfolio Summary ---")
        print(f"Total Portfolio Value: ${total_value:,.2f}\n")

        print("--- Asset Weights ---")
        for _, row in df.iterrows():
            print(f"{row['Ticker']}: ${row['Current Value']:,.2f} ({row['Weight (%)']:.2f}%)")

        print("\n--- By Asset Class ---")
        for cls, val in by_class.items():
            print(f"{cls}: ${val:,.2f} ({by_class_weights[cls]:.2f}%)")

        print("\n--- By Sector ---")
        for sec, val in by_sector.items():
            print(f"{sec}: ${val:,.2f} ({by_sector_weights[sec]:.2f}%)")


def fetch_asset_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        asset_class = 'Equity'  # Default
        latest_price = info.get('currentPrice', None)
        return sector, asset_class, latest_price
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, None, None

def fetch_ytd_performance(ticker):
    try:
        stock = yf.Ticker(ticker)
        current_year = datetime.now().year
        start_date = f"{current_year}-01-01"
        hist = stock.history(start=start_date)
        if hist.empty:
            print(f"No historical data for {ticker}.")
            return None
        price_start = hist['Close'].iloc[0]
        price_now = hist['Close'].iloc[-1]
        return (price_now - price_start) / price_start
    except Exception as e:
        print(f"Error fetching YTD performance for {ticker}: {e}")
        return None
    
def simulate_portfolio(portfolio, n_years=15, n_simulations=100000):
    if not portfolio.assets:
        return None, None

    df = pd.DataFrame(portfolio.assets)
    tickers = df['Ticker']
    quantities = df['Quantity']

    returns = []
    start_prices = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')

            if hist.empty:
                continue

            daily_returns = hist['Close'].pct_change().dropna()
            drift = daily_returns.mean()
            vol = daily_returns.std()
            start_price = hist['Close'].iloc[-1]

            returns.append((drift, vol))
            start_prices.append(start_price)

        except Exception:
            continue

    if not start_prices:
        return None, None

    start_prices = np.array(start_prices)
    drifts = np.array([x[0] for x in returns])
    vols = np.array([x[1] for x in returns])

    n_days = n_years * 252
    dt = 1/252

    np.random.seed(42)

    final_prices = []
    for i in range(len(start_prices)):
        total_drift = (drifts[i] - 0.5 * vols[i]**2) * n_days * dt
        total_vol = vols[i] * np.sqrt(n_days * dt)
        Z = np.random.standard_normal(n_simulations)
        S_T = start_prices[i] * np.exp(total_drift + total_vol * Z)
        final_prices.append(S_T)

    final_prices = np.array(final_prices)

    portfolio_end_values = (final_prices.T * quantities.values).sum(axis=1)
    initial_value = (start_prices * quantities.values).sum()

    return portfolio_end_values, initial_value
