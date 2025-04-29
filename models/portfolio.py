# models/portfolio.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.optimize as sco

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

    def purchase_gmv_portfolio(self, portfolio_df, total_budget):
    
        tickers = [str(t) if not isinstance(t, str) else t for t in portfolio_df['Ticker']]
        weights = portfolio_df['GMV Weight'].values

        print(f"\nPurchasing GMV portfolio with a total budget of ${total_budget:,.2f}...")
        data = yf.download(tickers, period='1d', auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            # Handle MultiIndex properly
            if ('Price', 'Close') in data.columns:
                data = data.xs(('Price', 'Close'), axis=1)
            elif 'Close' in data.columns.get_level_values(1):
                data = data.xs('Close', axis=1, level=1)
            else:
                print("Cannot find Close prices, using first available level.")
                data = data.iloc[:, :len(tickers)]

        # Now flatten to Series with tickers as columns
        latest_prices = data.iloc[-1]  # Last row = latest prices

        # Fix columns if still messy
        if isinstance(latest_prices.index, pd.MultiIndex):
            latest_prices.index = latest_prices.index.get_level_values(-1)

        print("Latest prices fetched for:", latest_prices.index.tolist())
        purchases = []

        for ticker, weight in zip(tickers, weights):
            allocation = total_budget * weight
            price = latest_prices.get(ticker, np.nan)
            if np.isnan(price):
                continue
            shares = np.floor(allocation / price)
            spent = shares * price
            purchases.append({
                "Ticker": ticker,
                "Sector": "Unknown",
                "Asset Class": "Equity",
                "Quantity": shares,
                "Purchase Price": price
            })

        # Clear old portfolio
        self.assets = []
        for asset in purchases:
            if asset["Quantity"] > 0:
                self.add_asset(
                    ticker=asset["Ticker"],
                    sector=asset["Sector"],
                    asset_class=asset["Asset Class"],
                    quantity=asset["Quantity"],
                    purchase_price=asset["Purchase Price"]
                )

        total_spent = sum(asset["Quantity"] * asset["Purchase Price"] for asset in self.assets)
        print(f"\nPurchase complete. Total Spent: ${total_spent:,.2f}")
        print(f"Remaining Cash: ${total_budget - total_spent:,.2f}")
    
    
    def global_minimum_variance(self):
        print("Fetching S&P 500 tickers...")

        # Step 1: Get S&P500 Tickers
        def get_sp500_tickers():
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)[0]
            tickers = table['Symbol'].tolist()
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            return tickers

        tickers = get_sp500_tickers()

        print(f"Fetched {len(tickers)} tickers. Downloading price data...")

        # Step 2: Download 1 year adjusted close prices
        data = yf.download(tickers, period="1y", auto_adjust=True)
        print("printing the top of the data", data.head())
        data = data.iloc[:,:len(tickers)]
        print("printing the top of the data after slicing", data.head())

        

        print("Filtering tickers with missing data...")

        # Step 3: Keep only tickers with complete data
        valid_tickers = data.columns[data.notna().all()]
        data = data[valid_tickers]

        if data.shape[1] < 2:
            print("Not enough valid tickers to optimize.")
            return

        print(f"Number of valid tickers after filtering: {len(valid_tickers)}")

        # Step 4: Calculate returns
        returns = data.pct_change().dropna()

        # Step 5: Covariance matrix
        cov_matrix = returns.cov()

        n_assets = len(cov_matrix.columns)
        print(f"Optimization over {n_assets} assets.")

        # Step 6: Set up the optimization
        x0 = np.array([1/n_assets] * n_assets)
        bounds = tuple((0, 1) for _ in range(n_assets))
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        print("Running optimization...")

        result = sco.minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            print("Optimization failed.")
            return

        gm_weights = result.x
        optimized_tickers = cov_matrix.columns

        # Step 7: Display optimized portfolio
        tickers_clean = [ticker[1] if isinstance(ticker, tuple) else ticker for ticker in cov_matrix.columns]

        portfolio_df = pd.DataFrame({
            'Ticker': tickers_clean,
            'GMV Weight': gm_weights
        }).sort_values('GMV Weight', ascending=False)

        print("\n--- Global Minimum Variance Portfolio ---")
        print(portfolio_df.head(20))  # Top 20 biggest weights
        
        return portfolio_df
    

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
