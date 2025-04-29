# models/portfolio.py

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import scipy.optimize as sco
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
            
            sector, asset_class, _ = fetch_asset_info(ticker)

            purchases.append({
                "Ticker": ticker,
                "Sector": sector if sector else "Unknown",
                "Asset Class": asset_class if asset_class else "Equity",
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

    def equal_weighted_portfolio(self):
        

        print("Fetching S&P 500 tickers...")

        def get_sp500_tickers():
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)[0]
            tickers = table['Symbol'].tolist()
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            return tickers

        tickers = get_sp500_tickers()

        n = len(tickers)
        weight = 1 / n

        portfolio_df = pd.DataFrame({
            'Ticker': tickers,
            'Equal Weight': [weight] * n
        })

        print("\n--- Equal Weighted Portfolio (1/N) ---")
        print(portfolio_df.head(10))  # Show top 10

        return portfolio_df
    
    def purchase_equal_weighted_portfolio(self, portfolio_df, total_budget):
        import yfinance as yf
        import numpy as np
        import pandas as pd

        # Extract tickers and weights
        tickers = portfolio_df['Ticker'].tolist()
        weights = portfolio_df['Equal Weight'].values

        print(f"\nPurchasing Equal Weighted portfolio with a total budget of ${total_budget:,.2f}...")

        print("Fetching latest prices...")
        data = yf.download(tickers, period='1d', auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            if ('Price', 'Close') in data.columns:
                data = data.xs(('Price', 'Close'), axis=1)
            elif 'Close' in data.columns.get_level_values(1):
                data = data.xs('Close', axis=1, level=1)
            else:
                print("Cannot find Close prices, using first available columns.")
                data = data.iloc[:, :len(tickers)]

        latest_prices = data.iloc[-1]

        # Flatten index if necessary
        if isinstance(latest_prices.index, pd.MultiIndex):
            latest_prices.index = latest_prices.index.get_level_values(-1)

        purchases = []

        for ticker, weight in zip(tickers, weights):
            allocation = total_budget * weight
            if ticker not in latest_prices:
                print(f"Ticker {ticker} price not found, skipping...")
                continue

            price = latest_prices[ticker]
            shares = np.floor(allocation / price)
            spent = shares * price

            purchases.append({
                "Ticker": ticker,
                "Sector": "Unknown",
                "Asset Class": "Equity",
                "Quantity": shares,
                "Purchase Price": price
            })

        # Clear existing portfolio and add new purchases
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
        leftover_cash = total_budget - total_spent

        print(f"\nPurchase complete. Total Spent: ${total_spent:,.2f}")
        print(f"Remaining Cash: ${leftover_cash:,.2f}")

        return total_spent, leftover_cash
    
    def ml_classification_strategy(self):
        print("Fetching S&P 500 tickers...")

        def get_sp500_tickers():
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table = pd.read_html(url)[0]
            tickers = table['Symbol'].tolist()
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            return tickers

        tickers = get_sp500_tickers()

        print("Downloading historical prices...")
        data = yf.download(tickers, period="1y", auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            if ('Price', 'Close') in data.columns:
                data = data.xs(('Price', 'Close'), axis=1)
            elif 'Close' in data.columns.get_level_values(1):
                data = data.xs('Close', axis=1, level=1)
            else:
                data = data.iloc[:, :len(tickers)]

        # FLATTEN column names to simple tickers
        data.columns = [c if isinstance(c, str) else c[1] for c in data.columns]

        data = data.dropna(axis=1)  # Remove stocks with missing data

        print("Building features and labels...")

        # Feature: previous day's returns
        past_returns = data.pct_change().shift(1)

        # Label: 5-day forward return
        forward_returns = data.pct_change(periods=5).shift(-5)
        labels = (forward_returns > 0).astype(int)

        # Stack into long format
        X = past_returns.stack().reset_index()
        X.columns = ['Date', 'Ticker', 'Return']

        y = labels.stack().reset_index()
        y.columns = ['Date', 'Ticker', 'Label']

        df = pd.merge(X, y, on=['Date', 'Ticker']).dropna()

        # Pivot: each row = stock on date, features = returns of all stocks
        print("Building cross-sectional features...")
        pivoted_features = past_returns.shift(1).dropna()
        latest_features = pivoted_features.iloc[-1]  # Used later for prediction
        pivoted_features = pivoted_features.iloc[:-1]  # Exclude last row for train

        # Build ticker ID mapping
        ticker_ids = {ticker: i for i, ticker in enumerate(pivoted_features.columns)}

        # Build training set: each row = market context + ticker ID
        merged = df[df['Date'].isin(pivoted_features.index)]

        X_list = []
        y_list = []
        for date, group in merged.groupby('Date'):
            try:
                X_row = pivoted_features.loc[date]
                for _, row in group.iterrows():
                    ticker_id = ticker_ids.get(row['Ticker'], -1)
                    feature_vector = np.append(X_row.values, ticker_id)
                    X_list.append(feature_vector)
                    y_list.append(row['Label'])
            except Exception as e:
                print(f"Skipping date {date} due to error: {e}")
                continue

        X_train = np.array(X_list)
        y_train = np.array(y_list)

        print(f"Training on {len(y_train)} samples with {X_train.shape[1]} features...")

        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)

        print("Predicting today's winners...")

        ticker_probs = []

        for ticker in latest_features.index:
            ticker_id = ticker_ids.get(ticker, -1)
            X_input = np.append(latest_features.values, ticker_id).reshape(1, -1)

            try:
                prob_up = model.predict_proba(X_input)[0][1]  # probability of class 1 ("up")
                ticker_probs.append((ticker, prob_up))
            except Exception as e:
                print(f"Error predicting for {ticker}: {e}")
                continue

        # Sort by probability
        ticker_probs = sorted(ticker_probs, key=lambda x: x[1], reverse=True)

        # Take top 10% (e.g., 50 stocks if 500 total)
        top_n = int(len(ticker_probs) * 0.10)

        rising_stocks = [ticker for ticker, prob in ticker_probs[:top_n]]

        print(f"Selected {len(rising_stocks)} most confident stocks predicted to go UP.")

        if not rising_stocks:
            print("No stocks predicted to go up. Exiting strategy.")
            return None

        weight = 1 / len(rising_stocks)
        portfolio_df = pd.DataFrame({
            'Ticker': rising_stocks,
            'ML Weight': [weight] * len(rising_stocks)
        })

        print("\n--- ML Classification Portfolio ---")
        print(portfolio_df.head())

        return portfolio_df
    
    def purchase_ml_classification_portfolio(self, portfolio_df, total_budget):
        tickers = portfolio_df['Ticker'].tolist()
        weights = portfolio_df['ML Weight'].values

        print(f"\nPurchasing ML Classification portfolio with a total budget of ${total_budget:,.2f}...")

        print("Fetching latest prices...")
        data = yf.download(tickers, period='1d', auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            if ('Price', 'Close') in data.columns:
                data = data.xs(('Price', 'Close'), axis=1)
            elif 'Close' in data.columns.get_level_values(1):
                data = data.xs('Close', axis=1, level=1)
            else:
                print("Cannot find Close prices, using first available columns.")
                data = data.iloc[:, :len(tickers)]

        latest_prices = data.iloc[-1]

        if isinstance(latest_prices.index, pd.MultiIndex):
            latest_prices.index = latest_prices.index.get_level_values(-1)

        purchases = []

        for ticker, weight in zip(tickers, weights):
            allocation = total_budget * weight
            if ticker not in latest_prices:
                print(f"Ticker {ticker} price not found, skipping...")
                continue

            price = latest_prices[ticker]
            shares = np.floor(allocation / price)
            spent = shares * price

            purchases.append({
                "Ticker": ticker,
                "Sector": "Unknown",
                "Asset Class": "Equity",
                "Quantity": shares,
                "Purchase Price": price
            })

        # Clear existing assets
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
        leftover_cash = total_budget - total_spent

        print(f"\nPurchase complete. Total Spent: ${total_spent:,.2f}")
        print(f"Remaining Cash: ${leftover_cash:,.2f}")

        return total_spent, leftover_cash
    

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
