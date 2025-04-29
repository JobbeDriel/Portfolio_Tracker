# --- portfolio_teracker.py --- 

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt 

def fetch_ytd_performance(ticker):
    try:
        stock = yf.Ticker(ticker)

        # Get the current year
        current_year = datetime.now().year
        start_date = f"{current_year}-01-01"

        # Fetch historical prices from Jan 1st to today
        hist = stock.history(start=start_date)

        if hist.empty:
            print(f"No historical data found for {ticker}.")
            return None

        price_start = hist['Close'].iloc[0]  # First close price of the year
        price_now = hist['Close'].iloc[-1]   # Most recent close price

        ytd_return = (price_now - price_start) / price_start

        return ytd_return

    except Exception as e:
        print(f"Error fetching YTD performance for {ticker}: {e}")
        return None

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

    def view_portfolio(self):
        if not self.assets:
            print("Portfolio is empty.")
        else:
            print("\nCurrent Portfolio:")
            for idx, asset in enumerate(sorted(self.assets, key=lambda x: x['Ticker']), 1):
                ticker = asset['Ticker']
                quantity = asset['Quantity']
                purchase_price = asset['Purchase Price']
                sector = asset['Sector']
                asset_class = asset['Asset Class']

                transaction_value = quantity * purchase_price

                # Fetch current market price live
                try:
                    stock = yf.Ticker(ticker)
                    current_price = stock.info.get('currentPrice', None)

                    if current_price is not None:
                        current_value = quantity * current_price
                    else:
                        current_value = None
                except Exception as e:
                    print(f"Error fetching current price for {ticker}: {e}")
                    current_value = None

                # Print nicely
                print(f"{idx}. {ticker} - {quantity} shares @ ${purchase_price:.2f} ({sector}, {asset_class})")
                print(f"    Transaction Value: ${transaction_value:.2f}")

                if current_value is not None:
                    print(f"    Current Market Value: ${current_value:.2f}")
                else:
                    print("    Current Market Value: Not available")

                print("-" * 50)
    
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

        # Pie Chart: Asset Weights
        plt.figure(figsize=(6, 6))
        plt.pie(df['Current Value'], labels=df['Ticker'], autopct='%1.1f%%', startangle=140)
        plt.title('Portfolio by Asset')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        # Pie Chart: Asset Class Weights
        plt.figure(figsize=(6, 6))
        plt.pie(by_class, labels=by_class.index, autopct='%1.1f%%', startangle=140)
        plt.title('Portfolio by Asset Class')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        # Pie Chart: Sector Weights
        plt.figure(figsize=(6, 6))
        plt.pie(by_sector, labels=by_sector.index, autopct='%1.1f%%', startangle=140)
        plt.title('Portfolio by Sector')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


def fetch_asset_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        sector = info.get('sector', 'Unknown')
        asset_class = 'Equity'  # For stocks, default to Equity
        latest_price = info.get('currentPrice', None)

        return sector, asset_class, latest_price

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, None, None
    
def simulate_portfolio(portfolio, n_years=15, n_simulations=100000):
    if not portfolio.assets:
        print("Portfolio is empty.")
        return

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
                print(f"No data for {ticker}. Skipping.")
                continue

            daily_returns = hist['Close'].pct_change().dropna()
            drift = daily_returns.mean()
            vol = daily_returns.std()
            start_price = hist['Close'].iloc[-1]

            returns.append((drift, vol))
            start_prices.append(start_price)

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue

    if not start_prices:
        print("No valid stocks to simulate.")
        return

    start_prices = np.array(start_prices)
    drifts = np.array([x[0] for x in returns])
    vols = np.array([x[1] for x in returns])

    n_days = n_years * 252
    dt = 1/252

    # Simulate final returns directly
    np.random.seed(42)

    final_prices = []

    for i in range(len(start_prices)):
        # Simulate log returns over all days
        total_drift = (drifts[i] - 0.5 * vols[i]**2) * n_days * dt
        total_vol = vols[i] * np.sqrt(n_days * dt)

        # Random draws
        Z = np.random.standard_normal(n_simulations)

        # Final price from GBM formula
        S_T = start_prices[i] * np.exp(total_drift + total_vol * Z)
        final_prices.append(S_T)

    final_prices = np.array(final_prices)  # Shape (n_assets, n_simulations)

    # Portfolio value at end
    portfolio_end_values = (final_prices.T * quantities.values).sum(axis=1)

    # Analyze results
    mean_ending = np.mean(portfolio_end_values)
    median_ending = np.median(portfolio_end_values)
    p5 = np.percentile(portfolio_end_values, 5)
    p95 = np.percentile(portfolio_end_values, 95)

    print(f"\n--- Portfolio Simulation ---")
    print(f"Simulated {n_simulations:,} paths over {n_years} years.")

    print(f"Expected Ending Value (Mean): ${mean_ending:,.2f}")
    print(f"Median Ending Value: ${median_ending:,.2f}")
    print(f"5th Percentile: ${p5:,.2f}")
    print(f"95th Percentile: ${p95:,.2f}")

    initial_value = (start_prices * quantities.values).sum()

    plt.figure(figsize=(10,6))
    plt.hist(portfolio_end_values, bins=100, density=True, alpha=0.7, color='skyblue')
    plt.axvline(initial_value, color='red', linestyle='--', linewidth=2, label='Initial Portfolio Value')
    plt.title('Distribution of Portfolio Ending Values')
    plt.xlabel('Portfolio Value ($)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    portfolio = Portfolio()

    while True:
        print("\nPortfolio Tracker CLI (Auto-fetch mode)")
        print("1. View Asset Info")
        print("2. Add Asset")
        print("3. View Portfolio")
        print("4. Portfolio Summary & Weights")
        print("5. Simulate Portfolio")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ").strip()

        if choice == '1':
            ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
    
            # Fetch YTD
            ytd = fetch_ytd_performance(ticker)
    
            if ytd is not None:
                print(f"YTD performance for {ticker}: {ytd:.2%}")
            else:
                print("YTD performance data not available.")

            # Fetch 1 year of historical prices
            stock = yf.Ticker(ticker)
            one_year_ago = datetime.now() - pd.DateOffset(years=1)
            hist = stock.history(start=one_year_ago)['Close']

            comparison_data = {ticker: hist}

            if hist.empty:
                print("No historical data available to plot.")
                continue 

            # Plot initially
            combined = pd.DataFrame(comparison_data).dropna()
            combined = combined / combined.iloc[0] * 100
            plt.figure(figsize=(10,5))
            for col in combined.columns:
                plt.plot(combined.index, combined[col], label=col)
            plt.title("1-Year Price Comparison (Indexed to 100)")
            plt.xlabel('Date')
            plt.ylabel('Indexed Price')
            plt.legend()
            plt.grid(True)
            plt.show()

            while True: 
                print("\nDo you want to compare with another stock?")
                print("1. Yes")
                print("2. No")
                compare = input("Enter 1 or 2: ").strip()

                if compare == '1':
                    new_ticker = input("Enter other ticker symbol: ").strip().upper()
                    stock2 = yf.Ticker(new_ticker)
                    hist2 = stock2.history(start=one_year_ago)['Close']

                    if hist2.empty:
                        print(f"No data for {new_ticker}. Skipping.")
                        continue

                    ytd2 = fetch_ytd_performance(new_ticker)
                    if ytd2 is not None:
                        print(f"YTD performance for {new_ticker}: {ytd2:.2%}")
                    else:
                        print("YTD performance not available.")

                    # Add to comparison
                    comparison_data[new_ticker] = hist2

                    # Align and normalize all added tickers
                    combined = pd.DataFrame(comparison_data).dropna()
                    combined = combined / combined.iloc[0] * 100

                    # Re-plot after each addition
                    plt.figure(figsize=(10, 5))
                    for col in combined.columns:
                        plt.plot(combined.index, combined[col], label=col)
                    plt.title("1-Year Price Comparison (Indexed to 100)")
                    plt.xlabel("Date")
                    plt.ylabel("Indexed Price")
                    plt.legend()
                    plt.grid(True)
                    plt.show()

                elif compare == '2':
                    break
                else:
                    print("Please enter 1 or 2.")
            continue 
            


        if choice == '2':
            ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
            sector, asset_class, latest_price = fetch_asset_info(ticker)
            ytd = fetch_ytd_performance(ticker)
           
            if sector is None or latest_price is None:
                print(f"Could not fetch data for {ticker}. Try again.")
                continue 
        
            if ytd is not None:
                print(f"Fetched data: Sector: {sector}, Asset Class: {asset_class}, Latest Price: ${latest_price:.2f}, YTD Performance: {ytd:.2%}")
            else:
                print(f"Fetched data: Sector: {sector}, Asset Class: {asset_class}, Latest Price: ${latest_price:.2f}")
                print("YTD performance data not available.")

            print("\nDo you want to buy this stock?")
            print("1. Yes")
            print("2. No")
            
            buy_choice = input("Enter 1 or 2: ").strip()

            if buy_choice == '1':
                quantity = float(input("Enter quantity to buy: ").strip())
                portfolio.add_asset(ticker, sector, asset_class, quantity, latest_price)
                print(f"{ticker} added to your portfolio.")
            elif buy_choice == '2':
                print("Returning to main menu without buying.")
                continue
            else:
                print("Invalid input. Returning to main menu.")
                continue

            
        elif choice == '3':
            portfolio.view_portfolio()

        elif choice == '4':
            portfolio.view_summary()
        
        elif choice == '5':
            simulate_portfolio(portfolio)

        elif choice == '6':
            print("Exiting Portfolio Tracker.")
            break

        else:
            print("Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()




        