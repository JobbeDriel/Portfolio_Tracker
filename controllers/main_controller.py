# controllers/main_controller.py

from models.portfolio import Portfolio, fetch_asset_info, fetch_ytd_performance, simulate_portfolio
from views.portfolio_view import display_asset_info, plot_price_comparison, display_portfolio_summary, plot_portfolio_summary,  show_simulation_results
import pandas as pd
import yfinance as yf
from datetime import datetime

def main():
    portfolio = Portfolio()

    while True:
        print("\nPortfolio Tracker CLI (MVC Version)")
        print("1. View Asset Info")
        print("2. Add Asset")
        print("3. View Portfolio")
        print("4. Portfolio Summary & Weights")
        print("5. Simulate Portfolio")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ").strip()

        if choice == '1':
            comparison_data = {}

            while True:
                ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
                sector, asset_class, latest_price = fetch_asset_info(ticker)
                ytd_return = fetch_ytd_performance(ticker)

                display_asset_info(ticker, sector, asset_class, latest_price, ytd_return)

                # Fetch 1-year price data
                one_year_ago = datetime.now() - pd.DateOffset(years=1)
                stock = yf.Ticker(ticker)
                hist = stock.history(start=one_year_ago)['Close']

                if hist.empty:
                    print("No historical data to compare.")
                else:
                    comparison_data[ticker] = hist

                    # Normalize and plot current state
                    combined = pd.DataFrame(comparison_data).dropna()
                    combined = combined / combined.iloc[0] * 100
                    plot_price_comparison(combined)

                print("\nDo you want to compare with another stock?")
                print("1. Yes")
                print("2. No")
                compare_more = input("Enter your choice: ").strip()
                if compare_more != '1':
                    break

        elif choice == '2':
            ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
            sector, asset_class, latest_price = fetch_asset_info(ticker)
            if sector is None or latest_price is None:
                print(f"Failed to fetch info for {ticker}.")
                continue

            ytd_return = fetch_ytd_performance(ticker)
            display_asset_info(ticker, sector, asset_class, latest_price, ytd_return)

            buy = input("Do you want to buy this stock? (1. Yes / 2. No): ").strip()
            if buy == '1':
                quantity = float(input("Enter quantity to buy: ").strip())
                portfolio.add_asset(ticker, sector, asset_class, quantity, latest_price)
                print(f"{ticker} added to your portfolio.")

        elif choice == '3':
            df, _ = portfolio.get_portfolio_dataframe()
            if df is None or df.empty:
                print("Portfolio is empty.")
            else:
                print("\nCurrent Portfolio:")
                for idx, row in df.iterrows():
                    print(f"{idx+1}. {row['Ticker']} - {row['Quantity']} shares @ ${row['Purchase Price']:.2f}")
                    print(f"   Sector: {row['Sector']}, Asset Class: {row['Asset Class']}")
                    print(f"   Current Value: ${row['Current Value']:,.2f} | Weight: {row['Weight (%)']:.2f}%\n")

        elif choice == '4':
            df, total_value = portfolio.get_portfolio_dataframe()
            if df is None or df.empty or total_value == 0:
                print("Portfolio is empty or contains invalid pricing.")
            else:
                by_class = df.groupby('Asset Class')['Current Value'].sum()
                by_sector = df.groupby('Sector')['Current Value'].sum()
                display_portfolio_summary(df, total_value)
                plot_portfolio_summary(df, by_class, by_sector)

        elif choice == '5':
            end_values, init_val = simulate_portfolio(portfolio)
            if end_values is not None:
                show_simulation_results(end_values, init_val)
            else:
                print("Simulation could not be performed.")

        elif choice == '6':
            print("Exiting Portfolio Tracker.")
            break

        else:
            print("Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()
