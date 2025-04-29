# views/portfolio_view.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def display_asset_info(ticker, sector, asset_class, latest_price, ytd_return):
    print(f"\nAsset Info for {ticker}:")
    print(f"Sector: {sector}")
    print(f"Asset Class: {asset_class}")
    print(f"Latest Price: ${latest_price:.2f}")
    if ytd_return is not None:
        print(f"YTD Performance: {ytd_return:.2%}")
    else:
        print("YTD Performance: Not available")

def plot_price_comparison(combined_data):
    plt.figure(figsize=(10, 5))
    for col in combined_data.columns:
        plt.plot(combined_data.index, combined_data[col], label=col)
    plt.title("1-Year Price Comparison (Indexed to 100)")
    plt.xlabel("Date")
    plt.ylabel("Indexed Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def display_portfolio_summary(df, total_value):
    if df is None or total_value == 0:
        print("Portfolio is empty or invalid.")
        return

    by_class = df.groupby('Asset Class')['Current Value'].sum()
    by_sector = df.groupby('Sector')['Current Value'].sum()

    by_class_weights = by_class / total_value * 100
    by_sector_weights = by_sector / total_value * 100

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

def plot_portfolio_summary(df, by_class, by_sector):
    # Pie Chart by Asset
    plt.figure(figsize=(6, 6))
    plt.pie(df['Current Value'], labels=df['Ticker'], autopct='%1.1f%%', startangle=140)
    plt.title('Portfolio by Asset')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Pie Chart by Asset Class
    plt.figure(figsize=(6, 6))
    plt.pie(by_class, labels=by_class.index, autopct='%1.1f%%', startangle=140)
    plt.title('Portfolio by Asset Class')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Pie Chart by Sector
    plt.figure(figsize=(6, 6))
    plt.pie(by_sector, labels=by_sector.index, autopct='%1.1f%%', startangle=140)
    plt.title('Portfolio by Sector')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()



def show_simulation_results(portfolio_end_values, initial_value):
    mean_ending = np.mean(portfolio_end_values)
    median_ending = np.median(portfolio_end_values)
    p5 = np.percentile(portfolio_end_values, 5)
    p95 = np.percentile(portfolio_end_values, 95)

    print(f"\n--- Portfolio Simulation ---")
    print(f"Expected Ending Value (Mean): ${mean_ending:,.2f}")
    print(f"Median Ending Value: ${median_ending:,.2f}")
    print(f"5th Percentile: ${p5:,.2f}")
    print(f"95th Percentile: ${p95:,.2f}")

    plt.figure(figsize=(10,6))
    plt.hist(portfolio_end_values, bins=100, density=True, alpha=0.7, color='skyblue')
    plt.axvline(initial_value, color='red', linestyle='--', linewidth=2, label='Initial Portfolio Value')
    plt.title('Distribution of Portfolio Ending Values')
    plt.xlabel('Portfolio Value ($)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)
    plt.show()
