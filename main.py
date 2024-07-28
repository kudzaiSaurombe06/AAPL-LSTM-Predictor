import yfinance as yf

# Download Apple stock data
apple = yf.Ticker("AAPL")
apple_data = apple.history(period="max")

# Save the data to a CSV file
apple_data.to_csv("apple_stock_data.csv")