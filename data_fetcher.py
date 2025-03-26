from data_pipeline.stage_03_score.score import run_sentiment_analysis

# Define the list of tickers (you can extend it to include all S&P 500 tickers)
tickers = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "NVDA",  # Nvidia
    "GOOGL", # Alphabet (Class A)
    "GOOG",  # Alphabet (Class C)
    "META",  # Meta Platforms (Facebook)
    "BRK.B", # Berkshire Hathaway (Class B)
    "TSLA",  # Tesla
    "UNH"    # UnitedHealth Group
]

# Define the date range
start_date = "2015-01-01"
end_date = "2025-03-01"

# Loop through each ticker and process sentiment analysis
for ticker in tickers:
    print(f"Processing {ticker}...")
    run_sentiment_analysis(symbol=ticker, start_date=start_date, end_date=end_date, categorical=False)

print("Processing completed for all tickers!")