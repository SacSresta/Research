import time
from data_pipeline.stage_03_score.score import run_sentiment_analysis

ticker_configs = {
    "AAPL": {
        "custom_keywords": ["iPhone", "Mac", "iPad", "Apple Watch", "iOS"],
        "false_positives": ["fruit", "tree"]
    },
    "MSFT": {
        "custom_keywords": ["Windows", "Azure", "Xbox", "Office 365", "LinkedIn"],
        "false_positives": ["microscope", "soft surface"]
    },
    "AMZN": {
        "custom_keywords": ["AWS", "Prime", "Alexa", "Kindle", "e-commerce"],
        "false_positives": ["rainforest", "river"]
    },
    "NVDA": {
        "custom_keywords": ["GPU", "CUDA", "AI chips", "Hopper", "DLSS"],
        "false_positives": ["video cards", "video editing"]
    },
    "TSLA": {
        "custom_keywords": ["Model 3", "Cybertruck", "Gigafactory", "Autopilot"],
        "false_positives": ["Nikola Tesla", "physics"]
    },
   
}


start_date = "2015-01-01"
end_date = "2025-03-01"

def process_tickers():
    """Process sentiment analysis for all configured tickers with error handling"""
    for symbol, config in ticker_configs.items():
        try:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}")
            print(f"Keywords: {config['custom_keywords']}")
            print(f"Excluding false positives: {config['false_positives']}")
            
            start_time = time.time()
            
            run_sentiment_analysis(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                categorical=False,
                custom_keywords=config["custom_keywords"],
                false_positives=config["false_positives"]
            )
            
            elapsed = time.time() - start_time
            print(f"Completed {symbol} in {elapsed:.2f} seconds")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue  

if __name__ == "__main__":
    print("Starting sentiment analysis pipeline...")
    process_tickers()
    print("\nPipeline completed!")