import time
from data_pipeline.stage_03_score.score import run_sentiment_analysis

ticker_configs = {
    "AAPL": {
        "custom_keywords": [
            "Apple Inc.", "iPhone", "MacBook", "iPad", "Apple Watch", "iOS",
            "Tim Cook", "App Store", "MacOS", "M1 Chip", "Apple Car", "Apple Pay",
            "Earnings", "Revenue", "Stock Split", "Supply Chain", "Foxconn", "China"
        ],
        "false_positives": ["fruit", "tree", "apple orchard"]
    },
    "MSFT": {
        "custom_keywords": [
            "Microsoft", "Windows", "Azure", "Xbox", "Office 365", "LinkedIn",
            "Satya Nadella", "GitHub", "AI Copilot", "Bing AI", "Surface", 
            "Cloud Computing", "Enterprise Software", "Earnings", "Stock Buyback"
        ],
        "false_positives": ["microscope", "soft surface"]
    },
    "AMZN": {
        "custom_keywords": [
            "Amazon", "AWS", "Amazon Prime", "Jeff Bezos", "Kindle", "E-commerce",
            "Fulfillment Center", "Whole Foods", "Alexa", "Cloud Revenue",
            "Retail Sales", "Earnings", "Stock Split", "Logistics", "Supply Chain"
        ],
        "false_positives": ["rainforest", "river", "Amazon forest"]
    },
    "NVDA": {
        "custom_keywords": [
            "NVIDIA", "GPU", "AI Chips", "CUDA", "Hopper", "DLSS", "RTX", "GeForce",
            "Jensen Huang", "Data Center", "Semiconductor", "Artificial Intelligence",
            "Earnings", "Chip Shortage", "Stock Buyback", "Revenue Growth"
        ],
        "false_positives": ["video cards", "video editing", "gaming accessories"]
    },
    "TSLA": {
        "custom_keywords": [
            "Tesla", "Elon Musk", "Model 3", "Model S", "Cybertruck", "Gigafactory",
            "Autopilot", "EV Market", "Tesla Energy", "Battery Production",
            "Self-Driving", "Earnings", "Stock Split", "EV Sales", "China Expansion"
        ],
        "false_positives": ["Nikola Tesla", "physics", "Tesla coil"]
    },
    "GOOGL": {
        "custom_keywords": [
            "Alphabet", "Google", "Sundar Pichai", "YouTube", "Cloud Revenue",
            "Search Ads", "Android", "Waymo", "AI", "DeepMind", "Google Pixel",
            "Earnings", "Stock Buyback", "Antitrust", "Regulation"
        ],
        "false_positives": ["googly eyes", "alphabet letters"]
    },
    "META": {
        "custom_keywords": [
            "Meta", "Facebook", "Instagram", "WhatsApp", "Mark Zuckerberg",
            "Reels", "Metaverse", "VR", "Oculus", "AI", "Advertising Revenue",
            "Earnings", "Stock Buyback", "User Growth", "Ad Revenue"
        ],
        "false_positives": ["metadata", "metaphor"]
    },
    "NFLX": {
        "custom_keywords": [
            "Netflix", "Streaming", "OTT", "Reed Hastings", "Original Content",
            "Subscriber Growth", "Earnings", "Ad-Supported Tier", "Licensing Deals",
            "Content Budget", "Stock Buyback", "Revenue Growth"
        ],
        "false_positives": ["internet speed", "cable TV"]
    }
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
                custom_keywords=None,
                false_positives=None
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