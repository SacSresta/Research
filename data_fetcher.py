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
    },
     "SPY": {
        "custom_keywords": [
            "S&P 500", "Index Fund", "Market Index", "SPDR", "Sector Rotation",
            "Bull Market", "Bear Market", "Risk-On", "Risk-Off", "Market Cap Weighted",
            "Top Holdings", "Sector Exposure", "Fed Policy", "Inflation", "VIX",
            "Tech Heavyweight", "Market Breadth", "Yield Curve", "Macro Trends"
        ],
        "false_positives": ["spy agency", "espionage"]
    },
    "JPM": {
        "custom_keywords": [
            "JPMorgan", "Jamie Dimon", "Investment Banking", "Commercial Banking",
            "Trading Revenue", "Interest Income", "Wealth Management", "Basel III",
            "Loan Portfolio", "Credit Quality", "Stress Test", "Regulatory Capital"
        ],
        "false_positives": ["JP Morgan chase (the game)"]
    },
    "BTC-USD": {
        "custom_keywords": [
            "Bitcoin", "BTC", "Satoshi", "Halving", "Blockchain", "HODL",
            "Cryptocurrency", "Digital Gold", "Hash Rate", "Lightning Network",
            "Coinbase", "ETF Approval", "On-Chain", "Whale Activity", "Mining Difficulty"
        ],
        "false_positives": ["bit torrent", "bit rate"]
    },
    "ETH-USD": {
        "custom_keywords": [
            "Ethereum", "ETH", "Vitalik Buterin", "Smart Contracts", "DeFi",
            "NFT", "Gas Fees", "Proof of Stake", "EIP-1559", "Layer 2",
            "Dencun Upgrade", "Staking Yield", "ERC-20", "zK-Rollups", "Validator"
        ],
        "false_positives": ["ethernet"]
    },
    "ADA-USD": {
        "custom_keywords": [
            "Cardano", "ADA", "Charles Hoskinson", "Ouroboros", "Plutus",
            "Hydra", "Staking Pool", "Mary Hard Fork", "Alonzo Upgrade",
            "dApp Development", "Proof of Stake", "eUTXO"
        ],
        "false_positives": ["Ada Lovelace"]
    },
    "BNB-USD": {
        "custom_keywords": [
            "Binance Coin", "CZ", "Binance Chain", "BNB Burn", "Launchpad",
            "Smart Chain", "CEX", "DEX", "Token Launch", "BEP-20",
            "Centralized Exchange", "Regulatory Scrutiny", "Fiat Gateway"
        ],
        "false_positives": ["binary options"]
    },
    "COIN": {
        "custom_keywords": [
            "Coinbase", "Brian Armstrong", "Trading Volume", "Custody Solutions",
            "Staking Rewards", "Base Chain", "SEC Actions", "Spot Bitcoin ETF",
            "Exchange Reserves", "Retail Investors", "Institutional Adoption"
        ],
        "false_positives": ["coin collection"]
    },
    "SOL-USD": {
        "custom_keywords": [
            "Solana", "SOL", "Anatoly Yakovenko", "Sealevel", "Proof of History",
            "Memecoins", "Network Outage", "TPS", "Serum DEX", "Validator Requirements",
            "Jupiter Exchange", "Phantom Wallet"
        ],
        "false_positives": ["solution", "sunspot"]
    },
    "XRP-USD": {
        "custom_keywords": [
            "Ripple", "XRP", "SEC Lawsuit", "Cross-Border Payments", "ODL",
            "Brad Garlinghouse", "Escrow", "Swift Alternative", "Bank Adoption",
            "Regulatory Clarity", "On-Demand Liquidity", "MoneyGram"
        ],
        "false_positives": ["X-Ray"]
    },
    "DOGE-USD": {
        "custom_keywords": [
            "Dogecoin", "Elon Musk Tweet", "Meme Coin", "Proof of Work",
            "Billy Markus", "Jackson Palmer", "Shiba Inu", "Community Driven",
            "Inflationary Supply", "Tesla Merchandise", "Crypto Payment"
        ],
        "false_positives": ["actual dogs"]
    },
    "AVAX-USD": {
        "custom_keywords": [
            "Avalanche", "Emin Gün Sirer", "Subnets", "C-Chain", "Snowman Consensus",
            "Multichain", "Institutional Adoption", "Custom Blockchain",
            "Staking Yield", "Tokenization", "Fire Drops"
        ],
        "false_positives": ["avalanche warnings"]
    },
    "UNH": {
        "custom_keywords": [
            "UnitedHealth", "Optum", "Health Insurance", "Medicare Advantage",
            "Provider Network", "Premiums", "Claims Ratio", "Healthcare Costs",
            "Aetna", "Cigna", "HCAHPS Scores", "Affordable Care Act"
        ],
        "false_positives": ["unhealthy food"]
    },
    "AMD": {
        "custom_keywords": [
            "Advanced Micro Devices", "Lisa Su", "Ryzen", "Radeon", "Instinct MI300X",
            "AI Accelerators", "Server Market Share", "GPU Competition", "Xilinx Merger",
            "TSMC Partnership", "Semi-Custom Chips"
        ],
        "false_positives": ["AMD processor vulnerabilities"]
    },
    "TSM": {
        "custom_keywords": [
            "TSMC", "Semiconductor Manufacturing", "3nm Process", "Fab Capacity",
            "CoWoS Packaging", "Apple Silicon", "Chip Lead Times", "Geopolitical Risks",
            "Taiwan Relations", "Capital Expenditure", "Advanced Packaging"
        ],
        "false_positives": ["time-sharing"]
    },
    "LTC-USD": {
        "custom_keywords": [
            "Litecoin", "Charlie Lee", "Digital Silver", "MWEB", "Atomic Swaps",
            "Scrypt Algorithm", "Payment Coin", "Halving", "SegWit Adoption",
            "Transaction Speed", "Privacy Features"
        ],
        "false_positives": ["Lite version"]
    },
    "DOT-USD": {
        "custom_keywords": [
            "Polkadot", "Gavin Wood", "Parachain", "Substrate", "Nominated Proof of Stake",
            "Interoperability", "Web3 Foundation", "Cross-Chain", "Kusama Network",
            "Governance Referenda", "Staking Parameters"
        ],
        "false_positives": ["punctuation mark"]
    },
    "XOM": {
        "custom_keywords": [
            "ExxonMobil", "Darren Woods", "Upstream Production", "Downstream Margins",
            "Guyana Discoveries", "Carbon Capture", "Permian Basin", "LNG Exports",
            "Dividend Aristocrat", "Refining Capacity", "Oil Prices"
        ],
        "false_positives": ["X-Men movies"]
    },
    "BAC": {
        "custom_keywords": [
            "Bank of America", "Brian Moynihan", "Net Interest Income", "Loan Growth",
            "Deposit Costs", "Global Markets", "Consumer Banking", "Merrill Lynch",
            "Zelle", "Provision for Credit Losses", "FICC Trading"
        ],
        "false_positives": ["blood alcohol content"]
    },
    "INTC": {
        "custom_keywords": [
            "Intel", "Pat Gelsinger", "Foundry Services", "Core Ultra", "Gaudi AI Chips",
            "Process Node", "Competitive Position", "IDM 2.0", "CHIPS Act Funding",
            "Mobileye", "OpenVINO Toolkit"
        ],
        "false_positives": ["intelligence"]
    },
    "CVX": {
        "custom_keywords": [
            "Chevron", "Mike Wirth", "Upstream Earnings", "Permian Development",
            "Hess Acquisition", "Dividend Growth", "Carbon Intensity", "Renewables",
            "Gorgon Project", "Shale Oil", "Reserve Replacement"
        ],
        "false_positives": ["Chevy cars"]
    },
    "COST": {
        "custom_keywords": [
            "Costco", "Membership Fee", "Kirkland Signature", "Retail Margins",
            "Same-Store Sales", "E-commerce Growth", "Gasoline Sales",
            "Inventory Turnover", "Merchandising", "Global Expansion"
        ],
        "false_positives": ["cost reduction"]
    },
    "WMT": {
        "custom_keywords": [
            "Walmart", "Doug McMillon", "Same Day Delivery", "Inventory Management",
            "Omnichannel Strategy", "Sam's Club", "Advertising Revenue",
            "Supply Chain Automation", "Store Modernization", "Price Rollbacks"
        ],
        "false_positives": ["wall mount"]
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