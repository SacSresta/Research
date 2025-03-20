from data_pipeline.stage_02.merger import merge_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import argparse
import os

# Load FinBERT model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Function to get sentiment scores
def get_finbert_sentiment(headlines):
    sentiments = []
    for headline in headlines:
        inputs = tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        # Get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().numpy()[0]
        # FinBERT classes: negative (0), neutral (1), positive (2)
        sentiment_score = probs[2] - probs[0]  # positive - negative
        sentiments.append(sentiment_score)

    return np.array(sentiments)
def parse_args():

    parser = argparse.ArgumentParser(description="Fetch and merge market and news data")
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (e.g., AAPL, MSFT)')
    parser.add_argument('--start_date', type=str, default='2025-01-01', help="Start date in 'YYYY-MM-DD' format")
    parser.add_argument('--end_date', type=str, default='2025-03-01', help="End date in 'YYYY-MM-DD' format")
    parser.add_argument('--categorical', type=bool, default=False, help="If required class or pos/neg/neutral")
    
    return parser.parse_args()

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def preprocess_text(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    return inputs


def classify_sentiment(text):
    inputs = preprocess_text(text)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    sentiment_labels = ["positive", "negative", "neutral"]
    predicted_class = sentiment_labels[torch.argmax(probs).item()]
    return predicted_class


if __name__ == "__main__":
    # Load data
    try:
        args = parse_args()
        print(args)
        print(args.end_date)  
        categorical = args.categorical
        merged_df = merge_data(symbol=args.symbol, start_date=args.start_date, end_date=args.end_date)
        print(merged_df.tail(2))
        merged_df['sentiment_score'] = get_finbert_sentiment(merged_df['headline'])
        if categorical:
            merged_df['sentiment_class'] = merged_df['headline'].apply(classify_sentiment)
            output_dir = 'sentiment_categorical'  
            output_file = f'merged_data_{args.symbol}_from_{args.start_date}_to_{args.end_date}.csv'  
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_file)
            merged_df.to_csv(output_path, index=False)
        
        else:
            output_dir = 'sentiment'  
            output_file = f'merged_data_{args.symbol}_from_{args.start_date}_to_{args.end_date}.csv'  
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_file)
            merged_df.to_csv(output_path, index=False)
        
    except Exception as e:
        print(f"An error occurred: {e}")


