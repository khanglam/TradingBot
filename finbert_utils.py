# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # For loading the FinBERT model
import torch  # For PyTorch operations
from typing import Tuple  # For type hints

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise CPU

# Initialize the FinBERT model and tokenizer
# FinBERT is a BERT model fine-tuned for financial sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]  # Possible sentiment outputs

def estimate_sentiment(news: list) -> Tuple[float, str]:
    """
    Analyze the sentiment of financial news articles using FinBERT
    
    Parameters:
    - news: list - List of news headlines to analyze
    
    Returns:
    - Tuple containing:
        - probability: float - Confidence score of the prediction
        - sentiment: str - The predicted sentiment ("positive", "negative", or "neutral")
    """
    if news:
        # Tokenize the input news articles
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        # Get model predictions
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        
        # Calculate probabilities using softmax
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        
        # Get the highest probability and corresponding sentiment
        probability = result[torch.argmax(result)]
        sentiment = labels[torch.argmax(result)]
        
        return probability, sentiment
    else:
        # Return neutral sentiment if no news is provided
        return 0, labels[-1]


if __name__ == "__main__":
    """
    Example usage of the sentiment analysis function
    """
    # Test with sample negative news headlines
    tensor, sentiment = estimate_sentiment([
        'markets responded negatively to the news!',
        'traders were displeased!'
    ])
    print(f"Sentiment Probability: {tensor}")
    print(f"Predicted Sentiment: {sentiment}")
    print(f"CUDA Available: {torch.cuda.is_available()}")