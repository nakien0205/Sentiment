"""
Sentiment Analysis Model using Hugging Face Transformers
This module implements a sentiment analysis model that can detect 7 basic emotions:
happiness, sadness, fear, anger, surprise, disgust, and neutral.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class SentimentAnalyzer:
    """
    A class for analyzing sentiment/emotion in text using pre-trained models from Hugging Face.
    """
    
    def __init__(self):
        """
        Initialize the sentiment analyzer with a pre-trained model for emotion detection.
        """
        # Using j-hartmann/emotion-english-distilroberta-base which is trained for emotion classification
        # It can detect 7 emotions: joy (happiness), sadness, fear, anger, surprise, disgust, and neutral
        self.model_name = "j-hartmann/emotion-english-distilroberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        # Define emotion labels (mapping from model output to our required emotions)
        self.emotion_labels = {
            0: "anger",
            1: "disgust",
            2: "fear",
            3: "happiness",  # joy in the model
            4: "neutral",
            5: "sadness",
            6: "surprise"
        }
    
    def analyze_text(self, text):
        """
        Analyze the sentiment/emotion in the given text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: A dictionary containing the detected emotion and confidence scores
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze().numpy()
            
        # Convert to probabilities
        probs = np.exp(scores) / np.sum(np.exp(scores))
        
        # Get the predicted emotion (highest probability)
        predicted_class = np.argmax(probs)
        emotion = self.emotion_labels[predicted_class]
        confidence = float(probs[predicted_class])
        
        # Get all emotion probabilities
        emotion_probs = {self.emotion_labels[i]: float(probs[i]) for i in range(len(probs))}
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "all_emotions": emotion_probs
        }

# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test with different emotion examples
    test_texts = [
        "I'm so happy today! Everything is going well.",
        "I feel really sad and disappointed about what happened.",
        "I'm terrified of what might happen next.",
        "This makes me so angry! I can't believe they did that.",
        "Wow! I didn't expect that at all!",
        "That's disgusting, I can't stand it.",
        "It's just an ordinary day, nothing special."
    ]
    
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"Text: {text}")
        print(f"Detected emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
        print("All emotions:", {k: f"{v:.2f}" for k, v in result['all_emotions'].items()})
        print("-" * 50)
