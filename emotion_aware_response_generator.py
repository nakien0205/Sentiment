"""
Emotion-Aware Response Generation System

This module implements the emotion-aware response generation system for the chatbot,
building on the T5 with LoRA integration.
"""

import torch
import numpy as np
from transformers import pipeline
from t5_emotion_chatbot import T5EmotionChatbot

class EmotionAwareResponseGenerator:
    """
    A system for generating emotion-aware responses using the fine-tuned T5 model.
    This class enhances the base T5EmotionChatbot with additional emotion-aware
    response generation capabilities.
    """
    
    def __init__(self, model_path=None, base_model="google/flan-t5-base"):
        """
        Initialize the emotion-aware response generator.
        
        Args:
            model_path (str): Path to a fine-tuned model, if available
            base_model (str): Base model to use if no fine-tuned model is provided
        """
        # Initialize the base T5 emotion chatbot
        self.chatbot = T5EmotionChatbot(model_name=base_model)
        
        # Load fine-tuned model if provided
        if model_path:
            try:
                self.chatbot.load_fine_tuned_model(model_path)
                print(f"Loaded fine-tuned model from {model_path}")
            except Exception as e:
                print(f"Error loading fine-tuned model: {e}")
                print("Using base model instead")
        
        # Initialize emotion intensity analyzer
        self.emotion_intensity = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        
        # Define emotion-specific response strategies
        self.emotion_strategies = {
            "happiness": self._happiness_strategy,
            "sadness": self._sadness_strategy,
            "fear": self._fear_strategy,
            "anger": self._anger_strategy,
            "surprise": self._surprise_strategy,
            "disgust": self._disgust_strategy,
            "neutral": self._neutral_strategy,
            "excitement": self._excitement_strategy,
            "gratitude": self._gratitude_strategy,
            "love": self._love_strategy,
            "confusion": self._confusion_strategy,
            "disappointment": self._disappointment_strategy
        }
    
    def generate_response(self, user_input, conversation_history=None):
        """
        Generate an emotion-aware response to the user input.
        
        Args:
            user_input (str): The user's message
            conversation_history (list): Optional conversation history
            
        Returns:
            dict: A dictionary containing the response, detected emotion, and confidence
        """
        # Update conversation history if provided
        if conversation_history:
            self.chatbot.conversation_history = conversation_history
        
        # Detect emotion and intensity
        emotion_data = self._analyze_emotion(user_input)
        detected_emotion = emotion_data["emotion"]
        emotion_intensity = emotion_data["intensity"]
        
        # Generate base response using the T5 model
        base_response = self.chatbot.generate_response(user_input)
        
        # Apply emotion-specific strategy to enhance the response
        if detected_emotion in self.emotion_strategies and emotion_intensity > 0.5:
            enhanced_response = self.emotion_strategies[detected_emotion](
                user_input, 
                base_response, 
                emotion_intensity
            )
        else:
            enhanced_response = base_response
        
        return {
            "response": enhanced_response,
            "emotion": detected_emotion,
            "intensity": emotion_intensity,
            "conversation_history": self.chatbot.conversation_history
        }
    
    def _analyze_emotion(self, text):
        """
        Analyze the emotion in the given text, including intensity.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: A dictionary containing the detected emotion and intensity
        """
        # Use the chatbot's emotion detection
        primary_emotion = self.chatbot.detect_emotion(text)
        
        # Get emotion intensity using the pipeline
        try:
            intensity_scores = self.emotion_intensity(text)[0]
            
            # Map pipeline emotion labels to our emotion set
            emotion_map = {
                "joy": "happiness",
                "sadness": "sadness",
                "anger": "anger",
                "fear": "fear",
                "surprise": "surprise",
                "disgust": "disgust",
                "neutral": "neutral"
            }
            
            # Find the intensity of the primary emotion
            intensity = 0.5  # Default medium intensity
            for score in intensity_scores:
                mapped_emotion = emotion_map.get(score["label"], "neutral")
                if mapped_emotion == primary_emotion:
                    intensity = score["score"]
                    break
        except Exception:
            # If the pipeline fails, use a default intensity
            intensity = 0.7
        
        return {
            "emotion": primary_emotion,
            "intensity": intensity
        }
    
    def _happiness_strategy(self, user_input, base_response, intensity):
        """Strategy for responding to happiness."""
        # For high intensity happiness, match the enthusiasm
        if intensity > 0.8:
            # Add enthusiasm markers like exclamation points if not already present
            if not base_response.endswith("!") and not "!" in base_response:
                base_response = base_response + "!"
            
            # Add positive reinforcement
            happiness_enhancers = [
                "That's wonderful to hear! ",
                "I'm so glad! ",
                "That's fantastic! ",
                "How wonderful! "
            ]
            
            if not any(enhancer in base_response for enhancer in happiness_enhancers):
                return np.random.choice(happiness_enhancers) + base_response
        
        return base_response
    
    def _sadness_strategy(self, user_input, base_response, intensity):
        """Strategy for responding to sadness."""
        # For high intensity sadness, offer support and validation
        if intensity > 0.7:
            # Check if the response already contains empathetic elements
            empathy_phrases = [
                "I'm sorry to hear that",
                "That sounds difficult",
                "I understand how you feel",
                "That must be hard"
            ]
            
            if not any(phrase in base_response.lower() for phrase in empathy_phrases):
                sadness_responses = [
                    "I'm sorry you're feeling this way. " + base_response,
                    "That sounds really difficult. " + base_response,
                    "I understand this is hard for you. " + base_response,
                    base_response + " Would you like to talk more about what's making you feel this way?"
                ]
                return np.random.choice(sadness_responses)
            
            # Add a supportive question if not already present
            if not "?" in base_response and intensity > 0.9:
                support_questions = [
                    " Would you like to talk more about it?",
                    " Is there anything specific that's troubling you?",
                    " Would it help to discuss what's on your mind?"
                ]
                return base_response + np.random.choice(support_questions)
        
        return base_response
    
    def _fear_strategy(self, user_input, base_response, intensity):
        """Strategy for responding to fear."""
        # For high intensity fear, provide reassurance and practical support
        if intensity > 0.7:
            reassurance_phrases = [
                "It's okay to feel scared",
                "I understand your concern",
                "That sounds frightening",
                "It's natural to feel anxious"
            ]
            
            if not any(phrase in base_response.lower() for phrase in reassurance_phrases):
                fear_responses = [
                    "I understand you're feeling afraid. " + base_response,
                    "It's okay to feel this way. " + base_response,
                    base_response + " Remember that facing fears often makes them less powerful.",
                    "I hear your concern. " + base_response
                ]
                return np.random.choice(fear_responses)
        
        return base_response
    
    def _anger_strategy(self, user_input, base_response, intensity):
        """Strategy for responding to anger."""
        # For high intensity anger, acknowledge feelings and de-escalate
        if intensity > 0.7:
            de_escalation_phrases = [
                "I understand you're frustrated",
                "I can see why you'd feel that way",
                "That sounds really frustrating",
                "I appreciate you sharing your feelings"
            ]
            
            if not any(phrase in base_response.lower() for phrase in de_escalation_phrases):
                anger_responses = [
                    "I understand you're feeling upset. " + base_response,
                    "I can see why that would be frustrating. " + base_response,
                    base_response + " Would it help to take a moment to discuss this calmly?",
                    "Your feelings are valid. " + base_response
                ]
                return np.random.choice(anger_responses)
        
        return base_response
    
    def _surprise_strategy(self, user_input, base_response, intensity):
        """Strategy for responding to surprise."""
        # For high intensity surprise, acknowledge the unexpected nature
        if intensity > 0.7:
            surprise_enhancers = [
                "Wow! ",
                "That's unexpected! ",
                "How surprising! ",
                "I didn't see that coming either! "
            ]
            
            if not any(enhancer in base_response for enhancer in surprise_enhancers):
                return np.random.choice(surprise_enhancers) + base_response
        
        return base_response
    
    def _disgust_strategy(self, user_input, base_response, intensity):
        """Strategy for responding to disgust."""
        # For high intensity disgust, validate feelings and offer perspective
        if intensity > 0.7:
            disgust_responses = [
                "I understand that's disgusting to you. " + base_response,
                "I can see why you'd feel that way. " + base_response,
                base_response + " Would you prefer to talk about something else?",
                "That does sound unpleasant. " + base_response
            ]
            return np.random.choice(disgust_responses)
        
        return base_response
    
    def _neutral_strategy(self, user_input, base_response, intensity):
        """Strategy for responding to neutral emotions."""
        # For neutral emotions, focus on engagement and information
        engagement_enhancers = [
            "Is there anything specific you'd like to know more about?",
            "Would you like to explore this topic further?",
            "Is there anything else you'd like to discuss?",
            "Does that answer your question?"
        ]
        
        # Add an engagement question if the response doesn't already have a question
        if not "?" in base_response and len(base_response.split()) > 5:
            return base_response + " " + np.random.choice(engagement_enhancers)
        
        return base_response
    
    def _excitement_strategy(self, user_input, base_response, intensity):
        """Strategy for responding to excitement."""
        # Similar to happiness but with more energy
        if intensity > 0.7:
            excitement_enhancers = [
                "That's so exciting! ",
                "How thrilling! ",
                "I'm excited for you! ",
                "That's amazing news! "
            ]
            
            if not any(enhancer in base_response for enhancer in excitement_enhancers):
                return np.random.choice(excitement_enhancers) + base_response
        
        return base_response
    
    def _gratitude_strategy(self, user_input, base_response, intensity):
        """Strategy for responding to gratitude."""
        gratitude_responses = [
            "You're very welcome! " + base_response,
            "I'm glad I could help. " + base_response,
            "It's my pleasure. " + base_response,
            "Anytime! " + base_response
        ]
        
        if "thank" in user_input.lower() and not any(phrase in base_response.lower() for phrase in ["welcome", "pleasure", "glad"]):
            return np.random.choice(gratitude_responses)
        
        return base_response
    
    def _love_strategy(self, user_input, base_response, intensity):
        """Strategy for responding to love/affection."""
        if intensity > 0.7:
            love_responses = [
                "That's really heartwarming. " + base_response,
                "I'm touched by your affection. " + base_response,
                base_response + " It's wonderful to experience such positive feelings.",
                "Those are beautiful sentiments. " + base_response
            ]
            return np.random.choice(love_responses)
        
        return base_response
    
    def _confusion_strategy(self, user_input, base_response, intensity):
        """Strategy for responding to confusion."""
        if intensity > 0.6:
            clarification_responses = [
                "Let me try to clarify. " + base_response,
                "I understand this might be confusing. " + base_response,
                base_response + " Does that help clear things up?",
                "To put it another way: " + base_response
            ]
            return np.random.choice(clarification_responses)
        
        return base_response
    
    def _disappointment_strategy(self, user_input, base_response, intensity):
        """Strategy for responding to disappointment."""
        if intensity > 0.7:
            disappointment_responses = [
                "I understand your disappointment. " + base_response,
                "That's unfortunate. " + base_response,
                base_response + " I hope things improve soon.",
                "I'm sorry things didn't work out as expected. " + base_response
            ]
            return np.random.choice(disappointment_responses)
        
        return base_response

# Example usage
if __name__ == "__main__":
    # Initialize the response generator
    response_generator = EmotionAwareResponseGenerator()
    
    # Example of generating a response
    # result = response_generator.generate_response("I'm feeling really anxious about my upcoming presentation.")
    # print(f"Emotion: {result['emotion']} (Intensity: {result['intensity']:.2f})")
    # print(f"Response: {result['response']}")
