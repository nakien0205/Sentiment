"""
Conversation Management System for Emotion-Aware Chatbot

This module implements a conversation management system that maintains context,
handles conversation flow, and ensures coherent interactions with the chatbot.
"""

import json
import time
from collections import deque
import numpy as np
from emotion_aware_response_generator import EmotionAwareResponseGenerator

class ConversationManager:
    """
    A system for managing conversations with the emotion-aware chatbot.
    This class handles conversation history, context tracking, and ensures
    coherent and contextually appropriate responses.
    """
    
    def __init__(self, model_path=None, base_model="google/flan-t5-base", max_history=10):
        """
        Initialize the conversation manager.
        
        Args:
            model_path (str): Path to a fine-tuned model, if available
            base_model (str): Base model to use if no fine-tuned model is provided
            max_history (int): Maximum number of conversation turns to maintain
        """
        # Initialize the emotion-aware response generator
        self.response_generator = EmotionAwareResponseGenerator(
            model_path=model_path,
            base_model=base_model
        )
        
        # Initialize conversation state
        self.conversation_id = self._generate_conversation_id()
        self.start_time = time.time()
        self.last_activity_time = self.start_time
        self.max_history = max_history
        self.conversation_history = []
        self.emotion_history = deque(maxlen=max_history)
        self.topic_history = []
        self.current_topic = None
        self.user_preferences = {}
        self.conversation_metrics = {
            "turn_count": 0,
            "avg_response_time": 0,
            "total_response_time": 0,
            "emotion_distribution": {}
        }
    
    def process_message(self, user_input):
        """
        Process a user message and generate a response.
        
        Args:
            user_input (str): The user's message
            
        Returns:
            dict: A dictionary containing the response and conversation state
        """
        # Update activity time
        self.last_activity_time = time.time()
        
        # Track conversation metrics
        start_time = time.time()
        self.conversation_metrics["turn_count"] += 1
        
        # Extract potential topics from user input
        self._update_topics(user_input)
        
        # Generate response
        result = self.response_generator.generate_response(
            user_input, 
            conversation_history=self.conversation_history
        )
        
        response = result["response"]
        detected_emotion = result["emotion"]
        emotion_intensity = result["intensity"]
        
        # Update conversation history from the response generator
        self.conversation_history = result["conversation_history"]
        
        # Update emotion history
        self.emotion_history.append({
            "emotion": detected_emotion,
            "intensity": emotion_intensity,
            "timestamp": time.time()
        })
        
        # Update emotion distribution metrics
        if detected_emotion in self.conversation_metrics["emotion_distribution"]:
            self.conversation_metrics["emotion_distribution"][detected_emotion] += 1
        else:
            self.conversation_metrics["emotion_distribution"][detected_emotion] = 1
        
        # Calculate response time and update metrics
        response_time = time.time() - start_time
        self.conversation_metrics["total_response_time"] += response_time
        self.conversation_metrics["avg_response_time"] = (
            self.conversation_metrics["total_response_time"] / 
            self.conversation_metrics["turn_count"]
        )
        
        # Check for and handle conversation drift
        if self._detect_conversation_drift():
            response = self._handle_conversation_drift(response)
        
        # Check for and extract user preferences
        self._extract_user_preferences(user_input)
        
        return {
            "response": response,
            "emotion": detected_emotion,
            "intensity": emotion_intensity,
            "conversation_id": self.conversation_id,
            "turn_count": self.conversation_metrics["turn_count"],
            "current_topic": self.current_topic,
            "conversation_duration": time.time() - self.start_time
        }
    
    def save_conversation(self, file_path):
        """
        Save the current conversation to a file.
        
        Args:
            file_path (str): Path to save the conversation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conversation_data = {
                "conversation_id": self.conversation_id,
                "start_time": self.start_time,
                "last_activity_time": self.last_activity_time,
                "conversation_history": self.conversation_history,
                "emotion_history": list(self.emotion_history),
                "topic_history": self.topic_history,
                "current_topic": self.current_topic,
                "user_preferences": self.user_preferences,
                "conversation_metrics": self.conversation_metrics
            }
            
            with open(file_path, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def load_conversation(self, file_path):
        """
        Load a conversation from a file.
        
        Args:
            file_path (str): Path to the conversation file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                conversation_data = json.load(f)
            
            self.conversation_id = conversation_data["conversation_id"]
            self.start_time = conversation_data["start_time"]
            self.last_activity_time = conversation_data["last_activity_time"]
            self.conversation_history = conversation_data["conversation_history"]
            self.emotion_history = deque(conversation_data["emotion_history"], maxlen=self.max_history)
            self.topic_history = conversation_data["topic_history"]
            self.current_topic = conversation_data["current_topic"]
            self.user_preferences = conversation_data["user_preferences"]
            self.conversation_metrics = conversation_data["conversation_metrics"]
            
            return True
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return False
    
    def get_conversation_summary(self):
        """
        Generate a summary of the current conversation.
        
        Returns:
            dict: A summary of the conversation
        """
        # Calculate emotion distribution percentages
        total_emotions = sum(self.conversation_metrics["emotion_distribution"].values())
        emotion_percentages = {}
        
        if total_emotions > 0:
            for emotion, count in self.conversation_metrics["emotion_distribution"].items():
                emotion_percentages[emotion] = (count / total_emotions) * 100
        
        # Get dominant emotion
        dominant_emotion = max(
            self.conversation_metrics["emotion_distribution"].items(), 
            key=lambda x: x[1]
        )[0] if self.conversation_metrics["emotion_distribution"] else "neutral"
        
        return {
            "conversation_id": self.conversation_id,
            "duration": time.time() - self.start_time,
            "turn_count": self.conversation_metrics["turn_count"],
            "avg_response_time": self.conversation_metrics["avg_response_time"],
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_percentages,
            "topics_discussed": self.topic_history,
            "current_topic": self.current_topic
        }
    
    def reset_conversation(self):
        """
        Reset the conversation to a new state.
        
        Returns:
            str: The new conversation ID
        """
        self.conversation_id = self._generate_conversation_id()
        self.start_time = time.time()
        self.last_activity_time = self.start_time
        self.conversation_history = []
        self.emotion_history.clear()
        self.topic_history = []
        self.current_topic = None
        self.user_preferences = {}
        self.conversation_metrics = {
            "turn_count": 0,
            "avg_response_time": 0,
            "total_response_time": 0,
            "emotion_distribution": {}
        }
        
        return self.conversation_id
    
    def _generate_conversation_id(self):
        """Generate a unique conversation ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _update_topics(self, text):
        """
        Extract and update conversation topics from text.
        
        Args:
            text (str): The text to analyze for topics
        """
        # This is a simplified topic extraction
        # In a real implementation, you might use keyword extraction or topic modeling
        
        # List of potential topic keywords
        topics = {
            "work": ["job", "work", "career", "office", "boss", "colleague"],
            "education": ["school", "university", "college", "study", "learn", "education", "student"],
            "health": ["health", "doctor", "hospital", "sick", "illness", "disease", "medicine"],
            "relationships": ["relationship", "friend", "family", "partner", "love", "marriage", "date"],
            "entertainment": ["movie", "music", "book", "game", "play", "watch", "read", "listen"],
            "technology": ["computer", "phone", "tech", "software", "app", "device", "internet"],
            "travel": ["travel", "trip", "vacation", "visit", "country", "city", "flight"],
            "food": ["food", "eat", "restaurant", "cook", "meal", "recipe", "taste"],
            "sports": ["sport", "game", "team", "play", "win", "match", "exercise"],
            "weather": ["weather", "rain", "sun", "temperature", "cold", "hot", "forecast"]
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topics.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        if detected_topics:
            # Update current topic to the most recently detected one
            self.current_topic = detected_topics[0]
            
            # Add to topic history if it's a new topic
            if not self.topic_history or self.topic_history[-1] != self.current_topic:
                self.topic_history.append(self.current_topic)
    
    def _detect_conversation_drift(self):
        """
        Detect if the conversation has drifted from its context.
        
        Returns:
            bool: True if drift is detected, False otherwise
        """
        # Check if we have enough history to detect drift
        if len(self.conversation_history) < 4:
            return False
        
        # Check for topic consistency in recent messages
        recent_topics = set()
        for i in range(min(3, len(self.topic_history))):
            if i < len(self.topic_history):
                recent_topics.add(self.topic_history[-(i+1)])
        
        # If there are multiple topics in recent history, potential drift
        return len(recent_topics) > 2
    
    def _handle_conversation_drift(self, response):
        """
        Handle detected conversation drift.
        
        Args:
            response (str): The current response
            
        Returns:
            str: Modified response to address drift
        """
        # If drift is detected, acknowledge the change in topic
        if self.current_topic and self.topic_history and len(self.topic_history) > 1:
            previous_topic = self.topic_history[-2]
            
            # Only acknowledge if it's a significant change
            if previous_topic != self.current_topic:
                topic_transitions = [
                    f"Shifting to {self.current_topic}, ",
                    f"On the topic of {self.current_topic}, ",
                    f"Regarding {self.current_topic}, ",
                    f"Speaking about {self.current_topic}, "
                ]
                
                # Add a transition phrase if the response doesn't already reference the topic
                if not self.current_topic.lower() in response.lower():
                    return np.random.choice(topic_transitions) + response
        
        return response
    
    def _extract_user_preferences(self, text):
        """
        Extract potential user preferences from text.
        
        Args:
            text (str): The text to analyze for preferences
        """
        # This is a simplified preference extraction
        # In a real implementation, you might use more sophisticated NLP techniques
        
        # Look for preference indicators
        like_indicators = ["like", "love", "enjoy", "prefer", "favorite"]
        dislike_indicators = ["dislike", "hate", "don't like", "can't stand"]
        
        text_lower = text.lower()
        
        # Extract likes
        for indicator in like_indicators:
            if indicator in text_lower:
                # Find what comes after the indicator
                parts = text_lower.split(indicator, 1)
                if len(parts) > 1 and parts[1].strip():
                    # Extract the first few words after the indicator
                    preference = parts[1].strip().split(".")[0].strip()
                    # Limit to a reasonable length
                    preference = " ".join(preference.split()[:5])
                    if preference:
                        self.user_preferences[preference] = "like"
        
        # Extract dislikes
        for indicator in dislike_indicators:
            if indicator in text_lower:
                # Find what comes after the indicator
                parts = text_lower.split(indicator, 1)
                if len(parts) > 1 and parts[1].strip():
                    # Extract the first few words after the indicator
                    preference = parts[1].strip().split(".")[0].strip()
                    # Limit to a reasonable length
                    preference = " ".join(preference.split()[:5])
                    if preference:
                        self.user_preferences[preference] = "dislike"

# Example usage
if __name__ == "__main__":
    # Initialize the conversation manager
    conversation_manager = ConversationManager()
    
    # Example of processing a message
    # result = conversation_manager.process_message("I'm feeling really anxious about my upcoming presentation at work.")
    # print(f"Response: {result['response']}")
    # print(f"Detected emotion: {result['emotion']} (Intensity: {result['intensity']:.2f})")
    # print(f"Current topic: {result['current_topic']}")
    
    # Example of saving and loading a conversation
    # conversation_manager.save_conversation("conversation.json")
    # conversation_manager.load_conversation("conversation.json")
    
    # Example of getting a conversation summary
    # summary = conversation_manager.get_conversation_summary()
    # print(f"Conversation summary: {summary}")
