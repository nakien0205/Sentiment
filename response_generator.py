"""
Chatbot Response Logic
This module implements the response logic for different emotions detected by the sentiment analyzer.
"""

class ResponseGenerator:
    """
    A class for generating appropriate responses based on detected emotions.
    """
    
    def __init__(self):
        """
        Initialize the response generator with strategies for different emotions.
        """
        pass
    
    def generate_response(self, text, emotion_data):
        """
        Generate an appropriate response based on the detected emotion.
        
        Args:
            text (str): The original user input text
            emotion_data (dict): The emotion analysis data from the sentiment analyzer
            
        Returns:
            str: An appropriate response based on the detected emotion
        """
        emotion = emotion_data["emotion"]
        confidence = emotion_data["confidence"]
        
        # If confidence is too low, use a neutral response
        if confidence < 0.5:
            return "I'm not quite sure how you're feeling. Could you tell me more?"
        
        # Generate response based on emotion
        if emotion == "sadness":
            return self._respond_to_sadness(text)
        elif emotion == "fear":
            return self._respond_to_fear(text)
        elif emotion == "anger":
            return self._respond_to_anger(text)
        elif emotion == "happiness":
            return self._respond_to_happiness(text)
        elif emotion == "surprise":
            return self._respond_to_surprise(text)
        elif emotion == "disgust":
            return self._respond_to_disgust(text)
        else:  # neutral
            return self._respond_to_neutral(text)
    
    def _respond_to_sadness(self, text):
        """Generate response for sadness emotion."""
        # Identify what might be making them sad
        sad_triggers = self._identify_triggers(text, [
            "miss", "lost", "alone", "sad", "unhappy", "depressed", 
            "disappointed", "hurt", "heartbroken", "grief"
        ])
        
        if sad_triggers:
            return f"I notice you seem sad about {sad_triggers}. Would you like some advice, or would you prefer I just listen?"
        else:
            return "You sound sad. Would you like to talk more about what's bothering you, or would you prefer some encouragement?"
    
    def _respond_to_fear(self, text):
        """Generate response for fear emotion."""
        # Identify what might be causing fear
        fear_triggers = self._identify_triggers(text, [
            "afraid", "scared", "terrified", "anxious", "worried", 
            "nervous", "panic", "dread", "frightened", "fear"
        ])
        
        if fear_triggers:
            return f"I understand you're feeling afraid about {fear_triggers}. Remember that facing your fears often makes them less powerful. Would you like to talk about some coping strategies?"
        else:
            return "I can sense you're feeling fearful. Remember that courage isn't the absence of fear, but the triumph over it. Would it help to talk about what's causing this feeling?"
    
    def _respond_to_anger(self, text):
        """Generate response for anger emotion."""
        # Identify what might be causing anger
        anger_triggers = self._identify_triggers(text, [
            "angry", "mad", "furious", "annoyed", "irritated", 
            "frustrated", "upset", "outraged", "hate", "unfair"
        ])
        
        if anger_triggers:
            return f"I understand you're feeling angry about {anger_triggers}. Taking a deep breath and counting to ten can help calm those feelings. Would you like to discuss this more calmly?"
        else:
            return "I can tell you're feeling angry. It's natural to feel this way sometimes, but remember that anger rarely solves problems. Would you like to talk about what happened?"
    
    def _respond_to_happiness(self, text):
        """Generate response for happiness emotion."""
        return "I'm glad to hear you're feeling happy! It's wonderful to experience positive emotions. What's bringing you joy today?"
    
    def _respond_to_surprise(self, text):
        """Generate response for surprise emotion."""
        return "Wow! That does sound surprising. Would you like to tell me more about what happened?"
    
    def _respond_to_disgust(self, text):
        """Generate response for disgust emotion."""
        return "I understand that's disgusting to you. Sometimes things can really put us off. Would you like to talk about something else instead?"
    
    def _respond_to_neutral(self, text):
        """Generate response for neutral emotion."""
        return "I see. Is there anything specific you'd like to talk about today?"
    
    def _identify_triggers(self, text, trigger_words):
        """
        Identify potential triggers in the text based on trigger words.
        
        Args:
            text (str): The user's input text
            trigger_words (list): List of words that might indicate triggers
            
        Returns:
            str: The identified trigger or empty string if none found
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        # Find sentences containing trigger words
        for trigger in trigger_words:
            if trigger in text_lower:
                # Find the context around the trigger word
                index = text_lower.find(trigger)
                start = max(0, text_lower.rfind(".", 0, index))
                if start == 0:
                    start = max(0, text_lower.rfind("!", 0, index))
                    start = max(start, text_lower.rfind("?", 0, index))
                
                end = text_lower.find(".", index)
                if end == -1:
                    end = text_lower.find("!", index)
                    if end == -1:
                        end = text_lower.find("?", index)
                        if end == -1:
                            end = len(text_lower)
                
                context = text_lower[start:end].strip()
                if context:
                    # Remove common filler words to get to the core issue
                    for filler in ["i am", "i'm", "i feel", "i am feeling", "i'm feeling"]:
                        context = context.replace(filler, "")
                    return context.strip()
        
        return ""
