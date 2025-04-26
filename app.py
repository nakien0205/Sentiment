"""
Sentiment Analysis Chatbot Web Application
This module implements a Flask web application for the sentiment analysis chatbot
with both text and voice interfaces.
"""

from flask import Flask, render_template, request, jsonify
import os
import json
import speech_recognition as sr
import pyttsx3
from sentiment_model import SentimentAnalyzer
from response_generator import ResponseGenerator

app = Flask(__name__)

# Initialize the sentiment analyzer and response generator
sentiment_analyzer = SentimentAnalyzer()
response_generator = ResponseGenerator()

# Initialize conversation history
conversation_history = []

@app.route('/')
def index():
    """Render the main page of the chatbot."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process text input from the user and return a response."""
    data = request.json
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({'response': 'I didn\'t catch that. Could you please say something?'})
    
    # Add user message to conversation history
    conversation_history.append({'role': 'user', 'content': user_input})
    
    # Analyze sentiment
    emotion_data = sentiment_analyzer.analyze_text(user_input)
    
    # Generate response based on emotion
    response = response_generator.generate_response(user_input, emotion_data)
    
    # Add bot response to conversation history
    conversation_history.append({'role': 'bot', 'content': response})
    
    # Keep conversation history to a reasonable size
    if len(conversation_history) > 20:
        conversation_history.pop(0)
        conversation_history.pop(0)
    
    return jsonify({
        'response': response,
        'emotion': emotion_data['emotion'],
        'confidence': emotion_data['confidence']
    })

@app.route('/api/voice', methods=['POST'])
def voice():
    """Process voice input from the user and return a response."""
    # In a real implementation, this would process audio data
    # For now, we'll simulate by accepting text that represents voice input
    data = request.json
    voice_text = data.get('voice_text', '')
    
    if not voice_text:
        return jsonify({'response': 'I didn\'t catch that. Could you please say something?'})
    
    # Process the voice text the same way as chat
    conversation_history.append({'role': 'user', 'content': voice_text})
    emotion_data = sentiment_analyzer.analyze_text(voice_text)
    response = response_generator.generate_response(voice_text, emotion_data)
    conversation_history.append({'role': 'bot', 'content': response})
    
    if len(conversation_history) > 20:
        conversation_history.pop(0)
        conversation_history.pop(0)
    
    return jsonify({
        'response': response,
        'emotion': emotion_data['emotion'],
        'confidence': emotion_data['confidence']
    })

@app.route('/api/history', methods=['GET'])
def history():
    """Return the conversation history."""
    return jsonify({'history': conversation_history})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
