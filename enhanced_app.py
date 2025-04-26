"""
Web Interface for Enhanced Emotion-Aware Chatbot

This module implements a Flask web application for the enhanced emotion-aware chatbot
with both text and voice interfaces.
"""

from flask import Flask, render_template, request, jsonify
import os
import json
from conversation_manager import ConversationManager

app = Flask(__name__)

# Initialize the conversation manager
conversation_manager = ConversationManager()

@app.route('/')
def index():
    """Render the main page of the chatbot."""
    return render_template('enhanced_index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process text input from the user and return a response."""
    data = request.json
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({'response': 'I didn\'t catch that. Could you please say something?'})
    
    # Process the message using the conversation manager
    result = conversation_manager.process_message(user_input)
    
    return jsonify({
        'response': result['response'],
        'emotion': result['emotion'],
        'intensity': result['intensity'],
        'turn_count': result['turn_count'],
        'current_topic': result['current_topic']
    })

@app.route('/api/voice', methods=['POST'])
def voice():
    """Process voice input from the user and return a response."""
    data = request.json
    voice_text = data.get('voice_text', '')
    
    if not voice_text:
        return jsonify({'response': 'I didn\'t catch that. Could you please say something?'})
    
    # Process the voice text using the conversation manager
    result = conversation_manager.process_message(voice_text)
    
    return jsonify({
        'response': result['response'],
        'emotion': result['emotion'],
        'intensity': result['intensity'],
        'turn_count': result['turn_count'],
        'current_topic': result['current_topic']
    })

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Reset the conversation to a new state."""
    new_conversation_id = conversation_manager.reset_conversation()
    
    return jsonify({
        'status': 'success',
        'conversation_id': new_conversation_id,
        'message': 'Conversation has been reset.'
    })

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get a summary of the current conversation."""
    summary = conversation_manager.get_conversation_summary()
    
    return jsonify(summary)

@app.route('/api/save', methods=['POST'])
def save_conversation():
    """Save the current conversation to a file."""
    data = request.json
    file_path = data.get('file_path', 'conversation.json')
    
    success = conversation_manager.save_conversation(file_path)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': f'Conversation saved to {file_path}'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to save conversation'
        }), 500

@app.route('/api/load', methods=['POST'])
def load_conversation():
    """Load a conversation from a file."""
    data = request.json
    file_path = data.get('file_path', 'conversation.json')
    
    success = conversation_manager.load_conversation(file_path)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': f'Conversation loaded from {file_path}'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to load conversation'
        }), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
