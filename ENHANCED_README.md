# Enhanced Emotion-Aware Chatbot

A sophisticated chatbot that uses fine-tuned language models to generate diverse, contextually appropriate responses while maintaining awareness of emotions.

## Features

- **Fine-tuned Language Model**: Uses T5 with LoRA (Low-Rank Adaptation) to generate natural, diverse responses
- **Emotion Detection**: Accurately identifies 12 different emotions in user messages
- **Emotion-Aware Responses**: Generates responses appropriate to the emotional context with varying strategies based on emotion intensity
- **Conversation Management**: Tracks history, detects topics, identifies conversation drift, and extracts user preferences
- **Dual Interface**: Supports both text and voice interaction
- **Conversation Analytics**: Provides summaries of conversation metrics and emotion distribution

## Technical Architecture

### Components

1. **T5 Language Model Integration**: 
   - Uses Google's T5 (Text-to-Text Transfer Transformer) model
   - Implements LoRA for parameter-efficient fine-tuning
   - Combines DailyDialog and GoEmotions datasets for training

2. **Emotion-Aware Response Generator**:
   - Detects emotions and their intensity in user messages
   - Applies emotion-specific response strategies
   - Enhances base responses with emotional context

3. **Conversation Manager**:
   - Maintains conversation history and context
   - Detects topics and conversation drift
   - Extracts user preferences
   - Provides conversation analytics

4. **Web Application**:
   - Flask-based backend
   - Responsive web interface
   - Text and voice interaction capabilities
   - Conversation reset and summary features

## Implementation Details

### T5 with LoRA Fine-tuning

The chatbot uses parameter-efficient fine-tuning with LoRA, which:
- Updates less than 1% of the model's parameters
- Keeps most of the pre-trained model frozen
- Adds lightweight, trainable layers
- Makes training faster and more resource-efficient

### Datasets

The model is fine-tuned on two datasets:
1. **DailyDialog**: Contains multi-turn conversations with emotion annotations
2. **GoEmotions**: Provides detailed emotion labels for text samples

These datasets are processed and combined to teach the model both conversational flow and emotional awareness.

### Emotion-Specific Response Strategies

The chatbot implements different response strategies for 12 emotions:
- **Happiness**: Matches enthusiasm and adds positive reinforcement
- **Sadness**: Offers support, validation, and empathetic responses
- **Fear**: Provides reassurance and practical support
- **Anger**: Acknowledges feelings and de-escalates
- **Surprise**: Acknowledges the unexpected nature
- **Disgust**: Validates feelings and offers perspective
- **Neutral**: Focuses on engagement and information
- **Excitement**: Responds with high energy and enthusiasm
- **Gratitude**: Acknowledges appreciation
- **Love**: Responds warmly to affection
- **Confusion**: Offers clarification and simplified explanations
- **Disappointment**: Acknowledges feelings and offers encouragement

### Conversation Management Features

- **Topic Detection**: Identifies conversation topics from user messages
- **Conversation Drift Detection**: Recognizes when the conversation changes topics
- **User Preference Extraction**: Identifies likes and dislikes mentioned by the user
- **Conversation Analytics**: Tracks metrics like turn count, response time, and emotion distribution

## Installation and Setup

### Prerequisites

- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- Flask
- SpeechRecognition
- pyttsx3
- NumPy
- GPU capabilities for fine-tuning (recommended)

### Installation Steps

1. Clone the repository or download the source code
2. Install the required dependencies:
   ```
   pip install torch transformers flask SpeechRecognition pyttsx3 numpy datasets evaluate
   ```
3. Fine-tune the model (requires GPU capabilities):
   ```
   python t5_emotion_chatbot.py
   ```
4. Run the application:
   ```
   python enhanced_app.py
   ```
5. Access the chatbot in your web browser at `http://localhost:5000`

## Usage Instructions

### Text Chat Mode

1. The default mode is text chat
2. Type your message in the input field at the bottom of the screen
3. Press Enter or click the send button to send your message
4. The chatbot will analyze your message, detect the emotion, and respond accordingly
5. The detected emotion will be displayed below the chatbot's response

### Voice Chat Mode

1. Click the "Voice Chat" button in the mode toggle at the top of the chat interface
2. Click the microphone button to start recording
3. Speak your message clearly
4. The recording will automatically stop when you finish speaking
5. The chatbot will process your speech, detect the emotion, and respond both textually and verbally

### Additional Features

- **Reset Conversation**: Click the "Reset Conversation" button to start a new conversation
- **Show Conversation Summary**: Click the "Show Conversation Summary" button to view metrics about the current conversation
- **Conversation Analytics**: View information about turn count, current topic, dominant emotion, and conversation duration

## Customization

### Fine-tuning on Custom Datasets

To fine-tune the model on your own datasets:
1. Prepare your data in the format expected by the `prepare_datasets` method
2. Modify the `fine_tune` method parameters as needed
3. Run the fine-tuning process

### Adding New Emotions

To add new emotions to the detection and response system:
1. Add the new emotion to the `emotions` list in the `T5EmotionChatbot` class
2. Create a corresponding response strategy method in the `EmotionAwareResponseGenerator` class
3. Add the new strategy to the `emotion_strategies` dictionary

### Modifying Response Strategies

To change how the chatbot responds to specific emotions:
1. Edit the corresponding strategy method in the `EmotionAwareResponseGenerator` class
2. For example, to modify how the chatbot responds to sadness, edit the `_sadness_strategy` method

## File Structure

- `t5_emotion_chatbot.py`: Implements the T5 model with LoRA fine-tuning
- `emotion_aware_response_generator.py`: Implements the emotion-aware response generation system
- `conversation_manager.py`: Implements the conversation management system
- `enhanced_app.py`: Implements the Flask web application
- `templates/enhanced_index.html`: Implements the web interface

## Future Improvements

- Implement multi-language support
- Add more sophisticated topic modeling
- Integrate with external APIs for more dynamic responses
- Implement user authentication for personalized experiences
- Add more advanced conversation analytics
- Optimize for mobile devices with a dedicated app

## License

This project is open source and available under the MIT License.
